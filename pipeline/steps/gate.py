"""BƯỚC 4 — Cửa chặn: quyết định Release / Hold / Rework.   ★

Test:  pytest tests/test_gate.py -q

Cây quyết định Day 07 (slide Day 09: "Lỗi bắt buộc chặn pipeline; lỗi cảnh báo chuyển người xem"):

  còn ≥ 1 lỗi bắt buộc (severity ERROR)                          -> "Rework"
  không còn lỗi bắt buộc, nhưng tỉ lệ ẢNH còn cảnh báo
      >= cfg["gate"]["warn_ratio_hold"]                          -> "Hold"
  còn lại                                                        -> "Release"

Tỉ lệ cảnh báo = (số ẢNH khác nhau có ≥ 1 cảnh báo) / (tổng số ảnh).  Chú ý:
- chỉ đếm issue có `image` nằm trong `image_stems` (bỏ "<dataset>", "data.yaml");
- bỏ qua ảnh nằm trong `reviewed` (người đã xem, cảnh báo của ảnh đó coi như đã xử lý) — nhưng LỖI BẮT BUỘC thì không bao giờ được bỏ qua;
- mẫu số là số ảnh, không phải số issue.

Trả về dict: decision, n_errors, n_warn_images, n_images, warn_ratio (làm tròn 4 chữ số), reasons (list[str], giải thích bằng tiếng Việt, có số liệu).
"""
# Dùng tên mức độ lỗi và cảnh báo thống nhất của dự án.
from ..common import ERROR, WARN


# ────────────────────────────────────────────────────────────────────
# Hàm chính của module
# ────────────────────────────────────────────────────────────────────
# Quyết định trạng thái dựa trên lỗi và ảnh đã duyệt.
def decide(issues, image_stems, cfg, reviewed=frozenset()):
    # Lấy tập tên ảnh để đếm mỗi ảnh một lần và kiểm tra nhanh.
    stems = set(image_stems)
    # Chuyển danh sách ảnh đã duyệt thành tập để tra cứu.
    reviewed_images = set(reviewed)
    # Đếm mọi lỗi bắt buộc, kể cả trên ảnh đã duyệt.
    n_errors = sum(1 for issue in issues if issue.severity == ERROR)
    # Chọn các ảnh còn cảnh báo chưa duyệt.
    warned_images = {issue.image for issue in issues if issue.severity == WARN and issue.image in stems and issue.image not in reviewed_images}
    # Đếm tổng số ảnh khác nhau trong bộ dữ liệu.
    n_images = len(stems)
    # Đếm số ảnh còn cảnh báo, không đếm số dòng cảnh báo.
    n_warn_images = len(warned_images)
    # Tính tỷ lệ và tránh chia cho 0 khi không có ảnh.
    warn_ratio = n_warn_images / n_images if n_images else 0.0
    # Đọc ngưỡng Hold đã quy định trong cấu hình.
    threshold = cfg["gate"]["warn_ratio_hold"]
    
    # Lỗi bắt buộc luôn được ưu tiên cao nhất.
    if n_errors > 0:
        # Yêu cầu sửa lại khi còn lỗi bắt buộc.
        decision = "Rework"
        # Nêu số lỗi làm căn cứ cho quyết định.
        reasons = [f"Còn {n_errors} lỗi bắt buộc cần xử lý."]
    # So tỷ lệ thực tế chưa làm tròn với ngưỡng Hold.
    elif warn_ratio >= threshold:
        # Tạm giữ khi tỷ lệ cảnh báo đạt hoặc vượt ngưỡng.
        decision = "Hold"
        # Nêu số ảnh và ngưỡng.
        reasons = [f"{n_warn_images}/{n_images} ảnh còn cảnh báo ({warn_ratio:.2%}), đạt hoặc vượt ngưỡng {threshold:.2%}."]
    # Không có lỗi bắt buộc và tỷ lệ cảnh báo dưới ngưỡng.
    else:
        # Cho phép phát hành bộ dữ liệu.
        decision = "Release"
        # Giải thích quyết định.
        reasons = [f"Không còn lỗi bắt buộc; {n_warn_images}/{n_images} ảnh còn cảnh báo ({warn_ratio:.2%}), dưới ngưỡng {threshold:.2%}."]
    # Trả quyết định cùng số liệu và lý do.
    return {"decision": decision, "n_errors": n_errors, "n_warn_images": n_warn_images, "n_images": n_images, "warn_ratio": round(warn_ratio, 4), "reasons": reasons}
