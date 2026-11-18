"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
# Nhóm lỗi theo ảnh và số dòng gốc.
from collections import defaultdict

# Đọc và ghi cấu hình lớp trong data.yaml.
import yaml

# Dùng mức lỗi và bộ đọc dòng YOLO chung.
from ..common import ERROR, parse_yolo_line


# ────────────────────────────────────────────────────────────────────
# Hàm chính của module
# ────────────────────────────────────────────────────────────────────
# Sửa các lỗi an toàn trên bản dữ liệu được truyền vào.
def fix(ds, issues, cfg):
    # Lưu nhật ký thay đổi để người dùng kiểm tra.
    changes = []
    # Chỉ xử lý lỗi bắt buộc, không sửa cảnh báo.
    errors = [issue for issue in issues if issue.severity == ERROR]
    # Các lỗi cần bỏ dòng nhãn.
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    # Các lỗi cần loại ảnh khỏi bộ dữ liệu.
    remove_codes = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    # Gom ảnh phải loại trước khi sửa dòng.
    remove_stems = {issue.image for issue in errors if issue.code in remove_codes or issue.code.startswith("DUP_IMAGE_")}
    
    # Xử lý mỗi ảnh cần loại đúng một lần.
    for stem in sorted(remove_stems):
        # Tìm mọi file ảnh cùng tên.
        image_paths = [path for path in ds.image_files() if path.stem == stem]
        # Tìm đường dẫn file nhãn tương ứng.
        label_path = ds.label_path(stem)
        # Ghi lại tên các ảnh bị loại.
        removed_names = [path.name for path in image_paths]
        # Kiểm tra file nhãn có tồn tại không.
        if label_path.exists():
            # Ghi lại tên file nhãn trước khi xóa.
            removed_names.append(label_path.name)
        # Duyệt các file ảnh cần loại.
        for path in image_paths:
            # Xóa ảnh khỏi bản dữ liệu đang xử lý.
            path.unlink()
        # Chỉ xóa nhãn nếu file còn tồn tại.
        if label_path.exists():
            # Xóa file nhãn đi kèm ảnh bị loại.
            label_path.unlink()
        # Chỉ ghi nhật ký khi đã thực sự loại file.
        if removed_names:
            # Lưu bằng chứng ảnh bị loại.
            changes.append({"image": stem, "line": None, "action": "remove_image", "before": ", ".join(removed_names), "after": "đã xóa"})
    # Xử lý mỗi ảnh thiếu nhãn còn tồn tại.
    for stem in sorted({issue.image for issue in errors if issue.code == "MISSING_LABEL_FILE"} - remove_stems):
        # Chỉ chuyển người khi ảnh tồn tại và vẫn thiếu file nhãn.
        if ds.image_path(stem) is not None and not ds.label_path(stem).exists():
            # Không tự tạo nhãn hoặc file rỗng.
            changes.append({"image": stem, "line": None, "action": "needs_label", "before": "(không có file)", "after": "chuyển người gán"})
    # Chỉ sửa tên lớp khi có lỗi và có file cấu hình.
    if any(issue.code == "BAD_CLASS_NAME" for issue in errors) and ds.yaml_path.exists():
        # Đọc nội dung YAML hiện tại.
        data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
        # Lưu tên lớp cũ để ghi nhật ký.
        before_names = data.get("names")
        # Không ghi lại nếu tên lớp đã đúng.
        if before_names != cfg["classes"]:
            # Đặt tên lớp đúng theo cấu hình dự án.
            data["names"] = list(cfg["classes"])
            # Ghi lại YAML và giữ các mục khác.
            ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            # Ghi tên lớp trước và sau.
            changes.append({"image": "data.yaml", "line": None, "action": "rename_classes", "before": str(before_names), "after": str(data["names"])})
    # Nhóm mã lỗi theo tên ảnh và số dòng ban đầu.
    line_errors = defaultdict(lambda: defaultdict(set))
    # Duyệt các lỗi bắt buộc để lập kế hoạch sửa dòng.
    for issue in errors:
        # Bỏ qua ảnh đã loại và lỗi không sửa theo dòng.
        if issue.image not in remove_stems and issue.line is not None and (issue.code in drop_codes or issue.code == "BOX_OUT_OF_IMAGE"):
            # Gom các lỗi cùng dòng để chỉ sửa một lần.
            line_errors[issue.image][issue.line].add(issue.code)
    # Xử lý lần lượt từng file nhãn cần sửa.
    for stem, by_line in sorted(line_errors.items()):
        # Lấy đường dẫn file nhãn của ảnh.
        path = ds.label_path(stem)
        # File có thể đã bị xóa hoặc không tồn tại.
        if not path.exists():
            # Bỏ qua an toàn nếu không có file.
            continue
        # Đọc toàn bộ dòng, kể cả dòng trống.
        original_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        # Lưu nội dung sau sửa theo đúng thứ tự cũ.
        updated_lines = []
        # Theo dõi xem file có thực sự thay đổi không.
        file_changed = False
        # Duyệt theo số dòng gốc, bắt đầu từ 1.
        for number, before in enumerate(original_lines, 1):
            # Lấy tất cả lỗi của dòng hiện tại.
            codes = by_line.get(number, set())
            # Ưu tiên xóa dòng khi dòng có lỗi cần xóa.
            if codes & drop_codes:
                # Ghi dòng đã xóa vào nhật ký.
                changes.append({"image": stem, "line": number, "action": "drop_line", "before": before, "after": ""})
                # Đánh dấu cần ghi lại file.
                file_changed = True
                # Không thêm dòng này và không cắt box của nó.
                continue
            # Chỉ cắt box nếu dòng không thuộc diện phải xóa.
            if "BOX_OUT_OF_IMAGE" in codes:
                # Thử đọc dòng nhãn trước khi tính cạnh box.
                try:
                    # Lấy lớp và hình học box.
                    class_id, x, y, width, height = parse_yolo_line(before)
                # Bỏ qua an toàn nếu dữ liệu đã thay đổi và không đọc được.
                except (ValueError, OverflowError):
                    # Giữ nguyên dòng không thể xử lý.
                    updated_lines.append(before)
                    # Chuyển sang dòng kế tiếp.
                    continue
                # Giới hạn cạnh trái và trên.
                left, top = max(0.0, x - width / 2), max(0.0, y - height / 2)
                # Giới hạn cạnh phải và dưới.
                right, bottom = min(1.0, x + width / 2), min(1.0, y + height / 2)
                # Box không còn diện tích sau khi cắt.
                if right <= left or bottom <= top:
                    # Ghi việc xóa box nằm ngoài ảnh.
                    changes.append({"image": stem, "line": number, "action": "drop_line", "before": before, "after": ""})
                    # Đánh dấu file đã đổi.
                    file_changed = True
                    # Không giữ box rỗng.
                    continue
                # Chỉ ghi nếu box thật sự vượt biên.
                if left != x - width / 2 or top != y - height / 2 or right != x + width / 2 or bottom != y + height / 2:
                    # Đổi cạnh đã cắt về định dạng YOLO.
                    after = f"{class_id} {(left + right) / 2:.8f} {(top + bottom) / 2:.8f} {right - left:.8f} {bottom - top:.8f}"
                    # Ghi giá trị trước và sau.
                    changes.append({"image": stem, "line": number, "action": "clip_box", "before": before, "after": after})
                    # Giữ dòng nhãn sau khi cắt.
                    updated_lines.append(after)
                    # Đánh dấu cần ghi lại file.
                    file_changed = True
                    # Chuyển sang dòng tiếp theo.
                    continue
            # Giữ nguyên mọi dòng không cần sửa.
            updated_lines.append(before)
        # Chỉ ghi file khi có thay đổi thực tế.
        if file_changed:
            # Ghi một lần để không làm lệch số dòng gốc.
            path.write_text("\n".join(updated_lines) + ("\n" if updated_lines else ""), encoding="utf-8")
    # Trả nhật ký tất cả thay đổi cho pipeline.
    return changes
