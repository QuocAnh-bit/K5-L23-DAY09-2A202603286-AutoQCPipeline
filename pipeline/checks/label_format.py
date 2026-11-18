"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
# Dùng các công cụ kiểm tra chung của dự án.
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line


# ────────────────────────────────────────────────────────────────────
# Hàm chính của module
# ────────────────────────────────────────────────────────────────────
# Kiểm tra các dòng nhãn trong bộ dữ liệu theo cấu hình.
def run(ds, cfg):
    # Lưu tất cả lỗi và cảnh báo tìm được.
    issues = []
    # Lấy các ngưỡng đã quy định cho kiểm tra nhãn.
    settings = cfg["label_format"]

    # Lần lượt xem từng file nhãn.
    for label_file in ds.label_files():
        # Lấy tên ảnh tương ứng, không có đuôi file.
        stem = label_file.stem
        # Lấy kích thước ảnh để kiểm tra box nhỏ.
        image_size = ds.image_size(stem)
        # Lưu box hợp lệ trước đó để phát hiện box trùng.
        previous_boxes = []

        # Dguyệt từng dòng không trốn, giữ số dòng gốc.
        for line_number, text in ds.read_label_lines(stem) or []:
            # Thử đọc năm cột theo định dạng YOLO.
            try:
                # Tách ID lớp, tâm và kích thước box.
                class_id, x, y, width, height = parse_yolo_line(text)

            # Dòng sai định dạng hoặc chứa ID không hữu hạn.
            except (ValueError, OverflowError):
                # Ghi lỗi định dạng.
                issues.append(Issue("label_format", ERROR, "BAD_FORMAT", stem, "Dòng nhãn sai định dạng", line_number))
                # Không kiểm tra hình học trên dòng không đọc được.
                continue

            # Kiểm tra ID có thuộc danh sách lớp không.
            if not 0 <= class_id < len(ds.classes):
                # Ghi lỗi ID.
                issues.append(Issue("label_format", ERROR, "BAD_CLASS_ID", stem, "ID lớp không hợp lệ", line_number))
                # Bỏ qua phần hình học của dòng có ID sai.
                continue

            # Box phải có chiều rộng và chiều cao dương.
            if width <= 0 or height <= 0:
                # Ghi lỗi box rỗng.
                issues.append(Issue("label_format", ERROR, "BOX_ZERO", stem, "Box có kích thước không dương", line_number))
                # Không kiểm tra tiếp box không hợp lệ.
                continue
            # Gom tọa độ và kích thước để so sánh IoU.
            box = (x, y, width, height)
            # Tìm độ tràn lớn nhất ở bốn cạnh.
            overflow = max(0.0, width / 2 - x, height / 2 - y, x + width / 2 - 1, y + height / 2 - 1)
            # Chỉ báo khi độ tràn vượt mức cho phép.
            if overflow > settings["overflow_tol"]:
                # Ghi lỗi và độ tràn.
                issues.append(Issue("label_format", ERROR, "BOX_OUT_OF_IMAGE", stem, "Box vượt biên ảnh", line_number, {"overflow": overflow}))
            # So với box cùng lớp ở các dòng trước.
            if any(old_class == class_id and iou_xywh(box, old_box) > settings["dup_box_iou"] for old_class, old_box in previous_boxes):
                # Báo box trùng tại dòng sau.
                issues.append(Issue("label_format", ERROR, "DUP_BOX", stem, "Box trùng với dòng trước", line_number))
            # Ghi nhớ box hiện tại cho những dòng tiếp theo.
            previous_boxes.append((class_id, box))
            # Đổi hai cạnh sang pixel và tìm cạnh ngắn.
            if image_size is not None and min(width * image_size[0], height * image_size[1]) < settings["min_box_px"]:
                # Cảnh báo box nhỏ.
                issues.append(Issue("label_format", WARN, "BOX_TINY", stem, "Box có cạnh quá nhỏ", line_number))
            # So diện tích box với phần diện tích ảnh cho phép.
            if width * height > settings["max_box_frac"]:
                # Cảnh báo box lớn.
                issues.append(Issue("label_format", WARN, "BOX_HUGE", stem, "Box chiếm diện tích quá lớn", line_number))
    # Trả toàn bộ lỗi và cảnh báo cho pipeline.
    return issues
