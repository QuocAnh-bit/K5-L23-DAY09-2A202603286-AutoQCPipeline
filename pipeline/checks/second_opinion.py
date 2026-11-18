"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
# Dùng công cụ đọc nhãn, tính IoU và ghi cảnh báo chung.
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
# Lấy dự đoán model đã lưu khi không được truyền trực tiếp.
from ..model_cache import get_preds


# ────────────────────────────────────────────────────────────────────
# Hàm chính của module
# ────────────────────────────────────────────────────────────────────
# So nhãn với dự đoán của model để đưa ra ý kiến thứ hai.
def run(ds, cfg, preds=None):
    # Kiểm tra người gọi có truyền sẵn dự đoán hay không.
    if preds is None:
        # Lấy dự đoán từ bộ nhớ đệm của model kiểm tra.
        preds = get_preds(ds, cfg, "checker")
    # Đọc các ngưỡng so sánh từ cấu hình.
    settings = cfg["second_opinion"]
    # Lưu các cảnh báo để chuyển cho người duyệt.
    issues = []
    # Xét ảnh theo thứ tự do bộ dữ liệu cung cấp.
    for image_path in ds.image_files():
        # Lấy tên ảnh không có phần mở rộng.
        stem = image_path.stem
        # Đọc các dòng nhãn cùng số dòng gốc.
        lines = ds.read_label_lines(stem)
        # Bỏ qua ảnh thiếu dự đoán hoặc chưa có file nhãn.
        if stem not in preds or lines is None:
            # Chuyển sang ảnh tiếp theo.
            continue
        # Lấy kích thước ảnh để tính diện tích dự đoán theo pixel.
        size = ds.image_size(stem)
        # Không thể tính diện tích khi ảnh không đọc được.
        if size is None:
            # Để bước kiểm tra ảnh xử lý trường hợp này.
            continue
        # Lưu nhãn hợp lệ cùng số dòng gốc.
        labels = []
        # Duyệt từng dòng nhãn không trống.
        for number, text in lines:
            # Thử phân tích dòng nhãn YOLO.
            try:
                # Lấy ID lớp và bốn thông số box.
                label = parse_yolo_line(text)
            # Dòng sai định dạng hoặc ID không hữu hạn.
            except (ValueError, OverflowError):
                # Bỏ qua dòng hỏng, vì check định dạng sẽ báo riêng.
                continue
            # Chỉ dùng nhãn có ID thuộc danh sách lớp.
            if 0 <= label[0] < len(ds.classes):
                # Giữ số dòng để báo false positive chính xác.
                labels.append((number, label))
        # Lấy danh sách dự đoán của ảnh hiện tại.
        image_preds = preds[stem]
        # Xét từng dự đoán theo thứ tự ban đầu.
        for prediction in image_preds:
            # Tách lớp, độ tin cậy và box dự đoán.
            class_id, confidence, x, y, width, height = prediction
            # Lọc dự đoán yếu hoặc quá nhỏ.
            if confidence < settings["conf"] or width * size[0] * height * size[1] < settings["min_area_px"]:
                # Không dùng dự đoán này để tìm thiếu nhãn, nhầm lớp hoặc box lỏng.
                continue
            # Gom hình học box để tính IoU.
            box = (x, y, width, height)
            # Tính IoU với mọi nhãn hợp lệ.
            candidates = [(label[0], iou_xywh(box, label[1:])) for _, label in labels]
            # Giữ các nhãn có mức chồng lấn đủ lớn.
            candidates = [(cls, overlap) for cls, overlap in candidates if overlap >= settings["iou_missing"]]
            # Không có nhãn nào khớp vật mà model dự đoán.
            if not candidates:
                # Cảnh báo có thể thiếu nhãn.
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_MISSING", stem, "Model có thể thấy đối tượng chưa được gán nhãn", extra={"conf": confidence}))
                # Một dự đoán chỉ tạo tối đa một cảnh báo.
                continue
            # Tìm các nhãn ứng viên cùng lớp.
            same_class = [overlap for cls, overlap in candidates if cls == class_id]
            # Ưu tiên nhãn cùng lớp dù có nhãn khác lớp chồng lên.
            if same_class:
                # Chọn nhãn cùng lớp có IoU cao nhất.
                best = max(same_class)
                # Box cùng lớp vẫn lệch hoặc lỏng so với model.
                if best < settings["iou_loose"]:
                    # Ghi cảnh báo để người xem.
                    issues.append(Issue("second_opinion", WARN, "POSSIBLE_LOOSE_BOX", stem, "Box nhãn có thể lỏng hoặc lệch", extra={"conf": confidence, "iou": best}))
                # Không báo nhầm lớp khi đã có ứng viên cùng lớp.
                continue
            # Tìm IoU cao nhất trong các ứng viên khác lớp.
            best = max(overlap for _, overlap in candidates)
            # Chỉ báo nhầm lớp khi mức trùng đủ chắc.
            if best >= settings["iou_match"]:
                # Ghi cảnh báo nhầm lớp.
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_WRONG_CLASS", stem, "Nhãn có thể bị gán nhầm lớp", extra={"conf": confidence, "iou": best}))
        # Lấy dự đoán cho false positive, không lọc diện tích.
        fp_predictions = [prediction for prediction in image_preds if prediction[1] >= settings["fp_conf"]]
        # Kiểm tra từng dòng nhãn hợp lệ theo thứ tự gốc.
        for number, label in labels:
            # Tìm mức khớp tốt nhất, hoặc 0 nếu không có dự đoán.
            best = max((iou_xywh(label[1:], prediction[2:]) for prediction in fp_predictions), default=0.0)
            # Không dự đoán nào khớp đủ với box nhãn.
            if best < settings["iou_missing"]:
                # Ghi cảnh báo cùng số dòng gốc.
                issues.append(Issue("second_opinion", WARN, "POSSIBLE_FALSE_POS", stem, "Box nhãn có thể không có đối tượng tương ứng", line=number, extra={"iou": best}))
    # Trả các cảnh báo cho pipeline và hàng đợi người duyệt.
    return issues
