# Báo cáo Day 09 — Dương Quốc Anh, 2A202603286

## 1. Kết luận
Pipeline kết luận **Hold**. Lô đầu vào có 224 ảnh; cửa nhận loại 19 ảnh và còn 205 ảnh. Không còn lỗi bắt buộc sau duyệt mô phỏng, nhưng 38/205 ảnh (18,54%) còn cảnh báo chưa được tính là xử lý, vượt ngưỡng Hold 10%. Dataset `clean_v1` gồm 205 ảnh và là bản nháp, chưa phát hành.

## 2. Lỗi và phân việc
Số liệu sau là **số dòng issue** trong `02_error_list_v1.csv`, gồm kết quả cửa nhận và QC ban đầu; một ảnh có thể có nhiều issue, vì vậy không cộng các dòng thành số ảnh duy nhất.

| Mã lỗi | Số issue | Phân việc và lý do |
|---|---:|---|
| BAD_SIZE / CORRUPT_IMAGE | 3 / 2 | Máy loại ở cửa nhận: ảnh quá nhỏ hoặc không đọc được. |
| DUP_IMAGE_EXACT / DUP_IMAGE_NEAR / LEAK_TEST | 8 / 2 / 4 | Máy loại ở cửa nhận; cần xem lại cặp gần trùng để tránh loại nhầm. |
| BAD_CLASS_ID / BAD_FORMAT / BOX_ZERO / DUP_BOX | 6 / 3 / 4 / 7 | Máy loại dòng không hợp lệ hoặc box trùng; thay đổi được ghi log. |
| BOX_OUT_OF_IMAGE / BAD_CLASS_NAME | 8 / 2 | Máy cắt box hoặc sửa tên lớp theo cấu hình. |
| MISSING_LABEL_FILE | 6 | Chuyển người gán; máy không tự tạo nhãn rỗng. |
| BOX_HUGE / BOX_TINY / EMPTY_LABEL / CLASS_RARE | 3 / 6 / 12 / 4 | Cảnh báo để người xem; CLASS_RARE là cảnh báo cấp bộ dữ liệu. |
| POSSIBLE_FALSE_POS / POSSIBLE_LOOSE_BOX | 29 / 28 | Ý kiến model, cần người xác minh. |
| POSSIBLE_MISSING / POSSIBLE_WRONG_CLASS | 53 / 18 | Ý kiến model, cần người xác minh thiếu nhãn hoặc nhầm lớp. |

Hàng đợi có 125 ảnh, xếp theo tổng trọng số lỗi và độ tin cậy; ví dụ `img_0020` đứng đầu với điểm 11,524, còn ảnh thiếu nhãn được cộng trọng số 10. Trong 80 ảnh được duyệt mô phỏng có **74 ảnh mang ít nhất một loại cảnh báo ngữ nghĩa**, trên tổng 103 ảnh như vậy trong hàng đợi. Đây là số ảnh có cảnh báo đã được xem, không phải số lỗi ngữ nghĩa thực sự được xác nhận.

## 3. Calibration
| Tham số | Khởi đầu | Chọn | Precision / recall trước → sau | Đánh đổi |
|---|---:|---:|---|---|
| `duplicates.phash_dist` | 2 | 4 | DUP_IMAGE_NEAR: 1,00 / 0,50 → 1,00 / 0,67 | Chọn giá trị thấp nhất đạt recall 0,67 trên sweep để hạn chế rủi ro loại nhầm. |
| `second_opinion.conf` | 0,25 | 0,60 | WRONG_CLASS: 0,56 / 0,83 → 0,62 / 0,83; MISSING: 0,50 / 1,00 → 0,57 / 1,00; LOOSE_BOX: 0,29 / 0,40 → 0,20 / 0,20 | Precision của hai loại cảnh báo đầu tăng, nhưng phát hiện box lỏng giảm; cần người xem. |

Calib chỉ có 39 ảnh nên kết quả có thể không đại diện lô raw. Lần chạy raw loại 19/224 ảnh ở cửa nhận. Đã mở xem hai cặp gần trùng `img_0076`–`img_0042` và `img_0143`–`img_0037`: mỗi cặp có cùng cảnh dù khác kích thước hoặc độ sáng; hai ví dụ chưa chứng minh mọi trường hợp loại đều đúng.

## 4. Cửa chặn và giới hạn
Thử `--budget 0 --reports work/rep_nb --export-dir work/exp_nb`: kết quả **Rework**, mã thoát 20, còn 6 lỗi `MISSING_LABEL_FILE` và không tạo dataset xuất. Không có người duyệt nên máy không bịa nhãn.

Hai giới hạn có bằng chứng từ calibration: (1) ở `phash_dist=4`, recall gần trùng 4/6, tức bỏ sót 2/6 trường hợp đã biết; (2) ở `conf=0,6`, recall cảnh báo box lỏng là 1/5, tức bỏ sót 4/5 trường hợp đã biết. Các số này đo trên calib, không chứng minh lỗi tương tự ở một ảnh raw cụ thể. Ngoài ra, dự đoán model là ý kiến thứ hai và có thể bỏ sót khi model cùng nhầm với nhãn.

Quan sát ảnh có vẽ nhãn: `img_0001` có người mặc quân phục ở giữa chưa có box; `img_0030` có các box người chồng lấn quanh cùng một người, cần đối chiếu từng dòng nhãn trước khi sửa. Với `img_0002`, học viên nghi có nhãn gán nhầm; hình chụp chưa đủ để khẳng định và cần người duyệt kiểm tra thêm.

## 5. Train raw và clean
Train trên Colab với `yolo26n.pt`, 20 epoch, batch 16, ảnh 640, seed 0 và 1. Kết quả ghi 224 ảnh raw và 205 ảnh clean; số ảnh raw thực sự được dùng có thể ít hơn nếu trình train bỏ ảnh hoặc nhãn lỗi, cần đối chiếu log.

| Biến thể | mAP50 seed 0 | mAP50 seed 1 | Trung bình | Độ lệch chuẩn giữa 2 seed |
|---|---:|---:|---:|---:|
| raw | 0,7256 | 0,7259 | 0,7258 | 0,0002 |
| clean | 0,6788 | 0,7219 | 0,7004 | 0,0305 |

Chênh lệch trung bình clean − raw là **−0,0254**, nhỏ hơn độ lệch chuẩn 0,0305 của clean. Với chỉ hai seed, kết quả chưa đủ chắc để kết luận clean cải thiện hay làm giảm hiệu quả; số đo hiện tại không ủng hộ khẳng định clean tốt hơn raw. Hai giả thuyết cần kiểm tra: raw có 4 trường hợp rò rỉ test được phát hiện ở cửa nhận, có thể làm phép đo raw lạc quan; clean vẫn ở trạng thái Hold, còn cảnh báo chưa duyệt. Cần đối chiếu log train và chạy thêm seed/đánh giá trên tập độc lập để kiểm chứng, không suy ra mức ảnh hưởng từ các số hiện có.

## 6. Quyết định và sử dụng AI
Giữ **Hold**. Cần người có thẩm quyền xem các cảnh báo còn lại và sửa nhãn thực sự có lỗi, sau đó chạy lại QC và gate; chỉ Release khi không còn lỗi bắt buộc và tỷ lệ cảnh báo chưa duyệt dưới 10%, không thay ngưỡng cố định để đạt kết quả.

- Prompt chính: nhờ AI hỗ trợ viết lần lượt bốn module QC và chú thích rõ chức năng từng dòng.
- Kiểm tra: chạy test từng module và toàn bộ 79 test; chạy calibration, đối chiếu hai cặp ảnh và thử gate với ngân sách 0.
- Lỗi AI đã phát hiện: phiên bản `autofix` ban đầu do AI viết không có chốt chặn khi bị gọi trực tiếp trên `data/raw`; nếu dùng sai đầu vào, nó có thể sửa hoặc xóa dữ liệu gốc. Đã thêm kiểm tra đường dẫn trước khi ghi và thử gọi với dataset raw: hàm báo lỗi, không thực hiện sửa. Chạy lại test để kiểm tra các tình huống thông thường.
