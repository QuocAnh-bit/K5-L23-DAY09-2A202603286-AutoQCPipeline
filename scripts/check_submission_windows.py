"""Chạy trình kiểm tra gốc với đường dẫn chuẩn hóa khi dùng Windows.

Sử dụng: python scripts/check_submission_windows.py [--no-train]
Không thay đổi file kiểm tra gốc; chỉ sửa cách so sánh đường dẫn trong bộ nhớ.
"""
from pathlib import Path  # Xác định vị trí trình kiểm tra gốc.
import sys  # Truyền các tham số dòng lệnh cho trình kiểm tra gốc.

checker = Path(__file__).with_name("check_submission.py")  # File được giảng viên cung cấp.
source = checker.read_text(encoding="utf-8")  # Đọc mà không ghi đè file gốc.
old = "str(p.relative_to(ROOT)) not in prot"  # Biểu thức gây lệch dấu phân cách trên Windows.
new = "p.relative_to(ROOT).as_posix() not in prot"  # Dùng dấu gạch giống danh sách file được bảo vệ.
if source.count(old) != 1:  # Ngừng nếu trình kiểm tra gốc đã thay đổi.
    raise RuntimeError("Không tìm thấy chính xác đoạn chuẩn hóa đường dẫn cần sửa.")
sys.argv[0] = str(checker)  # Giữ các tùy chọn như --no-train cho trình gốc.
exec(compile(source.replace(old, new), str(checker), "exec"), {"__name__": "__main__", "__file__": str(checker)})  # Chạy phiên bản đã chuẩn hóa trong bộ nhớ.
