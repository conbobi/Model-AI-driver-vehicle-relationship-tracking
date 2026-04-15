Bước 1: Chuẩn hóa video gốc (chuanhoavideo.txt)
Bước này thường sử dụng FFmpeg (một công cụ xử lý video cực mạnh). File chuanhoavideo.txt của bạn chắc chắn chứa các dòng lệnh tương tự như sau.

Lệnh chạy: (note: Chạy tại thư mục  "CCTV_VMR")
Bash
ffmpeg -i data/raw/cam02/video2.mp4 -c:v libx264 -preset veryfast -crf 23 -r 25 -vf "scale=-2:720" -c:a aac -b:a 128k data/raw/cam02/video2_norm.mp4


Bước 2: Vẽ vùng quan tâm ROI (draw_roi.py)
Tool này mở video lên để bạn trích xuất tọa độ khu vực cần theo dõi.
Lệnh chạy:
Bash
python draw_roi.py

Thao tác: 
1. Một cửa sổ video sẽ hiện lên (thường là frame đầu tiên).
2. Dùng chuột click các điểm tạo thành đa giác (hoặc kéo thả thành hình chữ nhật) bao quanh ngã tư/lòng đường.
3. Nhấn phím ENTER hoặc S (Save) để lưu.
4. Tool sẽ sinh ra một file tọa độ (ví dụ: roi.txt hoặc roi.json) lưu vào cùng thư mục.

Bước 3: Cắt clip sự kiện (cut_events_roi_v4.py)
Đây là tool lõi để tự động hóa việc cắt video.

Lệnh chạy:
Bash
python cut_events_roi_v4.py

Thao tác: Nhấn Enter, Màn hình terminal sẽ nhảy log liên tục báo hiệu: "Phát hiện chuyển động -> Đang cắt clip 001.mp4... Đang cắt clip 002.mp4...". Kết quả là bạn sẽ có một thư mục (ví dụ output_clips/) chứa hàng trăm clip ngắn 3-5s.

Bước 4: Làm sạch và sửa clip lỗi (fix_clips.py)
Dọn dẹp rác sau khi cắt tự động.

Lệnh chạy:
Bash
python fix_clips.py

Thao tác: Tool sẽ quét qua thư mục output_clips/. Nó sẽ tự động xóa các clip dung lượng quá nhỏ (ví dụ < 100KB), clip có thời lượng quá ngắn (dưới 1s), hoặc chạy lại FFmpeg nội bộ để vá lỗi header của các clip bị hỏng. Bạn sẽ thấy log báo: "Đã xóa 15 clip rác. Đã fix 3 clip lỗi".

Bước 5: Gán nhãn dữ liệu (label_tool_v26.py)
Phần việc thủ công mà chúng ta cần làm

Lệnh chạy:
Bash
python label_tool_v26.py

Thao tác: 1. Giao diện Terminal hiện lên yêu cầu bạn "Chọn (1/2/3)".
2. Tool tự động mở frame ảnh/video đầu tiên trong thư mục clip đã lọc.
3. Bạn vẽ Box, nhập thông số (màu áo, nón, loại xe, hành động...).
4. Nhấn Enter, tool tự sinh nhãn tiếng Việt/tiếng Anh lưu vào file .txt và chuyển sang clip tiếp theo.

Bước 6: Xác thực dữ liệu (validate_v2.py)
Kiểm tra chéo trước khi mang đi huấn luyện mô hình (Train AI).

Lệnh chạy:
Bash
python validate_v2.py

Thao tác: Tool sẽ quét 2 thư mục: thư mục chứa hình/clip và thư mục chứa file .txt nhãn. Nó sẽ báo lỗi chữ đỏ lên màn hình nếu phát hiện:
File ảnh có mà không có file nhãn (bạn bỏ sót).
Trong file nhãn bị trống nội dung hoặc sai format tiếng Anh/Việt.

Nếu mọi thứ báo Xanh (Passed) -> Data của bạn đã hoàn hảo 100%!