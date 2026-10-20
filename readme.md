# TRẠM 2: POSE MATCHING – SAO Y BẢN CHÍNH

> **Tagline:** *"Đọ dáng cực lầy – Chấm điểm bằng AI ngay tức thì!"*

Ứng dụng trò chơi tương tác thị giác máy tính (Computer Vision) thời gian thực dành cho gian hàng sự kiện, sử dụng **Python**, **OpenCV**, **MediaPipe Pose**, **Pillow** và **Pygame**. Được thiết kế tối ưu để vận hành liên tục nhiều giờ trên CPU ở độ phân giải **1280x720 (≥ 25–30 FPS)**.

## 1. Cấu trúc thư mục

```text
pose_booth/
├── main.py                # Vòng lặp chính, xử lý phím tắt, chế độ mô phỏng & chống sập
├── config.py              # Toàn bộ tham số cấu hình (ngưỡng, thời gian, SIGMA, ROI, màu sắc)
├── pose_engine.py         # MediaPipe Pose (solutions + fallback Tasks API), cắt ROI & ThreadedCamera
├── similarity.py          # Thuật toán chấm điểm vector đoạn chi trên không gian pixel & làm mượt EMA
├── game_state.py          # Máy trạng thái (IDLE -> COUNTDOWN -> PLAYING -> RESULT) & ghép ảnh Collage
├── ui.py                  # Giao diện đôi 1280x720, kết xuất chữ tiếng Việt có cache & âm thanh
├── build_templates.py     # Công cụ tiền xử lý ảnh mẫu sang templates.json (kèm bộ sinh mẫu tự động)
├── templates.json         # Dữ liệu khung xương của các tư thế mẫu (sinh tự động bởi build_templates.py)
├── assets/
│   ├── poses/             # Ảnh tư thế mẫu (.png/.jpg) phân loại theo easy_ / medium_ / hard_
│   ├── fonts/             # Font chữ TrueType hỗ trợ tiếng Việt (Be Vietnam Pro, Roboto...)
│   └── sounds/            # Hiệu ứng âm thanh .wav (tùy chọn, tự động tổng hợp sóng âm nếu trống)
├── captures/              # Ảnh chụp khoảnh khắc PASS sạch và ảnh ghép collage chia theo từng phiên
├── tests/
│   └── test_similarity.py # Bộ kiểm thử đơn vị cho thuật toán chấm điểm và máy trạng thái
├── requirements.txt       # Danh sách thư viện Python đã ghim phiên bản ổn định
└── README.md              # Tài liệu hướng dẫn cài đặt, vận hành và hiệu chỉnh tại sự kiện
```

## 2. Hướng dẫn cài đặt và chạy ngay

### Bước 1: Tạo môi trường ảo và cài đặt thư viện (Python 3.10 – 3.11)

```bash
cd pose_booth
python -m venv .venv

# Kích hoạt môi trường ảo (Windows):
.venv\Scripts\activate
# Hoặc trên macOS / Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý về phiên bản MediaPipe Offline:**
> Trong `requirements.txt`, phiên bản `mediapipe==0.10.14` được ghim cố định vì đây là nhánh ổn định đóng gói sẵn mô hình `mp.solutions.pose` (`model_complexity=1`) ngay trong wheel, chạy **100% offline** mà không cần kết nối mạng hay tải file ngoài. Nếu bạn dùng phiên bản MediaPipe mới hơn chỉ có Tasks API (`PoseLandmarker`), hãy tải sẵn file `pose_landmarker_lite.task` đặt vào `assets/pose_landmarker_lite.task` — `pose_engine.py` sẽ tự động nhận diện và chuyển sang Tasks API.

### Bước 2: Khởi tạo kho tư thế mẫu (`templates.json`)

Nếu bạn chưa có sẵn ảnh chụp người mẫu, hãy chạy lệnh sau để hệ thống tự động vẽ **8 ảnh tư thế mẫu chuẩn** (chia đủ 3 mức `easy`, `medium`, `hard`) vào `assets/poses/` và xuất `templates.json`:

```bash
python build_templates.py --generate-samples
```

Khi bạn thêm ảnh thật (`.png` hoặc `.jpg`) vào `assets/poses/` (đặt tên bắt đầu bằng `easy_`, `medium_` hoặc `hard_`, ví dụ `easy_superman.jpg`), hãy chạy lại:

```bash
python build_templates.py
```

### Bước 3: Chạy kiểm thử và khởi động trò chơi

```bash
# 1. Chạy bộ kiểm thử đơn vị (Unit Tests)
python -m unittest discover -s tests -v

# 2. Chạy chế độ mô phỏng tự động (không cần webcam hay người đứng trước máy)
python main.py --simulate

# 3. Chạy chính thức với Webcam tại sự kiện
python main.py --camera 0
```

## 3. Bảng phím tắt dành cho điều phối viên gian hàng

| Phím tắt | Chức năng |
| :--- | :--- |
| `SPACE` | Bắt đầu lượt chơi ngay lập tức (khi ở màn hình chờ `IDLE` / `RESULT`) hoặc bỏ qua tư thế đang chơi |
| `N` | Bỏ qua tư thế hiện tại (tính là trượt tư thế đó và chuyển sang hình kế tiếp) |
| `R` | Reset trò chơi ngay lập tức về màn hình chờ `IDLE` |
| `D` | Bật / tắt bảng thông số kỹ thuật Debug (xem FPS thực tế, sai số góc từng đoạn chi) |
| `F` | Bật / tắt chế độ Toàn màn hình (Fullscreen) |
| `Q` / `ESC` | Thoát ứng dụng an toàn |

## 4. Hướng dẫn hiệu chỉnh tại chỗ (On-site Calibration Guide)

### 4.1. Cách chỉnh `SIGMA_DEG` và ngưỡng điểm trong `config.py`
Công thức tính điểm của hệ thống là:
$$\text{score} = \exp\left(-\left(\frac{E}{\text{SIGMA\_DEG}}\right)^2\right)$$
trong đó $E$ là sai số góc trung bình có trọng số (đơn vị: độ).
- **Mặc định (`SIGMA_DEG = 32.0`, `PASS_THRESHOLD = 0.80`):** Đòi hỏi sai số trung bình $E \le 15.1^\circ$ và không có đoạn chi chính nào lệch quá `MAX_CORE_LIMB_ERROR_DEG = 45.0°`.
- **Nếu khách tham quan thấy hơi khó (trẻ em hoặc người mặc đồ rộng):** Tăng `SIGMA_DEG` lên `36.0` – `40.0` (cho phép lệch trung bình $17^\circ - 19^\circ$ vẫn đạt 80%), hoặc tăng `MAX_CORE_LIMB_ERROR_DEG` lên `52.0`.
- **Nếu muốn tăng độ thử thách cho vòng chung kết:** Giảm `SIGMA_DEG` xuống `27.0` – `29.0` và tăng `HOLD_DURATION_SEC` từ `2.0` lên `2.5` giây.

### 4.2. Chuẩn bị ảnh tư thế mẫu đạt chuẩn AI
1. **Thấy rõ toàn thân từ đầu đến cổ chân:** Không dùng ảnh cắt ngang đùi hoặc ảnh tự sướng nửa người vì thuật toán cần đủ các đoạn đùi và cẳng chân.
2. **Nền đơn giản, tương phản tốt với trang phục:** Tránh nền có nhiều người đứng phía sau để `build_templates.py` trích xuất khung xương chính xác nhất.
3. **Tránh các tư thế tự che khuất (Self-occlusion):** Ưu tiên các tư thế dang tay, chống hông, bước rộng hoặc tạo hình rõ ràng trên mặt phẳng 2D song song với ống kính thay vì chắp tay sau lưng hoặc xoay dọc người 90 độ.

### 4.3. Ánh sáng và vị trí đặt Webcam tại gian hàng
1. **Khoảng cách và độ cao Webcam:** Đặt webcam ở độ cao ngang ngực người lớn (**1.1m – 1.3m** so với mặt sàn), góc nhìn thẳng (không chúc quá sâu từ trần nhà xuống để tránh biến dạng phối cảnh).
2. **Dán băng keo đánh dấu vị trí đứng:** Dán chữ `X` nổi bật trên sàn cách webcam **1.8m – 2.2m** sao cho người cao 1.50m – 1.85m đều lọt trọn từ đầu đến mắt cá chân vào khung hướng dẫn trung tâm (ROI) trên màn hình.
3. **Ánh sáng:** Đặt đèn chiếu sáng (key light) từ phía màn hình hướng vào mặt trước của người chơi. **Tuyệt đối tránh ngược sáng** (không đặt webcam quay thẳng ra cửa kính nắng gắt hoặc màn hình LED sân khấu cực sáng phía lưng người chơi).
