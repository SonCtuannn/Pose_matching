"""
config.py - Cấu hình toàn cục cho ứng dụng TRẠM 2: POSE MATCHING – SAO Y BẢN CHÍNH.

Mọi tham số vận hành tại gian hàng (ngưỡng điểm, thời gian giữ, độ phân giải,
tham số thuật toán Similarity, ROI, màu sắc, đường dẫn) đều tập trung tại đây
để kỹ thuật viên dễ dàng tinh chỉnh tại sự kiện.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

# ==============================================================================
# 1. THÔNG TIN GIAN HÀNG & ĐƯỜNG DẪN HỆ THỐNG
# ==============================================================================
BOOTH_TITLE: str = "TRẠM 2: POSE MATCHING – SAO Y BẢN CHÍNH"
BOOTH_TAGLINE: str = "Đọ dáng cực lầy – Chấm điểm bằng AI ngay tức thì!"

BASE_DIR: Path = Path(__file__).resolve().parent
ASSETS_DIR: Path = BASE_DIR / "assets"
POSES_DIR: Path = ASSETS_DIR / "poses"
FONTS_DIR: Path = ASSETS_DIR / "fonts"
SOUNDS_DIR: Path = ASSETS_DIR / "sounds"
CAPTURES_DIR: Path = BASE_DIR / "captures"
TEMPLATES_JSON_PATH: Path = BASE_DIR / "templates.json"
POSE_LANDMARKER_TASK_PATH: Path = ASSETS_DIR / "pose_landmarker_lite.task"

# Danh sách đường dẫn font ưu tiên (hỗ trợ tiếng Việt đầy đủ dấu)
FONT_CANDIDATES: List[str] = [
    str(FONTS_DIR / "BeVietnamPro-Bold.ttf"),
    str(FONTS_DIR / "Roboto-Bold.ttf"),
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]

# ==============================================================================
# 2. CẤU HÌNH CAMERA & HIỂN THỊ
# ==============================================================================
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
TARGET_FPS: int = 30
CAMERA_RECONNECT_INTERVAL_SEC: float = 2.0
MIRROR_WEBCAM: bool = True  # Lật ngang khung hình webcam như soi gương

# Bố cục màn hình đôi (1280x720 chia 2 nửa 640x720)
PANEL_WIDTH: int = FRAME_WIDTH // 2
PANEL_HEIGHT: int = FRAME_HEIGHT

# Vùng đứng chơi ưu tiên (ROI chuẩn hóa [x1, y1, x2, y2] trên khung webcam 640x720)
# Dùng để lọc nhiễu đám đông xung quanh gian hàng, chỉ bắt người đứng giữa khung
USE_CENTER_ROI: bool = False
PLAYER_ROI_NORM: Tuple[float, float, float, float] = (0.12, 0.05, 0.88, 0.96)

# ==============================================================================
# 3. LUẬT CHƠI & THỜI GIAN (STATE MACHINE)
# ==============================================================================
NUM_POSES_PER_GAME: int = 5          # Tổng số tư thế mỗi lượt chơi
REQUIRED_PASSES: int = 3             # Số tư thế cần PASS để thắng (3/5)
GAME_DURATION_SEC: float = 45.0      # Đồng hồ tổng 45 giây
POSE_SOFT_TIMEOUT_SEC: float = 15.0  # Giới hạn mềm mỗi tư thế (quá giờ tự bỏ qua)
COUNTDOWN_SEC: float = 3.0           # Đếm ngược 3-2-1 trước khi bắt đầu
RESULT_SCREEN_SEC: float = 10.0      # Hiển thị màn hình kết quả 10 giây rồi về IDLE

# Tự động bắt đầu khi phát hiện người chơi đứng đúng ROI ở màn IDLE
AUTO_START_ENABLED: bool = True
AUTO_START_HOLD_SEC: float = 1.8     # Đứng đủ khớp trong khung 1.8 giây sẽ tự vào COUNTDOWN

# Cơ cấu trộn độ khó cho 5 tư thế trong 1 lượt chơi
DIFFICULTY_MIX: Dict[str, int] = {
    "easy": 2,
    "medium": 2,
    "hard": 1,
}

# ==============================================================================
# 4. NGƯỠNG CHẤM ĐIỂM & BỘ ĐẾM GIỮ TƯ THẾ (HOLD TIMER)
# ==============================================================================
PASS_THRESHOLD: float = 0.80         # Đạt >= 80% để bắt đầu tính giờ giữ
HYSTERESIS_THRESHOLD: float = 0.78   # Ngưỡng thoát khi đang giữ (chống reset oan do rung nhẹ)
HOLD_DURATION_SEC: float = 2.0       # Phải giữ liên tục 2.0 giây
HOLD_GRACE_PERIOD_SEC: float = 0.25  # Dung sai tụt điểm ngắn tối đa 0.25 giây không bị mất tiến trình
FLASH_DURATION_SEC: float = 0.35     # Thời gian hiệu ứng chớp trắng khi chụp ảnh PASS

# ==============================================================================
# 5. THUẬT TOÁN SIMILARITY & MEDIAPIPE POSE
# ==============================================================================
MP_MODEL_COMPLEXITY: int = 1         # 1 = cân bằng tốc độ CPU (>=25 FPS) và độ chính xác
MP_MIN_DETECTION_CONFIDENCE: float = 0.5
MP_MIN_TRACKING_CONFIDENCE: float = 0.5

# Các khớp chính sử dụng (bỏ qua bàn tay và các điểm mặt chi tiết, chỉ giữ mũi 0)
KEY_LANDMARK_INDICES: Tuple[int, ...] = (
    0,           # Mũi (định hướng đầu)
    11, 12,      # Vai trái, Vai phải
    13, 14,      # Khuỷu tay trái, Khuỷu tay phải
    15, 16,      # Cổ tay trái, Cổ tay phải
    23, 24,      # Hông trái, Hông phải
    25, 26,      # Đầu gối trái, Đầu gối phải
    27, 28,      # Cổ chân trái, Cổ chân phải
)

# Danh sách các đoạn chi (tên đoạn, khớp đầu, khớp cuối, trọng số cơ sở, có phải đoạn chính chặn gian lận không)
LIMB_DEFINITIONS: List[Tuple[str, int, int, float, bool]] = [
    ("Bắp tay trái", 11, 13, 1.2, True),
    ("Cẳng tay trái", 13, 15, 1.2, True),
    ("Bắp tay phải", 12, 14, 1.2, True),
    ("Cẳng tay phải", 14, 16, 1.2, True),
    ("Thân trái", 11, 23, 1.0, True),
    ("Thân phải", 12, 24, 1.0, True),
    ("Đùi trái", 23, 25, 1.1, True),
    ("Cẳng chân trái", 25, 27, 1.1, True),
    ("Đùi phải", 24, 26, 1.1, True),
    ("Cẳng chân phải", 26, 28, 1.1, True),
    ("Ngang vai", 11, 12, 0.8, False),
    ("Ngang hông", 23, 24, 0.8, False),
    ("Cổ - Đầu trái", 11, 0, 0.6, False),
    ("Cổ - Đầu phải", 12, 0, 0.6, False),
]

# Cặp hoán đổi khớp Trái <-> Phải khi lật gương tư thế mẫu (MIRROR_INVARIANT)
LEFT_RIGHT_SWAP_PAIRS: List[Tuple[int, int]] = [
    (11, 12),
    (13, 14),
    (15, 16),
    (23, 24),
    (25, 26),
    (27, 28),
]

MIN_LANDMARK_VISIBILITY: float = 0.35  # Giảm từ 0.5 xuống 0.35 để bắt khớp nhạy hơn nhiều
MIN_VALID_LIMBS: int = 4              # Giảm từ 6 xuống 4 (hỗ trợ cả khi test ngồi/đứng thấy nửa người)        # Tối thiểu 6 đoạn chi rõ ràng mới chấm điểm

# Tham số hàm Gauss: score = exp(-(E / SIGMA)^2)
# Với SIGMA = 32.0 độ -> sai số trung bình E = 15 độ cho score ~ 0.803 (80.3%)
SIGMA_DEG: float = 32.0

# Chặn gian lận may rủi: bất kỳ đoạn chi chính (is_core=True) nào lệch > 45 độ
# thì khóa điểm tối đa ở 79% (không thể qua ngưỡng 80%)
MAX_CORE_LIMB_ERROR_DEG: float = 65.0  # Tăng từ 45.0 lên 65.0 độ
CHEAT_CAP_SCORE: float = 0.79

# Cho phép khớp cả bản gốc lẫn bản đối xứng gương của tư thế mẫu
MIRROR_INVARIANT: bool = True

# Hệ số làm mượt EMA (Exponential Moving Average): alpha càng nhỏ càng mượt, càng lớn càng nhạy
# Giảm alpha để khung xương chuyển động "đầm", êm ru và triệt tiêu rung giật
EMA_ALPHA_LANDMARKS: float = 0.28
EMA_ALPHA_SCORE: float = 0.25

# ==============================================================================
# 6. BẢNG MÀU GIAO DIỆN (BGR cho OpenCV) & PHÍM TẮT
# ==============================================================================
COLOR_BG_DARK: Tuple[int, int, int] = (28, 20, 16)
COLOR_PANEL_HEADER: Tuple[int, int, int] = (48, 32, 24)
COLOR_NEON_GREEN: Tuple[int, int, int] = (80, 240, 60)
COLOR_NEON_YELLOW: Tuple[int, int, int] = (0, 215, 255)
COLOR_NEON_RED: Tuple[int, int, int] = (60, 60, 255)
COLOR_NEON_CYAN: Tuple[int, int, int] = (255, 220, 40)
COLOR_NEON_MAGENTA: Tuple[int, int, int] = (220, 60, 255)
COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)
COLOR_GRAY: Tuple[int, int, int] = (170, 170, 170)

# Mã phím tắt
KEY_SPACE: int = 32
KEY_ESC: int = 27