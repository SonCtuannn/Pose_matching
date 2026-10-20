"""
build_templates.py - Công cụ tiền xử lý ảnh mẫu (Offline Template Builder).

Chức năng:
1. Quét thư mục `assets/poses/` để tìm các ảnh tư thế mẫu (.png, .jpg, .jpeg).
2. Chạy MediaPipe Pose (`static_image_mode=True`) trên từng ảnh mẫu.
3. Kiểm tra chất lượng mẫu: cảnh báo rõ ràng nếu ảnh mẫu không phát hiện đủ các khớp chính
   (visibility < 0.5).
4. Xuất file `templates.json` chứa danh sách tư thế kèm tên hiển thị, độ khó
   (`easy` / `medium` / `hard`), kích thước ảnh gốc và tọa độ chuẩn hóa của các khớp.
5. Đặc biệt: Hỗ trợ cờ `--generate-samples` (hoặc tự kích hoạt khi `assets/poses/` trống)
   để sinh tự động 8 tư thế mẫu đặc sắc (meme, anime, thể thao) kèm ảnh minh họa rõ nét
   và `templates.json` chuẩn xác, giúp gian hàng chạy thử ngay lập tức.
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

import config
from pose_engine import PoseExtractor
from similarity import Landmark2D, PoseLandmarks

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("pose_booth.build_templates")


# ==============================================================================
# DỮ LIỆU 8 TƯ THẾ MẪU MẶC ĐỊNH (Dùng khi tạo bộ mẫu khởi đầu hoặc chạy mô phỏng)
# Tọa độ chuẩn hóa (x, y) trong khung 640x720, thiết kế giải phẫu chuẩn xác
# ==============================================================================
BUILTIN_SAMPLE_POSES: List[Dict[str, Any]] = [
    {
        "id": "easy_victory_v",
        "name": "Siêu Nhân Chiến Thắng",
        "difficulty": "easy",
        "filename": "easy_victory_v.png",
        "theme_color": (60, 180, 255),
        "landmarks": {
            0: (0.50, 0.20, 0.99),
            11: (0.39, 0.33, 0.99), 12: (0.61, 0.33, 0.99),
            13: (0.28, 0.22, 0.98), 14: (0.72, 0.22, 0.98),
            15: (0.20, 0.10, 0.97), 16: (0.80, 0.10, 0.97),
            23: (0.42, 0.56, 0.99), 24: (0.58, 0.56, 0.99),
            25: (0.38, 0.74, 0.98), 26: (0.62, 0.74, 0.98),
            27: (0.35, 0.90, 0.97), 28: (0.65, 0.90, 0.97),
        },
    },
    {
        "id": "easy_muscle_flex",
        "name": "Lực Sĩ Khoe Cơ",
        "difficulty": "easy",
        "filename": "easy_muscle_flex.png",
        "theme_color": (80, 220, 100),
        "landmarks": {
            0: (0.50, 0.21, 0.99),
            11: (0.38, 0.34, 0.99), 12: (0.62, 0.34, 0.99),
            13: (0.22, 0.34, 0.98), 14: (0.78, 0.34, 0.98),
            15: (0.25, 0.19, 0.97), 16: (0.75, 0.19, 0.97),
            23: (0.42, 0.57, 0.99), 24: (0.58, 0.57, 0.99),
            25: (0.39, 0.75, 0.98), 26: (0.61, 0.75, 0.98),
            27: (0.37, 0.91, 0.97), 28: (0.63, 0.91, 0.97),
        },
    },
    {
        "id": "easy_teapot_sassy",
        "name": "Ấm Trà Chảnh Chọe",
        "difficulty": "easy",
        "filename": "easy_teapot_sassy.png",
        "theme_color": (255, 140, 80),
        "landmarks": {
            0: (0.51, 0.21, 0.99),
            11: (0.39, 0.34, 0.99), 12: (0.61, 0.34, 0.99),
            13: (0.27, 0.45, 0.98), 14: (0.75, 0.25, 0.98),
            15: (0.39, 0.53, 0.97), 16: (0.85, 0.14, 0.97),
            23: (0.41, 0.56, 0.99), 24: (0.59, 0.56, 0.99),
            25: (0.40, 0.74, 0.98), 26: (0.60, 0.74, 0.98),
            27: (0.39, 0.90, 0.97), 28: (0.61, 0.90, 0.97),
        },
    },
    {
        "id": "medium_usain_bolt",
        "name": "Tia Chớp Usain Bolt",
        "difficulty": "medium",
        "filename": "medium_usain_bolt.png",
        "theme_color": (0, 215, 255),
        "landmarks": {
            0: (0.48, 0.21, 0.99),
            11: (0.38, 0.34, 0.99), 12: (0.60, 0.32, 0.99),
            13: (0.24, 0.25, 0.98), 14: (0.74, 0.24, 0.98),
            15: (0.12, 0.15, 0.97), 16: (0.58, 0.16, 0.97),
            23: (0.42, 0.56, 0.99), 24: (0.58, 0.56, 0.99),
            25: (0.35, 0.74, 0.98), 26: (0.65, 0.74, 0.98),
            27: (0.30, 0.91, 0.97), 28: (0.70, 0.91, 0.97),
        },
    },
    {
        "id": "medium_kamehameha",
        "name": "Chưởng Kamehameha",
        "difficulty": "medium",
        "filename": "medium_kamehameha.png",
        "theme_color": (255, 190, 0),
        "landmarks": {
            0: (0.48, 0.24, 0.99),
            11: (0.38, 0.36, 0.99), 12: (0.60, 0.36, 0.99),
            13: (0.24, 0.42, 0.98), 14: (0.44, 0.44, 0.98),
            15: (0.14, 0.38, 0.97), 16: (0.20, 0.41, 0.97),
            23: (0.40, 0.58, 0.99), 24: (0.58, 0.58, 0.99),
            25: (0.30, 0.74, 0.98), 26: (0.68, 0.74, 0.98),
            27: (0.26, 0.91, 0.97), 28: (0.72, 0.91, 0.97),
        },
    },
    {
        "id": "medium_egyptian_walk",
        "name": "Vũ Điệu Ai Cập",
        "difficulty": "medium",
        "filename": "medium_egyptian_walk.png",
        "theme_color": (180, 120, 255),
        "landmarks": {
            0: (0.50, 0.21, 0.99),
            11: (0.38, 0.34, 0.99), 12: (0.62, 0.34, 0.99),
            13: (0.22, 0.34, 0.98), 14: (0.78, 0.34, 0.98),
            15: (0.22, 0.20, 0.97), 16: (0.78, 0.48, 0.97),
            23: (0.42, 0.56, 0.99), 24: (0.58, 0.56, 0.99),
            25: (0.36, 0.73, 0.98), 26: (0.64, 0.73, 0.98),
            27: (0.34, 0.90, 0.97), 28: (0.66, 0.90, 0.97),
        },
    },
    {
        "id": "hard_crane_kick",
        "name": "Hạc Quyền Karate",
        "difficulty": "hard",
        "filename": "hard_crane_kick.png",
        "theme_color": (90, 90, 255),
        "landmarks": {
            0: (0.50, 0.20, 0.99),
            11: (0.38, 0.33, 0.99), 12: (0.62, 0.33, 0.99),
            13: (0.22, 0.26, 0.98), 14: (0.78, 0.26, 0.98),
            15: (0.15, 0.36, 0.97), 16: (0.85, 0.36, 0.97),
            23: (0.43, 0.55, 0.99), 24: (0.57, 0.55, 0.99),
            25: (0.30, 0.60, 0.98), 26: (0.57, 0.74, 0.98),
            27: (0.35, 0.75, 0.97), 28: (0.57, 0.91, 0.97),
        },
    },
    {
        "id": "hard_jojo_star",
        "name": "Tuyệt Kỹ JoJo Pose",
        "difficulty": "hard",
        "filename": "hard_jojo_star.png",
        "theme_color": (230, 80, 250),
        "landmarks": {
            0: (0.52, 0.21, 0.99),
            11: (0.38, 0.35, 0.99), 12: (0.61, 0.32, 0.99),
            13: (0.25, 0.22, 0.98), 14: (0.76, 0.42, 0.98),
            15: (0.42, 0.14, 0.97), 16: (0.62, 0.52, 0.97),
            23: (0.41, 0.56, 0.99), 24: (0.57, 0.55, 0.99),
            25: (0.31, 0.73, 0.98), 26: (0.67, 0.72, 0.98),
            27: (0.25, 0.90, 0.97), 28: (0.74, 0.90, 0.97),
        },
    },
]


def render_sample_pose_card(
    landmarks_dict: Dict[int, Tuple[float, float, float]],
    theme_color: Tuple[int, int, int],
    width: int = config.PANEL_WIDTH,
    height: int = config.PANEL_HEIGHT,
) -> np.ndarray:
    """
    Vẽ hình nhân minh họa phong cách Neon Mannequin lên nền studio tối
    để tạo file ảnh mẫu trực quan trong `assets/poses/`.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Vẽ nền chuyển sắc nhẹ và lưới sân khấu
    for y in range(height):
        ratio = y / float(height)
        canvas[y, :] = (
            int(22 + 18 * ratio),
            int(16 + 14 * ratio),
            int(30 + 25 * (1.0 - ratio)),
        )

    for gx in range(0, width, 64):
        cv2.line(canvas, (gx, 0), (gx, height), (42, 34, 48), 1)
    for gy in range(0, height, 64):
        cv2.line(canvas, (0, gy), (width, gy), (42, 34, 48), 1)

    def pt_px(idx: int) -> Tuple[int, int]:
        x, y, _ = landmarks_dict[idx]
        return (int(x * width), int(y * height))

    # Vẽ đa giác thân (vai trái - vai phải - hông phải - hông trái)
    torso_pts = np.array([pt_px(11), pt_px(12), pt_px(24), pt_px(23)], dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [torso_pts], theme_color)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    cv2.polylines(canvas, [torso_pts], isClosed=True, color=theme_color, thickness=4, lineType=cv2.LINE_AA)

    # Vẽ các đoạn chi dày dặn có hào quang neon
    limb_pairs = [
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
    ]
    for idx_a, idx_b in limb_pairs:
        p1 = pt_px(idx_a)
        p2 = pt_px(idx_b)
        cv2.line(canvas, p1, p2, theme_color, 18, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, (255, 255, 255), 6, cv2.LINE_AA)

    # Vẽ đầu và cổ
    shoulder_mid = (
        (pt_px(11)[0] + pt_px(12)[0]) // 2,
        (pt_px(11)[1] + pt_px(12)[1]) // 2,
    )
    head_center = pt_px(0)
    cv2.line(canvas, shoulder_mid, head_center, theme_color, 12, cv2.LINE_AA)
    cv2.circle(canvas, head_center, 36, theme_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, head_center, 36, (255, 255, 255), 4, cv2.LINE_AA)

    # Vẽ các khớp tròn
    for idx in config.KEY_LANDMARK_INDICES:
        if idx == 0:
            continue
        p = pt_px(idx)
        cv2.circle(canvas, p, 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p, 11, theme_color, 3, cv2.LINE_AA)

    return canvas


def infer_difficulty_from_filename(stem: str) -> str:
    """Suy luận độ khó từ tiền tố tên file (mặc định là 'medium')."""
    lower = stem.lower()
    if lower.startswith("easy") or "_easy" in lower:
        return "easy"
    if lower.startswith("hard") or "_hard" in lower:
        return "hard"
    return "medium"


def format_display_name(stem: str) -> str:
    """Chuyển tên file `easy_victory_v` thành nhãn hiển thị nếu không có metadata."""
    cleaned = stem
    for prefix in ("easy_", "medium_", "hard_"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    return cleaned.replace("_", " ").strip().title()


def validate_landmarks_quality(landmarks: Optional[PoseLandmarks]) -> Tuple[bool, List[int]]:
    """
    Kiểm tra xem ảnh mẫu có phát hiện đủ các khớp chính với visibility >= 0.5 hay không.
    Trả về (hợp_lệ, danh_sách_khớp_thiếu_hoặc_mờ).
    """
    if not landmarks:
        return False, list(config.KEY_LANDMARK_INDICES)

    missing: List[int] = []
    for idx in config.KEY_LANDMARK_INDICES:
        lm = landmarks.get(idx)
        if lm is None or lm.visibility < config.MIN_LANDMARK_VISIBILITY:
            missing.append(idx)

    # Cho phép thiếu tối đa 2 khớp ngoại vi nhưng phải đủ tối thiểu 11/13 khớp chính
    is_valid = len(missing) <= 2
    return is_valid, missing


def generate_builtin_samples(poses_dir: Path, output_json: Path) -> List[Dict[str, Any]]:
    """
    Sinh bộ 8 ảnh tư thế mẫu chuẩn vào `assets/poses/` và ghi `templates.json`.
    Đảm bảo gian hàng luôn có sẵn dữ liệu chạy ngay kể cả khi chưa chụp ảnh mẫu thật.
    """
    poses_dir.mkdir(parents=True, exist_ok=True)
    templates_data: List[Dict[str, Any]] = []

    for item in BUILTIN_SAMPLE_POSES:
        img_path = poses_dir / item["filename"]
        card_bgr = render_sample_pose_card(
            landmarks_dict=item["landmarks"],
            theme_color=item["theme_color"],
            width=config.PANEL_WIDTH,
            height=config.PANEL_HEIGHT,
        )
        cv2.imwrite(str(img_path), card_bgr)

        serialized_landmarks: Dict[str, Dict[str, float]] = {}
        for idx, (lx, ly, lvis) in item["landmarks"].items():
            serialized_landmarks[str(idx)] = {
                "x": round(float(lx), 5),
                "y": round(float(ly), 5),
                "visibility": round(float(lvis), 4),
            }

        templates_data.append({
            "id": item["id"],
            "name": item["name"],
            "difficulty": item["difficulty"],
            "image_path": str(img_path.relative_to(config.BASE_DIR)),
            "width": config.PANEL_WIDTH,
            "height": config.PANEL_HEIGHT,
            "landmarks": serialized_landmarks,
            "warning": None,
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"templates": templates_data}, f, ensure_ascii=False, indent=2)

    logger.info(
        "Đã khởi tạo %d ảnh mẫu mặc định tại '%s' và lưu '%s'.",
        len(templates_data),
        poses_dir,
        output_json,
    )
    return templates_data


def build_templates_from_directory(
    poses_dir: Path = config.POSES_DIR,
    output_json: Path = config.TEMPLATES_JSON_PATH,
    force_generate_samples: bool = False,
) -> List[Dict[str, Any]]:
    """
    Quét toàn bộ ảnh trong `poses_dir`, trích xuất khung xương bằng MediaPipe
    (static_image_mode=True) và xuất ra `templates.json`.
    """
    poses_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(
        p for p in poses_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )

    if force_generate_samples or len(image_files) == 0:
        logger.info("Thư mục ảnh mẫu trống hoặc được yêu cầu tạo mới. Đang sinh bộ mẫu mặc định...")
        return generate_builtin_samples(poses_dir, output_json)

    # Ánh xạ metadata từ bộ mẫu mặc định nếu trùng tên file
    builtin_lookup: Dict[str, Dict[str, Any]] = {
        item["filename"]: item for item in BUILTIN_SAMPLE_POSES
    }

    extractor = PoseExtractor(static_image_mode=True, use_roi=False)
    templates_data: List[Dict[str, Any]] = []

    try:
        for img_path in image_files:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                logger.warning("Không thể đọc file ảnh: %s -> Bỏ qua.", img_path)
                continue

            h, w = bgr.shape[:2]
            landmarks = extractor.extract(bgr)

            # Nếu chạy trong môi trường không có mediapipe và ảnh thuộc bộ mẫu chuẩn, dùng landmark chuẩn
            if landmarks is None and img_path.name in builtin_lookup:
                raw_map = builtin_lookup[img_path.name]["landmarks"]
                landmarks = {
                    int(k): Landmark2D(x=v[0], y=v[1], visibility=v[2])
                    for k, v in raw_map.items()
                }

            is_valid, missing_indices = validate_landmarks_quality(landmarks)
            warning_msg: Optional[str] = None
            if not is_valid or landmarks is None:
                warning_msg = (
                    f"CẢNH BÁO: Ảnh '{img_path.name}' thiếu hoặc mờ các khớp {missing_indices}. "
                    "Hãy chọn ảnh thấy rõ toàn thân từ đầu đến cổ chân trên nền đơn giản!"
                )
                logger.warning(warning_msg)
                if landmarks is None:
                    continue

            builtin_meta = builtin_lookup.get(img_path.name)
            pose_id = builtin_meta["id"] if builtin_meta else img_path.stem
            pose_name = (
                builtin_meta["name"]
                if builtin_meta
                else format_display_name(img_path.stem)
            )
            difficulty = (
                builtin_meta["difficulty"]
                if builtin_meta
                else infer_difficulty_from_filename(img_path.stem)
            )

            serialized_landmarks = {
                str(idx): {
                    "x": round(lm.x, 5),
                    "y": round(lm.y, 5),
                    "visibility": round(lm.visibility, 4),
                }
                for idx, lm in landmarks.items()
            }

            templates_data.append({
                "id": pose_id,
                "name": pose_name,
                "difficulty": difficulty,
                "image_path": str(img_path.relative_to(config.BASE_DIR)),
                "width": w,
                "height": h,
                "landmarks": serialized_landmarks,
                "warning": warning_msg,
            })
    finally:
        extractor.close()

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"templates": templates_data}, f, ensure_ascii=False, indent=2)

    logger.info(
        "Đã trích xuất thành công %d/%d tư thế mẫu vào '%s'.",
        len(templates_data),
        len(image_files),
        output_json,
    )
    return templates_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Công cụ tiền xử lý ảnh tư thế mẫu sang templates.json"
    )
    parser.add_argument(
        "--poses-dir",
        type=Path,
        default=config.POSES_DIR,
        help="Thư mục chứa ảnh tư thế mẫu (.png/.jpg)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.TEMPLATES_JSON_PATH,
        help="Đường dẫn file templates.json đầu ra",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Tự động sinh 8 ảnh tư thế mẫu chuẩn và tạo templates.json",
    )
    args = parser.parse_args()
    build_templates_from_directory(
        poses_dir=args.poses_dir,
        output_json=args.output,
        force_generate_samples=args.generate_samples,
    )


if __name__ == "__main__":
    main()