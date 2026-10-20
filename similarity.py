"""
similarity.py - Thuật toán tính độ tương đồng tư thế (Pose Similarity Engine).

Đặc điểm kỹ thuật:
1. Quy đổi tọa độ chuẩn hóa (0..1) sang PIXEL thực tế (nhân width, height) TRƯỚC khi tính
   vector đoạn chi để tránh biến dạng góc do tỉ lệ khung hình (aspect ratio).
2. Biểu diễn tư thế qua các vector hướng 2D của từng đoạn chi (vai-khuỷu, khuỷu-cổ tay,
   thân, đùi, cẳng chân, đầu-vai). Bất biến tuyệt đối với phép tịnh tiến và phóng đại.
3. Trọng số mỗi đoạn dựa trên min(visibility_A, visibility_B) * base_weight; loại bỏ
   các đoạn có visibility < MIN_LANDMARK_VISIBILITY (0.5).
4. Hàm điểm số phi tuyến mượt: score = exp(-(E / SIGMA)^2), với SIGMA ≈ 32 độ.
5. Chống gian lận may rủi: nếu bất kỳ đoạn chi chính (core limb) nào lệch > 45 độ,
   giới hạn điểm tối đa ở 0.79 (79%).
6. Hỗ trợ MIRROR_INVARIANT: tự động chấm điểm với cả mẫu gốc và mẫu lật ngang
   (hoán đổi nhãn trái/phải), lấy kết quả cao hơn.
7. Bộ lọc EMA (Exponential Moving Average) làm mượt landmark và điểm số.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

import config


@dataclass
class Landmark2D:
    """Biểu diễn một khớp xương 2D kèm độ tin cậy hiển thị (visibility)."""
    x: float          # Tọa độ chuẩn hóa [0..1] theo chiều ngang
    y: float          # Tọa độ chuẩn hóa [0..1] theo chiều dọc
    visibility: float = 1.0

    def to_pixel(self, width: int, height: int) -> Tuple[float, float]:
        """Chuyển đổi sang tọa độ pixel thực tế trước khi tính góc."""
        return (self.x * float(width), self.y * float(height))


# Kiểu dữ liệu ánh xạ từ chỉ số khớp MediaPipe (int) sang Landmark2D
PoseLandmarks = Dict[int, Landmark2D]


@dataclass
class SimilarityResult:
    """Kết quả chi tiết của một lần chấm điểm tư thế."""
    score: float                                 # Điểm đã chuẩn hóa [0.0 .. 1.0]
    raw_score: float                             # Điểm trước khi chặn trần gian lận
    weighted_mean_error_deg: float               # Sai số góc trung bình có trọng số E (độ)
    limb_errors_deg: Dict[str, float] = field(default_factory=dict)
    valid_limb_count: int = 0
    insufficient_landmarks: bool = False         # True nếu người chơi đứng khuất/quá gần
    capped_by_outlier: bool = False              # True nếu bị khóa ở 79% do 1 chi lệch > 45 độ
    outlier_limbs: List[str] = field(default_factory=list)
    used_mirrored_template: bool = False         # True nếu bản lật gương cho điểm cao hơn


def mirror_pose_landmarks(landmarks: PoseLandmarks) -> PoseLandmarks:
    """
    Tạo bản sao đối xứng gương (lật ngang x -> 1 - x) và hoán đổi cặp khớp Trái <-> Phải.
    Dùng cho chế độ MIRROR_INVARIANT để người chơi bắt chước xuôi hay ngược đều được chấm công bằng.
    """
    flipped: PoseLandmarks = {}
    for idx, lm in landmarks.items():
        flipped[idx] = Landmark2D(
            x=1.0 - lm.x,
            y=lm.y,
            visibility=lm.visibility,
        )

    swapped: PoseLandmarks = dict(flipped)
    for left_idx, right_idx in config.LEFT_RIGHT_SWAP_PAIRS:
        left_lm = flipped.get(left_idx)
        right_lm = flipped.get(right_idx)
        if left_lm is not None and right_lm is not None:
            swapped[left_idx] = right_lm
            swapped[right_idx] = left_lm
        elif left_lm is not None:
            swapped[right_idx] = left_lm
            swapped.pop(left_idx, None)
        elif right_lm is not None:
            swapped[left_idx] = right_lm
            swapped.pop(right_idx, None)

    return swapped


def compute_vector_angle_deg(
    vec_a: Tuple[float, float],
    vec_b: Tuple[float, float],
) -> Optional[float]:
    """
    Tính góc lệch (0..180 độ) giữa hai vector 2D trong không gian pixel.
    Trả về None nếu một trong hai vector có độ dài suy biến (~0).
    """
    ax, ay = vec_a
    bx, by = vec_b
    norm_a = math.hypot(ax, ay)
    norm_b = math.hypot(bx, by)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return None

    cos_theta = (ax * bx + ay * by) / (norm_a * norm_b)
    # Kẹp giá trị trong [-1.0, 1.0] để tránh lỗi số học dấu phẩy động với math.acos
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def _evaluate_single_orientation(
    player_landmarks: PoseLandmarks,
    template_landmarks: PoseLandmarks,
    player_size: Tuple[int, int],
    template_size: Tuple[int, int],
    sigma_deg: float,
    min_visibility: float,
    min_valid_limbs: int,
    max_core_error_deg: float,
    cheat_cap_score: float,
) -> SimilarityResult:
    """
    Tính điểm tương đồng giữa người chơi và một hướng của tư thế mẫu.
    """
    p_w, p_h = player_size
    t_w, t_h = template_size

    limb_errors: Dict[str, float] = {}
    outlier_limbs: List[str] = []
    weighted_error_sum: float = 0.0
    weight_sum: float = 0.0
    valid_limbs: int = 0

    for limb_name, idx_a, idx_b, base_weight, is_core in config.LIMB_DEFINITIONS:
        p_a = player_landmarks.get(idx_a)
        p_b = player_landmarks.get(idx_b)
        t_a = template_landmarks.get(idx_a)
        t_b = template_landmarks.get(idx_b)

        if p_a is None or p_b is None or t_a is None or t_b is None:
            continue

        # Độ tin cậy của đoạn chi tính theo đầu mút thấp hơn
        p_vis = min(p_a.visibility, p_b.visibility)
        t_vis = min(t_a.visibility, t_b.visibility)
        if p_vis < min_visibility or t_vis < min_visibility:
            continue

        # Bước 2 bắt buộc: chuyển sang PIXEL trước khi lấy hiệu vector và tính góc
        pa_x, pa_y = p_a.to_pixel(p_w, p_h)
        pb_x, pb_y = p_b.to_pixel(p_w, p_h)
        ta_x, ta_y = t_a.to_pixel(t_w, t_h)
        tb_x, tb_y = t_b.to_pixel(t_w, t_h)

        vec_player = (pb_x - pa_x, pb_y - pa_y)
        vec_template = (tb_x - ta_x, tb_y - ta_y)

        err_deg = compute_vector_angle_deg(vec_player, vec_template)
        if err_deg is None:
            continue

        valid_limbs += 1
        limb_errors[limb_name] = err_deg

        # Trọng số đoạn = min(visibility người chơi) * trọng số cơ sở của đoạn
        segment_weight = p_vis * base_weight
        weighted_error_sum += segment_weight * err_deg
        weight_sum += segment_weight

        # Kiểm tra điều kiện gian lận / sai tư thế cục bộ trên các đoạn chi chính
        if is_core and err_deg > max_core_error_deg:
            outlier_limbs.append(limb_name)

    if valid_limbs < min_valid_limbs or weight_sum <= 1e-6:
        return SimilarityResult(
            score=0.0,
            raw_score=0.0,
            weighted_mean_error_deg=180.0,
            limb_errors_deg=limb_errors,
            valid_limb_count=valid_limbs,
            insufficient_landmarks=True,
            capped_by_outlier=False,
            outlier_limbs=outlier_limbs,
            used_mirrored_template=False,
        )

    mean_error_deg = weighted_error_sum / weight_sum
    # Chuyển sang Cosine Similarity tự nhiên theo mắt nhìn con người:
    err_rad = math.radians(mean_error_deg)
    cos_sim = (1.0 + math.cos(err_rad)) / 2.0
    raw_score = float(max(0.0, min(1.0, cos_sim ** 1.6)))

    capped = len(outlier_limbs) > 0
    final_score = min(raw_score, cheat_cap_score) if capped else raw_score

    return SimilarityResult(
        score=final_score,
        raw_score=raw_score,
        weighted_mean_error_deg=mean_error_deg,
        limb_errors_deg=limb_errors,
        valid_limb_count=valid_limbs,
        insufficient_landmarks=False,
        capped_by_outlier=capped and (raw_score > cheat_cap_score),
        outlier_limbs=outlier_limbs,
        used_mirrored_template=False,
    )


def evaluate_pose_similarity(
    player_landmarks: Optional[PoseLandmarks],
    template_landmarks: Optional[PoseLandmarks],
    player_size: Tuple[int, int] = (config.PANEL_WIDTH, config.PANEL_HEIGHT),
    template_size: Tuple[int, int] = (config.PANEL_WIDTH, config.PANEL_HEIGHT),
    sigma_deg: float = config.SIGMA_DEG,
    min_visibility: float = config.MIN_LANDMARK_VISIBILITY,
    min_valid_limbs: int = config.MIN_VALID_LIMBS,
    max_core_error_deg: float = config.MAX_CORE_LIMB_ERROR_DEG,
    cheat_cap_score: float = config.CHEAT_CAP_SCORE,
    mirror_invariant: bool = config.MIRROR_INVARIANT,
) -> SimilarityResult:
    """
    Hàm chính chấm điểm tương đồng tư thế giữa người chơi và mẫu.
    Hỗ trợ MIRROR_INVARIANT (thử cả mẫu gốc và mẫu lật ngang, lấy điểm cao hơn).
    """
    if not player_landmarks or not template_landmarks:
        return SimilarityResult(
            score=0.0,
            raw_score=0.0,
            weighted_mean_error_deg=180.0,
            valid_limb_count=0,
            insufficient_landmarks=True,
        )

    direct_res = _evaluate_single_orientation(
        player_landmarks=player_landmarks,
        template_landmarks=template_landmarks,
        player_size=player_size,
        template_size=template_size,
        sigma_deg=sigma_deg,
        min_visibility=min_visibility,
        min_valid_limbs=min_valid_limbs,
        max_core_error_deg=max_core_error_deg,
        cheat_cap_score=cheat_cap_score,
    )

    if not mirror_invariant:
        return direct_res

    mirrored_template = mirror_pose_landmarks(template_landmarks)
    mirror_res = _evaluate_single_orientation(
        player_landmarks=player_landmarks,
        template_landmarks=mirrored_template,
        player_size=player_size,
        template_size=template_size,
        sigma_deg=sigma_deg,
        min_visibility=min_visibility,
        min_valid_limbs=min_valid_limbs,
        max_core_error_deg=max_core_error_deg,
        cheat_cap_score=cheat_cap_score,
    )
    mirror_res.used_mirrored_template = True

    # Nếu một bên không đủ khớp còn bên kia đủ khớp thì ưu tiên bên đủ khớp
    if direct_res.insufficient_landmarks and not mirror_res.insufficient_landmarks:
        return mirror_res
    if mirror_res.insufficient_landmarks and not direct_res.insufficient_landmarks:
        return direct_res

    # So sánh điểm sau chặn trần; nếu bằng nhau (ví dụ cùng bị chặn ở 0.79) thì chọn bên có sai số góc nhỏ hơn
    if (mirror_res.score > direct_res.score + 1e-6) or (
        abs(mirror_res.score - direct_res.score) <= 1e-6
        and mirror_res.weighted_mean_error_deg < direct_res.weighted_mean_error_deg
    ):
        return mirror_res

    return direct_res


class LandmarkEMASmoother:
    """Bộ lọc làm mượt Exponential Moving Average (EMA) cho tọa độ landmark."""

    def __init__(self, alpha: float = config.EMA_ALPHA_LANDMARKS) -> None:
        self.alpha: float = max(0.05, min(1.0, alpha))
        self._state: PoseLandmarks = {}

    def reset(self) -> None:
        """Xóa trạng thái lọc khi mất người chơi hoặc đổi phiên."""
        self._state.clear()

    def update(self, current: Optional[PoseLandmarks]) -> Optional[PoseLandmarks]:
        """Cập nhật và trả về bộ landmark đã làm mượt thông minh (chống rung lắc motion blur)."""
        if current is None or len(current) == 0:
            self.reset()
            return None

        if not self._state:
            self._state = {
                idx: Landmark2D(lm.x, lm.y, lm.visibility)
                for idx, lm in current.items()
            }
            return dict(self._state)

        smoothed: PoseLandmarks = {}
        for idx, lm in current.items():
            prev = self._state.get(idx)
            if prev is None:
                smoothed[idx] = Landmark2D(lm.x, lm.y, lm.visibility)
            else:
                # Tính khoảng cách di chuyển giữa 2 khung hình
                dist = math.hypot(lm.x - prev.x, lm.y - prev.y)
                
                # Nếu di chuyển li ti (< 0.005 khung hình) -> giữ nguyên để chống rung vi mô
                if dist < 0.005:
                    eff_alpha = 0.08
                else:
                    # Nếu đang cử động, điều chỉnh độ mượt theo độ tin cậy (visibility)
                    eff_alpha = self.alpha * max(0.4, min(1.0, lm.visibility))

                smoothed[idx] = Landmark2D(
                    x=eff_alpha * lm.x + (1.0 - eff_alpha) * prev.x,
                    y=eff_alpha * lm.y + (1.0 - eff_alpha) * prev.y,
                    visibility=0.5 * lm.visibility + 0.5 * prev.visibility,
                )
        self._state = smoothed
        return dict(self._state)


class ScoreEMASmoother:
    """Bộ lọc làm mượt EMA cho điểm Similarity Score chống nhấp nháy thanh tiến trình."""

    def __init__(self, alpha: float = config.EMA_ALPHA_SCORE) -> None:
        self.alpha: float = max(0.05, min(1.0, alpha))
        self._value: Optional[float] = None

    def reset(self) -> None:
        """Reset điểm làm mượt khi chuyển sang tư thế mới."""
        self._value = None

    def update(self, new_score: float) -> float:
        """Cập nhật điểm mới và trả về điểm đã làm mượt."""
        clamped = max(0.0, min(1.0, float(new_score)))
        if self._value is None:
            self._value = clamped
        else:
            self._value = self.alpha * clamped + (1.0 - self.alpha) * self._value
        return self._value