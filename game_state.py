"""
game_state.py - Máy trạng thái (State Machine) và xử lý Luật chơi.

Chu trình trạng thái:
IDLE -> COUNTDOWN (3-2-1) -> PLAYING -> RESULT (10 giây rồi tự về IDLE).

Luật chơi chính xác:
1. Chọn ngẫu nhiên 5 tư thế không lặp từ kho mẫu, ưu tiên trộn theo độ khó
   (2 easy, 2 medium, 1 hard) để mục tiêu thắng 3/5 vừa sức và kịch tính.
2. Đồng hồ tổng 45 giây đếm ngược; mỗi tư thế có giới hạn mềm 15 giây (hết 15s tự chuyển).
3. Bộ đếm giữ 2.0 giây liên tục ở ngưỡng >= 80%:
   - Áp dụng Hysteresis (khi đang giữ chỉ cần >= 78% vẫn tiếp tục tính).
   - Áp dụng Dung sai ngắn (Grace period 0.25 giây): tụt nhẹ dưới 78% chưa quá 0.25s
     thì chỉ tạm dừng bộ đếm chứ không xóa tiến trình về 0.
4. Kết thúc sớm:
   - THẮNG ngay khi đạt đủ 3 tư thế PASS.
   - THUA ngay khi hết 45 giây hoặc số tư thế còn lại không đủ để chạm mốc 3.
5. Khi PASS: kích hoạt flash trắng, phát âm thanh chụp, lưu ảnh webcam sạch (không dính UI)
   vào `captures/<session_id>/<thứ_tự>_<id_tư_thế>.jpg`.
6. Khi vào RESULT: tự động tạo ảnh ghép (Collage) đặt ảnh người chơi cạnh ảnh mẫu.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

import config
from similarity import Landmark2D, PoseLandmarks, SimilarityResult

logger = logging.getLogger("pose_booth.game_state")


class GamePhase(str, Enum):
    """Bốn trạng thái chính của máy trạng thái gian hàng."""
    IDLE = "IDLE"
    COUNTDOWN = "COUNTDOWN"
    PLAYING = "PLAYING"
    RESULT = "RESULT"


@dataclass
class PoseTemplate:
    """Thông tin một tư thế mẫu đã nạp vào bộ nhớ."""
    pose_id: str
    name: str
    difficulty: str
    image_path: Path
    width: int
    height: int
    landmarks: PoseLandmarks
    image_bgr: np.ndarray


@dataclass
class PassedCaptureRecord:
    """Bản ghi một khoảnh khắc người chơi vượt qua tư thế thành công."""
    pose_index: int              # Thứ tự tư thế trong lượt (1..5)
    template: PoseTemplate
    clean_frame_bgr: np.ndarray  # Ảnh sạch từ webcam (không vẽ UI)
    saved_path: Path
    achieved_score: float


def load_pose_templates(json_path: Path = config.TEMPLATES_JSON_PATH) -> List[PoseTemplate]:
    """Đọc danh sách tư thế mẫu từ `templates.json` và nạp sẵn ảnh BGR."""
    if not json_path.exists():
        from build_templates import build_templates_from_directory
        build_templates_from_directory(output_json=json_path, force_generate_samples=True)

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    templates: List[PoseTemplate] = []
    for item in payload.get("templates", []):
        rel_path = item.get("image_path", "")
        img_path = (config.BASE_DIR / rel_path).resolve()
        if not img_path.exists():
            img_path = Path(rel_path).resolve()

        bgr = cv2.imread(str(img_path)) if img_path.exists() else None
        w = int(item.get("width", config.PANEL_WIDTH))
        h = int(item.get("height", config.PANEL_HEIGHT))
        if bgr is None:
            bgr = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            bgr = cv2.resize(bgr, (config.PANEL_WIDTH, config.PANEL_HEIGHT))
            w, h = config.PANEL_WIDTH, config.PANEL_HEIGHT

        landmarks: PoseLandmarks = {}
        for k_str, v_dict in item.get("landmarks", {}).items():
            landmarks[int(k_str)] = Landmark2D(
                x=float(v_dict["x"]),
                y=float(v_dict["y"]),
                visibility=float(v_dict.get("visibility", 1.0)),
            )

        templates.append(
            PoseTemplate(
                pose_id=str(item.get("id", img_path.stem)),
                name=str(item.get("name", "Tư thế mẫu")),
                difficulty=str(item.get("difficulty", "medium")).lower(),
                image_path=img_path,
                width=w,
                height=h,
                landmarks=landmarks,
                image_bgr=bgr,
            )
        )

    return templates


def select_balanced_poses(
    all_templates: List[PoseTemplate],
    count: int = config.NUM_POSES_PER_GAME,
    difficulty_mix: Dict[str, int] = config.DIFFICULTY_MIX,
    rng: Optional[random.Random] = None,
) -> List[PoseTemplate]:
    """
    Chọn ngẫu nhiên `count` tư thế KHÔNG lặp, trộn theo độ khó (2 easy -> 2 medium -> 1 hard).
    Nếu kho mẫu không đủ số lượng ở một mức độ khó, tự động bù bằng các tư thế còn lại.
    """
    if not all_templates:
        return []

    r = rng or random
    by_diff: Dict[str, List[PoseTemplate]] = {"easy": [], "medium": [], "hard": []}
    for t in all_templates:
        bucket = t.difficulty if t.difficulty in by_diff else "medium"
        by_diff[bucket].append(t)

    selected: List[PoseTemplate] = []
    used_ids: set[str] = set()

    for diff_level in ("easy", "medium", "hard"):
        needed = difficulty_mix.get(diff_level, 0)
        candidates = [t for t in by_diff[diff_level] if t.pose_id not in used_ids]
        r.shuffle(candidates)
        picked = candidates[:needed]
        selected.extend(picked)
        for p in picked:
            used_ids.add(p.pose_id)

    # Bù thêm nếu tổng số chưa đủ `count`
    if len(selected) < count:
        remaining = [t for t in all_templates if t.pose_id not in used_ids]
        r.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])

    # Nếu kho mẫu có ít hơn `count` ảnh, cho phép lặp vòng tròn để đủ 5 lượt
    while len(selected) < count and len(all_templates) > 0:
        selected.append(r.choice(all_templates))

    return selected[:count]


class GameStateMachine:
    """Quản lý toàn bộ trạng thái trò chơi, bộ đếm thời gian và lưu trữ ảnh chụp."""

    def __init__(
        self,
        templates: List[PoseTemplate],
        captures_root: Path = config.CAPTURES_DIR,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.all_templates: List[PoseTemplate] = templates
        self.captures_root: Path = captures_root
        self.rng: random.Random = rng or random.Random()

        self.phase: GamePhase = GamePhase.IDLE
        self.phase_elapsed_sec: float = 0.0

        # Trạng thái tự động bắt đầu ở màn IDLE
        self.idle_presence_sec: float = 0.0

        # Dữ liệu lượt chơi hiện tại
        self.session_id: str = ""
        self.session_dir: Path = self.captures_root
        self.playlist: List[PoseTemplate] = []
        self.current_pose_idx: int = 0
        self.pose_outcomes: List[Optional[bool]] = []  # True=PASS, False=MISS, None=Chưa xong

        self.game_remaining_sec: float = config.GAME_DURATION_SEC
        self.pose_remaining_sec: float = config.POSE_SOFT_TIMEOUT_SEC

        # Bộ đếm giữ 2.0 giây kèm Hysteresis và Grace period
        self.hold_elapsed_sec: float = 0.0
        self.grace_elapsed_sec: float = 0.0
        self.is_holding_active: bool = False

        # Hiệu ứng & sự kiện âm thanh trả về cho tầng UI trong mỗi tick
        self.flash_timer_sec: float = 0.0
        self.last_event: Optional[str] = None  # "countdown_start" | "play_start" | "pose_pass" | "pose_miss" | "win" | "lose"

        # Kết quả cuối lượt
        self.is_winner: bool = False
        self.passed_records: List[PassedCaptureRecord] = []
        self.collage_bgr: Optional[np.ndarray] = None
        self.collage_path: Optional[Path] = None

    @property
    def passed_count(self) -> int:
        """Số tư thế đã PASS trong lượt hiện tại."""
        return sum(1 for x in self.pose_outcomes if x is True)

    @property
    def current_template(self) -> Optional[PoseTemplate]:
        """Tư thế mẫu hiện tại cần thực hiện."""
        if 0 <= self.current_pose_idx < len(self.playlist):
            return self.playlist[self.current_pose_idx]
        return None

    @property
    def hold_progress(self) -> float:
        """Tiến trình giữ tư thế từ 0.0 đến 1.0 (tương ứng 0.0s .. 2.0s)."""
        return max(0.0, min(1.0, self.hold_elapsed_sec / config.HOLD_DURATION_SEC))

    def reset_to_idle(self) -> None:
        """Đưa máy trạng thái về màn hình chờ IDLE."""
        self.phase = GamePhase.IDLE
        self.phase_elapsed_sec = 0.0
        self.idle_presence_sec = 0.0
        self.hold_elapsed_sec = 0.0
        self.grace_elapsed_sec = 0.0
        self.is_holding_active = False
        self.flash_timer_sec = 0.0
        self.last_event = None

    def start_countdown(self) -> None:
        """Khởi tạo lượt chơi mới và chuyển sang đếm ngược 3-2-1."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{timestamp}"
        self.session_dir = self.captures_root / self.session_id

        self.playlist = select_balanced_poses(
            self.all_templates,
            count=config.NUM_POSES_PER_GAME,
            difficulty_mix=config.DIFFICULTY_MIX,
            rng=self.rng,
        )
        self.current_pose_idx = 0
        self.pose_outcomes = [None] * len(self.playlist)

        self.game_remaining_sec = config.GAME_DURATION_SEC
        self.pose_remaining_sec = config.POSE_SOFT_TIMEOUT_SEC
        self.hold_elapsed_sec = 0.0
        self.grace_elapsed_sec = 0.0
        self.is_holding_active = False
        self.flash_timer_sec = 0.0
        self.is_winner = False
        self.passed_records.clear()
        self.collage_bgr = None
        self.collage_path = None

        self.phase = GamePhase.COUNTDOWN
        self.phase_elapsed_sec = 0.0
        self.last_event = "countdown_start"
        logger.info("Bắt đầu phiên chơi mới: %s (%d tư thế)", self.session_id, len(self.playlist))

    def skip_current_pose(self) -> None:
        """Phím tắt N / SPACE: bỏ qua tư thế hiện tại (tính là trượt) và sang tư thế kế tiếp."""
        if self.phase != GamePhase.PLAYING:
            return
        self._record_pose_miss_and_advance()

    def update(
        self,
        dt: float,
        similarity: SimilarityResult,
        clean_webcam_frame: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Cập nhật trạng thái game sau mỗi khung hình với bước thời gian `dt` (giây).
        Trả về tên sự kiện vừa phát sinh (nếu có) để tầng UI phát âm thanh.
        """
        dt = max(0.0, min(0.5, float(dt)))
        self.last_event = None
        self.phase_elapsed_sec += dt
        if self.flash_timer_sec > 0.0:
            self.flash_timer_sec = max(0.0, self.flash_timer_sec - dt)

        if self.phase == GamePhase.IDLE:
            self._update_idle(dt, similarity)
        elif self.phase == GamePhase.COUNTDOWN:
            self._update_countdown(dt)
        elif self.phase == GamePhase.PLAYING:
            self._update_playing(dt, similarity, clean_webcam_frame)
        elif self.phase == GamePhase.RESULT:
            self._update_result(dt)

        return self.last_event

    def _update_idle(self, dt: float, similarity: SimilarityResult) -> None:
        """Ở màn IDLE: nếu phát hiện người đứng đủ khớp trong ROI liên tục thì tự bắt đầu."""
        if not config.AUTO_START_ENABLED:
            return
        if not similarity.insufficient_landmarks and similarity.valid_limb_count >= config.MIN_VALID_LIMBS:
            self.idle_presence_sec += dt
            if self.idle_presence_sec >= config.AUTO_START_HOLD_SEC:
                self.start_countdown()
        else:
            self.idle_presence_sec = max(0.0, self.idle_presence_sec - dt * 2.0)

    def _update_countdown(self, dt: float) -> None:
        """Đếm ngược 3-2-1 trước khi bước vào PLAYING."""
        if self.phase_elapsed_sec >= config.COUNTDOWN_SEC:
            self.phase = GamePhase.PLAYING
            self.phase_elapsed_sec = 0.0
            self.last_event = "play_start"

    def _update_playing(
        self,
        dt: float,
        similarity: SimilarityResult,
        clean_webcam_frame: Optional[np.ndarray],
    ) -> None:
        """Cập nhật đồng hồ 45s, bộ đếm giữ 2.0s và kiểm tra điều kiện PASS / THẮNG / THUA."""
        self.game_remaining_sec = max(0.0, self.game_remaining_sec - dt)
        self.pose_remaining_sec = max(0.0, self.pose_remaining_sec - dt)

        score = similarity.score if not similarity.insufficient_landmarks else 0.0

        # Xử lý bộ đếm giữ có Hysteresis (0.80 vào / 0.78 duy trì) và Dung sai (0.25s)
        if not self.is_holding_active:
            if score >= config.PASS_THRESHOLD:
                self.is_holding_active = True
                self.hold_elapsed_sec = dt
                self.grace_elapsed_sec = 0.0
        else:
            if score >= config.HYSTERESIS_THRESHOLD:
                self.hold_elapsed_sec += dt
                self.grace_elapsed_sec = 0.0
            else:
                self.grace_elapsed_sec += dt
                if self.grace_elapsed_sec > config.HOLD_GRACE_PERIOD_SEC:
                    # Vượt quá thời gian dung sai -> reset bộ đếm giữ
                    self.is_holding_active = False
                    self.hold_elapsed_sec = 0.0
                    self.grace_elapsed_sec = 0.0

        # Kiểm tra nếu đã giữ đủ 2.0 giây -> PASS tư thế hiện tại
        if self.hold_elapsed_sec >= config.HOLD_DURATION_SEC:
            self._record_pose_pass_and_advance(score, clean_webcam_frame)
            return

        # Kiểm tra hết giờ tổng 45 giây
        if self.game_remaining_sec <= 0.0:
            self._finish_game(is_win=(self.passed_count >= config.REQUIRED_PASSES))
            return

        # Kiểm tra hết giờ mềm (15 giây) của tư thế hiện tại
        if self.pose_remaining_sec <= 0.0:
            self._record_pose_miss_and_advance()

    def _record_pose_pass_and_advance(
        self,
        achieved_score: float,
        clean_webcam_frame: Optional[np.ndarray],
    ) -> None:
        """Ghi nhận PASS, lưu ảnh sạch xuống đĩa và kiểm tra thắng ngay khi đủ 3/5."""
        tpl = self.current_template
        self.pose_outcomes[self.current_pose_idx] = True
        self.flash_timer_sec = config.FLASH_DURATION_SEC
        self.last_event = "pose_pass"

        if tpl is not None and clean_webcam_frame is not None:
            saved_path = self._save_clean_capture(
                pose_order=self.current_pose_idx + 1,
                template=tpl,
                clean_frame=clean_webcam_frame,
            )
            self.passed_records.append(
                PassedCaptureRecord(
                    pose_index=self.current_pose_idx + 1,
                    template=tpl,
                    clean_frame_bgr=clean_webcam_frame.copy(),
                    saved_path=saved_path,
                    achieved_score=achieved_score,
                )
            )

        # ĐIỀU KIỆN THẮNG NGAY: đạt đủ 3 tư thế PASS
        if self.passed_count >= config.REQUIRED_PASSES:
            self._finish_game(is_win=True)
            return

        self._advance_to_next_pose()

    def _record_pose_miss_and_advance(self) -> None:
        """Ghi nhận trượt tư thế hiện tại và kiểm tra xem còn đủ tư thế để đạt 3/5 không."""
        self.pose_outcomes[self.current_pose_idx] = False
        self.last_event = "pose_miss"

        remaining_after_this = len(self.playlist) - (self.current_pose_idx + 1)
        if self.passed_count + remaining_after_this < config.REQUIRED_PASSES:
            # Không còn đủ số tư thế để đạt 3 PASS -> kết thúc sớm báo THUA
            self._finish_game(is_win=False)
            return

        self._advance_to_next_pose()

    def _advance_to_next_pose(self) -> None:
        """Chuyển sang tư thế tiếp theo và đặt lại bộ đếm giữ."""
        self.current_pose_idx += 1
        self.pose_remaining_sec = min(config.POSE_SOFT_TIMEOUT_SEC, self.game_remaining_sec)
        self.hold_elapsed_sec = 0.0
        self.grace_elapsed_sec = 0.0
        self.is_holding_active = False

        if self.current_pose_idx >= len(self.playlist):
            self._finish_game(is_win=(self.passed_count >= config.REQUIRED_PASSES))

    def _finish_game(self, is_win: bool) -> None:
        """Chuyển sang màn hình RESULT và tạo ảnh ghép (collage) lưu niệm."""
        self.is_winner = is_win
        self.phase = GamePhase.RESULT
        self.phase_elapsed_sec = 0.0
        self.last_event = "win" if is_win else "lose"
        self.collage_bgr, self.collage_path = self._build_and_save_collage()
        logger.info(
            "Kết thúc phiên %s: %s (Đạt %d/%d tư thế)",
            self.session_id,
            "THẮNG" if is_win else "THUA",
            self.passed_count,
            config.REQUIRED_PASSES,
        )

    def _update_result(self, dt: float) -> None:
        """Giữ màn hình kết quả trong 10 giây rồi tự động quay về IDLE."""
        if self.phase_elapsed_sec >= config.RESULT_SCREEN_SEC:
            self.reset_to_idle()

    def _save_clean_capture(
        self,
        pose_order: int,
        template: PoseTemplate,
        clean_frame: np.ndarray,
    ) -> Path:
        """Lưu ảnh webcam sạch (không vẽ UI) vào `captures/<phiên>/<thứ_tự>_<tên>.jpg`."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", template.pose_id).strip("_")
        file_path = self.session_dir / f"{pose_order:02d}_{safe_slug}.jpg"
        try:
            cv2.imwrite(str(file_path), clean_frame)
        except Exception as exc:
            logger.error("Không thể lưu ảnh chụp %s: %s", file_path, exc)
        return file_path

    def _build_and_save_collage(self) -> Tuple[Optional[np.ndarray], Optional[Path]]:
        """
        Ghép các khoảnh khắc đã PASS cạnh ảnh tư thế mẫu tương ứng thành 1 bức ảnh kỷ niệm
        kích thước 1280x720 để hiển thị ở màn RESULT và lưu vào `captures/<phiên>/collage.jpg`.
        """
        if not self.passed_records:
            return None, None

        canvas_w, canvas_h = config.FRAME_WIDTH, config.FRAME_HEIGHT
        collage = np.full((canvas_h, canvas_w, 3), config.COLOR_BG_DARK, dtype=np.uint8)

        n = len(self.passed_records)
        margin_x = 30
        top_y = 130
        avail_h = canvas_h - top_y - 90
        slot_w = (canvas_w - margin_x * (n + 1)) // max(1, n)

        for idx, rec in enumerate(self.passed_records):
            x0 = margin_x + idx * (slot_w + margin_x)
            pair_w = slot_w
            half_w = pair_w // 2 - 4

            tpl_thumb = self._resize_crop_aspect(rec.template.image_bgr, half_w, avail_h)
            cap_thumb = self._resize_crop_aspect(rec.clean_frame_bgr, half_w, avail_h)

            collage[top_y : top_y + avail_h, x0 : x0 + half_w] = tpl_thumb
            collage[top_y : top_y + avail_h, x0 + half_w + 8 : x0 + half_w * 2 + 8] = cap_thumb

            # Viền neon quanh cặp ảnh mẫu - người chơi
            cv2.rectangle(
                collage,
                (x0 - 2, top_y - 2),
                (x0 + half_w * 2 + 10, top_y + avail_h + 2),
                config.COLOR_NEON_GREEN if self.is_winner else config.COLOR_NEON_YELLOW,
                2,
            )

        self.session_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.session_dir / "collage_result.jpg"
        try:
            cv2.imwrite(str(out_path), collage)
        except Exception as exc:
            logger.error("Không thể lưu ảnh collage %s: %s", out_path, exc)
        return collage, out_path

    @staticmethod
    def _resize_crop_aspect(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Co giãn và cắt trung tâm để không làm méo tỉ lệ hình ảnh người chơi."""
        if img is None or img.size == 0 or target_w <= 0 or target_h <= 0:
            return np.zeros((max(1, target_h), max(1, target_w), 3), dtype=np.uint8)
        h, w = img.shape[:2]
        scale = max(target_w / float(w), target_h / float(h))
        new_w = max(target_w, int(w * scale))
        new_h = max(target_h, int(h * scale))
        resized = cv2.resize(img, (new_w, new_h))
        x_start = (new_w - target_w) // 2
        y_start = (new_h - target_h) // 2
        return resized[y_start : y_start + target_h, x_start : x_start + target_w]