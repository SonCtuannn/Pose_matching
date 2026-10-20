"""
ui.py - Giao diện đồ họa toàn màn hình, kết xuất chữ tiếng Việt có bộ đệm (LRU Cache) và Âm thanh.

Điểm nhấn kỹ thuật:
1. `VietnameseTextRenderer`: dùng `PIL.ImageFont.truetype` để vẽ tiếng Việt có dấu sắc nét,
   kết hợp bộ nhớ đệm (bitmap cache) lưu sẵn mảng RGBA của từng chuỗi văn bản theo
   `(text, font_size, color_bgr, stroke_width)`. Nhờ đó việc ghép chữ tiếng Việt lên
   khung hình OpenCV chỉ tốn phép trộn alpha numpy siêu tốc, giữ vững >= 30 FPS trên CPU.
2. `SoundManager`: hỗ trợ phát hiệu ứng âm thanh qua `pygame.mixer`. Nếu thư mục `assets/sounds/`
   chưa có file `.wav`, tự động tổng hợp sóng âm arcade vui nhộn bằng numpy; nếu máy không
   cài `pygame` hoặc không có loa, tự động chuyển sang chế độ im lặng an toàn.
3. Giao diện chia đôi 1280x720 rực rỡ:
   - Nửa trái (640x720): Ảnh tư thế mẫu + khung xương mẫu + huy hiệu độ khó.
   - Nửa phải (640x720): Webcam dạng gương + khung xương người chơi đổi màu theo điểm
     (báo đỏ đoạn chi nào đang lệch > 45 độ để người chơi sửa ngay) + khung hướng dẫn ROI.
   - Cụm HUD trung tâm: Đồng hồ 45s, bộ đếm "ĐÃ PASS x/3", thanh Similarity Score (%),
     vòng tròn tiến trình giữ 2.0 giây và bảng Debug (phím D).
"""

from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from game_state import GamePhase, GameStateMachine, PoseTemplate
from pose_engine import PoseExtractor
from similarity import PoseLandmarks, SimilarityResult, mirror_pose_landmarks

logger = logging.getLogger("pose_booth.ui")


class VietnameseTextRenderer:
    """
    Trình kết xuất chữ tiếng Việt tốc độ cao sử dụng Pillow + bộ nhớ đệm Bitmap RGBA.
    """

    def __init__(self, max_cache_entries: int = 512) -> None:
        self.max_cache_entries: int = max_cache_entries
        self._font_path: Optional[str] = self._locate_vietnamese_font()
        self._font_cache: Dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
        # Cache lưu (bgr_patch, alpha_float_3ch)
        self._bitmap_cache: Dict[
            Tuple[str, int, Tuple[int, int, int], int],
            Tuple[np.ndarray, np.ndarray],
        ] = {}

    @staticmethod
    def _locate_vietnamese_font() -> Optional[str]:
        """Tìm font chữ hỗ trợ tiếng Việt trong `assets/fonts/` hoặc hệ điều hành."""
        for candidate in config.FONT_CANDIDATES:
            if Path(candidate).exists():
                logger.info("Sử dụng font tiếng Việt: %s", candidate)
                return candidate
        logger.warning("Không tìm thấy font TrueType trong danh sách ưu tiên, dùng font mặc định.")
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if size not in self._font_cache:
            if self._font_path is not None:
                try:
                    self._font_cache[size] = ImageFont.truetype(self._font_path, size)
                except Exception:
                    self._font_cache[size] = ImageFont.load_default()
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _render_to_cache(
        self,
        text: str,
        size: int,
        color_bgr: Tuple[int, int, int],
        stroke_width: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = (text, size, color_bgr, stroke_width)
        if key in self._bitmap_cache:
            return self._bitmap_cache[key]

        if len(self._bitmap_cache) >= self.max_cache_entries:
            # Xóa bớt nửa đầu cache nếu đầy
            for old_key in list(self._bitmap_cache.keys())[: self.max_cache_entries // 2]:
                self._bitmap_cache.pop(old_key, None)

        font = self._get_font(size)
        dummy_img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        pad = stroke_width + 4
        w = max(1, (bbox[2] - bbox[0]) + pad * 2)
        h = max(1, (bbox[3] - bbox[1]) + pad * 2)

        pil_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)
        color_rgba = (color_bgr[2], color_bgr[1], color_bgr[0], 255)
        stroke_rgba = (10, 10, 14, 240) if stroke_width > 0 else None

        draw.text(
            (pad - bbox[0], pad - bbox[1]),
            text,
            font=font,
            fill=color_rgba,
            stroke_width=stroke_width,
            stroke_fill=stroke_rgba,
        )

        rgba = np.array(pil_img, dtype=np.uint8)
        bgr_patch = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha_3ch = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
        self._bitmap_cache[key] = (bgr_patch, alpha_3ch)
        return bgr_patch, alpha_3ch

    def draw_text(
        self,
        canvas: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        size: int = 24,
        color_bgr: Tuple[int, int, int] = config.COLOR_WHITE,
        stroke_width: int = 2,
        anchor: str = "topleft",
    ) -> Tuple[int, int]:
        """
        Vẽ chuỗi tiếng Việt lên `canvas` tại vị trí `pos`.
        `anchor` hỗ trợ: `'topleft'`, `'center'`, `'midtop'`, `'topright'`.
        Trả về kích thước `(w, h)` của vùng chữ đã vẽ.
        """
        if not text:
            return (0, 0)

        bgr_patch, alpha = self._render_to_cache(text, size, color_bgr, stroke_width)
        th, tw = bgr_patch.shape[:2]
        x, y = pos

        if anchor == "center":
            x -= tw // 2
            y -= th // 2
        elif anchor == "midtop":
            x -= tw // 2
        elif anchor == "topright":
            x -= tw

        ch, cw = canvas.shape[:2]
        # Cắt vùng giao nhau với khung hình để không bao giờ lỗi tràn mảng
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(cw, x + tw)
        y2 = min(ch, y + th)
        if x1 >= x2 or y1 >= y2:
            return (tw, th)

        px1 = x1 - x
        py1 = y1 - y
        px2 = px1 + (x2 - x1)
        py2 = py1 + (y2 - y1)

        roi = canvas[y1:y2, x1:x2]
        sub_patch = bgr_patch[py1:py2, px1:px2]
        sub_alpha = alpha[py1:py2, px1:px2]

        blended = roi.astype(np.float32) * (1.0 - sub_alpha) + sub_patch.astype(np.float32) * sub_alpha
        canvas[y1:y2, x1:x2] = blended.astype(np.uint8)
        return (tw, th)


class SoundManager:
    """
    Quản lý hiệu ứng âm thanh sự kiện bằng `pygame.mixer` (có fallback an toàn).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled: bool = enabled
        self._ready: bool = False
        self._sounds: Dict[str, object] = {}
        if self.enabled:
            self._init_audio()

    def _init_audio(self) -> None:
        try:
            import pygame  # type: ignore

            pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
            pygame.mixer.init()
            self._ready = True

            # Tạo hoặc nạp các hiệu ứng âm thanh
            tone_specs = {
                "countdown_start": [523.25, 659.25],
                "play_start": [523.25, 659.25, 783.99, 1046.50],
                "pose_pass": [783.99, 1046.50, 1318.51],
                "pose_miss": [311.13, 233.08],
                "win": [523.25, 659.25, 783.99, 1046.50, 1318.51],
                "lose": [392.00, 349.23, 311.13, 261.63],
            }
            for event_name, freqs in tone_specs.items():
                wav_file = config.SOUNDS_DIR / f"{event_name}.wav"
                if wav_file.exists():
                    self._sounds[event_name] = pygame.mixer.Sound(str(wav_file))
                else:
                    self._sounds[event_name] = self._synthesize_chime(pygame, freqs)
            logger.info("Đã khởi tạo âm thanh pygame.mixer thành công.")
        except Exception as exc:
            logger.info("Chạy ở chế độ không âm thanh (pygame.mixer không khả dụng: %s).", exc)
            self._ready = False

    @staticmethod
    def _synthesize_chime(pygame_mod: object, freqs: list[float]) -> object:
        """Tổng hợp sóng âm hình sin mượt mà trực tiếp trong RAM."""
        sample_rate = 22050
        segments = []
        for f in freqs:
            dur = 0.09
            t = np.linspace(0, dur, int(sample_rate * dur), endpoint=False)
            envelope = np.exp(-t * 14.0)
            wave = (np.sin(2 * np.pi * f * t) * envelope * 16000).astype(np.int16)
            segments.append(wave)
        full_wave = np.concatenate(segments)
        return pygame_mod.mixer.Sound(buffer=full_wave.tobytes())  # type: ignore

    def play(self, event_name: Optional[str]) -> None:
        if not self._ready or not event_name:
            return
        snd = self._sounds.get(event_name)
        if snd is not None:
            try:
                snd.play()  # type: ignore
            except Exception:
                pass


class BoothUI:
    """Lớp vẽ toàn bộ giao diện sự kiện 1280x720 cho 4 trạng thái game."""

    def __init__(self, sound_enabled: bool = True) -> None:
        self.text_renderer: VietnameseTextRenderer = VietnameseTextRenderer()
        self.sound: SoundManager = SoundManager(enabled=sound_enabled)
        self.debug_mode: bool = False

    @staticmethod
    def score_to_color(score: float) -> Tuple[int, int, int]:
        """Nội suy màu sắc BGR từ Đỏ (thấp) -> Vàng (trung bình) -> Xanh lá Neon (>=80%)."""
        if score >= config.PASS_THRESHOLD:
            return config.COLOR_NEON_GREEN
        if score >= 0.55:
            return config.COLOR_NEON_YELLOW
        return config.COLOR_NEON_RED

    def draw_skeleton(
        self,
        panel: np.ndarray,
        landmarks: Optional[PoseLandmarks],
        base_color: Tuple[int, int, int],
        outlier_limbs: Optional[list[str]] = None,
    ) -> None:
        """Vẽ khung xương lên một nửa màn hình (640x720)."""
        if not landmarks:
            return
        h, w = panel.shape[:2]
        outlier_set = set(outlier_limbs or [])

        for limb_name, idx_a, idx_b, _, _ in config.LIMB_DEFINITIONS:
            lm_a = landmarks.get(idx_a)
            lm_b = landmarks.get(idx_b)
            if lm_a is None or lm_b is None:
                continue
            if min(lm_a.visibility, lm_b.visibility) < config.MIN_LANDMARK_VISIBILITY:
                continue

            pt1 = (int(lm_a.x * w), int(lm_a.y * h))
            pt2 = (int(lm_b.x * w), int(lm_b.y * h))
            color = config.COLOR_NEON_RED if limb_name in outlier_set else base_color
            cv2.line(panel, pt1, pt2, (20, 20, 20), 10, cv2.LINE_AA)
            cv2.line(panel, pt1, pt2, color, 5, cv2.LINE_AA)

        for idx in config.KEY_LANDMARK_INDICES:
            lm = landmarks.get(idx)
            if lm is None or lm.visibility < config.MIN_LANDMARK_VISIBILITY:
                continue
            pt = (int(lm.x * w), int(lm.y * h))
            radius = 9 if idx == 0 else 7
            cv2.circle(panel, pt, radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(panel, pt, radius, config.COLOR_WHITE, -1, cv2.LINE_AA)
            cv2.circle(panel, pt, radius - 2, base_color, -1, cv2.LINE_AA)

    def render_frame(
        self,
        webcam_panel_bgr: np.ndarray,
        player_landmarks: Optional[PoseLandmarks],
        similarity: SimilarityResult,
        state: GameStateMachine,
        fps: float,
        backend_name: str,
    ) -> np.ndarray:
        """Kết xuất khung hình hoàn chỉnh 1280x720 tùy theo trạng thái của `state.phase`."""
        canvas = np.full(
            (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3),
            config.COLOR_BG_DARK,
            dtype=np.uint8,
        )

        # Lấy mẫu hiển thị cho nửa bên trái
        if state.phase in (GamePhase.PLAYING, GamePhase.COUNTDOWN) and state.current_template:
            active_tpl = state.current_template
        elif state.all_templates:
            # Ở màn IDLE xoay vòng các tư thế mẫu mỗi 3 giây để thu hút khách
            idx = int(state.phase_elapsed_sec // 3.0) % len(state.all_templates)
            active_tpl = state.all_templates[idx]
        else:
            active_tpl = None

        if state.phase == GamePhase.RESULT and state.collage_bgr is not None:
            canvas[:] = state.collage_bgr
            self._draw_result_overlay(canvas, state)
        else:
            left_panel = self._build_left_template_panel(active_tpl, similarity)
            right_panel = self._build_right_webcam_panel(
                webcam_panel_bgr, player_landmarks, similarity
            )
            canvas[:, : config.PANEL_WIDTH] = left_panel
            canvas[:, config.PANEL_WIDTH :] = right_panel

            # Đường phân cách dọc ở giữa
            cv2.line(
                canvas,
                (config.PANEL_WIDTH, 70),
                (config.PANEL_WIDTH, config.FRAME_HEIGHT),
                config.COLOR_NEON_CYAN,
                3,
            )

            self._draw_top_header(canvas, state)

            if state.phase == GamePhase.IDLE:
                self._draw_idle_overlay(canvas, state)
            elif state.phase == GamePhase.COUNTDOWN:
                self._draw_countdown_overlay(canvas, state)
            elif state.phase == GamePhase.PLAYING:
                self._draw_playing_hud(canvas, state, similarity)

        # Hiệu ứng chớp trắng máy ảnh khi vừa PASS tư thế
        if state.flash_timer_sec > 0.0:
            alpha = min(0.85, state.flash_timer_sec / config.FLASH_DURATION_SEC)
            white = np.full_like(canvas, 255)
            cv2.addWeighted(white, alpha, canvas, 1.0 - alpha, 0, canvas)

        if self.debug_mode:
            self._draw_debug_overlay(canvas, similarity, fps, backend_name)

        return canvas

    def _build_left_template_panel(
        self,
        template: Optional[PoseTemplate],
        similarity: SimilarityResult,
    ) -> np.ndarray:
        """Tạo nửa màn hình bên trái hiển thị ảnh mẫu + khung xương mẫu + nhãn độ khó."""
        if template is None:
            return np.zeros((config.PANEL_HEIGHT, config.PANEL_WIDTH, 3), dtype=np.uint8)

        panel = template.image_bgr.copy()
        tpl_landmarks = (
            mirror_pose_landmarks(template.landmarks)
            if similarity.used_mirrored_template
            else template.landmarks
        )
        self.draw_skeleton(panel, tpl_landmarks, config.COLOR_NEON_CYAN)

        # Huy hiệu tên tư thế và mức độ khó ở góc dưới trái
        cv2.rectangle(panel, (16, 620), (624, 704), (32, 22, 18), -1)
        cv2.rectangle(panel, (16, 620), (624, 704), config.COLOR_NEON_CYAN, 2)

        diff_map = {
            "easy": ("DỄ", config.COLOR_NEON_GREEN),
            "medium": ("VỪA", config.COLOR_NEON_YELLOW),
            "hard": ("KHÓ", config.COLOR_NEON_MAGENTA),
        }
        diff_label, diff_color = diff_map.get(
            template.difficulty, ("VỪA", config.COLOR_NEON_YELLOW)
        )

        self.text_renderer.draw_text(
            panel,
            f"TƯ THẾ MẪU: {template.name.upper()}",
            (32, 632),
            size=24,
            color_bgr=config.COLOR_WHITE,
        )
        self.text_renderer.draw_text(
            panel,
            f"Độ khó: {diff_label}",
            (32, 668),
            size=20,
            color_bgr=diff_color,
        )
        return panel

    def _build_right_webcam_panel(
        self,
        webcam_bgr: np.ndarray,
        player_landmarks: Optional[PoseLandmarks],
        similarity: SimilarityResult,
    ) -> np.ndarray:
        """Tạo nửa màn hình bên phải hiển thị webcam đã lật gương + khung ROI + khung xương."""
        panel = cv2.resize(webcam_bgr, (config.PANEL_WIDTH, config.PANEL_HEIGHT))

        # Vẽ khung ROI hướng dẫn vị trí đứng ở trung tâm
        if config.USE_CENTER_ROI:
            x1, y1, x2, y2 = PoseExtractor.get_roi_pixel_box(
                config.PANEL_WIDTH, config.PANEL_HEIGHT
            )
            roi_color = (
                config.COLOR_NEON_RED
                if similarity.insufficient_landmarks
                else config.COLOR_NEON_CYAN
            )
            # Vẽ 4 góc định vị thay vì khung kín để thoáng tầm nhìn
            corner_len = 36
            for cx, cy, dx, dy in [
                (x1, y1, 1, 1),
                (x2, y1, -1, 1),
                (x1, y2, 1, -1),
                (x2, y2, -1, -1),
            ]:
                cv2.line(panel, (cx, cy), (cx + dx * corner_len, cy), roi_color, 3)
                cv2.line(panel, (cx, cy), (cx, cy + dy * corner_len), roi_color, 3)

        skel_color = self.score_to_color(similarity.score)
        self.draw_skeleton(
            panel,
            player_landmarks,
            base_color=skel_color,
            outlier_limbs=similarity.outlier_limbs,
        )

        # Cảnh báo khi người chơi đứng quá gần hoặc khuất khỏi khung hình
        if similarity.insufficient_landmarks:
            cv2.rectangle(panel, (50, 620), (590, 695), (25, 25, 160), -1)
            cv2.rectangle(panel, (50, 620), (590, 695), config.COLOR_NEON_YELLOW, 2)
            self.text_renderer.draw_text(
                panel,
                "Lùi lại / bước vào khung hình (1.5m - 2m)",
                (config.PANEL_WIDTH // 2, 657),
                size=22,
                color_bgr=config.COLOR_NEON_YELLOW,
                anchor="center",
            )
        elif similarity.capped_by_outlier and similarity.outlier_limbs:
            bad_limb = similarity.outlier_limbs[0]
            cv2.rectangle(panel, (50, 625), (590, 690), (20, 45, 120), -1)
            self.text_renderer.draw_text(
                panel,
                f"Chỉnh lại: {bad_limb} đang lệch > 45°!",
                (config.PANEL_WIDTH // 2, 657),
                size=21,
                color_bgr=config.COLOR_NEON_YELLOW,
                anchor="center",
            )

        return panel

    def _draw_top_header(self, canvas: np.ndarray, state: GameStateMachine) -> None:
        """Vẽ thanh tiêu đề sự kiện trên cùng."""
        cv2.rectangle(canvas, (0, 0), (config.FRAME_WIDTH, 72), config.COLOR_PANEL_HEADER, -1)
        cv2.line(canvas, (0, 72), (config.FRAME_WIDTH, 72), config.COLOR_NEON_CYAN, 2)

        self.text_renderer.draw_text(
            canvas,
            config.BOOTH_TITLE,
            (24, 12),
            size=26,
            color_bgr=config.COLOR_NEON_CYAN,
        )
        self.text_renderer.draw_text(
            canvas,
            config.BOOTH_TAGLINE,
            (24, 44),
            size=16,
            color_bgr=config.COLOR_WHITE,
            stroke_width=1,
        )

        # Bộ đếm tiến độ PASS x/3 ở góc phải thanh Header
        pass_text = f"ĐÃ PASS: {state.passed_count}/{config.REQUIRED_PASSES} (Tư thế {min(state.current_pose_idx + 1, config.NUM_POSES_PER_GAME)}/{config.NUM_POSES_PER_GAME})"
        self.text_renderer.draw_text(
            canvas,
            pass_text,
            (config.FRAME_WIDTH - 24, 22),
            size=24,
            color_bgr=config.COLOR_NEON_GREEN,
            anchor="topright",
        )

    def _draw_idle_overlay(self, canvas: np.ndarray, state: GameStateMachine) -> None:
        """Vẽ bảng mời gọi người chơi ở trạng thái IDLE."""
        box_w, box_h = 820, 170
        x0 = (config.FRAME_WIDTH - box_w) // 2
        y0 = config.FRAME_HEIGHT // 2 - box_h // 2

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (18, 12, 30), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), config.COLOR_NEON_YELLOW, 3)

        self.text_renderer.draw_text(
            canvas,
            "SẴN SÀNG THỬ THÁCH SAO Y BẢN CHÍNH?",
            (config.FRAME_WIDTH // 2, y0 + 40),
            size=32,
            color_bgr=config.COLOR_NEON_YELLOW,
            anchor="center",
        )
        self.text_renderer.draw_text(
            canvas,
            "Nhấn phím [SPACE] hoặc đứng vào khung hình để bắt đầu ngay!",
            (config.FRAME_WIDTH // 2, y0 + 92),
            size=22,
            color_bgr=config.COLOR_WHITE,
            anchor="center",
        )

        if state.idle_presence_sec > 0.1:
            ratio = min(1.0, state.idle_presence_sec / config.AUTO_START_HOLD_SEC)
            bar_w = int((box_w - 80) * ratio)
            cv2.rectangle(
                canvas,
                (x0 + 40, y0 + 130),
                (x0 + 40 + bar_w, y0 + 148),
                config.COLOR_NEON_GREEN,
                -1,
            )

    def _draw_countdown_overlay(self, canvas: np.ndarray, state: GameStateMachine) -> None:
        """Vẽ đồng hồ đếm ngược 3 - 2 - 1 giữa màn hình."""
        remaining = max(0.0, config.COUNTDOWN_SEC - state.phase_elapsed_sec)
        count_num = max(1, int(math.ceil(remaining)))
        center = (config.FRAME_WIDTH // 2, config.FRAME_HEIGHT // 2)

        cv2.circle(canvas, center, 110, (25, 18, 35), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 110, config.COLOR_NEON_YELLOW, 6, cv2.LINE_AA)
        self.text_renderer.draw_text(
            canvas,
            str(count_num),
            center,
            size=110,
            color_bgr=config.COLOR_NEON_YELLOW,
            anchor="center",
        )
        self.text_renderer.draw_text(
            canvas,
            "CHUẨN BỊ TẠO DÁNG!",
            (center[0], center[1] + 150),
            size=30,
            color_bgr=config.COLOR_WHITE,
            anchor="center",
        )

    def _draw_playing_hud(
        self,
        canvas: np.ndarray,
        state: GameStateMachine,
        similarity: SimilarityResult,
    ) -> None:
        """Vẽ thanh Similarity Score, đồng hồ 45s và vòng tròn tiến trình giữ 2.0 giây."""
        cx = config.FRAME_WIDTH // 2

        # 1. Hộp đồng hồ tổng 45 giây ở giữa trên
        timer_color = (
            config.COLOR_NEON_RED
            if state.game_remaining_sec <= 10.0
            else config.COLOR_NEON_YELLOW
        )
        cv2.rectangle(canvas, (cx - 110, 82), (cx + 110, 148), (25, 20, 30), -1)
        cv2.rectangle(canvas, (cx - 110, 82), (cx + 110, 148), timer_color, 2)
        self.text_renderer.draw_text(
            canvas,
            f"{state.game_remaining_sec:04.1f}s",
            (cx, 115),
            size=34,
            color_bgr=timer_color,
            anchor="center",
        )

        # 2. Thanh Similarity Score (%)
        bar_w, bar_h = 460, 34
        bx = cx - bar_w // 2
        by = 162
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 48), -1)

        pct = int(round(similarity.score * 100))
        fill_w = int(bar_w * max(0.0, min(1.0, similarity.score)))
        bar_color = self.score_to_color(similarity.score)
        if fill_w > 0:
            cv2.rectangle(canvas, (bx, by), (bx + fill_w, by + bar_h), bar_color, -1)

        # Vạch mốc chuẩn 80% trên thanh điểm
        thresh_x = bx + int(bar_w * config.PASS_THRESHOLD)
        cv2.line(canvas, (thresh_x, by - 5), (thresh_x, by + bar_h + 5), config.COLOR_WHITE, 3)
        cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), config.COLOR_WHITE, 2)

        self.text_renderer.draw_text(
            canvas,
            f"ĐỘ KHỚP: {pct}% (Mục tiêu >= 80%)",
            (cx, by + bar_h // 2),
            size=20,
            color_bgr=config.COLOR_WHITE,
            anchor="center",
        )

        # 3. Vòng tròn tiến trình giữ 2.0 giây ở tâm màn hình
        ring_center = (cx, config.FRAME_HEIGHT // 2 + 20)
        ring_radius = 68
        cv2.circle(canvas, ring_center, ring_radius, (30, 25, 38), -1, cv2.LINE_AA)
        cv2.circle(canvas, ring_center, ring_radius, (90, 90, 100), 6, cv2.LINE_AA)

        if state.hold_progress > 0.0:
            end_angle = int(360.0 * state.hold_progress)
            cv2.ellipse(
                canvas,
                ring_center,
                (ring_radius, ring_radius),
                -90,
                0,
                end_angle,
                config.COLOR_NEON_GREEN,
                10,
                cv2.LINE_AA,
            )
            hold_rem = max(0.0, config.HOLD_DURATION_SEC - state.hold_elapsed_sec)
            self.text_renderer.draw_text(
                canvas,
                f"GIỮ {hold_rem:.1f}s",
                ring_center,
                size=24,
                color_bgr=config.COLOR_NEON_GREEN,
                anchor="center",
            )
        else:
            self.text_renderer.draw_text(
                canvas,
                f"{pct}%",
                ring_center,
                size=32,
                color_bgr=bar_color,
                anchor="center",
            )

    def _draw_result_overlay(self, canvas: np.ndarray, state: GameStateMachine) -> None:
        """Vẽ tiêu đề chiến thắng / chia buồn và đếm ngược về IDLE trên màn hình RESULT."""
        cv2.rectangle(canvas, (0, 0), (config.FRAME_WIDTH, 115), (25, 18, 35), -1)
        title = (
            "CHÚC MỪNG! BẠN ĐÃ CHIẾN THẮNG!"
            if state.is_winner
            else "HẾT LƯỢT! HÃY THỬ LẠI NHÉ!"
        )
        color = config.COLOR_NEON_GREEN if state.is_winner else config.COLOR_NEON_YELLOW

        self.text_renderer.draw_text(
            canvas,
            title,
            (config.FRAME_WIDTH // 2, 42),
            size=40,
            color_bgr=color,
            anchor="center",
        )
        self.text_renderer.draw_text(
            canvas,
            f"Hoàn thành xuất sắc {state.passed_count}/{config.REQUIRED_PASSES} tư thế yêu cầu",
            (config.FRAME_WIDTH // 2, 88),
            size=24,
            color_bgr=config.COLOR_WHITE,
            anchor="center",
        )

        rem = max(0, int(math.ceil(config.RESULT_SCREEN_SEC - state.phase_elapsed_sec)))
        self.text_renderer.draw_text(
            canvas,
            f"Tự động về màn hình chờ sau {rem}s | Nhấn [SPACE] hoặc [R] để chơi lượt mới",
            (config.FRAME_WIDTH // 2, config.FRAME_HEIGHT - 36),
            size=20,
            color_bgr=config.COLOR_NEON_CYAN,
            anchor="center",
        )

    def _draw_debug_overlay(
        self,
        canvas: np.ndarray,
        similarity: SimilarityResult,
        fps: float,
        backend_name: str,
    ) -> None:
        """Hiển thị thông số kỹ thuật chi tiết khi bật chế độ Debug (phím D)."""
        x0, y0 = config.PANEL_WIDTH + 16, 84
        box_h = 80 + 22 * len(similarity.limb_errors_deg)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + 340, min(700, y0 + box_h)), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        info_lines = [
            f"FPS: {fps:.1f} | Backend: {backend_name}",
            f"Sai số TB (E): {similarity.weighted_mean_error_deg:.1f}° | Mirror: {similarity.used_mirrored_template}",
            f"Raw Score: {similarity.raw_score*100:.1f}% | Capped: {similarity.capped_by_outlier}",
        ]
        curr_y = y0 + 8
        for line in info_lines:
            self.text_renderer.draw_text(
                canvas, line, (x0 + 10, curr_y), size=15, color_bgr=config.COLOR_NEON_CYAN, stroke_width=1
            )
            curr_y += 22

        for limb_name, err in similarity.limb_errors_deg.items():
            c = config.COLOR_NEON_RED if err > config.MAX_CORE_LIMB_ERROR_DEG else config.COLOR_WHITE
            self.text_renderer.draw_text(
                canvas,
                f" • {limb_name}: {err:.1f}°",
                (x0 + 10, curr_y),
                size=14,
                color_bgr=c,
                stroke_width=1,
            )
            curr_y += 20
            if curr_y > 680:
                break