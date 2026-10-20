"""
main.py - Điểm vào chính của ứng dụng TRẠM 2: POSE MATCHING – SAO Y BẢN CHÍNH.

Tính năng vận hành bền bỉ tại sự kiện:
1. Vòng lặp chính bọc `try...except` toàn diện để trò chơi không bao giờ bị văng (crash)
   dù gặp lỗi khung hình hay ngắt kết nối thiết bị ngoại vi.
2. Hỗ trợ phím tắt đầy đủ:
   - `SPACE`: Bắt đầu lượt chơi (khi đang ở IDLE/RESULT) hoặc bỏ qua tư thế (khi đang PLAYING).
   - `N`: Bỏ qua tư thế hiện tại.
   - `R`: Reset ngay lập tức về màn hình chờ IDLE.
   - `D`: Bật/tắt bảng thông số kỹ thuật Debug (FPS, góc lệch từng đoạn chi).
   - `F`: Bật/tắt chế độ Toàn màn hình (Fullscreen).
   - `Q` hoặc `ESC`: Thoát ứng dụng an toàn.
3. Chế độ mô phỏng (`--simulate` và `--headless-frames`): cho phép kiểm thử toàn bộ
   luồng game bằng chính ảnh tư thế mẫu hoặc file video mà không cần người đứng trước webcam.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import random
import time
from typing import Optional
import cv2
import numpy as np

import config
from build_templates import build_templates_from_directory, render_sample_pose_card
from game_state import GamePhase, GameStateMachine, PoseTemplate, load_pose_templates
from pose_engine import PoseExtractor, ThreadedCamera
from similarity import (
    Landmark2D,
    PoseLandmarks,
    ScoreEMASmoother,
    evaluate_pose_similarity,
)
from ui import BoothUI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pose_booth.main")


def generate_simulated_player_pose(
    target_template: Optional[PoseTemplate],
    noise_deg_std: float = 0.012,
) -> Tuple[np.ndarray, Optional[PoseLandmarks]]:
    """
    Sinh khung hình giả lập và bộ khớp xương của người chơi dựa trên tư thế mẫu hiện tại
    (có nhiễu nhẹ tự nhiên) để chạy chế độ `--simulate` không cần webcam.
    """
    if target_template is None:
        blank = np.zeros((config.PANEL_HEIGHT, config.PANEL_WIDTH, 3), dtype=np.uint8)
        return blank, None

    sim_landmarks: PoseLandmarks = {}
    raw_tuple_map = {}
    for idx, lm in target_template.landmarks.items():
        jx = float(np.clip(lm.x + random.gauss(0.0, noise_deg_std), 0.05, 0.95))
        jy = float(np.clip(lm.y + random.gauss(0.0, noise_deg_std), 0.05, 0.95))
        sim_landmarks[idx] = Landmark2D(x=jx, y=jy, visibility=0.98)
        raw_tuple_map[idx] = (jx, jy, 0.98)

    sim_frame = render_sample_pose_card(
        landmarks_dict=raw_tuple_map,
        theme_color=(90, 210, 90),
        width=config.PANEL_WIDTH,
        height=config.PANEL_HEIGHT,
    )
    return sim_frame, sim_landmarks


def run_application(
    camera_source: int | str = config.CAMERA_INDEX,
    simulate: bool = False,
    headless_frames: int = 0,
    no_sound: bool = False,
) -> int:
    """Khởi chạy vòng lặp chính của gian hàng Pose Matching."""
    # 1. Đảm bảo đã có file `templates.json` và bộ ảnh mẫu trong `assets/poses/`
    if not config.TEMPLATES_JSON_PATH.exists():
        logger.info("Chưa có templates.json -> Đang tự động tạo bộ tư thế mẫu...")
        build_templates_from_directory()

    templates = load_pose_templates(config.TEMPLATES_JSON_PATH)
    if not templates:
        logger.error("Không nạp được bất kỳ tư thế mẫu nào! Vui lòng kiểm tra assets/poses/.")
        return 1

    state = GameStateMachine(templates=templates)
    ui = BoothUI(sound_enabled=(not no_sound and headless_frames == 0))
    score_smoother = ScoreEMASmoother(alpha=config.EMA_ALPHA_SCORE)
    pose_extractor = PoseExtractor(static_image_mode=False, use_roi=config.USE_CENTER_ROI)

    camera: Optional[ThreadedCamera] = None
    if not simulate:
        camera = ThreadedCamera(
            source=camera_source,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            mirror=config.MIRROR_WEBCAM,
        )
        camera.start()

    window_name = config.BOOTH_TITLE
    is_fullscreen = False
    if headless_frames == 0:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    prev_time = time.monotonic()
    fps_ema = float(config.TARGET_FPS)
    frame_counter = 0
    last_pose_idx = -1

    logger.info(
        "Ứng dụng đã sẵn sàng! Chế độ: %s | Tổng số mẫu: %d",
        "MÔ PHỎNG (--simulate)" if simulate else f"CAMERA ({camera_source})",
        len(templates),
    )

    try:
        while True:
            now = time.monotonic()
            dt = max(1e-3, min(0.25, now - prev_time))
            prev_time = now
            inst_fps = 1.0 / dt
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps

            try:
                # 1. Thu nhận khung hình và trích xuất landmark người chơi
                if simulate:
                    active_tpl = state.current_template or templates[0]
                    clean_webcam_panel, player_landmarks = generate_simulated_player_pose(
                        active_tpl
                    )
                else:
                    ok, raw_frame = camera.read() if camera else (False, None)
                    if ok and raw_frame is not None:
                        # Cắt vùng trung tâm tương đương khung nửa màn hình 640x720
                        fh, fw = raw_frame.shape[:2]
                        target_w = int(fh * (config.PANEL_WIDTH / float(config.PANEL_HEIGHT)))
                        if target_w <= fw:
                            x0 = (fw - target_w) // 2
                            cropped = raw_frame[:, x0 : x0 + target_w]
                        else:
                            cropped = raw_frame
                        clean_webcam_panel = cv2.resize(
                            cropped, (config.PANEL_WIDTH, config.PANEL_HEIGHT)
                        )
                        player_landmarks = pose_extractor.extract(clean_webcam_panel)
                    else:
                        clean_webcam_panel = np.zeros(
                            (config.PANEL_HEIGHT, config.PANEL_WIDTH, 3), dtype=np.uint8
                        )
                        player_landmarks = None

                # Reset bộ lọc điểm nếu vừa chuyển sang tư thế mới
                if state.current_pose_idx != last_pose_idx:
                    score_smoother.reset()
                    last_pose_idx = state.current_pose_idx

                # 2. Chấm điểm Similarity với tư thế mục tiêu
                target_tpl = state.current_template or (templates[0] if templates else None)
                sim_result = evaluate_pose_similarity(
                    player_landmarks=player_landmarks,
                    template_landmarks=target_tpl.landmarks if target_tpl else None,
                    player_size=(config.PANEL_WIDTH, config.PANEL_HEIGHT),
                    template_size=(
                        (target_tpl.width, target_tpl.height)
                        if target_tpl
                        else (config.PANEL_WIDTH, config.PANEL_HEIGHT)
                    ),
                )

                if sim_result.insufficient_landmarks:
                    score_smoother.reset()
                else:
                    sim_result.score = score_smoother.update(sim_result.score)

                # 3. Cập nhật máy trạng thái Game và phát âm thanh sự kiện
                event = state.update(
                    dt=dt,
                    similarity=sim_result,
                    clean_webcam_frame=clean_webcam_panel,
                )
                ui.sound.play(event)

                # 4. Kết xuất giao diện 1280x720
                composed_frame = ui.render_frame(
                    webcam_panel_bgr=clean_webcam_panel,
                    player_landmarks=player_landmarks,
                    similarity=sim_result,
                    state=state,
                    fps=fps_ema,
                    backend_name="simulation" if simulate else pose_extractor.backend_name,
                )

                if headless_frames == 0:
                    cv2.imshow(window_name, composed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if not _handle_keyboard_shortcut(
                        key=key,
                        state=state,
                        ui=ui,
                        window_name=window_name,
                        is_fullscreen=is_fullscreen,
                    ):
                        break
                    if key in (ord("f"), ord("F")):
                        is_fullscreen = not is_fullscreen
                else:
                    frame_counter += 1
                    if frame_counter >= headless_frames:
                        logger.info(
                            "Đã hoàn tất chạy kiểm thử %d khung hình mô phỏng (Phase=%s, Passed=%d).",
                            headless_frames,
                            state.phase.value,
                            state.passed_count,
                        )
                        break

            except Exception as loop_exc:
                # Bắt mọi ngoại lệ trong thân vòng lặp để gian hàng không bao giờ sập
                logger.exception("Đã bắt ngoại lệ an toàn trong vòng lặp chính: %s", loop_exc)
                time.sleep(0.02)

    finally:
        if camera is not None:
            camera.stop()
        pose_extractor.close()
        if headless_frames == 0:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    return 0


def _handle_keyboard_shortcut(
    key: int,
    state: GameStateMachine,
    ui: BoothUI,
    window_name: str,
    is_fullscreen: bool,
) -> bool:
    """
    Xử lý các phím tắt điều khiển tại gian hàng.
    Trả về False nếu người vận hành nhấn Q hoặc ESC để thoát.
    """
    if key == 255:
        return True

    if key in (config.KEY_ESC, ord("q"), ord("Q")):
        logger.info("Nhận lệnh thoát từ bàn phím (Q/ESC).")
        return False

    if key == config.KEY_SPACE:
        if state.phase in (GamePhase.IDLE, GamePhase.RESULT):
            state.start_countdown()
            ui.sound.play("countdown_start")
        elif state.phase == GamePhase.PLAYING:
            state.skip_current_pose()
            ui.sound.play("pose_miss")

    elif key in (ord("n"), ord("N")):
        if state.phase == GamePhase.PLAYING:
            state.skip_current_pose()
            ui.sound.play("pose_miss")

    elif key in (ord("r"), ord("R")):
        state.reset_to_idle()

    elif key in (ord("d"), ord("D")):
        ui.debug_mode = not ui.debug_mode
        logger.info("Chế độ Debug: %s", "BẬT" if ui.debug_mode else "TẮT")

    elif key in (ord("f"), ord("F")):
        new_prop = cv2.WINDOW_NORMAL if is_fullscreen else cv2.WINDOW_FULLSCREEN
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, new_prop)

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ứng dụng TRẠM 2: POSE MATCHING – SAO Y BẢN CHÍNH"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=str(config.CAMERA_INDEX),
        help="Chỉ số webcam (ví dụ: 0, 1) hoặc đường dẫn tới file video (.mp4) để test",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Chạy chế độ mô phỏng tự động bằng tư thế mẫu mà không cần webcam",
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=0,
        help="Chạy N khung hình không mở cửa sổ GUI (dùng cho kiểm thử tự động CI)",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Tắt hiệu ứng âm thanh",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cam_src: int | str = int(args.camera) if args.camera.isdigit() else args.camera
    raise SystemExit(
        run_application(
            camera_source=cam_src,
            simulate=args.simulate,
            headless_frames=args.headless_frames,
            no_sound=args.no_sound,
        )
    )