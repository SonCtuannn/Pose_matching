"""
pose_engine.py - Bộ trích xuất khung xương MediaPipe Pose & Luồng đọc Camera bất đồng bộ.

Tính năng chính:
1. Ưu tiên `mp.solutions.pose` (model_complexity=1) chạy hoàn toàn offline trên CPU.
2. Tự động fallback sang MediaPipe Tasks API (`PoseLandmarker`) nếu phiên bản mediapipe
   mới đã gỡ bỏ `mp.solutions`, hoặc fallback sang chế độ giả lập khi chạy test/mô phỏng.
3. Cắt vùng ROI trung tâm (Player ROI) để chỉ bắt người chơi đứng ở vị trí đánh dấu,
   loại bỏ nhiễu từ khán giả đứng xem hai bên gian hàng, sau đó quy đổi tọa độ khớp
   về lại hệ quy chiếu khung hình chuẩn hóa [0..1].
4. `ThreadedCamera`: đọc webcam trên luồng riêng với bộ đệm 1 khung hình (1-frame buffer)
   giúp giảm độ trễ xuống tối thiểu và tự động kết nối lại nếu dây cáp camera bị lỏng.
"""

from __future__ import annotations
import logging
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

import config
from similarity import Landmark2D, LandmarkEMASmoother, PoseLandmarks

logger = logging.getLogger("pose_booth.pose_engine")


class PoseExtractor:
    """
    Lớp bao đóng MediaPipe Pose hỗ trợ:
    - Chế độ luồng video thời gian thực (`static_image_mode=False`, có làm mượt EMA).
    - Chế độ xử lý ảnh mẫu tĩnh (`static_image_mode=True`).
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = config.MP_MODEL_COMPLEXITY,
        min_detection_confidence: float = config.MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MP_MIN_TRACKING_CONFIDENCE,
        use_roi: bool = config.USE_CENTER_ROI,
    ) -> None:
        self.static_image_mode: bool = static_image_mode
        self.use_roi: bool = use_roi and (not static_image_mode)
        self.smoother: LandmarkEMASmoother = LandmarkEMASmoother(config.EMA_ALPHA_LANDMARKS)
        self.backend_name: str = "uninitialized"
        self._mp_pose = None
        self._tasks_landmarker = None

        self._init_mediapipe(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _init_mediapipe(
        self,
        static_image_mode: bool,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        """Khởi tạo backend MediaPipe (ưu tiên solutions.pose, fallback sang Tasks API)."""
        try:
            import mediapipe as mp  # type: ignore

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self._mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=static_image_mode,
                    model_complexity=model_complexity,
                    smooth_landmarks=not static_image_mode,
                    enable_segmentation=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.backend_name = "mp.solutions.pose"
                logger.info("Đã khởi tạo MediaPipe Solutions Pose (complexity=%d).", model_complexity)
                return

            # Fallback sang Tasks API nếu dùng mediapipe >= 0.10.18 đã bỏ mp.solutions
            if hasattr(mp, "tasks") and config.POSE_LANDMARKER_TASK_PATH.exists():
                from mediapipe.tasks import python as mp_python  # type: ignore
                from mediapipe.tasks.python import vision as mp_vision  # type: ignore

                running_mode = (
                    mp_vision.RunningMode.IMAGE
                    if static_image_mode
                    else mp_vision.RunningMode.VIDEO
                )
                base_options = mp_python.BaseOptions(
                    model_asset_path=str(config.POSE_LANDMARKER_TASK_PATH)
                )
                options = mp_vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=running_mode,
                    num_poses=1,
                    min_pose_detection_confidence=min_detection_confidence,
                    min_pose_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._tasks_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
                self.backend_name = "mp.tasks.PoseLandmarker"
                logger.info("Đã khởi tạo MediaPipe Tasks PoseLandmarker từ file .task.")
                return

            logger.warning(
                "MediaPipe đã cài đặt nhưng không tìm thấy solutions.pose hoặc file %s.",
                config.POSE_LANDMARKER_TASK_PATH,
            )
            self.backend_name = "fallback_mock"
        except ImportError:
            logger.warning(
                "Chưa cài đặt thư viện 'mediapipe'. Hệ thống chuyển sang chế độ giả lập (fallback_mock)."
            )
            self.backend_name = "fallback_mock"
        except Exception as exc:
            logger.error("Lỗi khi khởi tạo MediaPipe: %s", exc)
            self.backend_name = "fallback_mock"

    @staticmethod
    def get_roi_pixel_box(width: int, height: int) -> Tuple[int, int, int, int]:
        """Tính tọa độ hộp ROI (x1, y1, x2, y2) bằng pixel trên khung hình."""
        rx1, ry1, rx2, ry2 = config.PLAYER_ROI_NORM
        x1 = max(0, min(width - 2, int(rx1 * width)))
        y1 = max(0, min(height - 2, int(ry1 * height)))
        x2 = max(x1 + 1, min(width, int(rx2 * width)))
        y2 = max(y1 + 1, min(height, int(ry2 * height)))
        return (x1, y1, x2, y2)

    def reset_smoothing(self) -> None:
        """Xóa bộ nhớ đệm EMA khi đổi người chơi."""
        self.smoother.reset()

    def extract(self, bgr_frame: np.ndarray) -> Optional[PoseLandmarks]:
        """
        Trích xuất các khớp chính (`config.KEY_LANDMARK_INDICES`) từ khung hình BGR.
        Nếu bật `use_roi`, chỉ cắt vùng trung tâm đưa vào MediaPipe rồi ánh xạ ngược
        tọa độ chuẩn hóa về toàn khung hình `bgr_frame`.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return None

        h, w = bgr_frame.shape[:2]
        if self.use_roi:
            x1, y1, x2, y2 = self.get_roi_pixel_box(w, h)
            roi_bgr = bgr_frame[y1:y2, x1:x2]
            rx_offset = x1 / float(w)
            ry_offset = y1 / float(h)
            rx_scale = (x2 - x1) / float(w)
            ry_scale = (y2 - y1) / float(h)
        else:
            roi_bgr = bgr_frame
            rx_offset, ry_offset = 0.0, 0.0
            rx_scale, ry_scale = 1.0, 1.0

        raw_landmarks = self._run_inference(roi_bgr)
        if not raw_landmarks:
            self.smoother.reset()
            return None

        # Quy đổi từ tọa độ chuẩn hóa trong ROI về tọa độ chuẩn hóa trên toàn khung hình
        remapped: PoseLandmarks = {}
        for idx, lm in raw_landmarks.items():
            full_x = rx_offset + lm.x * rx_scale
            full_y = ry_offset + lm.y * ry_scale
            remapped[idx] = Landmark2D(
                x=float(np.clip(full_x, 0.0, 1.0)),
                y=float(np.clip(full_y, 0.0, 1.0)),
                visibility=float(np.clip(lm.visibility, 0.0, 1.0)),
            )

        if self.static_image_mode:
            return remapped

        return self.smoother.update(remapped)

    def _run_inference(self, roi_bgr: np.ndarray) -> Optional[PoseLandmarks]:
        """Thực thi suy luận MediaPipe trên ảnh BGR."""
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        if self._mp_pose is not None:
            rgb.flags.writeable = False
            results = self._mp_pose.process(rgb)
            if not results or not results.pose_landmarks:
                return None

            lm_list = results.pose_landmarks.landmark
            extracted: PoseLandmarks = {}
            for idx in config.KEY_LANDMARK_INDICES:
                if idx < len(lm_list):
                    pt = lm_list[idx]
                    extracted[idx] = Landmark2D(
                        x=float(pt.x),
                        y=float(pt.y),
                        visibility=float(getattr(pt, "visibility", 1.0)),
                    )
            return extracted

        if self._tasks_landmarker is not None:
            import mediapipe as mp  # type: ignore

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if self.static_image_mode:
                res = self._tasks_landmarker.detect(mp_image)
            else:
                ts_ms = int(time.monotonic() * 1000)
                res = self._tasks_landmarker.detect_for_video(mp_image, ts_ms)

            if not res.pose_landmarks or len(res.pose_landmarks) == 0:
                return None

            lm_list = res.pose_landmarks[0]
            extracted = {}
            for idx in config.KEY_LANDMARK_INDICES:
                if idx < len(lm_list):
                    pt = lm_list[idx]
                    extracted[idx] = Landmark2D(
                        x=float(pt.x),
                        y=float(pt.y),
                        visibility=float(getattr(pt, "visibility", 1.0)),
                    )
            return extracted

        return None

    def close(self) -> None:
        """Giải phóng tài nguyên MediaPipe."""
        if self._mp_pose is not None:
            try:
                self._mp_pose.close()
            except Exception:
                pass
            self._mp_pose = None
        if self._tasks_landmarker is not None:
            try:
                self._tasks_landmarker.close()
            except Exception:
                pass
            self._tasks_landmarker = None


class ThreadedCamera:
    """
    Luồng đọc Camera độc lập với bộ đệm 1 khung hình (1-frame buffer).
    - Đảm bảo vòng lặp chính luôn lấy được khung hình mới nhất mà không bị chặn I/O.
    - Tự động thử kết nối lại mỗi `CAMERA_RECONNECT_INTERVAL_SEC` nếu webcam bị ngắt.
    - Hỗ trợ truyền đường dẫn file video để chạy chế độ mô phỏng.
    """

    def __init__(
        self,
        source: int | str = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        mirror: bool = config.MIRROR_WEBCAM,
    ) -> None:
        self.source: int | str = source
        self.width: int = width
        self.height: int = height
        self.mirror: bool = mirror

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock: threading.Lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._is_connected: bool = False
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Khởi chạy luồng đọc camera nền."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _open_capture(self) -> bool:
        """Mở kết nối tới thiết bị webcam hoặc file video."""
        try:
            if self._cap is not None:
                self._cap.release()

            if isinstance(self.source, int):
                self._cap = cv2.VideoCapture(self.source)
            else:
                self._cap = cv2.VideoCapture(str(self.source))

            if not self._cap or not self._cap.isOpened():
                self._is_connected = False
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._is_connected = True
            logger.info("Đã kết nối nguồn camera/video: %s", self.source)
            return True
        except Exception as exc:
            logger.error("Không thể mở camera %s: %s", self.source, exc)
            self._is_connected = False
            return False

    def _reader_loop(self) -> None:
        """Vòng lặp chạy nền liên tục cập nhật khung hình mới nhất."""
        last_reconnect_attempt = 0.0
        while self._running:
            if not self._is_connected or self._cap is None:
                now = time.monotonic()
                if now - last_reconnect_attempt >= config.CAMERA_RECONNECT_INTERVAL_SEC:
                    last_reconnect_attempt = now
                    self._open_capture()
                time.sleep(0.05)
                continue

            try:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    # Nếu nguồn là file video mô phỏng, tua lại từ đầu khi hết video
                    if isinstance(self.source, str) and Path(self.source).exists():
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.warning("Mất tín hiệu từ camera. Đang chuẩn bị kết nối lại...")
                    self._is_connected = False
                    continue

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                with self._lock:
                    self._latest_frame = frame
            except Exception as exc:
                logger.error("Lỗi khi đọc khung hình camera: %s", exc)
                self._is_connected = False
                time.sleep(0.1)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lấy bản sao của khung hình mới nhất từ bộ đệm."""
        with self._lock:
            if not self._is_connected or self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    @property
    def is_connected(self) -> bool:
        """Trạng thái kết nối hiện tại của camera."""
        return self._is_connected

    def stop(self) -> None:
        """Dừng luồng đọc và giải phóng thiết bị camera."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_connected = False