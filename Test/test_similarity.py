"""
tests/test_similarity.py - Bộ kiểm thử tự động cho thuật toán Similarity và State Machine.

Các kịch bản kiểm thử bắt buộc:
1. Cùng một tư thế cho điểm gần 100% (~1.0).
2. Bất biến tuyệt đối khi dịch chuyển vị trí đứng (translation) và đổi tỉ lệ cơ thể (scale).
3. Hỗ trợ MIRROR_INVARIANT: tư thế lật gương ngang vẫn đạt điểm ~100%.
4. Tư thế khác hẳn cho điểm thấp (< 50%).
5. Chặn gian lận may rủi: chỉ cần 1 đoạn chi chính lệch > 45° thì điểm bị khóa ở mức <= 79%.
6. Xử lý đúng khi thiếu landmark hoặc visibility < 0.5 (trả về score = 0.0 và cờ insufficient_landmarks).
7. Kiểm thử bộ đếm giữ 2.0 giây kèm Hysteresis (0.78) và Dung sai ngắn (0.25s) trong GameStateMachine.
"""

from __future__ import annotations
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

# Thêm thư mục gốc của dự án vào sys.path để import trực tiếp
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from build_templates import BUILTIN_SAMPLE_POSES
from game_state import GamePhase, GameStateMachine, PoseTemplate
from similarity import (
    Landmark2D,
    PoseLandmarks,
    evaluate_pose_similarity,
    mirror_pose_landmarks,
)


def _sample_to_landmarks(sample_index: int = 0) -> PoseLandmarks:
    """Chuyển mẫu tư thế có sẵn thành từ điển PoseLandmarks."""
    raw = BUILTIN_SAMPLE_POSES[sample_index]["landmarks"]
    return {
        int(k): Landmark2D(x=float(v[0]), y=float(v[1]), visibility=float(v[2]))
        for k, v in raw.items()
    }


class TestPoseSimilarity(unittest.TestCase):
    """Kiểm thử toàn diện thuật toán chấm điểm `similarity.py`."""

    def test_identical_pose_scores_near_100_percent(self) -> None:
        """Cùng một tư thế phải cho điểm xấp xỉ 100% (sai số góc = 0 độ)."""
        pose = _sample_to_landmarks(0)
        res = evaluate_pose_similarity(pose, pose)
        self.assertFalse(res.insufficient_landmarks)
        self.assertAlmostEqual(res.score, 1.0, places=5)
        self.assertAlmostEqual(res.weighted_mean_error_deg, 0.0, places=4)

    def test_translation_and_scale_invariance(self) -> None:
        """Dịch chuyển vị trí đứng và phóng to/thu nhỏ cơ thể không làm thay đổi điểm số."""
        template = _sample_to_landmarks(0)
        # Giả lập người chơi đứng lệch sang góc và nhỏ hơn (scale = 0.55, dịch +0.18, +0.07)
        transformed: PoseLandmarks = {}
        for idx, lm in template.items():
            transformed[idx] = Landmark2D(
                x=lm.x * 0.55 + 0.18,
                y=lm.y * 0.55 + 0.07,
                visibility=lm.visibility,
            )

        res = evaluate_pose_similarity(
            player_landmarks=transformed,
            template_landmarks=template,
            player_size=(640, 720),
            template_size=(640, 720),
        )
        self.assertFalse(res.insufficient_landmarks)
        self.assertAlmostEqual(res.score, 1.0, places=4)

    def test_mirror_invariance(self) -> None:
        """Tư thế bất đối xứng (Ấm Trà Chảnh Chọe) khi lật gương vẫn đạt 100% nếu bật MIRROR_INVARIANT."""
        asymmetric_pose = _sample_to_landmarks(2)  # easy_teapot_sassy
        mirrored_player = mirror_pose_landmarks(asymmetric_pose)

        res_on = evaluate_pose_similarity(
            mirrored_player, asymmetric_pose, mirror_invariant=True
        )
        self.assertAlmostEqual(res_on.score, 1.0, places=4)
        self.assertTrue(res_on.used_mirrored_template)

        res_off = evaluate_pose_similarity(
            mirrored_player, asymmetric_pose, mirror_invariant=False
        )
        self.assertLess(res_off.score, 0.80)

    def test_completely_different_pose_scores_low(self) -> None:
        """Hai tư thế ngược hướng hoàn toàn (tay giơ cao chữ V vs tay xuôi khép sát) phải có điểm < 50%."""
        victory_pose = _sample_to_landmarks(0)
        # Tạo tư thế hai tay buông thõng ngược hướng chữ V và chân chéo ngược
        opposite_pose: PoseLandmarks = dict(victory_pose)
        opposite_pose[13] = Landmark2D(x=0.39, y=0.52, visibility=0.99)
        opposite_pose[15] = Landmark2D(x=0.39, y=0.70, visibility=0.99)
        opposite_pose[14] = Landmark2D(x=0.61, y=0.52, visibility=0.99)
        opposite_pose[16] = Landmark2D(x=0.61, y=0.70, visibility=0.99)
        opposite_pose[25] = Landmark2D(x=0.62, y=0.40, visibility=0.99)
        opposite_pose[27] = Landmark2D(x=0.72, y=0.25, visibility=0.99)

        res = evaluate_pose_similarity(opposite_pose, victory_pose)
        self.assertFalse(res.insufficient_landmarks)
        self.assertLess(res.score, 0.50)

    def test_outlier_limb_caps_score_at_79_percent(self) -> None:
        """
        Chặn gian lận may rủi: dù 13/14 đoạn chi khớp tuyệt đối (0 độ), chỉ cần 1 đoạn chi chính
        (ví dụ cẳng tay trái 13->15) lệch > 45 độ thì điểm tổng bị khóa ở tối đa 79% (0.79).
        """
        template = _sample_to_landmarks(0)
        cheating_player: PoseLandmarks = {
            idx: Landmark2D(lm.x, lm.y, lm.visibility)
            for idx, lm in template.items()
        }
        # Bẻ ngược cẳng tay trái (khớp 15) xuống dưới sao cho góc lệch > 90 độ
        cheating_player[15] = Landmark2D(x=0.28, y=0.55, visibility=0.99)

        res = evaluate_pose_similarity(
            cheating_player, template, mirror_invariant=False
        )
        self.assertGreater(res.raw_score, 0.80)
        self.assertTrue(res.capped_by_outlier)
        self.assertIn("Cẳng tay trái", res.outlier_limbs)
        self.assertAlmostEqual(res.score, config.CHEAT_CAP_SCORE, places=4)

    def test_missing_or_low_visibility_landmarks(self) -> None:
        """Khi thiếu khớp hoặc visibility < 0.5 trên đa số khớp, trả về score=0 và cờ cảnh báo."""
        template = _sample_to_landmarks(0)
        low_vis_player: PoseLandmarks = {
            idx: Landmark2D(lm.x, lm.y, visibility=0.20)
            for idx, lm in template.items()
        }

        res = evaluate_pose_similarity(low_vis_player, template)
        self.assertTrue(res.insufficient_landmarks)
        self.assertEqual(res.score, 0.0)


class TestGameStateMachine(unittest.TestCase):
    """Kiểm thử máy trạng thái, bộ đếm giữ 2 giây (kèm hysteresis + grace period) và luật thắng/thua."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.templates = []
        for item in BUILTIN_SAMPLE_POSES:
            lms = {
                int(k): Landmark2D(v[0], v[1], v[2])
                for k, v in item["landmarks"].items()
            }
            self.templates.append(
                PoseTemplate(
                    pose_id=item["id"],
                    name=item["name"],
                    difficulty=item["difficulty"],
                    image_path=Path(self.tmp_dir.name) / item["filename"],
                    width=640,
                    height=720,
                    landmarks=lms,
                    image_bgr=np.zeros((720, 640, 3), dtype=np.uint8),
                )
            )

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_hold_timer_with_hysteresis_and_grace_period_and_early_win(self) -> None:
        """Kiểm tra giữ 2.0s với hysteresis (0.79) + dung sai 0.2s và thắng ngay khi đạt 3/5."""
        sm = GameStateMachine(self.templates, captures_root=Path(self.tmp_dir.name))
        sm.start_countdown()
        self.assertEqual(sm.phase, GamePhase.COUNTDOWN)
        self.assertEqual(len(sm.playlist), config.NUM_POSES_PER_GAME)

        pose = _sample_to_landmarks(0)
        high_sim = evaluate_pose_similarity(pose, pose)  # score = 1.0

        # Qua 3 giây đếm ngược
        sm.update(dt=0.5, similarity=high_sim)
        for _ in range(6):
            sm.update(dt=0.5, similarity=high_sim)
        self.assertEqual(sm.phase, GamePhase.PLAYING)

        # Tạo kết quả giả lập ở mức 0.79 (trong vùng hysteresis >= 0.78) và 0.70 (tụt nhẹ ngắn hạn)
        hyst_sim = evaluate_pose_similarity(pose, pose)
        hyst_sim.score = 0.79
        dip_sim = evaluate_pose_similarity(pose, pose)
        dip_sim.score = 0.70

        clean_img = np.zeros((720, 640, 3), dtype=np.uint8)

        # Vượt qua Tư thế 1: giữ 1.0s ở 1.0 -> tụt xuống 0.70 trong 0.20s (nhỏ hơn grace 0.25s) -> giữ tiếp 1.0s ở 0.79
        sm.update(dt=0.5, similarity=high_sim, clean_webcam_frame=clean_img)
        sm.update(dt=0.5, similarity=high_sim, clean_webcam_frame=clean_img)
        sm.update(dt=0.20, similarity=dip_sim, clean_webcam_frame=clean_img)
        self.assertGreater(sm.hold_elapsed_sec, 0.9)  # Không bị reset nhờ grace period 0.25s
        sm.update(dt=0.5, similarity=hyst_sim, clean_webcam_frame=clean_img)
        sm.update(dt=0.5, similarity=hyst_sim, clean_webcam_frame=clean_img)
        self.assertEqual(sm.passed_count, 1)

        # Vượt qua Tư thế 2 và Tư thế 3 -> Thắng ngay lập tức!
        for _ in range(2):
            for _ in range(5):
                sm.update(dt=0.45, similarity=high_sim, clean_webcam_frame=clean_img)

        self.assertEqual(sm.passed_count, 3)
        self.assertEqual(sm.phase, GamePhase.RESULT)
        self.assertTrue(sm.is_winner)
        self.assertIsNotNone(sm.collage_path)
        self.assertTrue(sm.collage_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)