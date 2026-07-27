from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.center_fakes import FakeVideoRecorder
from twopoint_project.tools import monitor


class MonitorTest(unittest.TestCase):
    def test_monitor_writes_annotated_and_raw_video(self) -> None:
        FakeVideoRecorder.instances = []
        captured = SimpleNamespace(
            frame_id=1,
            timestamp=0.0,
            frame_bgr=np.zeros((8, 12, 3), dtype=np.uint8),
        )
        update = SimpleNamespace(
            valid=True,
            moved=False,
            settled=True,
            reason="settled",
            target=None,
            step=None,
        )

        with patch.object(monitor, "VideoRecorder", FakeVideoRecorder):
            adapter = monitor.CenterRunMonitor(
                enabled=True,
                output_path=Path("outputs/center_flash_track_20260714_120000.mp4"),
                fps=15.0,
                save_raw_video=True,
                webrtc_host="127.0.0.1",
                webrtc_port=8080,
                backend="traditional",
                providers=["fake"],
                task_name="center_flash_track",
            )
            adapter.on_frame(captured, [], update)

        _, status = adapter.frame_buffer.snapshot()
        self.assertEqual(status["task"], "center_flash_track")
        self.assertEqual([item.label for item in FakeVideoRecorder.instances], ["annotated", "raw"])
        self.assertEqual(FakeVideoRecorder.instances[1].output_path.name, "center_flash_track_20260714_120000_raw.mp4")
        self.assertEqual(len(FakeVideoRecorder.instances[0].frames), 1)
        self.assertEqual(len(FakeVideoRecorder.instances[1].frames), 1)

    def test_draws_raw_target_without_image_center_marker(self) -> None:
        frame = np.zeros((101, 201, 3), dtype=np.uint8)
        points = [
            {"label": "target_center", "x": 0.5, "y": 0.5, "confidence": 1.0},
            {"label": "laser_point", "x": 0.25, "y": 0.75, "confidence": 1.0},
        ]
        update = SimpleNamespace(step=None, reason="test")
        corners = ((0.1, 0.2), (0.9, 0.2), (0.9, 0.8), (0.1, 0.8))
        raw_target = {"label": "target_center", "x": 0.6, "y": 0.4, "confidence": 1.0}

        with patch.object(monitor.cv2, "circle") as circle, patch.object(monitor.cv2, "line") as line:
            monitor.draw_aim_frame(frame, points, update, corners, raw_target)

        calls = [item.args[1:] for item in circle.call_args_list]
        self.assertIn(((100, 50), 4, (0, 220, 0), -1, monitor.cv2.LINE_AA), calls)
        self.assertIn(((120, 40), 3, (255, 255, 255), -1, monitor.cv2.LINE_AA), calls)
        self.assertNotIn(((100, 50), 22, (255, 255, 255), 1, monitor.cv2.LINE_AA), calls)
        line.assert_not_called()


if __name__ == "__main__":
    unittest.main()
