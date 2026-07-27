from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from twopoint_project.vision.capture import (
    CapturedFrame,
    LatestCapturedFrame,
    build_gstreamer_pipeline,
    resolve_camera_device,
)


def make_captured_frame(frame_id: int, generation: int = 1) -> CapturedFrame:
    return CapturedFrame(
        camera_id="test-camera",
        stream_generation=generation,
        frame_id=frame_id,
        captured_at_monotonic_ns=frame_id,
        pts_ns=frame_id,
        duration_ns=1,
        width=2,
        height=2,
        pixel_format="BGR",
        frame_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
        sample_interval_ns=1,
        copy_duration_ns=1,
    )


class LatestCapturedFrameTest(unittest.TestCase):
    def test_publish_overwrites_unread_frame(self) -> None:
        latest = LatestCapturedFrame()
        latest.publish(make_captured_frame(1))
        latest.publish(make_captured_frame(2))

        frame = latest.read_after(None, timeout=0)

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.frame_id, 2)
        self.assertEqual(latest.overwrite_count, 1)
        self.assertIsNone(latest.read_after(frame.identity, timeout=0))

    def test_stream_generation_is_part_of_identity(self) -> None:
        latest = LatestCapturedFrame()
        old = make_captured_frame(1, generation=1)
        new = make_captured_frame(1, generation=2)
        latest.publish(old)
        self.assertEqual(latest.read_after(None, timeout=0), old)
        latest.publish(new)
        self.assertEqual(latest.read_after(old.identity, timeout=0), new)


class GStreamerCaptureConfigTest(unittest.TestCase):
    def test_pipeline_uses_low_latency_drop_oldest_configuration(self) -> None:
        pipeline = build_gstreamer_pipeline("/dev/v4l/by-id/camera", 1280, 720, 30)

        self.assertIn("image/jpeg,width=1280,height=720,framerate=30/1", pipeline)
        self.assertIn("leaky=downstream", pipeline)
        self.assertIn("max-buffers=1", pipeline)
        self.assertIn("drop=true", pipeline)
        self.assertIn("sync=false", pipeline)
        self.assertIn("video/x-raw,format=BGR", pipeline)

    def test_auto_device_requires_exactly_one_stable_index_zero_path(self) -> None:
        with TemporaryDirectory() as directory:
            by_id = Path(directory)
            target = by_id / "video0"
            target.touch()
            stable = by_id / "usb-camera-video-index0"
            stable.symlink_to(target)

            self.assertEqual(resolve_camera_device(by_id_dir=by_id), str(stable))

            second = by_id / "usb-other-video-index0"
            second.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "multiple stable camera devices"):
                resolve_camera_device(by_id_dir=by_id)


if __name__ == "__main__":
    unittest.main()
