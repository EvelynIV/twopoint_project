from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from twopoint_project.vision.inferencer import build_vision_inferencer
from twopoint_project.vision3.infer_npu_pose import (
    LetterboxMeta,
    NativePoseDetection,
    NpuPoseInferencer,
    PoseDetection,
    PoseKeypoint,
    PoseTiming,
    letterbox_rgb_uint8,
    restore_detection,
    target_center_prediction,
)


class NpuPoseTest(unittest.TestCase):
    def test_letterbox_keeps_uint8_hwc_and_converts_bgr_to_rgb(self) -> None:
        frame = np.zeros((320, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 10
        frame[:, :, 1] = 20
        frame[:, :, 2] = 30

        tensor, meta = letterbox_rgb_uint8(frame)

        self.assertEqual(tensor.shape, (640, 640, 3))
        self.assertEqual(tensor.dtype, np.uint8)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(tuple(tensor[160, 0]), (30, 20, 10))
        self.assertEqual(tuple(tensor[0, 0]), (114, 114, 114))
        self.assertEqual(meta, LetterboxMeta(1.0, 0, 160, 640, 320))

    def test_restore_detection_removes_letterbox_padding(self) -> None:
        meta = LetterboxMeta(scale=0.5, left=0, top=140, original_width=1280, original_height=720)
        keypoints = tuple(PoseKeypoint(320.0, 320.0, 0.9) for _ in range(5))
        detection = PoseDetection((100.0, 190.0, 500.0, 590.0), 0.8, keypoints)

        restored = restore_detection(detection, meta)

        self.assertEqual(restored.box, (200.0, 100.0, 1000.0, 719.0))
        self.assertEqual(restored.keypoints[0], PoseKeypoint(640.0, 360.0, 0.9))

    def test_target_center_uses_semantic_center_without_point_threshold(self) -> None:
        meta = LetterboxMeta(scale=1.0, left=0, top=0, original_width=640, original_height=640)
        keypoints = (
            PoseKeypoint(100.0, 50.0, 0.01),
            PoseKeypoint(0.0, 0.0, 0.99),
            PoseKeypoint(639.0, 0.0, 0.99),
            PoseKeypoint(639.0, 639.0, 0.99),
            PoseKeypoint(0.0, 639.0, 0.99),
        )
        detection = PoseDetection((0.0, 0.0, 639.0, 639.0), 0.85, keypoints)

        prediction = target_center_prediction([detection], meta)

        assert prediction is not None
        self.assertEqual(prediction["label"], "target_center")
        self.assertAlmostEqual(prediction["x"], 100.0 / 639.0)
        self.assertAlmostEqual(prediction["y"], 50.0 / 639.0)
        self.assertAlmostEqual(prediction["confidence"], 0.01)

    def test_no_detection_returns_zero_confidence_target(self) -> None:
        meta = LetterboxMeta(scale=1.0, left=0, top=0, original_width=640, original_height=640)

        prediction = target_center_prediction([], meta)

        self.assertIsNone(prediction)

    def test_native_structure_matches_c_abi(self) -> None:
        self.assertEqual(NativePoseDetection.keypoints.size, 15 * 4)
        self.assertEqual(NativePoseDetection.score.offset, 4 * 4)
        self.assertEqual(NativePoseDetection.keypoints.offset, 5 * 4)

    def test_factory_builds_npu_pose_backend_without_loading_hardware(self) -> None:
        inferencer = build_vision_inferencer(
            backend="npu_pose",
            npu_model_path=Path("model-bin/pose/test.nb"),
            npu_library_path=Path("build/npu/test.so"),
            img_size=640,
            npu_box_confidence_threshold=0.25,
            npu_nms_threshold=0.5,
        )

        self.assertIsInstance(inferencer, NpuPoseInferencer)
        self.assertEqual(inferencer.box_confidence_threshold, 0.25)

    def test_details_bind_center_and_distance_to_most_likely_accepted_center(self) -> None:
        meta = LetterboxMeta(
            scale=1.0,
            left=0,
            top=0,
            original_width=1920,
            original_height=1080,
        )
        rectangle_width = 400.0
        rectangle_height = 37142.0 / rectangle_width

        def detection(center_x: float, box_score: float, center_confidence: float) -> PoseDetection:
            left = center_x - rectangle_width / 2.0
            right = center_x + rectangle_width / 2.0
            top = 540.0 - rectangle_height / 2.0
            bottom = 540.0 + rectangle_height / 2.0
            keypoints = (
                PoseKeypoint(center_x, 540.0, center_confidence),
                PoseKeypoint(left, top, 0.9),
                PoseKeypoint(right, top, 0.9),
                PoseKeypoint(right, bottom, 0.9),
                PoseKeypoint(left, bottom, 0.9),
            )
            return PoseDetection((left, top, right, bottom), box_score, keypoints)

        first = detection(480.0, 0.9, 0.7)
        second = detection(1440.0, 0.8, 0.95)
        rejected = detection(960.0, 0.3, 0.99)
        inferencer = NpuPoseInferencer(
            Path("unused.nb"),
            Path("unused.so"),
            box_confidence_threshold=0.4,
        )
        inferencer.last_timing = PoseTiming(1, 2, 3)
        inferencer.predict_detections = lambda frame: ([first, second, rejected], meta)  # type: ignore[method-assign]

        details = inferencer.predict_with_details(np.zeros((1080, 1920, 3), dtype=np.uint8))

        self.assertAlmostEqual(details.points[0]["x"], 1440.0 / 1919.0)
        self.assertAlmostEqual(details.points[0]["confidence"], 0.95)
        self.assertEqual(details.detections, (first, second))
        self.assertIsNotNone(details.target_area_normalized)
        self.assertIsNotNone(details.target_distance_cm)
        expected_corners = tuple(
            (keypoint.x / 1919.0, keypoint.y / 1079.0)
            for keypoint in second.keypoints[1:5]
        )
        for actual, expected in zip(details.target_corners_normalized, expected_corners):
            self.assertAlmostEqual(actual[0], expected[0])
            self.assertAlmostEqual(actual[1], expected[1])
        assert details.target_distance_cm is not None
        self.assertAlmostEqual(details.target_distance_cm, 150.0, places=3)
        self.assertIsNotNone(inferencer.last_timing)
        assert inferencer.last_timing is not None
        self.assertGreater(inferencer.last_timing.postprocess_ns, 3)

    def test_details_without_detection_have_no_distance(self) -> None:
        meta = LetterboxMeta(1.0, 0, 0, 1920, 1080)
        inferencer = NpuPoseInferencer(Path("unused.nb"), Path("unused.so"))
        inferencer.predict_detections = lambda frame: ([], meta)  # type: ignore[method-assign]

        details = inferencer.predict_with_details(np.zeros((1080, 1920, 3), dtype=np.uint8))

        self.assertEqual(
            details.points,
            [],
        )
        self.assertIsNone(details.target_area_normalized)
        self.assertIsNone(details.target_distance_cm)
        self.assertEqual(details.target_corners_normalized, ())


if __name__ == "__main__":
    unittest.main()
