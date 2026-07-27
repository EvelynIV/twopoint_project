from __future__ import annotations

import math
import unittest

from twopoint_project.vision3.area_distance import (
    REFERENCE_AREA_NORMALIZED,
    estimate_area_distance,
)


def rectangle_for_normalized_area(
    area_normalized: float,
    *,
    frame_width: int,
    frame_height: int,
    normalized_width: float = 0.4,
) -> list[tuple[float, float]]:
    normalized_height = area_normalized / normalized_width
    left = 0.5 - normalized_width / 2.0
    right = 0.5 + normalized_width / 2.0
    top = 0.5 - normalized_height / 2.0
    bottom = 0.5 + normalized_height / 2.0
    return [
        (left * frame_width, top * frame_height),
        (right * frame_width, top * frame_height),
        (right * frame_width, bottom * frame_height),
        (left * frame_width, bottom * frame_height),
    ]


class AreaDistanceTest(unittest.TestCase):
    def test_reference_area_returns_150_cm(self) -> None:
        estimate = estimate_area_distance(
            rectangle_for_normalized_area(
                REFERENCE_AREA_NORMALIZED,
                frame_width=1920,
                frame_height=1080,
            ),
            frame_width=1920,
            frame_height=1080,
        )

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.area_normalized, REFERENCE_AREA_NORMALIZED, places=7)
        self.assertAlmostEqual(estimate.distance_cm, 150.0, places=4)

    def test_nine_times_reference_area_returns_50_cm(self) -> None:
        estimate = estimate_area_distance(
            rectangle_for_normalized_area(
                REFERENCE_AREA_NORMALIZED * 9.0,
                frame_width=1920,
                frame_height=1080,
            ),
            frame_width=1920,
            frame_height=1080,
        )

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.distance_cm, 50.0, places=4)

    def test_resolution_and_corner_order_do_not_change_result(self) -> None:
        corners_1920 = rectangle_for_normalized_area(
            REFERENCE_AREA_NORMALIZED,
            frame_width=1920,
            frame_height=1080,
        )
        corners_1280 = rectangle_for_normalized_area(
            REFERENCE_AREA_NORMALIZED,
            frame_width=1280,
            frame_height=720,
        )
        shuffled_1280 = [corners_1280[index] for index in (2, 0, 3, 1)]

        estimate_1920 = estimate_area_distance(
            corners_1920,
            frame_width=1920,
            frame_height=1080,
        )
        estimate_1280 = estimate_area_distance(
            shuffled_1280,
            frame_width=1280,
            frame_height=720,
        )

        self.assertIsNotNone(estimate_1920)
        self.assertIsNotNone(estimate_1280)
        assert estimate_1920 is not None and estimate_1280 is not None
        self.assertAlmostEqual(
            estimate_1920.area_normalized,
            estimate_1280.area_normalized,
            places=7,
        )
        self.assertAlmostEqual(estimate_1920.distance_cm, estimate_1280.distance_cm, places=4)

    def test_invalid_geometry_and_confidence_return_none(self) -> None:
        valid = rectangle_for_normalized_area(
            REFERENCE_AREA_NORMALIZED,
            frame_width=1920,
            frame_height=1080,
        )

        self.assertIsNone(
            estimate_area_distance(valid[:3], frame_width=1920, frame_height=1080)
        )
        self.assertIsNone(
            estimate_area_distance(
                [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
                frame_width=1920,
                frame_height=1080,
            )
        )
        invalid_nan = list(valid)
        invalid_nan[0] = (math.nan, invalid_nan[0][1])
        self.assertIsNone(
            estimate_area_distance(invalid_nan, frame_width=1920, frame_height=1080)
        )
        self.assertIsNone(
            estimate_area_distance(
                valid,
                frame_width=1920,
                frame_height=1080,
                confidences=[0.9, 0.9, 0.2, 0.9],
                confidence_threshold=0.4,
            )
        )


if __name__ == "__main__":
    unittest.main()
