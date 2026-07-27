from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence

import cv2
import numpy as np


REFERENCE_DISTANCE_CM = 150.0
REFERENCE_AREA_PIXELS = 37142.0
REFERENCE_FRAME_WIDTH = 1920
REFERENCE_FRAME_HEIGHT = 1080
REFERENCE_AREA_NORMALIZED = REFERENCE_AREA_PIXELS / (
    REFERENCE_FRAME_WIDTH * REFERENCE_FRAME_HEIGHT
)


@dataclass(frozen=True)
class AreaDistanceEstimate:
    area_normalized: float
    distance_cm: float


def estimate_area_distance(
    corners_xy: Sequence[tuple[float, float]],
    *,
    frame_width: int,
    frame_height: int,
    confidences: Sequence[float] | None = None,
    confidence_threshold: float = 0.0,
) -> AreaDistanceEstimate | None:
    """Estimate distance from four target corners without perspective correction."""
    if frame_width <= 0 or frame_height <= 0 or len(corners_xy) != 4:
        return None
    if confidences is not None:
        if len(confidences) != 4:
            return None
        if any(
            not isfinite(float(confidence))
            or float(confidence) < confidence_threshold
            for confidence in confidences
        ):
            return None

    corners = np.asarray(corners_xy, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return None
    corners[:, 0] /= float(frame_width)
    corners[:, 1] /= float(frame_height)
    if np.any(corners < 0.0) or np.any(corners > 1.0):
        return None

    hull = cv2.convexHull(corners.astype(np.float32))
    if hull.shape[0] != 4:
        return None
    area_normalized = abs(float(cv2.contourArea(hull)))
    if not isfinite(area_normalized) or area_normalized <= 0.0:
        return None

    distance_cm = REFERENCE_DISTANCE_CM * sqrt(
        REFERENCE_AREA_NORMALIZED / area_normalized
    )
    if not isfinite(distance_cm):
        return None
    return AreaDistanceEstimate(
        area_normalized=area_normalized,
        distance_cm=distance_cm,
    )
