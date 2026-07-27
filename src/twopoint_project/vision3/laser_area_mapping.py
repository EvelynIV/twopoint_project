"""Map detected target area to the calibrated laser point.

The fitted equations come from the combined area-only calibration data and are
scaled to the production 1280x720 frame.  ``A`` is the target quadrilateral
area in pixels and ``log`` is the natural logarithm::

    x = 677.069517697516 - 2.189607919627936 * log(A)
    y = 516.669698671584 - 14.517075591063199 * log(A)

Runtime inputs are converted to that reference resolution through normalized
area, then the fitted point is scaled back to the runtime frame.  This assumes
the runtime image has the same field of view and crop as the calibration video.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log

from twopoint_project.vision.inferencer import PointPrediction


LASER_POINT_LABEL = "laser_point"
CALIBRATION_FRAME_WIDTH = 1280
CALIBRATION_FRAME_HEIGHT = 720
CALIBRATION_MIN_AREA_PX2 = 12241.043494837151
CALIBRATION_MAX_AREA_PX2 = 155146.30112338564

X_INTERCEPT = 677.069517697516
X_LOG_AREA_COEFFICIENT = -2.189607919627936
Y_INTERCEPT = 516.669698671584
Y_LOG_AREA_COEFFICIENT = -14.517075591063199


@dataclass(frozen=True)
class LaserAreaPrediction:
    """One area-derived laser prediction in the runtime frame.

    ``area_px2`` is the measured area at the runtime resolution.
    ``calibration_area_px2`` is the same normalized area expressed at
    1280x720.  ``fitted_calibration_area_px2`` is the value actually passed to
    the fitted equations after limiting it to the calibrated range.
    """

    area_px2: float
    area_normalized: float
    calibration_area_px2: float
    fitted_calibration_area_px2: float
    area_clamped: bool
    x_px: float
    y_px: float
    x: float
    y: float
    confidence: float

    def as_point_prediction(self) -> PointPrediction:
        """Return the normalized point contract already consumed by control."""
        return {
            "label": LASER_POINT_LABEL,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
        }


def predict_laser_point_from_area(
    area_px2: float,
    *,
    frame_width: int,
    frame_height: int,
    confidence: float = 1.0,
) -> LaserAreaPrediction | None:
    """Predict a laser point from target area measured in a runtime frame.

    Areas outside the calibration interval are clamped instead of extrapolated.
    Invalid measurements return ``None`` so a consumer cannot accidentally
    reuse a previous frame's prediction.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")

    area_px2 = float(area_px2)
    confidence = float(confidence)
    if (
        not isfinite(area_px2)
        or area_px2 <= 0.0
        or not isfinite(confidence)
    ):
        return None

    frame_area = float(frame_width * frame_height)
    area_normalized = area_px2 / frame_area
    calibration_frame_area = float(
        CALIBRATION_FRAME_WIDTH * CALIBRATION_FRAME_HEIGHT
    )
    calibration_area = area_normalized * calibration_frame_area
    fitted_area = min(
        max(calibration_area, CALIBRATION_MIN_AREA_PX2),
        CALIBRATION_MAX_AREA_PX2,
    )

    log_area = log(fitted_area)
    calibration_x = X_INTERCEPT + X_LOG_AREA_COEFFICIENT * log_area
    calibration_y = Y_INTERCEPT + Y_LOG_AREA_COEFFICIENT * log_area
    x_px = calibration_x * frame_width / CALIBRATION_FRAME_WIDTH
    y_px = calibration_y * frame_height / CALIBRATION_FRAME_HEIGHT
    x_px = min(max(x_px, 0.0), float(frame_width - 1))
    y_px = min(max(y_px, 0.0), float(frame_height - 1))

    return LaserAreaPrediction(
        area_px2=area_px2,
        area_normalized=area_normalized,
        calibration_area_px2=calibration_area,
        fitted_calibration_area_px2=fitted_area,
        area_clamped=calibration_area != fitted_area,
        x_px=x_px,
        y_px=y_px,
        x=x_px / max(frame_width - 1, 1),
        y=y_px / max(frame_height - 1, 1),
        confidence=min(max(confidence, 0.0), 1.0),
    )


def predict_laser_point_from_normalized_area(
    area_normalized: float,
    *,
    frame_width: int,
    frame_height: int,
    confidence: float = 1.0,
) -> LaserAreaPrediction | None:
    """Predict from an area divided by the source frame's total pixel area."""
    area_normalized = float(area_normalized)
    if not isfinite(area_normalized) or area_normalized <= 0.0:
        return None
    return predict_laser_point_from_area(
        area_normalized * frame_width * frame_height,
        frame_width=frame_width,
        frame_height=frame_height,
        confidence=confidence,
    )
