from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Protocol

from twopoint_project.vision.inferencer import (
    PointPrediction,
    TargetCorners,
    VisionInferenceDetails,
    VisionInferencer,
)
from twopoint_project.vision3.laser_area_mapping import (
    LASER_POINT_LABEL,
    LaserAreaPrediction,
    predict_laser_point_from_normalized_area,
)
from twopoint_project.vision.target_filter import (
    TargetCenterFilter,
    TargetCenterFilterConfig,
)


class FrameCapture(Protocol):
    def read_frame(self, timeout: float | None = None) -> Any:
        ...


@dataclass(frozen=True)
class VisionResult:
    camera_id: str
    stream_generation: int
    source_frame_id: int
    captured_at_monotonic_ns: int
    inference_started_monotonic_ns: int
    inference_finished_monotonic_ns: int
    captured: Any
    points: list[PointPrediction]
    raw_target_center: PointPrediction | None = None
    detections: tuple[Any, ...] = ()
    source_frames_skipped: int = 0
    preprocessing_duration_ns: int | None = None
    model_inference_duration_ns: int | None = None
    postprocessing_duration_ns: int | None = None
    target_area_normalized: float | None = None
    target_distance_cm: float | None = None
    laser_area_prediction: LaserAreaPrediction | None = None
    target_corners_normalized: TargetCorners = ()

    @property
    def inference_duration_ns(self) -> int:
        return self.inference_finished_monotonic_ns - self.inference_started_monotonic_ns

    @property
    def source_identity(self) -> tuple[int, int]:
        return self.stream_generation, self.source_frame_id

    def age_ns(self, now_monotonic_ns: int | None = None) -> int:
        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        return max(now - self.captured_at_monotonic_ns, 0)


VisionFrame = VisionResult


class LatestVisionState:
    """Single-consumer overwrite slot containing only the newest completed result."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: VisionResult | None = None
        self._version = 0
        self._last_read_version = 0
        self._closed = False
        self.overwrite_count = 0

    def publish(self, result: VisionResult) -> None:
        with self._condition:
            if self._closed:
                return
            if self._version > self._last_read_version:
                self.overwrite_count += 1
            self._latest = result
            self._version += 1
            self._condition.notify_all()

    def read_latest(self, timeout: float | None = None) -> VisionResult:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._version <= self._last_read_version:
                if self._closed:
                    raise RuntimeError("vision state closed before producing a new result")
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a vision result")
                self._condition.wait(remaining)
            self._last_read_version = self._version
            assert self._latest is not None
            return self._latest

    def read_nowait_latest(self) -> VisionResult | None:
        with self._condition:
            if self._version <= self._last_read_version:
                return None
            self._last_read_version = self._version
            return self._latest

    def empty(self) -> bool:
        with self._condition:
            return self._version <= self._last_read_version

    def clear(self) -> None:
        with self._condition:
            self._latest = None
            self._last_read_version = self._version
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


LatestVisionQueue = LatestVisionState


class VisionProducer:
    def __init__(
        self,
        *,
        capture: FrameCapture,
        inferencer: VisionInferencer,
        queue: LatestVisionState | None = None,
        join_timeout: float = 2.0,
        target_filter_config: TargetCenterFilterConfig | None = None,
    ) -> None:
        self.capture = capture
        self.inferencer = inferencer
        self.queue = queue or LatestVisionState()
        self.join_timeout = join_timeout
        self.target_filter = TargetCenterFilter(
            target_filter_config or TargetCenterFilterConfig()
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def __enter__(self) -> VisionProducer:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="twopoint-vision-inference",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.join_timeout)
        self.queue.close()

    def read_latest(self, timeout: float | None = None) -> VisionResult:
        if self._error is not None and self.queue.empty():
            raise RuntimeError("vision inference failed") from self._error
        try:
            return self.queue.read_latest(timeout=timeout)
        except TimeoutError:
            if self._error is not None:
                raise RuntimeError("vision inference failed") from self._error
            if self._stop_event.is_set():
                raise RuntimeError("vision inference stopped before producing a result")
            raise

    def read_nowait_latest(self) -> VisionResult | None:
        if self._error is not None and self.queue.empty():
            raise RuntimeError("vision inference failed") from self._error
        return self.queue.read_nowait_latest()

    def _run(self) -> None:
        last_source_identity: tuple[int, int] | None = None
        last_generation: int | None = None
        try:
            while not self._stop_event.is_set():
                try:
                    captured = self.capture.read_frame(timeout=0.5)
                except TimeoutError:
                    continue
                source_identity = (
                    int(getattr(captured, "stream_generation", 0)),
                    int(captured.frame_id),
                )
                if source_identity == last_source_identity:
                    continue
                previous_source_identity = last_source_identity
                if last_generation is not None and source_identity[0] != last_generation:
                    self.queue.clear()
                last_source_identity = source_identity
                last_generation = source_identity[0]
                inference_started = time.monotonic_ns()
                predict_with_details = getattr(self.inferencer, "predict_with_details", None)
                if callable(predict_with_details):
                    details = predict_with_details(captured.frame_bgr)
                    if isinstance(details, VisionInferenceDetails):
                        points = details.points
                        detections = details.detections
                        target_area_normalized = details.target_area_normalized
                        target_distance_cm = details.target_distance_cm
                        target_corners_normalized = details.target_corners_normalized
                    else:
                        points, detections = details
                        target_area_normalized = None
                        target_distance_cm = None
                        target_corners_normalized = ()
                else:
                    points = self.inferencer.predict(captured.frame_bgr)
                    detections = ()
                    target_area_normalized = None
                    target_distance_cm = None
                    target_corners_normalized = ()
                raw_target_center = max(
                    (
                        dict(point)
                        for point in points
                        if point["label"] == "target_center"
                    ),
                    key=lambda point: float(point["confidence"]),
                    default=None,
                )
                points = self.target_filter.filter_points(points)
                laser_area_prediction = self._predict_laser_point(
                    frame_bgr=captured.frame_bgr,
                    points=points,
                    target_area_normalized=target_area_normalized,
                )
                if laser_area_prediction is not None and not any(
                    point["label"] == LASER_POINT_LABEL for point in points
                ):
                    points.append(laser_area_prediction.as_point_prediction())
                inference_finished = time.monotonic_ns()
                timing = getattr(self.inferencer, "last_timing", None)
                skipped = 0
                if (
                    previous_source_identity is not None
                    and source_identity[0] == previous_source_identity[0]
                ):
                    skipped = max(source_identity[1] - previous_source_identity[1] - 1, 0)
                self.queue.publish(
                    VisionResult(
                        camera_id=str(getattr(captured, "camera_id", "unknown")),
                        stream_generation=int(getattr(captured, "stream_generation", 0)),
                        source_frame_id=int(captured.frame_id),
                        captured_at_monotonic_ns=int(
                            getattr(captured, "captured_at_monotonic_ns", inference_started)
                        ),
                        inference_started_monotonic_ns=inference_started,
                        inference_finished_monotonic_ns=inference_finished,
                        captured=captured,
                        points=points,
                        raw_target_center=raw_target_center,
                        detections=tuple(detections),
                        source_frames_skipped=skipped,
                        preprocessing_duration_ns=getattr(timing, "preprocess_ns", None),
                        model_inference_duration_ns=getattr(timing, "inference_ns", None),
                        postprocessing_duration_ns=getattr(timing, "postprocess_ns", None),
                        target_area_normalized=target_area_normalized,
                        target_distance_cm=target_distance_cm,
                        laser_area_prediction=laser_area_prediction,
                        target_corners_normalized=target_corners_normalized,
                    )
                )
        except BaseException as exc:
            self._error = exc
            self._stop_event.set()
        finally:
            close = getattr(self.inferencer, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    if self._error is None:
                        self._error = exc
                    self._stop_event.set()

    @staticmethod
    def _predict_laser_point(
        *,
        frame_bgr: Any,
        points: list[PointPrediction],
        target_area_normalized: float | None,
    ) -> LaserAreaPrediction | None:
        if target_area_normalized is None:
            return None
        shape = getattr(frame_bgr, "shape", None)
        if shape is None or len(shape) < 2:
            return None
        frame_height = int(shape[0])
        frame_width = int(shape[1])
        target_confidence = max(
            (
                float(point["confidence"])
                for point in points
                if point["label"] == "target_center"
            ),
            default=0.0,
        )
        return predict_laser_point_from_normalized_area(
            target_area_normalized,
            frame_width=frame_width,
            frame_height=frame_height,
            confidence=target_confidence,
        )
