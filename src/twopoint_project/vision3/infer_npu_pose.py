from __future__ import annotations

import ctypes
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from twopoint_project.vision.inferencer import PointPrediction, VisionInferenceDetails
from twopoint_project.vision3.area_distance import estimate_area_distance


INPUT_SIZE = 640
KEYPOINT_COUNT = 5
MAX_DETECTIONS = 100


class NativePoseDetection(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("score", ctypes.c_float),
        ("keypoints", ctypes.c_float * (KEYPOINT_COUNT * 3)),
    ]


class NativePoseTimingBreakdown(ctypes.Structure):
    _fields_ = [
        ("preprocess_ns", ctypes.c_uint64),
        ("input_flush_ns", ctypes.c_uint64),
        ("network_ns", ctypes.c_uint64),
        ("output_map_ns", ctypes.c_uint64),
        ("decode_ns", ctypes.c_uint64),
        ("output_unmap_ns", ctypes.c_uint64),
        ("result_copy_ns", ctypes.c_uint64),
        ("total_ns", ctypes.c_uint64),
    ]


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    left: int
    top: int
    original_width: int
    original_height: int


@dataclass(frozen=True)
class PoseKeypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseDetection:
    box: tuple[float, float, float, float]
    score: float
    keypoints: tuple[PoseKeypoint, ...]


@dataclass(frozen=True)
class PoseTiming:
    preprocess_ns: int
    inference_ns: int
    postprocess_ns: int


@dataclass(frozen=True)
class PoseNativeTiming:
    preprocess_ns: int
    input_flush_ns: int
    network_ns: int
    output_map_ns: int
    decode_ns: int
    output_unmap_ns: int
    result_copy_ns: int
    total_ns: int


def validate_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def letterbox_rgb_uint8(
    frame_bgr: np.ndarray,
    size: int = INPUT_SIZE,
) -> tuple[np.ndarray, LetterboxMeta]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"expected BGR frame shape [H, W, 3], got {frame_bgr.shape}")
    if frame_bgr.dtype != np.uint8:
        raise ValueError(f"expected uint8 BGR frame, got {frame_bgr.dtype}")

    original_height, original_width = frame_bgr.shape[:2]
    scale = min(size / original_width, size / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        image_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_width = size - resized_width
    pad_height = size - resized_height
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = np.ascontiguousarray(padded, dtype=np.uint8)
    if tensor.shape != (size, size, 3):
        raise RuntimeError(f"letterbox produced unexpected shape {tensor.shape}")
    return tensor, LetterboxMeta(
        scale=scale,
        left=left,
        top=top,
        original_width=original_width,
        original_height=original_height,
    )


def letterbox_meta(frame_bgr: np.ndarray, size: int = INPUT_SIZE) -> LetterboxMeta:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"expected BGR frame shape [H, W, 3], got {frame_bgr.shape}")
    if frame_bgr.dtype != np.uint8:
        raise ValueError(f"expected uint8 BGR frame, got {frame_bgr.dtype}")
    original_height, original_width = frame_bgr.shape[:2]
    scale = min(size / original_width, size / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    return LetterboxMeta(
        scale=scale,
        left=(size - resized_width) // 2,
        top=(size - resized_height) // 2,
        original_width=original_width,
        original_height=original_height,
    )


def restore_xy(x: float, y: float, meta: LetterboxMeta) -> tuple[float, float]:
    restored_x = (float(x) - meta.left) / meta.scale
    restored_y = (float(y) - meta.top) / meta.scale
    return (
        float(np.clip(restored_x, 0.0, meta.original_width - 1.0)),
        float(np.clip(restored_y, 0.0, meta.original_height - 1.0)),
    )


def restore_detection(detection: PoseDetection, meta: LetterboxMeta) -> PoseDetection:
    x1, y1 = restore_xy(detection.box[0], detection.box[1], meta)
    x2, y2 = restore_xy(detection.box[2], detection.box[3], meta)
    keypoints = tuple(
        PoseKeypoint(*restore_xy(point.x, point.y, meta), point.confidence)
        for point in detection.keypoints
    )
    return PoseDetection(
        box=(x1, y1, x2, y2),
        score=detection.score,
        keypoints=keypoints,
    )


def target_center_prediction(
    detections: list[PoseDetection],
    meta: LetterboxMeta,
) -> PointPrediction | None:
    restored = tuple(restore_detection(detection, meta) for detection in detections)
    selected = select_target_detection(restored)
    if selected is None:
        return None
    return target_center_from_restored_detection(selected, meta)


def select_target_detection(
    detections: tuple[PoseDetection, ...],
) -> PoseDetection | None:
    """Choose the accepted box whose semantic center keypoint is most likely."""
    candidates = []
    for detection in detections:
        center = detection.keypoints[0]
        x1, y1, x2, y2 = detection.box
        if (
            isfinite(center.x)
            and isfinite(center.y)
            and isfinite(center.confidence)
            and x1 <= center.x <= x2
            and y1 <= center.y <= y2
        ):
            candidates.append(detection)
    return max(
        candidates,
        key=lambda detection: detection.keypoints[0].confidence,
        default=None,
    )


def target_center_from_restored_detection(
    restored: PoseDetection,
    meta: LetterboxMeta,
) -> PointPrediction:
    keypoint = restored.keypoints[0]
    width_scale = max(meta.original_width - 1, 1)
    height_scale = max(meta.original_height - 1, 1)
    return {
        "label": "target_center",
        "x": float(np.clip(keypoint.x / width_scale, 0.0, 1.0)),
        "y": float(np.clip(keypoint.y / height_scale, 0.0, 1.0)),
        "confidence": float(keypoint.confidence),
    }


class NpuPoseInferencer:
    def __init__(
        self,
        model_path: Path,
        library_path: Path,
        *,
        img_size: int = INPUT_SIZE,
        box_confidence_threshold: float = 0.4,
        nms_threshold: float = 0.45,
    ) -> None:
        if img_size != INPUT_SIZE:
            raise ValueError(f"A733 pose model requires img_size={INPUT_SIZE}, got {img_size}")
        self.model_path = Path(model_path)
        self.library_path = Path(library_path)
        self.img_size = img_size
        self.box_confidence_threshold = validate_unit_interval(
            box_confidence_threshold,
            "box_confidence_threshold",
        )
        self.nms_threshold = validate_unit_interval(nms_threshold, "nms_threshold")
        self._library: ctypes.CDLL | None = None
        self._context: int | None = None
        self._owner_thread: int | None = None
        self._driver_version: int | None = None
        self._native_output = (NativePoseDetection * MAX_DETECTIONS)()
        self.last_timing: PoseTiming | None = None
        self.last_native_timing: PoseNativeTiming | None = None

    @property
    def providers(self) -> list[str]:
        provider = "VIPLite-A733"
        if self._driver_version is not None:
            provider += f"-0x{self._driver_version:08x}"
        return [provider]

    def _load_library(self) -> ctypes.CDLL:
        if self._library is not None:
            return self._library
        if not self.library_path.is_file():
            raise FileNotFoundError(f"NPU bridge library does not exist: {self.library_path}")
        library = ctypes.CDLL(str(self.library_path.resolve()))
        library.pose_create.argtypes = [ctypes.c_char_p, ctypes.c_float, ctypes.c_float]
        library.pose_create.restype = ctypes.c_void_p
        library.pose_infer_rgb640.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(NativePoseDetection),
            ctypes.c_int,
        ]
        library.pose_infer_rgb640.restype = ctypes.c_int
        library.pose_infer_bgr.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.POINTER(NativePoseDetection),
            ctypes.c_int,
        ]
        library.pose_infer_bgr.restype = ctypes.c_int
        library.pose_last_timing.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(NativePoseTimingBreakdown),
        ]
        library.pose_last_timing.restype = ctypes.c_int
        library.pose_driver_version.argtypes = [ctypes.c_void_p]
        library.pose_driver_version.restype = ctypes.c_uint32
        library.pose_last_error.argtypes = [ctypes.c_void_p]
        library.pose_last_error.restype = ctypes.c_char_p
        library.pose_destroy.argtypes = [ctypes.c_void_p]
        library.pose_destroy.restype = None
        self._library = library
        return library

    def _native_error(self, context: int | None = None) -> str:
        library = self._load_library()
        message = library.pose_last_error(context)
        return message.decode("utf-8", errors="replace") if message else "unknown VIPLite error"

    def _ensure_context(self) -> tuple[ctypes.CDLL, int]:
        current_thread = threading.get_ident()
        if self._context is not None:
            if self._owner_thread != current_thread:
                raise RuntimeError("NPU pose inferencer must stay on the thread that initialized it")
            return self._load_library(), self._context
        if not self.model_path.is_file():
            raise FileNotFoundError(f"NPU model does not exist: {self.model_path}")

        library = self._load_library()
        context = library.pose_create(
            str(self.model_path.resolve()).encode("utf-8"),
            self.box_confidence_threshold,
            self.nms_threshold,
        )
        if not context:
            raise RuntimeError(f"failed to initialize A733 NPU model: {self._native_error(None)}")
        self._context = context
        self._owner_thread = current_thread
        self._driver_version = int(library.pose_driver_version(context))
        return library, context

    def predict_detections(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[list[PoseDetection], LetterboxMeta]:
        meta = letterbox_meta(frame_bgr, self.img_size)
        if not frame_bgr.flags.c_contiguous:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        library, context = self._ensure_context()
        count = library.pose_infer_bgr(
            context,
            frame_bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            frame_bgr.nbytes,
            meta.original_width,
            meta.original_height,
            frame_bgr.strides[0],
            self._native_output,
            MAX_DETECTIONS,
        )
        if count < 0:
            raise RuntimeError(f"A733 NPU inference failed: {self._native_error(context)}")

        native_timing = NativePoseTimingBreakdown()
        if library.pose_last_timing(context, ctypes.byref(native_timing)) != 0:
            raise RuntimeError(f"failed to read A733 NPU timing: {self._native_error(context)}")
        self.last_native_timing = PoseNativeTiming(
            preprocess_ns=int(native_timing.preprocess_ns),
            input_flush_ns=int(native_timing.input_flush_ns),
            network_ns=int(native_timing.network_ns),
            output_map_ns=int(native_timing.output_map_ns),
            decode_ns=int(native_timing.decode_ns),
            output_unmap_ns=int(native_timing.output_unmap_ns),
            result_copy_ns=int(native_timing.result_copy_ns),
            total_ns=int(native_timing.total_ns),
        )

        postprocess_started = time.monotonic_ns()
        detections: list[PoseDetection] = []
        for index in range(count):
            native = self._native_output[index]
            keypoints = tuple(
                PoseKeypoint(
                    float(native.keypoints[keypoint * 3]),
                    float(native.keypoints[keypoint * 3 + 1]),
                    float(native.keypoints[keypoint * 3 + 2]),
                )
                for keypoint in range(KEYPOINT_COUNT)
            )
            detections.append(
                PoseDetection(
                    box=(
                        float(native.x1),
                        float(native.y1),
                        float(native.x2),
                        float(native.y2),
                    ),
                    score=float(native.score),
                    keypoints=keypoints,
                )
            )
        python_result_ns = time.monotonic_ns() - postprocess_started
        self.last_timing = PoseTiming(
            preprocess_ns=(
                self.last_native_timing.preprocess_ns
                + self.last_native_timing.input_flush_ns
            ),
            inference_ns=self.last_native_timing.network_ns,
            postprocess_ns=(
                self.last_native_timing.output_map_ns
                + self.last_native_timing.decode_ns
                + self.last_native_timing.output_unmap_ns
                + self.last_native_timing.result_copy_ns
                + python_result_ns
            ),
        )
        return detections, meta

    def predict(self, frame_bgr: np.ndarray) -> list[PointPrediction]:
        return self.predict_with_details(frame_bgr).points

    def predict_with_details(
        self,
        frame_bgr: np.ndarray,
    ) -> VisionInferenceDetails:
        detections, meta = self.predict_detections(frame_bgr)
        restore_started = time.monotonic_ns()
        restored = tuple(
            restore_detection(detection, meta)
            for detection in detections
            if detection.score >= self.box_confidence_threshold
        )
        selected = select_target_detection(restored)
        area_distance = None
        target_corners_normalized = ()
        if selected is None:
            points = []
        else:
            points = [target_center_from_restored_detection(selected, meta)]
            corners = selected.keypoints[1:5]
            width_scale = max(meta.original_width - 1, 1)
            height_scale = max(meta.original_height - 1, 1)
            target_corners_normalized = tuple(
                (
                    float(np.clip(corner.x / width_scale, 0.0, 1.0)),
                    float(np.clip(corner.y / height_scale, 0.0, 1.0)),
                )
                for corner in corners
            )
            area_distance = estimate_area_distance(
                [(corner.x, corner.y) for corner in corners],
                frame_width=meta.original_width,
                frame_height=meta.original_height,
                confidences=[corner.confidence for corner in corners],
            )
        if self.last_timing is not None:
            self.last_timing = PoseTiming(
                preprocess_ns=self.last_timing.preprocess_ns,
                inference_ns=self.last_timing.inference_ns,
                postprocess_ns=self.last_timing.postprocess_ns
                + time.monotonic_ns()
                - restore_started,
            )
        return VisionInferenceDetails(
            points=points,
            detections=restored,
            target_area_normalized=(
                None if area_distance is None else area_distance.area_normalized
            ),
            target_distance_cm=(
                None if area_distance is None else area_distance.distance_cm
            ),
            target_corners_normalized=target_corners_normalized,
        )

    def close(self) -> None:
        if self._context is None:
            return
        if self._owner_thread != threading.get_ident():
            raise RuntimeError("NPU pose inferencer must be closed on the thread that initialized it")
        assert self._library is not None
        self._library.pose_destroy(self._context)
        self._context = None
        self._owner_thread = None

    def __enter__(self) -> NpuPoseInferencer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
