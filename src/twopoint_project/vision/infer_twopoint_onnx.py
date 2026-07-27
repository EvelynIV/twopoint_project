from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError(
        "This script requires `onnxruntime`. Install it with `poetry add onnxruntime` "
        "or `poetry add onnxruntime-gpu`."
    ) from exc


POINT_NAMES = ("target_center", "laser_point")
POINT_COLORS = {
    "target_center": (0, 220, 0),
    "laser_point": (0, 0, 255),
}


class PointPrediction(TypedDict):
    label: str
    x: float
    y: float
    confidence: float


def default_output_path(image_path: Path, output_dir: Path) -> Path:
    suffix = image_path.suffix if image_path.suffix else ".jpg"
    return output_dir / f"{image_path.stem}_onnx_pred{suffix}"


def letterbox(
    image_rgb: np.ndarray,
    img_size: int,
    pad_value: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = image_rgb.shape[:2]
    ratio = min(img_size / height, img_size / width)
    new_w = int(round(width * ratio))
    new_h = int(round(height * ratio))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = img_size - new_w
    pad_h = img_size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    return padded, ratio, (left, top)


def create_session(onnx_path: Path) -> ort.InferenceSession:
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")

    available = ort.get_available_providers()
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(onnx_path), providers=providers)


def resolve_input_shape(session: ort.InferenceSession, fallback: int) -> tuple[int, int, int, int]:
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) != 4:
        raise RuntimeError(f"Expected input shape [B, 3, H, W], got {input_shape}.")

    batch, channels, height, width = input_shape
    batch = int(batch) if isinstance(batch, int) else 1
    channels = int(channels) if isinstance(channels, int) else 3
    height = int(height) if isinstance(height, int) else fallback
    width = int(width) if isinstance(width, int) else fallback

    if batch != 1:
        raise RuntimeError(f"This script expects fixed batch size 1, got batch={batch}.")
    if channels != 3:
        raise RuntimeError(f"Expected 3 input channels, got {channels}.")
    if height != width:
        raise RuntimeError(f"Expected square input size, got height={height}, width={width}.")
    return batch, channels, height, width


def preprocess_frame(frame_bgr: np.ndarray, img_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR frame shape [H, W, 3], got {frame_bgr.shape}.")

    image_bgr = frame_bgr
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    padded_rgb, ratio, pad = letterbox(image_rgb, img_size)
    tensor = padded_rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)

    meta = {
        "ratio": ratio,
        "pad": pad,
        "orig_h": image_bgr.shape[0],
        "orig_w": image_bgr.shape[1],
        "orig_bgr": image_bgr,
    }
    return tensor, meta


def preprocess_image(image_path: Path, img_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return preprocess_frame(image_bgr, img_size)


def parse_outputs(session: ort.InferenceSession, raw_outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    outputs_by_name = {
        output.name: value for output, value in zip(session.get_outputs(), raw_outputs)
    }

    if "points" in outputs_by_name and "scores" in outputs_by_name:
        points = outputs_by_name["points"]
        scores = outputs_by_name["scores"]
    elif len(raw_outputs) >= 2:
        points, scores = raw_outputs[:2]
    else:
        raise RuntimeError(
            f"Expected ONNX outputs `points` and `scores`, got {len(raw_outputs)} output(s)."
        )

    points = np.asarray(points, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if points.ndim != 3 or points.shape[1:] != (2, 2):
        raise RuntimeError(f"Expected points shape [B, 2, 2], got {points.shape}.")
    if scores.ndim == 3 and scores.shape[-1] == 1:
        scores = scores[..., 0]
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise RuntimeError(f"Expected scores shape [B, 2], got {scores.shape}.")
    return points, scores


def restore_normalized_points(
    points: np.ndarray,
    scores: np.ndarray,
    meta: dict[str, Any],
    img_size: int,
) -> list[PointPrediction]:
    pad_x, pad_y = meta["pad"]
    ratio = meta["ratio"]
    image_width = int(meta["orig_w"])
    image_height = int(meta["orig_h"])
    width_scale = max(image_width - 1, 1)
    height_scale = max(image_height - 1, 1)
    restored: list[PointPrediction] = []

    for point_index, name in enumerate(POINT_NAMES):
        x_model = float(points[point_index, 0]) * img_size
        y_model = float(points[point_index, 1]) * img_size
        x_orig = (x_model - pad_x) / ratio
        y_orig = (y_model - pad_y) / ratio

        x_orig = float(np.clip(x_orig, 0.0, meta["orig_w"] - 1.0))
        y_orig = float(np.clip(y_orig, 0.0, meta["orig_h"] - 1.0))
        restored.append({
            "label": name,
            "x": float(np.clip(x_orig / width_scale, 0.0, 1.0)),
            "y": float(np.clip(y_orig / height_scale, 0.0, 1.0)),
            "confidence": float(scores[point_index]),
        })
    return restored


def infer_points(
    session: ort.InferenceSession,
    frame_bgr: np.ndarray,
    img_size: int,
    input_name: str | None = None,
) -> list[PointPrediction]:
    input_name = input_name or session.get_inputs()[0].name
    batch, channels, resolved_img_size, _ = resolve_input_shape(session, img_size)
    tensor, meta = preprocess_frame(frame_bgr, resolved_img_size)
    expected_shape = (batch, channels, resolved_img_size, resolved_img_size)
    if tensor.shape != expected_shape:
        raise RuntimeError(f"Expected input tensor shape {expected_shape}, got {tensor.shape}.")

    raw_outputs = session.run(None, {input_name: tensor})
    points, scores = parse_outputs(session, raw_outputs)
    return restore_normalized_points(points[0], scores[0], meta, resolved_img_size)


class TwoPointOnnxInferencer:
    def __init__(self, onnx_path: Path, img_size: int = 640) -> None:
        self.session = create_session(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.img_size, _ = resolve_input_shape(self.session, img_size)

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def predict(self, frame_bgr: np.ndarray) -> list[PointPrediction]:
        return infer_points(self.session, frame_bgr, self.img_size, self.input_name)


def points_for_drawing(
    points: list[PointPrediction],
    image_width: int,
    image_height: int,
) -> dict[str, dict[str, float]]:
    width_scale = max(image_width - 1, 1)
    height_scale = max(image_height - 1, 1)
    return {
        point["label"]: {
            "x": point["x"] * width_scale,
            "y": point["y"] * height_scale,
            "confidence": point["confidence"],
        }
        for point in points
    }


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    padding = 4
    height, width = image.shape[:2]
    text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]

    x = min(max(origin[0], 0), max(width - text_width - padding * 2, 0))
    y = origin[1]
    if y - text_height - padding * 2 < 0:
        y = min(y + text_height + padding * 3, height - padding)
    y = min(max(y, text_height + padding * 2), height - padding)

    top_left = (x, y - text_height - padding * 2)
    bottom_right = (x + text_width + padding * 2, y)
    cv2.rectangle(image, top_left, bottom_right, (20, 20, 20), -1)
    cv2.rectangle(image, top_left, bottom_right, color, 1)
    cv2.putText(image, text, (x + padding, y - padding), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_predictions(
    image_bgr: np.ndarray,
    points: dict[str, dict[str, float]],
    radius: int,
) -> np.ndarray:
    annotated = image_bgr.copy()

    for name in POINT_NAMES:
        point = points[name]
        if point["confidence"] <= 0.0:
            continue

        x = int(round(point["x"]))
        y = int(round(point["y"]))
        color = POINT_COLORS[name]
        cv2.circle(annotated, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
        label = f"{name}: ({point['x']:.1f}, {point['y']:.1f}) conf={point['confidence']:.2f}"
        draw_label(annotated, label, (x + radius + 6, y - radius - 6), color)
    return annotated
