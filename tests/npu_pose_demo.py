from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import cv2
from dotenv import load_dotenv

from twopoint_project.vision3.infer_npu_pose import (
    NpuPoseInferencer,
    PoseDetection,
    restore_detection,
)


DEFAULT_MODEL_PATH = "model-bin/pose/best_pcq_a733.nb"
DEFAULT_LIBRARY_PATH = "build/npu/libyolo11_pose_npu.so"
DEFAULT_OUTPUT_PATH = "outputs/npu_pose_demo.mp4"


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run the A733 YOLO11n-Pose model on a video and save an annotated MP4."
    )
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("TWOPOINT_NPU_MODEL_PATH", DEFAULT_MODEL_PATH)),
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(os.getenv("TWOPOINT_NPU_LIBRARY_PATH", DEFAULT_LIBRARY_PATH)),
    )
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--box-confidence-threshold",
        type=float,
        default=env_float("TWOPOINT_NPU_BOX_CONFIDENCE_THRESHOLD", 0.4),
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=env_float("TWOPOINT_NPU_NMS_THRESHOLD", 0.45),
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=1,
        help="Keypoint radius in pixels (default: 1).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames; 0 processes the entire video.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N frames; 0 disables progress output.",
    )
    return parser.parse_args()


def draw_detection(
    frame: cv2.typing.MatLike,
    detection: PoseDetection,
    *,
    point_radius: int,
) -> None:
    x1, y1, x2, y2 = detection.box
    box_start = (int(round(x1)), int(round(y1)))
    box_end = (int(round(x2)), int(round(y2)))
    box_color = (255, 0, 0)
    cv2.rectangle(frame, box_start, box_end, box_color, 1, cv2.LINE_8)
    cv2.putText(
        frame,
        f"target {detection.score:.3f}",
        (box_start[0], max(box_start[1] - 4, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        box_color,
        1,
        cv2.LINE_AA,
    )

    for keypoint_index, keypoint in enumerate(detection.keypoints):
        color = (0, 255, 0) if keypoint_index == 0 else (0, 165, 255)
        point = (int(round(keypoint.x)), int(round(keypoint.y)))
        cv2.circle(frame, point, point_radius, color, -1, cv2.LINE_8)
        cv2.putText(
            frame,
            f"k{keypoint_index}",
            (point[0] + point_radius + 2, point[1] - point_radius - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            color,
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    if args.point_radius < 1:
        raise ValueError("--point-radius must be at least 1")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be 0 or greater")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be 0 or greater")
    if not args.video.is_file():
        raise FileNotFoundError(f"input video does not exist: {args.video}")
    if args.video.resolve() == args.output.resolve():
        raise ValueError("input and output video paths must be different")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"failed to open input video: {args.video}")

    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(f"input video has no readable frames: {args.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0.0:
        source_fps = 30.0
    height, width = frame.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(f"failed to open output video: {args.output}")

    inferencer = NpuPoseInferencer(
        args.model,
        args.library,
        box_confidence_threshold=args.box_confidence_threshold,
        nms_threshold=args.nms_threshold,
    )
    frame_count = 0
    detection_count = 0
    inference_seconds = 0.0
    started = time.perf_counter()
    try:
        while frame is not None:
            inference_started = time.perf_counter()
            detections, meta = inferencer.predict_detections(frame)
            inference_seconds += time.perf_counter() - inference_started
            restored = [restore_detection(detection, meta) for detection in detections]
            for detection in restored:
                draw_detection(
                    frame,
                    detection,
                    point_radius=args.point_radius,
                )
            writer.write(frame)

            frame_count += 1
            detection_count += len(restored)
            if args.progress_every and frame_count % args.progress_every == 0:
                print(f"processed={frame_count} detections={detection_count}")
            if args.max_frames and frame_count >= args.max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                frame = None
    finally:
        inferencer.close()
        writer.release()
        capture.release()

    elapsed = time.perf_counter() - started
    processing_fps = frame_count / elapsed if elapsed > 0.0 else 0.0
    inference_fps = frame_count / inference_seconds if inference_seconds > 0.0 else 0.0
    print(f"providers={inferencer.providers}")
    print(f"model={args.model}")
    print(f"input={args.video}")
    print(f"output={args.output}")
    print(f"video={width}x{height} source_fps={source_fps:.3f}")
    print(f"frames={frame_count} detections={detection_count}")
    print(f"processing_fps={processing_fps:.3f} inference_fps={inference_fps:.3f}")


if __name__ == "__main__":
    main()
