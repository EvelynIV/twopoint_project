#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import cv2
from dotenv import load_dotenv

from twopoint_project.config import runtime_config_from_env
from twopoint_project.vision.capture import CameraCapture


DEFAULT_DURATION_SECONDS = 120.0
DEFAULT_OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record the latest frames from the shared GStreamer camera capture path."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path. Defaults to outputs/camera_YYYYmmdd_HHMMSS.mp4.",
    )
    return parser.parse_args()


def default_output_path(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"camera_{stamp}.mp4"


def create_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    return writer


def record_video(args: argparse.Namespace) -> Path:
    if args.duration <= 0:
        raise ValueError("duration must be greater than 0")

    load_dotenv()
    camera_config = runtime_config_from_env().camera
    output_path = Path(args.output) if args.output else default_output_path()
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    first_frame_id: int | None = None
    last_frame_id: int | None = None
    start = 0.0

    with CameraCapture(
        width=camera_config.width,
        height=camera_config.height,
        fps=camera_config.fps,
    ) as capture:
        start = time.monotonic()
        try:
            while time.monotonic() - start < args.duration:
                captured = capture.read_frame(timeout=2.0)
                frame = captured.frame_bgr
                if writer is None:
                    writer = create_writer(
                        output_path,
                        camera_config.fps,
                        (captured.width, captured.height),
                    )
                    print(
                        f"Recording device={captured.camera_id}, "
                        f"size={captured.width}x{captured.height}, "
                        f"fps={camera_config.fps:g}, duration={args.duration:g}s"
                    )
                    print(f"Negotiated caps: {capture.negotiated_caps}")
                    print(f"Saving to: {output_path}")
                    first_frame_id = captured.frame_id

                writer.write(frame)
                last_frame_id = captured.frame_id
                frame_count += 1
        finally:
            if writer is not None:
                writer.release()

    elapsed = max(time.monotonic() - start, 1e-9)
    skipped = 0
    if first_frame_id is not None and last_frame_id is not None:
        skipped = max(last_frame_id - first_frame_id + 1 - frame_count, 0)
    print(
        f"Saved {frame_count} frame(s) in {elapsed:.2f}s, "
        f"actual fps={frame_count / elapsed:.2f}, skipped_latest={skipped}"
    )
    return output_path


def main() -> None:
    record_video(parse_args())


if __name__ == "__main__":
    main()
