from __future__ import annotations

import time
from typing import Any

from twopoint_project.contrl.target_center_servo import (
    TargetCenterServo,
    sleep_for_loop_rate,
)
from twopoint_project.f32c.gimbal import open_serial_gimbal
from twopoint_project.vision.inferencer import build_vision_inferencer
from twopoint_project.vision.pipeline import VisionProducer


def open_camera_capture(
    *,
    camera_index: int,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> object:
    from twopoint_project.vision.capture import CameraCapture

    return CameraCapture(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
    )


def run(args: Any) -> bool:
    inferencer = build_vision_inferencer(
        backend=args.vision_backend,
        onnx_path=args.onnx,
        img_size=args.img_size,
    )
    servo = TargetCenterServo(
        center_x=args.center_x,
        center_y=args.center_y,
        x_gain_deg=args.x_gain_deg,
        y_gain_deg=args.y_gain_deg,
        max_step_deg=args.max_step_deg,
        deadband=args.deadband,
    )

    deadline = time.monotonic() + args.timeout
    settled_frames = 0

    with open_camera_capture(
        camera_index=args.camera,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    ) as capture, open_serial_gimbal(
        port=args.port,
        baudrate=args.baudrate,
        x_id=args.x_id,
        y_id=args.y_id,
        speed_rpm=args.speed_rpm,
        startup_delay=args.startup_delay,
        command_interval=args.command_interval,
        enable_settle_delay=args.enable_settle_delay,
        debug_frames=args.debug_frames,
    ) as gimbal, VisionProducer(
        capture=capture,
        inferencer=inferencer,
    ) as vision:
        print(f"question2: backend={args.vision_backend}")
        print(f"question2: providers={inferencer.providers}")
        print(f"question2: target-only mode, timeout={args.timeout:.2f}s")
        gimbal.initialize()

        while time.monotonic() < deadline:
            loop_started_at = time.monotonic()
            read_timeout = min(max(deadline - time.monotonic(), 0.0), 1.0)
            try:
                vision_frame = vision.read_latest(timeout=read_timeout)
            except TimeoutError:
                continue

            captured = vision_frame.captured
            points = vision_frame.points
            update = servo.update(gimbal, points)

            if update.settled:
                settled_frames += 1
            else:
                settled_frames = 0

            if update.step is not None:
                print(
                    "frame={frame_id} err=({err_x:+.4f},{err_y:+.4f}) "
                    "step=({step_x:+.3f},{step_y:+.3f}) settled={settled}".format(
                        frame_id=captured.frame_id,
                        err_x=update.step.error.x,
                        err_y=update.step.error.y,
                        step_x=update.step.x_delta_deg,
                        step_y=update.step.y_delta_deg,
                        settled=update.settled,
                    )
                )
            else:
                print(f"frame={captured.frame_id} target_center invalid: {update.reason}")

            if settled_frames >= args.stable_frames:
                print(f"question2: settled for {settled_frames} frame(s)")
                return True

            sleep_for_loop_rate(loop_started_at, args.loop_hz)

    print("question2: timeout before stable centering")
    return False
