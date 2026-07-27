from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from twopoint_project.command.app import app
from twopoint_project.config import load_task_config


class CommandAppTest(unittest.TestCase):
    def test_run_loads_config_and_dispatches_task(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        config_path = repo_root / "configs/tasks/center_then_flash.json"
        runner = CliRunner()
        seen: dict[str, object] = {}

        def fake_run_task(task_config: object, runtime_config: object) -> bool:
            seen["task_config"] = task_config
            seen["runtime_config"] = runtime_config
            return True

        env = {
            "TWOPOINT_CAMERA_WIDTH": "1280",
            "TWOPOINT_CAMERA_HEIGHT": "720",
            "TWOPOINT_CAMERA_FPS": "30",
            "TWOPOINT_VISION_BACKEND": "traditional",
            "TWOPOINT_ONNX_PATH": "model-bin/test.onnx",
            "TWOPOINT_NPU_MODEL_PATH": "model-bin/pose/test.nb",
            "TWOPOINT_NPU_LIBRARY_PATH": "build/test/libpose.so",
            "TWOPOINT_IMG_SIZE": "320",
            "TWOPOINT_NPU_BOX_CONFIDENCE_THRESHOLD": "0.25",
            "TWOPOINT_NPU_NMS_THRESHOLD": "0.55",
            "TWOPOINT_WEBRTC_ENABLED": "false",
            "TWOPOINT_WEBRTC_HOST": "127.0.0.1",
            "TWOPOINT_WEBRTC_PORT": "18080",
        }
        with patch("twopoint_project.command.app.run_task", side_effect=fake_run_task):
            result = runner.invoke(app, ["run", "--config", str(config_path)], env=env)

        self.assertEqual(result.exit_code, 0, result.output)
        task_config = seen["task_config"]
        runtime_config = seen["runtime_config"]
        self.assertEqual(getattr(task_config, "mode"), "center_then_flash")
        self.assertEqual(runtime_config.camera.width, 1280)
        self.assertEqual(runtime_config.camera.height, 720)
        self.assertEqual(runtime_config.camera.fps, 30)
        self.assertEqual(runtime_config.vision.backend, "traditional")
        self.assertEqual(runtime_config.vision.onnx_path, "model-bin/test.onnx")
        self.assertEqual(runtime_config.vision.npu_model_path, "model-bin/pose/test.nb")
        self.assertEqual(runtime_config.vision.npu_library_path, "build/test/libpose.so")
        self.assertEqual(runtime_config.vision.img_size, 320)
        self.assertEqual(runtime_config.vision.npu_box_confidence_threshold, 0.25)
        self.assertEqual(runtime_config.vision.npu_nms_threshold, 0.55)
        self.assertFalse(runtime_config.webrtc.enabled)
        self.assertEqual(runtime_config.webrtc.host, "127.0.0.1")
        self.assertEqual(runtime_config.webrtc.port, 18080)

    def test_center_flash_track_config_loads_as_implemented_mode(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        task_config = load_task_config(repo_root / "configs/tasks/center_flash_track.json")

        self.assertEqual(task_config.mode, "center_flash_track")
        self.assertFalse(task_config.behavior.exit_after_fire)
        self.assertTrue(task_config.laser.on_during_run)
        self.assertTrue(task_config.recording.save_raw_video)
        self.assertEqual(task_config.closed_loop.motor_loop_hz, 50.0)
        self.assertEqual(task_config.closed_loop.feedback_timeout, 0.01)
        self.assertTrue(task_config.closed_loop.feedforward.enabled)
        self.assertEqual(task_config.closed_loop.feedforward.lead_time, 0.03)
        self.assertEqual(task_config.closed_loop.feedforward.max_prediction_error, 0.04)
        self.assertEqual(task_config.closed_loop.feedforward.max_velocity, 2.0)
        self.assertEqual(task_config.closed_loop.feedforward.velocity_alpha, 0.35)
        self.assertGreater(task_config.closed_loop.x_pid.output_limit_deg, 0)
        self.assertGreater(task_config.closed_loop.y_pid.output_limit_deg, 0)
        self.assertTrue(task_config.center.target_filter.enabled)
        self.assertEqual(task_config.center.target_filter.ema_alpha, 0.4)
        self.assertEqual(task_config.center.target_filter.max_jump, 0.08)
        self.assertEqual(task_config.center.target_filter.jump_confirm_frames, 2)


if __name__ == "__main__":
    unittest.main()
