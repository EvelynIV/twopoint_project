from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.center_fakes import (
    FakeCapture,
    FakeGimbal,
    FakeInferencer,
    FakeLaser,
    make_constant_laser_track_config,
    make_runtime_config,
    make_task_config,
    make_track_config,
)
from twopoint_project.tasks import center_flash_track, center_then_flash
from twopoint_project.tasks import resources as task_resources


def fake_resources() -> SimpleNamespace:
    monitor = SimpleNamespace(
        enabled=False,
        start=Mock(),
        on_frame=Mock(),
        stop_requested=lambda: False,
    )
    return SimpleNamespace(
        inferencer=SimpleNamespace(providers=["fake"]),
        gimbal=FakeGimbal(),
        laser=FakeLaser(),
        vision=Mock(),
        monitor=monitor,
    )


class TaskFlowTest(unittest.TestCase):
    def test_center_then_flash_owns_center_and_fire_flow(self) -> None:
        resources = fake_resources()
        with patch.object(
            center_then_flash,
            "open_task_resources",
            return_value=nullcontext(resources),
        ), patch.object(
            center_then_flash,
            "run_target_centering_loop",
            return_value=True,
        ) as center_loop:
            result = center_then_flash.run(make_task_config(), make_runtime_config())

        self.assertTrue(result)
        center_loop.assert_called_once()
        self.assertTrue(resources.gimbal.initialized)
        self.assertIn("on", resources.laser.events)
        self.assertEqual(resources.laser.events[-1], "off")

    def test_resource_context_cleans_up_after_task_error(self) -> None:
        capture = FakeCapture()
        gimbal = FakeGimbal()
        laser = FakeLaser()

        class ContextResource(SimpleNamespace):
            def __enter__(self) -> ContextResource:
                return self

            def __exit__(self, *args: object) -> None:
                self.closed = True

            def close(self) -> None:
                self.closed = True

        vision = ContextResource(closed=False)
        monitor = ContextResource(
            enabled=False,
            closed=False,
            on_frame=Mock(),
            stop_requested=lambda: False,
        )
        with patch.object(
            task_resources,
            "build_vision_inferencer",
            return_value=FakeInferencer(),
        ), patch.object(
            task_resources,
            "open_camera_capture",
            return_value=capture,
        ), patch.object(
            task_resources,
            "open_serial_gimbal",
            return_value=gimbal,
        ), patch.object(
            task_resources,
            "VisionProducer",
            return_value=vision,
        ), patch.object(
            task_resources,
            "open_laser_pointer",
            return_value=laser,
        ), patch.object(
            task_resources,
            "CenterRunMonitor",
            return_value=monitor,
        ):
            with self.assertRaisesRegex(RuntimeError, "task failed"):
                with task_resources.open_task_resources(
                    make_task_config(),
                    make_runtime_config(),
                ):
                    laser.on()
                    raise RuntimeError("task failed")

        self.assertEqual(laser.events[-1], "off")
        self.assertTrue(gimbal.disabled)
        self.assertTrue(capture.closed)
        self.assertTrue(vision.closed)
        self.assertTrue(monitor.closed)

    def test_track_task_owns_center_fire_and_track_flow(self) -> None:
        resources = fake_resources()
        with patch.object(
            center_flash_track,
            "open_task_resources",
            return_value=nullcontext(resources),
        ), patch.object(
            center_flash_track,
            "run_laser_alignment_loop",
            side_effect=[True, False],
        ) as alignment_loop:
            result = center_flash_track.run(
                make_track_config(),
                make_runtime_config(),
                stop_requested=lambda: False,
            )

        self.assertTrue(result)
        self.assertEqual(alignment_loop.call_count, 2)
        self.assertIn("on", resources.laser.events)
        self.assertEqual(resources.laser.events[-1], "off")

    def test_track_task_runs_one_continuous_loop_when_laser_stays_on(self) -> None:
        resources = fake_resources()
        events: list[str] = []
        resources.gimbal.initialize = Mock(side_effect=lambda: events.append("motor_enable"))
        resources.gimbal.move_to = Mock(
            side_effect=lambda x, y: events.append(f"move_to({x},{y})")
        )
        resources.monitor.start = Mock(side_effect=lambda: events.append("monitor_start"))
        resources.laser.on = Mock(side_effect=lambda: events.append("laser_on"))
        with patch.object(
            center_flash_track,
            "open_task_resources",
            return_value=nullcontext(resources),
        ), patch.object(
            center_flash_track,
            "run_laser_alignment_loop",
            side_effect=lambda **_: events.append("tracking") or False,
        ) as alignment_loop:
            result = center_flash_track.run(
                make_constant_laser_track_config(),
                make_runtime_config(),
                stop_requested=lambda: False,
            )

        self.assertTrue(result)
        alignment_loop.assert_called_once()
        call = alignment_loop.call_args.kwargs
        self.assertEqual(call["phase"], "track")
        self.assertNotIn("deadline", call)
        self.assertNotIn("stable_frames", call)
        self.assertEqual(
            events,
            [
                "motor_enable",
                "move_to(0.0,0.0)",
                "monitor_start",
                "laser_on",
                "tracking",
            ],
        )


if __name__ == "__main__":
    unittest.main()
