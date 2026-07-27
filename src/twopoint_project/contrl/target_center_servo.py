"""视觉目标居中控制。

这个模块负责把识别结果里的 ``target_center`` 点转换成云台运动命令。
输入点坐标使用归一化图像坐标：左上角为 (0, 0)，右下角为 (1, 1)，
默认目标是把点移动到画面中心 (0.5, 0.5)。

这里的 PID 是视觉外环 PID：它输出每次循环云台应移动的角度增量。
电机驱动器内部的角度环、速度环 PID 仍然在更底层负责执行这个角度命令。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot, isfinite
import time
from typing import Protocol, Sequence, TypedDict


TARGET_CENTER_LABEL = "target_center"


class PointPrediction(TypedDict):
    """单个视觉检测点。

    x/y 是归一化图像坐标，不是像素坐标。
    """

    label: str
    x: float
    y: float
    confidence: float


class GimbalLike(Protocol):
    """云台对象只需要提供相对角度移动接口。"""

    def move_by(self, x_delta_deg: float, y_delta_deg: float) -> None:
        ...


@dataclass(frozen=True)
class TargetCenterObservation:
    """选中的 target_center 检测点。"""

    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class FeedForwardConfig:
    """视觉前馈预测配置。

    enabled=True 时，根据 target_center 的图像坐标速度预测 lead_time 秒后的
    目标位置，再把预测位置送入 PID。坐标、速度都使用归一化图像坐标。
    """

    enabled: bool = False
    lead_time: float = 0.0
    max_prediction_error: float = 0.05
    max_velocity: float = 2.0
    velocity_alpha: float = 0.5


@dataclass(frozen=True)
class CenteringError:
    """目标点相对期望中心点的误差。"""

    x: float
    y: float
    distance: float


@dataclass(frozen=True)
class GimbalStep:
    """一次控制循环计算出的云台运动量。"""

    x_delta_deg: float
    y_delta_deg: float
    error: CenteringError
    settled: bool
    x_output: PIDOutput | None = None
    y_output: PIDOutput | None = None
    feedforward_x: float = 0.0
    feedforward_y: float = 0.0
    target_velocity_x: float = 0.0
    target_velocity_y: float = 0.0


@dataclass(frozen=True)
class PIDAxisGains:
    """单轴视觉 PID 参数。

    kp/ki/kd 使用图像归一化误差作为输入，输出单位是度。
    output_limit_deg 限制单次循环最大角度增量，防止视觉误差或噪声
    直接造成过大的云台动作。

    integral_separation_threshold > 0 时，只有误差进入该范围才累积积分；
    derivative_separation_threshold > 0 时，只有误差进入该范围才启用微分。
    fuzzy_enabled=True 时，只按误差大小自动缩放 KP：远处加强 P，
    近处降低 P。KI/KD 仍使用配置值，方便独立判断积分和微分效果。
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 0.0
    output_limit_deg: float = 1.0
    integral_separation_threshold: float = 0.0
    derivative_separation_threshold: float = 0.0
    fuzzy_enabled: bool = False
    fuzzy_error_low: float = 0.02
    fuzzy_error_high: float = 0.18
    fuzzy_kp_near_scale: float = 0.65
    fuzzy_kp_far_scale: float = 1.25


@dataclass(frozen=True)
class PIDOutput:
    """单次 PID 计算结果，便于日志或调试拆分 P/I/D 贡献。"""

    p: float
    i: float
    d: float
    raw: float
    clamped: float
    effective_kp: float
    effective_ki: float
    effective_kd: float
    derivative: float


@dataclass(frozen=True)
class PIDEffectiveGains:
    """模糊调节后的本轮有效 PID 参数。"""

    kp: float
    ki: float
    kd: float


@dataclass
class PIDAxisState:
    """单轴 PID 运行状态。

    integral 和 previous_error 必须跨帧保留，否则 I/D 项无法工作。
    当目标进入死区或目标丢失时，上层会重置状态，避免历史误差继续影响
    下一次有效控制。
    """

    integral: float = 0.0
    previous_error: float | None = None
    previous_time: float | None = None

    def reset(self) -> None:
        """清空积分和微分历史。"""
        self.integral = 0.0
        self.previous_error = None
        self.previous_time = None

    def update(self, error: float, gains: PIDAxisGains, now: float) -> PIDOutput:
        """根据当前误差计算一轴的 PID 输出。"""
        dt = 0.0 if self.previous_time is None else max(now - self.previous_time, 0.0)
        integral_allowed = (
            gains.integral_separation_threshold <= 0
            or abs(error) <= abs(gains.integral_separation_threshold)
        )
        if gains.integral_limit > 0 and dt > 0 and integral_allowed:
            # 积分项按时间累积误差，并用 integral_limit 防止长时间偏差
            # 把输出推得过大。
            self.integral = clamp(
                self.integral + error * dt,
                -abs(gains.integral_limit),
                abs(gains.integral_limit),
            )
        elif not integral_allowed:
            # 积分分离：误差较大时不让积分继续累积，避免回到中心附近后
            # 历史积分继续推着云台过冲。
            self.integral = 0.0
        elif gains.integral_limit <= 0:
            # integral_limit <= 0 表示关闭积分记忆。
            self.integral = 0.0

        derivative = 0.0
        if self.previous_error is not None and dt > 0:
            # 微分项看误差变化速度，通常用于给接近目标时增加阻尼。
            derivative = (error - self.previous_error) / dt
        derivative_allowed = (
            gains.derivative_separation_threshold <= 0
            or abs(error) <= abs(gains.derivative_separation_threshold)
        )
        derivative_for_output = derivative if derivative_allowed else 0.0
        effective_gains = resolve_effective_gains(error, gains)

        p = effective_gains.kp * error
        i = effective_gains.ki * self.integral
        d = effective_gains.kd * derivative_for_output
        raw = p + i + d
        output_limit = abs(gains.output_limit_deg)
        # raw 是理论输出；clamped 是真正会下发给云台的单次角度增量。
        clamped = clamp(raw, -output_limit, output_limit) if output_limit > 0 else raw

        self.previous_error = error
        self.previous_time = now
        return PIDOutput(
            p=p,
            i=i,
            d=d,
            raw=raw,
            clamped=clamped,
            effective_kp=effective_gains.kp,
            effective_ki=effective_gains.ki,
            effective_kd=effective_gains.kd,
            derivative=derivative_for_output,
        )


@dataclass(frozen=True)
class AimUpdate:
    """一次视觉控制更新的结果。"""

    valid: bool
    moved: bool
    settled: bool
    reason: str | None
    target: TargetCenterObservation | None
    step: GimbalStep | None


def select_target_center(
    points: Sequence[PointPrediction],
) -> TargetCenterObservation | None:
    """选择置信度最高的有效 target_center，不再应用外部点置信度阈值。"""
    candidates = [
        point
        for point in points
        if point["label"] == TARGET_CENTER_LABEL
        and isfinite(float(point["x"]))
        and isfinite(float(point["y"]))
        and isfinite(float(point["confidence"]))
    ]
    if not candidates:
        return None

    point = max(candidates, key=lambda item: item["confidence"])
    return TargetCenterObservation(
        x=float(point["x"]),
        y=float(point["y"]),
        confidence=float(point["confidence"]),
    )


def clamp(value: float, min_value: float, max_value: float) -> float:
    """把 value 限制在 [min_value, max_value] 范围内。"""
    return min(max(value, min_value), max_value)


def lerp(start: float, end: float, ratio: float) -> float:
    """线性插值。"""
    return start + (end - start) * ratio


def smooth_ratio(value: float, low: float, high: float) -> float:
    """把误差大小平滑映射到 0~1，供模糊 PID 使用。"""
    low = abs(low)
    high = abs(high)
    if high <= low:
        return 1.0 if abs(value) >= high else 0.0
    ratio = clamp((abs(value) - low) / (high - low), 0.0, 1.0)
    return ratio * ratio * (3.0 - 2.0 * ratio)


def resolve_effective_gains(
    error: float,
    gains: PIDAxisGains,
) -> PIDEffectiveGains:
    """根据误差大小得到本轮实际使用的 PID 参数。"""
    if not gains.fuzzy_enabled:
        return PIDEffectiveGains(kp=gains.kp, ki=gains.ki, kd=gains.kd)

    far_ratio = smooth_ratio(error, gains.fuzzy_error_low, gains.fuzzy_error_high)
    kp_scale = lerp(gains.fuzzy_kp_near_scale, gains.fuzzy_kp_far_scale, far_ratio)

    return PIDEffectiveGains(
        kp=gains.kp * kp_scale,
        ki=gains.ki,
        kd=gains.kd,
    )


class TargetCenterServo:
    """把视觉目标偏差转换成云台相对角度移动。

    参数中的 center_x/center_y 是期望目标落在画面中的位置，默认是中心。
    x_pid/y_pid 不传时，会退回到旧版的单纯比例控制参数
    x_gain_deg/y_gain_deg/max_step_deg。
    """

    def __init__(
        self,
        *,
        center_x: float = 0.5,
        center_y: float = 0.5,
        x_gain_deg: float = 8.0,
        y_gain_deg: float = -8.0,
        max_step_deg: float = 1.0,
        deadband: float = 0.006,
        x_pid: PIDAxisGains | None = None,
        y_pid: PIDAxisGains | None = None,
        feedforward: FeedForwardConfig | None = None,
    ) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.deadband = abs(deadband)
        self.x_pid = x_pid or PIDAxisGains(kp=x_gain_deg, output_limit_deg=max_step_deg)
        self.y_pid = y_pid or PIDAxisGains(kp=y_gain_deg, output_limit_deg=max_step_deg)
        self.feedforward = feedforward or FeedForwardConfig()
        self._x_state = PIDAxisState()
        self._y_state = PIDAxisState()
        self._previous_target: TargetCenterObservation | None = None
        self._previous_target_time: float | None = None
        self._target_velocity_x = 0.0
        self._target_velocity_y = 0.0

    def compute_error(self, target: TargetCenterObservation) -> CenteringError:
        """计算目标点到期望中心的归一化坐标误差。"""
        x_error = target.x - self.center_x
        y_error = target.y - self.center_y
        return CenteringError(
            x=x_error,
            y=y_error,
            distance=hypot(x_error, y_error),
        )

    def reset_feedforward(self) -> None:
        """清空目标运动预测状态。"""
        self._previous_target = None
        self._previous_target_time = None
        self._target_velocity_x = 0.0
        self._target_velocity_y = 0.0

    def predict_target(
        self,
        target: TargetCenterObservation,
        *,
        now: float,
    ) -> tuple[TargetCenterObservation, float, float]:
        """用目标点运动速度预测前馈目标位置。"""
        config = self.feedforward
        if not config.enabled or config.lead_time <= 0:
            return target, 0.0, 0.0

        previous = self._previous_target
        previous_time = self._previous_target_time
        velocity_x = self._target_velocity_x
        velocity_y = self._target_velocity_y
        dt = 0.0 if previous_time is None else max(now - previous_time, 0.0)

        if previous is not None and dt > 0:
            raw_velocity_x = (target.x - previous.x) / dt
            raw_velocity_y = (target.y - previous.y) / dt
            max_velocity = abs(config.max_velocity)
            if max_velocity > 0:
                raw_velocity_x = clamp(raw_velocity_x, -max_velocity, max_velocity)
                raw_velocity_y = clamp(raw_velocity_y, -max_velocity, max_velocity)
            alpha = clamp(config.velocity_alpha, 0.0, 1.0)
            velocity_x = lerp(velocity_x, raw_velocity_x, alpha)
            velocity_y = lerp(velocity_y, raw_velocity_y, alpha)

        max_prediction = abs(config.max_prediction_error)
        feedforward_x = velocity_x * config.lead_time
        feedforward_y = velocity_y * config.lead_time
        if max_prediction > 0:
            feedforward_x = clamp(feedforward_x, -max_prediction, max_prediction)
            feedforward_y = clamp(feedforward_y, -max_prediction, max_prediction)

        self._target_velocity_x = velocity_x
        self._target_velocity_y = velocity_y
        predicted = TargetCenterObservation(
            x=clamp(target.x + feedforward_x, 0.0, 1.0),
            y=clamp(target.y + feedforward_y, 0.0, 1.0),
            confidence=target.confidence,
        )
        return predicted, predicted.x - target.x, predicted.y - target.y

    def compute_step(self, target: TargetCenterObservation, *, now: float | None = None) -> GimbalStep:
        """根据当前目标点计算云台本轮应该移动的角度。"""
        now = time.monotonic() if now is None else now
        control_target, feedforward_x, feedforward_y = self.predict_target(
            target,
            now=now,
        )
        self._previous_target = target
        self._previous_target_time = now

        error = self.compute_error(control_target)
        x_settled = abs(error.x) <= self.deadband
        y_settled = abs(error.y) <= self.deadband
        settled = x_settled and y_settled

        if settled:
            # 已进入死区时不再移动，并重置 PID 历史，避免积分或微分
            # 在下一次目标离开死区时制造突跳。
            self._x_state.reset()
            self._y_state.reset()
            return GimbalStep(
                x_delta_deg=0.0,
                y_delta_deg=0.0,
                error=error,
                settled=True,
                x_output=None,
                y_output=None,
                feedforward_x=feedforward_x,
                feedforward_y=feedforward_y,
                target_velocity_x=self._target_velocity_x,
                target_velocity_y=self._target_velocity_y,
            )

        if x_settled:
            # 单轴已进入死区时，只停止该轴，另一轴仍可继续修正。
            self._x_state.reset()
            x_output = None
            x_delta = 0.0
        else:
            x_output = self._x_state.update(error.x, self.x_pid, now)
            x_delta = x_output.clamped

        if y_settled:
            self._y_state.reset()
            y_output = None
            y_delta = 0.0
        else:
            y_output = self._y_state.update(error.y, self.y_pid, now)
            y_delta = y_output.clamped

        return GimbalStep(
            x_delta_deg=x_delta,
            y_delta_deg=y_delta,
            error=error,
            settled=settled,
            x_output=x_output,
            y_output=y_output,
            feedforward_x=feedforward_x,
            feedforward_y=feedforward_y,
            target_velocity_x=self._target_velocity_x,
            target_velocity_y=self._target_velocity_y,
        )

    def update(
        self,
        gimbal: GimbalLike,
        points: Sequence[PointPrediction],
        *,
        step_scale: float = 1.0,
    ) -> AimUpdate:
        """从检测点完成一次闭环更新，并在需要时移动云台。"""
        if step_scale < 0:
            raise ValueError("step_scale must be non-negative")
        target = select_target_center(points)
        if target is None:
            self.reset_feedforward()
            return AimUpdate(
                valid=False,
                moved=False,
                settled=False,
                reason="target_center_unavailable",
                target=None,
                step=None,
            )

        step = self.compute_step(target)
        if step.settled:
            return AimUpdate(
                valid=True,
                moved=False,
                settled=True,
                reason=None,
                target=target,
                step=step,
            )

        if step_scale != 1.0:
            # 旧帧复用时可以缩小或关闭动作，避免用过期目标继续追。
            step = replace(
                step,
                x_delta_deg=step.x_delta_deg * step_scale,
                y_delta_deg=step.y_delta_deg * step_scale,
            )
        moved = step.x_delta_deg != 0.0 or step.y_delta_deg != 0.0
        if moved:
            gimbal.move_by(step.x_delta_deg, step.y_delta_deg)
        return AimUpdate(
            valid=True,
            moved=moved,
            settled=False,
            reason=None,
            target=target,
            step=step,
        )


def sleep_for_loop_rate(loop_started_at: float, loop_hz: float) -> None:
    """按目标控制频率补足睡眠时间。"""
    if loop_hz <= 0:
        return
    min_interval = 1.0 / loop_hz
    elapsed = time.monotonic() - loop_started_at
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
