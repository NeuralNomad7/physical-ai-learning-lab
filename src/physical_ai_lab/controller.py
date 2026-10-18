"""Push skill for the Panda: bounded push primitive, planning filters and execution (Phase 2).

A push primitive (``PushAction``) names the block face to contact (approach side), the
push direction in the table plane, the pusher speed and the push duration. The stroke
length is ``speed * duration``. ``PushSkill`` turns an action into Cartesian TCP
setpoints and executes them only through the arm's position servos:

1. rise to the transit height (if below it) and move above the pre-contact point,
2. descend to the push height a small standoff behind the chosen face,
3. stroke along the push direction (trapezoidal speed profile, cruise = action speed),
4. dwell, back off along the push direction, rise to the transit height.

Every control update solves damped least-squares IK on the controller's own
kinematics-only model (``ArmKinematics``), clamps the joint target inside the joint
ranges (minus a margin) and limits its change to the controller's joint-speed limit.
That kinematic command (``q_cmd``) is sent to the servos with the static gravity
offset of the arm added (``ArmKinematics.gravity_offset``), so the arm comes to rest at
the commanded joints instead of sagging under its own weight. Without the offset the
TCP ran 4.1-8.4 mm below its setpoints during the stroke, so the Cartesian bounds below
held only for commands, not for the motion (review of Phase 2, D25). The gripper stays
closed. The block is never moved directly.

Before an action is used, ``plan`` filters it: Cartesian setpoints inside the table
outline and above the minimum TCP height, the predicted block end inside the allowed
region, the push direction within ``max_off_normal`` of the face normal, and IK
reachability along the whole path (transit, descent, stroke, back-off and rise).

The skill only reads ``Observation`` fields. It never reads hidden physics and never
rolls out the simulation: the predicted block end is pure geometry (block follows the
pusher after contact).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from .config import SceneConfig
from .kinematics import ArmKinematics, IKResult, wrap_angle
from .observation import Observation
from .push_config import PushConfig

YAW_SPEED = 1.5  # rad/s, hand yaw rate limit for transit moves
CHECK_STEP = 0.02  # m, spacing of IK reachability samples along a path
CHECK_YAW_STEP = 0.25  # rad, spacing of IK samples for yaw changes


class EpisodeOver(Exception):
    """Raised by a plant when the evaluator has ended the episode (success or failure)."""


class PushAborted(Exception):
    """The skill could not follow its own plan (IK or bounds) and stopped the push."""


class Plant(Protocol):
    """What the controller may touch: observations in, actuator targets out."""

    control_dt: float

    def observe(self) -> Observation: ...

    def command(self, arm_q: np.ndarray, gripper: float) -> Observation:
        """Apply the actuator targets for one control update and return the next observation."""
        ...

    def note(self, event: str, **info: object) -> None:
        """Annotate the episode log (phase changes, push start/end); no effect on the simulation."""
        ...


@dataclass(frozen=True)
class PushAction:
    face: int  # 0..3: face whose outward normal is at block_yaw + face * pi/2
    direction: float  # rad, push direction in the table plane (world frame)
    speed: float  # m/s, cruise speed of the pusher during the stroke
    duration: float  # s, stroke length / speed

    @property
    def stroke(self) -> float:
        return self.speed * self.duration

    def to_dict(self) -> dict:
        return {
            "face": self.face,
            "direction_deg": math.degrees(self.direction),
            "speed": self.speed,
            "duration": self.duration,
            "stroke": self.stroke,
        }


@dataclass
class PushPlan:
    action: PushAction
    feasible: bool
    reason: str
    hand_yaw: float = math.nan
    off_normal: float = math.nan  # rad, push direction minus inward face normal
    pre_contact: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    stroke_end: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    retreat: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    predicted_displacement: float = math.nan
    predicted_block_end: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))
    min_joint_margin: float = math.nan

    def summary(self) -> dict:
        return {
            "action": self.action.to_dict(),
            "feasible": self.feasible,
            "reason": self.reason,
            "hand_yaw_deg": math.degrees(self.hand_yaw) if math.isfinite(self.hand_yaw) else None,
            "off_normal_deg": math.degrees(self.off_normal) if math.isfinite(self.off_normal) else None,
            "pre_contact": self.pre_contact.tolist(),
            "stroke_end": self.stroke_end.tolist(),
            "predicted_displacement": self.predicted_displacement,
            "predicted_block_end": self.predicted_block_end.tolist(),
            "min_joint_margin": self.min_joint_margin,
        }


@dataclass(frozen=True)
class Segment:
    phase: str
    start: np.ndarray
    end: np.ndarray
    yaw_start: float
    yaw_end: float  # unwrapped: yaw_start + signed change
    duration: float
    ramp: float

    def setpoint(self, t: float) -> tuple[np.ndarray, float]:
        u = unit_profile(t, self.duration, self.ramp)
        return self.start + u * (self.end - self.start), self.yaw_start + u * (self.yaw_end - self.yaw_start)


def unit_profile(t: float, total: float, ramp: float) -> float:
    """Trapezoidal-velocity progress in [0, 1] over ``total`` seconds with ``ramp`` s ramps."""
    if total <= 0 or t >= total:
        return 1.0
    if t <= 0:
        return 0.0
    ramp = min(ramp, total / 2)
    peak = 1.0 / (total - ramp)
    if t < ramp:
        return 0.5 * peak / ramp * t * t
    if t <= total - ramp:
        return peak * (t - ramp / 2)
    rest = total - t
    return 1.0 - 0.5 * peak / ramp * rest * rest


def move_time(distance: float, speed: float, ramp: float) -> float:
    """Duration of a trapezoidal move with cruise ``speed`` (triangular when short)."""
    if distance <= 0:
        return 0.0
    if distance >= speed * ramp:
        return distance / speed + ramp
    return 2.0 * math.sqrt(distance * ramp / speed)


@dataclass
class ExecutionStats:
    control_updates: int = 0
    rate_limited_updates: int = 0
    max_ik_position_error: float = 0.0
    max_ik_orientation_error: float = 0.0
    min_commanded_tcp_z: float = math.inf

    def to_dict(self) -> dict:
        return asdict(self)


class PushSkill:
    """Plans, filters and executes push primitives through the arm servos."""

    def __init__(self, scene: SceneConfig, push: PushConfig, kinematics: ArmKinematics | None = None) -> None:
        self.scene, self.push, self.ctl = scene, push, push.controller
        self.kin = kinematics or ArmKinematics(scene)
        self.half_edge = scene.block.edge / 2
        self.gripper_closed = scene.robot.gripper_ctrl_closed
        self.q_cmd: np.ndarray | None = None
        self.stats = ExecutionStats()

    # ----- geometry -----
    def rest_posture(self, xy: np.ndarray) -> np.ndarray:
        rest = self.kin.home_q.copy()
        rest[0] = math.atan2(float(xy[1]), float(xy[0]))
        return rest

    def inward_normal(self, block_yaw: float, face: int) -> float:
        """Angle of the direction pointing from ``face`` into the block."""
        return wrap_angle(block_yaw + face * math.pi / 2 + math.pi)

    def ik(self, target: np.ndarray, yaw: float, q_init: np.ndarray, *, iterations: int = 60) -> IKResult:
        return self.kin.solve(
            target,
            yaw,
            q_init,
            self.rest_posture(target),
            joint_margin=self.ctl.joint_margin,
            position_tolerance=self.ctl.ik_position_tolerance,
            orientation_tolerance=self.ctl.ik_orientation_tolerance,
            max_iterations=iterations,
        )

    def in_table(self, p: np.ndarray) -> bool:
        t, m = self.scene.table, self.ctl.table_margin
        return t.x.low + m <= p[0] <= t.x.high - m and t.y.low + m <= p[1] <= t.y.high - m

    def block_in_region(self, xy: np.ndarray) -> bool:
        w, m = self.scene.workspace, self.ctl.block_margin
        return (
            w.allowed_block_x.low + m <= xy[0] <= w.allowed_block_x.high - m
            and w.allowed_block_y.low + m <= xy[1] <= w.allowed_block_y.high - m
        )

    # ----- planning -----
    def start_state(self, obs: Observation) -> tuple[np.ndarray, np.ndarray, float]:
        """Commanded joints, their TCP and hand yaw (the controller plans from its own commands).

        Before the first command of an episode the servo targets are still the reset pose,
        which carries no gravity offset, so ``obs.arm_qpos_target`` is a kinematic command then.
        """
        q = np.array(obs.arm_qpos_target if self.q_cmd is None else self.q_cmd, dtype=float)
        tcp, _ = self.kin.forward(q)
        return q, tcp, self.kin.hand_yaw(q)

    def plan(self, obs: Observation, action: PushAction) -> PushPlan:
        """Geometry of ``action`` from the current observation, with all filters applied."""
        ctl = self.ctl
        if action.face not in (0, 1, 2, 3):
            return PushPlan(action, False, "invalid face")
        if action.speed not in self.push.action.speeds or action.duration not in self.push.action.durations:
            return PushPlan(action, False, "speed/duration not admissible")
        block = np.asarray(obs.block_pos[:2], dtype=float)
        normal = self.inward_normal(obs.block_yaw, action.face)
        off = wrap_angle(action.direction - normal)
        if abs(off) > ctl.max_off_normal + 1e-9:
            return PushPlan(action, False, f"direction {math.degrees(off):+.1f} deg off the face normal")
        d = np.array([math.cos(action.direction), math.sin(action.direction)])
        cos_off = math.cos(off)
        s_contact = -(self.half_edge + ctl.pusher_half_thickness) / cos_off
        s_pre = s_contact - ctl.standoff / cos_off
        s_end = s_pre + action.stroke
        z = ctl.push_height
        pre = np.array([*(block + s_pre * d), z])
        end = np.array([*(block + s_end * d), z])
        retreat = np.array([*(block + (s_end - ctl.retreat_distance) * d), z])
        displacement = max(0.0, action.stroke - ctl.standoff / cos_off)
        block_end = block + displacement * d
        plan = PushPlan(
            action,
            False,
            "",
            off_normal=off,
            pre_contact=pre,
            stroke_end=end,
            retreat=retreat,
            predicted_displacement=displacement,
            predicted_block_end=block_end,
        )
        for name, point in (("pre-contact", pre), ("stroke end", end), ("retreat", retreat)):
            if not self.in_table(point):
                plan.reason = f"{name} TCP outside table bounds"
                return plan
        if not self.block_in_region(block_end):
            plan.reason = "predicted block end outside the allowed region"
            return plan
        q0, tcp0, yaw0 = self.start_state(obs)
        best: tuple[float, float, float] | None = None  # (score, yaw, margin)
        failures = []
        for yaw in (normal, wrap_angle(normal + math.pi)):  # closed fingers are symmetric about hand z
            ok, margin, why = self._reachable(q0, tcp0, yaw0, yaw, pre, end)
            if not ok:
                failures.append(why)
                continue
            score = abs(wrap_angle(yaw - yaw0))
            if best is None or score < best[0]:
                best = (score, yaw, margin)
        if best is None:
            plan.reason = "unreachable: " + "; ".join(failures)
            return plan
        plan.hand_yaw, plan.min_joint_margin = best[1], best[2]
        plan.feasible, plan.reason = True, "ok"
        return plan

    def _reachable(
        self, q0: np.ndarray, tcp0: np.ndarray, yaw0: float, yaw: float, pre: np.ndarray, end: np.ndarray
    ) -> tuple[bool, float, str]:
        """IK along the planned path, warm-started sample to sample (as during execution)."""
        q, margin = q0.copy(), math.inf
        for phase, segment in self._segments(tcp0, yaw0, yaw, pre, end, end, check_only=True):
            span = float(np.linalg.norm(segment.end - segment.start))
            turn = abs(segment.yaw_end - segment.yaw_start)
            n = max(1, math.ceil(span / CHECK_STEP), math.ceil(turn / CHECK_YAW_STEP))
            for i in range(1, n + 1):
                u = i / n
                point = segment.start + u * (segment.end - segment.start)
                result = self.ik(point, segment.yaw_start + u * (segment.yaw_end - segment.yaw_start), q)
                if not result.converged:
                    return False, margin, f"yaw {math.degrees(yaw):+.0f} deg: IK fails in {phase}"
                q, margin = result.q, min(margin, result.joint_margin)
        return True, margin, ""

    def _segments(
        self,
        tcp0: np.ndarray,
        yaw0: float,
        yaw: float,
        pre: np.ndarray,
        end: np.ndarray,
        stroke_end: np.ndarray,
        *,
        speed: float | None = None,
        check_only: bool = False,
    ) -> Iterator[tuple[str, Segment]]:
        """Approach, descent, stroke, back-off and rise as straight Cartesian segments."""
        ctl = self.ctl
        safe = ctl.safe_height
        yaw_end = yaw0 + wrap_angle(yaw - yaw0)
        above_pre = np.array([pre[0], pre[1], safe])
        start = tcp0.copy()
        if start[2] < safe - 1e-6:
            up = np.array([start[0], start[1], safe])
            yield "rise", self._segment("rise", start, up, yaw0, yaw0, ctl.descend_speed)
            start = up
        yield "approach", self._segment("approach", start, above_pre, yaw0, yaw_end, ctl.transit_speed)
        yield "descend", self._segment("descend", above_pre, pre, yaw_end, yaw_end, ctl.descend_speed)
        if check_only:
            yield "stroke", self._segment("stroke", pre, end, yaw_end, yaw_end, 1.0)
        else:
            assert speed is not None
            stroke_time = float(np.linalg.norm(end - pre)) / speed + ctl.ramp_seconds
            yield "stroke", Segment("stroke", pre, end, yaw_end, yaw_end, stroke_time, ctl.ramp_seconds)
        d = (end - pre) / max(float(np.linalg.norm(end - pre)), 1e-9)
        back = stroke_end - ctl.retreat_distance * d
        yield "back-off", self._segment("back-off", stroke_end, back, yaw_end, yaw_end, ctl.descend_speed)
        yield "retreat", self._segment("retreat", back, np.array([back[0], back[1], safe]), yaw_end, yaw_end,
                                       ctl.descend_speed)

    def _segment(self, phase: str, a: np.ndarray, b: np.ndarray, yaw_a: float, yaw_b: float, speed: float) -> Segment:
        ramp = self.ctl.ramp_seconds
        duration = max(
            move_time(float(np.linalg.norm(b - a)), speed, ramp),
            move_time(abs(yaw_b - yaw_a), YAW_SPEED, ramp),
        )
        return Segment(phase, a.copy(), b.copy(), yaw_a, yaw_b, duration, ramp)

    # ----- execution -----
    def servo_target(self, q: np.ndarray) -> np.ndarray:
        """Servo targets that hold the arm at joint angles ``q`` (gravity offset added)."""
        kin = self.kin
        return np.clip(q + kin.gravity_offset(q), kin.joint_lo, kin.joint_hi)

    def _send(self, plant: Plant, q_target: np.ndarray) -> Observation:
        """Joint-limit and joint-speed-limit the target, then apply it for one control update.

        The limits apply to the kinematic command ``q_cmd``; the servos receive
        ``servo_target(q_cmd)``, which differs only by the static gravity offset.
        """
        assert self.q_cmd is not None
        kin, ctl = self.kin, self.ctl
        lo, hi = kin.joint_lo + ctl.joint_margin, kin.joint_hi - ctl.joint_margin
        step = np.asarray(ctl.max_joint_speed) * plant.control_dt
        wanted = np.clip(q_target, lo, hi)
        delta = np.clip(wanted - self.q_cmd, -step, step)
        if np.any(np.abs(wanted - self.q_cmd) > step + 1e-12):
            self.stats.rate_limited_updates += 1
        self.q_cmd = np.clip(self.q_cmd + delta, lo, hi)
        self.stats.control_updates += 1
        return plant.command(self.servo_target(self.q_cmd), self.gripper_closed)

    def _follow(self, plant: Plant, segment: Segment, q_ik: np.ndarray) -> np.ndarray:
        dt = plant.control_dt
        steps = max(1, math.ceil(segment.duration / dt - 1e-9))
        plant.note("phase", phase=segment.phase)
        for i in range(1, steps + 1):
            point, yaw = segment.setpoint(i * dt)
            if point[2] < self.ctl.min_tcp_z - 1e-9 or not self.in_table(point):
                raise PushAborted(f"{segment.phase}: setpoint {np.round(point, 4).tolist()} outside bounds")
            result = self.ik(point, yaw, q_ik, iterations=40)
            if not result.converged:
                raise PushAborted(
                    f"{segment.phase}: IK did not converge (error {result.position_error * 1e3:.2f} mm)"
                )
            q_ik = result.q
            self.stats.max_ik_position_error = max(self.stats.max_ik_position_error, result.position_error)
            self.stats.max_ik_orientation_error = max(self.stats.max_ik_orientation_error, result.orientation_error)
            self.stats.min_commanded_tcp_z = min(self.stats.min_commanded_tcp_z, float(point[2]))
            self._send(plant, q_ik)
        return q_ik

    def hold(self, plant: Plant, seconds: float, until: Callable[[Observation], bool] | None = None) -> Observation:
        """Hold the current joint targets for up to ``seconds`` (or until ``until(obs)``)."""
        obs = plant.observe()
        if self.q_cmd is None:
            self.q_cmd = np.array(obs.arm_qpos_target, dtype=float)
        for _ in range(max(0, round(seconds / plant.control_dt))):
            if until is not None and until(obs):
                break
            obs = self._send(plant, self.q_cmd)
        return obs

    def execute(self, plant: Plant, obs: Observation, plan: PushPlan) -> Observation:
        """Run approach, contact stroke, back-off and retreat for a feasible plan."""
        if not plan.feasible:
            raise ValueError(f"plan is not feasible: {plan.reason}")
        if self.q_cmd is None:
            self.q_cmd = np.array(obs.arm_qpos_target, dtype=float)
        q_ik, tcp0, yaw0 = self.start_state(obs)
        segments = list(
            self._segments(tcp0, yaw0, plan.hand_yaw, plan.pre_contact, plan.stroke_end, plan.stroke_end,
                           speed=plan.action.speed)
        )
        for phase, segment in segments:
            q_ik = self._follow(plant, segment, q_ik)
            if phase == "stroke":
                plant.note("phase", phase="dwell")
                self.hold(plant, self.ctl.dwell_seconds)
        return plant.observe()

    def recover(self, plant: Plant) -> Observation:
        """After an aborted push: rise straight up to the transit height from the commanded pose."""
        obs = plant.observe()
        q, tcp, yaw = self.start_state(obs)
        if tcp[2] < self.ctl.safe_height:
            up = np.array([tcp[0], tcp[1], self.ctl.safe_height])
            self._follow(plant, self._segment("recover", tcp, up, yaw, yaw, self.ctl.descend_speed), q)
        return plant.observe()
