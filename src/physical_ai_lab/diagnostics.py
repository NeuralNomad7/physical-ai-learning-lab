"""Simulator-side scene diagnostics. Never imported by agent code.

* Reach envelope: kinematic damped-least-squares IK puts the closed-gripper TCP
  on a grid over the block/target regions (expanded by a margin), pointing
  straight down. It checks feasibility only; it is not the Phase 2 controller.
* Hidden physics: block/table contact friction, normal force and an effective
  friction "kick test" (initial block velocity v0, slide distance d,
  mu_eff = v0^2 / (2 g d)).
* Hold stability: the arm holds its home pose and the block stays at rest.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from .config import SceneConfig
from .observation import ObservationAdapter
from .scenario import HiddenPhysics, ScenarioRunner

DOWN = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class _Indices:
    arm_qadr: np.ndarray
    arm_dadr: np.ndarray
    finger_qadr: np.ndarray
    joint_lo: np.ndarray
    joint_hi: np.ndarray
    hand: int
    tcp_offset: np.ndarray
    block_qadr: int
    block_dadr: int
    block_geom: int
    table_top_geom: int


def _indices(model: mujoco.MjModel, cfg: SceneConfig) -> _Indices:
    robot = cfg.robot
    joints = [model.joint(n) for n in robot.arm_joints]
    block_joint = model.joint(cfg.block.joint)
    return _Indices(
        arm_qadr=np.array([j.qposadr[0] for j in joints], dtype=int),
        arm_dadr=np.array([j.dofadr[0] for j in joints], dtype=int),
        finger_qadr=np.array([model.joint(n).qposadr[0] for n in robot.finger_joints], dtype=int),
        joint_lo=np.array([model.jnt_range[j.id][0] for j in joints]),
        joint_hi=np.array([model.jnt_range[j.id][1] for j in joints]),
        hand=model.body(robot.tcp_body).id,
        tcp_offset=np.array(robot.tcp_offset, dtype=float),
        block_qadr=int(block_joint.qposadr[0]),
        block_dadr=int(block_joint.dofadr[0]),
        block_geom=model.geom(cfg.block.geom).id,
        table_top_geom=model.geom(cfg.table.top_geom).id,
    )


# --------------------------------------------------------------------------- reach


@dataclass(frozen=True)
class ReachPoint:
    target: tuple[float, float, float]
    converged: bool
    position_error: float  # m
    orientation_error: float  # rad between hand z-axis and straight down
    joint_margin: float  # rad, smallest distance of any arm joint to its limit
    robot_table_contact: bool
    robot_self_contact: bool
    iterations: int
    ok: bool


@dataclass(frozen=True)
class ReachReport:
    points: tuple[ReachPoint, ...]
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    heights: tuple[float, ...]
    min_joint_margin_required: float

    @property
    def ok(self) -> bool:
        return all(p.ok for p in self.points)

    @property
    def failures(self) -> tuple[ReachPoint, ...]:
        return tuple(p for p in self.points if not p.ok)

    def summary(self) -> str:
        reached = sum(p.ok for p in self.points)
        return (
            f"{reached}/{len(self.points)} TCP-down poses reachable over x {self.x_range[0]:.2f}..{self.x_range[1]:.2f}, "
            f"y {self.y_range[0]:.2f}..{self.y_range[1]:.2f} m at heights {', '.join(f'{h:.3f}' for h in self.heights)} m; "
            f"worst position error {max(p.position_error for p in self.points) * 1e3:.3f} mm, "
            f"worst orientation error {max(p.orientation_error for p in self.points):.4f} rad, "
            f"min joint margin {min(p.joint_margin for p in self.points):.3f} rad "
            f"(required {self.min_joint_margin_required:.3f}), "
            f"robot/table contacts {sum(p.robot_table_contact for p in self.points)}, "
            f"self contacts {sum(p.robot_self_contact for p in self.points)}"
        )


def solve_tcp_down_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    idx: _Indices,
    target: np.ndarray,
    q_rest: np.ndarray,
    *,
    position_tolerance: float,
    orientation_tolerance: float,
    damping: float = 0.02,
    max_step: float = 0.2,
    max_iterations: int = 500,
    nullspace_gain: float = 0.05,
) -> tuple[np.ndarray, float, float, int, bool]:
    """Kinematic DLS IK: TCP position + hand z-axis down, yaw free, posture bias in the nullspace."""
    q = q_rest.copy()
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    eye7, eye6 = np.eye(len(q)), np.eye(6)
    position_error = orientation_error = math.inf
    for iteration in range(max_iterations + 1):
        data.qpos[idx.arm_qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        rotation = data.xmat[idx.hand].reshape(3, 3)
        tcp = data.xpos[idx.hand] + rotation @ idx.tcp_offset
        z_axis = rotation[:, 2]
        e_pos = target - tcp
        e_ori = np.cross(z_axis, DOWN)
        position_error = float(np.linalg.norm(e_pos))
        orientation_error = float(math.acos(max(-1.0, min(1.0, float(z_axis @ DOWN)))))
        if position_error <= position_tolerance and orientation_error <= orientation_tolerance:
            return q, position_error, orientation_error, iteration, True
        if iteration == max_iterations:
            break
        mujoco.mj_jac(model, data, jacp, jacr, tcp, idx.hand)
        projector = np.eye(3) - np.outer(z_axis, z_axis)  # rotation about the hand axis is free
        jac = np.vstack([jacp[:, idx.arm_dadr], projector @ jacr[:, idx.arm_dadr]])
        pinv = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * eye6, eye6)
        dq = pinv @ np.concatenate([e_pos, e_ori]) + (eye7 - pinv @ jac) @ (nullspace_gain * (q_rest - q))
        largest = float(np.max(np.abs(dq)))
        if largest > max_step:
            dq *= max_step / largest
        q = np.clip(q + dq, idx.joint_lo, idx.joint_hi)
    return q, position_error, orientation_error, max_iterations, False


def check_reach(runner: ScenarioRunner) -> ReachReport:
    """Validate the reach envelope over the block/target regions plus a margin."""
    cfg = runner.config
    rc, w = cfg.reach_check, cfg.workspace
    model = runner.sim_model
    data = mujoco.MjData(model)  # private copy: the runner's state is untouched
    idx = _indices(model, cfg)
    adapter = ObservationAdapter(model, cfg)
    robot_mask = adapter.robot_geom_mask

    x_lo = min(w.block_x.low, w.target_x.low) - rc.margin
    x_hi = max(w.block_x.high, w.target_x.high) + rc.margin
    y_lo = min(w.block_y.low, w.target_y.low) - rc.margin
    y_hi = max(w.block_y.high, w.target_y.high) + rc.margin
    points = []
    for z in rc.heights:
        for x in np.linspace(x_lo, x_hi, rc.grid):
            for y in np.linspace(y_lo, y_hi, rc.grid):
                target = np.array([x, y, cfg.table.top_z + z])
                q_rest = np.array(runner.home_arm_qpos)
                q_rest[0] = math.atan2(y, x)
                mujoco.mj_resetData(model, data)
                data.qpos[idx.block_qadr : idx.block_qadr + 3] = (10.0, 10.0, 1.0)  # block out of the way
                q, pos_err, ori_err, iterations, converged = solve_tcp_down_ik(
                    model,
                    data,
                    idx,
                    target,
                    q_rest,
                    position_tolerance=rc.position_tolerance,
                    orientation_tolerance=rc.orientation_tolerance,
                )
                data.qpos[idx.arm_qadr] = q
                data.qpos[idx.finger_qadr] = 0.0
                mujoco.mj_forward(model, data)
                margin = float(np.min(np.minimum(q - idx.joint_lo, idx.joint_hi - q)))
                _, table_contact = adapter.contact_flags(data)
                self_contact = False
                if data.ncon:
                    geom = np.asarray(data.contact.geom)
                    active = (np.asarray(data.contact.exclude) == 0) & (np.asarray(data.contact.dist) <= 0.0)
                    self_contact = bool((active & robot_mask[geom[:, 0]] & robot_mask[geom[:, 1]]).any())
                points.append(
                    ReachPoint(
                        target=(float(x), float(y), float(target[2])),
                        converged=converged,
                        position_error=pos_err,
                        orientation_error=ori_err,
                        joint_margin=margin,
                        robot_table_contact=table_contact,
                        robot_self_contact=self_contact,
                        iterations=iterations,
                        ok=converged and margin >= rc.min_joint_margin and not table_contact and not self_contact,
                    )
                )
    return ReachReport(
        points=tuple(points),
        x_range=(x_lo, x_hi),
        y_range=(y_lo, y_hi),
        heights=rc.heights,
        min_joint_margin_required=rc.min_joint_margin,
    )


# --------------------------------------------------------------------------- hidden physics


@dataclass(frozen=True)
class ContactSummary:
    count: int
    normal_force: float  # N, sum over block/table contacts
    frictions: tuple[float, ...]  # sliding coefficient of each block/table contact
    other_block_contacts: int


def block_table_contacts(runner: ScenarioRunner) -> ContactSummary:
    model, data = runner.sim_model, runner.sim_data
    idx = _indices(model, runner.config)
    pair = {idx.block_geom, idx.table_top_geom}
    force = np.zeros(6)
    normal, frictions, other = 0.0, [], 0
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.exclude != 0:
            continue
        geoms = {int(contact.geom[0]), int(contact.geom[1])}
        if geoms == pair:
            mujoco.mj_contactForce(model, data, i, force)
            normal += float(force[0])
            frictions.append(float(contact.friction[0]))
        elif idx.block_geom in geoms:
            other += 1
    return ContactSummary(len(frictions), normal, tuple(frictions), other)


@dataclass(frozen=True)
class KickResult:
    physics: HiddenPhysics
    direction_deg: float
    v0: float
    distance: float  # m, planar slide distance
    mu_effective: float
    relative_error: float  # (mu_eff - mu) / mu
    stop_time: float  # s, first control step with planar speed < 1e-3 m/s
    final_speed: float
    yaw_change: float  # rad
    normal_force: float  # N, at rest before the kick
    expected_normal_force: float  # m * g
    contact_frictions: tuple[float, ...]
    inertia: tuple[float, float, float]


def kick_test(
    runner: ScenarioRunner,
    physics: HiddenPhysics,
    *,
    seed: int = 0,
    direction_deg: float = 0.0,
    v0: float = 0.5,
    settle_seconds: float = 0.5,
    max_seconds: float = 1.5,
) -> KickResult:
    """Give the resting block a horizontal velocity and measure how far it slides.

    For Coulomb friction the slide distance is d = v0^2 / (2 mu g), so
    mu_eff = v0^2 / (2 g d). Diagnostic only: the task never sets block velocities.
    """
    model, data = runner.sim_model, runner.sim_data
    idx = _indices(model, runner.config)
    dt = runner.config.timing.control_dt
    gravity = float(np.linalg.norm(model.opt.gravity))

    runner.reset(seed, physics=physics)
    start_obs = runner.step(round(settle_seconds / dt))
    contacts = block_table_contacts(runner)
    inertia = tuple(float(v) for v in model.body_inertia[model.body(runner.config.block.body).id])

    angle = math.radians(direction_deg)
    data.qvel[idx.block_dadr : idx.block_dadr + 6] = (v0 * math.cos(angle), v0 * math.sin(angle), 0.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    stop_time = math.nan
    obs = start_obs
    for update in range(1, round(max_seconds / dt) + 1):
        obs = runner.step(1)
        if math.isnan(stop_time) and obs.block_planar_speed < 1e-3:
            stop_time = update * dt
    distance = float(np.hypot(*(obs.block_pos[:2] - start_obs.block_pos[:2])))
    mu_eff = v0 * v0 / (2.0 * gravity * distance) if distance > 0 else math.inf
    yaw_change = math.remainder(obs.block_yaw - start_obs.block_yaw, math.tau)
    return KickResult(
        physics=physics,
        direction_deg=direction_deg,
        v0=v0,
        distance=distance,
        mu_effective=mu_eff,
        relative_error=(mu_eff - physics.sliding_friction) / physics.sliding_friction,
        stop_time=stop_time,
        final_speed=obs.block_planar_speed,
        yaw_change=yaw_change,
        normal_force=contacts.normal_force,
        expected_normal_force=physics.block_mass * gravity,
        contact_frictions=contacts.frictions,
        inertia=inertia,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- hold stability


@dataclass(frozen=True)
class HoldReport:
    seed: int
    physics_source: str
    seconds: float
    stable: bool
    reason: str
    max_abs_qvel: float  # over the whole hold
    final_max_abs_qvel: float
    block_planar_drift: float  # m
    block_height_change: float  # m (settling into the soft contact)
    arm_max_deviation: float  # rad from the home pose (gravity sag of the position servos)
    robot_contacts: bool
    wall_seconds: float

    @property
    def realtime_factor(self) -> float:
        return self.seconds / self.wall_seconds if self.wall_seconds > 0 else math.inf


def hold_test(runner: ScenarioRunner, seed: int, *, seconds: float = 5.0, randomize_physics: bool = False) -> HoldReport:
    """Hold the home pose with the closed gripper and check that nothing moves or blows up."""
    first = runner.reset(seed, randomize_physics=randomize_physics)
    updates = round(seconds / runner.config.timing.control_dt)
    max_qvel, contacts, reason, stable = 0.0, False, "stable", True
    obs = first
    started = time.perf_counter()
    try:
        for _ in range(updates):
            obs = runner.step(1)
            max_qvel = max(max_qvel, runner.stability().max_abs_qvel)
            contacts = contacts or obs.robot_block_contact or obs.robot_table_contact
    except Exception as exc:  # SimulationUnstable or MuJoCo errors are reported, not hidden
        stable, reason = False, f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - started
    final = runner.stability()
    return HoldReport(
        seed=seed,
        physics_source=runner.hidden_truth().physics_source,
        seconds=seconds,
        stable=stable and final.ok,
        reason=reason if not stable else final.reason,
        max_abs_qvel=max_qvel,
        final_max_abs_qvel=final.max_abs_qvel,
        block_planar_drift=float(np.hypot(*(obs.block_pos[:2] - first.block_pos[:2]))),
        block_height_change=float(obs.block_pos[2] - first.block_pos[2]),
        arm_max_deviation=float(np.max(np.abs(obs.arm_qpos - runner.home_arm_qpos))),
        robot_contacts=contacts,
        wall_seconds=wall,
    )


# --------------------------------------------------------------------------- actuator response


@dataclass(frozen=True)
class ActuatorResponse:
    arm_step: float  # rad added to every arm joint target from the home pose
    seconds: float  # simulated time allowed per phase
    arm_errors: tuple[float, ...]  # rad, |q - target| per arm joint at the end of the arm phase
    gripper_open_command: float
    gripper_open_opening: float  # m, finger opening after commanding fully open
    gripper_open_expected: float  # m, both fingers at their upper joint limits
    gripper_closed_opening: float  # m, after commanding closed again
    max_abs_qvel: float
    ok: bool
    reason: str

    @property
    def arm_max_error(self) -> float:
        return max(self.arm_errors)


def actuator_response_test(
    runner: ScenarioRunner,
    *,
    seed: int = 0,
    arm_step: float = 0.1,
    seconds: float = 1.0,
    arm_tolerance: float = 0.02,
    gripper_tolerance: float = 0.002,
) -> ActuatorResponse:
    """Command a joint-target step on every arm joint, then open and close the gripper.

    Confirms by observed motion that each servo drives the intended joint with the
    expected sign and that gripper command 0 means closed. It is not a controller.
    """
    cfg = runner.config
    model, data = runner.sim_model, runner.sim_data
    robot = cfg.robot
    arm_act = np.array([model.actuator(n).id for n in robot.arm_actuators], dtype=int)
    gripper = model.actuator(robot.gripper_actuator).id
    finger_hi = sum(float(model.jnt_range[model.joint(n).id][1]) for n in robot.finger_joints)
    updates = round(seconds / cfg.timing.control_dt)

    runner.reset(seed)
    target = np.array(runner.home_arm_qpos) + arm_step
    data.ctrl[arm_act] = target
    max_qvel = 0.0
    obs = None
    for _ in range(updates):
        obs = runner.step(1)
        max_qvel = max(max_qvel, runner.stability().max_abs_qvel)
    assert obs is not None
    arm_errors = tuple(float(e) for e in np.abs(obs.arm_qpos - target))

    open_command = float(model.actuator_ctrlrange[gripper][1])
    data.ctrl[gripper] = open_command
    for _ in range(updates):
        obs = runner.step(1)
        max_qvel = max(max_qvel, runner.stability().max_abs_qvel)
    opened = obs.gripper_opening

    data.ctrl[gripper] = robot.gripper_ctrl_closed
    for _ in range(updates):
        obs = runner.step(1)
        max_qvel = max(max_qvel, runner.stability().max_abs_qvel)
    closed = obs.gripper_opening

    reasons = []
    if max(arm_errors) > arm_tolerance:
        reasons.append(f"arm tracking error {max(arm_errors):.4f} rad > {arm_tolerance}")
    if abs(opened - finger_hi) > gripper_tolerance:
        reasons.append(f"open gripper {opened * 1e3:.2f} mm, expected {finger_hi * 1e3:.2f} mm")
    if abs(closed) > gripper_tolerance:
        reasons.append(f"closed gripper still open {closed * 1e3:.2f} mm")
    return ActuatorResponse(
        arm_step=arm_step,
        seconds=seconds,
        arm_errors=arm_errors,
        gripper_open_command=open_command,
        gripper_open_opening=opened,
        gripper_open_expected=finger_hi,
        gripper_closed_opening=closed,
        max_abs_qvel=max_qvel,
        ok=not reasons,
        reason="; ".join(reasons) or "servos track their targets",
    )
