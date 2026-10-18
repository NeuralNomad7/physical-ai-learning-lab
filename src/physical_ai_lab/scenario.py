"""Scenario runner: the simulator side of the experiment.

Owns the MuJoCo model and data, seeded resets, hidden physical parameters and
ground truth. Agent-facing code receives `Observation` objects only. It must not
receive the live MjModel/MjData (``sim_model``/``sim_data``), `HiddenPhysics` or
`ScenarioTruth`; those exist for the viewer, renderer, diagnostics and evaluator.
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .config import SceneConfig, load_scene_config
from .observation import Observation, ObservationAdapter

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def _option_values(opt: mujoco.MjOption) -> dict[str, np.ndarray]:
    """Copies of every physics option (the fields of ``MjModel.opt``), by name."""
    names = [name for name in dir(opt) if not name.startswith("_") and not callable(getattr(opt, name))]
    return {name: np.array(getattr(opt, name)) for name in names}


class SimulationUnstable(RuntimeError):
    """Stepping produced NaN/Inf values, MuJoCo warnings or runaway velocities."""


def _print_mujoco_warning(message: str) -> None:
    print(f"MuJoCo warning: {message}", file=sys.stderr, flush=True)


def route_mujoco_warnings_to_stderr() -> None:
    """Send MuJoCo warnings to stderr.

    MuJoCo's default handler prints to stdout and appends to MUJOCO_LOG.TXT in the
    current directory. The handler is process-global.
    """
    mujoco.set_mju_user_warning(_print_mujoco_warning)


@contextmanager
def capture_mujoco_warnings() -> Iterator[list[str]]:
    """Collect MuJoCo warning messages in a list (nothing printed or logged), then restore the handler."""
    previous = mujoco.get_mju_user_warning()
    messages: list[str] = []
    mujoco.set_mju_user_warning(messages.append)
    try:
        yield messages
    finally:
        mujoco.set_mju_user_warning(previous)


@dataclass(frozen=True)
class HiddenPhysics:
    """Hidden physical parameters of one scenario. Simulator truth; never shown to the agent."""

    block_mass: float  # kg
    sliding_friction: float  # block/table Coulomb coefficient (explicit contact pair)

    def __post_init__(self) -> None:
        for name in ("block_mass", "sliding_friction"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive and finite, got {value!r}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ScenarioLayout:
    """Explicit initial state of a scenario (Phase 3 manifests). Simulator truth; never shown as such."""

    block_xy: tuple[float, float]
    block_yaw: float
    target_xy: tuple[float, float]

    def __post_init__(self) -> None:
        bx, by = (float(v) for v in self.block_xy)
        tx, ty = (float(v) for v in self.target_xy)
        yaw = float(self.block_yaw)
        if not all(math.isfinite(v) for v in (bx, by, tx, ty, yaw)):
            raise ValueError("layout values must be finite")
        object.__setattr__(self, "block_xy", (bx, by))
        object.__setattr__(self, "target_xy", (tx, ty))
        object.__setattr__(self, "block_yaw", yaw)

    def check(self, config: SceneConfig) -> None:
        """Raise unless the layout lies inside the reset regions of ``config.workspace``."""
        w = config.workspace
        (bx, by), (tx, ty) = self.block_xy, self.target_xy
        problems = []
        if not (w.block_x.contains(bx) and w.block_y.contains(by)):
            problems.append(f"block ({bx}, {by}) outside the block region")
        if not (w.target_x.contains(tx) and w.target_y.contains(ty)):
            problems.append(f"target ({tx}, {ty}) outside the target region")
        if not w.block_yaw.contains(self.block_yaw):
            problems.append(f"yaw {self.block_yaw} outside the yaw range")
        if not w.block_target_distance.contains(math.hypot(tx - bx, ty - by)):
            problems.append("block/target distance outside the configured range")
        if problems:
            raise ValueError("invalid layout: " + "; ".join(problems))


@dataclass(frozen=True)
class ScenarioTruth:
    """Ground truth of the current scenario, owned by the simulator/evaluator."""

    seed: int
    physics_source: str  # "nominal" | "randomized" | "explicit"
    physics: HiddenPhysics
    block_xy: tuple[float, float]
    block_yaw: float
    target_xy: tuple[float, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StabilityReport:
    ok: bool
    time: float
    finite: bool
    warnings: dict[str, int]
    max_abs_qvel: float
    reason: str


@dataclass(frozen=True)
class ActuatorInfo:
    name: str
    target: str
    kind: str
    kp: float
    kv: float
    ctrlrange: tuple[float, float]
    forcerange: tuple[float, float]


def _check_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return int(seed)


def verify_actuators(model: mujoco.MjModel, cfg: SceneConfig) -> tuple[ActuatorInfo, ...]:
    """Confirm the actuators are what the controls assume before any control is set.

    Arm: one position servo per joint (fixed gain kp, affine bias [0, -kp, -kv]),
    control range inside the joint range. Gripper: tendon actuator whose closed
    command maps to zero tendon length (both fingers at 0).
    """
    problems: list[str] = []
    infos: list[ActuatorInfo] = []
    robot = cfg.robot
    for joint_name, actuator_name in zip(robot.arm_joints, robot.arm_actuators):
        joint, act = model.joint(joint_name), model.actuator(actuator_name)
        i = act.id
        kp, bias = float(model.actuator_gainprm[i, 0]), model.actuator_biasprm[i]
        lo, hi = (float(v) for v in model.actuator_ctrlrange[i])
        jlo, jhi = (float(v) for v in model.jnt_range[joint.id])
        if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT or model.actuator_trnid[i, 0] != joint.id:
            problems.append(f"{actuator_name} does not drive {joint_name}")
        if model.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_NONE:
            problems.append(f"{actuator_name} has activation dynamics")
        if model.actuator_gaintype[i] != mujoco.mjtGain.mjGAIN_FIXED or (
            model.actuator_biastype[i] != mujoco.mjtBias.mjBIAS_AFFINE
        ):
            problems.append(f"{actuator_name} is not a fixed-gain/affine-bias actuator")
        if not (kp > 0 and bias[0] == 0 and math.isclose(bias[1], -kp) and bias[2] <= 0):
            problems.append(f"{actuator_name} is not a position servo (gain={kp}, bias={bias[:3]})")
        if not model.actuator_ctrllimited[i]:
            problems.append(f"{actuator_name} control is not limited")
        if not model.jnt_limited[joint.id] or lo < jlo - 1e-9 or hi > jhi + 1e-9:
            problems.append(f"{actuator_name} ctrlrange [{lo}, {hi}] exceeds joint range [{jlo}, {jhi}]")
        infos.append(
            ActuatorInfo(
                name=actuator_name,
                target=joint_name,
                kind="joint position servo",
                kp=kp,
                kv=float(-bias[2]),
                ctrlrange=(lo, hi),
                forcerange=tuple(float(v) for v in model.actuator_forcerange[i]),  # type: ignore[arg-type]
            )
        )

    act = model.actuator(robot.gripper_actuator)
    i = act.id
    tendon = model.tendon(robot.gripper_tendon)
    gain, bias = float(model.actuator_gainprm[i, 0]), model.actuator_biasprm[i]
    lo, hi = (float(v) for v in model.actuator_ctrlrange[i])
    if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_TENDON or model.actuator_trnid[i, 0] != tendon.id:
        problems.append(f"{robot.gripper_actuator} does not drive tendon {robot.gripper_tendon}")
    if not (bias[0] == 0 and bias[1] < 0 and gain > 0):
        problems.append(f"{robot.gripper_actuator} is not a tendon position servo")
    elif not (lo <= robot.gripper_ctrl_closed <= hi):
        problems.append(f"gripper closed command {robot.gripper_ctrl_closed} outside [{lo}, {hi}]")
    elif abs(gain * robot.gripper_ctrl_closed / -bias[1]) > 1e-9:
        problems.append("gripper closed command does not map to zero finger opening")
    infos.append(
        ActuatorInfo(
            name=robot.gripper_actuator,
            target=robot.gripper_tendon,
            kind="tendon position servo (0 = closed)",
            kp=float(-bias[1]),
            kv=float(-bias[2]),
            ctrlrange=(lo, hi),
            forcerange=tuple(float(v) for v in model.actuator_forcerange[i]),  # type: ignore[arg-type]
        )
    )
    if model.nu != len(robot.arm_actuators) + 1:
        problems.append(f"expected {len(robot.arm_actuators) + 1} actuators, model has {model.nu}")
    if problems:
        raise ValueError("actuator verification failed:\n  " + "\n  ".join(problems))
    return tuple(infos)


def verify_scene(model: mujoco.MjModel, cfg: SceneConfig) -> list[str]:
    """Check the MJCF matches the configuration; return human-readable confirmations."""
    problems: list[str] = []
    notes: list[str] = []
    if not math.isclose(model.opt.timestep, cfg.timing.physics_timestep, rel_tol=0, abs_tol=1e-15):
        problems.append(f"MJCF timestep {model.opt.timestep} != config {cfg.timing.physics_timestep}")
    notes.append(
        f"timestep {model.opt.timestep} s x {cfg.timing.substeps} substeps = {cfg.timing.control_dt:.3f} s "
        f"({cfg.timing.control_hz:g} Hz control)"
    )
    for camera in cfg.cameras:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            problems.append(f"camera {camera!r} missing")

    block_joint = model.joint(cfg.block.joint)
    if model.jnt_type[block_joint.id] != mujoco.mjtJoint.mjJNT_FREE:
        problems.append("block joint is not a free joint")
    block_body = model.body(cfg.block.body).id
    block_geom = model.geom(cfg.block.geom).id
    if model.jnt_bodyid[block_joint.id] != block_body or model.geom_bodyid[block_geom] != block_body:
        problems.append("block joint/geom are not attached to the block body")
    half = cfg.block.edge / 2
    if model.geom_type[block_geom] != mujoco.mjtGeom.mjGEOM_BOX or not np.allclose(model.geom_size[block_geom], half):
        problems.append(f"block geom is not a {cfg.block.edge} m cube")

    table_body = model.body(cfg.table.body).id
    top = model.geom(cfg.table.top_geom).id
    top_z = model.body_pos[table_body][2] + model.geom_pos[top][2] + model.geom_size[top][2]
    if not math.isclose(top_z, cfg.table.top_z, abs_tol=1e-12):
        problems.append(f"tabletop surface at z={top_z}, config says {cfg.table.top_z}")
    top_x = model.body_pos[table_body][0] + model.geom_pos[top][0] + np.array([-1, 1]) * model.geom_size[top][0]
    top_y = model.body_pos[table_body][1] + model.geom_pos[top][1] + np.array([-1, 1]) * model.geom_size[top][1]
    if not (np.allclose(top_x, cfg.table.x.as_list()) and np.allclose(top_y, cfg.table.y.as_list())):
        problems.append(f"tabletop extent x={top_x}, y={top_y} differs from config")

    target_body = model.body(cfg.target.mocap_body).id
    zone = model.geom(cfg.target.zone_geom).id
    if model.body_mocapid[target_body] < 0:
        problems.append("target is not a mocap body")
    if model.geom_contype[zone] or model.geom_conaffinity[zone]:
        problems.append("target zone must not collide")
    if not math.isclose(model.geom_size[zone][0], cfg.workspace.target_tolerance, abs_tol=1e-12):
        problems.append(
            f"target disc radius {model.geom_size[zone][0]} != success tolerance {cfg.workspace.target_tolerance}"
        )
    notes.append(f"target disc radius = success tolerance = {cfg.workspace.target_tolerance} m (centre-based)")

    pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, cfg.physics.contact_pair)
    if pair < 0:
        problems.append(f"contact pair {cfg.physics.contact_pair!r} missing")
    else:
        if {int(model.pair_geom1[pair]), int(model.pair_geom2[pair])} != {block_geom, top}:
            problems.append("contact pair does not join the block and the tabletop")
        if model.pair_dim[pair] != 3:
            problems.append("contact pair must use condim 3 (sliding friction only)")
    signature = (min(table_body, block_body) << 16) + max(table_body, block_body)
    if signature not in set(int(s) for s in model.exclude_signature):
        problems.append("block/table dynamic contacts are not excluded; the pair would be duplicated")
    notes.append("block/table contact generated only by the explicit pair (dynamic contact excluded)")
    fingers = sorted(int(model.jnt_bodyid[model.joint(n).id]) for n in cfg.robot.finger_joints)
    if len(fingers) == 2 and (fingers[0] << 16) + fingers[1] not in set(int(s) for s in model.exclude_signature):
        problems.append("closed finger pads must be excluded from mutual contact")
    if model.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC:
        problems.append("friction cone must be elliptic (pyramidal cones make sliding friction anisotropic)")
    if problems:
        raise ValueError("scene verification failed:\n  " + "\n  ".join(problems))
    return notes


class ScenarioRunner:
    """Seeded resets, hidden physics, headless stepping and stability monitoring."""

    def __init__(self, config: SceneConfig | None = None) -> None:
        self.config = config or load_scene_config()
        cfg = self.config
        self._model = mujoco.MjModel.from_xml_path(str(cfg.mjcf_path))
        self._data = mujoco.MjData(self._model)
        model = self._model
        self.scene_notes = verify_scene(model, cfg)
        self.actuators = verify_actuators(model, cfg)
        self._adapter = ObservationAdapter(model, cfg)

        robot = cfg.robot
        self._arm_qadr = np.array([model.joint(n).qposadr[0] for n in robot.arm_joints], dtype=int)
        self._arm_act = np.array([model.actuator(n).id for n in robot.arm_actuators], dtype=int)
        self._finger_qadr = np.array([model.joint(n).qposadr[0] for n in robot.finger_joints], dtype=int)
        self._gripper_act = model.actuator(robot.gripper_actuator).id
        key = model.key(robot.home_keyframe)
        self.home_arm_qpos = np.array(key.qpos[self._arm_qadr], dtype=float)
        self.home_arm_qpos.setflags(write=False)
        self._block_body = model.body(cfg.block.body).id
        self._block_qadr = int(model.joint(cfg.block.joint).qposadr[0])
        self._target_mocap = int(model.body_mocapid[model.body(cfg.target.mocap_body).id])
        self._pair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, cfg.physics.contact_pair)
        self._warning_names = tuple(mujoco.mjtWarning(i).name for i in range(len(self._data.warning)))
        # mj_setConst (run at every reset) recomputes the model statistics from the geometry. Keep the
        # authored <statistic> centre/extent: they only frame the free camera, clipping planes and shadows.
        self._authored_center = model.stat.center.copy()
        self._authored_extent = float(model.stat.extent)
        # The passive viewer's Physics panel writes its edits into model.opt (timestep, solver, gravity,
        # disable flags, ...), so they would otherwise outlive a reset. Keep the scene's options to put back.
        self._authored_options = _option_values(model.opt)
        self._truth: ScenarioTruth | None = None
        self._control_step = 0

    # ----- simulator-side handles (viewer, renderer, diagnostics; never the agent) -----
    @property
    def sim_model(self) -> mujoco.MjModel:
        return self._model

    @property
    def sim_data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_step(self) -> int:
        return self._control_step

    @property
    def arm_actuator_ids(self) -> np.ndarray:
        """Indices into ``ctrl`` of the seven arm position servos, joint 1 first (read-only)."""
        ids = self._arm_act.copy()
        ids.setflags(write=False)
        return ids

    @property
    def gripper_actuator_id(self) -> int:
        return int(self._gripper_act)

    @property
    def ctrl_range(self) -> np.ndarray:
        """Control limits as an (nu, 2) array of [low, high]; -inf/inf for actuators without limits."""
        model = self._model
        limited = model.actuator_ctrllimited.astype(bool)[:, None]
        return np.where(limited, model.actuator_ctrlrange, np.array([-np.inf, np.inf]))

    @property
    def nominal_physics(self) -> HiddenPhysics:
        p = self.config.physics
        return HiddenPhysics(block_mass=p.nominal_mass, sliding_friction=p.nominal_sliding_friction)

    def hidden_truth(self) -> ScenarioTruth:
        """Ground truth for logging and evaluation only."""
        self._require_reset()
        assert self._truth is not None
        return self._truth

    # ----- reset -----
    def sample_layout(self, rng: np.random.Generator) -> tuple[tuple[float, float], float, tuple[float, float]]:
        w = self.config.workspace
        for _ in range(w.max_reset_attempts):
            bx, by = rng.uniform(w.block_x.low, w.block_x.high), rng.uniform(w.block_y.low, w.block_y.high)
            tx, ty = rng.uniform(w.target_x.low, w.target_x.high), rng.uniform(w.target_y.low, w.target_y.high)
            if w.block_target_distance.contains(math.hypot(tx - bx, ty - by)):
                yaw = rng.uniform(w.block_yaw.low, w.block_yaw.high)
                return (float(bx), float(by)), float(yaw), (float(tx), float(ty))
        raise RuntimeError(f"no valid block/target layout after {w.max_reset_attempts} attempts")

    def sample_physics(self, rng: np.random.Generator) -> HiddenPhysics:
        p = self.config.physics
        return HiddenPhysics(
            block_mass=float(rng.uniform(p.mass_range.low, p.mass_range.high)),
            sliding_friction=float(rng.uniform(p.sliding_friction_range.low, p.sliding_friction_range.high)),
        )

    def _apply_physics(self, physics: HiddenPhysics) -> None:
        model, cfg = self._model, self.config
        edge = cfg.block.edge
        model.body_mass[self._block_body] = physics.block_mass
        model.body_inertia[self._block_body] = physics.block_mass * edge * edge / 6.0  # solid cube
        mu = physics.sliding_friction
        p = cfg.physics
        model.pair_friction[self._pair] = [mu, mu, p.torsional_friction, p.rolling_friction, p.rolling_friction]

    def restore_options(self) -> tuple[str, ...]:
        """Put back every physics option (``model.opt``) that differs from the scene's; return their names.

        Values are compared bit for bit. ``reset`` calls this first; the viewer also calls it
        before each step, because its Physics panel edits the model's options.
        """
        opt = self._model.opt
        changed = []
        for name, authored in self._authored_options.items():
            current = getattr(opt, name)
            if np.asarray(current).tobytes() == authored.tobytes():
                continue
            if authored.ndim:
                current[...] = authored
            else:
                setattr(opt, name, authored.item())
            changed.append(name)
        return tuple(changed)

    def reset(
        self,
        seed: int,
        *,
        randomize_physics: bool = False,
        physics: HiddenPhysics | None = None,
        layout: ScenarioLayout | None = None,
    ) -> Observation:
        """Reset to the scenario determined by ``seed`` and return the first observation.

        The seed spawns two independent streams: one for the block/target layout and
        one for hidden physics, so a seed gives the same layout whether or not physics
        is randomized. The state is rebuilt from scratch and the scene's physics options
        are restored (``restore_options``), so it does not depend on earlier episodes.

        ``layout`` (Phase 3 manifests) replaces the seeded layout with an explicit one; it
        must lie inside the configured reset regions. Without it the seeded path is unchanged.
        """
        seed = _check_seed(seed)
        if randomize_physics and physics is not None:
            raise ValueError("pass either randomize_physics=True or explicit physics, not both")
        layout_seq, physics_seq = np.random.SeedSequence(seed).spawn(2)
        if layout is None:
            block_xy, block_yaw, target_xy = self.sample_layout(np.random.default_rng(layout_seq))
        else:
            layout.check(self.config)
            block_xy, block_yaw, target_xy = layout.block_xy, layout.block_yaw, layout.target_xy
        if physics is not None:
            source = "explicit"
        elif randomize_physics:
            physics, source = self.sample_physics(np.random.default_rng(physics_seq)), "randomized"
        else:
            physics, source = self.nominal_physics, "nominal"

        model, data, cfg = self._model, self._data, self.config
        self.restore_options()
        self._apply_physics(physics)
        mujoco.mj_setConst(model, data)  # recompute mass-dependent constants for the new block mass
        model.stat.center[:] = self._authored_center
        model.stat.extent = self._authored_extent
        mujoco.mj_resetData(model, data)
        data.qpos[self._arm_qadr] = self.home_arm_qpos
        data.qpos[self._finger_qadr] = 0.0
        data.ctrl[self._arm_act] = self.home_arm_qpos
        data.ctrl[self._gripper_act] = cfg.robot.gripper_ctrl_closed
        qa = self._block_qadr
        data.qpos[qa : qa + 3] = (block_xy[0], block_xy[1], cfg.table.top_z + cfg.block.edge / 2)
        data.qpos[qa + 3 : qa + 7] = (math.cos(block_yaw / 2), 0.0, 0.0, math.sin(block_yaw / 2))
        data.mocap_pos[self._target_mocap] = (target_xy[0], target_xy[1], cfg.table.top_z)
        data.mocap_quat[self._target_mocap] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)

        self._truth = ScenarioTruth(
            seed=seed,
            physics_source=source,
            physics=physics,
            block_xy=block_xy,
            block_yaw=block_yaw,
            target_xy=target_xy,
        )
        self._control_step = 0
        return self.observe()

    # ----- actuation and stepping -----
    def set_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        """Set every actuator input, clamped to its control range, and return the values applied.

        Only the actuator inputs change: the next ``step`` moves the robot through its servos
        and the contact dynamics, so no body is moved directly.
        """
        self._require_reset()
        values = np.asarray(ctrl, dtype=float)
        if values.shape != (self._model.nu,):
            raise ValueError(f"ctrl must have shape ({self._model.nu},), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("ctrl must be finite")
        bounds = self.ctrl_range
        applied = np.clip(values, bounds[:, 0], bounds[:, 1])
        self._data.ctrl[:] = applied
        return applied

    def step(
        self,
        control_updates: int = 1,
        *,
        monitor: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
    ) -> Observation:
        """Advance ``control_updates`` x ``substeps`` physics steps holding the current controls.

        ``monitor`` (evaluator side only) is called after every physics step with the live
        model and data. It must only read them; it never changes the dynamics.
        """
        self._require_reset()
        if control_updates < 0:
            raise ValueError("control_updates must be non-negative")
        model, data = self._model, self._data
        substeps = self.config.timing.substeps
        for _ in range(control_updates):
            for _ in range(substeps):
                mujoco.mj_step(model, data)
                if monitor is not None:
                    monitor(model, data)
            self._control_step += 1
            report = self.stability()
            if not report.ok:
                raise SimulationUnstable(f"t={report.time:.3f} s: {report.reason}")
        mujoco.mj_forward(model, data)  # derived quantities consistent with the new state
        return self.observe()

    def observe(self) -> Observation:
        self._require_reset()
        return self._adapter.build(self._model, self._data, self._control_step)

    def stability(self) -> StabilityReport:
        data = self._data
        finite = bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
            and np.isfinite(data.ctrl).all()
        )
        warnings = {
            name: int(data.warning[i].number) for i, name in enumerate(self._warning_names) if data.warning[i].number
        }
        max_abs_qvel = float(np.max(np.abs(data.qvel))) if finite else math.inf
        reasons = []
        if not finite:
            reasons.append("non-finite state")
        if warnings:
            reasons.append(f"MuJoCo warnings {warnings}")
        if max_abs_qvel > self.config.stability.max_abs_qvel:
            reasons.append(f"|qvel| {max_abs_qvel:.3g} exceeds {self.config.stability.max_abs_qvel}")
        return StabilityReport(
            ok=not reasons,
            time=float(data.time),
            finite=finite,
            warnings=warnings,
            max_abs_qvel=max_abs_qvel,
            reason="; ".join(reasons) or "stable",
        )

    # ----- reproducibility helpers -----
    def state_vector(self) -> np.ndarray:
        """Full integration state (time, qpos, qvel, act, warmstart, ctrl, applied forces, mocap)."""
        state = np.empty(mujoco.mj_stateSize(self._model, STATE_SPEC))
        mujoco.mj_getState(self._model, self._data, state, STATE_SPEC)
        return state

    def state_digest(self) -> str:
        return hashlib.sha256(self.state_vector().astype("<f8").tobytes()).hexdigest()[:16]

    def applied_physics(self) -> HiddenPhysics:
        """Read the hidden parameters back from the compiled model (diagnostics)."""
        return HiddenPhysics(
            block_mass=float(self._model.body_mass[self._block_body]),
            sliding_friction=float(self._model.pair_friction[self._pair, 0]),
        )

    def _require_reset(self) -> None:
        if self._truth is None:
            raise RuntimeError("call reset(seed) before stepping or observing")
