"""Typed loader for ``configs/scene.toml``."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import paths


@dataclass(frozen=True)
class Range:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high) and self.low <= self.high):
            raise ValueError(f"invalid range [{self.low}, {self.high}]")

    @classmethod
    def parse(cls, value: object) -> "Range":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"expected [low, high], got {value!r}")
        return cls(float(value[0]), float(value[1]))

    def contains(self, value: float, tol: float = 0.0) -> bool:
        return self.low - tol <= value <= self.high + tol

    def as_list(self) -> list[float]:
        return [self.low, self.high]


@dataclass(frozen=True)
class TimingConfig:
    physics_timestep: float
    control_hz: float
    substeps: int

    @property
    def control_dt(self) -> float:
        return self.physics_timestep * self.substeps


@dataclass(frozen=True)
class RobotConfig:
    home_keyframe: str
    arm_joints: tuple[str, ...]
    arm_actuators: tuple[str, ...]
    finger_joints: tuple[str, ...]
    gripper_actuator: str
    gripper_tendon: str
    gripper_ctrl_closed: float
    tcp_body: str
    tcp_offset: tuple[float, float, float]


@dataclass(frozen=True)
class BlockConfig:
    body: str
    geom: str
    joint: str
    edge: float


@dataclass(frozen=True)
class TargetConfig:
    mocap_body: str
    zone_geom: str


@dataclass(frozen=True)
class TableConfig:
    body: str
    top_geom: str
    top_z: float
    x: Range
    y: Range


@dataclass(frozen=True)
class WorkspaceConfig:
    block_x: Range
    block_y: Range
    block_yaw: Range
    target_x: Range
    target_y: Range
    block_target_distance: Range
    max_reset_attempts: int
    target_tolerance: float
    allowed_block_x: Range
    allowed_block_y: Range


@dataclass(frozen=True)
class PhysicsConfig:
    contact_pair: str
    nominal_mass: float
    nominal_sliding_friction: float
    mass_range: Range
    sliding_friction_range: Range
    torsional_friction: float
    rolling_friction: float


@dataclass(frozen=True)
class StabilityConfig:
    max_abs_qvel: float


@dataclass(frozen=True)
class ReachCheckConfig:
    margin: float
    grid: int
    heights: tuple[float, ...]
    min_joint_margin: float
    position_tolerance: float
    orientation_tolerance: float


@dataclass(frozen=True)
class ChecksConfig:
    """Phase 1 acceptance tolerances (smoke test and pytest)."""

    reset_state_tolerance: float
    rollout_seconds: float
    layout_seeds: int
    hold_seconds: float
    hold_seeds: int
    long_hold_seconds: float
    hold_max_final_qvel: float
    hold_max_block_drift: float
    hold_max_arm_deviation: float
    kick_speed: float
    kick_mu_rel_tolerance: float
    normal_force_rel_tolerance: float
    actuator_step: float
    actuator_seconds: float
    actuator_tracking_tolerance: float
    gripper_tolerance: float
    observation_rest_tolerance: float
    visibility_seeds: int
    max_arm_occlusion: float


@dataclass(frozen=True)
class SceneConfig:
    source: Path
    mjcf_path: Path
    cameras: tuple[str, ...]
    timing: TimingConfig
    robot: RobotConfig
    block: BlockConfig
    target: TargetConfig
    table: TableConfig
    workspace: WorkspaceConfig
    physics: PhysicsConfig
    stability: StabilityConfig
    reach_check: ReachCheckConfig
    checks: ChecksConfig


def _validate(cfg: SceneConfig) -> None:
    t = cfg.timing
    if t.substeps < 1 or t.physics_timestep <= 0 or t.control_hz <= 0:
        raise ValueError("timing values must be positive")
    if not math.isclose(t.physics_timestep * t.substeps, 1.0 / t.control_hz, rel_tol=0, abs_tol=1e-12):
        raise ValueError(
            f"timing mismatch: {t.substeps} x {t.physics_timestep} s != 1/{t.control_hz} Hz"
        )
    if len(cfg.robot.arm_joints) != len(cfg.robot.arm_actuators):
        raise ValueError("arm_joints and arm_actuators must have the same length")
    if len(cfg.robot.tcp_offset) != 3:
        raise ValueError("tcp_offset must have 3 components")
    p = cfg.physics
    if not (p.mass_range.low > 0 and p.sliding_friction_range.low > 0):
        raise ValueError("mass and friction ranges must be positive")
    if not (p.mass_range.contains(p.nominal_mass) and p.sliding_friction_range.contains(p.nominal_sliding_friction)):
        raise ValueError("nominal physics must lie inside the development ranges")
    w = cfg.workspace
    if w.target_tolerance <= 0 or w.max_reset_attempts < 1:
        raise ValueError("target_tolerance and max_reset_attempts must be positive")
    for region, allowed in ((w.block_x, w.allowed_block_x), (w.block_y, w.allowed_block_y)):
        if not (allowed.contains(region.low) and allowed.contains(region.high)):
            raise ValueError("block reset region must lie inside the allowed block region")
    for region, table in (
        (w.allowed_block_x, cfg.table.x),
        (w.allowed_block_y, cfg.table.y),
        (w.target_x, cfg.table.x),
        (w.target_y, cfg.table.y),
    ):
        if not (table.contains(region.low) and table.contains(region.high)):
            raise ValueError("workspace regions must lie on the table")
    if cfg.block.edge <= 0:
        raise ValueError("block edge must be positive")
    c = cfg.checks
    if c.reset_state_tolerance < 0:
        raise ValueError("reset_state_tolerance must be non-negative")
    positive = {
        name: getattr(c, name)
        for name in (
            "rollout_seconds", "layout_seeds", "hold_seconds", "hold_seeds", "long_hold_seconds",
            "hold_max_final_qvel", "hold_max_block_drift", "hold_max_arm_deviation", "kick_speed",
            "kick_mu_rel_tolerance", "normal_force_rel_tolerance", "actuator_step", "actuator_seconds",
            "actuator_tracking_tolerance", "gripper_tolerance", "observation_rest_tolerance",
            "visibility_seeds", "max_arm_occlusion",
        )
    }
    not_positive = [name for name, value in positive.items() if not value > 0]
    if not_positive:
        raise ValueError(f"checks values must be positive: {not_positive}")
    if not c.max_arm_occlusion < 1:
        raise ValueError("max_arm_occlusion is a fraction of the object's area and must be below 1")


def load_scene_config(path: Path | str = paths.SCENE_CONFIG) -> SceneConfig:
    source = Path(path).resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported scene config schema_version {raw.get('schema_version')!r}")

    scene, timing, robot = raw["scene"], raw["timing"], raw["robot"]
    block, target, table = raw["block"], raw["target"], raw["table"]
    workspace, physics, stability, reach = raw["workspace"], raw["physics"], raw["stability"], raw["reach_check"]
    checks = raw["checks"]

    mjcf = Path(scene["mjcf"])
    cfg = SceneConfig(
        source=source,
        mjcf_path=(mjcf if mjcf.is_absolute() else paths.PROJECT_ROOT / mjcf).resolve(),
        cameras=tuple(scene["cameras"]),
        timing=TimingConfig(
            physics_timestep=float(timing["physics_timestep"]),
            control_hz=float(timing["control_hz"]),
            substeps=int(timing["substeps"]),
        ),
        robot=RobotConfig(
            home_keyframe=robot["home_keyframe"],
            arm_joints=tuple(robot["arm_joints"]),
            arm_actuators=tuple(robot["arm_actuators"]),
            finger_joints=tuple(robot["finger_joints"]),
            gripper_actuator=robot["gripper_actuator"],
            gripper_tendon=robot["gripper_tendon"],
            gripper_ctrl_closed=float(robot["gripper_ctrl_closed"]),
            tcp_body=robot["tcp_body"],
            tcp_offset=tuple(float(v) for v in robot["tcp_offset"]),  # type: ignore[arg-type]
        ),
        block=BlockConfig(
            body=block["body"], geom=block["geom"], joint=block["joint"], edge=float(block["edge"])
        ),
        target=TargetConfig(mocap_body=target["mocap_body"], zone_geom=target["zone_geom"]),
        table=TableConfig(
            body=table["body"],
            top_geom=table["top_geom"],
            top_z=float(table["top_z"]),
            x=Range.parse(table["x"]),
            y=Range.parse(table["y"]),
        ),
        workspace=WorkspaceConfig(
            block_x=Range.parse(workspace["block_x"]),
            block_y=Range.parse(workspace["block_y"]),
            block_yaw=Range.parse(workspace["block_yaw"]),
            target_x=Range.parse(workspace["target_x"]),
            target_y=Range.parse(workspace["target_y"]),
            block_target_distance=Range.parse(workspace["block_target_distance"]),
            max_reset_attempts=int(workspace["max_reset_attempts"]),
            target_tolerance=float(workspace["target_tolerance"]),
            allowed_block_x=Range.parse(workspace["allowed_block_x"]),
            allowed_block_y=Range.parse(workspace["allowed_block_y"]),
        ),
        physics=PhysicsConfig(
            contact_pair=physics["contact_pair"],
            nominal_mass=float(physics["nominal_mass"]),
            nominal_sliding_friction=float(physics["nominal_sliding_friction"]),
            mass_range=Range.parse(physics["mass_range"]),
            sliding_friction_range=Range.parse(physics["sliding_friction_range"]),
            torsional_friction=float(physics["torsional_friction"]),
            rolling_friction=float(physics["rolling_friction"]),
        ),
        stability=StabilityConfig(max_abs_qvel=float(stability["max_abs_qvel"])),
        reach_check=ReachCheckConfig(
            margin=float(reach["margin"]),
            grid=int(reach["grid"]),
            heights=tuple(float(h) for h in reach["heights"]),
            min_joint_margin=float(reach["min_joint_margin"]),
            position_tolerance=float(reach["position_tolerance"]),
            orientation_tolerance=float(reach["orientation_tolerance"]),
        ),
        checks=ChecksConfig(
            reset_state_tolerance=float(checks["reset_state_tolerance"]),
            rollout_seconds=float(checks["rollout_seconds"]),
            layout_seeds=int(checks["layout_seeds"]),
            hold_seconds=float(checks["hold_seconds"]),
            hold_seeds=int(checks["hold_seeds"]),
            long_hold_seconds=float(checks["long_hold_seconds"]),
            hold_max_final_qvel=float(checks["hold_max_final_qvel"]),
            hold_max_block_drift=float(checks["hold_max_block_drift"]),
            hold_max_arm_deviation=float(checks["hold_max_arm_deviation"]),
            kick_speed=float(checks["kick_speed"]),
            kick_mu_rel_tolerance=float(checks["kick_mu_rel_tolerance"]),
            normal_force_rel_tolerance=float(checks["normal_force_rel_tolerance"]),
            actuator_step=float(checks["actuator_step"]),
            actuator_seconds=float(checks["actuator_seconds"]),
            actuator_tracking_tolerance=float(checks["actuator_tracking_tolerance"]),
            gripper_tolerance=float(checks["gripper_tolerance"]),
            observation_rest_tolerance=float(checks["observation_rest_tolerance"]),
            visibility_seeds=int(checks["visibility_seeds"]),
            max_arm_occlusion=float(checks["max_arm_occlusion"]),
        ),
    )
    _validate(cfg)
    return cfg
