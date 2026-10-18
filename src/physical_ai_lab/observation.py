"""Agent-facing observation boundary.

`Observation` is the only simulator output that agent code (controller, predictor,
planner, learner) may consume. It is built from an explicit allow-list of state
quantities and holds read-only NumPy copies and plain Python scalars:

* no live ``mujoco.MjModel`` / ``mujoco.MjData`` references;
* no hidden physical parameters (block mass, inertia, friction);
* no scenario identity (seed, manifest entry) and no future state.

Version 1 uses simulator-derived poses instead of perception. Every observation
carries ``pose_source = "simulator_state"`` so that this stays explicit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from typing import Any

import mujoco
import numpy as np

from .config import SceneConfig

POSE_SOURCE = "simulator_state"

_ARRAY_FIELDS = (
    "arm_qpos",
    "arm_qvel",
    "arm_qpos_target",
    "tcp_pos",
    "tcp_quat",
    "block_pos",
    "block_quat",
    "block_linvel",
    "block_angvel",
    "target_pos",
)
_FIXED_SIZES = {
    "tcp_pos": 3,
    "tcp_quat": 4,
    "block_pos": 3,
    "block_quat": 4,
    "block_linvel": 3,
    "block_angvel": 3,
    "target_pos": 3,
}


def yaw_from_quat(quat: np.ndarray) -> float:
    """Rotation about world z of a (w, x, y, z) quaternion, in (-pi, pi]."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True, eq=False)
class Observation:
    """What the agent may see at one instant (SI units, world frame, quaternions w-x-y-z)."""

    time: float  # simulated seconds since reset
    control_step: int  # 50 Hz control updates since reset
    pose_source: str
    arm_qpos: np.ndarray  # rad, 7 arm joints
    arm_qvel: np.ndarray  # rad/s
    arm_qpos_target: np.ndarray  # rad, commanded joint position targets
    gripper_opening: float  # m, sum of both finger joint positions (0 = closed)
    tcp_pos: np.ndarray  # m, Franka Hand TCP (103.4 mm along the hand z-axis)
    tcp_quat: np.ndarray
    block_pos: np.ndarray  # m, block centre
    block_quat: np.ndarray
    block_yaw: float  # rad
    block_linvel: np.ndarray  # m/s
    block_angvel: np.ndarray  # rad/s
    target_pos: np.ndarray  # m, target centre on the tabletop
    target_tolerance: float  # m, success radius on the block centre
    robot_block_contact: bool
    robot_table_contact: bool

    def __post_init__(self) -> None:
        for name in _ARRAY_FIELDS:
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if array.ndim != 1 or (name in _FIXED_SIZES and array.shape[0] != _FIXED_SIZES[name]):
                raise ValueError(f"observation field {name!r} has shape {array.shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("time", "gripper_opening", "block_yaw", "target_tolerance"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "control_step", int(self.control_step))
        object.__setattr__(self, "pose_source", str(self.pose_source))
        for name in ("robot_block_contact", "robot_table_contact"):
            object.__setattr__(self, name, bool(getattr(self, name)))

    @property
    def block_planar_speed(self) -> float:
        return float(np.hypot(self.block_linvel[0], self.block_linvel[1]))

    @property
    def block_target_distance(self) -> float:
        """Planar distance between block centre and target centre (the success metric)."""
        return float(np.hypot(*(self.block_pos[:2] - self.target_pos[:2])))

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: (getattr(self, f.name).tolist() if f.name in _ARRAY_FIELDS else getattr(self, f.name))
            for f in fields(self)
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Observation":
        unknown = set(payload) - set(OBSERVATION_FIELDS)
        missing = set(OBSERVATION_FIELDS) - set(payload)
        if unknown or missing:
            raise ValueError(f"observation keys mismatch: unknown={sorted(unknown)} missing={sorted(missing)}")
        return cls(**payload)

    def max_abs_difference(self, other: "Observation") -> float:
        """Largest absolute difference over all numeric fields (inf if shapes/labels differ)."""
        worst = 0.0
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, str) or isinstance(b, str):
                if a != b:
                    return math.inf
                continue
            a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            if a_arr.shape != b_arr.shape:
                return math.inf
            if a_arr.size:
                worst = max(worst, float(np.max(np.abs(a_arr - b_arr))))
        return worst


OBSERVATION_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Observation))


class ObservationAdapter:
    """Builds `Observation` objects from simulator state using an explicit allow-list.

    Simulator side: it reads joint/body kinematics, actuator targets, mocap target
    pose and contact geometry. It never reads masses, inertias or friction values.
    """

    def __init__(self, model: mujoco.MjModel, cfg: SceneConfig) -> None:
        robot = cfg.robot
        self._arm_qadr = np.array([model.joint(n).qposadr[0] for n in robot.arm_joints], dtype=int)
        self._arm_dadr = np.array([model.joint(n).dofadr[0] for n in robot.arm_joints], dtype=int)
        self._arm_act = np.array([model.actuator(n).id for n in robot.arm_actuators], dtype=int)
        self._finger_qadr = np.array([model.joint(n).qposadr[0] for n in robot.finger_joints], dtype=int)
        self._hand = model.body(robot.tcp_body).id
        self._tcp_offset = np.array(robot.tcp_offset, dtype=float)
        self._block_body = model.body(cfg.block.body).id
        self._block_geom = model.geom(cfg.block.geom).id
        self._target_mocap = int(model.body_mocapid[model.body(cfg.target.mocap_body).id])
        if self._target_mocap < 0:
            raise ValueError(f"target body {cfg.target.mocap_body!r} is not a mocap body")
        self._target_tolerance = cfg.workspace.target_tolerance
        robot_root = model.body_rootid[model.jnt_bodyid[model.joint(robot.arm_joints[0]).id]]
        self._robot_geom = (model.body_rootid == robot_root)[model.geom_bodyid]
        self._table_geom = model.geom_bodyid == model.body(cfg.table.body).id

    @property
    def robot_geom_mask(self) -> np.ndarray:
        return self._robot_geom.copy()

    def contact_flags(self, data: mujoco.MjData) -> tuple[bool, bool]:
        """(robot touches block, robot touches table) from active penetrating contacts."""
        if data.ncon == 0:
            return False, False
        geom = np.asarray(data.contact.geom)
        active = (np.asarray(data.contact.exclude) == 0) & (np.asarray(data.contact.dist) <= 0.0)
        robot = self._robot_geom[geom]
        block = geom == self._block_geom
        table = self._table_geom[geom]
        robot_block = active & ((robot[:, 0] & block[:, 1]) | (robot[:, 1] & block[:, 0]))
        robot_table = active & ((robot[:, 0] & table[:, 1]) | (robot[:, 1] & table[:, 0]))
        return bool(robot_block.any()), bool(robot_table.any())

    def build(self, model: mujoco.MjModel, data: mujoco.MjData, control_step: int) -> Observation:
        """Requires kinematics consistent with ``data.qpos`` (call after mj_forward)."""
        rotation = data.xmat[self._hand].reshape(3, 3)
        tcp_pos = data.xpos[self._hand] + rotation @ self._tcp_offset
        block_quat = data.xquat[self._block_body]
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self._block_body, velocity, 0)
        robot_block, robot_table = self.contact_flags(data)
        return Observation(
            time=data.time,
            control_step=control_step,
            pose_source=POSE_SOURCE,
            arm_qpos=data.qpos[self._arm_qadr],
            arm_qvel=data.qvel[self._arm_dadr],
            arm_qpos_target=data.ctrl[self._arm_act],
            gripper_opening=float(np.sum(data.qpos[self._finger_qadr])),
            tcp_pos=tcp_pos,
            tcp_quat=data.xquat[self._hand],
            block_pos=data.xpos[self._block_body],
            block_quat=block_quat,
            block_yaw=yaw_from_quat(block_quat),
            block_linvel=velocity[3:],
            block_angvel=velocity[:3],
            target_pos=data.mocap_pos[self._target_mocap],
            target_tolerance=self._target_tolerance,
            robot_block_contact=robot_block,
            robot_table_contact=robot_table,
        )
