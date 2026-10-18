"""The controller's private model of the Panda: kinematics and static arm gravity (Phase 2).

``ArmKinematics`` loads its own ``MjModel`` from the scene MJCF and keeps its own
``MjData``. It only ever writes arm joint angles into that private data and runs
forward kinematics, plus one static computation: the gravity load of the robot's own
links (``gravity_offset``). It cannot see or change the simulation the evaluator runs
(no shared state, no time stepping, no rollouts). It never reads block mass, friction or
any other hidden value: the block is not part of the arm's gravity load, and contacts
are not modelled.

The TCP is ``robot.tcp_body`` + ``robot.tcp_offset`` (in the hand frame), as in
``observation.py``. A "TCP-down" pose has the hand z-axis pointing straight down and
the hand x-axis at a chosen yaw in the table plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .config import SceneConfig, load_scene_config
from .scenario import verify_actuators

DOWN = np.array([0.0, 0.0, -1.0])


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def down_rotation(yaw: float) -> np.ndarray:
    """Hand rotation (columns = hand axes in the world) with z down and x at ``yaw``."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    position_error: float  # m
    orientation_error: float  # rad (angle of the rotation between achieved and desired)
    iterations: int
    converged: bool
    joint_margin: float  # rad, smallest distance of any joint to its range limit


class ArmKinematics:
    """Forward and inverse kinematics on a private, kinematics-only model instance."""

    def __init__(self, config: SceneConfig | None = None) -> None:
        self.config = cfg = config or load_scene_config()
        self.model = mujoco.MjModel.from_xml_path(str(cfg.mjcf_path))
        self.data = mujoco.MjData(self.model)
        model, robot = self.model, cfg.robot
        joints = [model.joint(name) for name in robot.arm_joints]
        self.arm_qadr = np.array([j.qposadr[0] for j in joints], dtype=int)
        self.arm_dadr = np.array([j.dofadr[0] for j in joints], dtype=int)
        self.joint_lo = np.array([model.jnt_range[j.id][0] for j in joints])
        self.joint_hi = np.array([model.jnt_range[j.id][1] for j in joints])
        self.hand = model.body(robot.tcp_body).id
        self.tcp_offset = np.array(robot.tcp_offset, dtype=float)
        key = model.key(robot.home_keyframe)
        self.home_q = np.array(key.qpos[self.arm_qadr], dtype=float)
        mujoco.mj_resetDataKeyframe(model, self.data, key.id)
        # Position-servo stiffness per arm joint; verify_actuators raises unless every arm
        # actuator is a fixed-gain position servo on its joint.
        self.servo_kp = np.array([a.kp for a in verify_actuators(model, cfg)[: len(joints)]])
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._bias = np.zeros(model.nv)

    @property
    def n(self) -> int:
        return len(self.arm_qadr)

    def _set(self, q: np.ndarray) -> None:
        self.data.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP position and hand rotation matrix for arm joint angles ``q``."""
        self._set(np.asarray(q, dtype=float))
        rotation = self.data.xmat[self.hand].reshape(3, 3).copy()
        return self.data.xpos[self.hand] + rotation @ self.tcp_offset, rotation

    def hand_yaw(self, q: np.ndarray) -> float:
        """Yaw of the hand x-axis projected on the table plane."""
        _, rotation = self.forward(q)
        return math.atan2(rotation[1, 0], rotation[0, 0])

    def joint_margin(self, q: np.ndarray) -> float:
        q = np.asarray(q, dtype=float)
        return float(np.min(np.minimum(q - self.joint_lo, self.joint_hi - q)))

    def gravity_offset(self, q: np.ndarray) -> np.ndarray:
        """Servo-target offset (rad) that holds the arm at ``q`` against its own weight.

        A position servo pushes ``kp * (target - q)`` (at rest), so with ``target = q`` the arm
        settles where that force balances gravity, i.e. it sags by about ``gravity / kp``. This
        returns ``gravity(q) / kp``: the static joint torques from the robot's own links (zero
        velocity), divided by each servo's stiffness. Sending ``q + gravity_offset(q)`` makes
        ``q`` the resting pose. The block is not attached to the arm, so its (hidden) mass
        does not enter these torques.
        """
        self._set(np.asarray(q, dtype=float))
        self.data.qvel[:] = 0.0
        mujoco.mj_comVel(self.model, self.data)
        mujoco.mj_rne(self.model, self.data, 0, self._bias)
        return self._bias[self.arm_dadr] / self.servo_kp

    def solve(
        self,
        target: np.ndarray,
        yaw: float,
        q_init: np.ndarray,
        q_rest: np.ndarray | None = None,
        *,
        joint_margin: float = 0.0,
        position_tolerance: float = 1e-4,
        orientation_tolerance: float = 1e-3,
        damping: float = 0.02,
        max_step: float = 0.2,
        max_iterations: int = 200,
        nullspace_gain: float = 0.05,
    ) -> IKResult:
        """Damped least-squares IK: TCP at ``target``, hand z down, hand x at ``yaw``.

        Joint angles are kept ``joint_margin`` inside their ranges. The nullspace term pulls
        the redundant degree of freedom toward ``q_rest`` (default: ``q_init``).
        """
        target = np.asarray(target, dtype=float)
        desired = down_rotation(yaw)
        lo, hi = self.joint_lo + joint_margin, self.joint_hi - joint_margin
        q = np.clip(np.asarray(q_init, dtype=float).copy(), lo, hi)
        rest = q.copy() if q_rest is None else np.asarray(q_rest, dtype=float)
        eye_n, eye6 = np.eye(self.n), np.eye(6)
        position_error = orientation_error = math.inf
        for iteration in range(max_iterations + 1):
            self._set(q)
            rotation = self.data.xmat[self.hand].reshape(3, 3)
            tcp = self.data.xpos[self.hand] + rotation @ self.tcp_offset
            e_pos = target - tcp
            e_ori = 0.5 * sum(np.cross(rotation[:, i], desired[:, i]) for i in range(3))
            position_error = float(np.linalg.norm(e_pos))
            cos_angle = (float(np.trace(rotation.T @ desired)) - 1.0) / 2.0
            orientation_error = math.acos(max(-1.0, min(1.0, cos_angle)))
            if position_error <= position_tolerance and orientation_error <= orientation_tolerance:
                return IKResult(q, position_error, orientation_error, iteration, True, self.joint_margin(q))
            if iteration == max_iterations:
                break
            mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, tcp, self.hand)
            jac = np.vstack([self._jacp[:, self.arm_dadr], self._jacr[:, self.arm_dadr]])
            pinv = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * eye6, eye6)
            # The nullspace projector uses the undamped pseudo-inverse, so the posture term
            # cannot leak into the task error near convergence.
            nullspace = eye_n - np.linalg.pinv(jac, rcond=1e-4) @ jac
            dq = pinv @ np.concatenate([e_pos, e_ori]) + nullspace @ (nullspace_gain * (rest - q))
            largest = float(np.max(np.abs(dq)))
            if largest > max_step:
                dq *= max_step / largest
            q = np.clip(q + dq, lo, hi)
        return IKResult(q, position_error, orientation_error, max_iterations, False, self.joint_margin(q))
