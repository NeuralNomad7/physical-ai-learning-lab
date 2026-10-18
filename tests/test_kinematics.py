"""Controller kinematics (Phase 2): private model instance, forward kinematics, damped IK, gravity offset."""

from __future__ import annotations

import itertools
import math

import mujoco
import numpy as np
import pytest

from physical_ai_lab.config import SceneConfig
from physical_ai_lab.kinematics import ArmKinematics, down_rotation, wrap_angle
from physical_ai_lab.push_config import PushConfig, load_push_config
from physical_ai_lab.scenario import HiddenPhysics, ScenarioRunner

DEV_SEED = 21000  # a development seed (configs/push.toml); never a gate seed
# TCP-down poses (x, y, hand yaw) across and beyond the push area, near and far from the base.
HOLD_POSES = [(0.30, 0.0, 0.0), (0.40, -0.25, math.pi / 2), (0.40, 0.25, -math.pi / 4),
              (0.65, -0.20, -math.pi / 4), (0.65, 0.20, 0.0), (0.72, 0.0, math.pi / 2)]


@pytest.fixture(scope="module")
def kin(cfg: SceneConfig) -> ArmKinematics:
    return ArmKinematics(cfg)


@pytest.fixture(scope="module")
def push() -> PushConfig:
    return load_push_config()


def test_kinematics_use_their_own_model_instance(kin: ArmKinematics, runner: ScenarioRunner) -> None:
    runner.reset(3)
    assert kin.model is not runner.sim_model
    assert kin.data is not runner.sim_data
    digest = runner.state_digest()
    kin.solve(np.array([0.5, 0.1, 0.05]), 0.3, kin.home_q)
    kin.forward(kin.home_q + 0.2)
    assert runner.state_digest() == digest  # IK and FK never touch the simulation


def test_forward_kinematics_match_the_observed_tcp(kin: ArmKinematics, runner: ScenarioRunner) -> None:
    obs = runner.reset(5)
    tcp, rotation = kin.forward(obs.arm_qpos)
    np.testing.assert_allclose(tcp, obs.tcp_pos, atol=1e-9)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(kin.home_q, obs.arm_qpos, atol=1e-9)


def test_down_rotation_and_wrap_angle() -> None:
    for yaw in (-2.0, 0.0, 0.7, 3.0):
        r = down_rotation(yaw)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)
        np.testing.assert_allclose(r[:, 2], [0.0, 0.0, -1.0])
        assert math.atan2(r[1, 0], r[0, 0]) == pytest.approx(yaw)
    assert wrap_angle(math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_angle(-0.25) == pytest.approx(-0.25)


@pytest.mark.parametrize("yaw", [-math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2])
def test_ik_reaches_tcp_down_poses_over_the_push_area(kin: ArmKinematics, push: PushConfig, yaw: float) -> None:
    ctl = push.controller
    margin = ctl.joint_margin
    lo, hi = kin.joint_lo + margin, kin.joint_hi - margin
    for x, y, z in itertools.product((0.35, 0.50, 0.68), (-0.22, 0.0, 0.22), (ctl.push_height, ctl.safe_height)):
        target = np.array([x, y, z])
        rest = kin.home_q.copy()
        rest[0] = math.atan2(y, x)
        # The closed fingers are symmetric about the hand z-axis, so the push skill accepts the
        # hand yaw or the yaw + 180 deg (joint 7 cannot turn far enough for every yaw).
        for hand_yaw in (yaw, wrap_angle(yaw + math.pi)):
            result = kin.solve(target, hand_yaw, kin.home_q, rest, joint_margin=margin,
                               position_tolerance=ctl.ik_position_tolerance,
                               orientation_tolerance=ctl.ik_orientation_tolerance)
            if result.converged:
                break
        assert result.converged, (target, yaw, result.position_error)
        tcp, rotation = kin.forward(result.q)
        assert np.linalg.norm(tcp - target) <= ctl.ik_position_tolerance + 1e-12
        assert rotation[2, 2] == pytest.approx(-1.0, abs=1e-4)  # hand z points down
        assert abs(wrap_angle(kin.hand_yaw(result.q) - hand_yaw)) <= ctl.ik_orientation_tolerance + 1e-9
        assert np.all(result.q >= lo - 1e-12) and np.all(result.q <= hi + 1e-12)
        assert result.joint_margin == pytest.approx(kin.joint_margin(result.q))
        assert result.joint_margin >= margin - 1e-12


def test_ik_reports_unreachable_targets(kin: ArmKinematics) -> None:
    result = kin.solve(np.array([1.6, 0.0, 0.02]), 0.0, kin.home_q, max_iterations=100)
    assert not result.converged
    assert result.position_error > 0.5
    assert np.all(result.q >= kin.joint_lo) and np.all(result.q <= kin.joint_hi)


# ----- gravity offset (review of Phase 2, D25) -----


def push_height_pose(kin: ArmKinematics, push: PushConfig, x: float, y: float, yaw: float) -> np.ndarray:
    ctl = push.controller
    rest = kin.home_q.copy()
    rest[0] = math.atan2(y, x)
    for hand_yaw in (yaw, wrap_angle(yaw + math.pi)):
        result = kin.solve(np.array([x, y, ctl.push_height]), hand_yaw, kin.home_q, rest,
                           joint_margin=ctl.joint_margin, position_tolerance=ctl.ik_position_tolerance,
                           orientation_tolerance=ctl.ik_orientation_tolerance)
        if result.converged:
            return result.q
    raise AssertionError(f"no IK solution at ({x}, {y})")


def test_gravity_offset_holds_the_tcp_where_commanded(cfg: SceneConfig, kin: ArmKinematics,
                                                      push: PushConfig) -> None:
    """In a separate simulation, servo targets q + offset come to rest at q; targets q sag."""
    model = mujoco.MjModel.from_xml_path(str(cfg.mjcf_path))
    data = mujoco.MjData(model)
    arm = [model.actuator(name).id for name in cfg.robot.arm_actuators]
    gripper = model.actuator(cfg.robot.gripper_actuator).id
    fingers = [model.joint(name).qposadr[0] for name in cfg.robot.finger_joints]
    hand = model.body(cfg.robot.tcp_body).id
    robot = model.body_rootid[hand]

    def resting_tcp(q: np.ndarray, servo_targets: np.ndarray) -> tuple[np.ndarray, float]:
        mujoco.mj_resetDataKeyframe(model, data, model.key(cfg.robot.home_keyframe).id)
        data.qpos[kin.arm_qadr] = q
        data.qpos[fingers] = 0.0
        data.ctrl[arm] = servo_targets
        data.ctrl[gripper] = cfg.robot.gripper_ctrl_closed
        for _ in range(round(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            for contact in data.contact[: data.ncon]:  # the arm hangs free: no contact holds it up
                assert robot not in model.body_rootid[model.geom_bodyid[[contact.geom1, contact.geom2]]]
        rotation = data.xmat[hand].reshape(3, 3)
        return data.xpos[hand] + rotation @ kin.tcp_offset, float(np.max(np.abs(data.qvel[kin.arm_dadr])))

    sags = []
    for x, y, yaw in HOLD_POSES:
        q = push_height_pose(kin, push, x, y, yaw)
        commanded, _ = kin.forward(q)
        held, speed = resting_tcp(q, q + kin.gravity_offset(q))
        assert speed < 1e-6  # at rest
        assert np.linalg.norm(held - commanded) < 5e-4, (x, y, held - commanded)
        sagged, _ = resting_tcp(q, q)
        sags.append(commanded[2] - sagged[2])
    # Without the offset the same servos rest well below the setpoint, more at long reach.
    assert min(sags) > 1e-3 and max(sags) > 8e-3, sags


def test_gravity_offset_matches_the_simulator_and_ignores_the_block(cfg: SceneConfig, runner: ScenarioRunner,
                                                                   push: PushConfig) -> None:
    kin = ArmKinematics(cfg)  # a private instance: this test edits its model
    p = cfg.physics
    for physics in (HiddenPhysics(p.mass_range.low, p.sliding_friction_range.low),
                    HiddenPhysics(p.mass_range.high, p.sliding_friction_range.high)):
        obs = runner.reset(DEV_SEED, physics=physics)
        at_rest = runner.sim_data.qfrc_bias[kin.arm_dadr] / kin.servo_kp  # reset state: zero velocity
        np.testing.assert_allclose(kin.gravity_offset(obs.arm_qpos), at_rest, rtol=0, atol=1e-12)
    poses = [kin.home_q] + [push_height_pose(kin, push, x, y, yaw) for x, y, yaw in HOLD_POSES[3:]]
    before = [kin.gravity_offset(q) for q in poses]
    assert max(float(np.max(np.abs(o))) for o in before) > 5e-3  # not trivially zero
    block = kin.model.body(cfg.block.body).id
    kin.model.body_mass[block] *= 10.0
    kin.model.body_inertia[block] *= 10.0
    for q, offset in zip(poses, before):
        np.testing.assert_array_equal(kin.gravity_offset(q), offset)


def test_gravity_offset_leaves_forward_kinematics_unchanged(kin: ArmKinematics, push: PushConfig) -> None:
    q = push_height_pose(kin, push, 0.65, 0.20, 0.0)
    tcp, rotation = kin.forward(q)
    kin.gravity_offset(kin.home_q)
    again, rotation_again = kin.forward(q)
    np.testing.assert_array_equal(tcp, again)
    np.testing.assert_array_equal(rotation, rotation_again)
    np.testing.assert_allclose(kin.servo_kp, [4500, 4500, 3500, 3500, 2000, 2000, 2000])
