"""Manual control in the viewer, tested headless: actuator inputs, the joint jog, the viewer session and replay."""

from __future__ import annotations

import math
from collections.abc import Callable

import mujoco
import numpy as np
import pytest

from physical_ai_lab.config import SceneConfig
from physical_ai_lab.scenario import ScenarioRunner
from physical_ai_lab.viewer import (
    JOG_STEPS_DEG,
    KEY_DOWN,
    KEY_ENTER,
    KEY_EQUAL,
    KEY_HOME,
    KEY_KP_ADD,
    KEY_KP_SUBTRACT,
    KEY_LEFT,
    KEY_MINUS,
    KEY_RIGHT,
    KEY_SPACE,
    KEY_UP,
    JointJog,
    ViewerSession,
    replay,
)


@pytest.fixture(scope="module")
def fresh(cfg: SceneConfig) -> ScenarioRunner:
    """A second runner, for headless replays next to the session's runner."""
    return ScenarioRunner(cfg)


def make_jog(runner: ScenarioRunner, cfg: SceneConfig) -> JointJog:
    bounds = runner.ctrl_range
    return JointJog(
        runner.home_arm_qpos,
        bounds[runner.arm_actuator_ids],
        bounds[runner.gripper_actuator_id],
        cfg.robot.gripper_ctrl_closed,
    )


def press(session: ViewerSession, *keys: int) -> None:
    for key in keys:
        assert session.press(key)


def reset_digest(runner: ScenarioRunner, seed: int) -> str:
    runner.reset(seed)
    return runner.state_digest()


# ----- ScenarioRunner actuation API -----
def test_set_ctrl_clamps_to_the_control_ranges(runner: ScenarioRunner) -> None:
    runner.reset(0)
    data, arm, gripper = runner.sim_data, runner.arm_actuator_ids, runner.gripper_actuator_id
    bounds = runner.ctrl_range
    assert bounds.shape == (runner.sim_model.nu, 2) and np.isfinite(bounds).all()  # every Panda actuator is limited
    qpos = data.qpos.copy()
    ctrl = data.ctrl.copy()
    ctrl[arm[0]], ctrl[arm[3]], ctrl[gripper] = -10.0, 0.0, 300.0
    applied = runner.set_ctrl(ctrl)
    assert applied[arm[0]] == bounds[arm[0], 0]
    assert applied[arm[3]] == bounds[arm[3], 1] < 0.0  # the elbow cannot straighten fully
    assert applied[gripper] == bounds[gripper, 1]
    np.testing.assert_array_equal(data.ctrl, applied)
    np.testing.assert_array_equal(data.qpos, qpos)  # only the inputs change; nothing is moved directly


def test_set_ctrl_rejects_bad_input(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    runner.reset(0)
    before = runner.sim_data.ctrl.copy()
    with pytest.raises(ValueError, match="shape"):
        runner.set_ctrl(np.zeros(3))
    bad = before.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        runner.set_ctrl(bad)
    np.testing.assert_array_equal(runner.sim_data.ctrl, before)
    with pytest.raises(RuntimeError, match="reset"):
        ScenarioRunner(cfg).set_ctrl(before)


def test_actuator_ids_are_read_only(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    ids = runner.arm_actuator_ids
    assert [runner.sim_model.actuator(int(i)).name for i in ids] == list(cfg.robot.arm_actuators)
    with pytest.raises(ValueError):
        ids[0] = 3


# ----- JointJog -----
def test_jog_selection_and_step_size(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    jog = make_jog(runner, cfg)
    assert jog.selected == 0 and jog.step_deg == 5.0
    assert jog.press(KEY_UP) is False and jog.selected == 6  # wraps from J1 to J7
    assert jog.press(KEY_DOWN) is False and jog.selected == 0
    for key, step in [
        (KEY_MINUS, 2.0),
        (KEY_KP_SUBTRACT, 1.0),
        (KEY_MINUS, 1.0),
        (KEY_EQUAL, 2.0),
        (KEY_KP_ADD, 5.0),
        (KEY_EQUAL, 10.0),
        (KEY_EQUAL, 20.0),
        (KEY_EQUAL, 20.0),
    ]:
        assert jog.press(key) is False
        assert jog.step_deg == step
    assert jog.at_home and jog.gripper_fraction == 0.0  # selection and step keys change no target
    assert jog.press(KEY_SPACE) is None and jog.press(ord("A")) is None


def test_jog_turns_the_selected_joint_within_its_range(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    jog, home = make_jog(runner, cfg), runner.home_arm_qpos
    assert jog.press(KEY_RIGHT) is True
    assert jog.arm[0] == pytest.approx(home[0] + math.radians(5.0), abs=1e-12)
    np.testing.assert_array_equal(jog.arm[1:], home[1:])
    assert not jog.at_home
    assert jog.press(KEY_LEFT) is True and jog.at_home
    for key in (KEY_DOWN, KEY_DOWN, KEY_DOWN, KEY_EQUAL, KEY_EQUAL):  # J4 (elbow), 20-degree steps
        jog.press(key)
    for _ in range(5):  # from -90 degrees, five steps would pass the top of its range (about -4 degrees)
        assert jog.press(KEY_RIGHT) is True
    high = runner.ctrl_range[runner.arm_actuator_ids[3], 1]
    assert jog.arm[3] == high and jog.at_limit(3)
    assert jog.press(KEY_RIGHT) is False  # already at the limit
    assert jog.press(KEY_LEFT) is True and not jog.at_limit(3)
    assert jog.arm[3] == pytest.approx(high - math.radians(20.0), abs=1e-12)


def test_enter_toggles_the_gripper_and_home_restores_the_targets(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    jog = make_jog(runner, cfg)
    closed, opened = cfg.robot.gripper_ctrl_closed, runner.ctrl_range[runner.gripper_actuator_id, 1]
    for key in (KEY_DOWN, KEY_EQUAL, KEY_RIGHT):
        jog.press(key)
    assert jog.press(KEY_ENTER) is True and jog.gripper == opened and jog.gripper_fraction == 1.0
    assert jog.press(KEY_ENTER) is True and jog.gripper == closed
    jog.press(KEY_ENTER)
    assert jog.press(KEY_HOME) is True
    assert jog.at_home and jog.gripper == closed
    assert jog.selected == 1 and jog.step_deg == 10.0  # the selection and the step are kept
    assert jog.press(KEY_HOME) is False  # already there
    jog.load(jog.arm, 100.0)  # e.g. set with the gripper's Control slider
    assert jog.gripper_fraction == pytest.approx((100.0 - closed) / (opened - closed))
    assert jog.press(KEY_ENTER) is True and jog.gripper == opened  # less than half open: opens


# ----- ViewerSession -----
def test_jog_keys_move_the_arm_only_through_its_servos(runner: ScenarioRunner) -> None:
    session = ViewerSession(runner, 0)
    qpos = runner.sim_data.qpos.copy()
    press(session, *[KEY_RIGHT] * 4)
    target = runner.home_arm_qpos[0] + math.radians(20.0)
    assert session.operator_commands == 4 and list(session.commands) == [0]
    assert runner.sim_data.ctrl[runner.arm_actuator_ids[0]] == pytest.approx(target, abs=1e-12)
    np.testing.assert_array_equal(runner.sim_data.qpos, qpos)  # the target changed; no body was moved
    assert session.step(50) is None  # 1 s of control updates
    assert session.obs.arm_qpos[0] == pytest.approx(target, abs=0.02)
    assert (session.updates, session.since_reset, session.unstable, session.restores) == (50, 50, 0, 0)


def test_session_press_only_handles_jog_keys(runner: ScenarioRunner) -> None:
    session = ViewerSession(runner, 0)
    revision = session.revision
    assert session.press(KEY_SPACE) is False and session.revision == revision
    assert session.press(KEY_DOWN) is True and session.revision == revision + 1  # redraw the panel at once
    assert session.operator_commands == 0 and session.commands == {}


def test_control_slider_edits_are_clamped_operator_commands(runner: ScenarioRunner) -> None:
    session = ViewerSession(runner, 0)
    session.step(5)
    data, arm, bounds = runner.sim_data, runner.arm_actuator_ids, runner.ctrl_range
    qpos = data.qpos.copy()
    data.ctrl[:] = 0.0  # the Control panel's "Clear" button: 0 is outside the elbow's range
    assert session.check_external_changes() is None
    applied = data.ctrl.copy()
    np.testing.assert_array_equal(applied, np.clip(0.0, bounds[:, 0], bounds[:, 1]))
    assert applied[arm[3]] == bounds[arm[3], 1]
    assert session.operator_commands == 1 and session.restores == 0
    assert list(session.commands) == [5]
    np.testing.assert_array_equal(session.commands[5], applied)
    np.testing.assert_array_equal(session.jog.arm, applied[arm])
    assert not session.jog.at_home
    np.testing.assert_array_equal(data.qpos, qpos)
    assert session.check_external_changes() is None and session.operator_commands == 1  # nothing new


def _joint_slider(runner: ScenarioRunner) -> None:
    runner.sim_data.qpos[0] += 0.1


def _velocity(runner: ScenarioRunner) -> None:
    runner.sim_data.qvel[0] = 1.0


def _move_goal(runner: ScenarioRunner) -> None:
    runner.sim_data.mocap_pos[0, 0] += 0.05


def _time(runner: ScenarioRunner) -> None:
    runner.sim_data.time += 1.0


def _backspace(runner: ScenarioRunner) -> None:
    mujoco.mj_resetData(runner.sim_model, runner.sim_data)


def _non_finite_ctrl(runner: ScenarioRunner) -> None:
    runner.sim_data.ctrl[0] = np.nan


def _equality_toggle(runner: ScenarioRunner) -> None:
    runner.sim_data.eq_active[0] = 0  # the Equality panel's checkbox: frees the finger coupling


@pytest.mark.parametrize(
    "edit", [_joint_slider, _velocity, _move_goal, _time, _backspace, _non_finite_ctrl, _equality_toggle]
)
def test_other_viewer_edits_restore_the_seed(
    runner: ScenarioRunner, fresh: ScenarioRunner, edit: Callable[[ScenarioRunner], None]
) -> None:
    session = ViewerSession(runner, 0)
    press(session, KEY_DOWN, KEY_RIGHT, KEY_ENTER)
    session.step(10)
    edit(runner)
    message = session.check_external_changes()
    assert message is not None and "restored seed 0" in message
    assert (session.restores, session.since_reset, session.updates) == (1, 0, 10)
    assert session.commands == {} and session.jog.at_home and session.jog.gripper_fraction == 0.0
    assert session.jog.selected == 1  # the selection is kept
    assert runner.state_digest() == reset_digest(fresh, 0)
    assert session.check_external_changes() is None


def test_physics_panel_edits_are_put_back_without_a_restore(
    runner: ScenarioRunner, fresh: ScenarioRunner, cfg: SceneConfig
) -> None:
    session = ViewerSession(runner, 0)
    press(session, KEY_RIGHT)
    session.step(10)
    opt = runner.sim_model.opt
    opt.timestep = 2 * cfg.timing.physics_timestep  # what the viewer's sync copies from its Physics panel
    opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    message = session.check_external_changes()
    assert message == "viewer edited physics option(s) disableflags, timestep; put back the scene's values"
    assert (session.option_reverts, session.restores, session.since_reset, session.operator_commands) == (1, 0, 10, 1)
    assert opt.timestep == cfg.timing.physics_timestep and runner.restore_options() == ()
    assert session.check_external_changes() is None
    session.step(10)
    replay(fresh, 0, updates=session.since_reset, commands=session.commands)
    assert fresh.state_digest() == runner.state_digest()  # the edits never reached the physics


def test_option_and_state_edits_together_restore_the_seed(runner: ScenarioRunner, fresh: ScenarioRunner) -> None:
    session = ViewerSession(runner, 0)
    session.step(5)
    runner.sim_model.opt.iterations = 3
    _backspace(runner)
    message = session.check_external_changes()
    assert message is not None and "iterations" in message and "restored seed 0" in message
    assert (session.option_reverts, session.restores) == (1, 1)
    assert runner.state_digest() == reset_digest(fresh, 0)


def test_instability_restores_the_seed(runner: ScenarioRunner, fresh: ScenarioRunner, cfg: SceneConfig) -> None:
    session = ViewerSession(runner, 0)
    press(session, KEY_RIGHT)
    runner.sim_data.qvel[runner.sim_model.joint(cfg.block.joint).dofadr[0]] = 200.0  # m/s
    message = session.step(5)
    assert message is not None and "unstable" in message and "restored seed 0" in message
    assert (session.unstable, session.updates, session.since_reset) == (1, 0, 0)
    assert session.commands == {} and session.jog.at_home
    assert runner.state_digest() == reset_digest(fresh, 0)
    assert session.check_external_changes() is None


def test_selected_joint_axis_follows_the_selection(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    session = ViewerSession(runner, 0)
    data = runner.sim_data
    for name in cfg.robot.arm_joints:
        joint = runner.sim_model.joint(name).id
        anchor, axis = session.selected_joint_axis()
        np.testing.assert_array_equal(anchor, data.xanchor[joint])
        np.testing.assert_array_equal(axis, data.xaxis[joint])
        assert np.linalg.norm(axis) == pytest.approx(1.0)
        session.press(KEY_DOWN)
    np.testing.assert_allclose(session.selected_joint_axis()[1], [0.0, 0.0, 1.0], atol=1e-9)  # J1 again: vertical


# ----- replay -----
def test_replay_reproduces_a_jogged_run(runner: ScenarioRunner, fresh: ScenarioRunner) -> None:
    session = ViewerSession(runner, 3)
    press(session, KEY_RIGHT, KEY_RIGHT)  # J1 +10 degrees, recorded at 0 updates
    session.step(20)
    press(session, KEY_DOWN, KEY_EQUAL, KEY_LEFT)  # J2 -10 degrees at 20
    session.step(15)
    press(session, KEY_ENTER)  # open the gripper at 35
    session.step(25)
    press(session, KEY_HOME)  # after the last update, at 60
    assert sorted(session.commands) == [0, 20, 35, 60] and session.operator_commands == 5
    final = replay(fresh, 3, updates=session.since_reset, commands=session.commands)
    assert fresh.state_digest() == runner.state_digest()  # includes the actuator inputs
    assert final.to_json() == session.obs.to_json()
    replay(fresh, 3, updates=session.since_reset, commands={k: v for k, v in session.commands.items() if k < 60})
    assert fresh.state_digest() != runner.state_digest()  # the last command is part of the state


def test_replay_without_commands_steps_from_the_seed(runner: ScenarioRunner, fresh: ScenarioRunner) -> None:
    runner.reset(4, randomize_physics=True)
    for _ in range(7):
        runner.step(1)
    replay(fresh, 4, randomize_physics=True, updates=7)
    assert fresh.state_digest() == runner.state_digest()


@pytest.mark.parametrize("count", [-1, 4])
def test_replay_rejects_commands_outside_the_run(fresh: ScenarioRunner, count: int) -> None:
    fresh.reset(0)
    with pytest.raises(ValueError, match="control updates"):
        replay(fresh, 0, updates=3, commands={count: fresh.sim_data.ctrl.copy()})
