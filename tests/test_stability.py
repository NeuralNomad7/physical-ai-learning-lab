"""Headless stepping: stable holds, the instability detector and actuator response."""

from __future__ import annotations

import numpy as np
import pytest

from physical_ai_lab.config import SceneConfig
from physical_ai_lab.diagnostics import HoldReport, actuator_response_test, hold_test
from physical_ai_lab.scenario import ScenarioRunner, SimulationUnstable


def assert_quiet_hold(report: HoldReport, cfg: SceneConfig) -> None:
    c = cfg.checks
    assert report.stable, report.reason
    assert report.final_max_abs_qvel <= c.hold_max_final_qvel
    assert report.block_planar_drift <= c.hold_max_block_drift
    assert report.arm_max_deviation <= c.hold_max_arm_deviation
    assert not report.robot_contacts


@pytest.mark.parametrize("randomize", [False, True], ids=["nominal", "randomized"])
@pytest.mark.parametrize("seed", range(3))
def test_home_pose_hold_is_stable(runner: ScenarioRunner, cfg: SceneConfig, seed: int, randomize: bool) -> None:
    assert_quiet_hold(hold_test(runner, seed, seconds=cfg.checks.hold_seconds, randomize_physics=randomize), cfg)


def test_hold_over_the_full_episode_time_budget(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    assert_quiet_hold(hold_test(runner, 0, seconds=cfg.checks.long_hold_seconds, randomize_physics=True), cfg)


def test_one_control_update_is_exactly_the_configured_substeps(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    runner.reset(0)
    assert runner.step(0).control_step == 0
    obs = runner.step(1)
    assert obs.control_step == 1
    assert obs.time == pytest.approx(cfg.timing.substeps * cfg.timing.physics_timestep, rel=0, abs=1e-12)
    with pytest.raises(ValueError):
        runner.step(-1)


def test_runaway_velocity_is_detected_and_reset_recovers(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    runner.reset(0)
    initial = runner.state_vector().copy()
    dof = runner.sim_model.joint(cfg.block.joint).dofadr[0]
    runner.sim_data.qvel[dof] = 200.0  # m/s, far beyond [stability] max_abs_qvel
    with pytest.raises(SimulationUnstable, match="exceeds"):
        runner.step(5)
    assert not runner.stability().ok
    runner.reset(0)
    np.testing.assert_array_equal(runner.state_vector(), initial)
    assert runner.stability().ok


def test_non_finite_state_is_reported(runner: ScenarioRunner) -> None:
    runner.reset(0)
    runner.sim_data.qvel[0] = np.nan
    report = runner.stability()
    assert not report.ok and not report.finite and "non-finite" in report.reason
    runner.reset(0)
    assert runner.stability().ok


def test_servos_drive_the_intended_joints(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    c = cfg.checks
    result = actuator_response_test(
        runner,
        arm_step=c.actuator_step,
        seconds=c.actuator_seconds,
        arm_tolerance=c.actuator_tracking_tolerance,
        gripper_tolerance=c.gripper_tolerance,
    )
    assert result.ok, result.reason
    assert result.arm_max_error <= c.actuator_tracking_tolerance
    assert result.gripper_closed_opening <= c.gripper_tolerance
