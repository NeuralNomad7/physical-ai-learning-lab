"""Evaluator (Phase 2): success and failure detection, push budget, records and config validation.

Judge tests set the live simulation state directly and call the judge as the step monitor
would. That is test-harness access on the evaluator side; controllers never get it.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from physical_ai_lab import paths
from physical_ai_lab.controller import EpisodeOver
from physical_ai_lab.evaluation import (
    FAILURE_REASONS,
    TRAJECTORY_COLUMNS,
    ContactEvent,
    EpisodeJudge,
    EvaluatedPlant,
    run_episode,
)
from physical_ai_lab.push_config import PushConfig, load_push_config
from physical_ai_lab.scenario import ScenarioRunner

DEV_SEED = 21001


@pytest.fixture(scope="module")
def push() -> PushConfig:
    return load_push_config()


def place_block(runner: ScenarioRunner, xyz: np.ndarray) -> None:
    model, data = runner.sim_model, runner.sim_data
    joint = model.joint(runner.config.block.joint)
    adr, dadr = int(joint.qposadr[0]), int(joint.dofadr[0])
    data.qpos[adr:adr + 3] = xyz
    data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def target_xy(runner: ScenarioRunner) -> np.ndarray:
    return runner.observe().target_pos[:2].copy()


def tick(judge: EpisodeJudge, runner: ScenarioRunner, steps: int) -> None:
    """Call the judge as the step monitor would, on a frozen state with advancing time."""
    model, data = runner.sim_model, runner.sim_data
    for _ in range(steps):
        data.time += model.opt.timestep
        judge(model, data)


# ----- success -----


def test_success_needs_the_full_hold_at_rest_in_the_target(runner: ScenarioRunner, push: PushConfig) -> None:
    runner.reset(DEV_SEED)
    place_block(runner, np.r_[target_xy(runner) + [0.015, 0.0], 0.025])  # 15 mm off: inside 20 mm
    judge = EpisodeJudge(runner, push)
    hold_steps = round(push.episode.success_hold_seconds / runner.sim_model.opt.timestep)
    tick(judge, runner, hold_steps - 1)
    assert judge.outcome is None
    tick(judge, runner, 1)
    assert judge.outcome is not None and judge.outcome.success and judge.outcome.reason == "success"


def test_no_success_outside_tolerance_or_while_moving(runner: ScenarioRunner, push: PushConfig) -> None:
    runner.reset(DEV_SEED)
    place_block(runner, np.r_[target_xy(runner) + [0.021, 0.0], 0.025])  # 21 mm: outside
    judge = EpisodeJudge(runner, push)
    tick(judge, runner, 400)
    assert judge.outcome is None

    runner.reset(DEV_SEED)
    place_block(runner, np.r_[target_xy(runner), 0.025])
    dadr = int(runner.sim_model.joint(runner.config.block.joint).dofadr[0])
    runner.sim_data.qvel[dadr] = 0.025  # planar speed above 0.02 m/s
    judge = EpisodeJudge(runner, push)
    tick(judge, runner, 400)
    assert judge.outcome is None


def test_hold_restarts_when_the_block_moves_again(runner: ScenarioRunner, push: PushConfig) -> None:
    runner.reset(DEV_SEED)
    place_block(runner, np.r_[target_xy(runner), 0.025])
    judge = EpisodeJudge(runner, push)
    tick(judge, runner, 200)  # 0.4 s
    dadr = int(runner.sim_model.joint(runner.config.block.joint).dofadr[0])
    runner.sim_data.qvel[dadr] = 0.05
    tick(judge, runner, 1)
    runner.sim_data.qvel[dadr] = 0.0
    tick(judge, runner, 200)
    assert judge.outcome is None  # 0.4 s since the interruption
    tick(judge, runner, 50)
    assert judge.outcome is not None and judge.outcome.success


def test_no_success_while_the_fingers_hold_the_block(runner: ScenarioRunner, push: PushConfig) -> None:
    obs = runner.reset(DEV_SEED)
    place_block(runner, obs.tcp_pos)  # block between the fingers, block centre on the target
    target = runner.sim_model.body_mocapid[runner.sim_model.body(runner.config.target.mocap_body).id]
    runner.sim_data.mocap_pos[target][:2] = obs.tcp_pos[:2]
    mujoco.mj_forward(runner.sim_model, runner.sim_data)
    judge = EpisodeJudge(runner, push)
    tick(judge, runner, 400)
    assert judge.outcome is None  # D21: at rest in the target, but touched by the robot
    fingers = {e.body_a for e in judge.events}
    assert fingers == {"panda/left_finger", "panda/right_finger"}
    assert all(e.kind == "robot-block" and not e.forbidden and e.end is None for e in judge.events)


# ----- failures -----


def test_forbidden_collision_for_a_non_finger_body(runner: ScenarioRunner, push: PushConfig) -> None:
    runner.reset(DEV_SEED)
    hand = runner.sim_model.body(runner.config.robot.tcp_body).id
    place_block(runner, runner.sim_data.xpos[hand].copy())
    judge = EpisodeJudge(runner, push)
    tick(judge, runner, 1)
    assert judge.outcome is not None and judge.outcome.reason == "forbidden_collision"
    assert any(e.forbidden and e.body_a in ("panda/hand", "panda/link7") for e in judge.events)


def test_block_out_of_bounds(runner: ScenarioRunner, push: PushConfig) -> None:
    runner.reset(DEV_SEED)
    place_block(runner, np.array([0.87, 0.0, 0.025]))  # on the table, beyond allowed x <= 0.85
    judge = EpisodeJudge(runner, push)
    tick(judge, runner, 1)
    assert judge.outcome is not None and judge.outcome.reason == "block_out_of_bounds"


def test_joint_speed_and_range_limits(runner: ScenarioRunner, push: PushConfig) -> None:
    runner.reset(DEV_SEED)
    joint = runner.sim_model.joint(runner.config.robot.arm_joints[0])
    runner.sim_data.qvel[joint.dofadr[0]] = push.limits.max_joint_speed[0] + 0.05
    judge = EpisodeJudge(runner, push)
    tick(judge, runner, 1)
    assert judge.outcome is not None and judge.outcome.reason == "limit_violation"
    assert "speed" in judge.outcome.detail

    runner.reset(DEV_SEED)
    joint = runner.sim_model.joint(runner.config.robot.arm_joints[6])  # wrist roll: no self-contact
    hi = runner.sim_model.jnt_range[joint.id][1]
    runner.sim_data.qpos[joint.qposadr[0]] = hi + 2 * push.limits.joint_range_tolerance
    mujoco.mj_forward(runner.sim_model, runner.sim_data)
    judge = EpisodeJudge(runner, push)
    tick(judge, runner, 1)
    assert judge.outcome is not None and judge.outcome.reason == "limit_violation"
    assert "outside its range" in judge.outcome.detail


def test_tcp_height_limit(runner: ScenarioRunner, push: PushConfig) -> None:
    runner.reset(DEV_SEED)  # home TCP is about 0.52 m high; raise the limit above it
    strict = dataclasses.replace(push, limits=dataclasses.replace(push.limits, min_tcp_z=0.6))
    judge = EpisodeJudge(runner, strict)
    tick(judge, runner, 1)
    assert judge.outcome is not None and judge.outcome.reason == "limit_violation"
    assert "TCP height" in judge.outcome.detail
    assert judge.stats["min_tcp_z"] == pytest.approx(0.5211, abs=2e-3)


def test_non_finite_state_is_unstable(runner: ScenarioRunner, push: PushConfig) -> None:
    runner.reset(DEV_SEED)
    judge = EpisodeJudge(runner, push)
    runner.sim_data.qvel[0] = np.nan
    tick(judge, runner, 1)
    runner.reset(DEV_SEED)
    assert judge.outcome is not None and judge.outcome.reason == "unstable"


def test_timeout(runner: ScenarioRunner, push: PushConfig) -> None:
    runner.reset(DEV_SEED)
    judge = EpisodeJudge(runner, push)
    runner.sim_data.time = push.episode.max_seconds - 0.001
    tick(judge, runner, 1)
    assert judge.outcome is not None and judge.outcome.reason == "timeout"


def test_first_outcome_is_final(runner: ScenarioRunner, push: PushConfig) -> None:
    runner.reset(DEV_SEED)
    judge = EpisodeJudge(runner, push)
    judge.fail("timeout", "first", 1.0)
    judge.fail("unstable", "second", 2.0)
    judge.finish(True, "success", "third", 3.0)
    assert (judge.outcome.reason, judge.outcome.detail) == ("timeout", "first")
    with pytest.raises(AssertionError):
        judge.fail("made_up_reason", "", 0.0)


def test_ninth_push_exceeds_the_budget(runner: ScenarioRunner, push: PushConfig) -> None:
    runner.reset(DEV_SEED)
    judge = EpisodeJudge(runner, push)
    plant = EvaluatedPlant(runner, judge, push)
    for i in range(1, push.episode.max_pushes + 1):
        plant.note("push_start", index=i, plan={}, candidates=[])
        plant.note("push_end", index=i, aborted=None)
    assert judge.outcome is None
    with pytest.raises(EpisodeOver):
        plant.note("push_start", index=9, plan={}, candidates=[])
    assert judge.outcome.reason == "push_budget_exceeded"
    with pytest.raises(EpisodeOver):
        plant.command(np.asarray(runner.observe().arm_qpos_target), 0.0)


# ----- whole episodes -----


class StopPolicy:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def run(self, plant: EvaluatedPlant) -> str:
        return self.reason


class CrashPolicy:
    def run(self, plant: EvaluatedPlant) -> str:
        raise RuntimeError("bug in controller code")


class TeleportHarness:
    """Test harness, not a controller: declares pushes, then puts the block on the target
    and records robot/block contacts that happened outside any declared push."""

    def __init__(self, runner: ScenarioRunner, declared: int, unplanned: int) -> None:
        self.runner, self.declared, self.unplanned = runner, declared, unplanned

    def run(self, plant: EvaluatedPlant) -> str:
        for i in range(1, self.declared + 1):
            plant.note("push_start", index=i, plan={}, candidates=[])
            plant.note("push_end", index=i, aborted=None)
        place_block(self.runner, np.r_[target_xy(self.runner), 0.025])
        for k in range(self.unplanned):
            plant._judge.events.append(ContactEvent("robot-block", "panda/left_finger", "block", float(k),
                                                    float(k) + 0.01, None, "between-pushes", False))
        return "done"


@pytest.mark.parametrize("reason", ["no_feasible_push", "push_budget_exhausted"])
def test_a_stopped_controller_fails_with_its_reason(runner: ScenarioRunner, push: PushConfig, reason: str,
                                                    tmp_path: Path) -> None:
    trajectory = tmp_path / "episode.csv"
    record = run_episode(runner, StopPolicy(reason), push, DEV_SEED, trajectory_path=trajectory)
    assert not record.success and record.reason == reason and record.stop_reason == reason
    assert record.pushes_used == 0 and record.contact_events == []
    assert record.end_time == pytest.approx(
        push.episode.success_hold_seconds + push.controller.settle_max_seconds, abs=0.03)
    with trajectory.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == TRAJECTORY_COLUMNS and len(rows) > 50


def test_unknown_stop_and_controller_exceptions_are_failures(runner: ScenarioRunner, push: PushConfig) -> None:
    record = run_episode(runner, StopPolicy("gave up"), push, DEV_SEED)
    assert not record.success and record.reason == "controller_error"
    record = run_episode(runner, CrashPolicy(), push, DEV_SEED)
    assert not record.success and record.reason == "controller_error"
    assert "bug in controller code" in record.detail and "RuntimeError" in record.controller_traceback


def test_unplanned_contacts_count_against_the_push_budget(runner: ScenarioRunner, push: PushConfig) -> None:
    budget = push.episode.max_pushes
    ok = run_episode(runner, TeleportHarness(runner, budget - 1, 1), push, DEV_SEED)
    assert ok.success and ok.pushes_used == budget and ok.unplanned_robot_block_contacts == 1
    over = run_episode(runner, TeleportHarness(runner, budget - 1, 2), push, DEV_SEED)
    assert not over.success and over.reason == "push_budget_exceeded"
    assert over.pushes_used == budget + 1


def test_failure_reasons_are_the_documented_set() -> None:
    assert set(FAILURE_REASONS) == {
        "unstable", "forbidden_collision", "limit_violation", "block_out_of_bounds", "timeout",
        "push_budget_exceeded", "push_budget_exhausted", "no_feasible_push", "controller_error",
    }


# ----- configs/push.toml -----


def test_push_config_matches_section_4(push: PushConfig) -> None:
    assert push.episode.max_pushes == 8 and push.episode.max_seconds == 30.0
    assert push.episode.success_hold_seconds == 0.5 and push.episode.success_max_planar_speed == 0.02
    assert len(push.action.speeds) == 3 and len(push.action.durations) == 3
    assert len(push.seeds.gate) == 20 and push.seeds.gate_required_successes == 18
    assert not set(push.seeds.gate) & set(push.seeds.development)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("20000, 20001", "21000, 20001", "disjoint"),
        ("schema_version = 1", "schema_version = 2", "schema_version"),
        ("max_joint_speed = [1.2, 1.2", "max_joint_speed = [2.5, 1.2", "must not exceed"),
        ("min_tcp_z = 0.015", "min_tcp_z = 0.005", "min_tcp_z"),
        ("max_off_normal_deg = 20.0", "max_off_normal_deg = 50.0", "max_off_normal"),
        ("gate_required_successes = 18", "gate_required_successes = 21", "gate_required_successes"),
    ],
)
def test_push_config_validation(tmp_path: Path, old: str, new: str, message: str) -> None:
    text = paths.PUSH_CONFIG.read_text(encoding="utf-8")
    assert old in text
    bad = tmp_path / "push.toml"
    bad.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_push_config(bad)
