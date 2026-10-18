"""Push skill (Phase 2): limits on joint targets, planning filters, execution and retreat."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from physical_ai_lab.baseline import FeedbackBaseline
from physical_ai_lab.config import SceneConfig
from physical_ai_lab.controller import PushAction, PushSkill, move_time, unit_profile
from physical_ai_lab.evaluation import TRAJECTORY_COLUMNS, EpisodeJudge, EvaluatedPlant
from physical_ai_lab.kinematics import ArmKinematics, IKResult, wrap_angle
from physical_ai_lab.observation import Observation
from physical_ai_lab.push_config import PushConfig, load_push_config
from physical_ai_lab.scenario import ScenarioRunner

DEV_SEED = 21000  # a development seed (configs/push.toml); never a gate seed


@pytest.fixture(scope="module")
def push() -> PushConfig:
    return load_push_config()


@pytest.fixture(scope="module")
def kin(cfg: SceneConfig) -> ArmKinematics:
    return ArmKinematics(cfg)


@pytest.fixture
def skill(cfg: SceneConfig, push: PushConfig, kin: ArmKinematics) -> PushSkill:
    return PushSkill(cfg, push, kin)


class FakePlant:
    """Records the commands it receives; the observation never changes."""

    control_dt = 0.02

    def __init__(self, obs: Observation) -> None:
        self.obs = obs
        self.commands: list[tuple[np.ndarray, float]] = []
        self.notes: list[tuple[str, dict]] = []

    def observe(self) -> Observation:
        return self.obs

    def command(self, arm_q: np.ndarray, gripper: float) -> Observation:
        self.commands.append((np.array(arm_q), gripper))
        return self.obs

    def note(self, event: str, **info: object) -> None:
        self.notes.append((event, info))


def face_toward(skill: PushSkill, yaw: float, direction: float) -> int:
    """Face whose inward normal is closest to ``direction``."""
    return min(range(4), key=lambda f: abs(wrap_angle(skill.inward_normal(yaw, f) - direction)))


def at(obs: Observation, x: float, y: float, yaw: float = 0.0) -> Observation:
    """The same observation with the block moved (test input for the planner only)."""
    c, s = math.cos(yaw / 2), math.sin(yaw / 2)
    return dataclasses.replace(obs, block_pos=np.array([x, y, obs.block_pos[2]]),
                               block_quat=np.array([c, 0.0, 0.0, s]), block_yaw=yaw)


# ----- joint targets: range and speed limits -----


def test_send_limits_joint_speed(skill: PushSkill, runner: ScenarioRunner, push: PushConfig) -> None:
    plant = FakePlant(runner.reset(DEV_SEED))
    skill.q_cmd = skill.kin.home_q.copy()
    target = skill.kin.home_q + 1.0
    step = np.asarray(push.controller.max_joint_speed) * plant.control_dt
    previous = skill.q_cmd.copy()
    for _ in range(5):
        skill._send(plant, target)
        assert np.all(np.abs(skill.q_cmd - previous) <= step + 1e-12)
        # The servos get the limited command plus the arm's static gravity offset.
        sent = plant.commands[-1][0]
        np.testing.assert_array_equal(sent, skill.q_cmd + skill.kin.gravity_offset(skill.q_cmd))
        previous = skill.q_cmd.copy()
    assert skill.stats.rate_limited_updates == 5
    assert skill.stats.control_updates == 5
    assert all(gripper == skill.gripper_closed for _, gripper in plant.commands)


def test_send_keeps_targets_inside_the_joint_ranges(skill: PushSkill, runner: ScenarioRunner,
                                                    push: PushConfig) -> None:
    plant = FakePlant(runner.reset(DEV_SEED))
    margin = push.controller.joint_margin
    lo, hi = skill.kin.joint_lo + margin, skill.kin.joint_hi - margin
    skill.q_cmd = skill.kin.home_q.copy()
    commanded = []
    for target in (skill.kin.joint_hi + 1.0, skill.kin.joint_lo - 1.0):
        for _ in range(400):  # long enough to reach either limit at the joint-speed limit
            skill._send(plant, target)
            commanded.append(skill.q_cmd.copy())
        np.testing.assert_allclose(skill.q_cmd, hi if target[0] > 0 else lo, atol=1e-12)
    for q, (sent, _) in zip(commanded, plant.commands, strict=True):
        assert np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12)
        # The gravity offset is far smaller than the margin, so servo targets stay inside the ranges.
        assert np.all(np.abs(sent - q) < 0.02)
        assert np.all(sent >= skill.kin.joint_lo) and np.all(sent <= skill.kin.joint_hi)


def test_hold_repeats_the_commanded_targets(skill: PushSkill, runner: ScenarioRunner) -> None:
    obs = runner.reset(DEV_SEED)
    plant = FakePlant(obs)
    skill.hold(plant, 0.1)
    assert len(plant.commands) == 5
    np.testing.assert_array_equal(skill.q_cmd, obs.arm_qpos_target)  # the reset pose is the command
    held = obs.arm_qpos_target + skill.kin.gravity_offset(obs.arm_qpos_target)
    for sent, _ in plant.commands:
        np.testing.assert_array_equal(sent, held)


def test_unit_profile_and_move_time() -> None:
    values = [unit_profile(t, 1.0, 0.2) for t in np.linspace(-0.1, 1.1, 121)]
    assert values[0] == 0.0 and values[-1] == 1.0
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    assert unit_profile(0.5, 1.0, 0.2) == pytest.approx(0.5)
    assert move_time(0.0, 0.1, 0.15) == 0.0
    assert move_time(0.3, 0.3, 0.15) == pytest.approx(1.15)  # cruise: distance / speed + ramp
    short = move_time(0.001, 0.3, 0.15)
    assert 0 < short < 0.15  # triangular profile


# ----- planning filters -----


def test_bounds_helpers(skill: PushSkill, push: PushConfig, cfg: SceneConfig) -> None:
    m = push.controller.table_margin
    assert skill.in_table(np.array([0.5, 0.0, 0.02]))
    assert not skill.in_table(np.array([cfg.table.x.high - m + 1e-3, 0.0, 0.02]))
    assert not skill.in_table(np.array([0.5, cfg.table.y.low + m - 1e-3, 0.02]))
    b = push.controller.block_margin
    assert skill.block_in_region(np.array([0.5, 0.0]))
    assert not skill.block_in_region(np.array([cfg.workspace.allowed_block_x.low + b - 1e-3, 0.0]))
    assert not skill.block_in_region(np.array([0.5, cfg.workspace.allowed_block_y.high - b + 1e-3]))


def test_plan_accepts_a_nominal_push(skill: PushSkill, runner: ScenarioRunner, push: PushConfig) -> None:
    obs = at(runner.reset(DEV_SEED), 0.50, 0.0, yaw=0.3)
    direction = 0.3  # along the normal of one face
    face = face_toward(skill, obs.block_yaw, direction)
    plan = skill.plan(obs, PushAction(face, direction, 0.08, 1.0))
    assert plan.feasible and plan.reason == "ok", plan.reason
    d = np.array([math.cos(direction), math.sin(direction)])
    ctl = push.controller
    contact_gap = skill.half_edge + ctl.pusher_half_thickness + ctl.standoff
    np.testing.assert_allclose(plan.pre_contact[:2], obs.block_pos[:2] - contact_gap * d, atol=1e-12)
    np.testing.assert_allclose(plan.stroke_end[:2] - plan.pre_contact[:2], 0.08 * d, atol=1e-12)
    assert plan.pre_contact[2] == plan.stroke_end[2] == ctl.push_height
    assert plan.predicted_displacement == pytest.approx(0.08 - ctl.standoff)
    np.testing.assert_allclose(plan.predicted_block_end, obs.block_pos[:2] + plan.predicted_displacement * d)
    assert plan.min_joint_margin >= ctl.joint_margin - 1e-12
    assert abs(wrap_angle(plan.hand_yaw - skill.inward_normal(obs.block_yaw, face))) in (
        pytest.approx(0.0, abs=1e-12), pytest.approx(math.pi, abs=1e-12))


def test_plan_rejects_inadmissible_actions(skill: PushSkill, runner: ScenarioRunner) -> None:
    obs = at(runner.reset(DEV_SEED), 0.50, 0.0)
    face = face_toward(skill, 0.0, 0.0)
    assert skill.plan(obs, PushAction(face, 0.0, 0.05, 1.0)).reason == "speed/duration not admissible"
    assert skill.plan(obs, PushAction(face, 0.0, 0.08, 2.0)).reason == "speed/duration not admissible"
    assert skill.plan(obs, PushAction(7, 0.0, 0.08, 1.0)).reason == "invalid face"
    off = skill.plan(obs, PushAction(face, math.radians(25.0), 0.08, 1.0))
    assert not off.feasible and "off the face normal" in off.reason
    edge = skill.plan(obs, PushAction(face, math.radians(19.0), 0.08, 1.0))
    assert edge.feasible, edge.reason


def test_plan_rejects_setpoints_off_the_table(skill: PushSkill, runner: ScenarioRunner) -> None:
    # Pushing toward -y from a block near the +y table edge needs a pre-contact point off the table.
    obs = at(runner.reset(DEV_SEED), 0.50, 0.40)
    direction = -math.pi / 2
    plan = skill.plan(obs, PushAction(face_toward(skill, 0.0, direction), direction, 0.04, 0.5))
    assert not plan.feasible and plan.reason == "pre-contact TCP outside table bounds"


def test_plan_rejects_a_block_end_outside_the_allowed_region(skill: PushSkill, runner: ScenarioRunner) -> None:
    obs = at(runner.reset(DEV_SEED), 0.50, 0.30)
    direction = math.pi / 2
    plan = skill.plan(obs, PushAction(face_toward(skill, 0.0, direction), direction, 0.12, 1.5))
    assert not plan.feasible and plan.reason == "predicted block end outside the allowed region"
    short = skill.plan(obs, PushAction(face_toward(skill, 0.0, direction), direction, 0.04, 0.5))
    assert short.feasible, short.reason


def test_plan_rejects_unreachable_paths(skill: PushSkill, runner: ScenarioRunner,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    obs = at(runner.reset(DEV_SEED), 0.50, 0.0)
    q = skill.kin.home_q
    monkeypatch.setattr(skill, "ik", lambda *a, **k: IKResult(q, 0.01, 0.0, 60, False, 0.1))
    plan = skill.plan(obs, PushAction(face_toward(skill, 0.0, 0.0), 0.0, 0.08, 1.0))
    assert not plan.feasible and plan.reason.startswith("unreachable")


def test_execute_refuses_an_infeasible_plan(skill: PushSkill, runner: ScenarioRunner) -> None:
    obs = at(runner.reset(DEV_SEED), 0.50, 0.0)
    plan = skill.plan(obs, PushAction(0, 0.0, 0.05, 1.0))
    with pytest.raises(ValueError, match="not feasible"):
        skill.execute(FakePlant(obs), obs, plan)


def test_baseline_candidates_face_the_target(cfg: SceneConfig, push: PushConfig, skill: PushSkill,
                                             runner: ScenarioRunner) -> None:
    obs = runner.reset(DEV_SEED)
    policy = FeedbackBaseline(cfg, push, skill)
    delta = obs.target_pos[:2] - obs.block_pos[:2]
    wanted = math.atan2(delta[1], delta[0])
    actions = policy.candidates(obs)
    assert actions and len(actions) % (len(push.action.speeds) * len(push.action.durations)) == 0
    for a in actions:
        assert abs(wrap_angle(a.direction - wanted)) < math.pi / 2
        assert abs(wrap_angle(a.direction - skill.inward_normal(obs.block_yaw, a.face))) <= (
            push.controller.max_off_normal + 1e-9)
    plan, report = policy.choose(obs)
    assert plan is not None and plan.feasible
    assert len(report) == len(actions)


# ----- execution through the actuators -----


def test_one_push_moves_the_block_through_finger_contact_only(
    cfg: SceneConfig, push: PushConfig, skill: PushSkill, runner: ScenarioRunner
) -> None:
    obs = runner.reset(DEV_SEED)
    policy = FeedbackBaseline(cfg, push, skill)
    plan, _ = policy.choose(obs)
    assert plan is not None
    judge = EpisodeJudge(runner, push)
    plant = EvaluatedPlant(runner, judge, push)
    before = obs.block_pos[:2].copy()
    plant.note("push_start", index=1, plan=plan.summary(), candidates=[])
    end = skill.execute(plant, obs, plan)
    plant.note("push_end", index=1, aborted=None)
    end = skill.hold(plant, 1.0, until=lambda o: o.block_planar_speed < push.controller.settle_speed)

    moved = end.block_pos[:2] - before
    d = np.array([math.cos(plan.action.direction), math.sin(plan.action.direction)])
    assert float(moved @ d) > 0.5 * plan.predicted_displacement  # the block went the planned way
    assert end.block_target_distance < obs.block_target_distance

    assert judge.outcome is None or judge.outcome.success, judge.outcome
    kinds = {e.kind for e in judge.events}
    assert kinds == {"robot-block"}
    assert all(not e.forbidden and e.push_index == 1 for e in judge.events)
    assert {e.body_a for e in judge.events} <= set(push.limits.pusher_bodies)
    assert judge.stats["min_tcp_z"] > push.limits.min_tcp_z
    assert judge.stats["max_joint_speed_ratio"] < 1.0

    # Retreat ends at the transit height, clear of the block and with the gripper closed.
    tcp, _ = skill.kin.forward(skill.q_cmd)
    assert tcp[2] == pytest.approx(push.controller.safe_height, abs=1e-3)
    # The measured TCP trails the command (position-servo lag) but is well above the block.
    assert end.tcp_pos[2] > push.controller.safe_height - 0.02 > 2 * skill.half_edge
    assert not end.robot_block_contact
    assert end.gripper_opening < 0.002
    phases = [row[2] for row in plant.trajectory]
    for phase in ("approach", "descend", "stroke", "dwell", "back-off", "retreat"):
        assert phase in phases
    assert skill.stats.max_ik_position_error <= push.controller.ik_position_tolerance


def test_executed_tcp_holds_the_push_height(cfg: SceneConfig, push: PushConfig, skill: PushSkill,
                                            runner: ScenarioRunner) -> None:
    """The measured TCP, not only the command, stays at the push height through the stroke (D25)."""
    obs = runner.reset(DEV_SEED)
    plan, _ = FeedbackBaseline(cfg, push, skill).choose(obs)
    assert plan is not None
    judge = EpisodeJudge(runner, push)
    plant = EvaluatedPlant(runner, judge, push)
    plant.note("push_start", index=1, plan=plan.summary(), candidates=[])
    skill.execute(plant, obs, plan)
    plant.note("push_end", index=1, aborted=None)
    column = TRAJECTORY_COLUMNS.index
    height = push.controller.push_height
    # The stroke starts while the TCP is still settling from above after the descent (servo lag,
    # ~5 mm); the standoff lets it settle before contact, so check the rows that touch the block.
    contact = [row for row in plant.trajectory if row[column("robot_block_contact")]]
    assert contact and all(row[column("phase")] in ("stroke", "dwell", "back-off") for row in contact)
    z = np.array([row[column("tcp_z")] for row in contact])
    assert np.all(np.abs(z - height) <= 0.002), (z.min(), z.max())
    assert min(row[column("tcp_z")] for row in plant.trajectory) >= height - 0.002


def test_recover_rises_to_the_transit_height(cfg: SceneConfig, push: PushConfig, skill: PushSkill,
                                              runner: ScenarioRunner) -> None:
    obs = runner.reset(DEV_SEED)
    low = np.array([0.40, -0.30, 0.05])  # away from the block
    result = skill.ik(low, 0.0, skill.kin.home_q, iterations=200)
    assert result.converged
    judge = EpisodeJudge(runner, push)
    plant = EvaluatedPlant(runner, judge, push)
    skill.q_cmd = np.array(obs.arm_qpos_target)
    for _ in range(150):
        skill._send(plant, result.q)
    skill.recover(plant)
    tcp, _ = skill.kin.forward(skill.q_cmd)
    np.testing.assert_allclose(tcp, [low[0], low[1], push.controller.safe_height], atol=1e-3)
    assert judge.outcome is None and not judge.events
