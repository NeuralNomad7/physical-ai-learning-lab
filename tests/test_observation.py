"""The agent-facing observation boundary: an explicit allow-list, plain copies, no hidden physics."""

from __future__ import annotations

import dataclasses
import json

import mujoco
import numpy as np
import pytest

from physical_ai_lab.config import SceneConfig
from physical_ai_lab.observation import OBSERVATION_FIELDS, POSE_SOURCE, Observation
from physical_ai_lab.scenario import HiddenPhysics, ScenarioRunner

# Reviewed allow-list. Adding a field is a deliberate change to the agent boundary: update this tuple.
EXPECTED_FIELDS = (
    "time",
    "control_step",
    "pose_source",
    "arm_qpos",
    "arm_qvel",
    "arm_qpos_target",
    "gripper_opening",
    "tcp_pos",
    "tcp_quat",
    "block_pos",
    "block_quat",
    "block_yaw",
    "block_linvel",
    "block_angvel",
    "target_pos",
    "target_tolerance",
    "robot_block_contact",
    "robot_table_contact",
)
FORBIDDEN_WORDS = ("mass", "friction", "inertia", "seed", "truth", "physics", "model", "data", "future")
SCALAR_TYPES = {
    "time": float,
    "control_step": int,
    "pose_source": str,
    "gripper_opening": float,
    "block_yaw": float,
    "target_tolerance": float,
    "robot_block_contact": bool,
    "robot_table_contact": bool,
}


def test_fields_are_the_reviewed_allow_list() -> None:
    assert OBSERVATION_FIELDS == EXPECTED_FIELDS
    for name in OBSERVATION_FIELDS:
        assert not any(word in name for word in FORBIDDEN_WORDS), name


def test_observation_holds_plain_read_only_copies(runner: ScenarioRunner) -> None:
    obs = runner.reset(0)
    assert set(vars(obs)) == set(OBSERVATION_FIELDS)
    assert obs.pose_source == POSE_SOURCE == "simulator_state"
    for name in OBSERVATION_FIELDS:
        value = getattr(obs, name)
        assert not isinstance(value, (mujoco.MjModel, mujoco.MjData)), name
        if name in SCALAR_TYPES:
            assert type(value) is SCALAR_TYPES[name], name
        else:
            assert isinstance(value, np.ndarray) and value.dtype == np.float64 and value.ndim == 1, name
            assert value.flags.owndata and not value.flags.writeable, name
    assert obs.arm_qpos.shape == obs.arm_qvel.shape == obs.arm_qpos_target.shape == (7,)


def test_observation_is_a_snapshot(runner: ScenarioRunner) -> None:
    obs = runner.reset(0)
    before = obs.to_dict()
    runner.step(10)
    assert obs.to_dict() == before
    with pytest.raises(ValueError):
        obs.block_pos[0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.time = 1.0  # type: ignore[misc]


def test_json_round_trip_is_exact(runner: ScenarioRunner) -> None:
    runner.reset(1)
    obs = runner.step(25)
    payload = json.loads(obs.to_json())
    assert tuple(payload) == OBSERVATION_FIELDS
    assert Observation.from_dict(payload).max_abs_difference(obs) == 0.0


def test_unknown_or_missing_keys_are_rejected(runner: ScenarioRunner) -> None:
    payload = runner.reset(0).to_dict()
    with pytest.raises(ValueError, match=r"unknown=\['block_mass'\]"):
        Observation.from_dict({**payload, "block_mass": 0.25})
    del payload["target_tolerance"]
    with pytest.raises(ValueError, match=r"missing=\['target_tolerance'\]"):
        Observation.from_dict(payload)


def test_hidden_physics_is_not_observable_at_rest(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    """Episodes that share a seed but differ in hidden mass/friction look the same until something moves the block."""
    p = cfg.physics
    extremes = [
        HiddenPhysics(mass, mu)
        for mass in (p.mass_range.low, p.mass_range.high)
        for mu in (p.sliding_friction_range.low, p.sliding_friction_range.high)
    ]
    for seed in (0, 5):
        at_reset = runner.reset(seed)
        settled = runner.step(50)
        for physics in extremes:
            assert runner.reset(seed, physics=physics).max_abs_difference(at_reset) == 0.0
            assert runner.step(50).max_abs_difference(settled) <= cfg.checks.observation_rest_tolerance
