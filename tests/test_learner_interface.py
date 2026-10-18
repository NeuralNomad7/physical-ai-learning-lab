"""Phase 3 learner-facing interface: what crosses the boundary, and that nothing hidden does.

The agent (the baseline now, the Phase 4 learner later) runs in its own process and receives only
JSON lines validated against an allow-list (learner_interface.TO_AGENT). These tests check the
allow-lists, the agent-side object, the modules the agent process loads, every message of real
scratch episodes (traffic scan), the learner part of the episode records and the transitions.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys

import numpy as np
import pytest

from physical_ai_lab import learner_interface as li
from physical_ai_lab import paths
from physical_ai_lab.benchmark import AgentConnection, TrafficScan, run_manifest_episode, scenario_secrets
from physical_ai_lab.episode_schema import (
    HIDDEN_KEYS,
    LEARNER_KEYS,
    SchemaError,
    learner_view,
    validate_learner_view,
    validate_transition,
    walk_keys,
    walk_values,
)
from physical_ai_lab.manifest import open_split
from physical_ai_lab.observation import OBSERVATION_FIELDS, Observation

EVALUATOR_MODULES = ("benchmark", "manifest", "evaluation", "instrumentation", "episode_schema", "contact_checks",
                     "benchmark_config", "smoke", "cli")


@pytest.fixture(scope="module")
def direct_episode(cfg, push_cfg, scratch_manifests, tmp_path_factory):
    """One scratch validation episode run directly, keeping its transitions and step log."""
    from physical_ai_lab.scenario import ScenarioRunner

    bench, manifest_dir, freeze_dir = scratch_manifests
    manifest = open_split("validation", manifest_dir=manifest_dir, freeze_dir=freeze_dir, bench=bench)
    scenario = manifest.scenarios[1]
    runner = ScenarioRunner(cfg)
    transitions: list[dict] = []
    commands = tmp_path_factory.mktemp("direct") / "ep.npz"
    conn = AgentConnection("baseline", reply_timeout=bench.agent.reply_timeout_seconds)
    try:
        agent_pid = conn.proc.pid
        record, steps = run_manifest_episode(runner, conn, manifest, scenario, episode=1, push=push_cfg, bench=bench,
                                             agent="baseline", commands_path=commands, transitions=transitions)
    finally:
        exit_code = conn.close()
    return {"record": record, "steps": steps, "transitions": transitions, "manifest": manifest, "scenario": scenario,
            "runner": runner, "commands": commands, "exit_code": exit_code, "agent_pid": agent_pid}


# --------------------------------------------------------------------------- allow-lists


def test_messages_to_the_agent_have_no_hidden_keys():
    for kind, keys in li.TO_AGENT.items():
        assert not keys & HIDDEN_KEYS, kind
    assert not li.BUDGET_KEYS & HIDDEN_KEYS
    assert not set(OBSERVATION_FIELDS) & HIDDEN_KEYS
    assert li.TO_AGENT["episode_over"] == {"type"}  # no outcome, reason or truth at the end of an episode
    assert len(OBSERVATION_FIELDS) == 18


def test_learner_record_keys_are_not_hidden():
    assert not LEARNER_KEYS & HIDDEN_KEYS


def _start(obs: Observation) -> dict:
    return {"type": "episode_start", "protocol": li.PROTOCOL_VERSION, "episode": 1, "control_dt": 0.02,
            "budget": {"max_pushes": 8, "max_seconds": 30.0}, "observation": obs.to_dict()}


@pytest.mark.parametrize("extra", ["block_mass", "sliding_friction", "scenario_id", "split", "cohort", "seed",
                                   "outcome", "future_observation"])
def test_extra_keys_to_the_agent_are_rejected(extra, runner):
    runner.reset(0)
    obs = runner.observe()
    li.validate_to_agent(_start(obs))
    with pytest.raises(li.ProtocolError):
        li.validate_to_agent({**_start(obs), extra: 0.25})
    with pytest.raises(li.ProtocolError):
        li.validate_to_agent({"type": "observation", "observation": obs.to_dict(), extra: 1})
    with pytest.raises(li.ProtocolError):
        li.validate_to_agent({"type": "episode_over", extra: "x"})
    bad_obs = {**obs.to_dict(), extra: 0.25}
    with pytest.raises(li.ProtocolError):
        li.validate_to_agent({"type": "observation", "observation": bad_obs})
    bad_budget = {**_start(obs), "budget": {"max_pushes": 8, "max_seconds": 30.0, extra: 1}}
    with pytest.raises(li.ProtocolError):
        li.validate_to_agent(bad_budget)


def test_unknown_message_types_are_rejected_both_ways():
    for validate in (li.validate_to_agent, li.validate_from_agent):
        with pytest.raises(li.ProtocolError):
            validate({"type": "hidden_truth"})


def test_agent_messages_are_validated():
    li.validate_from_agent({"type": "command", "arm_q": [0.0] * 7, "gripper": 0.0})
    for bad in ({"type": "command", "arm_q": [0.0] * 6, "gripper": 0.0},
                {"type": "command", "arm_q": [0.0] * 7, "gripper": float("nan")},
                {"type": "command", "arm_q": [0.0] * 7, "gripper": 0.0, "physics": 1},
                {"type": "note", "event": "outcome", "info": {}},
                {"type": "note", "event": "push_start", "info": {"index": 1, "probe": "yes"}},
                {"type": "ready", "protocol": 99, "agent": "baseline"}):
        with pytest.raises(li.ProtocolError):
            li.validate_from_agent(bad)


def test_only_plain_json_values_cross():
    with pytest.raises(li.ProtocolError):
        li.encode({"type": "ack", "obj": object()})
    data = li.encode({"type": "observation", "x": np.float64(0.5), "v": np.arange(3.0)})
    assert json.loads(data) == {"type": "observation", "x": 0.5, "v": [0.0, 1.0, 2.0]}


# --------------------------------------------------------------------------- agent-side object


def test_agent_plant_holds_no_simulator_or_evaluator_object(runner):
    runner.reset(0)
    plant = li.AgentPlant(_start(runner.observe()), io.BytesIO(), io.BytesIO())
    assert li.AgentPlant.__slots__ == ("control_dt", "budget", "_obs", "_reader", "_writer")
    assert not hasattr(plant, "__dict__")
    held = [getattr(plant, name) for name in li.AgentPlant.__slots__]
    assert [type(v).__name__ for v in held] == ["float", "dict", "Observation", "BytesIO", "BytesIO"]
    assert set(plant.budget) == li.BUDGET_KEYS
    # The observation is a fresh object decoded from JSON, not the runner's.
    obs = plant.observe()
    assert obs is not runner.observe() and set(obs.to_dict()) == set(OBSERVATION_FIELDS)
    for name in ("hidden_truth", "applied_physics", "sim_model", "sim_data", "_runner", "runner", "model", "data"):
        assert not hasattr(plant, name)


def test_agent_process_loads_no_evaluator_module():
    code = ("import json, sys\n"
            "from physical_ai_lab import agent_process\n"
            "from physical_ai_lab.learner_interface import make_agent\n"
            "make_agent('baseline')\n"
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith('physical_ai_lab'))))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         cwd=str(paths.PROJECT_ROOT), check=True)
    loaded = {m.split(".", 1)[1] for m in json.loads(out.stdout.strip().splitlines()[-1]) if "." in m}
    assert not loaded & set(EVALUATOR_MODULES), sorted(loaded)
    assert "learner_interface" in loaded and "baseline" in loaded


def test_agent_side_code_never_names_hidden_accessors():
    # Code (not docstrings) of the modules the agent process loads. scenario.py is imported only for
    # kinematics.verify_actuators (a static check on the agent's own nominal model) and is excluded.
    import ast

    names = {"hidden_truth", "applied_physics", "sim_model", "sim_data", "ScenarioRunner", "open_split",
             "block_mass", "sliding_friction", "read_sealed_for_audit"}
    for module in ("agent_process", "learner_interface", "baseline", "controller", "kinematics", "observation"):
        tree = ast.parse((paths.PACKAGE_DIR / f"{module}.py").read_text(encoding="utf-8"))
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
               {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not used & names, (module, sorted(used & names))


# --------------------------------------------------------------------------- traffic of real episodes


def test_traffic_scan_catches_injected_values():
    scan = TrafficScan()
    scan.arm({"s0123456789ab", "out_of_range"}, {0.3125, 0.47})
    scan({"type": "observation", "observation": {"x": 0.3125}}, b'{"x":0.3125}')
    scan({"type": "ack"}, b'{"type":"ack","id":"s0123456789ab"}')
    scan({"type": "ack"}, b'{"type":"ack"}')
    report = scan.report()
    assert report["messages"] == 3 and len(report["hits"]) == 2


def test_direct_episode_crosses_cleanly(direct_episode):
    rec = direct_episode["record"]
    scan = rec["evaluator"]["traffic_scan"]
    assert scan["hits"] == [] and scan["messages"] > 100
    assert scan["messages"] == rec["learner"]["observations_received"] + len(rec["learner"]["notes"]) + \
        (1 if rec["learner"]["end"]["kind"] == "evaluator_ended" else 0)
    assert direct_episode["exit_code"] == 0
    import os

    assert direct_episode["agent_pid"] != os.getpid()


def test_learner_views_hold_no_hidden_key_identifier_or_value(direct_episode, scratch_run):
    runner = direct_episode["runner"]
    records = [direct_episode["record"], *scratch_run["records"]]
    manifest = direct_episode["manifest"]
    by_id = {s.scenario_id: s for s in manifest.scenarios}
    for rec in records:
        view = learner_view(rec)
        validate_learner_view(view, frozenset(str(rec["evaluator"][k]) for k in
                                              ("scenario_id", "condition_id", "split", "cohort", "band")))
        assert not walk_keys(view) & HIDDEN_KEYS
        strings, numbers = scenario_secrets(manifest, by_id[rec["evaluator"]["scenario_id"]], runner)
        text = json.dumps(view)
        assert not any(s and s in text for s in strings)
        observed = [v for o in (view["first_observation"], view["last_observation"]) for v in walk_values(o)
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
        assert not {float(v) for v in observed} & numbers


def test_learner_view_validation_rejects_leaks(direct_episode):
    view = json.loads(json.dumps(learner_view(direct_episode["record"])))
    leaked = json.loads(json.dumps(view))
    leaked["notes"].append({"control_step": 0, "event": "decision", "info": {"block_mass": 0.3}})
    with pytest.raises(SchemaError, match="hidden keys"):
        validate_learner_view(leaked)
    sid = direct_episode["scenario"].scenario_id
    leaked = json.loads(json.dumps(view))
    leaked["notes"].append({"control_step": 0, "event": "decision", "info": {"decision": sid}})
    with pytest.raises(SchemaError, match="evaluator identifiers"):
        validate_learner_view(leaked, frozenset({sid}))
    leaked = json.loads(json.dumps(view))
    leaked["outcome"] = "success"
    with pytest.raises(SchemaError):
        validate_learner_view(leaked)


def test_observations_are_never_from_the_future(direct_episode):
    rec, steps = direct_episode["record"], direct_episode["steps"]
    view = rec["learner"]
    # One observation at the start and one after each command, except when the evaluator ends the
    # episode during the agent's last command: then the agent gets ``episode_over`` (no observation,
    # no outcome). The last observation is never later than the state after the last command.
    ended_by_evaluator = view["end"]["kind"] == "evaluator_ended"
    assert view["observations_received"] == view["commands_sent"] + (0 if ended_by_evaluator else 1)
    assert view["last_observation"]["control_step"] == view["observations_received"] - 1
    assert view["first_observation"]["control_step"] == 0
    assert len(steps.ctrl) == view["commands_sent"]


def test_transitions_are_valid_and_carry_nothing_hidden(direct_episode):
    transitions, rec = direct_episode["transitions"], direct_episode["record"]
    view = rec["learner"]
    # A transition needs the next observation, so the last command of an episode the evaluator ended
    # has none (see test_observations_are_never_from_the_future).
    assert len(transitions) == view["observations_received"] - 1 > 0
    for t in transitions:
        validate_transition(t)
        assert not walk_keys(t) & HIDDEN_KEYS
    assert transitions[0]["observation"] == rec["learner"]["first_observation"]
    assert transitions[-1]["next_observation"] == rec["learner"]["last_observation"]
    for a, b in zip(transitions, transitions[1:]):
        assert a["next_observation"] == b["observation"]


def test_baseline_and_learner_share_the_same_budget(direct_episode, push_cfg):
    view = direct_episode["record"]["learner"]
    assert view["budget"] == {"max_pushes": push_cfg.episode.max_pushes, "max_seconds": push_cfg.episode.max_seconds}
    assert view["agent"] == "baseline"
