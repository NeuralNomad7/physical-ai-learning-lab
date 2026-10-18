"""Phase 3 episode / event / transition schema (episode_schema.py), on real scratch records."""

from __future__ import annotations

import copy

import pytest

from physical_ai_lab import episode_schema as es
from physical_ai_lab.evaluation import FAILURE_REASONS


@pytest.fixture()
def record(scratch_run):
    return copy.deepcopy(scratch_run["records"][0])


def test_scratch_records_validate_and_carry_versions(scratch_run, bench_cfg, push_cfg):
    from physical_ai_lab.push_config import PUSH_SCHEMA_VERSION

    assert len(scratch_run["records"]) == 2
    for rec in scratch_run["records"]:
        es.validate_episode_record(rec)
        b = rec["benchmark"]
        assert b["config_sha256"] == bench_cfg.sha256
        assert b["metrics_version"] == bench_cfg.metrics["version"]
        assert b["push_config_schema_version"] == PUSH_SCHEMA_VERSION == 1
        assert b["episode_schema_version"] == es.EPISODE_SCHEMA_VERSION
        assert set(rec) == es.TOP_KEYS


def test_records_count_probes_failures_and_limits(scratch_run):
    for rec in scratch_run["records"]:
        ev = rec["evaluator"]
        assert set(ev["counts"]) == es.COUNT_KEYS
        assert ev["counts"]["probe_pushes"] == ev["probe_pushes"] == rec["learner"]["probe_pushes"]
        assert ev["counts"]["failure"] == int(not ev["outcome"]["success"])
        assert ev["limits"]["passed"] is (ev["outcome"]["reason"] != "limit_violation")
        types = [e["type"] for e in ev["events"]]
        assert types[-2:] == ["limit_check", "outcome"]
        assert types.count("push_start") == len(rec["learner"]["pushes"]) == ev["pushes_declared"]
        assert ev["pushes_used"] == ev["pushes_declared"] + ev["unplanned_robot_block_contacts"]
        contacts = [e for e in ev["events"] if e["type"] == "contact"]
        assert len(contacts) == ev["counts"]["contact_events"]


@pytest.mark.parametrize("mutate, match", [
    (lambda r: r.update(episode_schema_version=99), "episode_schema_version"),
    (lambda r: r.update(extra=1), "keys"),
    (lambda r: r["benchmark"].pop("metrics_version"), "benchmark"),
    (lambda r: r["evaluator"].pop("traffic_scan"), "evaluator"),
    (lambda r: r["evaluator"]["traffic_scan"].update(hits=["string 'x'"]), "traffic scan"),
    (lambda r: r["evaluator"]["outcome"].update(success=False, reason="lost"), "outcome.reason"),
    (lambda r: r["evaluator"]["counts"].update(failure=1), "counts.failure"),
    (lambda r: r["evaluator"]["counts"].update(probe_pushes=3), "probe"),
    (lambda r: r["evaluator"].update(pushes_used=r["evaluator"]["pushes_used"] + 1), "pushes_used"),
    (lambda r: r["evaluator"]["events"].reverse(), "last event"),
    (lambda r: r["evaluator"]["events"].append(dict(r["evaluator"]["events"][-1])), "outcome"),
    (lambda r: r["evaluator"]["events"].insert(0, {"t": 0.0, "source": "agent", "type": "contact"}), "must come from"),
    (lambda r: r["evaluator"]["events"].insert(0, {"t": 0.0, "source": "agent", "type": "teleport"}), "unknown"),
    (lambda r: r["evaluator"].update(final_distance_m=float("nan")), "final_distance"),
    (lambda r: r["learner"]["budget"].update(block_mass=0.3), "hidden keys"),
    (lambda r: r["learner"]["notes"].append({"control_step": 1, "event": "decision",
                                             "info": {"decision": r["evaluator"]["condition_id"]}}), "identifiers"),
    (lambda r: r["learner"]["first_observation"].pop("block_pos"), "observation"),
    (lambda r: r["learner"]["end"].update(kind="won"), "end.kind"),
])
def test_invalid_records_are_rejected(record, mutate, match):
    mutate(record)
    with pytest.raises(es.SchemaError, match=match):
        es.validate_episode_record(record)


def test_failure_records_validate_with_any_reason(record):
    for reason in FAILURE_REASONS:
        r = copy.deepcopy(record)
        ev = r["evaluator"]
        ev["outcome"].update(success=False, reason=reason)
        ev["events"][-1].update(success=False, reason=reason)
        ev["counts"].update(failure=1, limit_violation=int(reason == "limit_violation"))
        ev["limits"]["passed"] = reason != "limit_violation"
        es.validate_episode_record(r)


def test_events_validate_individually():
    es.validate_event({"t": 1.0, "source": "agent", "type": "push_start", "index": 1, "probe": True, "plan": None}, 0)
    es.validate_event({"t": 2.0, "source": "evaluator", "type": "outcome", "success": True, "reason": "success",
                       "detail": ""}, 0)
    for bad in ({"t": 1.0, "source": "agent", "type": "push_start", "index": 1, "probe": "no", "plan": None},
                {"t": 1.0, "source": "agent", "type": "push_start", "index": 1},
                {"t": float("inf"), "source": "agent", "type": "phase", "phase": "approach"},
                {"t": 1.0, "source": "evaluator", "type": "outcome", "success": True, "reason": "timeout",
                 "detail": ""}):
        with pytest.raises(es.SchemaError):
            es.validate_event(bad, 0)


def test_actions_and_transitions(record):
    obs = record["learner"]["first_observation"]
    nxt = dict(obs, control_step=obs["control_step"] + 1)
    action = {"arm_q": [0.0] * 7, "gripper": 0.0}
    es.validate_transition({"observation": obs, "action": action, "next_observation": nxt})
    for bad in ({"observation": obs, "action": action, "next_observation": obs},  # not the next update
                {"observation": obs, "action": {"arm_q": [0.0] * 7}, "next_observation": nxt},
                {"observation": obs, "action": {**action, "mass": 1.0}, "next_observation": nxt},
                {"observation": obs, "action": action, "next_observation": nxt, "reward": 1.0},
                {"observation": {**obs, "block_mass": 0.25}, "action": action, "next_observation": nxt}):
        with pytest.raises(es.SchemaError):
            es.validate_transition(bad)


def test_hidden_keys_cover_the_evaluator_secrets():
    must_hide = {"block_mass", "sliding_friction", "inertia", "scenario_id", "condition_id", "split", "cohort", "band",
                 "reset_seed", "seed", "layout", "physics", "applied_physics", "manifest_sha256", "outcome", "success"}
    assert must_hide <= es.HIDDEN_KEYS
    assert not es.LEARNER_KEYS & es.HIDDEN_KEYS
    assert {"scenario_id", "condition_id", "split", "cohort", "band", "physics", "applied_physics"} <= es.EVALUATOR_KEYS
