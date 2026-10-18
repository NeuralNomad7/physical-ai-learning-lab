"""Versioned episode / event / transition schema (Phase 3; BUILD_PLAN.md sections 3 and 6).

An episode record has three parts, kept apart on purpose:

* ``learner``: only what crossed the learner interface during the episode (learner_interface.py):
  the agent name, protocol version, episode ordinal, control period, budget, the agent's own notes
  (phases, decisions, push start/end with the chosen action, probe flags), the pushes it declared,
  how its side of the episode ended, and the first and last observation it received. Nothing in it
  is hidden from the agent. ``learner_view(record)`` returns exactly this part, and
  ``validate_learner_view`` rejects any hidden key or evaluator identifier inside it.
* ``evaluator``: hidden truth and judgement, never sent to an agent: scenario / condition ids,
  split, cohort, band, reset seed, mass and friction (requested and read back), the layout, the
  outcome, pushes as judged (declared + unplanned contacts), the contact events, limit checks,
  contact metrics (instrumentation.py), event timeline, counts, compute time, replay digests and
  the traffic scan of every message sent to the agent.
* ``benchmark``: versions and hashes (benchmark version, config SHA-256, metrics version, manifest
  SHA-256, push config schema version, this schema's version).

Events (``evaluator.events``), each with ``t`` (simulated seconds), ``source`` (agent | evaluator)
and ``type``:

* agent: ``phase``, ``decision``, ``push_start`` (index, probe, plan), ``push_end`` (index, aborted),
  ``agent_stopped`` (reason), ``agent_error`` (error);
* evaluator: ``contact`` (kind, bodies, start, end, push index, forbidden, max normal force),
  ``limit_check`` (largest joint-speed ratio, joint-range excess, lowest TCP height, passed),
  ``outcome`` (success, reason, detail).

A transition (``TRANSITION_FIELDS``) is one control update as the learner sees it: the observation
it acted on, the action it sent (7 arm joint targets and the gripper input) and the next
observation it received. Transitions are built from learner-interface traffic only.
"""

from __future__ import annotations

import math
from typing import Any

from .evaluation import FAILURE_REASONS, SUCCESS
from .instrumentation import ContactSummaryMetrics
from .learner_interface import AGENT_NOTE_EVENTS, BUDGET_KEYS, N_ARM, PROTOCOL_VERSION
from .observation import OBSERVATION_FIELDS, Observation

EPISODE_SCHEMA_VERSION = 1

TOP_KEYS = frozenset({"record", "episode_schema_version", "benchmark", "learner", "evaluator"})
BENCHMARK_KEYS = frozenset({"benchmark_version", "config_sha256", "metrics_version", "manifest_sha256",
                            "push_config_schema_version", "episode_schema_version"})
LEARNER_KEYS = frozenset({"agent", "protocol_version", "episode", "control_dt", "budget", "observations_received",
                          "commands_sent", "notes", "pushes", "probe_pushes", "end", "first_observation",
                          "last_observation"})
LEARNER_END_KINDS = frozenset({"agent_stopped", "agent_error", "agent_timeout", "agent_died", "evaluator_ended"})
EVALUATOR_KEYS = frozenset({
    "episode_index", "scenario_id", "condition_id", "split", "cohort", "band", "reset_seed", "goal_sector", "physics",
    "applied_physics", "layout", "outcome", "pushes_declared", "unplanned_robot_block_contacts", "pushes_used",
    "probe_pushes", "final_distance_m", "initial", "final", "pushes", "events", "counts", "limits", "contact",
    "compute", "replay", "traffic_scan", "agent_traceback",
})
COUNT_KEYS = frozenset({"failure", "limit_violation", "forbidden_collision", "forbidden_contact_events",
                        "contact_events", "aborted_pushes", "probe_pushes", "push_budget_exceeded"})
LIMIT_KEYS = frozenset({"max_joint_speed_ratio", "max_joint_range_excess", "min_tcp_z", "physics_steps", "passed"})
COMPUTE_KEYS = frozenset({"wall_seconds", "agent_wait_seconds", "evaluator_seconds"})
REPLAY_KEYS = frozenset({"commands_file", "control_updates", "initial_state_digest", "final_state_digest"})
EVENT_TYPES = {
    "phase": ("agent", frozenset({"phase"})),
    "decision": ("agent", frozenset({"decision"})),
    "push_start": ("agent", frozenset({"index", "probe", "plan"})),
    "push_end": ("agent", frozenset({"index", "aborted"})),
    "agent_stopped": ("agent", frozenset({"reason"})),
    "agent_error": ("agent", frozenset({"error"})),
    "contact": ("evaluator", frozenset({"kind", "body_a", "body_b", "start", "end", "push_index", "forbidden",
                                        "max_normal_force"})),
    "limit_check": ("evaluator", frozenset({"max_joint_speed_ratio", "max_joint_range_excess", "min_tcp_z",
                                            "passed"})),
    "outcome": ("evaluator", frozenset({"success", "reason", "detail"})),
}
TRANSITION_FIELDS = ("observation", "action", "next_observation")
ACTION_KEYS = frozenset({"arm_q", "gripper"})

# Keys that name hidden or evaluator-only quantities. None may appear anywhere in a learner view.
HIDDEN_KEYS = frozenset({
    "block_mass", "mass", "sliding_friction", "friction", "inertia", "physics", "applied_physics",
    "scenario_id", "condition_id", "split", "cohort", "band", "seed", "reset_seed", "goal_sector", "layout",
    "manifest", "manifest_sha256", "physics_source", "truth", "hidden", "outcome", "success",
})


class SchemaError(ValueError):
    """A record does not match the episode schema."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise SchemaError(message)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _keys(obj: Any, expected: frozenset[str], where: str) -> None:
    _need(isinstance(obj, dict), f"{where} must be an object")
    _need(set(obj) == set(expected), f"{where}: keys {sorted(set(obj) ^ set(expected))} differ from the schema")


def _observation(obj: Any, where: str) -> None:
    _need(isinstance(obj, dict) and set(obj) == set(OBSERVATION_FIELDS), f"{where} must be an observation")
    try:
        Observation.from_dict(obj)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{where}: {exc}") from None


def walk_keys(obj: Any) -> set[str]:
    """Every dictionary key at any depth."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.add(str(key))
            found |= walk_keys(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found |= walk_keys(value)
    return found


def walk_values(obj: Any) -> list[Any]:
    """Every scalar value at any depth."""
    if isinstance(obj, dict):
        return [v for value in obj.values() for v in walk_values(value)]
    if isinstance(obj, (list, tuple)):
        return [v for value in obj for v in walk_values(value)]
    return [obj]


# --------------------------------------------------------------------------- learner part


def validate_action(action: Any, where: str = "action") -> None:
    _keys(action, ACTION_KEYS, where)
    arm_q = action["arm_q"]
    _need(isinstance(arm_q, list) and len(arm_q) == N_ARM and all(_finite(v) for v in arm_q),
          f"{where}.arm_q must be {N_ARM} finite numbers")
    _need(_finite(action["gripper"]), f"{where}.gripper must be finite")


def validate_transition(transition: Any) -> None:
    _need(isinstance(transition, dict) and set(transition) == set(TRANSITION_FIELDS), "transition keys differ")
    _observation(transition["observation"], "transition.observation")
    validate_action(transition["action"], "transition.action")
    _observation(transition["next_observation"], "transition.next_observation")
    _need(transition["next_observation"]["control_step"] == transition["observation"]["control_step"] + 1,
          "transition.next_observation must be the very next control update")


def validate_learner_view(view: Any, forbidden_strings: frozenset[str] = frozenset()) -> None:
    """Schema check of the learner part, plus: no hidden key and no evaluator identifier anywhere in it."""
    _keys(view, LEARNER_KEYS, "learner")
    hidden = walk_keys(view) & HIDDEN_KEYS
    _need(not hidden, f"learner view contains hidden keys {sorted(hidden)}")
    if forbidden_strings:
        leaked = {v for v in walk_values(view) if isinstance(v, str) and v in forbidden_strings}
        _need(not leaked, f"learner view contains evaluator identifiers {sorted(leaked)}")
    _need(isinstance(view["agent"], str) and view["protocol_version"] == PROTOCOL_VERSION, "learner agent/protocol")
    _need(isinstance(view["episode"], int) and view["episode"] >= 1, "learner.episode must be a positive ordinal")
    _need(_finite(view["control_dt"]) and view["control_dt"] > 0, "learner.control_dt")
    _keys(view["budget"], BUDGET_KEYS, "learner.budget")
    for key in ("observations_received", "commands_sent", "probe_pushes"):
        _need(isinstance(view[key], int) and view[key] >= 0, f"learner.{key} must be a count")
    _need(isinstance(view["notes"], list), "learner.notes must be a list")
    for i, note in enumerate(view["notes"]):
        _keys(note, frozenset({"control_step", "event", "info"}), f"learner.notes[{i}]")
        _need(note["event"] in AGENT_NOTE_EVENTS, f"learner.notes[{i}].event {note['event']!r}")
    _need(isinstance(view["pushes"], list), "learner.pushes must be a list")
    for i, push in enumerate(view["pushes"]):
        _keys(push, frozenset({"index", "probe", "start_step", "end_step", "aborted", "plan"}), f"learner.pushes[{i}]")
        _need(isinstance(push["probe"], bool), f"learner.pushes[{i}].probe must be true or false")
    _need(view["probe_pushes"] == sum(1 for p in view["pushes"] if p["probe"]), "learner.probe_pushes != probe pushes")
    _keys(view["end"], frozenset({"kind", "reason"}), "learner.end")
    _need(view["end"]["kind"] in LEARNER_END_KINDS, f"learner.end.kind {view['end']['kind']!r}")
    _observation(view["first_observation"], "learner.first_observation")
    _observation(view["last_observation"], "learner.last_observation")


def learner_view(record: dict[str, Any]) -> dict[str, Any]:
    """The part of an episode record a learner may read (a deep copy is not needed: it is JSON data)."""
    return record["learner"]


# --------------------------------------------------------------------------- evaluator part


def validate_event(event: Any, i: int) -> None:
    _need(isinstance(event, dict), f"events[{i}] must be an object")
    kind = event.get("type")
    _need(kind in EVENT_TYPES, f"events[{i}].type {kind!r} unknown")
    source, required = EVENT_TYPES[kind]
    _need(event.get("source") == source, f"events[{i}] ({kind}) must come from {source}")
    _need(_finite(event.get("t")), f"events[{i}].t must be finite")
    missing = required - set(event)
    _need(not missing, f"events[{i}] ({kind}) lacks {sorted(missing)}")
    if kind == "push_start":
        _need(isinstance(event["probe"], bool), f"events[{i}].probe must be true or false")
    if kind == "outcome":
        _need(event["reason"] == SUCCESS if event["success"] else event["reason"] in FAILURE_REASONS,
              f"events[{i}]: outcome reason {event['reason']!r}")


def validate_episode_record(record: Any) -> None:
    """Full schema check of one episode record (raises SchemaError)."""
    _keys(record, TOP_KEYS, "record")
    _need(record["record"] == "episode", "record must be 'episode'")
    _need(record["episode_schema_version"] == EPISODE_SCHEMA_VERSION,
          f"episode_schema_version {record['episode_schema_version']!r} != {EPISODE_SCHEMA_VERSION}")
    _keys(record["benchmark"], BENCHMARK_KEYS, "benchmark")
    ev = record["evaluator"]
    _keys(ev, EVALUATOR_KEYS, "evaluator")
    ids = frozenset(str(ev[k]) for k in ("scenario_id", "condition_id", "split", "cohort", "band"))
    validate_learner_view(record["learner"], forbidden_strings=ids)
    outcome = ev["outcome"]
    _keys(outcome, frozenset({"success", "reason", "detail", "time"}), "evaluator.outcome")
    _need(outcome["reason"] == SUCCESS if outcome["success"] else outcome["reason"] in FAILURE_REASONS,
          f"evaluator.outcome.reason {outcome['reason']!r}")
    _need(ev["pushes_used"] == ev["pushes_declared"] + ev["unplanned_robot_block_contacts"], "pushes_used mismatch")
    _need(ev["probe_pushes"] == record["learner"]["probe_pushes"], "probe count mismatch")
    _keys(ev["counts"], COUNT_KEYS, "evaluator.counts")
    _need(all(isinstance(v, int) and v >= 0 for v in ev["counts"].values()), "counts must be non-negative integers")
    _need(ev["counts"]["failure"] == (0 if outcome["success"] else 1), "counts.failure mismatch")
    _need(ev["counts"]["limit_violation"] == int(outcome["reason"] == "limit_violation"), "counts.limit_violation")
    _need(ev["counts"]["probe_pushes"] == ev["probe_pushes"], "counts.probe_pushes mismatch")
    _keys(ev["limits"], LIMIT_KEYS, "evaluator.limits")
    _keys(ev["contact"], frozenset(ContactSummaryMetrics.__dataclass_fields__), "evaluator.contact")
    _keys(ev["compute"], COMPUTE_KEYS, "evaluator.compute")
    _keys(ev["replay"], REPLAY_KEYS, "evaluator.replay")
    _need(_finite(ev["final_distance_m"]) and ev["final_distance_m"] >= 0, "final_distance_m")
    _need(isinstance(ev["events"], list) and ev["events"], "evaluator.events must be a non-empty list")
    for i, event in enumerate(ev["events"]):
        validate_event(event, i)
    _need(ev["events"][-1]["type"] == "outcome", "the last event must be the outcome")
    _need(sum(1 for e in ev["events"] if e["type"] == "outcome") == 1, "exactly one outcome event")
    times = [e["t"] for e in ev["events"]]
    _need(all(a <= b + 1e-12 for a, b in zip(times, times[1:])), "events must be in time order")
    scan = ev["traffic_scan"]
    _need(isinstance(scan, dict) and scan.get("hits") == [], "traffic scan must report no hidden values")
