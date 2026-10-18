"""Benchmark evaluation, statistics and replay (Phase 3; BUILD_PLAN.md sections 3, 6 and 7). Evaluator side.

``evaluate_split`` runs one controller on every scenario of one manifest split, in manifest order,
once each (no reruns, no skipping). The controller runs in a separate process behind the learner
interface (learner_interface.py): ``AgentConnection`` starts it, ``RemoteAgentPolicy`` relays its
commands and notes to ``evaluation.EvaluatedPlant`` and sends it only observations. Hidden mass and
friction come from the manifest and are applied by ``ScenarioRunner.reset`` on the evaluator side.

Every episode is written as one JSON line (episode_schema.py, validated before it is written) with
the learner-visible part and the evaluator-only part kept apart, plus a ``.npz`` file with the
actuator inputs of every control update and a digest of the full integration state after each one
(for replay). ``summarize`` computes the metrics of benchmark.toml [metrics] per cohort.

Sealed splits (final test, retention) open only through ``manifest.open_split``, which refuses them
without a valid candidate freeze. That includes the baseline.

``replay_run`` checks determinism on this machine: it replays the logged actuator inputs of chosen
episodes without any agent, and reruns the same scenarios through a fresh agent process, comparing
the state digest after every control update with the logged one.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from . import paths
from .benchmark_config import BenchmarkConfig
from .config import SceneConfig
from .episode_schema import EPISODE_SCHEMA_VERSION, validate_episode_record, walk_values
from .evaluation import FAILURE_REASONS, EpisodeRecord, run_episode
from .instrumentation import ContactMetrics
from .learner_interface import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode,
    encode,
    validate_from_agent,
    validate_to_agent,
)
from .manifest import Manifest, Scenario, open_split
from .push_config import PUSH_SCHEMA_VERSION, PushConfig
from .scenario import ScenarioRunner

MODES = ("baseline",)
# Keys an agent note may not use: the evaluator adds them itself (time and block pose at the note).
RESERVED_NOTE_KEYS = frozenset({"event", "time", "block_pos", "block_yaw", "block_target_distance"})


class AgentFailure(RuntimeError):
    """The agent process misbehaved (timeout, exit, protocol error). Recorded as controller_error."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# --------------------------------------------------------------------------- traffic scan


class TrafficScan:
    """Checks every message sent to the agent for hidden values of the current scenario.

    Message keys are already restricted by ``validate_to_agent``. This scan also looks at the
    values: no string equal to (or containing) a scenario / condition id, split, cohort or band
    label, and no number equal to the block mass, friction, inertia or reset seed.
    """

    def __init__(self) -> None:
        self.strings: frozenset[str] = frozenset()
        self.numbers: frozenset[float] = frozenset()
        self.messages = 0
        self.bytes = 0
        self.hits: list[str] = []

    def arm(self, strings: set[str], numbers: set[float]) -> None:
        self.strings, self.numbers = frozenset(strings), frozenset(float(n) for n in numbers)
        self.messages, self.bytes, self.hits = 0, 0, []

    def __call__(self, message: dict[str, Any], data: bytes) -> None:
        self.messages += 1
        self.bytes += len(data)
        text = data.decode("utf-8")
        for s in self.strings:
            if s and s in text:
                self.hits.append(f"string {s!r} in a {message['type']} message")
        for value in walk_values(message):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if float(value) in self.numbers:
                self.hits.append(f"hidden number {value!r} in a {message['type']} message")

    def report(self) -> dict[str, Any]:
        return {"messages": self.messages, "bytes": self.bytes, "hits": list(self.hits)}


# --------------------------------------------------------------------------- agent process


def agent_command(agent: str) -> list[str]:
    return [sys.executable, "-m", "physical_ai_lab.agent_process", "--agent", agent]


class AgentConnection:
    """One agent process; JSON lines over its stdin/stdout. Only this class talks to it."""

    def __init__(self, agent: str, *, reply_timeout: float, stderr_path: Path | None = None) -> None:
        self.agent, self.reply_timeout = agent, float(reply_timeout)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(paths.PACKAGE_DIR.parent), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        self._stderr = stderr_path.open("ab") if stderr_path is not None else subprocess.DEVNULL
        self.proc = subprocess.Popen(agent_command(agent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._stderr, cwd=str(paths.PROJECT_ROOT), env=env)
        self._lines: queue.Queue[bytes] = queue.Queue()
        self._reader = threading.Thread(target=self._read, args=(self.proc.stdout,), daemon=True)
        self._reader.start()
        self.scan: TrafficScan | None = None
        self.alive = True
        ready = self.recv()
        if ready["type"] != "ready" or ready["agent"] != agent:
            raise AgentFailure("protocol", f"expected ready from {agent!r}, got {ready}")

    def _read(self, stream: BinaryIO) -> None:
        for line in iter(stream.readline, b""):
            self._lines.put(line)
        self._lines.put(b"")

    def send(self, message: dict[str, Any]) -> None:
        validate_to_agent(message)
        data = encode(message)
        if self.scan is not None:
            self.scan(message, data)
        try:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()
        except OSError as exc:
            self.alive = False
            raise AgentFailure("agent_died", f"agent process closed its input: {exc}") from None

    def recv(self) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=self.reply_timeout)
        except queue.Empty:
            self.alive = False
            raise AgentFailure("agent_timeout", f"no reply from the agent within {self.reply_timeout:g} s") from None
        if not line:
            self.alive = False
            raise AgentFailure("agent_died", f"agent process ended (exit code {self.proc.poll()})")
        try:
            message = decode(line)
            validate_from_agent(message)
        except ProtocolError as exc:
            self.alive = False
            raise AgentFailure("protocol", f"agent broke the protocol: {exc}") from None
        return message

    def close(self) -> int | None:
        if self.proc.poll() is None:
            try:
                if self.alive:
                    self.send({"type": "shutdown"})
                self.proc.stdin.close()
                self.proc.wait(timeout=20)
            except (AgentFailure, OSError, subprocess.TimeoutExpired):
                self.proc.kill()  # our own child process only
                self.proc.wait(timeout=20)
        if self._stderr is not subprocess.DEVNULL:
            self._stderr.close()
        return self.proc.returncode


# --------------------------------------------------------------------------- relay


class StepLog:
    """Evaluator-side ``on_update`` hook: actuator inputs and a state digest after every control update."""

    def __init__(self, runner: ScenarioRunner) -> None:
        self.runner = runner
        self.ctrl: list[np.ndarray] = []
        self.digests: list[int] = []

    def __call__(self, plant: Any, obs: Any) -> None:
        self.ctrl.append(np.array(self.runner.sim_data.ctrl, dtype=float))
        self.digests.append(state_hash64(self.runner.state_vector()))


def state_hash64(state: np.ndarray) -> int:
    return int.from_bytes(hashlib.sha256(np.asarray(state, dtype="<f8").tobytes()).digest()[:8], "little")


class RemoteAgentPolicy:
    """``policy.run(plant)`` for ``evaluation.run_episode``: relays one episode to the agent process."""

    def __init__(self, conn: AgentConnection, episode: int, budget: dict[str, float], runner: ScenarioRunner,
                 transitions: list[dict[str, Any]] | None = None) -> None:
        self.conn, self.episode, self.budget, self._runner = conn, episode, budget, runner
        # Optional learner-side transitions (episode_schema.TRANSITION_FIELDS), built only from the
        # observations sent to the agent and the commands it sent back.
        self.transitions = transitions
        self.notes: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.observations_sent = 0
        self.commands = 0
        self.first_observation: dict[str, Any] | None = None
        self.last_observation: dict[str, Any] | None = None
        self.end: dict[str, Any] = {"kind": "evaluator_ended", "reason": None}
        self.agent_traceback: str | None = None
        self.wait_seconds = 0.0
        self.initial_state_digest: str | None = None
        self._awaiting = False

    def _send_observation(self, kind: str, obs_dict: dict[str, Any], **extra: Any) -> None:
        message = {"type": kind, **extra, "observation": obs_dict}
        self.conn.send(message)
        self.observations_sent += 1
        if self.first_observation is None:
            self.first_observation = obs_dict
        self.last_observation = obs_dict

    def _recv(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            return self.conn.recv()
        finally:
            self.wait_seconds += time.perf_counter() - t0

    def run(self, plant: Any) -> str:
        obs = plant.observe()
        self.initial_state_digest = self._runner.state_digest()
        self._send_observation("episode_start", obs.to_dict(), protocol=PROTOCOL_VERSION, episode=self.episode,
                               control_dt=plant.control_dt, budget=dict(self.budget))
        self._awaiting = True
        try:
            while True:
                message = self._recv()
                kind = message["type"]
                if kind == "command":
                    self.commands += 1
                    before = self.last_observation
                    obs = plant.command(np.asarray(message["arm_q"], dtype=float), float(message["gripper"]))
                    self._send_observation("observation", obs.to_dict())
                    if self.transitions is not None:
                        self.transitions.append({"observation": before,
                                                 "action": {"arm_q": list(message["arm_q"]),
                                                            "gripper": message["gripper"]},
                                                 "next_observation": self.last_observation})
                elif kind == "note":
                    self._note(plant, message["event"], dict(message["info"]))
                    self.conn.send({"type": "ack"})
                elif kind == "stopped":
                    self._awaiting = False
                    self.end = {"kind": "agent_stopped", "reason": message["reason"]}
                    self.events.append({"t": plant.observe().time, "source": "agent", "type": "agent_stopped",
                                        "reason": message["reason"]})
                    return message["reason"]
                elif kind == "error":
                    self._awaiting = False
                    self.end = {"kind": "agent_error", "reason": message["error"]}
                    self.agent_traceback = message["traceback"]
                    self.events.append({"t": plant.observe().time, "source": "agent", "type": "agent_error",
                                        "error": message["error"]})
                    raise AgentFailure("agent_error", f"agent error: {message['error']}")
                else:
                    raise AgentFailure("protocol", f"unexpected {kind!r} message during an episode")
        except AgentFailure as exc:
            if exc.kind == "protocol":
                self.conn.alive = False  # the agent's state is unknown: restart it before the next episode
            if exc.kind != "agent_error":
                self._awaiting = False
                self.end = {"kind": "agent_died" if exc.kind == "agent_died" else
                            ("agent_timeout" if exc.kind == "agent_timeout" else "agent_error"), "reason": str(exc)}
            raise
        finally:
            if self._awaiting and self.conn.alive:
                # The evaluator ended the episode (outcome, instability, budget) while the agent waits.
                self._awaiting = False
                self.end = {"kind": "evaluator_ended", "reason": None}
                self.conn.send({"type": "episode_over"})

    def _note(self, plant: Any, event: str, info: dict[str, Any]) -> None:
        reserved = RESERVED_NOTE_KEYS & set(info)
        if reserved:
            raise AgentFailure("protocol", f"note info uses reserved keys {sorted(reserved)}")
        obs = plant.observe()
        step = int(obs.control_step)
        if event == "push_start":
            info.setdefault("plan", None)
            info.setdefault("candidates", None)
            info["probe"] = bool(info.get("probe", False))
            self.events.append({"t": obs.time, "source": "agent", "type": "push_start", "index": info["index"],
                                "probe": info["probe"], "plan": info["plan"]})
        elif event == "push_end":
            info.setdefault("aborted", None)
            self.events.append({"t": obs.time, "source": "agent", "type": "push_end", "index": info["index"],
                                "aborted": info["aborted"]})
        elif event == "phase":
            self.events.append({"t": obs.time, "source": "agent", "type": "phase", "phase": info["phase"]})
        elif event == "decision":
            self.events.append({"t": obs.time, "source": "agent", "type": "decision",
                                "decision": info.get("decision")})
        self.notes.append({"control_step": step, "event": event, "info": info})
        plant.note(event, **info)

    def learner_record(self, agent: str, control_dt: float) -> dict[str, Any]:
        pushes: list[dict[str, Any]] = []
        for note in self.notes:
            info = note["info"]
            if note["event"] == "push_start":
                pushes.append({"index": info["index"], "probe": bool(info.get("probe", False)),
                               "start_step": note["control_step"], "end_step": None, "aborted": None,
                               "plan": info.get("plan")})
            elif note["event"] == "push_end" and pushes and pushes[-1]["index"] == info["index"]:
                pushes[-1].update(end_step=note["control_step"], aborted=info.get("aborted"))
        return {
            "agent": agent,
            "protocol_version": PROTOCOL_VERSION,
            "episode": self.episode,
            "control_dt": control_dt,
            "budget": dict(self.budget),
            "observations_received": self.observations_sent,
            "commands_sent": self.commands,
            "notes": self.notes,
            "pushes": pushes,
            "probe_pushes": sum(1 for p in pushes if p["probe"]),
            "end": dict(self.end),
            "first_observation": self.first_observation,
            "last_observation": self.last_observation,
        }


# --------------------------------------------------------------------------- one episode


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return _clean(value.item())
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def scenario_secrets(manifest: Manifest, scenario: Scenario, runner: ScenarioRunner) -> tuple[set[str], set[float]]:
    """Strings and numbers that must never reach the agent for this scenario."""
    cond = manifest.condition(scenario)
    strings = {scenario.scenario_id, cond.condition_id, manifest.split, cond.cohort, cond.band}
    numbers = {cond.physics.block_mass, cond.physics.sliding_friction, float(scenario.reset_seed)}
    model = runner.sim_model
    block = model.body(runner.config.block.body).id
    numbers.update(float(v) for v in model.body_inertia[block])
    return strings, numbers


def run_manifest_episode(
    runner: ScenarioRunner,
    conn: AgentConnection,
    manifest: Manifest,
    scenario: Scenario,
    *,
    episode: int,
    push: PushConfig,
    bench: BenchmarkConfig,
    agent: str,
    commands_path: Path | None,
    transitions: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], StepLog]:
    """Run one manifest scenario through the agent process and return its validated record.

    If ``transitions`` is a list, the learner-side transitions of the episode are appended to it.
    """
    cond = manifest.condition(scenario)
    budget = {"max_pushes": push.episode.max_pushes, "max_seconds": push.episode.max_seconds}
    policy = RemoteAgentPolicy(conn, episode, budget, runner, transitions)
    steps = StepLog(runner)
    metrics = ContactMetrics(runner)
    # Arm the scan with this scenario's secrets (the reset sets the inertia; run_episode resets again
    # from scratch, which is deterministic).
    runner.reset(scenario.reset_seed, physics=cond.physics, layout=scenario.layout)
    strings, numbers = scenario_secrets(manifest, scenario, runner)
    conn.scan = TrafficScan()
    conn.scan.arm(strings, numbers)
    try:
        rec: EpisodeRecord = run_episode(runner, policy, push, scenario.reset_seed, physics=cond.physics,
                                         layout=scenario.layout, monitors=(metrics,), on_update=steps)
        scan = conn.scan.report()
    finally:
        conn.scan = None
    contact = metrics.summary().to_dict()
    applied = runner.applied_physics()
    block = runner.sim_model.body(runner.config.block.body).id
    final_digest = runner.state_digest()
    if commands_path is not None:
        commands_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(commands_path, ctrl=np.asarray(steps.ctrl, dtype=float).reshape(-1, runner.sim_model.nu),
                            digests=np.asarray(steps.digests, dtype=np.uint64))
    learner = policy.learner_record(agent, float(runner.config.timing.substeps * runner.config.timing.physics_timestep))
    events = list(policy.events)
    for e in rec.contact_events:
        events.append({"t": e["start"], "source": "evaluator", "type": "contact", "kind": e["kind"],
                       "body_a": e["body_a"], "body_b": e["body_b"], "start": e["start"], "end": e["end"],
                       "push_index": e["push_index"], "forbidden": e["forbidden"],
                       "max_normal_force": e["max_normal_force"]})
    events.sort(key=lambda e: e["t"])
    limits = {k: rec.stats[k] for k in ("max_joint_speed_ratio", "max_joint_range_excess", "min_tcp_z", "physics_steps")}
    limits["passed"] = rec.reason != "limit_violation"
    events.append({"t": rec.end_time, "source": "evaluator", "type": "limit_check",
                   **{k: limits[k] for k in ("max_joint_speed_ratio", "max_joint_range_excess", "min_tcp_z", "passed")}})
    events.append({"t": rec.end_time, "source": "evaluator", "type": "outcome", "success": rec.success,
                   "reason": rec.reason, "detail": rec.detail})
    probes = learner["probe_pushes"]
    record = {
        "record": "episode",
        "episode_schema_version": EPISODE_SCHEMA_VERSION,
        "benchmark": {"benchmark_version": bench.benchmark_version, "config_sha256": bench.sha256,
                      "metrics_version": bench.metrics["version"], "manifest_sha256": manifest.sha256,
                      "push_config_schema_version": PUSH_SCHEMA_VERSION,
                      "episode_schema_version": EPISODE_SCHEMA_VERSION},
        "learner": learner,
        "evaluator": {
            "episode_index": episode,
            "scenario_id": scenario.scenario_id,
            "condition_id": cond.condition_id,
            "split": manifest.split,
            "cohort": cond.cohort,
            "band": cond.band,
            "reset_seed": scenario.reset_seed,
            "goal_sector": scenario.goal_sector,
            "physics": {"block_mass": cond.physics.block_mass, "sliding_friction": cond.physics.sliding_friction},
            "applied_physics": {"block_mass": applied.block_mass, "sliding_friction": applied.sliding_friction,
                                "body_inertia": [float(v) for v in runner.sim_model.body_inertia[block]]},
            "layout": {"block_xy": list(scenario.layout.block_xy), "block_yaw": scenario.layout.block_yaw,
                       "target_xy": list(scenario.layout.target_xy)},
            "outcome": {"success": rec.success, "reason": rec.reason, "detail": rec.detail, "time": rec.end_time},
            "pushes_declared": rec.pushes_declared,
            "unplanned_robot_block_contacts": rec.unplanned_robot_block_contacts,
            "pushes_used": rec.pushes_used,
            "probe_pushes": probes,
            "final_distance_m": rec.final["block_target_distance"],
            "initial": rec.initial,
            "final": rec.final,
            "pushes": [{k: v for k, v in p.items() if k != "candidates"} for p in rec.pushes],
            "events": events,
            "counts": {
                "failure": int(not rec.success),
                "limit_violation": int(rec.reason == "limit_violation"),
                "forbidden_collision": int(rec.reason == "forbidden_collision"),
                "forbidden_contact_events": sum(1 for e in rec.contact_events if e["forbidden"]),
                "contact_events": len(rec.contact_events),
                "aborted_pushes": sum(1 for p in rec.pushes if p.get("aborted")),
                "probe_pushes": probes,
                "push_budget_exceeded": int(rec.reason == "push_budget_exceeded"),
            },
            "limits": limits,
            "contact": contact,
            "compute": {"wall_seconds": rec.wall_seconds, "agent_wait_seconds": policy.wait_seconds,
                        "evaluator_seconds": rec.wall_seconds - policy.wait_seconds},
            "replay": {"commands_file": None if commands_path is None else commands_path.name,
                       "control_updates": len(steps.ctrl), "initial_state_digest": policy.initial_state_digest,
                       "final_state_digest": final_digest},
            "traffic_scan": scan,
            "agent_traceback": policy.agent_traceback or rec.controller_traceback,
        },
    }
    record = _clean(record)
    validate_episode_record(record)
    return record, steps


# --------------------------------------------------------------------------- statistics


def wilson(k: float, n: float, confidence: float) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    z = _z(confidence)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _z(confidence: float) -> float:
    # Two-sided normal quantile by bisection on erf (no SciPy dependency).
    target = confidence
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erf(mid / math.sqrt(2)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cluster_bootstrap(successes_by_cluster: list[tuple[int, int]], *, resamples: int, seed: int,
                      confidence: float) -> tuple[float, float]:
    """Percentile CI of the pooled success rate, resampling physical conditions with replacement."""
    if not successes_by_cluster:
        return (math.nan, math.nan)
    k = np.array([c[0] for c in successes_by_cluster], dtype=float)
    n = np.array([c[1] for c in successes_by_cluster], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(k), size=(resamples, len(k)))
    rates = k[idx].sum(axis=1) / n[idx].sum(axis=1)
    alpha = (1 - confidence) / 2
    return (float(np.quantile(rates, alpha)), float(np.quantile(rates, 1 - alpha)))


def _describe(values: list[float], scale: float = 1.0) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    a = np.asarray(values, dtype=float) * scale
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.quantile(a, 0.9)), "min": float(a.min()), "max": float(a.max())}


def summarize_group(records: list[dict[str, Any]], bench: BenchmarkConfig) -> dict[str, Any]:
    st = bench.statistics
    ev = [r["evaluator"] for r in records]
    n = len(ev)
    k = sum(1 for e in ev if e["outcome"]["success"])
    clusters: dict[str, list[int]] = {}
    for e in ev:
        c = clusters.setdefault(e["condition_id"], [0, 0])
        c[0] += int(e["outcome"]["success"])
        c[1] += 1
    cl = [tuple(v) for v in clusters.values()]
    succ = [e for e in ev if e["outcome"]["success"]]
    reasons = Counter(e["outcome"]["reason"] for e in ev if not e["outcome"]["success"])
    tol_mm = bench.rules.success_tolerance * 1e3
    distances = [e["final_distance_m"] for e in ev]
    contact = [e["contact"] for e in ev if e["contact"] is not None]
    return {
        "episodes": n,
        "conditions": len(clusters),
        "successes": k,
        "failures": n - k,
        "success_rate": k / n if n else None,
        "success_ci": {
            "confidence": st.confidence,
            "cluster_bootstrap": list(cluster_bootstrap(cl, resamples=st.bootstrap_resamples, seed=st.bootstrap_seed,
                                                        confidence=st.confidence)) if n else None,
            "wilson_conditions": list(wilson(k / n * len(clusters), len(clusters), st.confidence)) if n else None,
            "wilson_episodes_unclustered": list(wilson(k, n, st.confidence)) if n else None,
        },
        "conditions_with_any_failure": sum(1 for v in clusters.values() if v[0] < v[1]),
        "final_distance_mm": {
            "all": _describe(distances, 1e3),
            "successes": _describe([e["final_distance_m"] for e in succ], 1e3),
            "failures": _describe([e["final_distance_m"] for e in ev if not e["outcome"]["success"]], 1e3),
            "episodes_above_15mm": sum(1 for d in distances if d * 1e3 > 15.0),
            "episodes_above_tolerance": sum(1 for d in distances if d * 1e3 > tol_mm),
        },
        "pushes_used": {"all": _describe([e["pushes_used"] for e in ev]),
                        "successes": _describe([e["pushes_used"] for e in succ])},
        "sim_seconds": {"all": _describe([e["outcome"]["time"] for e in ev]),
                        "successes": _describe([e["outcome"]["time"] for e in succ])},
        "failures_by_reason": {r: reasons.get(r, 0) for r in FAILURE_REASONS},
        "violations": {
            "limit_violation_episodes": sum(e["counts"]["limit_violation"] for e in ev),
            "forbidden_collision_episodes": sum(e["counts"]["forbidden_collision"] for e in ev),
            "forbidden_contact_events": sum(e["counts"]["forbidden_contact_events"] for e in ev),
            "push_budget_exceeded_episodes": sum(e["counts"]["push_budget_exceeded"] for e in ev),
            "aborted_pushes": sum(e["counts"]["aborted_pushes"] for e in ev),
            "max_joint_speed_ratio": max((e["limits"]["max_joint_speed_ratio"] for e in ev), default=None),
            "max_joint_range_excess": max((e["limits"]["max_joint_range_excess"] for e in ev), default=None),
            "min_tcp_z": min((e["limits"]["min_tcp_z"] for e in ev), default=None),
        },
        "probe_pushes": sum(e["probe_pushes"] for e in ev),
        "contact": {
            "robot_block_contact_seconds": _describe([c["robot_block_contact_seconds"] for c in contact]),
            "robot_block_contact_intervals": _describe([c["robot_block_contact_intervals"] for c in contact]),
            "robot_block_max_normal_force_N": _describe([c["robot_block_max_normal_force"] for c in contact]),
            "robot_block_mean_normal_force_N": _describe([c["robot_block_mean_normal_force"] for c in contact]),
            "table_normal_ratio_min": _describe([c["table_normal_ratio_min"] for c in contact]),
            "table_normal_ratio_max": _describe([c["table_normal_ratio_max"] for c in contact]),
            "table_normal_ratio_free_mean": _describe([c["table_normal_ratio_free_mean"] for c in contact]),
            "block_airborne_seconds": _describe([c["block_airborne_seconds"] for c in contact]),
            "max_block_lift_mm": _describe([c["max_block_lift"] for c in contact], 1e3),
            "max_block_tilt_deg": _describe([c["max_block_tilt_deg"] for c in contact]),
            "final_block_tilt_deg": _describe([c["final_block_tilt_deg"] for c in contact]),
        },
        "compute_seconds": {
            "total_wall": float(sum(e["compute"]["wall_seconds"] for e in ev)),
            "per_episode_wall": _describe([e["compute"]["wall_seconds"] for e in ev]),
            "per_episode_agent_wait": _describe([e["compute"]["agent_wait_seconds"] for e in ev]),
            "per_episode_evaluator": _describe([e["compute"]["evaluator_seconds"] for e in ev]),
        },
        "traffic_scan": {"messages": sum(e["traffic_scan"]["messages"] for e in ev),
                         "bytes": sum(e["traffic_scan"]["bytes"] for e in ev),
                         "hits": sum(len(e["traffic_scan"]["hits"]) for e in ev)},
    }


def summarize(records: list[dict[str, Any]], bench: BenchmarkConfig) -> dict[str, Any]:
    groups: dict[str, Any] = {"all": summarize_group(records, bench)}
    for cohort in ("in_range", "out_of_range"):
        subset = [r for r in records if r["evaluator"]["cohort"] == cohort]
        if subset:
            groups[cohort] = summarize_group(subset, bench)
    bands = sorted({r["evaluator"]["band"] for r in records if r["evaluator"]["cohort"] == "out_of_range"})
    by_band = {}
    for band in bands:
        subset = [r for r in records if r["evaluator"]["band"] == band]
        g = summarize_group(subset, bench)
        by_band[band] = {k: g[k] for k in ("episodes", "conditions", "successes", "success_rate", "failures_by_reason",
                                            "final_distance_mm", "pushes_used", "sim_seconds")}
    groups["out_of_range_by_band"] = by_band
    return groups


def ceiling_note(group: dict[str, Any]) -> str:
    n, k = group["episodes"], group["successes"]
    if n and k == n:
        return (f"CEILING: the fixed baseline succeeded in all {n} episodes; this split cannot show a learner "
                "beating it on success rate (BUILD_PLAN.md section 6: report the ceiling).")
    if n and k / n >= 0.95:
        return (f"NEAR CEILING: {k}/{n} successes; a +15 percentage-point improvement is impossible here.")
    return ""


def summary_text(summary: dict[str, Any]) -> str:
    lines = [f"Phase 3 baseline evaluation: split {summary['split']}, mode {summary['mode']} "
             f"({summary['benchmark']['benchmark_version']}, config {summary['benchmark']['config_sha256'][:16]}, "
             f"manifest {summary['manifest_sha256'][:16]}, metrics v{summary['benchmark']['metrics_version']}, "
             f"episode schema v{summary['episode_schema_version']})",
             f"started {summary['started_utc']}, finished {summary['finished_utc']}, "
             f"{summary['episodes_run']}/{summary['episodes_in_manifest']} episodes"
             + (" (PARTIAL)" if summary["partial"] else "")]
    for name, g in summary["groups"].items():
        if name == "out_of_range_by_band":
            for band, b in g.items():
                lines.append(f"  band {band:<14} {b['successes']}/{b['episodes']} successes "
                             f"({b['conditions']} conditions); final distance mean "
                             f"{_fmt(b['final_distance_mm']['all']['mean'])} mm")
            continue
        ci = g["success_ci"]
        lines += [
            f"[{name}] {g['successes']}/{g['episodes']} successes = {100 * g['success_rate']:.1f}% "
            f"over {g['conditions']} physical conditions; 95% CI cluster bootstrap "
            f"[{100 * ci['cluster_bootstrap'][0]:.1f}, {100 * ci['cluster_bootstrap'][1]:.1f}]%, "
            f"Wilson (n = conditions) [{100 * ci['wilson_conditions'][0]:.1f}, {100 * ci['wilson_conditions'][1]:.1f}]%",
            f"  final distance mm (all): mean {_fmt(g['final_distance_mm']['all']['mean'])}, median "
            f"{_fmt(g['final_distance_mm']['all']['median'])}, p90 {_fmt(g['final_distance_mm']['all']['p90'])}, "
            f"max {_fmt(g['final_distance_mm']['all']['max'])}; above 15 mm: {g['final_distance_mm']['episodes_above_15mm']}",
            f"  pushes: all mean {_fmt(g['pushes_used']['all']['mean'])} (max {_fmt(g['pushes_used']['all']['max'])}); "
            f"successes mean {_fmt(g['pushes_used']['successes']['mean'])}",
            f"  sim time s: all mean {_fmt(g['sim_seconds']['all']['mean'])} (max {_fmt(g['sim_seconds']['all']['max'])}); "
            f"successes mean {_fmt(g['sim_seconds']['successes']['mean'])}",
            "  failures by reason: " + (", ".join(f"{r} {c}" for r, c in g["failures_by_reason"].items() if c) or "none"),
            f"  violations: limit {g['violations']['limit_violation_episodes']}, forbidden-collision episodes "
            f"{g['violations']['forbidden_collision_episodes']}, forbidden contact events "
            f"{g['violations']['forbidden_contact_events']}, aborted pushes {g['violations']['aborted_pushes']}, "
            f"probe pushes {g['probe_pushes']}",
            f"  contact: robot/block contact s mean {_fmt(g['contact']['robot_block_contact_seconds']['mean'])}, "
            f"max normal force N max {_fmt(g['contact']['robot_block_max_normal_force_N']['max'])}, "
            f"max lift mm {_fmt(g['contact']['max_block_lift_mm']['max'])}, "
            f"max tilt deg {_fmt(g['contact']['max_block_tilt_deg']['max'])}, "
            f"table normal ratio [{_fmt(g['contact']['table_normal_ratio_min']['min'])}, "
            f"{_fmt(g['contact']['table_normal_ratio_max']['max'])}]",
            f"  compute: {g['compute_seconds']['total_wall']:.1f} s wall, "
            f"{_fmt(g['compute_seconds']['per_episode_wall']['mean'])} s/episode "
            f"(agent wait {_fmt(g['compute_seconds']['per_episode_agent_wait']['mean'])} s)",
            f"  traffic scan: {g['traffic_scan']['messages']} messages to the agent, hidden-value hits "
            f"{g['traffic_scan']['hits']}",
        ]
        note = ceiling_note(g)
        if note:
            lines.append("  " + note)
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.2f}"


# --------------------------------------------------------------------------- evaluate a split


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(paths.PROJECT_ROOT))
    except ValueError:
        return str(path)


def evaluate_split(
    split: str,
    *,
    mode: str,
    out_dir: Path,
    scene: SceneConfig,
    push: PushConfig,
    bench: BenchmarkConfig,
    manifest_dir: Path = paths.MANIFESTS_DIR,
    freeze_dir: Path = paths.FREEZES_DIR,
    limit: int | None = None,
    command: str = "",
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run ``mode`` once on every scenario of ``split`` (manifest order) and write the report."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not available in Phase 3 (available: {', '.join(MODES)})")
    manifest = open_split(split, manifest_dir=manifest_dir, freeze_dir=freeze_dir, bench=bench)
    scenarios = manifest.scenarios if limit is None else manifest.scenarios[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    wall = time.perf_counter()
    runner = ScenarioRunner(scene)
    records: list[dict[str, Any]] = []
    episodes_path = out_dir / "episodes.jsonl"
    stderr_path = out_dir / "agent_stderr.log"
    conn: AgentConnection | None = None
    restarts = 0
    progress(f"{split}: {len(scenarios)} scenarios, mode {mode}, manifest {manifest.sha256[:16]}; writing to {out_dir}")
    with episodes_path.open("w", encoding="utf-8", newline="\n") as out:
        try:
            for i, scenario in enumerate(scenarios, start=1):
                if conn is None or not conn.alive:
                    if conn is not None:
                        conn.close()
                        restarts += 1
                    conn = AgentConnection(mode, reply_timeout=bench.agent.reply_timeout_seconds,
                                           stderr_path=stderr_path)
                record, _ = run_manifest_episode(runner, conn, manifest, scenario, episode=i, push=push, bench=bench,
                                                 agent=mode, commands_path=out_dir / "commands" / f"ep{i:04d}.npz")
                out.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                out.flush()
                records.append(record)
                ev = record["evaluator"]
                status = "SUCCESS" if ev["outcome"]["success"] else f"FAILURE {ev['outcome']['reason']}"
                progress(f"[{i}/{len(scenarios)}] {ev['cohort']:<12} {status}: pushes {ev['pushes_used']}, "
                         f"sim {ev['outcome']['time']:.2f} s, final {ev['final_distance_m'] * 1e3:.1f} mm, "
                         f"lift {ev['contact']['max_block_lift'] * 1e3:.2f} mm, tilt {ev['contact']['max_block_tilt_deg']:.2f} deg, "
                         f"wall {ev['compute']['wall_seconds']:.1f} s")
        finally:
            if conn is not None:
                conn.close()
    summary = {
        "kind": "phase3_evaluation",
        "split": split,
        "mode": mode,
        "command": command,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": time.perf_counter() - wall,
        "benchmark": bench.version_record(),
        "manifest_sha256": manifest.sha256,
        "manifest_provenance": manifest.data["provenance"],
        "push_config_schema_version": PUSH_SCHEMA_VERSION,
        "episode_schema_version": EPISODE_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "episodes_in_manifest": len(manifest.scenarios),
        "episodes_run": len(records),
        "partial": len(records) != len(manifest.scenarios),
        "agent_process_restarts": restarts,
        "episodes_file": episodes_path.name,
        "groups": summarize(records, bench),
        "notes": [n for n in (ceiling_note(summarize_group(records, bench)),) if n],
    }
    summary = _clean(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = summary_text(summary)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    progress(text.rstrip())
    return summary


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "episodes.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------- replay


def replay_commands(runner: ScenarioRunner, manifest: Manifest, scenario: Scenario, ctrl: np.ndarray,
                    digests: np.ndarray, tolerance: float) -> dict[str, Any]:
    """Reset to the scenario and apply the logged actuator inputs, with no agent and no judge."""
    cond = manifest.condition(scenario)
    runner.reset(scenario.reset_seed, physics=cond.physics, layout=scenario.layout)
    initial = runner.state_digest()
    first_mismatch = None
    for i, row in enumerate(ctrl):
        runner.set_ctrl(row)
        runner.step(1)
        if state_hash64(runner.state_vector()) != int(digests[i]) and first_mismatch is None:
            first_mismatch = i + 1
    return {"control_updates": int(len(ctrl)), "initial_state_digest": initial, "final_state_digest": runner.state_digest(),
            "first_mismatch_update": first_mismatch, "tolerance": tolerance}


def replay_run(run_dir: Path, *, scene: SceneConfig, push: PushConfig, bench: BenchmarkConfig,
               manifest_dir: Path = paths.MANIFESTS_DIR, freeze_dir: Path = paths.FREEZES_DIR,
               count: int | None = None, progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Replay the first ``count`` episodes of an evaluation run (manifest order) two ways.

    1. command replay: logged actuator inputs only (no agent, no judge, no monitors);
    2. rerun: the same scenario through a fresh agent process and the full evaluator.

    Both must match the logged state digest after every control update (bitwise; the tolerance
    in benchmark.toml [replay] is 0.0) and the logged initial and final state digests; the rerun
    must also give the same outcome, pushes and final distance.
    """
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = load_records(run_dir)
    split, mode = summary["split"], summary["mode"]
    manifest = open_split(split, manifest_dir=manifest_dir, freeze_dir=freeze_dir, bench=bench)
    if manifest.sha256 != summary["manifest_sha256"]:
        raise ValueError("the run used a different manifest")
    by_id = {s.scenario_id: s for s in manifest.scenarios}
    count = bench.replay.scenarios_per_split if count is None else count
    tolerance = bench.replay.state_abs_tolerance
    results = []
    runner = ScenarioRunner(scene)
    conn = AgentConnection(mode, reply_timeout=bench.agent.reply_timeout_seconds)
    try:
        for record in records[:count]:
            ev = record["evaluator"]
            scenario = by_id[ev["scenario_id"]]
            logged = np.load(run_dir / "commands" / ev["replay"]["commands_file"])
            ctrl, digests = logged["ctrl"], logged["digests"]
            t0 = time.perf_counter()
            cmd = replay_commands(runner, manifest, scenario, ctrl, digests, tolerance)
            cmd["wall_seconds"] = time.perf_counter() - t0
            cmd["matches"] = (cmd["first_mismatch_update"] is None
                              and cmd["initial_state_digest"] == ev["replay"]["initial_state_digest"]
                              and cmd["final_state_digest"] == ev["replay"]["final_state_digest"])
            t0 = time.perf_counter()
            rerun, steps = run_manifest_episode(runner, conn, manifest, scenario, episode=1, push=push, bench=bench,
                                                agent=mode, commands_path=None)
            rev = rerun["evaluator"]
            step_digests = np.asarray(steps.digests, dtype=np.uint64)
            same_len = step_digests.shape == digests.shape
            mismatch = None
            if same_len:
                diff = np.nonzero(step_digests != digests)[0]
                mismatch = int(diff[0]) + 1 if diff.size else None
            fields = ("outcome", "pushes_used", "final_distance_m", "contact", "limits")
            rer = {
                "wall_seconds": time.perf_counter() - t0,
                "control_updates": len(steps.ctrl),
                "same_number_of_updates": same_len,
                "first_mismatch_update": mismatch,
                "ctrl_identical": same_len and bool(np.array_equal(np.asarray(steps.ctrl), ctrl)),
                "fields_identical": {f: rev[f] == ev[f] for f in fields},
                "learner_view_identical_except_episode_ordinal": {**rerun["learner"], "episode": 0}
                == {**record["learner"], "episode": 0},
                "final_state_digest": rev["replay"]["final_state_digest"],
            }
            rer["matches"] = (same_len and mismatch is None and rer["ctrl_identical"]
                              and all(rer["fields_identical"].values())
                              and rer["learner_view_identical_except_episode_ordinal"]
                              and rev["replay"]["final_state_digest"] == ev["replay"]["final_state_digest"]
                              and rev["replay"]["initial_state_digest"] == ev["replay"]["initial_state_digest"])
            results.append({"episode_index": ev.get("episode_index"), "scenario_id": ev["scenario_id"],
                            "outcome": ev["outcome"]["reason"], "logged_final_state_digest": ev["replay"]["final_state_digest"],
                            "command_replay": cmd, "rerun": rer})
            progress(f"replay {split} episode {ev.get('episode_index')}: {len(ctrl)} control updates; command replay "
                     f"{'IDENTICAL' if cmd['matches'] else 'DIFFERS'}; rerun through the agent "
                     f"{'IDENTICAL' if rer['matches'] else 'DIFFERS'}")
    finally:
        conn.close()
    return {"kind": "phase3_replay", "split": split, "mode": mode, "run": _rel(run_dir),
            "manifest_sha256": manifest.sha256, "tolerance": tolerance, "comparison": "bitwise state digest "
            "(SHA-256 of the full MuJoCo integration state, first 8 bytes) after every control update",
            "episodes": results, "all_match": all(r["command_replay"]["matches"] and r["rerun"]["matches"]
                                                  for r in results)}

