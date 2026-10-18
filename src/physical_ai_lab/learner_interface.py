"""Learner-facing interface (Phase 3; BUILD_PLAN.md sections 3 and 6). Agent side.

The agent (the fixed baseline now, the Phase 4 learner later) runs in its own operating-system
process (``python -m physical_ai_lab.agent_process``) and talks to the evaluator only through
JSON lines on its standard input and output. Nothing else crosses: no Python object, no MuJoCo
model or data, no ``ScenarioRunner``, so the agent cannot reach hidden truth through attributes.

What the agent receives (``TO_AGENT``), all validated against an allow-list:

* ``episode_start``: protocol version, an episode ordinal within this process (1, 2, ...),
  the control period, the episode budget (``max_pushes``, ``max_seconds``) and the first
  ``Observation`` (the 18 fields of observation.py, nothing else);
* ``observation``: the ``Observation`` after the agent's last command (never a later one);
* ``ack``: a note was recorded;
* ``episode_over``: the evaluator ended the episode (no outcome, reason or truth attached);
* ``shutdown``.

What the agent sends (``FROM_AGENT``): ``ready``, ``command`` (7 arm joint targets and the
gripper input for one control update), ``note`` (phase changes, decisions, push start/end with
the chosen action; ``probe: true`` on ``push_start`` declares a probe push, which counts against
the budget like any push), ``stopped`` (the agent stopped by itself) and ``error``.

Never part of any message to the agent: block mass, inertia or friction, scenario or condition
identifiers, split or cohort labels, reset seeds, manifest contents, outcomes of other
episodes, or any state later than the last command. tests/test_learner_interface.py checks this.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from typing import Any, BinaryIO

import numpy as np

from .controller import EpisodeOver
from .observation import OBSERVATION_FIELDS, Observation

PROTOCOL_VERSION = 1
N_ARM = 7

TO_AGENT: dict[str, frozenset[str]] = {
    "episode_start": frozenset({"type", "protocol", "episode", "control_dt", "budget", "observation"}),
    "observation": frozenset({"type", "observation"}),
    "ack": frozenset({"type"}),
    "episode_over": frozenset({"type"}),
    "shutdown": frozenset({"type"}),
}
FROM_AGENT: dict[str, frozenset[str]] = {
    "ready": frozenset({"type", "protocol", "agent"}),
    "command": frozenset({"type", "arm_q", "gripper"}),
    "note": frozenset({"type", "event", "info"}),
    "stopped": frozenset({"type", "reason"}),
    "error": frozenset({"type", "error", "traceback"}),
}
BUDGET_KEYS = frozenset({"max_pushes", "max_seconds"})
# Note events an agent may send. "policy_stopped" and the evaluator's own events are reserved.
AGENT_NOTE_EVENTS = frozenset({"phase", "decision", "push_start", "push_end"})
MAX_LINE_BYTES = 1 << 20


class ProtocolError(ValueError):
    """A message broke the learner-interface protocol."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _plain(value: Any) -> Any:
    """JSON-ready copy: NumPy to Python, non-finite floats to strings (the JSON stays standard)."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ProtocolError(f"value of type {type(value).__name__} cannot cross the learner interface")


def encode(message: dict[str, Any]) -> bytes:
    data = json.dumps(_plain(message), allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(data) > MAX_LINE_BYTES:
        raise ProtocolError(f"message of {len(data)} bytes exceeds {MAX_LINE_BYTES}")
    return data


def decode(line: bytes | str) -> dict[str, Any]:
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"not JSON: {exc}") from None
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ProtocolError("a message must be a JSON object with a string 'type'")
    return message


def _check_keys(message: dict[str, Any], allowed: dict[str, frozenset[str]]) -> str:
    kind = message.get("type")
    if kind not in allowed:
        raise ProtocolError(f"unknown message type {kind!r}")
    keys = set(message)
    if keys != allowed[kind]:
        raise ProtocolError(f"{kind}: keys {sorted(keys)} != {sorted(allowed[kind])}")
    return kind


def validate_observation_payload(payload: Any) -> Observation:
    if not isinstance(payload, dict):
        raise ProtocolError("observation must be an object")
    try:
        return Observation.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid observation: {exc}") from None


def validate_to_agent(message: dict[str, Any]) -> str:
    """Raise ProtocolError unless ``message`` is an allowed evaluator-to-agent message."""
    kind = _check_keys(message, TO_AGENT)
    if kind == "episode_start":
        if message["protocol"] != PROTOCOL_VERSION:
            raise ProtocolError(f"protocol {message['protocol']!r} != {PROTOCOL_VERSION}")
        if not isinstance(message["episode"], int) or message["episode"] < 1:
            raise ProtocolError("episode must be a positive ordinal")
        if not _finite(message["control_dt"]) or message["control_dt"] <= 0:
            raise ProtocolError("control_dt must be positive")
        budget = message["budget"]
        if not isinstance(budget, dict) or set(budget) != BUDGET_KEYS or not all(_finite(v) for v in budget.values()):
            raise ProtocolError(f"budget must have exactly {sorted(BUDGET_KEYS)} with finite values")
    if kind in ("episode_start", "observation"):
        validate_observation_payload(message["observation"])
        if set(message["observation"]) != set(OBSERVATION_FIELDS):
            raise ProtocolError("observation fields differ from observation.OBSERVATION_FIELDS")
    return kind


def validate_from_agent(message: dict[str, Any]) -> str:
    """Raise ProtocolError unless ``message`` is an allowed agent-to-evaluator message."""
    kind = _check_keys(message, FROM_AGENT)
    if kind == "command":
        arm_q, gripper = message["arm_q"], message["gripper"]
        if not isinstance(arm_q, list) or len(arm_q) != N_ARM or not all(_finite(v) for v in arm_q):
            raise ProtocolError(f"command.arm_q must be {N_ARM} finite numbers")
        if not _finite(gripper):
            raise ProtocolError("command.gripper must be a finite number")
    elif kind == "note":
        event, info = message["event"], message["info"]
        if event not in AGENT_NOTE_EVENTS:
            raise ProtocolError(f"note event {event!r} not in {sorted(AGENT_NOTE_EVENTS)}")
        if not isinstance(info, dict):
            raise ProtocolError("note.info must be an object")
        if event == "phase" and not isinstance(info.get("phase"), str):
            raise ProtocolError("phase note needs a string 'phase'")
        if event in ("push_start", "push_end") and not (isinstance(info.get("index"), int) and info["index"] >= 1):
            raise ProtocolError(f"{event} note needs a positive integer 'index'")
        if event == "push_start" and "probe" in info and not isinstance(info["probe"], bool):
            raise ProtocolError("push_start 'probe' must be true or false")
    elif kind == "ready":
        if message["protocol"] != PROTOCOL_VERSION or not isinstance(message["agent"], str):
            raise ProtocolError("ready needs the protocol version and the agent name")
    elif kind == "stopped":
        if not isinstance(message["reason"], str):
            raise ProtocolError("stopped.reason must be a string")
    elif kind == "error":
        if not isinstance(message["error"], str) or not isinstance(message["traceback"], str):
            raise ProtocolError("error needs string 'error' and 'traceback'")
    return kind


class AgentPlant:
    """The agent's only handle on the episode (implements controller.Plant).

    It holds the control period, the budget, the last observation (a fresh copy decoded from
    JSON) and the two pipe ends. Nothing else: no simulator, evaluator or scenario object.
    """

    __slots__ = ("control_dt", "budget", "_obs", "_reader", "_writer")

    def __init__(self, start: dict[str, Any], reader: BinaryIO, writer: BinaryIO) -> None:
        self.control_dt = float(start["control_dt"])
        self.budget = {"max_pushes": int(start["budget"]["max_pushes"]),
                       "max_seconds": float(start["budget"]["max_seconds"])}
        self._obs = validate_observation_payload(start["observation"])
        self._reader, self._writer = reader, writer

    def _exchange(self, message: dict[str, Any]) -> dict[str, Any]:
        validate_from_agent(message)
        self._writer.write(encode(message))
        self._writer.flush()
        line = self._reader.readline()
        if not line:
            raise EOFError("evaluator closed the connection")
        reply = decode(line)
        kind = validate_to_agent(reply)
        if kind == "episode_over":
            raise EpisodeOver()
        return reply

    def observe(self) -> Observation:
        return self._obs

    def command(self, arm_q: np.ndarray, gripper: float) -> Observation:
        reply = self._exchange({"type": "command", "arm_q": [float(v) for v in np.asarray(arm_q, dtype=float)],
                                "gripper": float(gripper)})
        if reply["type"] != "observation":
            raise ProtocolError(f"expected an observation, got {reply['type']!r}")
        self._obs = validate_observation_payload(reply["observation"])
        return self._obs

    def note(self, event: str, **info: Any) -> None:
        reply = self._exchange({"type": "note", "event": event, "info": _plain(info)})
        if reply["type"] != "ack":
            raise ProtocolError(f"expected ack, got {reply['type']!r}")


def make_agent(name: str) -> Any:
    """Agents available behind the interface. Phase 3 has only the fixed-feedback baseline."""
    if name == "baseline":
        from .baseline import FeedbackBaseline
        from .config import load_scene_config
        from .push_config import load_push_config

        return FeedbackBaseline(load_scene_config(), load_push_config())
    raise ValueError(f"unknown agent {name!r}; available: baseline")


def serve(agent_name: str, reader: BinaryIO, writer: BinaryIO) -> int:
    """Agent process main loop: wait for episodes, run the agent, report how it stopped."""

    def send(message: dict[str, Any]) -> None:
        validate_from_agent(message)
        writer.write(encode(message))
        writer.flush()

    agent = make_agent(agent_name)
    send({"type": "ready", "protocol": PROTOCOL_VERSION, "agent": agent_name})
    while True:
        line = reader.readline()
        if not line:
            return 0
        message = decode(line)
        kind = validate_to_agent(message)
        if kind == "shutdown":
            return 0
        if kind != "episode_start":
            raise ProtocolError(f"expected episode_start or shutdown, got {kind!r}")
        plant = AgentPlant(message, reader, writer)
        try:
            reason = agent.run(plant)
        except EpisodeOver:
            continue  # the evaluator ended the episode; wait for the next one
        except Exception as exc:  # reported to the evaluator as a controller error, never hidden
            send({"type": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            continue
        send({"type": "stopped", "reason": str(reason)})


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m physical_ai_lab.agent_process")
    parser.add_argument("--agent", default="baseline")
    args = parser.parse_args(argv)
    reader, writer = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # stdout carries protocol lines only; stray prints go to stderr
    return serve(args.agent, reader, writer)
