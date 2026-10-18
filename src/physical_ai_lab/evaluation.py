"""Evaluator for Phase 2 push episodes: judge, contact log, trajectories and the gate.

Evaluator side only. ``EpisodeJudge`` is called after every physics step (through
``ScenarioRunner.step(monitor=...)``) and reads the live MuJoCo state to decide the
episode outcome with the BUILD_PLAN.md section 4 rules from ``configs/push.toml``:

* success: block centre within ``workspace.target_tolerance`` of the target centre AND
  planar block speed below ``success_max_planar_speed``, continuously for
  ``success_hold_seconds``, counted only while no robot geom touches the block (D21);
* failures: ``unstable`` (NaN / blow-up), ``forbidden_collision`` (robot/table,
  robot self-contact, or a robot body other than the fingers touching the block),
  ``limit_violation`` (joint speed above the Panda limit, joint outside its range,
  TCP below the minimum height), ``block_out_of_bounds``, ``timeout`` (simulated time
  budget), ``push_budget_exceeded`` (the controller started more pushes than allowed),
  ``push_budget_exhausted`` / ``no_feasible_push`` (the controller stopped without
  success), ``controller_error`` (exception in controller code).

The controller only ever sees ``Observation`` objects through ``EvaluatedPlant``.
"""

from __future__ import annotations

import csv
import json
import math
import time
import traceback
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ContextManager

import mujoco
import numpy as np

from . import paths
from .config import SceneConfig
from .controller import EpisodeOver
from .observation import Observation
from .push_config import PushConfig
from .scenario import HiddenPhysics, ScenarioLayout, ScenarioRunner, SimulationUnstable

SUCCESS = "success"
FAILURE_REASONS = (
    "unstable",
    "forbidden_collision",
    "limit_violation",
    "block_out_of_bounds",
    "timeout",
    "push_budget_exceeded",
    "push_budget_exhausted",
    "no_feasible_push",
    "controller_error",
)


@dataclass
class Outcome:
    success: bool
    reason: str  # "success" or one of FAILURE_REASONS
    detail: str
    time: float


@dataclass
class ContactEvent:
    kind: str  # robot-block | robot-table | robot-robot
    body_a: str
    body_b: str
    start: float
    end: float | None
    push_index: int | None  # push in progress when the contact started (None = outside any push)
    phase: str
    forbidden: bool
    max_normal_force: float = 0.0
    max_penetration: float = 0.0


class EpisodeJudge:
    """Per-physics-step outcome and contact bookkeeping (reads the live simulation)."""

    def __init__(self, runner: ScenarioRunner, push: PushConfig) -> None:
        model, scene = runner.sim_model, runner.config
        self.scene, self.push, self.rules, self.limits = scene, push, push.episode, push.limits
        robot = scene.robot
        joints = [model.joint(n) for n in robot.arm_joints]
        self._arm_qadr = np.array([j.qposadr[0] for j in joints], dtype=int)
        self._arm_dadr = np.array([j.dofadr[0] for j in joints], dtype=int)
        self._jlo = np.array([model.jnt_range[j.id][0] for j in joints])
        self._jhi = np.array([model.jnt_range[j.id][1] for j in joints])
        self._max_qvel = np.array(push.limits.max_joint_speed)
        self._hand = model.body(robot.tcp_body).id
        self._tcp_offset = np.array(robot.tcp_offset, dtype=float)
        block_joint = model.joint(scene.block.joint)
        self._block_qadr, self._block_dadr = int(block_joint.qposadr[0]), int(block_joint.dofadr[0])
        self._block_geom = model.geom(scene.block.geom).id
        self._target_mocap = int(model.body_mocapid[model.body(scene.target.mocap_body).id])
        robot_root = model.body_rootid[model.jnt_bodyid[joints[0].id]]
        self._robot_geom = (model.body_rootid == robot_root)[model.geom_bodyid]
        self._table_geom = model.geom_bodyid == model.body(scene.table.body).id
        self._geom_body = np.array(model.geom_bodyid)
        self._body_names = [model.body(i).name for i in range(model.nbody)]
        self._pushers = {model.body(name).id for name in push.limits.pusher_bodies}
        self._timestep = float(model.opt.timestep)
        self._force = np.zeros(6)
        self.outcome: Outcome | None = None
        self.context = {"push_index": None, "phase": "start"}
        self.events: list[ContactEvent] = []
        self._open: dict[tuple[str, str, str], ContactEvent] = {}
        self._hold_steps = 0
        self.stats = {
            "max_joint_speed_ratio": 0.0,
            "min_tcp_z": math.inf,
            "max_joint_range_excess": 0.0,
            "physics_steps": 0,
        }

    # ----- outcome -----
    def finish(self, success: bool, reason: str, detail: str, t: float) -> None:
        if self.outcome is None:
            self.outcome = Outcome(success, reason, detail, float(t))

    def fail(self, reason: str, detail: str, t: float) -> None:
        assert reason in FAILURE_REASONS, reason
        self.finish(False, reason, detail, t)

    # ----- monitor -----
    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        t = float(data.time)
        self.stats["physics_steps"] += 1
        self._contacts(model, data, t)
        if self.outcome is not None:
            return
        qpos, qvel = data.qpos, data.qvel
        if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
            self.fail("unstable", "non-finite state", t)
            return
        forbidden = [e for e in self._open.values() if e.forbidden]
        if forbidden:
            e = forbidden[0]
            self.fail("forbidden_collision", f"{e.kind} contact {e.body_a} / {e.body_b}", t)
            return
        q, dq = qpos[self._arm_qadr], qvel[self._arm_dadr]
        ratio = np.abs(dq) / self._max_qvel
        excess = np.maximum(self._jlo - q, q - self._jhi)
        rotation = data.xmat[self._hand].reshape(3, 3)
        tcp_z = float(data.xpos[self._hand][2] + rotation[2] @ self._tcp_offset)
        stats = self.stats
        stats["max_joint_speed_ratio"] = max(stats["max_joint_speed_ratio"], float(ratio.max()))
        stats["max_joint_range_excess"] = max(stats["max_joint_range_excess"], float(excess.max()))
        stats["min_tcp_z"] = min(stats["min_tcp_z"], tcp_z)
        if ratio.max() > 1.0:
            j = int(ratio.argmax())
            self.fail("limit_violation", f"joint {j + 1} speed {abs(dq[j]):.3f} rad/s > {self._max_qvel[j]}", t)
            return
        if excess.max() > self.limits.joint_range_tolerance:
            j = int(excess.argmax())
            self.fail("limit_violation", f"joint {j + 1} {excess[j]:.4f} rad outside its range", t)
            return
        if tcp_z < self.limits.min_tcp_z:
            self.fail("limit_violation", f"TCP height {tcp_z * 1e3:.1f} mm < {self.limits.min_tcp_z * 1e3:.1f} mm", t)
            return
        bx, by = float(qpos[self._block_qadr]), float(qpos[self._block_qadr + 1])
        w = self.scene.workspace
        if not (w.allowed_block_x.contains(bx) and w.allowed_block_y.contains(by)):
            self.fail("block_out_of_bounds", f"block centre ({bx:.3f}, {by:.3f}) left the allowed region", t)
            return
        target = data.mocap_pos[self._target_mocap]
        distance = math.hypot(bx - float(target[0]), by - float(target[1]))
        speed = math.hypot(float(qvel[self._block_dadr]), float(qvel[self._block_dadr + 1]))
        # Stricter than the section 4 wording (D21): the hold only counts once the robot has
        # released the block, so a block pinned at rest by the fingers never counts as success.
        touching = any(e.kind == "robot-block" for e in self._open.values())
        if distance <= w.target_tolerance and speed < self.rules.success_max_planar_speed and not touching:
            self._hold_steps += 1
            if self._hold_steps * self._timestep >= self.rules.success_hold_seconds - 1e-9:
                self.finish(True, SUCCESS, f"block {distance * 1e3:.1f} mm from target, at rest", t)
                return
        else:
            self._hold_steps = 0
        if t >= self.rules.max_seconds - 1e-9:
            self.fail("timeout", f"{self.rules.max_seconds:g} s simulated-time budget used", t)

    def _contacts(self, model: mujoco.MjModel, data: mujoco.MjData, t: float) -> None:
        seen: set[tuple[str, str, str]] = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.exclude != 0 or contact.dist > 0.0:
                continue
            g1, g2 = int(contact.geom[0]), int(contact.geom[1])
            r1, r2 = bool(self._robot_geom[g1]), bool(self._robot_geom[g2])
            if not (r1 or r2):
                continue
            if r1 and r2:
                kind, a, b = "robot-robot", g1, g2
            else:
                robot, other = (g1, g2) if r1 else (g2, g1)
                if other == self._block_geom:
                    kind = "robot-block"
                elif self._table_geom[other]:
                    kind = "robot-table"
                else:
                    kind = "robot-other"
                a, b = robot, other
            body_a, body_b = self._body_names[self._geom_body[a]], self._body_names[self._geom_body[b]]
            key = (kind, body_a, body_b)
            mujoco.mj_contactForce(model, data, i, self._force)
            event = self._open.get(key)
            if event is None:
                forbidden = kind != "robot-block" or int(self._geom_body[a]) not in self._pushers
                event = ContactEvent(kind, body_a, body_b, t, None, self.context["push_index"],
                                     str(self.context["phase"]), forbidden)
                self._open[key] = event
                self.events.append(event)
            event.max_normal_force = max(event.max_normal_force, float(self._force[0]))
            event.max_penetration = max(event.max_penetration, float(-contact.dist))
            seen.add(key)
        for key in [k for k in self._open if k not in seen]:
            self._open.pop(key).end = t

    def close(self, t: float) -> None:
        for event in self._open.values():
            event.end = t
        self._open.clear()


TRAJECTORY_COLUMNS = (
    ["time", "push_index", "phase"]
    + [f"q{i}" for i in range(1, 8)]
    + [f"q{i}_target" for i in range(1, 8)]
    + ["tcp_x", "tcp_y", "tcp_z", "block_x", "block_y", "block_z", "block_yaw", "block_planar_speed",
       "block_target_distance", "robot_block_contact", "robot_table_contact"]
)


class EpisodeInterrupted(Exception):
    """Raised by an ``on_update`` hook (e.g. the viewer window was closed); not a gate outcome."""


class EvaluatedPlant:
    """The controller's only access to the simulation: observations in, actuator targets out."""

    def __init__(
        self,
        runner: ScenarioRunner,
        judge: EpisodeJudge,
        push: PushConfig,
        on_update: Callable[["EvaluatedPlant", Observation], None] | None = None,
        lock: Callable[[], ContextManager[Any]] | None = None,
        monitor: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None,
    ) -> None:
        self._runner, self._judge, self._push = runner, judge, push
        # Per-physics-step monitor: the judge, or a chain that starts with it (Phase 3 instrumentation).
        self._monitor = monitor or judge
        self._lock = lock or nullcontext
        self.control_dt = runner.config.timing.substeps * runner.config.timing.physics_timestep
        self._arm_ids = np.asarray(runner.arm_actuator_ids, dtype=int)
        self._gripper_id = int(runner.gripper_actuator_id)
        self._nu = runner.sim_model.nu
        self._on_update = on_update
        self.trajectory: list[list[Any]] = []
        self.notes: list[dict[str, Any]] = []
        self.pushes_declared = 0
        self._record(runner.observe())

    def _record(self, obs: Observation) -> None:
        ctx = self._judge.context
        self.trajectory.append(
            [round(obs.time, 4), ctx["push_index"] or 0, ctx["phase"]]
            + [round(float(v), 6) for v in obs.arm_qpos]
            + [round(float(v), 6) for v in obs.arm_qpos_target]
            + [round(float(v), 6) for v in obs.tcp_pos]
            + [round(float(v), 6) for v in obs.block_pos]
            + [round(obs.block_yaw, 6), round(obs.block_planar_speed, 6), round(obs.block_target_distance, 6),
               int(obs.robot_block_contact), int(obs.robot_table_contact)]
        )

    @property
    def over(self) -> bool:
        return self._judge.outcome is not None

    def observe(self) -> Observation:
        return self._runner.observe()

    def command(self, arm_q: np.ndarray, gripper: float) -> Observation:
        if self.over:
            raise EpisodeOver()
        ctrl = np.zeros(self._nu)
        ctrl[self._arm_ids] = arm_q
        ctrl[self._gripper_id] = gripper
        with self._lock():
            self._runner.set_ctrl(ctrl)
            obs = self._runner.step(1, monitor=self._monitor)
        self._record(obs)
        if self._on_update is not None:
            self._on_update(self, obs)
        if self.over:
            raise EpisodeOver()
        return obs

    def note(self, event: str, **info: Any) -> None:
        obs = self._runner.observe()
        ctx = self._judge.context
        if event == "phase":
            ctx["phase"] = info["phase"]
            return
        entry = {"event": event, "time": obs.time, **info,
                 "block_pos": obs.block_pos[:2].tolist(), "block_yaw": obs.block_yaw,
                 "block_target_distance": obs.block_target_distance}
        if event == "push_start":
            self.pushes_declared += 1
            ctx["push_index"] = info["index"]
            ctx["phase"] = "push-start"
            if self.pushes_declared > self._push.episode.max_pushes:
                self._judge.fail("push_budget_exceeded", f"push {self.pushes_declared} started", obs.time)
                self.notes.append(entry)
                raise EpisodeOver()
        elif event == "push_end":
            ctx["push_index"] = None
            ctx["phase"] = "between-pushes"
        self.notes.append(entry)


@dataclass
class EpisodeRecord:
    seed: int
    physics_source: str
    success: bool
    reason: str
    detail: str
    end_time: float
    pushes_declared: int
    unplanned_robot_block_contacts: int
    pushes_used: int
    initial: dict
    final: dict
    pushes: list[dict]
    contact_events: list[dict]
    stats: dict
    stop_reason: str | None
    wall_seconds: float
    trajectory_file: str | None = None
    controller_traceback: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _pose(obs: Observation) -> dict:
    return {
        "time": obs.time,
        "block_pos": obs.block_pos.tolist(),
        "block_yaw_deg": math.degrees(obs.block_yaw),
        "target_pos": obs.target_pos.tolist(),
        "block_target_distance": obs.block_target_distance,
        "block_planar_speed": obs.block_planar_speed,
        "tcp_pos": obs.tcp_pos.tolist(),
    }


def run_episode(
    runner: ScenarioRunner,
    policy: Any,
    push: PushConfig,
    seed: int,
    *,
    randomize_physics: bool = False,
    on_update: Callable[[EvaluatedPlant, Observation], None] | None = None,
    lock: Callable[[], ContextManager[Any]] | None = None,
    trajectory_path: Path | None = None,
    physics: HiddenPhysics | None = None,
    layout: ScenarioLayout | None = None,
    monitors: tuple[Callable[[mujoco.MjModel, mujoco.MjData], None], ...] = (),
) -> EpisodeRecord:
    """Reset to ``seed``, let ``policy.run(plant)`` act, judge and record the episode.

    Phase 3: ``physics`` / ``layout`` give an explicit manifest scenario (evaluator side) and
    ``monitors`` are extra read-only per-physics-step monitors called after the judge.
    """
    wall = time.perf_counter()
    with (lock or nullcontext)():
        first = runner.reset(seed, randomize_physics=randomize_physics, physics=physics, layout=layout)
    truth = runner.hidden_truth()
    judge = EpisodeJudge(runner, push)
    chain = None
    if monitors:
        chain_monitors = (judge, *monitors)

        def chain(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            for m in chain_monitors:
                m(model, data)

    plant = EvaluatedPlant(runner, judge, push, on_update, lock, monitor=chain)
    stop_reason: str | None = None
    tb: str | None = None
    try:
        stop_reason = policy.run(plant)
        plant.note("policy_stopped", reason=stop_reason)
        plant.note("phase", phase="after-stop")
        # Give a block that is already in the target the full hold window before judging.
        rules = push.episode
        hold = np.asarray(runner.observe().arm_qpos_target, dtype=float)
        for _ in range(math.ceil((rules.success_hold_seconds + push.controller.settle_max_seconds) / plant.control_dt)):
            plant.command(hold, runner.config.robot.gripper_ctrl_closed)
        judge.fail(stop_reason if stop_reason in FAILURE_REASONS else "controller_error",
                   f"controller stopped: {stop_reason}", runner.observe().time)
    except EpisodeOver:
        pass
    except EpisodeInterrupted as exc:
        judge.finish(False, "interrupted", str(exc) or "interrupted", float(runner.sim_data.time))
    except SimulationUnstable as exc:
        judge.fail("unstable", str(exc), float(runner.sim_data.time))
    except Exception as exc:  # a controller bug is a visible failure, never a skipped episode
        tb = traceback.format_exc()
        judge.fail("controller_error", f"{type(exc).__name__}: {exc}", float(runner.sim_data.time))
    last = runner.observe()
    judge.close(last.time)
    unplanned = sum(1 for e in judge.events if e.kind == "robot-block" and e.push_index is None)
    if judge.outcome is not None and judge.outcome.success and plant.pushes_declared + unplanned > push.episode.max_pushes:
        # Every robot/block contact outside a declared push counts as a push (section 4 budget).
        judge.outcome = Outcome(False, "push_budget_exceeded",
                                f"{plant.pushes_declared} declared pushes + {unplanned} unplanned robot/block "
                                f"contacts > {push.episode.max_pushes}", judge.outcome.time)
    outcome = judge.outcome
    assert outcome is not None

    pushes: list[dict] = []
    for note in plant.notes:
        if note["event"] == "push_start":
            pushes.append({"index": note["index"], "start_time": note["time"], "plan": note["plan"],
                           "candidates": note["candidates"], "block_before": note["block_pos"],
                           "yaw_before_deg": math.degrees(note["block_yaw"]),
                           "distance_before": note["block_target_distance"],
                           "end_time": None, "aborted": None, "block_after": None, "distance_after": None})
        elif note["event"] == "push_end" and pushes:
            pushes[-1].update(end_time=note["time"], aborted=note["aborted"], block_after=note["block_pos"],
                              yaw_after_deg=math.degrees(note["block_yaw"]),
                              distance_after=note["block_target_distance"])
    events = [asdict(e) for e in judge.events]
    for p in pushes:
        p["contact_events"] = [i for i, e in enumerate(events) if e["push_index"] == p["index"]]
    trajectory_file = None
    if trajectory_path is not None:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with trajectory_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(TRAJECTORY_COLUMNS)
            writer.writerows(plant.trajectory)
        trajectory_file = str(trajectory_path)
    initial = _pose(first)
    initial["block_yaw_deg_truth"] = math.degrees(truth.block_yaw)
    skill_stats = getattr(getattr(policy, "skill", None), "stats", None)
    return EpisodeRecord(
        seed=seed,
        physics_source=truth.physics_source,
        success=outcome.success,
        reason=outcome.reason,
        detail=outcome.detail,
        end_time=outcome.time,
        pushes_declared=plant.pushes_declared,
        unplanned_robot_block_contacts=unplanned,
        pushes_used=plant.pushes_declared + unplanned,
        initial=initial,
        final=_pose(last),
        pushes=pushes,
        contact_events=events,
        stats={**judge.stats, **(skill_stats.to_dict() if skill_stats is not None else {})},
        stop_reason=stop_reason,
        wall_seconds=time.perf_counter() - wall,
        trajectory_file=trajectory_file,
        controller_traceback=tb,
    )


# --------------------------------------------------------------------------- gate


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _clean(value: Any) -> Any:
    """Replace non-finite floats so the JSON stays standard."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(paths.PROJECT_ROOT))
    except ValueError:
        return str(path)


def episode_line(rec: EpisodeRecord) -> str:
    status = "SUCCESS" if rec.success else ("INTERRUPTED" if rec.reason == "interrupted" else "FAILURE")
    final = rec.final["block_target_distance"] * 1e3
    base = (f"[{status}] seed {rec.seed}: pushes {rec.pushes_used} (declared {rec.pushes_declared}, "
            f"unplanned contacts {rec.unplanned_robot_block_contacts}), sim {rec.end_time:.2f} s, "
            f"start distance {rec.initial['block_target_distance'] * 1e3:.1f} mm, final {final:.1f} mm")
    return base if rec.success else f"{base}; reason {rec.reason}: {rec.detail}"


def run_evaluation(
    seeds: tuple[int, ...] | list[int],
    *,
    seed_set: str,
    out_dir: Path,
    scene: SceneConfig | None = None,
    push: PushConfig | None = None,
    echo: Callable[[str], None] = print,
    command: str = "",
) -> dict:
    """Run one baseline episode per seed (nominal physics), log every episode, write JSON."""
    from .baseline import FeedbackBaseline
    from .environment import environment_info
    from .push_config import load_push_config

    push = push or load_push_config()
    runner = ScenarioRunner(scene)
    scene = runner.config
    policy = FeedbackBaseline(scene, push)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    log_path = out_dir / f"{seed_set}_log.txt"
    lines: list[str] = []

    def line(text: str = "") -> None:
        echo(text)
        lines.append(text)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    env = environment_info()
    line(f"Phase 2 baseline evaluation, seed set '{seed_set}' ({len(seeds)} episodes, nominal physics)")
    line(f"started (UTC): {started.isoformat(timespec='seconds')}")
    if command:
        line(f"command: {command}")
    line(f"config: {_rel(push.source)}, {_rel(scene.source)}")
    line(f"python {env.get('python')} ({env.get('process_machine')}), mujoco {env.get('versions', {}).get('mujoco')}")
    line(f"seeds: {list(seeds)}")
    line()
    records = []
    for seed in seeds:
        rec = run_episode(runner, policy, push, seed, trajectory_path=out_dir / "trajectories" / f"seed_{seed}.csv")
        rec.trajectory_file = _rel(Path(rec.trajectory_file)) if rec.trajectory_file else None
        records.append(rec)
        line(episode_line(rec))
    successes = sum(r.success for r in records)
    required = push.seeds.gate_required_successes if seed_set == "gate" else None
    reasons: dict[str, int] = {}
    for r in records:
        if not r.success:
            reasons[r.reason] = reasons.get(r.reason, 0) + 1
    finished = datetime.now(timezone.utc)
    line()
    line(f"result: {successes}/{len(records)} episodes succeeded")
    line(f"failures by reason: {reasons or 'none'}")
    used = [r.pushes_used for r in records if r.success]
    if used:
        line(f"successful episodes: pushes used mean {np.mean(used):.2f}, max {max(used)}; "
             f"sim time mean {np.mean([r.end_time for r in records if r.success]):.2f} s")
    forbidden = sum(1 for r in records for e in r.contact_events if e["forbidden"])
    line(f"forbidden contact events over all episodes: {forbidden}")
    if required is not None:
        verdict = "PASS" if successes >= required else "FAIL"
        line(f"gate (>= {required}/{len(records)} successes): {verdict}")
    line(f"finished (UTC): {finished.isoformat(timespec='seconds')}")
    summary = {
        "seed_set": seed_set,
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": finished.isoformat(timespec="seconds"),
        "command": command,
        "episodes": len(records),
        "successes": successes,
        "required_successes": required,
        "gate_passed": None if required is None else successes >= required,
        "failures_by_reason": reasons,
        "push_config": _rel(push.source),
        "push_config_values": asdict(push),
        "environment": env,
        "records": [r.to_dict() for r in records],
    }
    result_path = out_dir / f"{seed_set}_result.json"
    result_path.write_text(json.dumps(_clean(summary), indent=1, default=_jsonable), encoding="utf-8", newline="\n")
    line(f"log: {_rel(log_path)}")
    line(f"results: {_rel(result_path)}")
    summary["log_path"], summary["result_path"] = log_path, result_path
    return summary
