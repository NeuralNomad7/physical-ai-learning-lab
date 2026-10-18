"""Effective-contact and scene-feasibility checks (Phase 3; BUILD_PLAN.md section 4).

Changing a geom property does not prove the contact changed, so these checks measure the
effect of every applied value in the running simulation:

1. readback: block mass, inertia (solid cube m e^2/6) and contact-pair friction read back from
   the compiled model, and every live block/table contact uses the pair friction;
2. rest: resting block/table normal force against m g, and the block's resting height and tilt
   (measured; run 1 and run 2 found the same 0.108 mm sink at every tested mass);
3. kick: effective sliding friction from the slide distance of a kicked block
   (diagnostics.kick_test; diagnostic only, the task never sets block velocities);
4. hold: 30 s at the home pose with the given physics, checked for stability;
5. pushes: the fixed baseline pushes the block on fixed seeds; block lift and tilt, contact time
   and forces are measured every physics step (instrumentation.ContactMetrics).

Mass alone (nominal friction), friction alone (nominal mass), then both together at the corners
of the in-range box and of the candidate out-of-range box. The acceptance rule is declared in
configs/benchmark.toml [feasibility] before the checks ran.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .baseline import FeedbackBaseline
from .benchmark_config import BenchmarkConfig
from .config import SceneConfig
from .diagnostics import block_table_contacts, kick_test
from .evaluation import _clean, run_episode
from .instrumentation import ContactMetrics, block_tilt_deg
from .push_config import PushConfig
from .scenario import HiddenPhysics, ScenarioRunner


def _label(p: HiddenPhysics) -> str:
    return f"m={p.block_mass:g} kg, mu={p.sliding_friction:g}"


def readback_check(runner: ScenarioRunner, physics: HiddenPhysics, seed: int, settle_seconds: float,
                   inertia_rel_tolerance: float, normal_rel_tolerance: float) -> dict[str, Any]:
    cfg = runner.config
    model = runner.sim_model
    runner.reset(seed, physics=physics)
    applied = runner.applied_physics()
    body = model.body(cfg.block.body).id
    inertia = np.array(model.body_inertia[body], dtype=float)
    expected_inertia = physics.block_mass * cfg.block.edge ** 2 / 6.0
    inertia_err = float(np.max(np.abs(inertia - expected_inertia))) / expected_inertia
    pair_friction = float(model.pair_friction[model.pair(cfg.physics.contact_pair).id, 0])
    geom_friction = float(model.geom_friction[model.geom(cfg.block.geom).id, 0])
    runner.step(round(settle_seconds / cfg.timing.control_dt))
    contacts = block_table_contacts(runner)
    gravity = float(np.linalg.norm(model.opt.gravity))
    weight = physics.block_mass * gravity
    data = runner.sim_data
    rest_height = float(data.xpos[body][2]) - (cfg.table.top_z + cfg.block.edge / 2)
    checks = {
        "mass_readback": applied.block_mass == physics.block_mass,
        "friction_readback": applied.sliding_friction == physics.sliding_friction and pair_friction == physics.sliding_friction,
        "inertia": inertia_err <= inertia_rel_tolerance,
        "contacts_use_pair_friction": contacts.count > 0 and all(f == physics.sliding_friction for f in contacts.frictions),
        "no_other_block_contacts": contacts.other_block_contacts == 0,
        "normal_force": abs(contacts.normal_force - weight) / weight <= normal_rel_tolerance,
    }
    return {
        "physics": asdict(physics),
        "applied": asdict(applied),
        "inertia": inertia.tolist(),
        "expected_inertia": expected_inertia,
        "inertia_rel_error": inertia_err,
        "block_geom_friction_unused": geom_friction,
        "contact_count": contacts.count,
        "contact_frictions": list(contacts.frictions),
        "normal_force": contacts.normal_force,
        "expected_normal_force": weight,
        "normal_force_rel_error": (contacts.normal_force - weight) / weight,
        "rest_height_m": rest_height,
        "rest_tilt_deg": block_tilt_deg(data.xmat[body]),
        "checks": checks,
        "ok": all(checks.values()),
    }


def kick_checks(runner: ScenarioRunner, physics: HiddenPhysics, seed: int, directions: tuple[float, ...],
                v0: float, mu_tolerance: float) -> dict[str, Any]:
    kicks = []
    for direction in directions:
        k = kick_test(runner, physics, seed=seed, direction_deg=direction, v0=v0)
        kicks.append({
            "direction_deg": direction, "distance_m": k.distance, "mu_effective": k.mu_effective,
            "relative_error": k.relative_error, "stop_time": k.stop_time, "final_speed": k.final_speed,
            "yaw_change_deg": math.degrees(k.yaw_change), "normal_force": k.normal_force,
            "expected_normal_force": k.expected_normal_force,
        })
    worst = max(abs(k["relative_error"]) for k in kicks)
    return {"physics": asdict(physics), "kicks": kicks, "worst_relative_error": worst, "ok": worst <= mu_tolerance}


def hold_check(runner: ScenarioRunner, physics: HiddenPhysics, seed: int, seconds: float,
               max_final_qvel: float, max_drift: float) -> dict[str, Any]:
    first = runner.reset(seed, physics=physics)
    metrics = ContactMetrics(runner)
    updates = round(seconds / runner.config.timing.control_dt)
    started = time.perf_counter()
    stable, reason, obs, max_qvel = True, "stable", first, 0.0
    try:
        for _ in range(updates):
            obs = runner.step(1, monitor=metrics)
            max_qvel = max(max_qvel, runner.stability().max_abs_qvel)
    except Exception as exc:  # reported, never hidden
        stable, reason = False, f"{type(exc).__name__}: {exc}"
    final = runner.stability()
    drift = float(np.hypot(*(obs.block_pos[:2] - first.block_pos[:2])))
    ok = stable and final.ok and final.max_abs_qvel <= max_final_qvel and drift <= max_drift
    return {
        "physics": asdict(physics), "seed": seed, "seconds": seconds, "stable": stable and final.ok,
        "reason": reason if not stable else final.reason, "max_abs_qvel": max_qvel,
        "final_max_abs_qvel": final.max_abs_qvel, "block_planar_drift_m": drift,
        "contact": metrics.summary().to_dict(), "wall_seconds": time.perf_counter() - started, "ok": ok,
    }


def push_checks(runner: ScenarioRunner, scene: SceneConfig, push: PushConfig, physics: HiddenPhysics,
                seeds: tuple[int, ...], max_lift: float, max_tilt: float) -> dict[str, Any]:
    episodes = []
    for seed in seeds:
        metrics = ContactMetrics(runner)
        rec = run_episode(runner, FeedbackBaseline(scene, push), push, seed, physics=physics, monitors=(metrics,))
        m = metrics.summary()
        episodes.append({
            "seed": seed, "success": rec.success, "reason": rec.reason, "detail": rec.detail,
            "pushes_used": rec.pushes_used, "end_time": rec.end_time,
            "final_distance_mm": rec.final["block_target_distance"] * 1e3, "contact": m.to_dict(),
            "wall_seconds": rec.wall_seconds,
        })
    lift = max(e["contact"]["max_block_lift"] for e in episodes)
    tilt = max(e["contact"]["max_block_tilt_deg"] for e in episodes)
    unstable = [e["seed"] for e in episodes if e["reason"] == "unstable"]
    return {"physics": asdict(physics), "episodes": episodes, "max_block_lift_m": lift, "max_block_tilt_deg": tilt,
            "unstable_seeds": unstable, "ok": lift <= max_lift and tilt <= max_tilt and not unstable}


def _conditions(bench: BenchmarkConfig, nominal: HiddenPhysics) -> dict[str, list[HiddenPhysics]]:
    f, inner = bench.feasibility, bench.in_range
    corners = lambda box_m, box_f: [HiddenPhysics(m, mu) for m in box_m for mu in box_f]  # noqa: E731
    return {
        "mass_only": [HiddenPhysics(m, nominal.sliding_friction) for m in f.mass_only],
        "friction_only": [HiddenPhysics(nominal.block_mass, mu) for mu in f.friction_only],
        "joint_in_range_corners": corners(inner.mass, inner.sliding_friction),
        "joint_candidate_corners": corners(f.candidate_mass, f.candidate_sliding_friction),
    }


def assess(results: dict[str, Any], bench: BenchmarkConfig) -> dict[str, Any]:
    """Apply the declared acceptance rule: feasible tested values per axis and the resulting box."""
    f = bench.feasibility
    failing: dict[str, set[float]] = {"mass": set(), "sliding_friction": set()}
    for section in ("readback", "kick", "hold", "push"):
        for item in results[section]:
            if not item["ok"]:
                failing["mass"].add(item["physics"]["block_mass"])
                failing["sliding_friction"].add(item["physics"]["sliding_friction"])
    tested = {"mass": sorted(set(f.mass_only) | set(f.candidate_mass)),
              "sliding_friction": sorted(set(f.friction_only) | set(f.candidate_sliding_friction))}
    box = {}
    for axis, (low, high) in (("mass", f.candidate_mass), ("sliding_friction", f.candidate_sliding_friction)):
        values = tested[axis]
        ok = [v for v in values if v not in failing[axis]]
        new_low = low if low not in failing[axis] else min((v for v in ok if v > low), default=math.nan)
        new_high = high if high not in failing[axis] else max((v for v in ok if v < high), default=math.nan)
        box[axis] = [new_low, new_high]
    unchanged = box["mass"] == list(f.candidate_mass) and box["sliding_friction"] == list(f.candidate_sliding_friction)
    return {"failing_values": {k: sorted(v) for k, v in failing.items()}, "feasible_box": box,
            "candidate_box_accepted": unchanged}


def run_contact_checks(scene: SceneConfig, push: PushConfig, bench: BenchmarkConfig, *,
                       progress: Callable[[str], None] = print) -> dict[str, Any]:
    runner = ScenarioRunner(scene)
    f, checks = bench.feasibility, scene.checks
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    groups = _conditions(bench, runner.nominal_physics)
    unique: list[HiddenPhysics] = []
    for items in groups.values():
        for p in items:
            if p not in unique:
                unique.append(p)
    corners = groups["joint_in_range_corners"] + groups["joint_candidate_corners"] + [runner.nominal_physics]
    results: dict[str, Any] = {"readback": [], "kick": [], "hold": [], "push": []}
    for p in unique:
        r = readback_check(runner, p, f.kick_seed, 0.5, f.inertia_rel_tolerance, checks.normal_force_rel_tolerance)
        k = kick_checks(runner, p, f.kick_seed, f.kick_directions_deg, checks.kick_speed, checks.kick_mu_rel_tolerance)
        results["readback"].append(r)
        results["kick"].append(k)
        progress(f"  readback+kick {_label(p):24s} normal {r['normal_force_rel_error']:+.4f}  "
                 f"rest height {r['rest_height_m'] * 1e6:+7.1f} um  kick mu_eff err "
                 f"{', '.join(f'{x['relative_error']:+.4f}' for x in k['kicks'])}  "
                 f"{'ok' if r['ok'] and k['ok'] else 'FAIL'}")
    for p in corners:
        h = hold_check(runner, p, f.hold_seed, f.hold_seconds, checks.hold_max_final_qvel, checks.hold_max_block_drift)
        results["hold"].append(h)
        progress(f"  hold {f.hold_seconds:g} s  {_label(p):24s} stable={h['stable']} drift "
                 f"{h['block_planar_drift_m'] * 1e3:.3f} mm  final |qvel| {h['final_max_abs_qvel']:.2e}  "
                 f"{'ok' if h['ok'] else 'FAIL'}")
    for p in corners:
        pc = push_checks(runner, scene, push, p, f.push_seeds, f.max_block_lift, f.max_block_tilt_deg)
        results["push"].append(pc)
        outcomes = ", ".join(f"{e['seed']}:{e['reason']}" for e in pc["episodes"])
        progress(f"  pushes {_label(p):24s} lift {pc['max_block_lift_m'] * 1e3:.2f} mm  tilt "
                 f"{pc['max_block_tilt_deg']:.2f} deg  [{outcomes}]  {'ok' if pc['ok'] else 'FAIL'}")
    return {
        "kind": "phase3_contact_checks",
        "started_utc": started_utc,
        "wall_seconds": time.perf_counter() - started,
        "benchmark": bench.version_record(),
        "criteria": {
            "inertia_rel_tolerance": f.inertia_rel_tolerance,
            "normal_force_rel_tolerance": checks.normal_force_rel_tolerance,
            "kick_mu_rel_tolerance": checks.kick_mu_rel_tolerance, "kick_speed": checks.kick_speed,
            "hold_seconds": f.hold_seconds, "hold_max_final_qvel": checks.hold_max_final_qvel,
            "hold_max_block_drift": checks.hold_max_block_drift, "max_block_lift": f.max_block_lift,
            "max_block_tilt_deg": f.max_block_tilt_deg, "push_seeds": list(f.push_seeds),
            "kick_seed": f.kick_seed, "hold_seed": f.hold_seed,
        },
        "groups": {k: [asdict(p) for p in v] for k, v in groups.items()},
        "results": results,
        "assessment": assess(results, bench),
    }


def write_contact_checks(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "contact_checks.json"
    path.write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
