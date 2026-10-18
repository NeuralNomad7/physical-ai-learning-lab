"""Phase 1 headless smoke test (``python -m physical_ai_lab smoke``).

Checks, without opening a window:

* environment and pinned dependency versions;
* integrity of the vendored Panda asset (SHA-256 manifest);
* scene structure (``verify_scene`` / ``verify_actuators``, model sizes, solver options);
* seeded-reset reproducibility within ``[checks] reset_state_tolerance``: across runner
  history, with a fresh runner and in a separate process;
* reset layouts over ``layout_seeds`` seeds (regions, distances, initial contacts);
* hidden physics (mass and friction really applied, measured with a kick test);
* the agent-facing observation boundary (allow-list, copies, no hidden values);
* passive stability, the instability detector and actuator response;
* TCP-down reachability of the block region;
* offscreen rendering of the scene cameras (screenshot evidence);
* camera views: at reset the arm hides at most ``[checks] max_arm_occlusion`` of the block
  and of the goal disc in any camera, over ``visibility_seeds`` seeds (segmentation renders).

Results go to ``runs/smoke/<UTC timestamp>/`` (``smoke_log.txt``, ``smoke_result.json``
and the camera PNGs). The exit status is non-zero if any check fails.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from . import __version__, assets, paths
from .config import Range, SceneConfig, load_scene_config
from .diagnostics import HoldReport, actuator_response_test, check_reach, hold_test, kick_test
from .environment import environment_info
from .observation import OBSERVATION_FIELDS, Observation
from .scenario import HiddenPhysics, ScenarioRunner, SimulationUnstable, capture_mujoco_warnings

PASS, FAIL, WARN, INFO, SKIP = "PASS", "FAIL", "WARN", "INFO", "SKIP"

# The agent-facing allow-list, written out independently of observation.py so that any
# change to the observation boundary has to be made deliberately in two places.
EXPECTED_OBSERVATION_FIELDS = (
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
FORBIDDEN_FIELD_WORDS = (
    "mass", "friction", "inertia", "force", "damping", "solref", "solimp", "seed", "scenario",
    "truth", "physics", "hidden", "future", "label", "manifest",
)
# MjData/MjModel arrays an observation must never alias.
_SIM_ARRAYS = {
    "data": (
        "qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "act", "mocap_pos", "mocap_quat", "xpos", "xquat",
        "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat", "site_xpos", "site_xmat", "cvel", "cacc",
        "subtree_com", "qfrc_applied", "xfrc_applied", "qfrc_actuator", "actuator_force", "sensordata",
    ),
    "model": ("body_mass", "body_inertia", "pair_friction", "geom_friction", "body_pos", "geom_size"),
}


@dataclass
class CheckResult:
    section: str
    name: str
    status: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SmokeOutcome:
    ok: bool
    gate_passed: bool
    counts: dict[str, int]
    out_dir: Path
    log_path: Path
    result_path: Path
    screenshot: Path | None


class Recorder:
    """Collects check results and mirrors every line to the console and the log file."""

    def __init__(self, log_path: Path, echo: Callable[[str], None]) -> None:
        self.results: list[CheckResult] = []
        self.section = ""
        self._echo = echo
        self._file = log_path.open("w", encoding="utf-8", newline="\n")

    def line(self, text: str = "") -> None:
        self._echo(text)
        self._file.write(text + "\n")
        self._file.flush()

    def begin(self, section: str, title: str) -> None:
        self.section = section
        self.line()
        self.line(f"== {title}")

    def record(self, status: str, name: str, detail: str, data: dict[str, Any] | None = None) -> bool:
        self.results.append(CheckResult(self.section, name, status, detail, data or {}))
        self.line(f"[{status}] {name}: {detail}")
        return status != FAIL

    def expect(self, name: str, ok: bool, detail: str, data: dict[str, Any] | None = None) -> bool:
        return self.record(PASS if ok else FAIL, name, detail, data)

    def info(self, name: str, detail: str, data: dict[str, Any] | None = None) -> None:
        self.record(INFO, name, detail, data)

    def warn(self, name: str, detail: str, data: dict[str, Any] | None = None) -> None:
        self.record(WARN, name, detail, data)

    def skip(self, name: str, detail: str) -> None:
        self.record(SKIP, name, detail)

    def section_failed(self, section: str) -> bool:
        return any(r.section == section and r.status == FAIL for r in self.results)

    def close(self) -> None:
        self._file.close()


# --------------------------------------------------------------------------- helpers


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(paths.PROJECT_ROOT))
    except ValueError:
        return str(path)


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return math.inf
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _field_differences(a: Observation, b: Observation) -> dict[str, float]:
    diffs = {}
    for name in OBSERVATION_FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, str):
            diffs[name] = 0.0 if x == y else math.inf
        else:
            diffs[name] = _max_diff(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return diffs


def _grid(r: Range) -> tuple[float, float, float]:
    return (r.low, (r.low + r.high) / 2.0, r.high)


def _seeds(seeds: list[int], limit: int = 10) -> str:
    shown = ", ".join(str(s) for s in seeds[:limit])
    return shown + (f", ... ({len(seeds)} total)" if len(seeds) > limit else "")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return repr(value)


# --------------------------------------------------------------------------- sections


def _check_environment(rec: Recorder, env: dict[str, Any]) -> None:
    rec.info(
        "python",
        f"{env['python']} {env['python_implementation']}, {env['process_bits']}-bit {env['process_machine']} "
        f"process ({env['python_executable']})",
    )
    emulated = f"; this {env['process_machine']} interpreter ({env['python_platform']}) runs under emulation" if env["emulated"] else ""
    rec.info("host", f"{env['os']}; native CPU {env['host_machine']}{emulated}")
    rec.info("versions", ", ".join(f"{k} {v}" for k, v in env["versions"].items()), env["versions"])
    mismatches = env["pin_mismatches"]
    rec.expect(
        "dependency pins",
        not mismatches,
        f"installed versions match the {len(env['pins'])} exact pins in pyproject.toml"
        if not mismatches
        else f"installed versions differ from pyproject.toml pins: {mismatches}",
        {"pins": env["pins"], "mismatches": mismatches},
    )
    if env["git_root"] is None:
        rec.warn("version control", "the project folder is not inside a git repository (changes are not tracked)")
    if env["onedrive_folder"]:
        rec.warn(
            "OneDrive",
            "the project is in a OneDrive-synced folder; if `uv sync` reports 'Access is denied' see README.md",
        )


def _check_assets(rec: Recorder) -> None:
    report = assets.verify()
    source = assets.load_manifest()["source"]
    rec.expect(
        "Panda asset integrity",
        report.ok,
        report.summary(),
        {"commit": report.commit, "checked": report.checked, "missing": report.missing, "mismatched": report.mismatched},
    )
    rec.info(
        "asset source",
        f"{source['repository']} @ {source['commit']} / {source['path']} ({source['license']}), used unmodified",
    )


def _check_scene(rec: Recorder, cfg: SceneConfig) -> ScenarioRunner:
    started = time.perf_counter()
    runner = ScenarioRunner(cfg)  # compiles the MJCF; verify_scene/verify_actuators raise on any mismatch
    elapsed = time.perf_counter() - started
    model = runner.sim_model
    rec.expect(
        "scene loads and matches configs/scene.toml",
        True,
        f"{_rel(cfg.mjcf_path)} compiled; verify_scene and verify_actuators passed ({elapsed:.2f} s)",
    )
    robot = cfg.robot
    joints = len(robot.arm_joints) + len(robot.finger_joints)
    expected = {"nq": joints + 7, "nv": joints + 6, "nu": len(robot.arm_actuators) + 1}
    actual = {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu)}
    rec.expect(
        "model size",
        actual == expected,
        f"nq {model.nq}, nv {model.nv}, nu {model.nu} (7 arm joints + 2 fingers + free block; 7 arm servos + gripper); "
        f"{model.nbody} bodies, {model.ngeom} geoms, {model.npair} explicit contact pair(s)",
        {"actual": actual, "expected": expected},
    )
    opt = model.opt
    rec.info(
        "solver",
        f"timestep {opt.timestep} s, integrator {mujoco.mjtIntegrator(opt.integrator).name}, "
        f"cone {mujoco.mjtCone(opt.cone).name}, solver {mujoco.mjtSolver(opt.solver).name}, "
        f"iterations {opt.iterations}, impratio {opt.impratio:g}",
    )
    for note in runner.scene_notes:
        rec.info("scene", note)
    for a in runner.actuators:
        rec.info(
            "actuator",
            f"{a.name} -> {a.target}: {a.kind}, kp {a.kp:g}, kv {a.kv:g}, ctrlrange {a.ctrlrange}, forcerange {a.forcerange}",
        )
    return runner


def _check_reset(rec: Recorder, runner: ScenarioRunner, cfg: SceneConfig) -> None:
    c, p = cfg.checks, cfg.physics
    tol = c.reset_state_tolerance
    dt = cfg.timing.control_dt
    updates = round(c.rollout_seconds / dt)

    def rollout(r: ScenarioRunner, seed: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, str]:
        r.reset(seed, **kwargs)
        start = r.state_vector().copy()
        r.step(updates)
        return start, r.state_vector().copy(), r.state_digest()

    s0, e0, d0 = rollout(runner, 0)
    rec.info(
        "reference rollout",
        f"seed 0, nominal hidden physics, {updates} control updates ({updates * dt:g} s): state digest {d0} "
        f"({e0.size} values, mjSTATE_INTEGRATION)",
        {"digest": d0, "updates": updates},
    )
    rollout(runner, 7, randomize_physics=True)
    rollout(runner, 3, physics=HiddenPhysics(p.mass_range.high, p.sliding_friction_range.low))
    s1, e1, d1 = rollout(runner, 0)
    diff = _max_diff(e0, e1)
    rec.expect(
        "same runner after other episodes",
        diff <= tol,
        f"seed 0 again after a randomized and an explicit-physics episode: max |state difference| {diff:.3g} "
        f"(tolerance {tol:g}); digest {d1}",
    )
    s2, e2, d2 = rollout(ScenarioRunner(cfg), 0)
    diff = _max_diff(e0, e2)
    rec.expect(
        "fresh runner",
        diff <= tol,
        f"new ScenarioRunner, seed 0: max |state difference| {diff:.3g} (tolerance {tol:g}); digest {d2}",
    )
    # The viewer's Physics panel writes its edits into the model's options; a reset must undo them.
    opt = runner.sim_model.opt
    opt.timestep *= 2.0
    opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
    s3, e3, d3 = rollout(runner, 0)
    diff = _max_diff(e0, e3)
    left = runner.restore_options()
    rec.expect(
        "physics options restored",
        diff <= tol and not left,
        f"seed 0 after editing the timestep, the gravity flag and the contact override as the viewer's Physics "
        f"panel would: max |state difference| {diff:.3g} (tolerance {tol:g}); digest {d3}"
        + (f"; still edited after the reset: {', '.join(left)}" if left else ""),
    )
    diff = max(_max_diff(s0, s1), _max_diff(s0, s2), _max_diff(s0, s3))
    rec.expect("initial state", diff <= tol, f"t = 0 state after reset(0): max |difference| {diff:.3g}")

    first = rollout(runner, 5, randomize_physics=True)
    truth_a = runner.hidden_truth()
    second = rollout(runner, 5, randomize_physics=True)
    truth_b = runner.hidden_truth()
    diff = _max_diff(first[1], second[1])
    rec.expect(
        "randomized physics is seeded",
        truth_a == truth_b and diff <= tol,
        f"seed 5 with randomized hidden physics twice: same hidden values {truth_a == truth_b}, "
        f"max |state difference| {diff:.3g}",
    )
    runner.reset(1)
    diff = _max_diff(s0, runner.state_vector())
    rec.expect("different seeds differ", diff > 0, f"seed 1 vs seed 0 initial state: max |difference| {diff:.3g}")

    command = [
        sys.executable, "-m", "physical_ai_lab", "step", "--seed", "0", "--seconds", repr(c.rollout_seconds), "--json",
    ]
    proc = subprocess.run(command, cwd=paths.PROJECT_ROOT, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        rec.expect("separate process", False, f"`step` exited with {proc.returncode}: {proc.stderr.strip()[-500:]}")
        return
    payload = json.loads(proc.stdout)
    same = payload["state_digest"] == d0 and payload["control_updates"] == updates
    rec.expect(
        "separate process",
        same,
        f"`{paths.python_command()} -m physical_ai_lab step --seed 0 --seconds {c.rollout_seconds:g} --json` "
        f"in a new interpreter: digest {payload['state_digest']} ({'bitwise identical' if same else 'DIFFERENT'})",
    )
    rec.info(
        "reproducibility scope",
        "same seed, same machine, same MuJoCo build and config; other machines/CPU architectures/builds not tested",
    )


def _check_layouts(rec: Recorder, runner: ScenarioRunner, cfg: SceneConfig) -> None:
    w, n = cfg.workspace, cfg.checks.layout_seeds
    top_z = cfg.table.top_z
    rest_z = top_z + cfg.block.edge / 2
    problems: dict[str, list[int]] = {
        k: [] for k in ("block region", "target region", "distance", "block pose", "target pose", "contact", "motion")
    }
    rows = []
    started = time.perf_counter()
    for seed in range(n):
        obs = runner.reset(seed)
        truth = runner.hidden_truth()
        (bx, by), yaw, (tx, ty) = truth.block_xy, truth.block_yaw, truth.target_xy
        distance = math.hypot(tx - bx, ty - by)
        rows.append((bx, by, yaw, tx, ty, distance))
        if not (w.block_x.contains(bx) and w.block_y.contains(by)):
            problems["block region"].append(seed)
        if not (w.target_x.contains(tx) and w.target_y.contains(ty)):
            problems["target region"].append(seed)
        if not w.block_target_distance.contains(distance):
            problems["distance"].append(seed)
        pose_error = max(
            abs(obs.block_pos[0] - bx),
            abs(obs.block_pos[1] - by),
            abs(obs.block_pos[2] - rest_z),
            abs(math.remainder(obs.block_yaw - yaw, math.tau)),
        )
        if pose_error > 1e-9:
            problems["block pose"].append(seed)
        if _max_diff(obs.target_pos, np.array([tx, ty, top_z])) > 1e-12 or abs(obs.block_target_distance - distance) > 1e-9:
            problems["target pose"].append(seed)
        if obs.robot_block_contact or obs.robot_table_contact:
            problems["contact"].append(seed)
        if np.any(obs.block_linvel) or np.any(obs.block_angvel) or np.any(obs.arm_qvel):
            problems["motion"].append(seed)
    elapsed = time.perf_counter() - started
    labels = {
        "block region": f"block centre inside x {w.block_x.as_list()}, y {w.block_y.as_list()} m",
        "target region": f"target inside x {w.target_x.as_list()}, y {w.target_y.as_list()} m",
        "distance": f"block-target distance inside {w.block_target_distance.as_list()} m",
        "block pose": "block placed flat on the tabletop at the sampled x, y, yaw",
        "target pose": "target marker at the sampled position on the tabletop",
        "contact": "no robot/block or robot/table contact at reset",
        "motion": "everything starts at rest",
    }
    for key, seeds in problems.items():
        rec.expect(
            f"layouts: {key}",
            not seeds,
            f"{labels[key]}: {n - len(seeds)}/{n} seeds" + (f"; failing seeds {_seeds(seeds)}" if seeds else ""),
        )
    rec.expect("layouts: distinct", len(set(rows)) == n, f"{len(set(rows))} distinct layouts from {n} seeds")
    changed = []
    for seed in range(20):
        runner.reset(seed)
        nominal = runner.hidden_truth()
        runner.reset(seed, randomize_physics=True)
        randomized = runner.hidden_truth()
        if (nominal.block_xy, nominal.block_yaw, nominal.target_xy) != (
            randomized.block_xy, randomized.block_yaw, randomized.target_xy,
        ):
            changed.append(seed)
    rec.expect(
        "layout independent of physics randomization",
        not changed,
        "seeds 0-19 give the same layout with nominal and randomized hidden physics"
        if not changed
        else f"layout changed for seeds {_seeds(changed)}",
    )
    a = np.array(rows)
    rec.info(
        "layout coverage",
        f"block x {a[:, 0].min():.3f}..{a[:, 0].max():.3f}, y {a[:, 1].min():.3f}..{a[:, 1].max():.3f}; "
        f"target x {a[:, 3].min():.3f}..{a[:, 3].max():.3f}, y {a[:, 4].min():.3f}..{a[:, 4].max():.3f}; "
        f"distance {a[:, 5].min():.3f}..{a[:, 5].max():.3f} m; yaw {a[:, 2].min():.2f}..{a[:, 2].max():.2f} rad; "
        f"{n / elapsed:.0f} resets/s",
    )


def _check_hidden_physics(rec: Recorder, runner: ScenarioRunner, cfg: SceneConfig) -> None:
    c, p = cfg.checks, cfg.physics
    edge = cfg.block.edge
    distances: dict[tuple[float, float], float] = {}
    masses, frictions = _grid(p.mass_range), _grid(p.sliding_friction_range)
    for mass in masses:
        for mu in frictions:
            result = kick_test(runner, HiddenPhysics(mass, mu), seed=0, v0=c.kick_speed)
            applied = runner.applied_physics()
            normal_error = abs(result.normal_force - result.expected_normal_force) / result.expected_normal_force
            reasons = []
            if abs(result.relative_error) > c.kick_mu_rel_tolerance:
                reasons.append(f"friction error {result.relative_error:+.2%} > {c.kick_mu_rel_tolerance:.0%}")
            if normal_error > c.normal_force_rel_tolerance:
                reasons.append(f"normal force error {normal_error:.2%} > {c.normal_force_rel_tolerance:.0%}")
            if not result.contact_frictions or any(not math.isclose(f, mu, rel_tol=1e-12) for f in result.contact_frictions):
                reasons.append(f"contact friction {result.contact_frictions} != {mu}")
            if not np.allclose(result.inertia, mass * edge * edge / 6.0, rtol=1e-12, atol=0.0):
                reasons.append(f"inertia {result.inertia} != m s^2 / 6")
            if applied != HiddenPhysics(mass, mu):
                reasons.append(f"model holds {applied}")
            if result.final_speed >= 1e-3:
                reasons.append(f"block still sliding at {result.final_speed:.3g} m/s")
            distances[(mass, mu)] = result.distance
            rec.expect(
                f"kick m={mass:.3f} kg, mu={mu:.3f}",
                not reasons,
                f"slid {result.distance * 1e3:.1f} mm from {c.kick_speed:g} m/s -> mu_eff {result.mu_effective:.4f} "
                f"({result.relative_error:+.2%}); normal force {result.normal_force:.4f} N vs m*g "
                f"{result.expected_normal_force:.4f} N ({normal_error:.3%}); {len(result.contact_frictions)} contacts "
                f"at mu {mu:g}; stopped after {result.stop_time:.2f} s, yaw change {result.yaw_change:+.1e} rad"
                + (f"; {'; '.join(reasons)}" if reasons else ""),
                dataclasses.asdict(result),
            )
    for mu in frictions:
        values = [distances[(m, mu)] for m in masses]
        spread = (max(values) - min(values)) / float(np.mean(values))
        rec.expect(
            f"slide distance independent of mass (mu={mu:.3f})",
            spread <= c.kick_mu_rel_tolerance,
            f"distances {', '.join(f'{v * 1e3:.1f}' for v in values)} mm for m = {', '.join(f'{m:g}' for m in masses)} kg; "
            f"spread {spread:.2%} (limit {c.kick_mu_rel_tolerance:.0%})",
        )
    for mass in masses:
        values = [distances[(mass, mu)] for mu in frictions]
        rec.expect(
            f"higher friction, shorter slide (m={mass:.3f})",
            all(a > b for a, b in zip(values, values[1:])),
            f"{', '.join(f'{v * 1e3:.1f}' for v in values)} mm for mu = {', '.join(f'{mu:g}' for mu in frictions)}",
        )
    nominal = runner.nominal_physics
    for direction in (45.0, 90.0):
        result = kick_test(runner, nominal, seed=0, v0=c.kick_speed, direction_deg=direction)
        rec.expect(
            f"kick direction {direction:g} deg",
            abs(result.relative_error) <= c.kick_mu_rel_tolerance,
            f"mu_eff {result.mu_effective:.4f} vs {nominal.sliding_friction:g} ({result.relative_error:+.2%}); "
            f"elliptic friction cone is isotropic",
        )
    seen, out_of_range, not_applied, not_repeatable = [], [], [], []
    for seed in range(50):
        runner.reset(seed, randomize_physics=True)
        truth = runner.hidden_truth()
        physics = truth.physics
        if not (p.mass_range.contains(physics.block_mass) and p.sliding_friction_range.contains(physics.sliding_friction)):
            out_of_range.append(seed)
        if runner.applied_physics() != physics:
            not_applied.append(seed)
        runner.reset(seed, randomize_physics=True)
        if runner.hidden_truth() != truth:
            not_repeatable.append(seed)
        seen.append((physics.block_mass, physics.sliding_friction))
    a = np.array(seen)
    ok = not (out_of_range or not_applied or not_repeatable) and len(set(seen)) == len(seen)
    rec.expect(
        "randomized hidden physics",
        ok,
        f"seeds 0-49: mass {a[:, 0].min():.3f}..{a[:, 0].max():.3f} kg within {p.mass_range.as_list()}, "
        f"friction {a[:, 1].min():.3f}..{a[:, 1].max():.3f} within {p.sliding_friction_range.as_list()}; "
        f"{len(set(seen))} distinct; applied to the model and repeatable"
        + (f"; out of range {out_of_range}, not applied {not_applied}, not repeatable {not_repeatable}" if not ok else ""),
    )


def _check_observation(rec: Recorder, runner: ScenarioRunner, cfg: SceneConfig) -> None:
    c, p = cfg.checks, cfg.physics
    rec.expect(
        "allow-listed fields",
        OBSERVATION_FIELDS == EXPECTED_OBSERVATION_FIELDS,
        f"{len(OBSERVATION_FIELDS)} fields: {', '.join(OBSERVATION_FIELDS)}",
    )
    leaked = [name for name in OBSERVATION_FIELDS if any(word in name for word in FORBIDDEN_FIELD_WORDS)]
    rec.expect(
        "no hidden-quantity fields",
        not leaked,
        "no field names mass, friction, inertia, force, seed, scenario, truth, future state or labels"
        if not leaked
        else f"suspicious fields: {leaked}",
    )
    obs = runner.reset(0)
    values = vars(obs)
    bad = [
        name
        for name, value in values.items()
        if not (
            (isinstance(value, np.ndarray) and value.dtype == np.float64 and not value.flags.writeable)
            or (not isinstance(value, np.ndarray) and isinstance(value, (bool, int, float, str)))
        )
    ]
    rec.expect(
        "plain read-only values",
        not bad and set(values) == set(OBSERVATION_FIELDS),
        "every field is a read-only float64 array or a Python scalar; no MuJoCo objects"
        if not bad
        else f"unexpected values in {bad}",
    )
    model, data = runner.sim_model, runner.sim_data
    shared = [
        f"{name}~{owner}.{attr}"
        for name, value in values.items()
        if isinstance(value, np.ndarray)
        for owner, attrs in _SIM_ARRAYS.items()
        for attr in attrs
        if np.shares_memory(value, getattr(model if owner == "model" else data, attr))
    ]
    rec.expect(
        "copies, not views",
        not shared,
        f"no observation array shares memory with {sum(len(v) for v in _SIM_ARRAYS.values())} MjData/MjModel arrays"
        if not shared
        else f"shared memory: {shared}",
    )
    before = obs.to_json()
    runner.step(10)
    rec.expect("snapshot isolation", obs.to_json() == before, "an observation does not change when the simulation steps")
    errors = []
    try:
        obs.block_pos[0] = 1.0
        errors.append("array write allowed")
    except ValueError:
        pass
    try:
        obs.time = 1.0  # type: ignore[misc]
        errors.append("attribute write allowed")
    except dataclasses.FrozenInstanceError:
        pass
    rec.expect("immutable", not errors, "array and attribute writes raise" if not errors else "; ".join(errors))
    restored = Observation.from_dict(json.loads(obs.to_json()))
    rec.expect(
        "JSON round trip",
        restored.to_json() == obs.to_json() and restored.max_abs_difference(obs) == 0.0,
        "to_json -> from_dict reproduces the observation exactly",
    )
    rejected = []
    extra = obs.to_dict()
    extra["block_mass"] = 0.25
    missing = obs.to_dict()
    del missing["block_pos"]
    for label, payload in (("extra key block_mass", extra), ("missing key block_pos", missing)):
        try:
            Observation.from_dict(payload)
        except ValueError:
            rejected.append(label)
    rec.expect("schema enforced", len(rejected) == 2, f"from_dict rejects: {', '.join(rejected)}")

    variants: list[tuple[str, dict[str, Any]]] = [
        ("randomized", {"randomize_physics": True}),
        ("light/low friction", {"physics": HiddenPhysics(p.mass_range.low, p.sliding_friction_range.low)}),
        ("heavy/high friction", {"physics": HiddenPhysics(p.mass_range.high, p.sliding_friction_range.high)}),
    ]
    worst = 0.0
    for seed in (0, 1, 2):
        reference = runner.reset(seed)
        for _, kwargs in variants:
            worst = max(worst, runner.reset(seed, **kwargs).max_abs_difference(reference))
    rec.expect(
        "hidden physics invisible at reset",
        worst == 0.0,
        f"seeds 0-2, nominal vs {', '.join(v[0] for v in variants)}: max observation difference {worst:.3g}",
    )
    for updates in (50, 250):
        runner.reset(0)
        reference = runner.step(updates)
        per_field: dict[str, float] = {}
        for _, kwargs in variants:
            runner.reset(0, **kwargs)
            for name, diff in _field_differences(reference, runner.step(updates)).items():
                per_field[name] = max(per_field.get(name, 0.0), diff)
        name, diff = max(per_field.items(), key=lambda item: item[1])
        rec.expect(
            f"hidden physics at rest after {updates * cfg.timing.control_dt:g} s",
            diff <= c.observation_rest_tolerance,
            f"max observation difference {diff:.3g} ({name}) across hidden physics "
            f"(limit {c.observation_rest_tolerance:g}); the block rests at the same height for every mass",
            {"per_field": per_field},
        )
    rec.info(
        "pose source",
        f"poses come from simulator state (pose_source = {obs.pose_source!r}); contact forces are not observed",
    )


def _judge_hold(report: HoldReport, cfg: SceneConfig) -> list[str]:
    c = cfg.checks
    reasons = []
    if not report.stable:
        reasons.append(report.reason)
    if report.final_max_abs_qvel > c.hold_max_final_qvel:
        reasons.append(f"final |qvel| {report.final_max_abs_qvel:.2e} > {c.hold_max_final_qvel:g}")
    if report.block_planar_drift > c.hold_max_block_drift:
        reasons.append(f"block drift {report.block_planar_drift:.2e} m > {c.hold_max_block_drift:g}")
    if report.arm_max_deviation > c.hold_max_arm_deviation:
        reasons.append(f"arm deviation {report.arm_max_deviation:.4f} rad > {c.hold_max_arm_deviation:g}")
    if report.robot_contacts:
        reasons.append("robot touched the block or the table")
    return reasons


def _hold_line(report: HoldReport) -> str:
    return (
        f"{report.seconds:g} s: max |qvel| {report.max_abs_qvel:.2e}, final {report.final_max_abs_qvel:.2e}; "
        f"block drift {report.block_planar_drift:.1e} m, settled {report.block_height_change * 1e3:+.4f} mm; "
        f"arm sag {report.arm_max_deviation:.4f} rad; {report.realtime_factor:.0f}x real time"
    )


def _check_stability(rec: Recorder, runner: ScenarioRunner, cfg: SceneConfig) -> None:
    c = cfg.checks
    holds = [(seed, randomize, c.hold_seconds) for seed in range(c.hold_seeds) for randomize in (False, True)]
    holds += [(0, False, c.long_hold_seconds), (1, True, c.long_hold_seconds)]
    for seed, randomize, seconds in holds:
        report = hold_test(runner, seed, seconds=seconds, randomize_physics=randomize)
        reasons = _judge_hold(report, cfg)
        rec.expect(
            f"hold seed {seed}, {report.physics_source} physics, {seconds:g} s",
            not reasons,
            _hold_line(report) + (f"; {'; '.join(reasons)}" if reasons else ""),
            dataclasses.asdict(report),
        )
    model, data = runner.sim_model, runner.sim_data
    block_dof = int(model.jnt_dofadr[model.joint(cfg.block.joint).id])
    runner.reset(0)
    reference = runner.state_vector().copy()
    limit = cfg.stability.max_abs_qvel

    def nan_velocity() -> None:
        data.qvel[0] = np.nan

    def runaway_velocity() -> None:
        data.qvel[block_dof] = 4.0 * limit

    for label, corrupt in (("NaN joint velocity", nan_velocity), (f"block at {4 * limit:g} m/s", runaway_velocity)):
        runner.reset(0)
        with capture_mujoco_warnings() as mujoco_warnings:
            corrupt()
            try:
                runner.step(1)
                raised = "not detected"
            except SimulationUnstable as exc:
                raised = str(exc)
        runner.reset(0)
        recovered = runner.stability().ok and _max_diff(runner.state_vector(), reference) == 0.0
        captured = f"; MuJoCo warning captured: {mujoco_warnings[0]!r}" if mujoco_warnings else ""
        rec.expect(
            f"instability detector: {label}",
            raised != "not detected" and recovered,
            f"SimulationUnstable raised ({raised}){captured}; reset(0) restores the exact initial state: {recovered}",
        )
    response = actuator_response_test(
        runner,
        seed=0,
        arm_step=c.actuator_step,
        seconds=c.actuator_seconds,
        arm_tolerance=c.actuator_tracking_tolerance,
        gripper_tolerance=c.gripper_tolerance,
    )
    rec.expect(
        "actuator response",
        response.ok,
        f"{response.reason}: +{response.arm_step:g} rad on all 7 joints -> worst error {response.arm_max_error:.4f} rad "
        f"after {response.seconds:g} s (limit {c.actuator_tracking_tolerance:g}); gripper open "
        f"{response.gripper_open_opening * 1e3:.2f} mm (expected {response.gripper_open_expected * 1e3:.2f}), "
        f"closed {response.gripper_closed_opening * 1e3:.3f} mm",
        dataclasses.asdict(response),
    )


def _check_reach(rec: Recorder, runner: ScenarioRunner) -> None:
    report = check_reach(runner)
    detail = report.summary()
    if report.failures:
        detail += "; failing targets " + ", ".join(str(tuple(round(v, 3) for v in f.target)) for f in report.failures[:5])
    rec.expect("TCP-down reach over the block region", report.ok, detail)


def _check_speed(rec: Recorder, runner: ScenarioRunner, cfg: SceneConfig) -> None:
    dt = cfg.timing.control_dt
    updates = round(10.0 / dt)
    runner.reset(0)
    started = time.perf_counter()
    runner.step(updates)
    wall = time.perf_counter() - started
    steps = updates * cfg.timing.substeps
    rec.info(
        "headless stepping speed",
        f"{updates * dt:g} s simulated ({steps} physics steps) in {wall:.3f} s: {updates * dt / wall:.0f}x real time, "
        f"{steps / wall:.0f} steps/s (this process; varies with load)",
    )


def _check_rendering(rec: Recorder, runner: ScenarioRunner, cfg: SceneConfig, out_dir: Path) -> Path | None:
    from .render import DEFAULT_SIZE, gl_info, image_stats, is_nonblank, render_cameras, save_renders

    try:
        gl = gl_info()
    except Exception as exc:  # strings are informational; rendering below is the real check
        gl = {"error": f"{type(exc).__name__}: {exc}"}
    rec.info("OpenGL", " | ".join(f"{k}: {v}" for k, v in gl.items()), gl)
    runner.reset(0)
    obs = runner.step(round(0.5 / cfg.timing.control_dt))
    images = render_cameras(runner.sim_model, runner.sim_data, cfg.cameras, DEFAULT_SIZE)
    nonblank = {}
    for name, image in images.items():
        stats = image_stats(image)
        nonblank[name] = is_nonblank(image, stats)
        rec.expect(
            f"camera {name}",
            nonblank[name],
            f"{image.shape[1]}x{image.shape[0]} px, mean {stats['mean']:.1f}, std {stats['std']:.1f}, "
            f"{stats['unique_colours']:.0f} colours",
            stats,
        )
    caption = (
        f"Phase 1 push scene | seed 0, nominal hidden physics, t = {obs.time:.2f} s | "
        f"block-target {obs.block_target_distance * 1e3:.0f} mm | MuJoCo {mujoco.__version__}"
    )
    written = save_renders(images, out_dir, caption)
    rec.info("screenshots", ", ".join(_rel(path) for path in written.values()))
    return written["composite"] if all(nonblank.values()) else None


def _check_camera_views(rec: Recorder, runner: ScenarioRunner, cfg: SceneConfig) -> None:
    from .render import VISIBILITY_SIZE, VisibilityProbe

    c = cfg.checks
    block, goal = cfg.block.geom, cfg.target.zone_geom
    seeds = range(c.visibility_seeds)
    pairs = [(camera, geom) for camera in cfg.cameras for geom in (block, goal)]
    worst: dict[tuple[str, str], tuple[float, int]] = {}
    not_in_view: dict[tuple[str, str], list[int]] = {pair: [] for pair in pairs}
    with VisibilityProbe(runner.sim_model, cfg.robot.tcp_body) as probe:
        for seed in seeds:
            runner.reset(seed)
            for v in probe.measure(runner.sim_data, cfg.cameras, (block, goal)):
                pair = (v.camera, v.geom)
                if not v.in_view:
                    not_in_view[pair].append(seed)
                elif pair not in worst or v.robot_occlusion > worst[pair][0]:
                    worst[pair] = (v.robot_occlusion, seed)
        robot_geoms = probe.robot_geom_count
    rec.info(
        "method",
        f"{VISIBILITY_SIZE[0]}x{VISIBILITY_SIZE[1]} segmentation renders at reset (arm in the home pose), "
        f"seeds 0..{c.visibility_seeds - 1}; robot occlusion = 1 - pixels / pixels with the robot's "
        f"{robot_geoms} geoms not drawn",
    )
    limit = c.max_arm_occlusion
    for camera, geom in pairs:
        missing = not_in_view[(camera, geom)]
        value, seed = worst.get((camera, geom), (math.nan, -1))
        rec.expect(
            f"{camera}: robot hides <= {limit:.0%} of {geom}",
            value <= limit,
            f"worst {value:.1%} (seed {seed}) over the {len(seeds) - len(missing)} seeds where it is in view"
            if seed >= 0
            else "never in view",
            {"worst": value, "worst_seed": seed, "seeds_in_view": len(seeds) - len(missing)},
        )
    block_missing = {camera: not_in_view[(camera, block)] for camera in cfg.cameras}
    rec.expect(
        f"{block} in view in every camera",
        not any(block_missing.values()),
        "; ".join(f"{camera}: {len(s)} seeds not in view {s[:10]}" for camera, s in block_missing.items() if s)
        or f"all {len(seeds)} seeds, {len(cfg.cameras)} cameras",
        {"not_in_view": block_missing},
    )
    goal_unseen = [s for s in seeds if all(s in not_in_view[(camera, goal)] for camera in cfg.cameras)]
    rec.expect(
        f"{goal} in view in at least one camera",
        not goal_unseen,
        f"not in view in any camera for seeds {goal_unseen[:10]}" if goal_unseen else f"all {len(seeds)} seeds",
        {"seeds_not_in_view": goal_unseen},
    )
    for camera in cfg.cameras:
        hidden = not_in_view[(camera, goal)]
        if hidden:
            rec.info(
                f"{camera}: {goal} not visible even with the robot not drawn",
                f"{len(hidden)}/{len(seeds)} seeds {hidden[:10]}; another scene object or the frame edge, not the arm",
                {"seeds": hidden},
            )


# --------------------------------------------------------------------------- driver


def run_smoke(
    out_dir: Path | None = None,
    *,
    render: bool = True,
    command: str | None = None,
    echo: Callable[[str], None] = print,
) -> SmokeOutcome:
    started_at = datetime.now(timezone.utc)
    out_dir = Path(out_dir) if out_dir else paths.RUNS_DIR / "smoke" / started_at.strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path, result_path = out_dir / "smoke_log.txt", out_dir / "smoke_result.json"
    rec = Recorder(log_path, echo)
    started = time.perf_counter()
    env = environment_info()
    cfg = load_scene_config()
    manifest = assets.load_manifest()["source"]
    hashes = {_rel(p): _sha256(p) for p in (cfg.source, cfg.mjcf_path, paths.PANDA_MANIFEST)}

    rec.line("Physical AI Lab - Phase 1 smoke test")
    rec.line(f"command      : {command or ' '.join([Path(sys.executable).name, *sys.argv])}")
    rec.line(f"started (UTC): {started_at.isoformat(timespec='seconds')}")
    rec.line(f"package      : physical_ai_lab {__version__}")
    rec.line(
        f"python       : {env['python']} ({env['process_bits']}-bit {env['process_machine']}; host CPU "
        f"{env['host_machine']}{', emulated' if env['emulated'] else ''}) {env['python_executable']}"
    )
    rec.line(f"os           : {env['os']}")
    rec.line(f"versions     : {', '.join(f'{k} {v}' for k, v in env['versions'].items())}")
    rec.line(f"asset        : {manifest['repository_slug']}@{manifest['commit']} ({manifest['path']}, {manifest['license']})")
    for rel, digest in hashes.items():
        rec.line(f"sha256       : {digest}  {rel}")
    rec.line(f"git          : {env['git_root'] or 'not a git repository'}")
    rec.line(f"output       : {_rel(out_dir)}")

    sections: list[tuple[str, str, Callable[..., Any]]] = [
        ("environment", "Environment", lambda: _check_environment(rec, env)),
        ("assets", "Vendored Panda asset", lambda: _check_assets(rec)),
    ]
    runner_box: dict[str, ScenarioRunner] = {}

    def scene() -> None:
        runner_box["runner"] = _check_scene(rec, cfg)

    def needs_runner(fn: Callable[..., Any], *args: Any) -> Callable[[], Any]:
        def call() -> Any:
            if "runner" not in runner_box:
                rec.skip("not run", "the scene did not load")
                return None
            return fn(rec, runner_box["runner"], *args)

        return call

    screenshot: dict[str, Path | None] = {"path": None}

    def rendering() -> None:
        if not render:
            rec.skip("offscreen rendering", "--no-render given; no screenshot produced")
            return
        screenshot["path"] = needs_runner(_check_rendering, cfg, out_dir)()

    def camera_views() -> None:
        if not render:
            rec.skip("camera views", "--no-render given")
            return
        needs_runner(_check_camera_views, cfg)()

    sections += [
        ("scene", "Scene", scene),
        ("reset", "Seeded reset reproducibility", needs_runner(_check_reset, cfg)),
        ("layouts", f"Reset layouts ({cfg.checks.layout_seeds} seeds)", needs_runner(_check_layouts, cfg)),
        ("hidden_physics", "Hidden physics (kick test)", needs_runner(_check_hidden_physics, cfg)),
        ("observation", "Observation boundary", needs_runner(_check_observation, cfg)),
        ("stability", "Stability, instability detector, actuators", needs_runner(_check_stability, cfg)),
        ("reach", "Reachability", needs_runner(_check_reach)),
        ("speed", "Headless speed", needs_runner(_check_speed, cfg)),
        ("rendering", "Offscreen rendering", rendering),
        ("camera_views", f"Camera views: arm occlusion at reset ({cfg.checks.visibility_seeds} seeds)", camera_views),
    ]
    for key, title, fn in sections:
        rec.begin(key, title)
        section_started = time.perf_counter()
        try:
            fn()
        except Exception as exc:
            rec.record(FAIL, "unexpected error", f"{type(exc).__name__}: {exc}", {"traceback": traceback.format_exc()})
            for text in traceback.format_exc().rstrip().splitlines():
                rec.line(f"    {text}")
        rec.line(f"   ({time.perf_counter() - section_started:.2f} s)")

    counts = {status: sum(r.status == status for r in rec.results) for status in (PASS, FAIL, WARN, INFO, SKIP)}
    ok = counts[FAIL] == 0
    gate_sections = ("assets", "scene", "reset", "layouts", "stability", "rendering")
    gate = ok and screenshot["path"] is not None and not any(rec.section_failed(s) for s in gate_sections)
    finished_at = datetime.now(timezone.utc)
    rec.line()
    rec.line("== Summary")
    rec.line("  ".join(f"{status} {count}" for status, count in counts.items()))
    for result in rec.results:
        if result.status == FAIL:
            rec.line(f"FAILED [{result.section}] {result.name}: {result.detail}")
    rec.line(
        "Phase 1 gate (scene loads, resets reproducibly, steps stably, screenshot saved): "
        + ("PASSED" if gate else "NOT PASSED")
        + ("" if screenshot["path"] is not None else " - no screenshot was produced")
    )
    rec.line(
        "Not covered here: the interactive viewer window (from the project folder run "
        f"`{paths.python_command()} -m physical_ai_lab view --seed 0`) and "
        "determinism across machines, CPU architectures or MuJoCo builds."
    )
    rec.line(f"finished (UTC): {finished_at.isoformat(timespec='seconds')} ({time.perf_counter() - started:.1f} s)")
    rec.line(f"log: {_rel(log_path)}  results: {_rel(result_path)}")
    rec.close()

    result = {
        "schema_version": 1,
        "ok": ok,
        "phase1_gate_passed": gate,
        "counts": counts,
        "started_utc": started_at.isoformat(timespec="seconds"),
        "finished_utc": finished_at.isoformat(timespec="seconds"),
        "command": command,
        "environment": env,
        "asset_source": manifest,
        "sha256": hashes,
        "screenshot": _rel(screenshot["path"]) if screenshot["path"] else None,
        "checks": [dataclasses.asdict(r) for r in rec.results],
    }
    result_path.write_text(json.dumps(result, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    return SmokeOutcome(ok, gate, counts, out_dir, log_path, result_path, screenshot["path"])
