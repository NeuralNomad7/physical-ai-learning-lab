"""Command line: ``python -m physical_ai_lab <command>``.

Phase 1 commands: doctor, fetch-assets, view, step, render, smoke.
Phase 2 commands: gate (headless baseline gate), demo (watch a baseline episode).
Phase 3 commands: contact-check (effective-contact and feasibility checks), manifests (generate or
--verify the benchmark manifests), evaluate (a policy on a manifest split, through the learner
interface), replay (determinism check of an evaluation run).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, paths

PROG = "python -m physical_ai_lab"


def open_scene_command() -> str:
    """The exact command that opens the scene with this interpreter, run from the project root."""
    return f"{paths.python_command()} -m physical_ai_lab view --seed 0"


def _seed(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"seed must be a non-negative integer, got {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"seed must be a non-negative integer, got {value}")
    return value


def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of seconds, got {text!r}") from None
    if not (math.isfinite(value) and value >= 0):
        raise argparse.ArgumentTypeError(f"seconds must be finite and non-negative, got {text}")
    return value


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _speed(text: str) -> float:
    value = _seconds(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"speed must be positive, got {text}")
    return value


# --------------------------------------------------------------------------- doctor


def cmd_doctor(args: argparse.Namespace) -> int:
    from . import assets
    from .environment import environment_info

    failures = 0

    def report(level: str, label: str, text: str) -> None:
        nonlocal failures
        failures += level == "FAIL"
        print(f"[{level}] {label}: {text}")

    env = environment_info()
    report(
        "INFO",
        "python",
        f"{env['python']} {env['process_bits']}-bit {env['process_machine']} ({env['python_platform']}) "
        f"at {env['python_executable']}",
    )
    report("INFO", "os", f"{env['os']} (native CPU {env['host_machine']})")
    if env["emulated"]:
        why = (
            " (MuJoCo 3.14.0 on PyPI has win_amd64 wheels only; checked 2026-10-02)"
            if sys.platform == "win32" and env["host_machine"] == "ARM64"
            else ""
        )
        report("INFO", "emulation", f"this {env['process_machine']} Python runs under emulation on {env['host_machine']}{why}")
    report("INFO", "versions", ", ".join(f"{k} {v}" for k, v in env["versions"].items()))
    mismatches = env["pin_mismatches"]
    report(
        "FAIL" if mismatches else "PASS",
        "dependency pins",
        f"differ from pyproject.toml: {mismatches}; run `uv sync`" if mismatches else "installed versions match pyproject.toml",
    )
    verify = assets.verify()
    report("PASS" if verify.ok else "FAIL", "Panda asset", verify.summary())

    try:
        from .scenario import ScenarioRunner

        started = time.perf_counter()
        runner = ScenarioRunner()
        model = runner.sim_model
        report(
            "PASS",
            "scene",
            f"{paths.SCENE_XML.name} loaded and verified in {time.perf_counter() - started:.2f} s "
            f"(nq {model.nq}, nv {model.nv}, nu {model.nu})",
        )
    except Exception as exc:
        report("FAIL", "scene", f"{type(exc).__name__}: {exc}")
        return 1

    try:
        runner.reset(0)
        updates = round(1.0 / runner.config.timing.control_dt)
        started = time.perf_counter()
        runner.step(updates)
        wall = time.perf_counter() - started
        report("PASS", "headless step", f"1 s simulated in {wall:.3f} s ({1.0 / wall:.0f}x real time); {runner.stability().reason}")
    except Exception as exc:
        report("FAIL", "headless step", f"{type(exc).__name__}: {exc}")

    try:
        from .render import gl_info, is_nonblank, render_cameras

        gl = gl_info()
        images = render_cameras(runner.sim_model, runner.sim_data, runner.config.cameras, (320, 240))
        blank = [name for name, image in images.items() if not is_nonblank(image)]
        report(
            "FAIL" if blank else "PASS",
            "offscreen rendering",
            (f"blank images from {blank}; " if blank else f"{len(images)} cameras rendered; ")
            + f"OpenGL {gl.get('version', '?')} on {gl.get('renderer', '?')}",
        )
    except Exception as exc:
        hint = "" if sys.platform == "win32" else " (on headless Linux try MUJOCO_GL=egl or MUJOCO_GL=osmesa)"
        report("FAIL", "offscreen rendering", f"{type(exc).__name__}: {exc}{hint}")

    report(
        "INFO",
        "viewer",
        f"not opened by doctor (needs a desktop session); from the project folder run `{open_scene_command()}`",
    )
    if env["git_root"] is None:
        report("WARN", "git", "the project folder is not a git repository; changes are not tracked (see docs/PROGRESS.md)")
    if env["onedrive_folder"]:
        report(
            "WARN",
            "OneDrive",
            "project is in a OneDrive-synced folder; if `uv sync` fails with 'Access is denied', see README.md",
        )
    print("doctor: " + ("all required checks passed" if not failures else f"{failures} check(s) FAILED"))
    return 1 if failures else 0


# --------------------------------------------------------------------------- assets


def cmd_fetch_assets(args: argparse.Namespace) -> int:
    from . import assets

    before = assets.verify()
    if before.ok and not args.force:
        print(before.summary())
        print("Nothing to download (use --force to download every file again).")
        return 0
    print(f"Downloading from {assets.REPOSITORY} at the pinned commit {before.commit} ...")
    after = assets.fetch(force=args.force)
    print(after.summary())
    return 0 if after.ok else 1


# --------------------------------------------------------------------------- viewer


def cmd_view(args: argparse.Namespace) -> int:
    import numpy as np

    from .scenario import ScenarioRunner
    from .viewer import replay, run_viewer

    try:
        summary = run_viewer(
            args.seed,
            randomize_physics=args.randomize_physics,
            duration=args.duration,
            screenshot=args.screenshot,
            screenshot_delay=args.screenshot_delay,
            show_ui=args.show_ui,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    pace = (
        f"{summary.realtime_factor:.2f}x real time while running"
        if summary.realtime_factor is not None
        else "too short to measure the pace"
    )
    print(
        f"Viewer closed ({summary.closed_by}) after {summary.wall_seconds:.1f} s"
        + (f" ({summary.paused_seconds:.1f} s paused)" if summary.paused_seconds >= 0.05 else "")
        + f": {summary.control_updates} control updates ({summary.sim_seconds:.2f} s simulated, {pace}), "
        f"{summary.operator_commands} operator command(s), "
        f"{summary.restores} scenario restore(s), {summary.option_reverts} physics-option revert(s), "
        f"{summary.unstable_events} instability event(s), "
        f"{summary.resyncs} stall resync(s); window closed in {summary.close_seconds:.2f} s."
    )
    print(
        f"Final state: t = {summary.final_time:.2f} s since the last reset, block-target distance "
        f"{summary.final_block_target_distance * 1e3:.1f} mm, state digest {summary.final_state_digest}"
    )
    commands = summary.commands_since_reset
    headless = ScenarioRunner()
    replay(
        headless,
        args.seed,
        randomize_physics=args.randomize_physics,
        updates=summary.updates_since_reset,
        commands=commands,
    )
    digest = headless.state_digest()
    what = f"Headless replay of {summary.updates_since_reset} updates from seed {args.seed}" + (
        f" with the {len(commands)} recorded operator command(s) since the last reset" if commands else ""
    )
    if digest == summary.final_state_digest:
        print(
            f"{what}: identical state (digest {digest}); "
            + (
                "the arm moved only through its position servos and the viewer did not change the physics."
                if commands
                else "the viewer did not change the physics."
            )
        )
    else:
        difference = float(np.max(np.abs(headless.state_vector() - summary.final_state)))
        print(
            f"{what} differs (digest {digest}, max |state difference| {difference:.3g}); expected only if a body "
            "was pushed with Ctrl+right-drag in the viewer."
        )
    if summary.screenshot is not None:
        shot = summary.screenshot
        if shot.get("saved"):
            check = {True: " (status panels pixel-exact)", False: " (status panels differ from those sent)"}
            print(f"Screenshot: {shot['path']}{check.get(shot.get('overlay_exact'), '')}")
        else:
            print(f"Screenshot not saved: {shot.get('error')}")
    return 0 if summary.unstable_events == 0 else 1


# --------------------------------------------------------------------------- step


def cmd_step(args: argparse.Namespace) -> int:
    from .scenario import ScenarioRunner, SimulationUnstable

    runner = ScenarioRunner()
    runner.reset(args.seed, randomize_physics=args.randomize_physics)
    dt = runner.config.timing.control_dt
    updates = round(args.seconds / dt)
    started = time.perf_counter()
    try:
        obs = runner.step(updates)
        failure = None
    except SimulationUnstable as exc:
        obs, failure = runner.observe(), str(exc)
    wall = time.perf_counter() - started
    stability = runner.stability()
    truth = runner.hidden_truth()
    result = {
        "seed": args.seed,
        "physics_source": truth.physics_source,
        "control_updates": runner.control_step,
        "sim_time": obs.time,
        "wall_seconds": wall,
        "realtime_factor": obs.time / wall if wall > 0 else None,
        "stable": failure is None and stability.ok,
        "reason": failure or stability.reason,
        "max_abs_qvel": stability.max_abs_qvel,
        "state_digest": runner.state_digest(),
        "observation": obs.to_dict(),
    }
    if args.show_truth:
        result["hidden_truth"] = truth.to_dict()
    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if result["stable"] else 1

    label = truth.physics_source if args.show_truth or truth.physics_source == "nominal" else "randomized (values hidden)"
    print(f"seed {args.seed}, hidden physics {label}")
    print(
        f"stepped {result['control_updates']} control updates = {obs.time:.2f} s simulated in {wall:.3f} s wall"
        + (f" ({obs.time / wall:.0f}x real time)" if wall > 0 and obs.time > 0 else "")
    )
    print(f"stable: {'yes' if result['stable'] else 'NO'} ({result['reason']}; max |qvel| {stability.max_abs_qvel:.3g})")
    print(f"state digest: {result['state_digest']}")
    print(
        f"block at ({obs.block_pos[0]:.4f}, {obs.block_pos[1]:.4f}, {obs.block_pos[2]:.4f}) m, yaw {obs.block_yaw:+.4f} rad; "
        f"target at ({obs.target_pos[0]:.4f}, {obs.target_pos[1]:.4f}) m; distance {obs.block_target_distance * 1e3:.1f} mm "
        f"(success radius {obs.target_tolerance * 1e3:.0f} mm)"
    )
    print(
        f"TCP at ({obs.tcp_pos[0]:.4f}, {obs.tcp_pos[1]:.4f}, {obs.tcp_pos[2]:.4f}) m; gripper opening "
        f"{obs.gripper_opening * 1e3:.2f} mm; contacts robot/block {obs.robot_block_contact}, robot/table {obs.robot_table_contact}"
    )
    if args.show_truth:
        physics = truth.physics
        print(f"hidden truth: block mass {physics.block_mass:.4f} kg, block/table friction {physics.sliding_friction:.4f}")
    return 0 if result["stable"] else 1


# --------------------------------------------------------------------------- render


def cmd_render(args: argparse.Namespace) -> int:
    import mujoco

    from .render import DEFAULT_SIZE, gl_info, is_nonblank, render_cameras, save_renders
    from .scenario import ScenarioRunner

    runner = ScenarioRunner()
    runner.reset(args.seed, randomize_physics=args.randomize_physics)
    obs = runner.step(round(args.seconds / runner.config.timing.control_dt))
    images = render_cameras(runner.sim_model, runner.sim_data, runner.config.cameras, DEFAULT_SIZE)
    physics = "randomized hidden physics" if args.randomize_physics else "nominal hidden physics"
    caption = (
        f"seed {args.seed}, {physics}, t = {obs.time:.2f} s | block-target {obs.block_target_distance * 1e3:.0f} mm | "
        f"MuJoCo {mujoco.__version__}"
    )
    out_dir = args.out or paths.RUNS_DIR / "render" / _stamp()
    written = save_renders(images, Path(out_dir), caption)
    for name, path in written.items():
        print(f"{name}: {path}")
    gl = gl_info()
    print(f"OpenGL {gl.get('version', '?')} on {gl.get('renderer', '?')}")
    blank = [name for name, image in images.items() if not is_nonblank(image)]
    if blank:
        print(f"error: blank render from {blank}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- smoke


def cmd_smoke(args: argparse.Namespace) -> int:
    from .smoke import run_smoke

    command = PROG + " smoke" + (f" --out {args.out}" if args.out else "") + (" --no-render" if args.no_render else "")
    outcome = run_smoke(args.out, render=not args.no_render, command=command)
    return 0 if outcome.ok else 1


# --------------------------------------------------------------------------- Phase 2: gate and demo


def cmd_gate(args: argparse.Namespace) -> int:
    from .evaluation import run_evaluation
    from .push_config import load_push_config

    push = load_push_config()
    seeds = push.seeds.gate if args.seeds == "gate" else push.seeds.development
    out_dir = Path(args.out) if args.out else paths.RUNS_DIR / "phase2" / args.seeds / _stamp()
    command = PROG + " gate" + (f" --seeds {args.seeds}" if args.seeds != "gate" else "") + (
        f" --out {args.out}" if args.out else ""
    )
    summary = run_evaluation(seeds, seed_set=args.seeds, out_dir=out_dir, push=push, command=command)
    if args.seeds == "gate":
        return 0 if summary["gate_passed"] else 1
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import headless_matches, render_episode_frames, run_demo
    from .evaluation import episode_line

    if args.frames is not None:
        record, written = render_episode_frames(args.seed, Path(args.frames), mode=args.mode)
        print(episode_line(record))
        for label, path in written.items():
            print(f"{label}: {path}")
        return 0
    try:
        summary = run_demo(
            args.seed, mode=args.mode, speed=args.speed, duration=args.duration, linger=args.linger, show_ui=args.show_ui
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Viewer closed ({summary.closed_by}) after {summary.wall_seconds:.1f} s; "
        f"{summary.option_reverts} physics-option revert(s); final state digest {summary.state_digest}"
    )
    match = headless_matches(summary)
    if match is None:
        print("Headless check skipped (episode interrupted before it finished).")
    elif match:
        print(
            f"Headless rerun of seed {args.seed}: identical outcome and state (digest {summary.headless_digest}); "
            "the viewer did not change the episode."
        )
    else:
        print(
            f"Headless rerun of seed {args.seed} differs: {episode_line(summary.headless_record)} "
            f"(digest {summary.headless_digest}); expected only if a body was pushed with Ctrl+right-drag in the viewer."
        )
    return 0


# --------------------------------------------------------------------------- Phase 3: benchmark


def cmd_contact_check(args: argparse.Namespace) -> int:
    from .benchmark_config import load_benchmark_config
    from .config import load_scene_config
    from .contact_checks import run_contact_checks, write_contact_checks
    from .push_config import load_push_config

    scene, push = load_scene_config(), load_push_config()
    bench = load_benchmark_config(scene=scene, push=push)
    out_dir = Path(args.out) if args.out else paths.RUNS_DIR / "phase3" / "contact_checks" / _stamp()
    print(f"Effective-contact and feasibility checks (benchmark config {bench.sha256[:16]}); writing to {out_dir}")
    report = run_contact_checks(scene, push, bench, progress=lambda line: print(line, flush=True))
    report["command"] = PROG + " contact-check" + (f" --out {args.out}" if args.out else "")
    path = write_contact_checks(report, out_dir)
    a = report["assessment"]
    sections = {k: sum(1 for r in v if r["ok"]) for k, v in report["results"].items()}
    totals = {k: len(v) for k, v in report["results"].items()}
    print("passed: " + ", ".join(f"{k} {sections[k]}/{totals[k]}" for k in sections))
    print(f"failing values: {a['failing_values']}")
    print(f"feasible box: mass {a['feasible_box']['mass']}, friction {a['feasible_box']['sliding_friction']} "
          f"(candidate box {'accepted' if a['candidate_box_accepted'] else 'REDUCED'})")
    print(f"wall time {report['wall_seconds']:.1f} s; report {path}")
    return 0 if all(sections[k] == totals[k] for k in sections) else 1


def _load_benchmark():
    from .benchmark_config import load_benchmark_config
    from .config import load_scene_config
    from .push_config import load_push_config

    scene, push = load_scene_config(), load_push_config()
    return scene, push, load_benchmark_config(scene=scene, push=push)


def cmd_manifests(args: argparse.Namespace) -> int:
    from . import manifest as mf

    scene, _push, bench = _load_benchmark()
    out_dir = Path(args.out) if args.out else paths.MANIFESTS_DIR
    if args.verify:
        manifests = {s: mf.read_sealed_for_audit(s, out_dir) for s in mf.GENERATION_ORDER}
        report = mf.check_disjoint(manifests, bench)
        hashes = mf.load_hashes(out_dir)
        stale = [s for s, m in manifests.items() if m.data["benchmark"]["config_sha256"] != bench.sha256]
        print(f"manifests in {out_dir}: hashes match hashes.json; disjointness checks passed "
              f"({report['layout_pairs_checked']} initial-state pairs compared)")
        for split, entry in hashes["manifests"].items():
            print(f"  {split:<10} {entry['sha256']}  conditions {entry['conditions']:>3}  scenarios {entry['scenarios']:>3}"
                  f"  {entry['by_cohort']}  {'SEALED' if entry['sealed'] else 'open'}")
        if stale:
            print(f"STALE: {stale} were generated with a different benchmark.toml", file=sys.stderr)
            return 1
        return 0
    if (out_dir / mf.HASHES_FILE).exists():
        print(f"REFUSED: manifests already exist in {out_dir} ({mf.HASHES_FILE}); they are frozen. Regenerating "
              "them needs a revised, declared benchmark (BUILD_PLAN.md section 6). Use --verify to check them.",
              file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"Generating benchmark manifests ({bench.benchmark_version}, config {bench.sha256[:16]}) into {out_dir}")
    docs = mf.generate_manifests(scene, bench, mf.manifest_provenance(scene))
    command = PROG + " manifests" + (f" --out {args.out}" if args.out else "")
    record = mf.write_manifests(docs, out_dir, command=command)
    manifests = {s: mf.read_sealed_for_audit(s, out_dir) for s in mf.GENERATION_ORDER}
    report = mf.check_disjoint(manifests, bench)
    print(f"started {started}; written {record['generated_utc']}; disjointness checks passed "
          f"({report['layout_pairs_checked']} initial-state pairs compared)")
    for split, entry in record["manifests"].items():
        print(f"  {split:<10} {entry['sha256']}  conditions {entry['conditions']:>3}  scenarios {entry['scenarios']:>3}"
              f"  {entry['by_cohort']}  {'SEALED' if entry['sealed'] else 'open'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .benchmark import evaluate_split
    from .manifest import ManifestError, SealedSplitError

    scene, push, bench = _load_benchmark()
    out_dir = Path(args.out) if args.out else paths.RUNS_DIR / "phase3" / "evaluate" / args.split / _stamp()
    command = f"{PROG} evaluate --split {args.split} --mode {args.mode}"
    command += (f" --limit {args.limit}" if args.limit is not None else "") + (f" --out {args.out}" if args.out else "")
    try:
        summary = evaluate_split(args.split, mode=args.mode, out_dir=out_dir, scene=scene, push=push, bench=bench,
                                 limit=args.limit, command=command, progress=lambda line: print(line, flush=True))
    except SealedSplitError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 1
    print(f"report: {out_dir / 'summary.txt'}")
    return 0 if not summary["partial"] or args.limit is not None else 1


def _positive_int(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def cmd_replay(args: argparse.Namespace) -> int:
    from .benchmark import replay_run
    from .manifest import SealedSplitError

    scene, push, bench = _load_benchmark()
    run_dir = Path(args.run)
    if not (run_dir / "summary.json").is_file():
        print(f"{run_dir} is not an evaluation run folder (no summary.json)", file=sys.stderr)
        return 1
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        report = replay_run(run_dir, scene=scene, push=push, bench=bench, count=args.count,
                            progress=lambda line: print(line, flush=True))
    except SealedSplitError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    report["started_utc"] = started
    report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["command"] = f"{PROG} replay --run {args.run}" + (f" --count {args.count}" if args.count else "")
    out = Path(args.out) if args.out else run_dir / "replay.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{'ALL IDENTICAL' if report['all_match'] else 'MISMATCH'}: {len(report['episodes'])} episodes; report {out}")
    return 0 if report["all_match"] else 1


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Physical AI Lab: MuJoCo Franka Panda push scene with seeded resets, an interactive viewer, "
        "headless stepping and a smoke test (Phase 1), a fixed-feedback push baseline with its gate (Phase 2), and "
        "benchmark manifests, effective-contact checks, a sealed final test, evaluation through the learner "
        "interface and replay (Phase 3).",
    )
    parser.add_argument("--version", action="version", version=f"physical_ai_lab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("doctor", help="check Python, dependencies, assets, scene, stepping and rendering")
    p.set_defaults(handler=cmd_doctor)

    p = sub.add_parser("fetch-assets", help="download the pinned Franka Panda files and verify their SHA-256")
    p.add_argument("--force", action="store_true", help="download every file again")
    p.set_defaults(handler=cmd_fetch_assets)

    p = sub.add_parser(
        "view", help="open the scene in the MuJoCo viewer (real time; jog the arm joint by joint with the arrow keys)"
    )
    p.add_argument("--seed", type=_seed, default=0, help="scenario seed (default 0)")
    p.add_argument("--randomize-physics", action="store_true", help="sample hidden block mass/friction from the seed")
    p.add_argument("--duration", type=_seconds, help="close the window automatically after this many wall-clock seconds")
    p.add_argument(
        "--screenshot", type=Path, help="capture the viewer window as a PNG (written when the viewer closes)"
    )
    p.add_argument("--screenshot-delay", type=_seconds, default=3.0, help="seconds before the capture (default 3)")
    p.add_argument("--show-ui", action="store_true", help="show the MuJoCo side panels")
    p.set_defaults(handler=cmd_view)

    p = sub.add_parser("step", help="reset and step the scene headless; print the state digest and observation")
    p.add_argument("--seed", type=_seed, default=0, help="scenario seed (default 0)")
    p.add_argument("--seconds", type=_seconds, default=5.0, help="simulated seconds (default 5)")
    p.add_argument("--randomize-physics", action="store_true", help="sample hidden block mass/friction from the seed")
    p.add_argument("--show-truth", action="store_true", help="also print the hidden scenario truth (diagnostics)")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p.set_defaults(handler=cmd_step)

    p = sub.add_parser("render", help="render the scene cameras offscreen to PNG files")
    p.add_argument("--seed", type=_seed, default=0, help="scenario seed (default 0)")
    p.add_argument("--seconds", type=_seconds, default=0.5, help="simulated seconds before rendering (default 0.5)")
    p.add_argument("--randomize-physics", action="store_true", help="sample hidden block mass/friction from the seed")
    p.add_argument("--out", type=Path, help="output folder (default runs/render/<UTC time>)")
    p.set_defaults(handler=cmd_render)

    p = sub.add_parser("smoke", help="run the Phase 1 headless smoke test and write a log")
    p.add_argument("--out", type=Path, help="output folder (default runs/smoke/<UTC time>)")
    p.add_argument(
        "--no-render", action="store_true", help="skip offscreen rendering and the camera-view check (no screenshot)"
    )
    p.set_defaults(handler=cmd_smoke)

    p = sub.add_parser(
        "gate", help="Phase 2: run the fixed-feedback baseline on the 20 fixed gate seeds headless and write logs"
    )
    p.add_argument(
        "--seeds",
        choices=("gate", "development"),
        default="gate",
        help="seed set from configs/push.toml (default gate; 'development' is the tuning set)",
    )
    p.add_argument("--out", type=Path, help="output folder (default runs/phase2/<seed set>/<UTC time>)")
    p.set_defaults(handler=cmd_gate)

    p = sub.add_parser(
        "demo", help="Phase 2: watch one baseline push episode in the MuJoCo viewer (inspection only, not the gate)"
    )
    p.add_argument("--mode", choices=("baseline",), default="baseline", help="controller to run (default baseline)")
    p.add_argument("--seed", type=_seed, default=20000, help="scenario seed (default 20000, the first gate seed)")
    p.add_argument("--speed", type=_speed, default=1.0, help="playback speed relative to real time (default 1)")
    p.add_argument("--duration", type=_seconds, help="close the window automatically after this many wall-clock seconds")
    p.add_argument("--linger", type=_seconds, default=5.0, help="seconds to keep the window open after the episode (default 5)")
    p.add_argument("--show-ui", action="store_true", help="show the MuJoCo side panels")
    p.add_argument(
        "--frames",
        type=Path,
        help="instead of opening the viewer, run the episode headless and save offscreen frames (start, "
        "mid-stroke of every push, end) to this folder",
    )
    p.set_defaults(handler=cmd_demo)

    p = sub.add_parser(
        "contact-check",
        help="Phase 3: measure that hidden mass/friction take effect (readback, normal force, kick friction, "
        "hold, lift/tilt under baseline pushes) across the in-range and out-of-range boxes",
    )
    p.add_argument("--out", type=Path, help="output folder (default runs/phase3/contact_checks/<UTC time>)")
    p.set_defaults(handler=cmd_contact_check)

    p = sub.add_parser(
        "manifests",
        help="Phase 3: generate the training, validation, sealed final-test and sealed retention manifests "
        "(refuses to overwrite), or --verify existing ones",
    )
    p.add_argument("--out", type=Path, help="manifest folder (default configs/manifests)")
    p.add_argument("--verify", action="store_true", help="check hashes and disjointness of existing manifests only")
    p.set_defaults(handler=cmd_manifests)

    p = sub.add_parser(
        "evaluate",
        help="Phase 3: run a policy through the learner interface (separate process) on every scenario of a "
        "manifest split and write episode records and a report; sealed splits refuse without a frozen candidate",
    )
    p.add_argument("--split", choices=("train", "validation", "test", "retention"), required=True)
    p.add_argument("--mode", choices=("baseline",), default="baseline", help="policy to run (Phase 3: baseline only)")
    p.add_argument("--limit", type=_positive_int, help="run only the first N scenarios (marked PARTIAL; not a report)")
    p.add_argument("--out", type=Path, help="output folder (default runs/phase3/evaluate/<split>/<UTC time>)")
    p.set_defaults(handler=cmd_evaluate)

    p = sub.add_parser(
        "replay",
        help="Phase 3: replay logged episodes of an evaluation run (logged commands, and a fresh rerun through "
        "the agent) and compare the simulator state after every control update",
    )
    p.add_argument("--run", type=Path, required=True, help="evaluation run folder (contains summary.json)")
    p.add_argument("--count", type=_positive_int, help="episodes to replay (default: benchmark.toml [replay])")
    p.add_argument("--out", type=Path, help="report file (default <run>/replay.json)")
    p.set_defaults(handler=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from .scenario import route_mujoco_warnings_to_stderr
    except ImportError:  # broken install: let the command report it
        pass
    else:
        route_mujoco_warnings_to_stderr()
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
