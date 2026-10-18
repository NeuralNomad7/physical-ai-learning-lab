"""Watch a baseline push episode (inspection only; not part of the gate).

``run_demo`` opens the MuJoCo viewer and runs the fixed-feedback baseline through the same
``EvaluatedPlant`` and judge as the gate, paced to real time. Afterwards the same seed is
run headless and compared, so a viewer that changed the physics (e.g. a Ctrl+right-drag
perturbation) is reported instead of hidden.

``render_episode_frames`` runs the same episode headless and renders the scene cameras
offscreen at fixed moments (start, middle of each push stroke, end) for evidence images.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco

from .baseline import FeedbackBaseline
from .config import SceneConfig
from .evaluation import EpisodeInterrupted, EpisodeRecord, EvaluatedPlant, episode_line, run_episode
from .observation import Observation
from .push_config import PushConfig, load_push_config
from .scenario import ScenarioRunner

MODES = ("baseline",)


def make_policy(mode: str, scene: SceneConfig, push: PushConfig) -> FeedbackBaseline:
    if mode != "baseline":
        raise ValueError(f"unknown mode {mode!r}; available: {', '.join(MODES)}")
    return FeedbackBaseline(scene, push)


@dataclass
class DemoSummary:
    record: EpisodeRecord
    state_digest: str
    headless_record: EpisodeRecord | None
    headless_digest: str | None
    wall_seconds: float
    closed_by: str
    option_reverts: int


def _status(plant: EvaluatedPlant, obs: Observation, push: PushConfig, outcome: str | None) -> str:
    ctx = plant._judge.context  # evaluator-side display only
    lines = [
        f"t {obs.time:5.2f} s / {push.episode.max_seconds:g} s",
        f"push {ctx['push_index'] or '-'} (started {plant.pushes_declared}/{push.episode.max_pushes})",
        f"phase {ctx['phase']}",
        f"block-target {obs.block_target_distance * 1e3:6.1f} mm",
        f"block speed {obs.block_planar_speed * 1e3:5.1f} mm/s",
        f"finger contact {'yes' if obs.robot_block_contact else 'no'}",
    ]
    if outcome:
        lines.append(outcome)
    return "\n".join(lines)


def run_demo(
    seed: int,
    *,
    mode: str = "baseline",
    speed: float = 1.0,
    duration: float | None = None,
    linger: float = 5.0,
    show_ui: bool = False,
    verify: bool = True,
    log: Callable[[str], None] = print,
) -> DemoSummary:
    """Run one episode in the viewer; ``speed`` scales real time, ``duration`` closes the window."""
    from .viewer import FREE_CAMERA, _join_viewer_thread, _launch, _mark_target

    push = load_push_config()
    runner = ScenarioRunner()
    policy = make_policy(mode, runner.config, push)
    model, data = runner.sim_model, runner.sim_data
    first = runner.reset(seed)
    log(
        f"Demo: {mode} on seed {seed}, nominal physics; block at ({first.block_pos[0]:.3f}, {first.block_pos[1]:.3f}) m, "
        f"target at ({first.target_pos[0]:.3f}, {first.target_pos[1]:.3f}) m, distance "
        f"{first.block_target_distance * 1e3:.1f} mm. Close the window to stop."
    )
    handle = _launch(model, data, lambda key: None, show_ui)
    started = time.perf_counter()
    closed_by = "episode finished"
    reverts = 0
    pace = {"sim0": None, "wall0": None}

    def expired() -> bool:
        return duration is not None and time.perf_counter() - started >= duration

    def on_update(plant: EvaluatedPlant, obs: Observation) -> None:
        nonlocal reverts
        if not handle.is_running():
            raise EpisodeInterrupted("viewer window closed")
        if expired():
            raise EpisodeInterrupted(f"duration of {duration:g} s elapsed")
        with handle.lock():
            if runner.restore_options():
                reverts += 1
        # set_texts takes the viewer lock itself, so it must be called outside handle.lock().
        handle.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           _status(plant, obs, push, None), "")])
        handle.sync()
        if pace["sim0"] is None:
            pace["sim0"], pace["wall0"] = obs.time, time.perf_counter()
        ahead = (obs.time - pace["sim0"]) / speed - (time.perf_counter() - pace["wall0"])
        if ahead > 0:
            time.sleep(ahead)

    record = None
    try:
        with handle:
            with handle.lock():
                handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                handle.cam.lookat[:] = FREE_CAMERA["lookat"]
                handle.cam.distance = FREE_CAMERA["distance"]
                handle.cam.azimuth = FREE_CAMERA["azimuth"]
                handle.cam.elevation = FREE_CAMERA["elevation"]
                _mark_target(handle.user_scn, first.target_pos)
            handle.sync()
            record = run_episode(runner, policy, push, seed, on_update=on_update, lock=handle.lock)
            if record.reason == "interrupted":
                closed_by = record.detail
            log(episode_line(record))
            verdict = "SUCCESS" if record.success else f"FAILURE: {record.reason}"
            final = runner.observe()
            handle.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               f"{verdict}\npushes {record.pushes_used}, t {record.end_time:.2f} s\n"
                               f"block-target {final.block_target_distance * 1e3:.1f} mm", "")])
            handle.sync()
            linger_until = time.perf_counter() + (0.0 if record.reason == "interrupted" else linger)
            while handle.is_running() and time.perf_counter() < linger_until and not expired():
                time.sleep(0.05)
            if handle.is_running() and record.reason != "interrupted":
                closed_by = "episode finished" + ("" if not expired() else f"; duration of {duration:g} s elapsed")
    finally:
        _join_viewer_thread()
    assert record is not None
    digest = runner.state_digest()
    headless, headless_digest = None, None
    if verify and record.reason != "interrupted":
        check = ScenarioRunner()
        headless = run_episode(check, make_policy(mode, check.config, push), push, seed)
        headless_digest = check.state_digest()
    return DemoSummary(record, digest, headless, headless_digest, time.perf_counter() - started, closed_by, reverts)


def render_episode_frames(
    seed: int,
    out_dir: Path,
    *,
    mode: str = "baseline",
    size: tuple[int, int] = (800, 600),
    cameras: tuple[str, ...] = ("oblique", "overhead"),
) -> tuple[EpisodeRecord, dict[str, Path]]:
    """Run one episode headless and render frames at the start, mid-stroke of each push, and end."""
    from .render import save_renders

    push = load_push_config()
    runner = ScenarioRunner()
    policy = make_policy(mode, runner.config, push)
    renderer = mujoco.Renderer(runner.sim_model, height=size[1], width=size[0])
    written: dict[str, Path] = {}
    stroke: dict[str, Any] = {"index": None, "start": 0.0, "duration": 1.0, "done": set()}

    def grab(label: str, obs: Observation, push_index: Any, phase: str) -> None:
        images = {}
        for camera in cameras:
            renderer.update_scene(runner.sim_data, camera=camera)
            images[camera] = renderer.render().copy()
        caption = (f"{mode} seed {seed} | {label} | t {obs.time:.2f} s | push {push_index or '-'} "
                   f"phase {phase} | block-target {obs.block_target_distance * 1e3:.0f} mm | "
                   f"finger contact {'yes' if obs.robot_block_contact else 'no'}")
        paths = save_renders(images, out_dir, caption, prefix=f"seed{seed}_{label}")
        written[label] = paths["composite"]

    def on_update(plant: EvaluatedPlant, obs: Observation) -> None:
        ctx = plant._judge.context  # evaluator-side, for choosing frames only
        if "start" not in written:
            grab("start", obs, None, ctx["phase"])
        index = ctx["push_index"]
        if index is None or index in stroke["done"]:
            return
        if ctx["phase"] == "stroke" and stroke["index"] != index:
            plan = next((n["plan"] for n in reversed(plant.notes) if n["event"] == "push_start"), None)
            stroke.update(index=index, start=obs.time,
                          duration=float(plan["action"]["duration"]) if plan else 1.0)
        if stroke["index"] != index:
            return
        # First update past mid-stroke with finger/block contact (the contact flag chatters during
        # a stroke); if none, the first update after the stroke ends.
        past_half = obs.time - stroke["start"] >= stroke["duration"] / 2
        if (ctx["phase"] == "stroke" and past_half and obs.robot_block_contact) or ctx["phase"] != "stroke":
            stroke["done"].add(index)
            grab(f"push{index}_mid_stroke", obs, index, ctx["phase"])

    try:
        record = run_episode(runner, policy, push, seed, on_update=on_update)
        grab("end", runner.observe(), None, f"end ({'success' if record.success else record.reason})")
    finally:
        renderer.close()
    return record, written


def headless_matches(summary: DemoSummary) -> bool | None:
    if summary.headless_record is None:
        return None
    a, b = summary.record, summary.headless_record
    return (a.success, a.reason, a.pushes_declared, round(a.end_time, 6)) == (
        b.success, b.reason, b.pushes_declared, round(b.end_time, 6)
    ) and summary.state_digest == summary.headless_digest


__all__ = ["MODES", "DemoSummary", "headless_matches", "make_policy", "render_episode_frames", "run_demo"]
