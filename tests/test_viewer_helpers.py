"""Viewer helpers that need no window: status panels, their placement, the overlay check and the markers."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from physical_ai_lab import viewer
from physical_ai_lab.scenario import ScenarioRunner
from physical_ai_lab.viewer import (
    DEG,
    JOINT_DISC_HALF_THICKNESS,
    JOINT_DISC_RADIUS,
    JOINT_DISC_RGBA,
    KEY_DOWN,
    KEY_ENTER,
    KEY_EQUAL,
    KEY_RIGHT,
    PANEL_BACKGROUND,
    PANEL_HIGHLIGHT_COLOR,
    STATUS_RESERVE,
    TARGET_ARROW_RGBA,
    TARGET_ARROW_Z,
    ViewerSession,
    overlay_age,
    panel_layout,
    render_panel,
    selected_rows,
    status_rows,
)

STATUS_LABELS = ["seed", "hidden physics", "sim time", "block to target", "success radius", "state"]
JOG_LABELS = ["arm targets", "jog step", "gripper", "J1", "J2", "J3", "J4", "J5", "J6", "J7"]


def test_status_rows(runner: ScenarioRunner) -> None:
    obs = runner.reset(0)
    rows = status_rows(0, "nominal", obs, paused=False)
    assert [label for label, _ in rows] == STATUS_LABELS
    values = dict(rows)
    assert values["seed"] == "0"
    assert values["hidden physics"] == "nominal"
    assert values["sim time"] == "0.00 s"
    assert values["block to target"] == f"{obs.block_target_distance * 1e3:.1f} mm"
    assert values["success radius"] == "20 mm"
    assert values["state"] == "running"
    assert dict(status_rows(0, "randomized (values hidden)", obs, paused=True))["state"] == "PAUSED (Space)"


def test_status_panel_is_deterministic_with_a_fixed_size(runner: ScenarioRunner) -> None:
    obs = runner.reset(0)
    running = render_panel(status_rows(0, "nominal", obs, paused=False), 24, STATUS_RESERVE)
    paused = render_panel(status_rows(0, "nominal", obs, paused=True), 24, STATUS_RESERVE)
    assert running.dtype == np.uint8 and running.ndim == 3 and running.shape[2] == 3
    np.testing.assert_array_equal(running, render_panel(status_rows(0, "nominal", obs, paused=False), 24, STATUS_RESERVE))
    assert running.shape == paused.shape  # the panel does not jump when the state text changes
    assert not np.array_equal(running, paused)
    assert tuple(running[0, 0]) == PANEL_BACKGROUND


def test_jog_rows_show_targets_limits_and_the_selection(runner: ScenarioRunner) -> None:
    session = ViewerSession(runner, 0)
    jog, obs = session.jog, session.obs
    rows = status_rows(0, "nominal", obs, paused=False, jog=jog)
    assert [label for label, _ in rows] == STATUS_LABELS + ["arm targets", "jog step", "gripper", "> J1", *JOG_LABELS[4:]]
    values = dict(rows)
    assert (values["arm targets"], values["jog step"], values["gripper"]) == ("home pose", f"5{DEG}", "closed (0 mm)")
    joints = [value for _, value in rows[-7:]]
    assert joints == [f"{angle}{DEG}" for angle in ("+0.0", "+0.0", "+0.0", "-90.0", "+0.0", "+90.0", "-45.0")]
    assert selected_rows(rows) == [len(STATUS_LABELS) + 3]
    at_reset = render_panel(rows, 24, STATUS_RESERVE, selected_rows(rows))

    for key in (KEY_DOWN,) * 3 + (KEY_EQUAL,) * 2 + (KEY_RIGHT,) * 5 + (KEY_ENTER,):  # J4 to its upper limit
        session.press(key)
    rows = status_rows(0, "nominal", session.obs, paused=False, jog=jog)
    values = dict(rows)
    assert (values["arm targets"], values["jog step"]) == ("operator-set", f"20{DEG}")
    assert values["gripper"] == "open (0 mm)"  # the target; the fingers have not moved yet
    assert values["> J4"] == f"-4.0{DEG} (limit, now -90.0{DEG})"  # target and measured angle
    assert selected_rows(rows) == [len(STATUS_LABELS) + 6]
    assert render_panel(rows, 24, STATUS_RESERVE, selected_rows(rows)).shape == at_reset.shape  # fixed size

    session.step(100)  # 2 s: the servos reach the targets
    values = dict(status_rows(0, "nominal", session.obs, paused=False, jog=jog))
    assert values["> J4"] == f"-4.0{DEG} (limit)"
    assert values["gripper"] == "open (80 mm)"


def test_selected_joint_row_is_highlighted() -> None:
    rows = [("seed", "0"), ("> J2", f"+5.0{DEG}"), ("J3", f"+0.0{DEG}")]
    plain, marked = render_panel(rows, 24), render_panel(rows, 24, highlight=[1])
    pad, line = 24 // 2, round(24 * 1.35)

    def shows_highlight(panel: np.ndarray, rows_slice: slice) -> bool:
        return bool(np.all(panel[rows_slice] == np.array(PANEL_HIGHLIGHT_COLOR, dtype=np.uint8), axis=-1).any())

    assert plain.shape == marked.shape
    assert shows_highlight(marked, slice(pad + line, pad + 2 * line))
    assert not shows_highlight(marked, slice(0, pad + line)) and not shows_highlight(marked, slice(pad + 2 * line, None))
    assert not shows_highlight(plain, slice(None))


@pytest.mark.parametrize("viewport", [(0, 0, 2560, 1440), (300, 0, 1800, 1300), (0, 0, 1280, 720)])
def test_panels_sit_in_the_right_hand_corners(runner: ScenarioRunner, viewport: tuple[int, int, int, int]) -> None:
    session = ViewerSession(runner, 0)  # the viewer shows the jog rows too
    rows = status_rows(0, "nominal", session.obs, paused=False, jog=session.jog)
    (status_rect, status), (hint_rect, hint) = panel_layout(viewport, rows)
    left, bottom, width, height = viewport
    assert (status_rect.width, status_rect.height) == (status.shape[1], status.shape[0])
    assert (hint_rect.width, hint_rect.height) == (hint.shape[1], hint.shape[0])
    assert (status_rect.left + status_rect.width, status_rect.bottom + status_rect.height) == (left + width, bottom + height)
    assert (hint_rect.left + hint_rect.width, hint_rect.bottom) == (left + width, bottom)
    assert hint_rect.bottom + hint_rect.height <= status_rect.bottom


@pytest.mark.parametrize("viewport", [None, (0, 0, 0, 0), (0, 0, 100, 1440), (0, 0, 2560, 50)])
def test_no_panels_without_room(runner: ScenarioRunner, viewport: tuple[int, int, int, int] | None) -> None:
    rows = status_rows(0, "nominal", runner.reset(0), paused=False)
    assert panel_layout(viewport, rows) == []


def _window_capture(panels: list, size: tuple[int, int] = (1280, 720), channels: int = 3) -> np.ndarray:
    """A top-down window capture showing the given panels (their rectangles use a bottom-left origin)."""
    width, height = size
    capture = np.zeros((height, width, channels), dtype=np.uint8)
    for rect, image in panels:
        top = height - rect.bottom - rect.height
        capture[top : top + rect.height, rect.left : rect.left + rect.width, :3] = image
    return capture


def test_overlay_age_matches_captures_pixel_for_pixel(runner: ScenarioRunner) -> None:
    viewport = (0, 0, 1280, 720)
    older = panel_layout(viewport, status_rows(0, "nominal", runner.reset(0), paused=False))
    newer = panel_layout(viewport, status_rows(0, "nominal", runner.step(50), paused=False))
    sent = [older, newer]
    assert overlay_age(_window_capture(newer), sent) == 0
    assert overlay_age(_window_capture(older), sent) == 1
    assert overlay_age(_window_capture(newer, channels=4), sent) == 0  # BGRA/RGBA captures: alpha ignored
    corrupted = _window_capture(newer)
    rect = newer[0][0]
    corrupted[720 - rect.bottom - rect.height + 3, rect.left + 3, 0] ^= 0xFF
    assert overlay_age(corrupted, sent) is None
    assert overlay_age(_window_capture(newer), []) is None


def test_goal_arrow_points_down_at_the_target(runner: ScenarioRunner) -> None:
    scene = mujoco.MjvScene(runner.sim_model, maxgeom=10)
    target = np.array([0.55, -0.12, 0.0])
    viewer._mark_target(scene, target)
    assert scene.ngeom == 1
    geom = scene.geoms[0]
    tail, tip = TARGET_ARROW_Z
    assert geom.type == mujoco.mjtGeom.mjGEOM_ARROW
    np.testing.assert_allclose(geom.pos, [target[0], target[1], tail], atol=1e-6)
    np.testing.assert_allclose(np.asarray(geom.mat).reshape(3, 3)[:, 2], [0.0, 0.0, -1.0], atol=1e-6)
    assert geom.size[2] == pytest.approx(tail - tip, abs=1e-6)
    np.testing.assert_allclose(geom.rgba, TARGET_ARROW_RGBA, atol=1e-6)
    viewer._mark_target(None, target)  # no user scene available: nothing to draw


def test_joint_disc_lies_in_the_rotation_plane(runner: ScenarioRunner) -> None:
    scene = mujoco.MjvScene(runner.sim_model, maxgeom=10)
    anchor, axis = np.array([0.1, -0.2, 0.5]), np.array([1.0, 2.0, 2.0]) / 3.0
    viewer._mark_target(scene, np.array([0.55, -0.12, 0.0]))
    viewer._mark_joint(scene, anchor, axis)
    assert scene.ngeom == 2 and scene.geoms[0].type == mujoco.mjtGeom.mjGEOM_ARROW
    disc = scene.geoms[1]
    assert disc.type == mujoco.mjtGeom.mjGEOM_CYLINDER
    np.testing.assert_allclose(disc.pos, anchor, atol=1e-6)
    np.testing.assert_allclose(np.asarray(disc.mat).reshape(3, 3)[:, 2], axis, atol=1e-6)  # normal along the axis
    np.testing.assert_allclose(disc.size, [JOINT_DISC_RADIUS, JOINT_DISC_RADIUS, JOINT_DISC_HALF_THICKNESS], atol=1e-6)
    np.testing.assert_allclose(disc.rgba, JOINT_DISC_RGBA, atol=1e-6)
    viewer._mark_joint(None, anchor, axis)
