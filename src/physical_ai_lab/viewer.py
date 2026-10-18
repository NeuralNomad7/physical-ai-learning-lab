"""Interactive MuJoCo viewer for the push scene (simulator side, human inspection only).

The viewer steps the scenario in real time. The arm starts in its home pose and can be
jogged by hand, one joint at a time, from the keyboard (``JointJog``); the Control
sliders in MuJoCo's right-hand panel set the same servo targets. Both change only the
actuator inputs, clamped to their control ranges, so the arm moves through its position
servos and the contact dynamics, and the block moves only when the robot pushes it. The
viewer is not part of any experiment: nothing seen or done in it is logged as data, and
agent code never uses it.

Any other direct change the viewer makes to the simulation state (Backspace reset,
keyframe load, dragging a body while paused, moving the target, editing joint sliders,
toggling the equality constraint in the right-hand panel) is detected by comparing time,
qpos, qvel, the target pose and the equality states with the values left by the last
step; the scenario is then restored from the same seed, so the viewer cannot teleport
objects. Physics options edited in the left-hand Physics panel (timestep, solver,
gravity, disable flags, ...) are written into the model by the viewer; they are put
back to the scene's values before the next step (``ScenarioRunner.restore_options``),
so they never change the simulation. Ctrl+right-drag force perturbations act through
the physics and remain available for manual inspection. Operator commands are recorded
with the number of control updates since the last reset, so that ``replay`` can
reproduce the run headless.

The status and key-hint panels are rendered with Pillow and drawn as images
(``Handle.set_images``) instead of MuJoCo's bitmap-font text (``Handle.set_texts``):
with the Mesa D3D12 (GLon12) OpenGL driver on Windows on Arm, bitmap-font glyphs in
the live window are corrupted at random from frame to frame (offscreen rendering is
not affected). MuJoCo's own text (F1 help, F2 info, Tab UI panels) uses the same font
path and can still glitch on that driver. Two markers are added only to the viewer's
``user_scn`` (never to the model or to any rendering used as data): a green arrow above
the goal, which stays visible when the block or the arm hides the goal marker, and an
orange disc in the rotation plane of the joint that the arrow keys turn.
"""

from __future__ import annotations

import collections
import functools
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import SceneConfig
from .observation import Observation
from .scenario import ScenarioRunner, SimulationUnstable
from .wincapture import capture_own_window

KEY_SPACE = 32
# GLFW key codes that the passive viewer leaves unused. Its key callback fires on press only
# (no auto-repeat), so every press is one jog step.
KEY_ENTER = 257
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 262, 263, 264, 265
KEY_HOME = 268
KEY_MINUS, KEY_EQUAL = 45, 61  # the + key is Shift+=
KEY_KP_SUBTRACT, KEY_KP_ADD = 333, 334
JOG_KEYS = frozenset(
    {KEY_ENTER, KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_HOME, KEY_MINUS, KEY_EQUAL, KEY_KP_SUBTRACT, KEY_KP_ADD}
)
JOG_STEPS_DEG = (1.0, 2.0, 5.0, 10.0, 20.0)
JOG_DEFAULT_STEP_DEG = 5.0
# A joint whose actual angle differs from its target by more than this (e.g. blocked by a
# contact) shows both in the status panel.
JOG_SHOW_ACTUAL_DEG = 1.0
LAUNCH_TIMEOUT_SECONDS = 60.0
# At most this many control updates are stepped per loop iteration to catch up with the wall
# clock (0.2 s at 50 Hz); a longer stall drops the backlog instead of running fast afterwards.
CATCH_UP_LIMIT = 10
# The status panel shows the sim time, so it is re-rendered at 5 Hz, and immediately when the
# paused / restored state changes.
OVERLAY_INTERVAL_SECONDS = 0.2
# Free camera showing the whole arm, the work area and the goal.
FREE_CAMERA = {"lookat": (0.40, 0.0, 0.26), "distance": 1.75, "azimuth": 145.0, "elevation": -40.0}

PANEL_BACKGROUND = (24, 30, 40)
PANEL_LABEL_COLOR = (170, 180, 195)
PANEL_VALUE_COLOR = (255, 255, 255)
PANEL_HIGHLIGHT_COLOR = (255, 150, 40)  # the selected joint's row, in the colour of its disc
SELECTED_MARK = "> "
DEG = "\u00b0"
# Values the status panel can show, measured so that its width stays fixed while they change.
STATUS_RESERVE = (
    "PAUSED (Space)",
    "running",
    "0000.00 s",
    "000.0 mm",
    "operator-set",
    "00% open (00 mm)",
    f"-000.0{DEG} (limit, now -000.0{DEG})",
)
HINT_ROWS = (
    ("Up / Down", "select a joint (orange disc)"),
    ("Left / Right", "turn it by one step"),
    ("- / +", "smaller / larger step"),
    ("Enter", "open / close the gripper"),
    ("Home", "home pose, gripper closed"),
    ("Space", "pause / resume"),
    ("Backspace", "reset to the same seed"),
    ("[ and ]", "cycle the fixed cameras (Esc: free)"),
    ("Tab", "MuJoCo UI panels"),
)
# Viewer-only goal arrow, pointing down to 1 cm above the top of the 8 cm marker pole.
TARGET_ARROW_RGBA = (0.10, 0.85, 0.25, 1.0)
TARGET_ARROW_WIDTH = 0.008  # m
TARGET_ARROW_Z = (0.25, 0.09)  # m, tail and tip heights above the tabletop
# Viewer-only disc in the rotation plane of the selected joint, centred on the joint axis.
# It is wider than the links, so it shows even where the axis runs inside the arm.
JOINT_DISC_RGBA = (1.0, 0.55, 0.10, 0.45)
JOINT_DISC_RADIUS = 0.10  # m
JOINT_DISC_HALF_THICKNESS = 0.002  # m

CONTROLS = """\
Viewer controls (MuJoCo simulate window; click the 3D view first so that it has keyboard focus):
  Up / Down arrows     select an arm joint, J1 (base) to J7 (flange); an orange disc marks it
  Left / Right arrows  turn the selected joint by one step (Right = positive angle)
  - and +              smaller / larger step: 1, 2, 5, 10 or 20 degrees, default 5 (+ is the = key)
  Enter                open / close the gripper
  Home                 back to the home pose with the gripper closed (the block stays where it is)
  Space                pause / resume (keys still set targets while paused)
  Backspace            reset to the same seed (the scenario is restored)
  Left drag            rotate camera;  right drag: pan;  scroll: zoom
  [ and ]              cycle the fixed cameras (overhead, oblique);  Esc: free camera
  Tab / Shift+Tab      show / hide the left / right UI panels
  F1                   MuJoCo help overlay
  Double-click a body, then Ctrl+right drag: push it with a force (manual inspection only)
  Close the window (or press Ctrl+C in this terminal) to exit.
Each key press is one step; holding a key does not repeat. The keys and the Control sliders
(right UI panel) only set the servo targets: the arm moves through its motors and contacts,
and the block moves only when the robot pushes it. Joint sliders and other direct edits
restore the seed. Physics options edited in the left UI panel are put back to the scene's
values before the next step (the panel can show the edited number until it is redrawn).
The green arrow marks the goal (viewer only). MuJoCo's own text (F1 help, UI panels) can
show garbled letters with some OpenGL drivers (seen with Mesa D3D12 on Windows on Arm);
the status panels on the right are drawn as images and are not affected."""


@dataclass(frozen=True)
class ViewerSummary:
    seed: int
    physics_source: str
    closed_by: str
    wall_seconds: float  # from the first frame until the loop ended (window still open)
    paused_seconds: float  # wall time spent paused (Space)
    close_seconds: float  # time to close the window and stop the viewer thread
    sim_seconds: float  # simulated time stepped in total, across restores
    realtime_factor: float | None  # sim_seconds / unpaused wall time (None if barely ran)
    resyncs: int  # stalls longer than CATCH_UP_LIMIT updates whose backlog was dropped
    control_updates: int
    restores: int  # scenario restored after a viewer-side state change
    unstable_events: int
    operator_commands: int  # jog keys that changed a target, plus Control-slider edits
    option_reverts: int  # Physics-panel edits to the model options, put back before the next step
    final_time: float
    final_block_target_distance: float
    updates_since_reset: int  # control updates since the last (re)reset
    final_state_digest: str  # compare with replay() of updates_since_reset updates and commands_since_reset
    screenshot: dict[str, Any] | None
    final_state: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))  # full integration state
    # Applied ctrl vectors by the number of control updates since the last reset (the last one at each count).
    commands_since_reset: dict[int, np.ndarray] = field(repr=False, default_factory=dict)


def _launch(model: mujoco.MjModel, data: mujoco.MjData, key_callback: Callable[[int], None], show_ui: bool):
    """Open the passive viewer, failing with a clear error instead of hanging."""
    result: queue.Queue = queue.Queue(1)

    def target() -> None:
        try:
            handle = mujoco.viewer.launch_passive(
                model, data, key_callback=key_callback, show_left_ui=show_ui, show_right_ui=show_ui
            )
            result.put(("ok", handle))
        except BaseException as exc:  # reported to the caller
            result.put(("error", exc))

    threading.Thread(target=target, name="viewer-launch", daemon=True).start()
    try:
        kind, value = result.get(timeout=LAUNCH_TIMEOUT_SECONDS)
    except queue.Empty:
        raise RuntimeError(
            f"the MuJoCo viewer window did not open within {LAUNCH_TIMEOUT_SECONDS:.0f} s "
            "(see any GLFW/OpenGL error printed above)"
        ) from None
    if kind == "error":
        raise RuntimeError(f"could not open the MuJoCo viewer: {type(value).__name__}: {value}") from value
    return value


def _snapshot(data: mujoco.MjData) -> tuple:
    return (
        float(data.time),
        data.qpos.copy(),
        data.qvel.copy(),
        data.ctrl.copy(),
        data.mocap_pos.copy(),
        data.mocap_quat.copy(),
        data.eq_active.copy(),
    )


class JointJog:
    """Keyboard jog of the arm's servo targets, one joint at a time (viewer only).

    Up/Down select joint J1-J7, Left/Right move the selected joint's target by one step,
    -/+ change the step, Enter opens or closes the gripper and Home restores the reset
    targets. Targets stay inside the actuators' control ranges. Only targets change here;
    the position servos then move the arm through the physics.
    """

    def __init__(
        self, home_arm: np.ndarray, arm_range: np.ndarray, gripper_range: Sequence[float], gripper_closed: float
    ) -> None:
        self.home_arm = np.array(home_arm, dtype=float)
        self.arm_range = np.array(arm_range, dtype=float).reshape(len(self.home_arm), 2)
        low, high = float(gripper_range[0]), float(gripper_range[1])
        self.gripper_closed = float(gripper_closed)
        # The open end of the gripper's range is the one farther from the closed command.
        self.gripper_open = high if abs(high - self.gripper_closed) >= abs(low - self.gripper_closed) else low
        self.selected = 0
        self.step_index = JOG_STEPS_DEG.index(JOG_DEFAULT_STEP_DEG)
        self.arm = self.home_arm.copy()
        self.gripper = self.gripper_closed

    def reset(self) -> None:
        """Return to the reset targets (home pose, gripper closed); the selection and step are kept."""
        self.arm = self.home_arm.copy()
        self.gripper = self.gripper_closed

    def load(self, arm: np.ndarray, gripper: float) -> None:
        """Adopt targets set elsewhere (the Control sliders)."""
        self.arm = np.array(arm, dtype=float)
        self.gripper = float(gripper)

    @property
    def step_deg(self) -> float:
        return JOG_STEPS_DEG[self.step_index]

    @property
    def at_home(self) -> bool:
        return bool(np.allclose(self.arm, self.home_arm, rtol=0.0, atol=1e-9))

    @property
    def gripper_fraction(self) -> float:
        """Gripper target from 0 (closed) to 1 (fully open)."""
        span = self.gripper_open - self.gripper_closed
        return (self.gripper - self.gripper_closed) / span if span else 0.0

    def at_limit(self, joint: int) -> bool:
        low, high = self.arm_range[joint]
        return bool(min(self.arm[joint] - low, high - self.arm[joint]) <= 1e-9)

    def press(self, keycode: int) -> bool | None:
        """Apply one key press: None if it is not a jog key, else whether a target changed."""
        if keycode in (KEY_UP, KEY_DOWN):
            self.selected = (self.selected + (1 if keycode == KEY_DOWN else -1)) % len(self.arm)
            return False
        if keycode in (KEY_MINUS, KEY_KP_SUBTRACT, KEY_EQUAL, KEY_KP_ADD):
            change = -1 if keycode in (KEY_MINUS, KEY_KP_SUBTRACT) else 1
            self.step_index = min(max(self.step_index + change, 0), len(JOG_STEPS_DEG) - 1)
            return False
        arm, gripper = self.arm.copy(), self.gripper
        if keycode in (KEY_LEFT, KEY_RIGHT):
            joint = self.selected
            step = math.radians(self.step_deg) * (1.0 if keycode == KEY_RIGHT else -1.0)
            low, high = self.arm_range[joint]
            self.arm[joint] = min(max(self.arm[joint] + step, low), high)
        elif keycode == KEY_ENTER:
            self.gripper = self.gripper_closed if self.gripper_fraction > 0.5 else self.gripper_open
        elif keycode == KEY_HOME:
            self.reset()
        else:
            return None
        return not (np.array_equal(self.arm, arm) and self.gripper == gripper)


class ViewerSession:
    """The viewer's simulation bookkeeping, kept apart from the window so that it can be tested headless.

    Operator commands (jog keys that change a target, Control-slider edits) only set actuator
    inputs, through ``ScenarioRunner.set_ctrl``. Each is recorded with the number of control
    updates since the last reset, so that ``replay`` can reproduce the run. Any other direct
    change to the state restores the scenario from its seed.
    """

    def __init__(self, runner: ScenarioRunner, seed: int, *, randomize_physics: bool = False) -> None:
        self.runner = runner
        self.seed = seed
        self.randomize_physics = randomize_physics
        model, robot = runner.sim_model, runner.config.robot
        self._arm_act = runner.arm_actuator_ids
        self._gripper_act = runner.gripper_actuator_id
        self._arm_joints = np.array([model.joint(name).id for name in robot.arm_joints], dtype=int)
        bounds = runner.ctrl_range
        self.jog = JointJog(runner.home_arm_qpos, bounds[self._arm_act], bounds[self._gripper_act], robot.gripper_ctrl_closed)
        self.updates = self.restores = self.unstable = self.since_reset = self.operator_commands = 0
        self.option_reverts = 0
        self.revision = 0  # changes whenever the status panel should be redrawn at once
        self.commands: dict[int, np.ndarray] = {}
        self.obs = runner.reset(seed, randomize_physics=randomize_physics)
        self._snapshot = _snapshot(runner.sim_data)

    def _restore(self) -> None:
        self.obs = self.runner.reset(self.seed, randomize_physics=self.randomize_physics)
        self.jog.reset()
        self.commands.clear()
        self.since_reset = 0
        self.revision += 1
        self._snapshot = _snapshot(self.runner.sim_data)

    def _command(self, ctrl: np.ndarray) -> None:
        applied = self.runner.set_ctrl(ctrl)
        self.jog.load(applied[self._arm_act], applied[self._gripper_act])
        self.commands[self.since_reset] = applied
        self.operator_commands += 1
        self.revision += 1
        self.obs = self.runner.observe()  # the observation includes the commanded targets
        self._snapshot = _snapshot(self.runner.sim_data)

    def check_external_changes(self) -> str | None:
        """Undo what the viewer changed in the model options or the state; return a message if it did.

        Physics options edited in the viewer's Physics panel are put back to the scene's values;
        the state is untouched, so the seed is not restored for them. A change to ``ctrl`` alone
        is a Control-slider edit: it is clamped and accepted as an operator command. Any other
        change (Backspace, a keyframe, dragging a body while paused, moving the goal, a joint
        slider, toggling the equality constraint) restores the scenario from its seed.
        """
        messages = []
        reverted = self.runner.restore_options()
        if reverted:
            self.option_reverts += 1
            messages.append(f"viewer edited physics option(s) {', '.join(reverted)}; put back the scene's values")
        data = self.runner.sim_data
        time_, qpos, qvel, ctrl, mocap_pos, mocap_quat, eq_active = self._snapshot
        state_unchanged = (
            data.time == time_
            and np.array_equal(data.qpos, qpos)
            and np.array_equal(data.qvel, qvel)
            and np.array_equal(data.mocap_pos, mocap_pos)
            and np.array_equal(data.mocap_quat, mocap_quat)
            and np.array_equal(data.eq_active, eq_active)
        )
        if not state_unchanged or not np.isfinite(data.ctrl).all():
            self._restore()
            self.restores += 1
            messages.append(f"viewer changed the simulation state; restored seed {self.seed}")
        elif not np.array_equal(data.ctrl, ctrl):
            self._command(data.ctrl.copy())
        return "; ".join(messages) or None

    def press(self, keycode: int) -> bool:
        """Apply a jog key; return False if the key is not a jog key."""
        changed = self.jog.press(keycode)
        if changed is None:
            return False
        if changed:
            ctrl = self.runner.sim_data.ctrl.copy()
            ctrl[self._arm_act] = self.jog.arm
            ctrl[self._gripper_act] = self.jog.gripper
            self._command(ctrl)
        else:
            self.revision += 1  # the selected joint or the step changed
        return True

    def step(self, updates: int) -> str | None:
        """Step up to ``updates`` control updates; return a message if instability restored the seed."""
        message = None
        for _ in range(updates):
            try:
                self.runner.step(1)
            except SimulationUnstable as exc:
                self.unstable += 1
                self._restore()
                message = f"simulation unstable ({exc}); restored seed {self.seed}"
                break
            self.updates += 1
            self.since_reset += 1
        self.obs = self.runner.observe()
        self._snapshot = _snapshot(self.runner.sim_data)
        return message

    def selected_joint_axis(self) -> tuple[np.ndarray, np.ndarray]:
        """World position and unit axis of the selected joint."""
        data, joint = self.runner.sim_data, self._arm_joints[self.jog.selected]
        return data.xanchor[joint].copy(), data.xaxis[joint].copy()


def replay(
    runner: ScenarioRunner,
    seed: int,
    *,
    randomize_physics: bool = False,
    updates: int = 0,
    commands: Mapping[int, np.ndarray] | None = None,
) -> Observation:
    """Reproduce a viewer run headless from its seed, update count and recorded operator commands.

    Each command is applied after the number of control updates it was recorded at, as in
    ``ViewerSession``; a command recorded after the last update is applied too.
    """
    commands = commands or {}
    if any(not 0 <= count <= updates for count in commands):
        raise ValueError(f"commands must be recorded at 0 to {updates} control updates")
    runner.reset(seed, randomize_physics=randomize_physics)
    for count in range(updates + 1):
        if count in commands:
            runner.set_ctrl(commands[count])
        if count < updates:
            runner.step(1)
    return runner.observe()


@functools.lru_cache(maxsize=8)
def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=px)


def render_panel(
    rows: Sequence[tuple[str, str]], px: int, reserve: Iterable[str] = (), highlight: Iterable[int] = ()
) -> np.ndarray:
    """Render (label, value) rows as a two-column RGB panel (uint8, height x width x 3).

    ``reserve`` lists other values the panel may show, so that its width does not change
    when the values do. Rows whose indices are in ``highlight`` are drawn in the highlight colour.
    """
    font = _font(px)
    pad, gap, line = px // 2, px, round(px * 1.35)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_width = max(measure.textlength(label, font=font) for label, _ in rows)
    value_width = max(measure.textlength(text, font=font) for text in [*(value for _, value in rows), *reserve])
    size = (math.ceil(2 * pad + label_width + gap + value_width), 2 * pad + line * len(rows))
    panel = Image.new("RGB", size, PANEL_BACKGROUND)
    draw = ImageDraw.Draw(panel)
    marked = set(highlight)
    for i, (label, value) in enumerate(rows):
        y = pad + i * line
        label_color, value_color = (
            (PANEL_HIGHLIGHT_COLOR, PANEL_HIGHLIGHT_COLOR) if i in marked else (PANEL_LABEL_COLOR, PANEL_VALUE_COLOR)
        )
        draw.text((pad, y), label, font=font, fill=label_color)
        draw.text((pad + label_width + gap, y), value, font=font, fill=value_color)
    return np.asarray(panel)


@functools.lru_cache(maxsize=4)
def _hint_panel(px: int) -> np.ndarray:
    return render_panel(HINT_ROWS, px)


def _degrees(angle: float) -> str:
    return f"{round(math.degrees(angle), 1) + 0.0:+.1f}{DEG}"  # + 0.0 turns -0.0 into 0.0


def jog_rows(jog: JointJog, obs: Observation) -> list[tuple[str, str]]:
    """Status rows for the operator's targets: the arm, the jog step, the gripper and each joint.

    Joint rows show the servo target; ``(now ...)`` adds the measured angle when it differs by
    more than ``JOG_SHOW_ACTUAL_DEG`` (still moving, or blocked by a contact) and ``(limit)``
    marks a target at the end of its control range. The selected joint's label starts with
    ``SELECTED_MARK``.
    """
    fraction = jog.gripper_fraction
    state = "closed" if fraction <= 0.005 else "open" if fraction >= 0.995 else f"{fraction:.0%} open"
    rows = [
        ("arm targets", "home pose" if jog.at_home else "operator-set"),
        ("jog step", f"{jog.step_deg:g}{DEG}"),
        ("gripper", f"{state} ({obs.gripper_opening * 1e3:.0f} mm)"),
    ]
    for joint, (target, actual) in enumerate(zip(jog.arm, obs.arm_qpos)):
        notes = []
        if jog.at_limit(joint):
            notes.append("limit")
        if abs(math.degrees(actual - target)) > JOG_SHOW_ACTUAL_DEG:
            notes.append(f"now {_degrees(actual)}")
        label = f"{SELECTED_MARK if joint == jog.selected else ''}J{joint + 1}"
        rows.append((label, _degrees(target) + (f" ({', '.join(notes)})" if notes else "")))
    return rows


def status_rows(
    seed: int, physics_label: str, obs: Observation, paused: bool, jog: JointJog | None = None
) -> list[tuple[str, str]]:
    rows = [
        ("seed", str(seed)),
        ("hidden physics", physics_label),
        ("sim time", f"{obs.time:.2f} s"),
        ("block to target", f"{obs.block_target_distance * 1e3:.1f} mm"),
        ("success radius", f"{obs.target_tolerance * 1e3:.0f} mm"),
        ("state", "PAUSED (Space)" if paused else "running"),
    ]
    return rows if jog is None else rows + jog_rows(jog, obs)


def selected_rows(rows: Sequence[tuple[str, str]]) -> list[int]:
    """Indices of the rows whose label marks the selected joint."""
    return [i for i, (label, _) in enumerate(rows) if label.startswith(SELECTED_MARK)]


def panel_layout(
    viewport: tuple[int, int, int, int] | None, rows: Sequence[tuple[str, str]]
) -> list[tuple[mujoco.MjrRect, np.ndarray]]:
    """Place the status panel in the top-right corner of the 3D view and the key hints in the bottom-right one.

    ``viewport`` is (left, bottom, width, height) in framebuffer pixels with a bottom-left
    origin, as for ``Handle.viewport``. The right-hand corners keep the panels clear of
    MuJoCo's own F1 help (top left) and F2 info (bottom left) text. The selected joint's row
    is highlighted. Returns no panels if the view is too small to hold them.
    """
    if viewport is None or viewport[2] <= 0 or viewport[3] <= 0:
        return []
    left, bottom, width, height = viewport
    px = min(40, max(12, round(height / 60)))
    status = render_panel(rows, px, STATUS_RESERVE, selected_rows(rows))
    hint = _hint_panel(max(12, round(px * 0.8)))
    if width < max(status.shape[1], hint.shape[1]) or height < status.shape[0] + hint.shape[0]:
        return []
    right, top = left + width, bottom + height
    return [
        (mujoco.MjrRect(right - status.shape[1], top - status.shape[0], status.shape[1], status.shape[0]), status),
        (mujoco.MjrRect(right - hint.shape[1], bottom, hint.shape[1], hint.shape[0]), hint),
    ]


def overlay_age(capture: np.ndarray, sent: Sequence[list[tuple[mujoco.MjrRect, np.ndarray]]]) -> int | None:
    """Return how many panel updates before the latest one the captured panels were sent.

    0 means the latest update. None means that no recent update matches the window capture
    pixel for pixel (the capture is top-down; panel rectangles have a bottom-left origin).
    """
    height = capture.shape[0]
    for age, panels in enumerate(reversed(sent)):
        if panels and all(
            np.array_equal(
                capture[height - rect.bottom - rect.height : height - rect.bottom, rect.left : rect.left + rect.width, :3],
                image,
            )
            for rect, image in panels
        ):
            return age
    return None


def _mark_target(scene: mujoco.MjvScene | None, target_pos: np.ndarray) -> None:
    """Point a viewer-only arrow down at the goal marker, which the block or the arm can hide."""
    if scene is None:
        return
    x, y = float(target_pos[0]), float(target_pos[1])
    tail, tip = TARGET_ARROW_Z
    geom = scene.geoms[0]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.asarray(TARGET_ARROW_RGBA, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_ARROW, TARGET_ARROW_WIDTH, np.array([x, y, tail]), np.array([x, y, tip])
    )
    scene.ngeom = 1


def _mark_joint(scene: mujoco.MjvScene | None, anchor: np.ndarray, axis: np.ndarray) -> None:
    """Add a viewer-only disc in the rotation plane of the selected joint (call after ``_mark_target``)."""
    if scene is None:
        return
    quat, mat = np.zeros(4), np.zeros(9)
    mujoco.mju_quatZ2Vec(quat, np.asarray(axis, dtype=float))  # the disc's normal along the joint axis
    mujoco.mju_quat2Mat(mat, quat)
    mujoco.mjv_initGeom(
        scene.geoms[1],
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([JOINT_DISC_RADIUS, JOINT_DISC_HALF_THICKNESS, 0.0]),
        np.asarray(anchor, dtype=float),
        mat,
        np.asarray(JOINT_DISC_RGBA, dtype=np.float32),
    )
    scene.ngeom = 2


def _save_window_capture(path: Path, image: np.ndarray | None, info: dict[str, Any]) -> dict[str, Any]:
    """Check and save a window capture taken by ``wincapture.capture_own_window``."""
    from .render import image_stats, is_nonblank

    if image is None:
        return {"saved": False, **info}
    info = {**info, "stats": image_stats(image)}
    if not is_nonblank(image, info["stats"]):
        return {"saved": False, **info, "error": "window capture is blank (OpenGL content was not captured)"}
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return {"saved": True, "path": str(path), **info}


def _join_viewer_thread(timeout: float = 5.0) -> bool:
    """Wait for MuJoCo's viewer thread to finish; return False if it is still running."""
    stopped = True
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and "_launch_internal" in thread.name:
            thread.join(timeout=timeout)
            stopped = stopped and not thread.is_alive()
    return stopped


def run_viewer(
    seed: int = 0,
    *,
    randomize_physics: bool = False,
    duration: float | None = None,
    screenshot: Path | None = None,
    screenshot_delay: float = 3.0,
    show_ui: bool = False,
    config: SceneConfig | None = None,
    log: Callable[[str], None] = print,
) -> ViewerSummary:
    """Open the scene in the MuJoCo viewer and step it in real time until closed.

    Control updates are paced by the wall clock: an iteration steps every update that is
    due (up to ``CATCH_UP_LIMIT``), so slow rendering lowers the frame rate, not the
    simulation rate. Jog keys are queued by the viewer thread and applied by this loop,
    under the viewer lock, before the updates that are due. ``duration`` (wall-clock
    seconds) closes the window automatically; ``screenshot`` captures the viewer window
    after ``screenshot_delay`` seconds and saves it once the window has closed.
    """
    runner = ScenarioRunner(config)
    cfg = runner.config
    session = ViewerSession(runner, seed, randomize_physics=randomize_physics)
    obs = session.obs
    truth = runner.hidden_truth()
    physics_label = "randomized (values hidden)" if randomize_physics else "nominal"
    model, data = runner.sim_model, runner.sim_data
    dt = cfg.timing.control_dt
    paused = threading.Event()
    keys: queue.SimpleQueue[int] = queue.SimpleQueue()

    def on_key(keycode: int) -> None:  # runs on the viewer thread; must not take the viewer lock
        if keycode == KEY_SPACE:
            if paused.is_set():
                paused.clear()
            else:
                paused.set()
        elif keycode in JOG_KEYS:
            keys.put(keycode)

    log(
        f"Scene: seed {seed}, hidden physics {physics_label}; block at "
        f"({truth.block_xy[0]:.3f}, {truth.block_xy[1]:.3f}) m, target at "
        f"({truth.target_xy[0]:.3f}, {truth.target_xy[1]:.3f}) m, distance {obs.block_target_distance * 1e3:.1f} mm"
    )
    log(f"Stepping in real time: {cfg.timing.substeps} x {cfg.timing.physics_timestep} s per {dt:.2f} s update.")
    log(CONTROLS)

    handle = _launch(model, data, on_key, show_ui)
    started = previous = time.perf_counter()
    next_due = started  # wall-clock time at which the next control update is due
    ended: float | None = None
    closed_by = "window closed"
    resyncs = 0
    paused_seconds = 0.0
    overlay_at, overlay_key = started - OVERLAY_INTERVAL_SECONDS, None
    sent_panels: collections.deque[list[tuple[mujoco.MjrRect, np.ndarray]]] = collections.deque(maxlen=4)
    capture: tuple[np.ndarray | None, dict[str, Any]] | None = None
    try:
        with handle:
            with handle.lock():
                handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                handle.cam.lookat[:] = FREE_CAMERA["lookat"]
                handle.cam.distance = FREE_CAMERA["distance"]
                handle.cam.azimuth = FREE_CAMERA["azimuth"]
                handle.cam.elevation = FREE_CAMERA["elevation"]
                _mark_target(handle.user_scn, obs.target_pos)
                _mark_joint(handle.user_scn, *session.selected_joint_axis())
            # set_images (mjr_drawPixels) sets no OpenGL state of its own: the panels are only
            # drawn after a text overlay has run init2D, which turns off the depth test left on
            # by the 3D scene. An empty overlay does that; the status panel covers its box.
            handle.set_texts([(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "", "")])
            while handle.is_running():
                now = time.perf_counter()
                if duration is not None and now - started >= duration:
                    closed_by = f"duration of {duration:g} s elapsed"
                    break
                is_paused = paused.is_set()
                if is_paused:
                    paused_seconds += now - previous
                    due = 0
                    next_due = now + dt  # resume without catching up the pause
                else:
                    due = int((now - next_due) / dt) + 1 if now >= next_due else 0
                    if due > CATCH_UP_LIMIT:  # long stall (e.g. window dragged): drop the backlog
                        resyncs += 1
                        due = CATCH_UP_LIMIT
                        next_due = now + dt
                    else:
                        next_due += due * dt
                previous = now
                messages = []
                with handle.lock():
                    messages.append(session.check_external_changes())
                    while True:
                        try:
                            session.press(keys.get_nowait())
                        except queue.Empty:
                            break
                    messages.append(session.step(due))
                    obs = session.obs
                    _mark_target(handle.user_scn, obs.target_pos)
                    _mark_joint(handle.user_scn, *session.selected_joint_axis())
                    vp = handle.viewport
                    viewport = None if vp is None else (vp.left, vp.bottom, vp.width, vp.height)
                for message in filter(None, messages):
                    log(message)
                key = (is_paused, session.revision)
                if key != overlay_key or now - overlay_at >= OVERLAY_INTERVAL_SECONDS:
                    panels = panel_layout(viewport, status_rows(seed, physics_label, obs, is_paused, session.jog))
                    if panels:
                        handle.set_images(panels)
                    elif sent_panels and sent_panels[-1]:  # view too small or minimized
                        handle.clear_images()
                    sent_panels.append(panels)
                    overlay_at, overlay_key = now, key
                handle.sync()
                if screenshot is not None and capture is None and now - started >= screenshot_delay:
                    capture = capture_own_window()  # pixels only; saved after the loop
                    image, info = capture
                    if image is None:
                        log(f"Viewer window capture failed: {info.get('error')}")
                    else:
                        age = overlay_age(image, sent_panels)
                        info.update(overlay_exact=age is not None, overlay_age=age)
                        log(
                            f"Captured the viewer window ({info['width']}x{info['height']} px) at {now - started:.1f} s; "
                            + (
                                "status panels pixel-exact."
                                if age is not None
                                else "status panels differ from those sent (not drawn or not shown yet)."
                            )
                        )
                delay = next_due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            ended = time.perf_counter()
    except KeyboardInterrupt:
        closed_by = "keyboard interrupt"
    finally:
        if ended is None:
            ended = time.perf_counter()
        if not _join_viewer_thread():
            log("note: the MuJoCo viewer thread was still running 5 s after the window was closed")
    close_seconds = time.perf_counter() - ended
    shot: dict[str, Any] | None = None
    if screenshot is not None:
        if capture is None:
            shot = {"saved": False, "error": "viewer closed before the screenshot delay elapsed"}
        else:
            shot = _save_window_capture(screenshot, *capture)
    wall = ended - started
    active = wall - paused_seconds
    obs = runner.observe()
    return ViewerSummary(
        seed=seed,
        physics_source=truth.physics_source,
        closed_by=closed_by,
        wall_seconds=wall,
        paused_seconds=paused_seconds,
        close_seconds=close_seconds,
        sim_seconds=session.updates * dt,
        realtime_factor=session.updates * dt / active if active >= 0.5 else None,
        resyncs=resyncs,
        control_updates=session.updates,
        restores=session.restores,
        unstable_events=session.unstable,
        operator_commands=session.operator_commands,
        option_reverts=session.option_reverts,
        final_time=obs.time,
        final_block_target_distance=obs.block_target_distance,
        updates_since_reset=session.since_reset,
        final_state_digest=runner.state_digest(),
        screenshot=shot,
        final_state=runner.state_vector(),
        commands_since_reset={count: ctrl.copy() for count, ctrl in session.commands.items()},
    )
