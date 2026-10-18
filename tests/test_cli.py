"""Command line: the registered command set, argument validation and reproducibility across processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from physical_ai_lab import cli as lab_cli
from physical_ai_lab import paths
from physical_ai_lab.config import SceneConfig
from physical_ai_lab.observation import OBSERVATION_FIELDS
from physical_ai_lab.scenario import ScenarioRunner

PHASE1_COMMANDS = {"doctor", "fetch-assets", "view", "step", "render", "smoke"}
PHASE2_COMMANDS = {"gate", "demo"}  # DECISIONS.md D22: Phase 2 adds exactly these two
# DECISIONS.md D30: Phase 3 adds exactly these four; collect, train, dashboard and a freeze command come later.
PHASE3_COMMANDS = {"contact-check", "manifests", "evaluate", "replay"}


def test_only_phase1_to_phase3_commands_are_registered() -> None:
    subparsers = [a for a in lab_cli.build_parser()._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(subparsers) == 1
    assert set(subparsers[0].choices) == PHASE1_COMMANDS | PHASE2_COMMANDS | PHASE3_COMMANDS


@pytest.mark.parametrize(
    "argv",
    [
        ["step", "--seed", "-1"],
        ["step", "--seed", "abc"],
        ["step", "--seed", "1.5"],
        ["step", "--seconds", "-1"],
        ["step", "--seconds", "nan"],
        ["view", "--duration", "inf"],
        ["predict"],
        ["train"],
        ["evaluate"],
        ["evaluate", "--split", "holdout"],
        ["evaluate", "--split", "train", "--mode", "learner"],
        ["evaluate", "--split", "train", "--limit", "0"],
        ["replay"],
        [],
    ],
)
def test_invalid_arguments_are_usage_errors(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        lab_cli.main(argv)
    assert excinfo.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_open_scene_command_uses_this_interpreter() -> None:
    assert lab_cli.open_scene_command() == f"{paths.python_command()} -m physical_ai_lab view --seed 0"


def test_step_in_a_new_process_reproduces_the_in_process_state(
    cli, runner: ScenarioRunner, cfg: SceneConfig
) -> None:
    seconds = cfg.checks.rollout_seconds
    updates = round(seconds / cfg.timing.control_dt)
    runner.reset(0)
    runner.step(updates)
    result = cli("step", "--seed", "0", "--seconds", f"{seconds:g}", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["state_digest"] == runner.state_digest()
    assert payload["stable"] is True and payload["reason"] == "stable"
    assert payload["physics_source"] == "nominal"
    assert payload["control_updates"] == updates
    assert tuple(payload["observation"]) == OBSERVATION_FIELDS
    assert "hidden_truth" not in payload
    assert "MuJoCo warning" not in result.stderr


def test_hidden_truth_is_printed_only_on_request(cli, runner: ScenarioRunner) -> None:
    hidden = cli("step", "--seed", "3", "--seconds", "0.2", "--randomize-physics")
    assert hidden.returncode == 0, hidden.stderr
    assert "hidden physics randomized (values hidden)" in hidden.stdout
    assert "hidden truth" not in hidden.stdout and " kg" not in hidden.stdout

    shown = cli("step", "--seed", "3", "--seconds", "0.2", "--randomize-physics", "--show-truth", "--json")
    assert shown.returncode == 0, shown.stderr
    truth = json.loads(shown.stdout)["hidden_truth"]
    runner.reset(3, randomize_physics=True)
    assert truth == json.loads(json.dumps(runner.hidden_truth().to_dict()))


@pytest.mark.usefixtures("opengl")
def test_doctor_passes(cli) -> None:
    result = cli("doctor")
    assert result.returncode == 0, result.stdout + result.stderr
    assert lab_cli.open_scene_command() in result.stdout


@pytest.mark.usefixtures("opengl")
def test_render_writes_the_camera_images(cli, tmp_path: Path, cfg: SceneConfig) -> None:
    result = cli("render", "--seed", "2", "--seconds", "0.1", "--out", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    expected = {f"scene_{camera}.png" for camera in cfg.cameras} | {"scene_composite.png"}
    assert {path.name for path in tmp_path.iterdir()} == expected
