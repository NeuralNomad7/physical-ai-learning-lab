"""Interpreter architecture, dependency pins and the interpreter command shown to the user."""

from __future__ import annotations

import struct
import sys
import sysconfig
from pathlib import Path

import mujoco
import pytest

from physical_ai_lab import environment, paths


def test_installed_versions_match_pins() -> None:
    pins = environment.dependency_pins()
    assert {"mujoco", "numpy", "pillow", "pytest"} <= set(pins)
    mismatches = {
        name: (pinned, environment.installed_version(name))
        for name, pinned in pins.items()
        if environment.installed_version(name) != pinned
    }
    assert not mismatches
    assert mujoco.__version__ == pins["mujoco"]


def test_process_machine_is_the_interpreter_architecture() -> None:
    assert struct.calcsize("P") * 8 == 64
    if sys.platform == "win32":
        # platform.machine() may report the native CPU (ARM64) inside an emulated x64 interpreter.
        assert environment.process_machine() == environment._WINDOWS_PLATFORMS[sysconfig.get_platform()]


def test_python_command_names_this_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    command = paths.python_command()
    if command != "python":
        assert command.startswith(".")
        assert (paths.PROJECT_ROOT / command).resolve() == Path(sys.executable).resolve()
    monkeypatch.setattr(sys, "executable", str(Path(paths.PROJECT_ROOT.anchor) / "elsewhere" / "python.exe"))
    assert paths.python_command() == "python"
