"""Environment facts for `doctor`, the smoke-test log and docs/PROGRESS.md."""

from __future__ import annotations

import os
import platform
import re
import struct
import sys
import sysconfig
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__, paths

_MACHINES = {0x014C: "x86", 0x8664: "AMD64", 0xAA64: "ARM64", 0x01C4: "ARM"}
_WINDOWS_PLATFORMS = {"win-amd64": "AMD64", "win-arm64": "ARM64", "win32": "x86"}


def process_machine() -> str:
    """CPU architecture this Python interpreter is built for (the wheels it can load).

    On Windows, ``platform.machine()`` can report the native CPU instead (Python 3.12
    queries WMI), so the interpreter's platform tag is used there.
    """
    if sys.platform == "win32":
        return _WINDOWS_PLATFORMS.get(sysconfig.get_platform(), platform.machine())
    return platform.machine()


def host_machine() -> str:
    """Native CPU architecture of the host, even when this process runs under emulation (Windows only)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.IsWow64Process2.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(wintypes.USHORT), ctypes.POINTER(wintypes.USHORT),
            ]
            kernel32.IsWow64Process2.restype = wintypes.BOOL
            process, native = wintypes.USHORT(), wintypes.USHORT()
            if kernel32.IsWow64Process2(kernel32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)):
                return _MACHINES.get(native.value, hex(native.value))
        except (AttributeError, OSError):
            pass
        return os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE") or "unknown"
    return platform.machine()


def dependency_pins(pyproject: Path = paths.PROJECT_ROOT / "pyproject.toml") -> dict[str, str]:
    """Exact ``name==version`` pins from the project dependencies (runtime and dev groups)."""
    raw = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    requirements = list(raw["project"]["dependencies"])
    for group in raw.get("dependency-groups", {}).values():
        requirements.extend(r for r in group if isinstance(r, str))
    pins = {}
    for requirement in requirements:
        match = re.fullmatch(r"\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)\s*", requirement)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_root(start: Path = paths.PROJECT_ROOT) -> Path | None:
    for folder in (start, *start.parents):
        if (folder / ".git").exists():
            return folder
    return None


def environment_info() -> dict[str, Any]:
    import mujoco
    import numpy
    import PIL

    machine = process_machine()
    host = host_machine()
    pins = dependency_pins()
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "python_platform": sysconfig.get_platform(),
        "process_machine": machine,
        "process_bits": struct.calcsize("P") * 8,
        "host_machine": host,
        "emulated": host.upper() != machine.upper(),
        "os": platform.platform(),
        "versions": {
            "physical_ai_lab": __version__,
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
            "pillow": PIL.__version__,
            "pytest": installed_version("pytest"),
        },
        "pins": pins,
        "pin_mismatches": {
            name: {"pinned": pinned, "installed": installed_version(name)}
            for name, pinned in pins.items()
            if installed_version(name) != pinned
        },
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "project_root": str(paths.PROJECT_ROOT),
        "onedrive_folder": "onedrive" in str(paths.PROJECT_ROOT).lower(),
        "git_root": str(git_root()) if git_root() else None,
    }
