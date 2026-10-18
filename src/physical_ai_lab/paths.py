"""Filesystem locations used by the lab.

The project is used from a source checkout (editable install), so paths are
resolved relative to this file. Set PHYSICAL_AI_LAB_ROOT to override.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PHYSICAL_AI_LAB_ROOT", PACKAGE_DIR.parent.parent)).resolve()

ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"

MENAGERIE_DIR = ASSETS_DIR / "third_party" / "mujoco_menagerie"
PANDA_DIR = MENAGERIE_DIR / "franka_emika_panda"
PANDA_MANIFEST = MENAGERIE_DIR / "franka_emika_panda.manifest.json"
PANDA_XML = PANDA_DIR / "panda.xml"

SCENE_XML = ASSETS_DIR / "scenes" / "panda_push_scene.xml"
SCENE_CONFIG = CONFIGS_DIR / "scene.toml"
PUSH_CONFIG = CONFIGS_DIR / "push.toml"
BENCHMARK_CONFIG = CONFIGS_DIR / "benchmark.toml"
# Phase 3 benchmark manifests (versioned, hashed). The sealed ones are opened only after a freeze.
MANIFESTS_DIR = CONFIGS_DIR / "manifests"
# Frozen-candidate records (Phase 5 writes the first one; Phase 3 only defines the step).
FREEZES_DIR = PROJECT_ROOT / "freezes"


def python_command() -> str:
    """This interpreter as typed from the project root, e.g. ``.\\.venv\\Scripts\\python.exe``.

    Falls back to ``python`` when the interpreter is not inside the project folder.
    """
    try:
        relative = Path(sys.executable).absolute().relative_to(PROJECT_ROOT)
    except ValueError:
        return "python"
    return f".{os.sep}{relative}"
