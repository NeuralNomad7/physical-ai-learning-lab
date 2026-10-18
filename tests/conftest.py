"""Shared fixtures for the test suite (Phases 1 to 3).

Run from the project folder with the project interpreter:
``.\\.venv\\Scripts\\python.exe -m pytest`` (Windows) or ``.venv/bin/python -m pytest``.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterator

import mujoco
import pytest

from physical_ai_lab import paths
from physical_ai_lab.config import SceneConfig, load_scene_config
from physical_ai_lab.scenario import ScenarioRunner, capture_mujoco_warnings


@pytest.fixture(scope="session")
def cfg() -> SceneConfig:
    return load_scene_config()


@pytest.fixture(scope="session")
def runner(cfg: SceneConfig) -> ScenarioRunner:
    """One runner for the session; every test that uses it resets it first."""
    return ScenarioRunner(cfg)


@pytest.fixture(autouse=True)
def no_mujoco_warnings() -> Iterator[list[str]]:
    """Fail a test if MuJoCo emits any warning while it runs (sign of an invalid or unstable scene)."""
    with capture_mujoco_warnings() as messages:
        yield messages
    assert not messages, f"MuJoCo warnings during the test: {messages}"


@pytest.fixture(scope="session")
def opengl() -> None:
    """Skip rendering tests when no OpenGL context can be created (e.g. a headless machine)."""
    try:
        context = mujoco.GLContext(64, 64)
    except Exception as exc:  # noqa: BLE001 - any failure means rendering is unavailable
        pytest.skip(f"no OpenGL context for offscreen rendering: {exc}")
    context.free()


CliRunner = Callable[..., subprocess.CompletedProcess]


@pytest.fixture(scope="session")
def cli() -> CliRunner:
    """Run ``python -m physical_ai_lab ...`` in a new interpreter from the project folder."""

    def run(*args: str, timeout: float = 300.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "physical_ai_lab", *args],
            cwd=paths.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    return run


# --------------------------------------------------------------------------- Phase 3 fixtures

SCRATCH_GENERATOR_SEED = 999  # not the benchmark's generator seed: tests never run the real manifests
SCRATCH_RESET_SEED_START = 90000


@pytest.fixture(scope="session")
def push_cfg():
    from physical_ai_lab.push_config import load_push_config

    return load_push_config()


@pytest.fixture(scope="session")
def bench_cfg(cfg, push_cfg):
    from physical_ai_lab.benchmark_config import load_benchmark_config

    return load_benchmark_config(scene=cfg, push=push_cfg)


@pytest.fixture(scope="session")
def scratch_manifests(cfg, bench_cfg, tmp_path_factory):
    """A throwaway manifest set (different generator seed, temporary folder) for harness tests.

    Episodes in tests run on these, never on configs/manifests, so no benchmark scenario is used
    by a test and the sealed splits stay untouched.
    """
    import dataclasses

    from physical_ai_lab import manifest as mf

    bench = dataclasses.replace(bench_cfg, manifests=dataclasses.replace(
        bench_cfg.manifests, generator_seed=SCRATCH_GENERATOR_SEED, reset_seed_start=SCRATCH_RESET_SEED_START))
    root = tmp_path_factory.mktemp("scratch_manifests")
    mf.write_manifests(mf.generate_manifests(cfg, bench, mf.manifest_provenance(cfg)), root / "manifests")
    return bench, root / "manifests", root / "freezes"


@pytest.fixture(scope="session")
def scratch_run(cfg, push_cfg, scratch_manifests, tmp_path_factory):
    """Two scratch validation episodes through ``evaluate_split`` (the real agent process, the baseline)."""
    from physical_ai_lab.benchmark import evaluate_split, load_records

    bench, manifest_dir, freeze_dir = scratch_manifests
    out = tmp_path_factory.mktemp("scratch_run") / "validation"
    summary = evaluate_split("validation", mode="baseline", out_dir=out, scene=cfg, push=push_cfg, bench=bench,
                             manifest_dir=manifest_dir, freeze_dir=freeze_dir, limit=2, command="pytest",
                             progress=lambda line: None)
    return {"summary": summary, "records": load_records(out), "dir": out, "bench": bench,
            "manifest_dir": manifest_dir, "freeze_dir": freeze_dir}
