"""configs/scene.toml: the timing contract, the brief's ranges and validation of bad values."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from physical_ai_lab import paths
from physical_ai_lab.config import Range, SceneConfig, load_scene_config


def test_timing_contract(cfg: SceneConfig) -> None:
    # BUILD_PLAN.md section 4: 0.002 s physics steps, 50 Hz control, exactly ten physics steps per update.
    t = cfg.timing
    assert (t.physics_timestep, t.control_hz, t.substeps) == (0.002, 50.0, 10)
    assert math.isclose(t.control_dt, 1.0 / t.control_hz, rel_tol=0, abs_tol=1e-12)


def test_brief_values(cfg: SceneConfig) -> None:
    # BUILD_PLAN.md section 4: ~5 cm block, 20 mm centre-based success radius, development ranges.
    assert cfg.block.edge == 0.05
    assert cfg.workspace.target_tolerance == 0.02
    assert cfg.physics.mass_range.as_list() == [0.10, 0.40]
    assert cfg.physics.sliding_friction_range.as_list() == [0.20, 0.80]
    assert set(cfg.cameras) == {"overhead", "oblique"}


def test_mjcf_path_is_inside_the_project(cfg: SceneConfig) -> None:
    assert cfg.mjcf_path == paths.SCENE_XML.resolve()
    assert cfg.mjcf_path.is_file()


def _variant(tmp_path: Path, old: str, new: str) -> Path:
    text = paths.SCENE_CONFIG.read_text(encoding="utf-8")
    assert text.count(old) == 1, f"{old!r} must occur exactly once in scene.toml"
    path = tmp_path / "scene.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("substeps = 10", "substeps = 9", "timing mismatch"),
        ("schema_version = 1", "schema_version = 2", "schema_version"),
        ("nominal_mass = 0.25", "nominal_mass = 0.5", "nominal physics"),
        ("mass_range = [0.10, 0.40]", "mass_range = [0.40, 0.10]", "invalid range"),
        ("target_tolerance = 0.02", "target_tolerance = 0.0", "target_tolerance"),
        ("reset_state_tolerance = 0.0", "reset_state_tolerance = -1.0", "reset_state_tolerance"),
        ("block_x = [0.42, 0.62]", "block_x = [0.42, 0.90]", "allowed block region"),
        ("visibility_seeds = 50", "visibility_seeds = 0", "visibility_seeds"),
        ("max_arm_occlusion = 0.05", "max_arm_occlusion = 1.0", "max_arm_occlusion"),
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_scene_config(_variant(tmp_path, old, new))


def test_range_parsing() -> None:
    r = Range.parse([0.1, 0.4])
    assert r.contains(0.1) and r.contains(0.4) and not r.contains(0.41)
    assert r.contains(0.41, tol=0.02)
    for bad in ([1.0], [0.0, float("nan")], "0.1,0.4", [0.5, 0.1]):
        with pytest.raises(ValueError):
            Range.parse(bad)
