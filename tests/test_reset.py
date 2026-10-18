"""Seeded reset: reproducibility within the documented tolerance, layouts and argument checks.

Reproducibility scope (configs/scene.toml [checks]): same seed, same machine, same MuJoCo
build. The tolerance is 0.0, i.e. the full integration state must be bitwise identical.
The separate-process comparison is in test_cli.py.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from physical_ai_lab.config import SceneConfig
from physical_ai_lab.scenario import HiddenPhysics, ScenarioRunner

ROLLOUT_UPDATES = 100  # 2 s at 50 Hz

# Every model option that the Physics panel of MuJoCo 3.14's viewer (simulate.cc, MakePhysicsSection)
# can edit. In the passive viewer the edits are written into the user's MjModel.opt.
PHYSICS_PANEL_OPTIONS = (
    "integrator", "cone", "jacobian", "solver", "timestep", "iterations", "tolerance", "ls_iterations",
    "ls_tolerance", "noslip_iterations", "noslip_tolerance", "ccd_iterations", "ccd_tolerance", "sleep_tolerance",
    "sdf_iterations", "sdf_initpoints", "gravity", "wind", "magnetic", "density", "viscosity", "impratio",
    "disableflags", "enableflags", "o_margin", "o_solimp", "o_solref", "o_friction", "disableactuator",
)


def rollout(runner: ScenarioRunner, seed: int, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
    runner.reset(seed, **kwargs)
    start = runner.state_vector().copy()
    runner.step(ROLLOUT_UPDATES)
    return start, runner.state_vector().copy()


def max_diff(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    return float(np.max(np.abs(a - b)))


def test_same_seed_is_reproduced_after_other_episodes(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    tol = cfg.checks.reset_state_tolerance
    p = cfg.physics
    start, end = rollout(runner, 0)
    rollout(runner, 7, randomize_physics=True)
    rollout(runner, 3, physics=HiddenPhysics(p.mass_range.high, p.sliding_friction_range.low))
    start_again, end_again = rollout(runner, 0)
    assert max_diff(start, start_again) <= tol
    assert max_diff(end, end_again) <= tol


def test_fresh_runner_reproduces_the_same_rollout(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    start, end = rollout(runner, 0)
    other_start, other_end = rollout(ScenarioRunner(cfg), 0)
    assert max_diff(start, other_start) <= cfg.checks.reset_state_tolerance
    assert max_diff(end, other_end) <= cfg.checks.reset_state_tolerance


def test_reset_undoes_physics_option_edits(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    tol = cfg.checks.reset_state_tolerance
    start, end = rollout(runner, 0)
    opt = runner.sim_model.opt
    opt.timestep = 2 * cfg.timing.physics_timestep
    opt.iterations = 5
    opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_OVERRIDE)
    opt.o_friction[0] = 0.1
    start_again, end_again = rollout(runner, 0)
    assert max_diff(start, start_again) <= tol
    assert max_diff(end, end_again) <= tol
    assert opt.timestep == cfg.timing.physics_timestep
    assert runner.restore_options() == ()


def test_restore_options_puts_back_every_physics_panel_option(runner: ScenarioRunner) -> None:
    opt = runner.sim_model.opt
    before = {name: np.array(getattr(opt, name)) for name in PHYSICS_PANEL_OPTIONS}
    try:
        for name in PHYSICS_PANEL_OPTIONS:
            value = getattr(opt, name)
            if isinstance(value, np.ndarray):
                value += 1.0  # a view into the model
            else:
                setattr(opt, name, value + 1)
        assert set(runner.restore_options()) == set(PHYSICS_PANEL_OPTIONS)
    finally:
        runner.restore_options()
    for name, value in before.items():
        assert np.asarray(getattr(opt, name)).tobytes() == value.tobytes(), name
    assert runner.restore_options() == ()


def test_different_seeds_give_different_states(runner: ScenarioRunner) -> None:
    runner.reset(0)
    first = runner.state_vector().copy()
    runner.reset(1)
    assert max_diff(first, runner.state_vector()) > 0.0


def test_layouts_are_inside_the_regions_and_at_rest(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    w = cfg.workspace
    rest_z = cfg.table.top_z + cfg.block.edge / 2
    layouts = set()
    for seed in range(100):
        obs = runner.reset(seed)
        truth = runner.hidden_truth()
        (bx, by), yaw, (tx, ty) = truth.block_xy, truth.block_yaw, truth.target_xy
        assert w.block_x.contains(bx) and w.block_y.contains(by), seed
        assert w.target_x.contains(tx) and w.target_y.contains(ty), seed
        assert w.block_target_distance.contains(math.hypot(tx - bx, ty - by)), seed
        np.testing.assert_allclose(obs.block_pos, [bx, by, rest_z], rtol=0, atol=1e-9)
        assert abs(math.remainder(obs.block_yaw - yaw, math.tau)) < 1e-9
        np.testing.assert_array_equal(obs.target_pos, [tx, ty, cfg.table.top_z])
        assert not obs.robot_block_contact and not obs.robot_table_contact, seed
        assert not (obs.block_linvel.any() or obs.block_angvel.any() or obs.arm_qvel.any()), seed
        np.testing.assert_array_equal(obs.arm_qpos, runner.home_arm_qpos)
        assert obs.time == 0.0 and obs.control_step == 0
        layouts.add((bx, by, yaw, tx, ty))
    assert len(layouts) == 100


def test_layout_does_not_depend_on_physics_randomization(runner: ScenarioRunner) -> None:
    for seed in range(20):
        runner.reset(seed)
        nominal = runner.hidden_truth()
        runner.reset(seed, randomize_physics=True)
        randomized = runner.hidden_truth()
        assert (nominal.block_xy, nominal.block_yaw, nominal.target_xy) == (
            randomized.block_xy,
            randomized.block_yaw,
            randomized.target_xy,
        )
        assert nominal.physics_source == "nominal" and randomized.physics_source == "randomized"


def test_randomized_physics_is_seeded_and_inside_the_ranges(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    p = cfg.physics
    seen = set()
    for seed in range(50):
        runner.reset(seed, randomize_physics=True)
        truth = runner.hidden_truth()
        assert p.mass_range.contains(truth.physics.block_mass)
        assert p.sliding_friction_range.contains(truth.physics.sliding_friction)
        assert runner.applied_physics() == truth.physics
        runner.reset(seed, randomize_physics=True)
        assert runner.hidden_truth() == truth
        seen.add(truth.physics)
    assert len(seen) == 50


def test_explicit_physics_is_applied(runner: ScenarioRunner) -> None:
    physics = HiddenPhysics(block_mass=0.33, sliding_friction=0.61)
    runner.reset(2, physics=physics)
    assert runner.hidden_truth().physics_source == "explicit"
    assert runner.applied_physics() == physics


@pytest.mark.parametrize(
    ("seed", "error"),
    [(-1, ValueError), (1.5, TypeError), (True, TypeError), ("0", TypeError), (None, TypeError)],
)
def test_invalid_seeds_are_rejected(runner: ScenarioRunner, seed: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        runner.reset(seed)  # type: ignore[arg-type]


def test_numpy_integer_seed_equals_python_int(runner: ScenarioRunner) -> None:
    runner.reset(np.int64(4))
    from_numpy = runner.state_vector().copy()
    runner.reset(4)
    np.testing.assert_array_equal(from_numpy, runner.state_vector())


def test_conflicting_physics_options_are_rejected(runner: ScenarioRunner) -> None:
    with pytest.raises(ValueError, match="either"):
        runner.reset(0, randomize_physics=True, physics=runner.nominal_physics)


@pytest.mark.parametrize(("mass", "friction"), [(0.0, 0.5), (-0.1, 0.5), (0.25, float("nan")), (float("inf"), 0.5)])
def test_hidden_physics_values_must_be_positive_and_finite(mass: float, friction: float) -> None:
    with pytest.raises(ValueError):
        HiddenPhysics(mass, friction)


def test_runner_requires_a_reset(cfg: SceneConfig) -> None:
    fresh = ScenarioRunner(cfg)
    for call in (fresh.observe, fresh.step, fresh.hidden_truth):
        with pytest.raises(RuntimeError, match="reset"):
            call()
