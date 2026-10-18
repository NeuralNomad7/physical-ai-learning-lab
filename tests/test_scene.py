"""Scene structure: Panda, table, free block, goal, cameras, contact model and actuators."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from physical_ai_lab.config import SceneConfig
from physical_ai_lab.scenario import ScenarioRunner, verify_actuators, verify_scene


@pytest.fixture
def fresh_model(cfg: SceneConfig) -> mujoco.MjModel:
    """A separate model that a test may break without affecting the shared runner."""
    return mujoco.MjModel.from_xml_path(str(cfg.mjcf_path))


def test_model_sizes_and_solver_options(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    model = runner.sim_model
    # 7 arm hinges + 2 finger slides + the block's free joint (7 qpos, 6 dof); 7 arm servos + gripper.
    assert (model.nq, model.nv, model.nu) == (16, 15, 8)
    assert model.opt.timestep == cfg.timing.physics_timestep
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.nmocap == 1


def test_scene_matches_configuration(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    notes = verify_scene(runner.sim_model, cfg)
    assert any("target disc radius" in note for note in notes)
    infos = verify_actuators(runner.sim_model, cfg)
    assert [info.name for info in infos] == [*cfg.robot.arm_actuators, cfg.robot.gripper_actuator]
    assert all(info.kp > 0 for info in infos)


def test_goal_marker_never_collides(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    model = runner.sim_model
    target = model.body(cfg.target.mocap_body).id
    geoms = np.flatnonzero(model.geom_bodyid == target)
    assert geoms.size >= 1
    assert not model.geom_contype[geoms].any() and not model.geom_conaffinity[geoms].any()


def test_verify_scene_rejects_a_pyramidal_cone(fresh_model: mujoco.MjModel, cfg: SceneConfig) -> None:
    verify_scene(fresh_model, cfg)
    fresh_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    with pytest.raises(ValueError, match="elliptic"):
        verify_scene(fresh_model, cfg)


def test_verify_scene_rejects_a_colliding_goal(fresh_model: mujoco.MjModel, cfg: SceneConfig) -> None:
    fresh_model.geom_contype[fresh_model.geom(cfg.target.zone_geom).id] = 1
    with pytest.raises(ValueError, match="target zone must not collide"):
        verify_scene(fresh_model, cfg)


def test_verify_actuators_rejects_a_non_servo(fresh_model: mujoco.MjModel, cfg: SceneConfig) -> None:
    verify_actuators(fresh_model, cfg)
    fresh_model.actuator_biasprm[fresh_model.actuator(cfg.robot.arm_actuators[3]).id, 1] = 0.0
    with pytest.raises(ValueError, match="not a position servo"):
        verify_actuators(fresh_model, cfg)
