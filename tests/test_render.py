"""Offscreen rendering of the scene cameras and the checks applied to the images."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

from physical_ai_lab.config import SceneConfig
from physical_ai_lab.render import (
    VISIBILITY_SIZE,
    Visibility,
    VisibilityProbe,
    image_stats,
    is_nonblank,
    render_cameras,
    save_renders,
)
from physical_ai_lab.scenario import ScenarioRunner

SIZE = (320, 240)


def test_image_stats_on_known_images() -> None:
    flat = np.full((4, 5, 3), 7, dtype=np.uint8)
    assert image_stats(flat) == {"mean": 7.0, "std": 0.0, "unique_colours": 1.0}
    assert not is_nonblank(flat)
    one_channel_apart = np.zeros((1, 2, 3), dtype=np.uint8)
    one_channel_apart[0, 1, 2] = 1
    assert image_stats(one_channel_apart)["unique_colours"] == 2.0
    noise = np.random.default_rng(0).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    stats = image_stats(noise)
    assert stats["unique_colours"] > 4000 and stats["std"] > 50
    assert is_nonblank(noise, stats)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32), np.zeros((4, 4, 5), np.uint8)],
    ids=["2d", "float", "5-channel"],
)
def test_image_stats_rejects_non_images(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        image_stats(bad)


def test_save_renders_writes_each_camera_and_a_composite(tmp_path: Path) -> None:
    images = {"overhead": np.full((30, 40, 3), 200, np.uint8), "oblique": np.zeros((30, 40, 3), np.uint8)}
    written = save_renders(images, tmp_path / "out", "caption")
    assert {name: path.name for name, path in written.items()} == {
        "overhead": "scene_overhead.png",
        "oblique": "scene_oblique.png",
        "composite": "scene_composite.png",
    }
    for name, image in images.items():
        with Image.open(written[name]) as png:
            np.testing.assert_array_equal(np.asarray(png), image)
    with Image.open(written["composite"]) as png:
        assert png.width > 2 * 40 and png.height > 30


def test_oversized_render_is_rejected_before_using_opengl(runner: ScenarioRunner) -> None:
    model = runner.sim_model
    with pytest.raises(ValueError, match="exceeds offscreen buffer"):
        render_cameras(model, runner.sim_data, ["overhead"], size=(model.vis.global_.offwidth + 1, 240))


@pytest.mark.usefixtures("opengl")
def test_cameras_show_the_scene(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    runner.reset(0)
    first = render_cameras(runner.sim_model, runner.sim_data, cfg.cameras, size=SIZE)
    assert tuple(first) == cfg.cameras
    for name, image in first.items():
        assert image.shape == (SIZE[1], SIZE[0], 3) and image.dtype == np.uint8, name
        assert is_nonblank(image), (name, image_stats(image))
    runner.reset(1)  # another layout: the block and the goal move in the overhead view
    second = render_cameras(runner.sim_model, runner.sim_data, ["overhead"], size=SIZE)
    assert not np.array_equal(first["overhead"], second["overhead"])


def test_visibility_fractions() -> None:
    assert Visibility("cam", "geom", pixels=30, pixels_without_robot=40).robot_occlusion == 0.25
    assert Visibility("cam", "geom", pixels=40, pixels_without_robot=40).robot_occlusion == 0.0
    hidden = Visibility("cam", "geom", pixels=0, pixels_without_robot=0)
    assert not hidden.in_view and math.isnan(hidden.robot_occlusion)


@pytest.mark.usefixtures("opengl")
def test_the_arm_hides_neither_the_block_nor_the_goal(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    # The smoke test checks [checks] visibility_seeds seeds; these include seed 0, the viewer's default.
    model = runner.sim_model
    groups = (model.geom_group.copy(), model.site_group.copy())
    with VisibilityProbe(model, cfg.robot.tcp_body) as probe:
        for seed in range(5):
            runner.reset(seed)
            for v in probe.measure(runner.sim_data, cfg.cameras, (cfg.block.geom, cfg.target.zone_geom)):
                if v.camera == "overhead" or v.geom == cfg.block.geom:
                    assert v.in_view, (seed, v)
                if v.in_view:
                    assert v.robot_occlusion <= cfg.checks.max_arm_occlusion, (seed, v)
    np.testing.assert_array_equal(model.geom_group, groups[0])
    np.testing.assert_array_equal(model.site_group, groups[1])


@pytest.mark.usefixtures("opengl")
def test_visibility_probe_detects_the_arm_hiding_the_block(cfg: SceneConfig) -> None:
    # Positive control on a private model: a straight-down camera above the hand, like the first
    # version of the overhead camera, with the block directly under the hand.
    runner = ScenarioRunner(cfg)
    runner.reset(0)
    model, data = runner.sim_model, runner.sim_data
    hand = data.xpos[model.body(cfg.robot.tcp_body).id].copy()
    camera = model.camera("overhead").id
    model.cam_pos[camera] = (hand[0], hand[1], 1.35)
    model.cam_quat[camera] = (math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5))  # xyaxes "0 -1 0 1 0 0": straight down
    qadr = int(model.joint(cfg.block.joint).qposadr[0])
    data.qpos[qadr : qadr + 3] = (hand[0], hand[1], cfg.table.top_z + cfg.block.edge / 2)
    mujoco.mj_forward(model, data)
    groups = model.geom_group.copy()
    with VisibilityProbe(model, cfg.robot.tcp_body) as probe:
        (block,) = probe.measure(data, ["overhead"], [cfg.block.geom])
    assert block.in_view and block.robot_occlusion > 0.9, block
    np.testing.assert_array_equal(model.geom_group, groups)


@pytest.mark.usefixtures("opengl")
def test_visibility_probe_counts_exact_pixels(cfg: SceneConfig) -> None:
    # A camera 0.30 m above the block's top face sees only the block and the table top, so in an exact
    # segmentation the two cover every pixel. A multisampled buffer fails this: it averages the id colours
    # of edge pixels, which then decode as other geoms (here the table legs, hidden under the top).
    runner = ScenarioRunner(cfg)
    runner.reset(0)
    model, data = runner.sim_model, runner.sim_data
    x, y, yaw, height_above = 0.45, -0.12, math.radians(30.0), 0.30
    top = cfg.table.top_z + cfg.block.edge
    qadr = int(model.joint(cfg.block.joint).qposadr[0])
    data.qpos[qadr : qadr + 7] = (x, y, top - cfg.block.edge / 2, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    data.mocap_pos[model.body(cfg.target.mocap_body).mocapid[0]] = (0.90, 0.45, cfg.table.top_z)  # out of view
    camera = model.camera("overhead").id
    model.cam_pos[camera] = (x, y, top + height_above)
    model.cam_quat[camera] = (1.0, 0.0, 0.0, 0.0)  # looking straight down
    mujoco.mj_forward(model, data)
    samples = int(model.vis.quality.offsamples)
    with VisibilityProbe(model, cfg.robot.tcp_body) as probe:
        assert model.vis.quality.offsamples == samples  # the model's own setting is left as it was
        block, table = probe.measure(data, ["overhead"], [cfg.block.geom, cfg.table.top_geom])
    width, height = VISIBILITY_SIZE
    assert block.pixels + table.pixels == width * height, (block, table)
    assert block.pixels_without_robot + table.pixels_without_robot == width * height, (block, table)
    metres_per_pixel = 2 * height_above * math.tan(math.radians(model.cam_fovy[camera]) / 2) / height
    assert block.pixels == pytest.approx((cfg.block.edge / metres_per_pixel) ** 2, rel=0.01), block
