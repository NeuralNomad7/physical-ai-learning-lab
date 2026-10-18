"""Offscreen rendering of the scene cameras (simulator side).

Rendering needs a working OpenGL context (GLFW on Windows by default; set
``MUJOCO_GL=egl`` or ``osmesa`` on headless Linux). Headless stepping does not
depend on this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_SIZE = (800, 600)  # width, height; must fit <visual><global offwidth/offheight> in the MJCF
VISIBILITY_SIZE = (640, 480)  # segmentation renders for the occlusion check
_UNDRAWN_GROUP = 5  # geom/site group that VisibilityProbe switches off to hide the robot


def _check_size(model: mujoco.MjModel, size: tuple[int, int]) -> None:
    width, height = size
    if width > model.vis.global_.offwidth or height > model.vis.global_.offheight:
        raise ValueError(
            f"requested {width}x{height} exceeds offscreen buffer "
            f"{model.vis.global_.offwidth}x{model.vis.global_.offheight}"
        )


def render_cameras(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cameras: Sequence[str],
    size: tuple[int, int] = DEFAULT_SIZE,
) -> dict[str, np.ndarray]:
    """Render RGB images (H x W x 3, uint8) from the named cameras for the current state."""
    _check_size(model, size)
    width, height = size
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        images = {}
        for camera in cameras:
            renderer.update_scene(data, camera=camera)
            images[camera] = renderer.render().copy()
        return images
    finally:
        renderer.close()


@dataclass(frozen=True)
class Visibility:
    """Pixel counts of one geom in one camera, rendered with and without the robot."""

    camera: str
    geom: str
    pixels: int  # in the normal render
    pixels_without_robot: int  # robot not drawn; other scene objects still occlude

    @property
    def in_view(self) -> bool:
        return self.pixels_without_robot > 0

    @property
    def robot_occlusion(self) -> float:
        """Fraction of the geom's otherwise visible area that the robot hides (nan if not in view)."""
        if not self.in_view:
            return math.nan
        return max(0.0, 1.0 - self.pixels / self.pixels_without_robot)


class VisibilityProbe:
    """Measure with segmentation renders how much of chosen geoms the robot hides in each camera.

    The robot is every geom and site in the kinematic tree that contains ``robot_body``. For the
    second render they are moved to a group the renderer does not draw; ``measure`` restores the
    model's groups before returning, also on error. Groups only affect drawing, not physics.
    """

    def __init__(self, model: mujoco.MjModel, robot_body: str, size: tuple[int, int] = VISIBILITY_SIZE) -> None:
        _check_size(model, size)
        root = int(model.body_rootid[model.body(robot_body).id])
        if root == 0:
            raise ValueError(f"{robot_body!r} is the world body, not part of a robot tree")
        self._model = model
        self._robot_geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] == root)
        self._robot_sites = np.flatnonzero(model.body_rootid[model.site_bodyid] == root)
        self._option = mujoco.MjvOption()
        self._option.geomgroup[_UNDRAWN_GROUP] = 0
        self._option.sitegroup[_UNDRAWN_GROUP] = 0
        width, height = size
        # Segmentation ids are encoded as colours. A multisampled buffer averages the colours of edge
        # pixels, which then decode as some other object, so the probe's context is single-sample
        # (the setting is read once, when the context is created).
        samples = model.vis.quality.offsamples
        model.vis.quality.offsamples = 0
        try:
            self._renderer = mujoco.Renderer(model, height=height, width=width)
        finally:
            model.vis.quality.offsamples = samples
        self._renderer.enable_segmentation_rendering()

    @property
    def robot_geom_count(self) -> int:
        return int(self._robot_geoms.size)

    def _pixels(self, data: mujoco.MjData, camera: str, geom_ids: Sequence[int]) -> list[int]:
        self._renderer.update_scene(data, camera=camera, scene_option=self._option)
        seg = self._renderer.render()  # [..., 0] object id, [..., 1] object type; -1 = background
        ids = seg[..., 0][seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)]
        counts = np.bincount(ids, minlength=self._model.ngeom)
        return [int(counts[g]) for g in geom_ids]

    def measure(self, data: mujoco.MjData, cameras: Sequence[str], geoms: Sequence[str]) -> list[Visibility]:
        """Pixel counts for every (camera, geom) pair at the current state of ``data``."""
        model = self._model
        ids = [model.geom(name).id for name in geoms]
        shown = {camera: self._pixels(data, camera, ids) for camera in cameras}
        geom_groups = model.geom_group[self._robot_geoms].copy()
        site_groups = model.site_group[self._robot_sites].copy()
        try:
            model.geom_group[self._robot_geoms] = _UNDRAWN_GROUP
            model.site_group[self._robot_sites] = _UNDRAWN_GROUP
            alone = {camera: self._pixels(data, camera, ids) for camera in cameras}
        finally:
            model.geom_group[self._robot_geoms] = geom_groups
            model.site_group[self._robot_sites] = site_groups
        return [
            Visibility(camera, name, shown[camera][i], alone[camera][i])
            for camera in cameras
            for i, name in enumerate(geoms)
        ]

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> VisibilityProbe:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def gl_info() -> dict[str, str]:
    """OpenGL vendor/renderer/version strings of the context MuJoCo creates for offscreen rendering."""
    import ctypes
    import ctypes.util
    import sys

    context = mujoco.GLContext(64, 64)
    context.make_current()
    try:
        if sys.platform == "win32":
            library = ctypes.WinDLL("opengl32")
        else:
            name = ctypes.util.find_library("GL") or ctypes.util.find_library("OpenGL")
            if name is None:
                return {"note": "OpenGL library not found for querying strings"}
            library = ctypes.CDLL(name)
        library.glGetString.argtypes = [ctypes.c_uint]
        library.glGetString.restype = ctypes.c_char_p
        strings = {}
        for key, enum in (("vendor", 0x1F00), ("renderer", 0x1F01), ("version", 0x1F02)):
            value = library.glGetString(enum)
            strings[key] = value.decode("utf-8", "replace") if value else "(unavailable)"
        return strings
    finally:
        context.free()


def image_stats(image: np.ndarray) -> dict[str, float]:
    """Simple checks that a render is not blank (all one colour)."""
    if image.dtype != np.uint8 or image.ndim != 3 or not 1 <= image.shape[-1] <= 4:
        raise ValueError(f"expected an H x W x C uint8 image with C <= 4, got {image.dtype} {image.shape}")
    # Pack each pixel into one integer: np.unique(axis=0) on rows is ~50x slower for large images.
    packed = np.zeros(image.shape[:2], dtype=np.uint32)
    for channel in range(image.shape[-1]):
        packed = (packed << 8) | image[..., channel]
    return {
        "mean": float(image.mean()),
        "std": float(image.std()),
        "unique_colours": float(np.unique(packed).size),
    }


def is_nonblank(image: np.ndarray, stats: dict[str, float] | None = None) -> bool:
    stats = image_stats(image) if stats is None else stats
    return stats["std"] > 5.0 and stats["unique_colours"] > 50


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def compose(images: dict[str, np.ndarray], caption: str) -> Image.Image:
    """Place camera images side by side with a label above each and a caption below."""
    tiles = [Image.fromarray(image) for image in images.values()]
    label_h, caption_h, pad = 36, 40, 8
    width = sum(t.width for t in tiles) + pad * (len(tiles) + 1)
    height = max(t.height for t in tiles) + label_h + caption_h + pad
    canvas = Image.new("RGB", (width, height), (24, 26, 30))
    draw = ImageDraw.Draw(canvas)
    label_font, caption_font = _font(22), _font(18)
    x = pad
    for name, tile in zip(images, tiles):
        draw.text((x, 8), f"camera: {name}", fill=(235, 235, 235), font=label_font)
        canvas.paste(tile, (x, label_h))
        x += tile.width + pad
    draw.text((pad, label_h + max(t.height for t in tiles) + 10), caption, fill=(200, 200, 200), font=caption_font)
    return canvas


def save_renders(
    images: dict[str, np.ndarray], out_dir: Path, caption: str, prefix: str = "scene"
) -> dict[str, Path]:
    """Write one PNG per camera plus a labelled composite; return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, image in images.items():
        path = out_dir / f"{prefix}_{name}.png"
        Image.fromarray(image).save(path)
        written[name] = path
    composite = out_dir / f"{prefix}_composite.png"
    compose(images, caption).save(composite)
    written["composite"] = composite
    return written
