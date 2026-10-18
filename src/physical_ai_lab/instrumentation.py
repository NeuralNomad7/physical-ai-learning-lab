"""Per-episode contact and block-pose instrumentation (Phase 3, evaluator side only).

``ContactMetrics`` is a read-only monitor for ``ScenarioRunner.step(monitor=...)``: after every
physics step it measures

* robot/block contact: time in contact, number of separate contact intervals (the finger
  contacts chatter on and off during a stroke, D21, so intervals are a contact measure, never a
  push count), largest and mean summed normal force while in contact;
* block/table contact: number of contact points and summed normal force, as a ratio to the
  block weight m*g (m is hidden: evaluator only): its minimum and maximum, and its mean over steps
  without robot contact (all after a 0.2 s settle), and the time the block had no table contact;
* block lift: height of the block centre above its placement height (table top + edge/2);
* block tilt: angle between the world vertical and the nearest block face normal, i.e. how far
  the cube is from lying flat on any face (0 = flat). Measured from the body orientation every
  physics step, not inferred from the height.

It reads ``data`` and never writes it, so it does not change the dynamics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .scenario import ScenarioRunner

# Normal-force ratios ignore the first 0.2 s of physics: contact build-up after placement, not a property.
SETTLE_SECONDS = 0.2


@dataclass
class ContactSummaryMetrics:
    physics_steps: int
    robot_block_contact_seconds: float
    robot_block_contact_intervals: int
    robot_block_max_normal_force: float  # N, summed over robot/block contact points
    robot_block_mean_normal_force: float  # N, over physics steps with robot/block contact
    # Block/table normal force / (m g), from the applied (hidden) mass, over physics steps after the first
    # SETTLE_SECONDS (the block is placed at exactly table height, so the first steps are a transient).
    table_normal_ratio_min: float
    table_normal_ratio_max: float
    table_normal_ratio_free_mean: float  # mean over settled steps without robot/block contact
    block_airborne_seconds: float  # no block/table contact point at all
    max_table_contacts: int
    max_block_lift: float  # m, block centre above placement height
    max_block_tilt_deg: float  # deg, from lying flat on some face
    final_block_tilt_deg: float

    def to_dict(self) -> dict:
        return asdict(self)


def block_tilt_deg(xmat: np.ndarray) -> float:
    """Tilt of a cube from lying flat on any face, from its 3x3 rotation matrix (row-major)."""
    rotation = np.asarray(xmat, dtype=float).reshape(3, 3)
    cos_tilt = float(np.max(np.abs(rotation[2, :])))  # world-z component of each body axis
    return math.degrees(math.acos(min(1.0, cos_tilt)))


class ContactMetrics:
    """Read-only per-physics-step monitor of contact and block pose (see module docstring)."""

    def __init__(self, runner: ScenarioRunner) -> None:
        model, scene = runner.sim_model, runner.config
        self._block_body = model.body(scene.block.body).id
        self._block_geom = model.geom(scene.block.geom).id
        robot_root = model.body_rootid[model.jnt_bodyid[model.joint(scene.robot.arm_joints[0]).id]]
        self._robot_geom = np.asarray((model.body_rootid == robot_root)[model.geom_bodyid], dtype=bool)
        self._table_geom = np.asarray(model.geom_bodyid == model.body(scene.table.body).id, dtype=bool)
        self._placement_z = scene.table.top_z + scene.block.edge / 2
        self._timestep = float(model.opt.timestep)
        self._gravity = float(np.linalg.norm(model.opt.gravity))
        self._weight = math.nan  # m g, read from the compiled model at the first step (after reset)
        self._force = np.zeros(6)
        self._steps = 0
        self._rb_steps = 0
        self._rb_intervals = 0
        self._rb_prev = False
        self._rb_max = 0.0
        self._rb_sum = 0.0
        self._ratio_min = math.inf
        self._ratio_max = -math.inf
        self._settle_steps = round(SETTLE_SECONDS / self._timestep)
        self._free_sum = 0.0
        self._free_steps = 0
        self._airborne_steps = 0
        self._max_table_contacts = 0
        self._max_lift = -math.inf
        self._max_tilt = 0.0
        self._tilt = 0.0

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._steps == 0:
            # The applied (hidden) block mass, after reset. Evaluator only.
            self._weight = float(model.body_mass[self._block_body]) * self._gravity
        self._steps += 1
        rb_force, table_force, table_count, touching = 0.0, 0.0, 0, False
        block, robot, table = self._block_geom, self._robot_geom, self._table_geom
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.exclude != 0 or contact.dist > 0.0:
                continue
            g1, g2 = int(contact.geom[0]), int(contact.geom[1])
            if g1 == block:
                other = g2
            elif g2 == block:
                other = g1
            else:
                continue
            if robot[other]:
                # A penetrating robot/block contact counts as touching even at zero normal force.
                touching = True
                mujoco.mj_contactForce(model, data, i, self._force)
                rb_force += float(self._force[0])
            elif table[other]:
                mujoco.mj_contactForce(model, data, i, self._force)
                table_force += float(self._force[0])
                table_count += 1
        if touching:
            self._rb_steps += 1
            self._rb_sum += rb_force
            self._rb_max = max(self._rb_max, rb_force)
            if not self._rb_prev:
                self._rb_intervals += 1
        self._rb_prev = touching
        if self._steps > self._settle_steps:
            ratio = table_force / self._weight if self._weight > 0 else math.nan
            self._ratio_min = min(self._ratio_min, ratio)
            self._ratio_max = max(self._ratio_max, ratio)
            if not touching:
                self._free_sum += ratio
                self._free_steps += 1
        if table_count == 0:
            self._airborne_steps += 1
        self._max_table_contacts = max(self._max_table_contacts, table_count)
        lift = float(data.xpos[self._block_body][2]) - self._placement_z
        self._max_lift = max(self._max_lift, lift)
        self._tilt = block_tilt_deg(data.xmat[self._block_body])
        self._max_tilt = max(self._max_tilt, self._tilt)

    def summary(self) -> ContactSummaryMetrics:
        dt = self._timestep
        return ContactSummaryMetrics(
            physics_steps=self._steps,
            robot_block_contact_seconds=self._rb_steps * dt,
            robot_block_contact_intervals=self._rb_intervals,
            robot_block_max_normal_force=self._rb_max,
            robot_block_mean_normal_force=self._rb_sum / self._rb_steps if self._rb_steps else 0.0,
            table_normal_ratio_min=self._ratio_min if self._steps > self._settle_steps else math.nan,
            table_normal_ratio_max=self._ratio_max if self._steps > self._settle_steps else math.nan,
            table_normal_ratio_free_mean=self._free_sum / self._free_steps if self._free_steps else math.nan,
            block_airborne_seconds=self._airborne_steps * dt,
            max_table_contacts=self._max_table_contacts,
            max_block_lift=self._max_lift if self._steps else math.nan,
            max_block_tilt_deg=self._max_tilt,
            final_block_tilt_deg=self._tilt,
        )


class MonitorChain:
    """Calls several read-only monitors after every physics step, in order."""

    def __init__(self, *monitors) -> None:
        self.monitors = tuple(monitors)

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for monitor in self.monitors:
            monitor(model, data)
