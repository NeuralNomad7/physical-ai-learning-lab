"""Fixed-feedback push baseline (Phase 2; BUILD_PLAN.md section 6).

Before every push the policy waits for the block to come to rest, measures the block
centre, block yaw and target from the ``Observation`` and chooses one push primitive:

* candidate faces: the faces whose inward normal is within 90 degrees of the
  block-to-target direction (pushing any other face moves the block away);
* direction: the block-to-target direction, clipped to within ``max_off_normal`` of the
  face normal;
* speed and duration: every admissible pair (3 x 3).

Every candidate is filtered by ``PushSkill.plan`` (reachability, table bounds, block
region). Among the feasible ones the policy takes the one whose geometric prediction
of the block end is closest to the target (ties broken toward slower pushes). It
stops pushing once the block is within ``stop_tolerance`` of the target and simply
holds the arm still, or when the push budget is used up.

No learning and no state carried between episodes: the same observation always gives
the same choice. Hidden mass and friction are never read; errors in the geometric
prediction are corrected by the next measurement (that is the feedback).
"""

from __future__ import annotations

import math

import numpy as np

from .config import SceneConfig
from .controller import ExecutionStats, Plant, PushAborted, PushAction, PushPlan, PushSkill
from .kinematics import wrap_angle
from .observation import Observation
from .push_config import PushConfig

STOP_HOLD_SECONDS = 1.0e6  # hold "forever"; the evaluator ends the episode


class FeedbackBaseline:
    name = "baseline"

    def __init__(self, scene: SceneConfig, push: PushConfig, skill: PushSkill | None = None) -> None:
        self.scene, self.push = scene, push
        self.skill = skill or PushSkill(scene, push)

    def candidates(self, obs: Observation) -> list[PushAction]:
        delta = np.asarray(obs.target_pos[:2], dtype=float) - np.asarray(obs.block_pos[:2], dtype=float)
        wanted = math.atan2(delta[1], delta[0])
        limit = self.push.controller.max_off_normal
        actions = []
        for face in range(4):
            normal = self.skill.inward_normal(obs.block_yaw, face)
            off = wrap_angle(wanted - normal)
            if abs(off) >= math.pi / 2:
                continue
            direction = wrap_angle(normal + max(-limit, min(limit, off)))
            for speed in self.push.action.speeds:
                for duration in self.push.action.durations:
                    actions.append(PushAction(face, direction, speed, duration))
        return actions

    def cost(self, obs: Observation, plan: PushPlan) -> float:
        miss = float(np.linalg.norm(plan.predicted_block_end - np.asarray(obs.target_pos[:2], dtype=float)))
        return miss + self.push.baseline.speed_preference * plan.action.speed

    def choose(self, obs: Observation) -> tuple[PushPlan | None, list[dict]]:
        """Best feasible plan and a record of every candidate considered."""
        best: tuple[float, PushPlan] | None = None
        report = []
        for action in self.candidates(obs):
            plan = self.skill.plan(obs, action)
            cost = self.cost(obs, plan) if plan.feasible else None
            report.append(
                {
                    "face": action.face,
                    "direction_deg": round(math.degrees(action.direction), 2),
                    "speed": action.speed,
                    "duration": action.duration,
                    "feasible": plan.feasible,
                    "reason": plan.reason,
                    "predicted_miss": None if cost is None else round(cost, 5),
                }
            )
            if cost is not None and (best is None or cost < best[0] - 1e-12):
                best = (cost, plan)
        return (None if best is None else best[1]), report

    def wait_for_rest(self, plant: Plant) -> Observation:
        ctl = self.push.controller
        plant.note("phase", phase="settle")
        return self.skill.hold(plant, ctl.settle_max_seconds, until=lambda o: o.block_planar_speed < ctl.settle_speed)

    def run(self, plant: Plant) -> str:
        """Push until done. Returns why it stopped by itself; the evaluator may end it earlier."""
        self.skill.q_cmd = None
        self.skill.stats = ExecutionStats()
        pushes = 0
        while True:
            obs = self.wait_for_rest(plant)
            if obs.block_target_distance <= self.push.baseline.stop_tolerance:
                plant.note("decision", decision="stop", distance=obs.block_target_distance)
                plant.note("phase", phase="hold")
                self.skill.hold(plant, STOP_HOLD_SECONDS)
                return "holding"
            if pushes >= self.push.episode.max_pushes:
                plant.note("decision", decision="budget_exhausted", distance=obs.block_target_distance)
                return "push_budget_exhausted"
            plan, report = self.choose(obs)
            if plan is None:
                plant.note("decision", decision="no_feasible_push", candidates=report)
                return "no_feasible_push"
            pushes += 1
            plant.note("push_start", index=pushes, plan=plan.summary(), candidates=report)
            try:
                self.skill.execute(plant, obs, plan)
                plant.note("push_end", index=pushes, aborted=None)
            except PushAborted as exc:
                plant.note("push_end", index=pushes, aborted=str(exc))
                self.skill.recover(plant)
