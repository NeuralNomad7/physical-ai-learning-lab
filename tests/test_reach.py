"""Kinematic feasibility: the TCP can point straight down over the block/target area plus a margin."""

from __future__ import annotations

import numpy as np

from physical_ai_lab.config import SceneConfig
from physical_ai_lab.diagnostics import check_reach
from physical_ai_lab.scenario import ScenarioRunner


def test_tcp_reaches_the_work_area(runner: ScenarioRunner, cfg: SceneConfig) -> None:
    runner.reset(0)
    before = runner.state_vector().copy()
    report = check_reach(runner)
    rc, w = cfg.reach_check, cfg.workspace
    assert len(report.points) == rc.grid * rc.grid * len(rc.heights)
    assert report.x_range[0] <= min(w.block_x.low, w.target_x.low) - rc.margin + 1e-12
    assert report.x_range[1] >= max(w.block_x.high, w.target_x.high) + rc.margin - 1e-12
    assert report.y_range[0] <= min(w.block_y.low, w.target_y.low) - rc.margin + 1e-12
    assert report.y_range[1] >= max(w.block_y.high, w.target_y.high) + rc.margin - 1e-12
    assert report.ok, report.summary() + "".join(f"\n  {point}" for point in report.failures)
    # The check works on a private MjData: the runner's episode is untouched.
    np.testing.assert_array_equal(runner.state_vector(), before)
