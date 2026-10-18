"""Phase 3 statistics, instrumentation and deterministic replay, on scratch runs only."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from physical_ai_lab import benchmark as bm
from physical_ai_lab.manifest import open_split
from physical_ai_lab.scenario import ScenarioRunner

# --------------------------------------------------------------------------- statistics


@pytest.mark.parametrize("k, n, expected", [
    (10, 10, (0.7225, 1.0)),
    (0, 10, (0.0, 0.2775)),
    (5, 10, (0.2366, 0.7634)),
    (95, 100, (0.8882, 0.9785)),
])
def test_wilson_matches_reference_values(k, n, expected):
    lo, hi = bm.wilson(k, n, 0.95)
    assert lo == pytest.approx(expected[0], abs=5e-4) and hi == pytest.approx(expected[1], abs=5e-4)


def test_wilson_empty():
    assert all(math.isnan(v) for v in bm.wilson(0, 0, 0.95))


def test_cluster_bootstrap_is_deterministic_and_clustered():
    clusters = [(5, 5)] * 15 + [(0, 5)] * 5  # 75 % success, failures concentrated in 5 conditions
    a = bm.cluster_bootstrap(clusters, resamples=2000, seed=7, confidence=0.95)
    assert a == bm.cluster_bootstrap(clusters, resamples=2000, seed=7, confidence=0.95)
    assert a[0] < 0.75 < a[1]
    # Treating the 100 episodes as independent understates the uncertainty.
    unclustered = bm.wilson(75, 100, 0.95)
    assert (a[1] - a[0]) > (unclustered[1] - unclustered[0])
    assert bm.cluster_bootstrap([(5, 5)] * 20, resamples=500, seed=1, confidence=0.95) == (1.0, 1.0)


def test_ceiling_notes():
    assert bm.ceiling_note({"episodes": 100, "successes": 100}).startswith("CEILING")
    assert bm.ceiling_note({"episodes": 100, "successes": 96}).startswith("NEAR CEILING")
    assert bm.ceiling_note({"episodes": 100, "successes": 80}) == ""


def test_summary_reports_all_episodes_and_successes(scratch_run, bench_cfg):
    records = scratch_run["records"]
    g = bm.summarize_group(records, bench_cfg)
    k = sum(r["evaluator"]["outcome"]["success"] for r in records)
    assert (g["episodes"], g["successes"], g["failures"]) == (2, k, 2 - k)
    assert g["pushes_used"]["all"]["n"] == 2 and g["pushes_used"]["successes"]["n"] == k
    assert g["sim_seconds"]["all"]["n"] == 2 and g["sim_seconds"]["successes"]["n"] == k
    assert g["final_distance_mm"]["all"]["n"] == 2
    distances = [r["evaluator"]["final_distance_m"] * 1e3 for r in records]
    assert g["final_distance_mm"]["all"]["mean"] == pytest.approx(np.mean(distances))
    assert set(g["failures_by_reason"]) >= {"timeout", "unstable", "limit_violation"}
    assert g["traffic_scan"]["hits"] == 0 and g["traffic_scan"]["messages"] > 0
    assert g["success_ci"]["cluster_bootstrap"] is not None


def test_scratch_summary_files(scratch_run):
    run = scratch_run["dir"]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["split"] == "validation" and summary["mode"] == "baseline"
    assert summary["episodes_run"] == 2 and summary["partial"] is True
    assert summary["benchmark"]["config_sha256"] == scratch_run["bench"].sha256
    assert "all" in summary["groups"]
    assert (run / "summary.txt").read_text(encoding="utf-8").startswith("Phase 3 baseline evaluation")
    assert sorted(p.name for p in (run / "commands").iterdir()) == ["ep0001.npz", "ep0002.npz"]


# --------------------------------------------------------------------------- instrumentation


def test_contact_metrics_are_recorded_and_plausible(scratch_run, cfg):
    for rec in scratch_run["records"]:
        ev = rec["evaluator"]
        c = ev["contact"]
        assert c["physics_steps"] == ev["limits"]["physics_steps"] > 0
        # Without robot contact the table carries the applied (hidden) weight: the mass reaches the contact.
        assert c["table_normal_ratio_free_mean"] == pytest.approx(1.0, abs=0.02)
        assert c["table_normal_ratio_min"] <= c["table_normal_ratio_free_mean"] <= c["table_normal_ratio_max"]
        assert 0 <= c["max_block_lift"] < 0.01 and 0 <= c["max_block_tilt_deg"] < 45
        assert c["final_block_tilt_deg"] <= c["max_block_tilt_deg"] + 1e-9
        if ev["pushes_used"]:
            assert c["robot_block_contact_seconds"] > 0 and c["robot_block_contact_intervals"] >= 1
            assert c["robot_block_max_normal_force"] >= c["robot_block_mean_normal_force"] > 0


def test_applied_physics_match_the_manifest(scratch_run, cfg):
    for rec in scratch_run["records"]:
        ev = rec["evaluator"]
        req, app = ev["physics"], ev["applied_physics"]
        assert app["block_mass"] == req["block_mass"] and app["sliding_friction"] == req["sliding_friction"]
        expected = req["block_mass"] * cfg.block.edge ** 2 / 6.0
        assert app["body_inertia"] == pytest.approx([expected] * 3, rel=1e-12)


# --------------------------------------------------------------------------- replay


def test_command_replay_is_bitwise_identical(scratch_run, cfg, bench_cfg):
    manifest = open_split("validation", manifest_dir=scratch_run["manifest_dir"], freeze_dir=scratch_run["freeze_dir"],
                          bench=scratch_run["bench"])
    by_id = {s.scenario_id: s for s in manifest.scenarios}
    runner = ScenarioRunner(cfg)
    for rec in scratch_run["records"]:
        ev = rec["evaluator"]
        logged = np.load(scratch_run["dir"] / "commands" / ev["replay"]["commands_file"])
        assert len(logged["ctrl"]) == ev["replay"]["control_updates"] == rec["learner"]["commands_sent"]
        result = bm.replay_commands(runner, manifest, by_id[ev["scenario_id"]], logged["ctrl"], logged["digests"],
                                    bench_cfg.replay.state_abs_tolerance)
        assert result["first_mismatch_update"] is None
        assert result["initial_state_digest"] == ev["replay"]["initial_state_digest"]
        assert result["final_state_digest"] == ev["replay"]["final_state_digest"]


def test_replay_detects_a_changed_command(scratch_run, cfg):
    manifest = open_split("validation", manifest_dir=scratch_run["manifest_dir"], freeze_dir=scratch_run["freeze_dir"],
                          bench=scratch_run["bench"])
    ev = scratch_run["records"][0]["evaluator"]
    scenario = next(s for s in manifest.scenarios if s.scenario_id == ev["scenario_id"])
    logged = np.load(scratch_run["dir"] / "commands" / ev["replay"]["commands_file"])
    ctrl = np.array(logged["ctrl"])
    ctrl[10, 0] += 1e-9
    result = bm.replay_commands(ScenarioRunner(cfg), manifest, scenario, ctrl, logged["digests"], 0.0)
    assert result["first_mismatch_update"] == 11
    assert result["final_state_digest"] != ev["replay"]["final_state_digest"]


def test_replay_run_reruns_through_the_agent(scratch_run, cfg, push_cfg):
    report = bm.replay_run(scratch_run["dir"], scene=cfg, push=push_cfg, bench=scratch_run["bench"],
                           manifest_dir=scratch_run["manifest_dir"], freeze_dir=scratch_run["freeze_dir"], count=1,
                           progress=lambda line: None)
    assert report["tolerance"] == 0.0 and len(report["episodes"]) == 1
    ep = report["episodes"][0]
    assert ep["command_replay"]["matches"], ep
    assert ep["rerun"]["matches"], ep
    assert report["all_match"]
