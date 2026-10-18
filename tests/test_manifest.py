"""Phase 3 manifests: sizes, cohorts, disjointness, determinism, hashes and opaque identifiers.

Only structural reads of the real manifests (``read_sealed_for_audit``): no episode runs on them here.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from physical_ai_lab import manifest as mf
from physical_ai_lab import paths

EXPECTED = {  # split: (conditions, scenarios, in_range scenarios, out_of_range scenarios, sealed)
    "train": (40, 200, 200, 0, False),
    "validation": (20, 100, 60, 40, False),
    "test": (20, 100, 70, 30, True),
    "retention": (10, 50, 50, 0, True),
}
SCENARIO_KEYS = {"scenario_id", "condition_id", "reset_seed", "goal_sector", "layout"}
SEEDS_USED_BEFORE_PHASE3 = set(range(0, 200)) | set(range(20000, 20020)) | set(range(21000, 21030)) | {30000, 30001, 30002}


@pytest.fixture(scope="module")
def real():
    return {split: mf.read_sealed_for_audit(split) for split in mf.SPLITS}


def test_hashes_file_matches_manifest_bytes(real):
    hashes = mf.load_hashes()
    for split, m in real.items():
        entry = hashes["manifests"][split]
        assert entry["sha256"] == mf.sha256_bytes((paths.MANIFESTS_DIR / entry["file"]).read_bytes()) == m.sha256
        assert entry["sealed"] is EXPECTED[split][4]


def test_sizes_cohorts_and_sealing(real, bench_cfg):
    for split, (n_cond, n_scen, n_in, n_out, sealed) in EXPECTED.items():
        m = real[split]
        assert (len(m.conditions), len(m.scenarios), m.sealed) == (n_cond, n_scen, sealed), split
        by = {c: sum(1 for s in m.scenarios if m.condition(s).cohort == c) for c in mf.COHORTS}
        assert by == {"in_range": n_in, "out_of_range": n_out}, split
        assert m.data["counts"]["by_cohort"] == by
        assert m.data["benchmark"]["config_sha256"] == bench_cfg.sha256


def test_cohort_labels_match_physics(real, bench_cfg):
    for m in real.values():
        for c in m.conditions.values():
            inside = bench_cfg.in_range.contains(c.physics.block_mass, c.physics.sliding_friction)
            assert inside == (c.cohort == "in_range")
            if c.cohort == "out_of_range":
                assert bench_cfg.out_of_range.box.contains(c.physics.block_mass, c.physics.sliding_friction)
                assert c.band in ("mass_low", "mass_high", "friction_low", "friction_high")


def test_development_range_is_the_scene_range(bench_cfg, cfg):
    phys = cfg.physics
    assert bench_cfg.in_range.mass == (phys.mass_range.low, phys.mass_range.high)
    assert bench_cfg.in_range.sliding_friction == (phys.sliding_friction_range.low, phys.sliding_friction_range.high)


def test_out_of_range_cohort_was_fixed_before_the_manifests(real, bench_cfg):
    assert bench_cfg.out_of_range.fixed and bench_cfg.out_of_range.fixed_utc
    generated = mf.load_hashes()["generated_utc"]
    assert bench_cfg.out_of_range.fixed_utc < generated
    for m in real.values():
        assert m.data["benchmark"]["out_of_range_fixed_utc"] == bench_cfg.out_of_range.fixed_utc


def test_splits_are_disjoint_in_physics_and_initial_states(real, bench_cfg):
    report = mf.check_disjoint(real, bench_cfg)
    assert report["ok"]
    assert report["layout_pairs_checked"] == 450 * 449 // 2
    primary = ("train", "validation", "test")
    physics = {s: {(c.physics.block_mass, c.physics.sliding_friction) for c in real[s].conditions.values()}
               for s in primary}
    assert not (physics["train"] & physics["validation"] or physics["train"] & physics["test"]
                or physics["validation"] & physics["test"])
    layouts = [s.layout for m in real.values() for s in m.scenarios]
    assert len({(l.block_xy, l.block_yaw, l.target_xy) for l in layouts}) == len(layouts)


def test_retention_reuses_train_conditions_with_held_out_initial_states(real):
    train, retention = real["train"], real["retention"]
    assert set(retention.conditions) <= set(train.conditions)
    for cid, c in retention.conditions.items():
        assert train.conditions[cid] == c
    train_layouts = {(s.layout.block_xy, s.layout.target_xy) for s in train.scenarios}
    assert not any((s.layout.block_xy, s.layout.target_xy) in train_layouts for s in retention.scenarios)


def test_test_split_covers_conditions_and_goal_directions(real, bench_cfg):
    test = real["test"]
    sectors = bench_cfg.manifests.goal_direction_sectors
    assert {s.goal_sector for s in test.scenarios} == set(range(sectors))
    bands = {test.condition(s).band for s in test.scenarios if test.condition(s).cohort == "out_of_range"}
    assert bands == {"mass_low", "mass_high", "friction_low", "friction_high"}
    for s in test.scenarios:
        assert mf.goal_sector(s.layout, sectors) == s.goal_sector


def test_reset_seeds_are_new(real):
    seeds = [s.reset_seed for m in real.values() for s in m.scenarios]
    assert len(set(seeds)) == len(seeds)
    assert not set(seeds) & SEEDS_USED_BEFORE_PHASE3


def test_identifiers_and_order_do_not_encode_the_condition(real):
    for split, m in real.items():
        for s in m.scenarios:
            assert re.fullmatch(r"s[0-9a-f]{12}", s.scenario_id)
            assert re.fullmatch(r"c[0-9a-f]{12}", s.condition_id)
        for entry in m.data["scenarios"]:
            assert set(entry) == SCENARIO_KEYS  # no mass, friction, cohort or band next to a scenario
        # Scenarios are shuffled: the condition sequence is not grouped by condition or cohort.
        conds = [s.condition_id for s in m.scenarios]
        runs = 1 + sum(1 for a, b in zip(conds, conds[1:]) if a != b)
        assert runs > len(conds) // 2, split
        # Hex ids carry no ordering: sorting by id does not sort by mass or friction.
        ordered = sorted(m.conditions.values(), key=lambda c: c.condition_id)
        masses = [c.physics.block_mass for c in ordered]
        assert masses != sorted(masses) and masses != sorted(masses, reverse=True)
    names = sorted(p.name for p in paths.MANIFESTS_DIR.iterdir())
    assert names == ["hashes.json", "retention.json", "test.json", "train.json", "validation.json"]


def test_generation_is_deterministic(real, cfg, bench_cfg):
    docs = mf.generate_manifests(cfg, bench_cfg, real["train"].data["provenance"])
    for split, m in real.items():
        assert mf.canonical_bytes(docs[split]) == m.path.read_bytes(), split


def test_provenance_records_versions_and_simulation(real):
    prov = real["test"].data["provenance"]
    assert prov["mujoco_version"] == "3.14.0"
    assert prov["panda_asset"]["commit"].startswith("4d038b3feae2") and prov["panda_asset"]["files"] == 72
    assert {"scene.toml", "push.toml", "benchmark.toml", "panda_push_scene.xml"} == set(prov["config_sha256"])
    sim = prov["simulation"]
    assert sim["cone"] == 1 and sim["substeps"] > 0 and sim["timestep"] > 0  # elliptic cones (D6)


def test_tampered_manifest_is_rejected(scratch_manifests, tmp_path):
    bench, manifest_dir, _ = scratch_manifests
    copy = tmp_path / "manifests"
    shutil.copytree(manifest_dir, copy)
    mf.open_split("validation", manifest_dir=copy, freeze_dir=tmp_path / "freezes", bench=bench)
    path = copy / "validation.json"
    doc = json.loads(path.read_bytes())
    doc["conditions"][0]["block_mass"] += 0.01
    path.write_bytes(mf.canonical_bytes(doc))
    with pytest.raises(mf.ManifestError, match="SHA-256"):
        mf.open_split("validation", manifest_dir=copy, freeze_dir=tmp_path / "freezes", bench=bench)


def test_manifests_are_never_overwritten(scratch_manifests, cfg):
    bench, manifest_dir, _ = scratch_manifests
    before = {p.name: p.read_bytes() for p in manifest_dir.iterdir()}
    with pytest.raises(mf.ManifestError, match="already written"):
        mf.write_manifests(mf.generate_manifests(cfg, bench, {"note": "x"}), manifest_dir)
    assert {p.name: p.read_bytes() for p in manifest_dir.iterdir()} == before


def test_split_opened_with_another_benchmark_config_is_rejected(bench_cfg):
    import dataclasses

    mf.open_split("train", bench=bench_cfg)
    with pytest.raises(mf.ManifestError, match="benchmark config"):
        mf.open_split("train", bench=dataclasses.replace(bench_cfg, sha256="0" * 64))


def test_manifests_verify_cli(cli):
    result = cli("manifests", "--verify")
    assert result.returncode == 0, result.stderr
    for sha in ("6eadee60c6ae2a2f", "88dabaa6cb811ea4", "5673ce2e3763c041", "e8c5857a7d7151a1"):
        assert sha in result.stdout
