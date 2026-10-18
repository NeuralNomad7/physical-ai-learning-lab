"""Phase 3 sealing: the final test and retention splits refuse to open without a frozen candidate.

The baseline is not a candidate, so no baseline outcome can exist on a sealed split. The freeze path
is exercised only on a throwaway manifest set in a temporary folder.
"""

from __future__ import annotations

import shutil

import pytest

from physical_ai_lab import manifest as mf
from physical_ai_lab import paths
from physical_ai_lab.benchmark import evaluate_split


def test_no_freeze_record_exists_in_the_project():
    assert not paths.FREEZES_DIR.exists() or not any(paths.FREEZES_DIR.glob("freeze_*.json"))


@pytest.mark.parametrize("split", ["test", "retention"])
def test_real_sealed_splits_refuse_to_open(split, bench_cfg):
    with pytest.raises(mf.SealedSplitError, match="sealed"):
        mf.open_split(split, bench=bench_cfg)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_real_open_splits_open(split, bench_cfg):
    assert not mf.open_split(split, bench=bench_cfg).sealed


@pytest.mark.parametrize("split", ["test", "retention"])
def test_evaluate_cli_refuses_sealed_split(split, cli):
    out = paths.RUNS_DIR / "phase3" / "evaluate" / split
    existed = out.exists()
    result = cli("evaluate", "--split", split, "--mode", "baseline")
    assert result.returncode == 2
    assert "REFUSED" in result.stderr and "sealed" in result.stderr
    assert out.exists() == existed  # nothing was written


def test_manifests_cli_refuses_to_regenerate_existing_manifests(cli):
    hashes = paths.MANIFESTS_DIR / mf.HASHES_FILE
    before = {p.name: p.read_bytes() for p in paths.MANIFESTS_DIR.iterdir()}
    result = cli("manifests")
    assert result.returncode == 2
    assert "REFUSED" in result.stderr and "Traceback" not in result.stderr
    assert hashes.exists()
    assert {p.name: p.read_bytes() for p in paths.MANIFESTS_DIR.iterdir()} == before


def test_evaluate_split_refuses_before_writing(scratch_manifests, cfg, push_cfg, tmp_path):
    bench, manifest_dir, freeze_dir = scratch_manifests
    out = tmp_path / "run"
    with pytest.raises(mf.SealedSplitError):
        evaluate_split("test", mode="baseline", out_dir=out, scene=cfg, push=push_cfg, bench=bench,
                       manifest_dir=manifest_dir, freeze_dir=freeze_dir, progress=lambda line: None)
    assert not out.exists()


@pytest.mark.parametrize("name", ["baseline", "Baseline", "", "  "])
def test_baseline_cannot_be_frozen(name, scratch_manifests, tmp_path):
    _, manifest_dir, _ = scratch_manifests
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"weights")
    with pytest.raises(ValueError, match="baseline"):
        mf.freeze_candidate(candidate, candidate_name=name, manifest_dir=manifest_dir, freeze_dir=tmp_path / "freezes")
    assert not (tmp_path / "freezes").exists()


def test_freeze_opens_sealed_splits_and_is_bound_to_the_candidate(scratch_manifests, tmp_path):
    _, manifest_dir, _ = scratch_manifests
    freezes = tmp_path / "freezes"
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"frozen learned state v1")
    with pytest.raises(mf.SealedSplitError):
        mf.open_split("test", manifest_dir=manifest_dir, freeze_dir=freezes)

    record_path = mf.freeze_candidate(candidate, candidate_name="learner-v1", manifest_dir=manifest_dir,
                                      freeze_dir=freezes)
    assert mf.open_split("test", manifest_dir=manifest_dir, freeze_dir=freezes).sealed
    assert mf.open_split("retention", manifest_dir=manifest_dir, freeze_dir=freezes).sealed

    # A freeze record is never overwritten.
    before = record_path.read_bytes()
    with pytest.raises(FileExistsError):
        mf.freeze_candidate(candidate, candidate_name="learner-v1", manifest_dir=manifest_dir, freeze_dir=freezes)
    assert record_path.read_bytes() == before

    # Changing the candidate after the freeze closes the sealed splits again.
    candidate.write_bytes(b"frozen learned state v2 (changed after the freeze)")
    with pytest.raises(mf.SealedSplitError):
        mf.open_split("test", manifest_dir=manifest_dir, freeze_dir=freezes)


def test_freeze_is_bound_to_the_sealed_manifest_hash(scratch_manifests, tmp_path):
    _, manifest_dir, _ = scratch_manifests
    copy, freezes = tmp_path / "manifests", tmp_path / "freezes"
    shutil.copytree(manifest_dir, copy)
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"learned")
    mf.freeze_candidate(candidate, candidate_name="learner", manifest_dir=copy, freeze_dir=freezes)
    # Swap the test manifest for a different one (and patch the hash file to match): the freeze no
    # longer covers it, so the split stays sealed.
    test_path = copy / "test.json"
    data = test_path.read_bytes().replace(b'"sealed": true', b'"sealed": true ', 1)
    test_path.write_bytes(data)
    hashes = mf.load_hashes(copy)
    hashes["manifests"]["test"]["sha256"] = mf.sha256_bytes(data)
    (copy / mf.HASHES_FILE).write_bytes(mf.canonical_bytes(hashes))
    with pytest.raises(mf.SealedSplitError):
        mf.open_split("test", manifest_dir=copy, freeze_dir=freezes)


def test_freeze_refuses_a_tampered_sealed_manifest(scratch_manifests, tmp_path):
    _, manifest_dir, _ = scratch_manifests
    copy = tmp_path / "manifests"
    shutil.copytree(manifest_dir, copy)
    (copy / "retention.json").write_bytes((copy / "retention.json").read_bytes() + b" ")
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"learned")
    with pytest.raises(mf.ManifestError):
        mf.freeze_candidate(candidate, candidate_name="learner", manifest_dir=copy, freeze_dir=tmp_path / "freezes")
    assert not (tmp_path / "freezes").exists()


def test_replay_of_a_sealed_split_run_is_refused(scratch_manifests, cfg, push_cfg, tmp_path):
    from physical_ai_lab.benchmark import replay_run

    bench, manifest_dir, freeze_dir = scratch_manifests
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text('{"split": "test", "mode": "baseline", "manifest_sha256": "x"}', encoding="utf-8")
    (run / "episodes.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(mf.SealedSplitError):
        replay_run(run, scene=cfg, push=push_cfg, bench=bench, manifest_dir=manifest_dir, freeze_dir=freeze_dir)
