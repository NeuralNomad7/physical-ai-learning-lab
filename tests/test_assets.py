"""Pinned MuJoCo Menagerie Panda: exact revision, hashes, licence and tamper detection."""

from __future__ import annotations

import json
import re
from pathlib import Path

from physical_ai_lab import assets, paths


def test_vendored_files_match_the_manifest() -> None:
    report = assets.verify()
    assert report.ok, report.summary()
    assert report.checked == len(assets.load_manifest()["files"])


def test_manifest_pins_an_exact_commit_and_licence() -> None:
    manifest = assets.load_manifest()
    source = manifest["source"]
    assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
    assert source["repository_slug"] == assets.REPOSITORY
    assert source["path"] == assets.SUBDIR
    assert source["license"] == "Apache-2.0"
    listed = {entry["path"] for entry in manifest["files"]}
    assert {"LICENSE", "README.md", "panda.xml"} <= listed
    assert "Apache License" in (paths.PANDA_DIR / "LICENSE").read_text(encoding="utf-8")
    for entry in manifest["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["path"]


def test_every_file_panda_xml_references_is_pinned() -> None:
    listed = {entry["path"] for entry in assets.load_manifest()["files"]}
    referenced = set(assets.referenced_files(paths.PANDA_XML.read_bytes()))
    assert referenced and referenced <= listed


def test_verify_detects_missing_and_modified_files(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_bytes(b"pinned")
    (tmp_path / "changed.txt").write_bytes(b"edited locally")
    manifest = {
        "source": {"commit": "0" * 40},
        "files": [
            {"path": "good.txt", "sha256": assets.sha256_hex(b"pinned")},
            {"path": "changed.txt", "sha256": assets.sha256_hex(b"original")},
            {"path": "gone.txt", "sha256": assets.sha256_hex(b"whatever")},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = assets.verify(manifest_path, tmp_path)
    assert not report.ok
    assert report.checked == 3
    assert report.missing == ["gone.txt"]
    assert report.mismatched == ["changed.txt"]
    assert "fetch-assets" in report.summary()
