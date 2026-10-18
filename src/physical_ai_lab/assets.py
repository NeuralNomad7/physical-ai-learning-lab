"""Pinned MuJoCo Menagerie Franka Emika Panda asset: pin, fetch and verify.

The manifest records the Menagerie commit, the Apache-2.0 license file and the
SHA-256 of every vendored file. ``fetch`` restores missing or modified files
from that exact commit; ``verify`` checks the local copy without network access.
Only the files that ``panda.xml`` references (plus license/docs) are vendored.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from . import paths

REPOSITORY = "google-deepmind/mujoco_menagerie"
SUBDIR = "franka_emika_panda"
LICENSE_SPDX = "Apache-2.0"
DOC_FILES = ("LICENSE", "README.md", "CHANGELOG.md", "panda.xml", "scene.xml")

_RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
_API = "https://api.github.com/repos/{repo}"
_USER_AGENT = "physical-ai-lab-asset-fetch"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """SHA-1 that git (and the GitHub tree API) assigns to a file's content."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _http_get(url: str, retries: int = 3, timeout: float = 60.0) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # network errors are retried, then reported
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_error}")


def referenced_files(panda_xml: bytes) -> list[str]:
    """Mesh/texture files referenced by panda.xml, relative to the Panda folder."""
    root = ET.fromstring(panda_xml)
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    files: set[str] = set()
    for element in root.iter():
        file_attr = element.get("file")
        if not file_attr:
            continue
        if element.tag in ("mesh", "hfield", "skin"):
            files.add(str(PurePosixPath(meshdir) / file_attr) if meshdir else file_attr)
        elif element.tag == "texture":
            files.add(str(PurePosixPath(texturedir) / file_attr) if texturedir else file_attr)
        elif element.tag == "include":
            raise ValueError("panda.xml unexpectedly uses <include>; update the asset pinning logic")
    return sorted(files)


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_manifest(manifest_path: Path = paths.PANDA_MANIFEST) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@dataclass
class VerifyReport:
    commit: str
    checked: int = 0
    missing: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.checked > 0 and not self.missing and not self.mismatched

    def summary(self) -> str:
        if self.ok:
            return f"OK: {self.checked} files match manifest (Menagerie commit {self.commit[:12]})"
        return (
            f"FAILED: {len(self.missing)} missing, {len(self.mismatched)} mismatched of {self.checked} "
            f"(Menagerie commit {self.commit[:12]}); run `{paths.python_command()} -m physical_ai_lab fetch-assets`"
        )


def verify(manifest_path: Path = paths.PANDA_MANIFEST, asset_dir: Path = paths.PANDA_DIR) -> VerifyReport:
    manifest = load_manifest(manifest_path)
    report = VerifyReport(commit=manifest["source"]["commit"])
    for entry in manifest["files"]:
        report.checked += 1
        local = asset_dir / entry["path"]
        if not local.is_file():
            report.missing.append(entry["path"])
        elif sha256_hex(local.read_bytes()) != entry["sha256"]:
            report.mismatched.append(entry["path"])
    return report


def fetch(
    manifest_path: Path = paths.PANDA_MANIFEST,
    asset_dir: Path = paths.PANDA_DIR,
    force: bool = False,
) -> VerifyReport:
    """Download missing/modified files from the pinned commit and verify SHA-256."""
    manifest = load_manifest(manifest_path)
    source = manifest["source"]
    before = verify(manifest_path, asset_dir)
    wanted = {e["path"] for e in manifest["files"]} if force else set(before.missing + before.mismatched)
    entries = [e for e in manifest["files"] if e["path"] in wanted]

    def download(entry: dict) -> None:
        url = _RAW_URL.format(
            repo=source["repository_slug"], commit=source["commit"], path=f"{source['path']}/{entry['path']}"
        )
        data = _http_get(url)
        digest = sha256_hex(data)
        if digest != entry["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {entry['path']}: got {digest}, expected {entry['sha256']}")
        _write_atomic(asset_dir / entry["path"], data)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, entries))
    return verify(manifest_path, asset_dir)


def pin(
    commit: str,
    manifest_path: Path = paths.PANDA_MANIFEST,
    asset_dir: Path = paths.PANDA_DIR,
) -> dict:
    """Maintainer operation: vendor the Panda folder at ``commit`` and write the manifest.

    Uses the GitHub API to resolve the commit and git blob hashes, downloads only
    the files panda.xml needs (plus license/docs) and checks each against git.
    """
    api = _API.format(repo=REPOSITORY)
    commit_info = json.loads(_http_get(f"{api}/commits/{commit}"))
    full_sha = commit_info["sha"]
    path_history = json.loads(_http_get(f"{api}/commits?path={SUBDIR}&sha={full_sha}&per_page=1"))
    tree = json.loads(_http_get(f"{api}/git/trees/{full_sha}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree listing was truncated; cannot pin reliably")
    blobs = {
        item["path"][len(SUBDIR) + 1 :]: item
        for item in tree["tree"]
        if item["type"] == "blob" and item["path"].startswith(SUBDIR + "/")
    }

    def get(rel: str) -> bytes:
        data = _http_get(_RAW_URL.format(repo=REPOSITORY, commit=full_sha, path=f"{SUBDIR}/{rel}"))
        if git_blob_sha1(data) != blobs[rel]["sha"]:
            raise RuntimeError(f"git blob hash mismatch for {rel}")
        return data

    panda_xml = get("panda.xml")
    wanted = sorted(set(DOC_FILES) | set(referenced_files(panda_xml)))
    missing_upstream = [rel for rel in wanted if rel not in blobs]
    if missing_upstream:
        raise RuntimeError(f"files referenced by panda.xml are missing upstream: {missing_upstream}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        contents = dict(zip(wanted, pool.map(get, wanted)))

    for rel, data in contents.items():
        _write_atomic(asset_dir / rel, data)

    manifest = {
        "schema_version": 1,
        "description": "Pinned subset of MuJoCo Menagerie franka_emika_panda used by physical_ai_lab",
        "source": {
            "repository": f"https://github.com/{REPOSITORY}",
            "repository_slug": REPOSITORY,
            "commit": full_sha,
            "commit_date": commit_info["commit"]["committer"]["date"],
            "path": SUBDIR,
            "last_commit_touching_path": path_history[0]["sha"] if path_history else None,
            "last_commit_touching_path_date": (
                path_history[0]["commit"]["committer"]["date"] if path_history else None
            ),
            "license": LICENSE_SPDX,
            "license_file": "LICENSE",
            "pinned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "files": [
            {
                "path": rel,
                "size": len(data),
                "git_blob_sha1": git_blob_sha1(data),
                "sha256": sha256_hex(data),
            }
            for rel, data in sorted(contents.items())
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
