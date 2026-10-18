"""Benchmark manifests and sealing (Phase 3; BUILD_PLAN.md section 6).

A manifest lists the scenarios of one split: each scenario has an opaque random ``scenario_id``,
an opaque ``condition_id`` (its physical condition: block mass and sliding friction), a cohort,
a reset seed and an explicit initial layout (block pose, target). Manifests are evaluator-side
files: the agent never sees them, their ids, their order or their file names; it only receives
observations through the learner interface (learner_interface.py).

Generation is deterministic from ``configs/benchmark.toml`` [manifests].generator_seed:

* physical conditions per split: in-range conditions by Latin-hypercube sampling of the in-range
  box; out-of-range conditions cycle through the four outer bands (one parameter in the band, the
  other uniform over the outer box); every condition is at least ``min_condition_separation`` from
  every other condition of every split (parameters normalised by the in-range widths);
* layouts: goal directions stratified over ``goal_direction_sectors`` sectors (a condition's five
  layouts use five different sectors), block position, yaw and block/target distance uniform in the
  scene's reset regions; every layout differs from every other layout of every split;
* retention reuses ``in_range_conditions`` train conditions (same condition ids and physics, by
  design: earlier training regimes) with new layouts that differ from every train layout;
* scenario order within a manifest is shuffled, so position does not encode the condition.

Train, validation and test share no physical condition and no initial state. Retention shares
conditions only with train and no initial state with any split.

Sealing: test and retention are sealed. ``open_split`` refuses a sealed split unless a valid
candidate-freeze record (``freeze_candidate``, Phase 5) commits to the current manifest hash.
This applies to every controller, including the fixed baseline.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import paths
from .benchmark_config import COHORTS, SPLITS, BenchmarkConfig
from .config import SceneConfig
from .scenario import HiddenPhysics, ScenarioLayout

GENERATION_ORDER = ("train", "validation", "test", "retention")
HASHES_FILE = "hashes.json"
FREEZE_KIND = "candidate_freeze"


class SealedSplitError(PermissionError):
    """Raised when a sealed split (final test, retention) is opened without a frozen candidate."""


class ManifestError(ValueError):
    pass


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Condition:
    condition_id: str
    cohort: str
    band: str  # "in_range" or the out-of-range band that made it out of range
    physics: HiddenPhysics


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    condition_id: str
    reset_seed: int
    goal_sector: int
    layout: ScenarioLayout


@dataclass(frozen=True)
class Manifest:
    split: str
    sealed: bool
    sha256: str  # of the manifest file bytes
    path: Path
    data: dict[str, Any]
    conditions: dict[str, Condition]
    scenarios: tuple[Scenario, ...]

    def condition(self, scenario: Scenario) -> Condition:
        return self.conditions[scenario.condition_id]


# --------------------------------------------------------------------------- canonical files


def canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False, ensure_ascii=True) + "\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


# --------------------------------------------------------------------------- generation


def _norm(bench: BenchmarkConfig, physics: HiddenPhysics) -> np.ndarray:
    (m0, m1), (f0, f1) = bench.in_range.mass, bench.in_range.sliding_friction
    return np.array([physics.block_mass / (m1 - m0), physics.sliding_friction / (f1 - f0)])


def _hex_id(rng: np.random.Generator, used: set[str], prefix: str) -> str:
    while True:
        value = prefix + "".join(f"{int(b):02x}" for b in rng.integers(0, 256, size=6))
        if value not in used:
            used.add(value)
            return value


def _round_physics(mass: float, friction: float) -> HiddenPhysics:
    return HiddenPhysics(round(float(mass), 4), round(float(friction), 4))


def _in_range_conditions(rng, bench: BenchmarkConfig, n: int) -> list[tuple[str, HiddenPhysics]]:
    if n == 0:
        return []
    box = bench.in_range
    u_m = (rng.permutation(n) + rng.uniform(size=n)) / n
    u_f = (rng.permutation(n) + rng.uniform(size=n)) / n
    return [("in_range", _round_physics(box.mass[0] + a * (box.mass[1] - box.mass[0]),
                                        box.sliding_friction[0] + b * (box.sliding_friction[1] - box.sliding_friction[0])))
            for a, b in zip(u_m, u_f)]


def _out_of_range_condition(rng, bench: BenchmarkConfig, k: int) -> tuple[str, HiddenPhysics]:
    oor = bench.out_of_range
    outer = oor.box
    band = ("mass_low", "friction_high", "mass_high", "friction_low")[k % 4]
    lo_hi = {"mass_low": oor.mass_low_band, "mass_high": oor.mass_high_band,
             "friction_low": oor.friction_low_band, "friction_high": oor.friction_high_band}[band]
    if band.startswith("mass"):
        mass = rng.uniform(*lo_hi)
        friction = rng.uniform(*outer.sliding_friction)
    else:
        friction = rng.uniform(*lo_hi)
        mass = rng.uniform(*outer.mass)
    return band, _round_physics(mass, friction)


def _yaw_mod(yaw: float) -> float:
    return yaw % (math.pi / 2)


def layouts_same(a: ScenarioLayout, b: ScenarioLayout, min_sep: float, min_yaw_deg: float) -> bool:
    """True if two initial states count as the same (no coordinate differs by the separations)."""
    db = math.dist(a.block_xy, b.block_xy)
    dt = math.dist(a.target_xy, b.target_xy)
    dy = abs(_yaw_mod(a.block_yaw) - _yaw_mod(b.block_yaw))
    dy = min(dy, math.pi / 2 - dy)
    return db < min_sep and dt < min_sep and math.degrees(dy) < min_yaw_deg


def goal_sector(layout: ScenarioLayout, sectors: int) -> int:
    angle = math.atan2(layout.target_xy[1] - layout.block_xy[1], layout.target_xy[0] - layout.block_xy[0])
    return int((angle % math.tau) // (math.tau / sectors)) % sectors


def _sample_layout(rng, scene: SceneConfig, sector: int, sectors: int) -> ScenarioLayout:
    w = scene.workspace
    width = math.tau / sectors
    for _ in range(100000):
        bx, by = rng.uniform(w.block_x.low, w.block_x.high), rng.uniform(w.block_y.low, w.block_y.high)
        d = rng.uniform(w.block_target_distance.low, w.block_target_distance.high)
        angle = (sector + rng.uniform()) * width
        tx, ty = bx + d * math.cos(angle), by + d * math.sin(angle)
        yaw = rng.uniform(w.block_yaw.low, w.block_yaw.high)
        layout = ScenarioLayout((round(bx, 5), round(by, 5)), round(yaw, 5), (round(tx, 5), round(ty, 5)))
        try:
            layout.check(scene)
        except ValueError:
            continue
        if goal_sector(layout, sectors) != sector:
            continue  # rounding moved it across a sector edge
        return layout
    raise ManifestError(f"no valid layout in goal sector {sector}")


def generate_manifests(scene: SceneConfig, bench: BenchmarkConfig, provenance: dict[str, Any]) -> dict[str, dict]:
    """Build the four manifest documents (deterministic for a given benchmark config)."""
    if not bench.out_of_range.fixed:
        raise ManifestError("the out-of-range cohort is still a candidate: run contact-check and fix it first")
    ms = bench.manifests
    rng = np.random.default_rng(ms.generator_seed)
    ids: set[str] = set()
    sep = ms.min_condition_separation
    all_conditions: list[np.ndarray] = []
    split_conditions: dict[str, list[Condition]] = {}

    def accept(physics: HiddenPhysics) -> bool:
        point = _norm(bench, physics)
        return all(np.linalg.norm(point - q) >= sep for q in all_conditions)

    for split in ("train", "validation", "test"):
        spec = bench.splits[split]
        conditions: list[Condition] = []
        for attempt in range(1000):
            proposals = _in_range_conditions(rng, bench, spec.in_range_conditions)
            pts = [_norm(bench, p) for _, p in proposals]
            ok = all(accept(p) for _, p in proposals) and all(
                np.linalg.norm(pts[i] - pts[j]) >= sep for i in range(len(pts)) for j in range(i))
            if ok:
                break
        else:
            raise ManifestError(f"{split}: could not place separated in-range conditions")
        for band, physics in proposals:
            all_conditions.append(_norm(bench, physics))
            conditions.append(Condition(_hex_id(rng, ids, "c"), "in_range", band, physics))
        for k in range(spec.out_of_range_conditions):
            for attempt in range(1000):
                band, physics = _out_of_range_condition(rng, bench, k)
                if accept(physics) and not bench.in_range.contains(physics.block_mass, physics.sliding_friction) \
                        and bench.out_of_range.box.contains(physics.block_mass, physics.sliding_friction):
                    break
            else:
                raise ManifestError(f"{split}: could not place out-of-range condition {k}")
            all_conditions.append(_norm(bench, physics))
            conditions.append(Condition(_hex_id(rng, ids, "c"), "out_of_range", band, physics))
        split_conditions[split] = conditions

    ret = bench.splits["retention"]
    train_in = [c for c in split_conditions["train"] if c.cohort == "in_range"]
    picks = sorted(rng.choice(len(train_in), size=ret.in_range_conditions, replace=False).tolist())
    split_conditions["retention"] = [train_in[i] for i in picks]

    all_layouts: list[ScenarioLayout] = []
    reset_seed = ms.reset_seed_start
    docs: dict[str, dict] = {}
    for split in GENERATION_ORDER:
        spec = bench.splits[split]
        scenarios: list[Scenario] = []
        slot = 0
        for condition in split_conditions[split]:
            for _ in range(spec.layouts_per_condition):
                sector = slot % ms.goal_direction_sectors
                slot += 1
                for attempt in range(1000):
                    layout = _sample_layout(rng, scene, sector, ms.goal_direction_sectors)
                    if not any(layouts_same(layout, other, ms.min_layout_separation, ms.min_layout_yaw_separation_deg)
                               for other in all_layouts):
                        break
                else:
                    raise ManifestError(f"{split}: could not place a distinct layout")
                all_layouts.append(layout)
                scenarios.append(Scenario(_hex_id(rng, ids, "s"), condition.condition_id, reset_seed, sector, layout))
                reset_seed += 1
        order = rng.permutation(len(scenarios))
        scenarios = [scenarios[i] for i in order]
        docs[split] = _document(split, spec.sealed, split_conditions[split], scenarios, bench, provenance)
    return docs


def _document(split: str, sealed: bool, conditions: list[Condition], scenarios: list[Scenario],
              bench: BenchmarkConfig, provenance: dict[str, Any]) -> dict[str, Any]:
    spec = bench.splits[split]
    return {
        "kind": "phase3_manifest",
        "manifest_schema_version": bench.manifests.manifest_schema_version,
        "split": split,
        "sealed": sealed,
        "from_split": spec.from_split,
        "benchmark": {"benchmark_version": bench.benchmark_version, "config_sha256": bench.sha256,
                      "schema_version": bench.schema_version, "metrics_version": bench.metrics["version"],
                      "out_of_range_fixed_utc": bench.out_of_range.fixed_utc},
        "generator_seed": bench.manifests.generator_seed,
        "provenance": provenance,
        "counts": {"conditions": len(conditions), "scenarios": len(scenarios),
                   "by_cohort": {c: sum(1 for s in scenarios
                                        if next(x for x in conditions if x.condition_id == s.condition_id).cohort == c)
                                 for c in COHORTS}},
        "conditions": [{"condition_id": c.condition_id, "cohort": c.cohort, "band": c.band,
                        "block_mass": c.physics.block_mass, "sliding_friction": c.physics.sliding_friction}
                       for c in sorted(conditions, key=lambda c: c.condition_id)],
        "scenarios": [{"scenario_id": s.scenario_id, "condition_id": s.condition_id, "reset_seed": s.reset_seed,
                       "goal_sector": s.goal_sector,
                       "layout": {"block_xy": list(s.layout.block_xy), "block_yaw": s.layout.block_yaw,
                                  "target_xy": list(s.layout.target_xy)}}
                      for s in scenarios],
    }


def manifest_provenance(scene: SceneConfig) -> dict[str, Any]:
    """Software versions and simulation settings recorded in every manifest (deterministic per machine)."""
    import mujoco

    from .environment import environment_info

    env = environment_info()
    model_opt = {}
    try:
        from .scenario import ScenarioRunner

        runner = ScenarioRunner(scene)
        opt = runner.sim_model.opt
        model_opt = {"timestep": float(opt.timestep), "integrator": int(opt.integrator), "cone": int(opt.cone),
                     "solver": int(opt.solver), "iterations": int(opt.iterations), "tolerance": float(opt.tolerance),
                     "noslip_iterations": int(opt.noslip_iterations)}
    except Exception as exc:  # recorded, not hidden
        model_opt = {"error": f"{type(exc).__name__}: {exc}"}
    assets = json.loads(paths.PANDA_MANIFEST.read_text(encoding="utf-8")) if paths.PANDA_MANIFEST.exists() else {}
    source = assets.get("source", {})
    return {
        "python": env["python"], "python_platform": env["python_platform"],
        "versions": env["versions"], "mujoco_version": mujoco.__version__,
        "panda_asset": {"repository": source.get("repository"), "commit": source.get("commit"),
                        "path": source.get("path"), "files": len(assets.get("files", [])),
                        "manifest_sha256": _file_sha(paths.PANDA_MANIFEST) if paths.PANDA_MANIFEST.exists() else None},
        "config_sha256": {name: _file_sha(path) for name, path in (
            ("scene.toml", paths.SCENE_CONFIG), ("push.toml", paths.PUSH_CONFIG),
            ("benchmark.toml", paths.BENCHMARK_CONFIG), ("panda_push_scene.xml", paths.SCENE_XML))},
        "simulation": {"control_dt": scene.timing.control_dt, "substeps": scene.timing.substeps, **model_opt},
    }


def write_manifests(docs: dict[str, dict], out_dir: Path = paths.MANIFESTS_DIR, *,
                    command: str = "", overwrite: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes_path = out_dir / HASHES_FILE
    if hashes_path.exists() and not overwrite:
        raise ManifestError(f"{hashes_path} exists: manifests are already written (regenerating needs a revised benchmark)")
    files = {}
    for split in SPLITS:
        data = canonical_bytes(docs[split])
        path = out_dir / f"{split}.json"
        path.write_bytes(data)
        files[split] = {"file": path.name, "sha256": sha256_bytes(data), "sealed": docs[split]["sealed"],
                        "conditions": docs[split]["counts"]["conditions"],
                        "scenarios": docs[split]["counts"]["scenarios"],
                        "by_cohort": docs[split]["counts"]["by_cohort"]}
    record = {"kind": "phase3_manifest_hashes", "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "command": command, "benchmark": docs["train"]["benchmark"], "manifests": files}
    hashes_path.write_bytes(canonical_bytes(record))
    return record


# --------------------------------------------------------------------------- loading and verification


def _parse(split: str, path: Path, data: bytes) -> Manifest:
    doc = json.loads(data)
    if doc.get("kind") != "phase3_manifest" or doc.get("split") != split:
        raise ManifestError(f"{path}: not the {split} manifest")
    conditions = {c["condition_id"]: Condition(c["condition_id"], c["cohort"], c["band"],
                                                HiddenPhysics(c["block_mass"], c["sliding_friction"]))
                  for c in doc["conditions"]}
    scenarios = tuple(Scenario(s["scenario_id"], s["condition_id"], int(s["reset_seed"]), int(s["goal_sector"]),
                               ScenarioLayout(tuple(s["layout"]["block_xy"]), s["layout"]["block_yaw"],
                                              tuple(s["layout"]["target_xy"])))
                      for s in doc["scenarios"])
    return Manifest(split, bool(doc["sealed"]), sha256_bytes(data), path, doc, conditions, scenarios)


def load_hashes(manifest_dir: Path = paths.MANIFESTS_DIR) -> dict[str, Any]:
    path = manifest_dir / HASHES_FILE
    if not path.exists():
        raise ManifestError(f"{path} missing: run `python -m physical_ai_lab manifests` first")
    return json.loads(path.read_text(encoding="ascii"))


def _read_verified(split: str, manifest_dir: Path) -> tuple[Path, bytes, dict[str, Any]]:
    if split not in SPLITS:
        raise ManifestError(f"unknown split {split!r}")
    hashes = load_hashes(manifest_dir)
    entry = hashes["manifests"][split]
    path = manifest_dir / entry["file"]
    data = path.read_bytes()
    if sha256_bytes(data) != entry["sha256"]:
        raise ManifestError(f"{path}: SHA-256 differs from {HASHES_FILE}; the manifest was changed")
    return path, data, entry


def open_split(split: str, *, manifest_dir: Path = paths.MANIFESTS_DIR, freeze_dir: Path = paths.FREEZES_DIR,
               bench: BenchmarkConfig | None = None) -> Manifest:
    """Load a split for evaluation. Sealed splits need a valid candidate-freeze record (Phase 5)."""
    path, data, entry = _read_verified(split, manifest_dir)
    if entry["sealed"]:
        freeze = valid_freeze(split, entry["sha256"], freeze_dir)
        if freeze is None:
            raise SealedSplitError(
                f"the {split} split is sealed: it opens only after a candidate is frozen (BUILD_PLAN.md Phase 5); "
                f"no valid freeze record in {freeze_dir}. This applies to the baseline too.")
    manifest = _parse(split, path, data)
    if bench is not None and manifest.data["benchmark"]["config_sha256"] != bench.sha256:
        raise ManifestError(f"{path} was generated with benchmark config {manifest.data['benchmark']['config_sha256'][:16]}, "
                            f"current is {bench.sha256[:16]}")
    return manifest


def read_sealed_for_audit(split: str, manifest_dir: Path = paths.MANIFESTS_DIR) -> Manifest:
    """Read a manifest (any split) for structural checks only: disjointness, counts, hashes.

    Evaluator-side tests use this to verify the sealed manifests without running any episode on them.
    """
    path, data, _ = _read_verified(split, manifest_dir)
    return _parse(split, path, data)


# --------------------------------------------------------------------------- freeze records


def freeze_candidate(candidate_path: Path, *, candidate_name: str, manifest_dir: Path = paths.MANIFESTS_DIR,
                     freeze_dir: Path = paths.FREEZES_DIR, note: str = "") -> Path:
    """Phase 5 step: freeze a learned candidate and commit to the sealed manifests' hashes.

    The candidate file's SHA-256 is recorded; the freeze cannot be made for the fixed baseline
    (it is not a candidate), and an existing freeze record is never overwritten.
    """
    if candidate_name.strip().lower() in ("", "baseline"):
        raise ValueError("a freeze needs a learned candidate; the fixed baseline is not one")
    candidate_path = Path(candidate_path)
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate artifact {candidate_path} not found")
    hashes = load_hashes(manifest_dir)
    sealed = {s: e["sha256"] for s, e in hashes["manifests"].items() if e["sealed"]}
    for split in sealed:
        _read_verified(split, manifest_dir)
    record = {"kind": FREEZE_KIND, "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "candidate": {"name": candidate_name, "file": str(candidate_path),
                            "sha256": _file_sha(candidate_path)},
              "sealed_manifests": sealed, "benchmark": hashes["benchmark"], "note": note}
    freeze_dir.mkdir(parents=True, exist_ok=True)
    path = freeze_dir / f"freeze_{record['candidate']['sha256'][:16]}.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; freeze records are never overwritten")
    path.write_bytes(canonical_bytes(record))
    return path


def valid_freeze(split: str, manifest_sha: str, freeze_dir: Path = paths.FREEZES_DIR) -> dict[str, Any] | None:
    if not freeze_dir.is_dir():
        return None
    for path in sorted(freeze_dir.glob("freeze_*.json")):
        try:
            record = json.loads(path.read_text(encoding="ascii"))
        except (OSError, ValueError):
            continue
        if record.get("kind") != FREEZE_KIND or record.get("sealed_manifests", {}).get(split) != manifest_sha:
            continue
        candidate = record.get("candidate", {})
        file = Path(candidate.get("file", ""))
        if candidate.get("name", "").lower() in ("", "baseline") or not file.is_file():
            continue
        if _file_sha(file) != candidate.get("sha256"):
            continue  # the candidate changed after the freeze
        return record
    return None


# --------------------------------------------------------------------------- checks


def check_disjoint(manifests: dict[str, Manifest], bench: BenchmarkConfig) -> dict[str, Any]:
    """Structural checks of a full manifest set; returns a report and raises on any violation."""
    ms = bench.manifests
    problems: list[str] = []
    phys = {s: {c.condition_id: c.physics for c in m.conditions.values()} for s, m in manifests.items()}
    primary = ("train", "validation", "test")
    for i, a in enumerate(primary):
        for b in primary[i + 1:]:
            if set(phys[a]) & set(phys[b]):
                problems.append(f"{a}/{b} share condition ids")
            for pa in phys[a].values():
                for pb in phys[b].values():
                    if np.linalg.norm(_norm(bench, pa) - _norm(bench, pb)) < ms.min_condition_separation:
                        problems.append(f"{a}/{b} conditions closer than the separation: {pa} {pb}")
    if not set(phys["retention"]) <= set(phys["train"]):
        problems.append("retention conditions are not train conditions")
    for cid, p in phys["retention"].items():
        if phys["train"].get(cid) != p:
            problems.append(f"retention condition {cid} physics differ from train")
    layouts = [(s, sc) for s, m in manifests.items() for sc in m.scenarios]
    for i in range(len(layouts)):
        for j in range(i):
            if layouts_same(layouts[i][1].layout, layouts[j][1].layout, ms.min_layout_separation,
                            ms.min_layout_yaw_separation_deg):
                problems.append(f"same initial state in {layouts[i][0]} and {layouts[j][0]}")
    ids = [sc.scenario_id for _, sc in layouts]
    if len(set(ids)) != len(ids):
        problems.append("duplicate scenario ids")
    seeds = [sc.reset_seed for _, sc in layouts]
    if len(set(seeds)) != len(seeds):
        problems.append("duplicate reset seeds")
    counts = {}
    for split, m in manifests.items():
        spec = bench.splits[split]
        by = {c: sum(1 for sc in m.scenarios if m.conditions[sc.condition_id].cohort == c) for c in COHORTS}
        counts[split] = {"conditions": len(m.conditions), "scenarios": len(m.scenarios), "by_cohort": by}
        if len(m.scenarios) != spec.scenarios or len(m.conditions) != spec.conditions:
            problems.append(f"{split}: counts differ from benchmark.toml")
        for c in m.conditions.values():
            p = c.physics
            inside = bench.in_range.contains(p.block_mass, p.sliding_friction)
            if (c.cohort == "in_range") != inside:
                problems.append(f"{split}: condition {c.condition_id} cohort label does not match its physics")
            if c.cohort == "out_of_range" and not bench.out_of_range.box.contains(p.block_mass, p.sliding_friction):
                problems.append(f"{split}: condition {c.condition_id} outside the out-of-range box")
        per_cond: dict[str, set[int]] = {}
        for sc in m.scenarios:
            per_cond.setdefault(sc.condition_id, set()).add(sc.goal_sector)
            if goal_sector(sc.layout, ms.goal_direction_sectors) != sc.goal_sector:
                problems.append(f"{split}: goal sector label wrong")
        if any(len(v) != spec.layouts_per_condition for v in per_cond.values()):
            problems.append(f"{split}: a condition repeats a goal sector")
    if problems:
        raise ManifestError("; ".join(problems[:10]) + (f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""))
    return {"ok": True, "counts": counts, "layout_pairs_checked": len(layouts) * (len(layouts) - 1) // 2}
