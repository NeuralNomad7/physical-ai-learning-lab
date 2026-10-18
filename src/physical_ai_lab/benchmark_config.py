"""Typed loader for ``configs/benchmark.toml`` (Phase 3 benchmark definition).

The loader cross-checks the benchmark against ``push.toml`` (budgets, success rule) and
``scene.toml`` (target tolerance, development physics ranges) and refuses to load if they differ,
so the versioned benchmark rules and the rules the evaluator applies cannot drift apart.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import paths
from .config import SceneConfig, load_scene_config
from .push_config import PushConfig, load_push_config

SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test", "retention")
COHORTS = ("in_range", "out_of_range")


@dataclass(frozen=True)
class Box:
    mass: tuple[float, float]
    sliding_friction: tuple[float, float]

    def contains(self, mass: float, friction: float) -> bool:
        return (self.mass[0] <= mass <= self.mass[1]
                and self.sliding_friction[0] <= friction <= self.sliding_friction[1])


@dataclass(frozen=True)
class Rules:
    push_config_schema_version: int
    max_pushes: int
    max_seconds: float
    success_tolerance: float
    success_hold_seconds: float
    success_max_planar_speed: float
    count_unplanned_contacts_as_pushes: bool
    count_probe_pushes: bool


@dataclass(frozen=True)
class AgentConfig:
    protocol_version: int
    reply_timeout_seconds: float


@dataclass(frozen=True)
class OutOfRangeCohort:
    status: str
    fixed_utc: str
    box: Box
    mass_low_band: tuple[float, float]
    mass_high_band: tuple[float, float]
    friction_low_band: tuple[float, float]
    friction_high_band: tuple[float, float]

    @property
    def fixed(self) -> bool:
        return self.status == "fixed"


@dataclass(frozen=True)
class SplitSpec:
    name: str
    in_range_conditions: int
    out_of_range_conditions: int
    layouts_per_condition: int
    sealed: bool
    from_split: str | None

    @property
    def conditions(self) -> int:
        return self.in_range_conditions + self.out_of_range_conditions

    @property
    def scenarios(self) -> int:
        return self.conditions * self.layouts_per_condition


@dataclass(frozen=True)
class ManifestSettings:
    manifest_schema_version: int
    generator_seed: int
    reset_seed_start: int
    min_condition_separation: float
    min_layout_separation: float
    min_layout_yaw_separation_deg: float
    goal_direction_sectors: int


@dataclass(frozen=True)
class StatisticsConfig:
    confidence: float
    bootstrap_resamples: int
    bootstrap_seed: int
    cluster_key: str


@dataclass(frozen=True)
class ReplayConfig:
    state_abs_tolerance: float
    scenarios_per_split: int


@dataclass(frozen=True)
class FeasibilityConfig:
    candidate_mass: tuple[float, float]
    candidate_sliding_friction: tuple[float, float]
    mass_only: tuple[float, ...]
    friction_only: tuple[float, ...]
    kick_seed: int
    kick_directions_deg: tuple[float, ...]
    hold_seconds: float
    hold_seed: int
    push_seeds: tuple[int, ...]
    inertia_rel_tolerance: float
    max_block_lift: float
    max_block_tilt_deg: float


@dataclass(frozen=True)
class BenchmarkConfig:
    source: Path
    sha256: str  # of the file bytes; recorded with every manifest and result
    schema_version: int
    benchmark_version: str
    rules: Rules
    agent: AgentConfig
    in_range: Box
    out_of_range: OutOfRangeCohort
    splits: dict[str, SplitSpec]
    manifests: ManifestSettings
    statistics: StatisticsConfig
    metrics: dict[str, Any]
    replay: ReplayConfig
    feasibility: FeasibilityConfig

    def version_record(self) -> dict[str, Any]:
        return {"benchmark_version": self.benchmark_version, "schema_version": self.schema_version,
                "metrics_version": self.metrics["version"], "config_sha256": self.sha256,
                "out_of_range_status": self.out_of_range.status,
                "out_of_range_fixed_utc": self.out_of_range.fixed_utc}


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [low, high]")
    low, high = float(value[0]), float(value[1])
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
        raise ValueError(f"{name} must satisfy 0 < low < high")
    return low, high


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def validate(cfg: BenchmarkConfig, scene: SceneConfig, push: PushConfig) -> None:
    r, ep = cfg.rules, push.episode
    if r.push_config_schema_version != 1:
        raise ValueError("rules.push_config_schema_version must match push.toml schema_version 1")
    mismatches = [name for name, a, b in (
        ("max_pushes", r.max_pushes, ep.max_pushes),
        ("max_seconds", r.max_seconds, ep.max_seconds),
        ("success_hold_seconds", r.success_hold_seconds, ep.success_hold_seconds),
        ("success_max_planar_speed", r.success_max_planar_speed, ep.success_max_planar_speed),
        ("success_tolerance", r.success_tolerance, scene.workspace.target_tolerance),
    ) if not _close(float(a), float(b))]
    if mismatches:
        raise ValueError(f"benchmark.toml [rules] differ from push.toml/scene.toml: {mismatches}")
    if not (r.count_unplanned_contacts_as_pushes and r.count_probe_pushes):
        raise ValueError("unplanned contacts and probe pushes must count against the push budget")
    ph = scene.physics
    if not (all(map(_close, cfg.in_range.mass, (ph.mass_range.low, ph.mass_range.high)))
            and all(map(_close, cfg.in_range.sliding_friction,
                        (ph.sliding_friction_range.low, ph.sliding_friction_range.high)))):
        raise ValueError("cohorts.in_range must equal scene.toml [physics] development ranges")
    oor = cfg.out_of_range
    if oor.status not in ("candidate", "fixed"):
        raise ValueError("cohorts.out_of_range.status must be 'candidate' or 'fixed'")
    if oor.fixed and not oor.fixed_utc:
        raise ValueError("a fixed out-of-range cohort needs fixed_utc")
    outer, inner = oor.box, cfg.in_range
    if not (outer.mass[0] < inner.mass[0] and outer.mass[1] > inner.mass[1]
            and outer.sliding_friction[0] < inner.sliding_friction[0]
            and outer.sliding_friction[1] > inner.sliding_friction[1]):
        raise ValueError("the out-of-range box must strictly contain the in-range box")
    for band, lo, hi, inside in (
        (oor.mass_low_band, outer.mass[0], inner.mass[0], "below"),
        (oor.mass_high_band, inner.mass[1], outer.mass[1], "above"),
        (oor.friction_low_band, outer.sliding_friction[0], inner.sliding_friction[0], "below"),
        (oor.friction_high_band, inner.sliding_friction[1], outer.sliding_friction[1], "above"),
    ):
        if not (lo <= band[0] and band[1] <= hi):
            raise ValueError(f"out-of-range band {band} must lie {inside} the in-range box, inside the outer box")
        if (inside == "below" and _close(band[1], hi)) or (inside == "above" and _close(band[0], lo)):
            raise ValueError(f"out-of-range band {band} must not touch the in-range box")
    if set(cfg.splits) != set(SPLITS):
        raise ValueError(f"splits must be exactly {SPLITS}")
    for spec in cfg.splits.values():
        if spec.in_range_conditions < 0 or spec.out_of_range_conditions < 0 or spec.conditions < 1:
            raise ValueError(f"split {spec.name}: condition counts must be non-negative, total >= 1")
        if spec.layouts_per_condition < 1:
            raise ValueError(f"split {spec.name}: layouts_per_condition must be >= 1")
    if cfg.splits["test"].scenarios != 100 or not cfg.splits["test"].sealed:
        raise ValueError("the final test must be a sealed split of 100 scenarios (BUILD_PLAN.md section 6)")
    ret = cfg.splits["retention"]
    if ret.scenarios != 50 or not ret.sealed or ret.from_split != "train" or ret.out_of_range_conditions:
        raise ValueError("retention must be 50 sealed in-range scenarios drawn from train conditions")
    if ret.in_range_conditions > cfg.splits["train"].in_range_conditions:
        raise ValueError("retention cannot use more conditions than train has")
    if cfg.splits["train"].sealed or cfg.splits["validation"].sealed:
        raise ValueError("train and validation are the unsealed development splits")
    m = cfg.manifests
    if m.goal_direction_sectors < 1 or m.min_condition_separation <= 0 or m.min_layout_separation <= 0:
        raise ValueError("manifest separations and sectors must be positive")
    used = set(range(0, 200)) | set(push.seeds.gate) | set(push.seeds.development) | set(cfg.feasibility.push_seeds)
    used |= {cfg.feasibility.kick_seed, cfg.feasibility.hold_seed}
    total = sum(s.scenarios for s in cfg.splits.values())
    if used & set(range(m.reset_seed_start, m.reset_seed_start + total)):
        raise ValueError("manifest reset seeds overlap seeds used before")
    s = cfg.statistics
    if not 0 < s.confidence < 1 or s.bootstrap_resamples < 100 or s.cluster_key != "condition_id":
        raise ValueError("statistics: need 0 < confidence < 1, >= 100 resamples, cluster_key condition_id")
    if cfg.replay.state_abs_tolerance < 0 or cfg.replay.scenarios_per_split < 1:
        raise ValueError("replay tolerance must be >= 0 and scenarios_per_split >= 1")
    f = cfg.feasibility
    if f.candidate_mass != outer.mass or f.candidate_sliding_friction != outer.sliding_friction:
        raise ValueError("feasibility candidate ranges must equal the out-of-range box")
    if cfg.metrics.get("version") != 1:
        raise ValueError("metrics.version must be 1")


def load_benchmark_config(
    path: Path | str = paths.BENCHMARK_CONFIG,
    scene: SceneConfig | None = None,
    push: PushConfig | None = None,
) -> BenchmarkConfig:
    path = Path(path)
    data = path.read_bytes()
    raw = tomllib.loads(data.decode("utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {raw.get('schema_version')!r}")
    r, a, co, m, st, rp, fe = (raw[k] for k in ("rules", "agent", "cohorts", "manifests", "statistics",
                                                 "replay", "feasibility"))
    oor = co["out_of_range"]
    splits = {}
    for name in raw["splits"]:
        sp = raw["splits"][name]
        splits[name] = SplitSpec(
            name=name,
            in_range_conditions=int(sp["in_range_conditions"]),
            out_of_range_conditions=int(sp["out_of_range_conditions"]),
            layouts_per_condition=int(sp["layouts_per_condition"]),
            sealed=bool(sp["sealed"]),
            from_split=sp.get("from_split"),
        )
    cfg = BenchmarkConfig(
        source=path,
        sha256=hashlib.sha256(data).hexdigest(),
        schema_version=int(raw["schema_version"]),
        benchmark_version=str(raw["benchmark_version"]),
        rules=Rules(**{k: (bool(v) if isinstance(v, bool) else v) for k, v in r.items()}),
        agent=AgentConfig(protocol_version=int(a["protocol_version"]),
                          reply_timeout_seconds=float(a["reply_timeout_seconds"])),
        in_range=Box(_pair(co["in_range"]["mass"], "in_range.mass"),
                     _pair(co["in_range"]["sliding_friction"], "in_range.sliding_friction")),
        out_of_range=OutOfRangeCohort(
            status=str(oor["status"]),
            fixed_utc=str(oor["fixed_utc"]),
            box=Box(_pair(oor["mass"], "out_of_range.mass"),
                    _pair(oor["sliding_friction"], "out_of_range.sliding_friction")),
            mass_low_band=_pair(oor["mass_low_band"], "mass_low_band"),
            mass_high_band=_pair(oor["mass_high_band"], "mass_high_band"),
            friction_low_band=_pair(oor["friction_low_band"], "friction_low_band"),
            friction_high_band=_pair(oor["friction_high_band"], "friction_high_band"),
        ),
        splits=splits,
        manifests=ManifestSettings(**m),
        statistics=StatisticsConfig(**st),
        metrics=dict(raw["metrics"]),
        replay=ReplayConfig(state_abs_tolerance=float(rp["state_abs_tolerance"]),
                            scenarios_per_split=int(rp["scenarios_per_split"])),
        feasibility=FeasibilityConfig(
            candidate_mass=_pair(fe["candidate_mass"], "candidate_mass"),
            candidate_sliding_friction=_pair(fe["candidate_sliding_friction"], "candidate_sliding_friction"),
            mass_only=tuple(float(v) for v in fe["mass_only"]),
            friction_only=tuple(float(v) for v in fe["friction_only"]),
            kick_seed=int(fe["kick_seed"]),
            kick_directions_deg=tuple(float(v) for v in fe["kick_directions_deg"]),
            hold_seconds=float(fe["hold_seconds"]),
            hold_seed=int(fe["hold_seed"]),
            push_seeds=tuple(int(v) for v in fe["push_seeds"]),
            inertia_rel_tolerance=float(fe["inertia_rel_tolerance"]),
            max_block_lift=float(fe["max_block_lift"]),
            max_block_tilt_deg=float(fe["max_block_tilt_deg"]),
        ),
    )
    validate(cfg, scene or load_scene_config(), push or load_push_config())
    return cfg
