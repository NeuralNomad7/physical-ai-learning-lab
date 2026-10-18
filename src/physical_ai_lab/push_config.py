"""Typed loader for ``configs/push.toml`` (Phase 2 push skill, baseline and episode rules)."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import paths

N_ARM = 7


@dataclass(frozen=True)
class EpisodeRules:
    max_pushes: int
    max_seconds: float
    success_hold_seconds: float
    success_max_planar_speed: float


@dataclass(frozen=True)
class EvaluatorLimits:
    max_joint_speed: tuple[float, ...]
    joint_range_tolerance: float
    min_tcp_z: float
    pusher_bodies: tuple[str, ...]


@dataclass(frozen=True)
class ControllerConfig:
    push_height: float
    safe_height: float
    pusher_half_thickness: float
    standoff: float
    retreat_distance: float
    transit_speed: float
    descend_speed: float
    ramp_seconds: float
    dwell_seconds: float
    max_joint_speed: tuple[float, ...]
    joint_margin: float
    min_tcp_z: float
    table_margin: float
    block_margin: float
    max_off_normal_deg: float
    ik_position_tolerance: float
    ik_orientation_tolerance: float
    settle_speed: float
    settle_max_seconds: float

    @property
    def max_off_normal(self) -> float:
        return math.radians(self.max_off_normal_deg)


@dataclass(frozen=True)
class ActionSpace:
    speeds: tuple[float, ...]
    durations: tuple[float, ...]


@dataclass(frozen=True)
class BaselineConfig:
    stop_tolerance: float
    speed_preference: float


@dataclass(frozen=True)
class SeedLists:
    gate: tuple[int, ...]
    gate_required_successes: int
    development: tuple[int, ...]


PUSH_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PushConfig:
    source: Path
    episode: EpisodeRules
    limits: EvaluatorLimits
    controller: ControllerConfig
    action: ActionSpace
    baseline: BaselineConfig
    seeds: SeedLists


def _floats(values: object, name: str, length: int | None = None) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    out = tuple(float(v) for v in values)
    if length is not None and len(out) != length:
        raise ValueError(f"{name} must have {length} entries, got {len(out)}")
    if not all(math.isfinite(v) and v > 0 for v in out):
        raise ValueError(f"{name} entries must be finite and positive")
    return out


def _validate(cfg: PushConfig) -> None:
    ep, lim, ctl, seeds = cfg.episode, cfg.limits, cfg.controller, cfg.seeds
    if ep.max_pushes < 1 or ep.max_seconds <= 0 or ep.success_hold_seconds <= 0:
        raise ValueError("episode budget and success hold must be positive")
    if not 0 < lim.min_tcp_z < ctl.min_tcp_z <= ctl.push_height < ctl.safe_height:
        raise ValueError("need 0 < limits.min_tcp_z < controller.min_tcp_z <= push_height < safe_height")
    if any(c > e for c, e in zip(ctl.max_joint_speed, lim.max_joint_speed)):
        raise ValueError("controller.max_joint_speed must not exceed limits.max_joint_speed")
    if not 0 < ctl.max_off_normal_deg < 45:
        raise ValueError("max_off_normal_deg must be in (0, 45)")
    if set(seeds.gate) & set(seeds.development):
        raise ValueError("gate and development seeds must be disjoint")
    if len(set(seeds.gate)) != len(seeds.gate) or not 0 < seeds.gate_required_successes <= len(seeds.gate):
        raise ValueError("gate seeds must be unique and gate_required_successes in 1..len(gate)")
    if any(s < 0 for s in seeds.gate + seeds.development):
        raise ValueError("seeds must be non-negative")


def load_push_config(path: Path | str = paths.PUSH_CONFIG) -> PushConfig:
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != PUSH_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {raw.get('schema_version')!r}")
    e, lim, c, a, b, s = (raw[k] for k in ("episode", "limits", "controller", "action", "baseline", "seeds"))
    cfg = PushConfig(
        source=path,
        episode=EpisodeRules(
            max_pushes=int(e["max_pushes"]),
            max_seconds=float(e["max_seconds"]),
            success_hold_seconds=float(e["success_hold_seconds"]),
            success_max_planar_speed=float(e["success_max_planar_speed"]),
        ),
        limits=EvaluatorLimits(
            max_joint_speed=_floats(lim["max_joint_speed"], "limits.max_joint_speed", N_ARM),
            joint_range_tolerance=float(lim["joint_range_tolerance"]),
            min_tcp_z=float(lim["min_tcp_z"]),
            pusher_bodies=tuple(str(x) for x in lim["pusher_bodies"]),
        ),
        controller=ControllerConfig(
            **{k: float(v) for k, v in c.items() if k != "max_joint_speed"},
            max_joint_speed=_floats(c["max_joint_speed"], "controller.max_joint_speed", N_ARM),
        ),
        action=ActionSpace(
            speeds=_floats(a["speeds"], "action.speeds"),
            durations=_floats(a["durations"], "action.durations"),
        ),
        baseline=BaselineConfig(
            stop_tolerance=float(b["stop_tolerance"]),
            speed_preference=float(b["speed_preference"]),
        ),
        seeds=SeedLists(
            gate=tuple(int(x) for x in s["gate"]),
            gate_required_successes=int(s["gate_required_successes"]),
            development=tuple(int(x) for x in s["development"]),
        ),
    )
    _validate(cfg)
    return cfg
