"""Review check (not a project feature) for the Phase 2 review (D25).

  run OUT [settings]     unchanged-config baseline on the 30 development seeds under forced hidden
                         physics (same settings as review2_stress.py), trajectories + records to OUT
  analyze DIR [DIR ...]  the same trajectory metrics for any stress output folder (before and after)
  compare BEFORE AFTER   per-setting metrics side by side plus per-seed pairing

Trajectory metrics use the evaluator's per-control-update CSV rows (values rounded to 1e-6).
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from physical_ai_lab.baseline import FeedbackBaseline  # noqa: E402
from physical_ai_lab.config import load_scene_config  # noqa: E402
from physical_ai_lab.evaluation import run_episode  # noqa: E402
from physical_ai_lab.push_config import load_push_config  # noqa: E402
from physical_ai_lab.scenario import HiddenPhysics, ScenarioRunner  # noqa: E402

SETTINGS = {"nominal": (0.25, 0.50), "heavy_grippy": (0.40, 0.80), "light_slippery": (0.10, 0.20)}
PUSH = load_push_config()
SCENE = load_scene_config()
H = PUSH.controller.push_height
DT = SCENE.timing.substeps * SCENE.timing.physics_timestep
STEP = np.asarray(PUSH.controller.max_joint_speed) * DT


class ForcedRunner(ScenarioRunner):
    forced = None

    def reset(self, seed, randomize_physics=False, physics=None, **kw):
        return super().reset(seed, physics=self.forced, **kw)


def episode_metrics(path: Path) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    z = np.array([float(r["tcp_z"]) for r in rows])
    contact = np.array([int(float(r["robot_block_contact"])) for r in rows], dtype=bool)
    block_z = np.array([float(r["block_z"]) for r in rows])
    targets = np.array([[float(r[f"q{i}_target"]) for i in range(1, 8)] for r in rows])
    ratio = np.abs(np.diff(targets, axis=0)) / STEP
    zc = z[contact]
    return dict(
        min_tcp_mm=round(z.min() * 1e3, 2),
        below_push_mm=round((H - z.min()) * 1e3, 2),
        contact_rows=int(contact.sum()),
        contact_z_lo_mm=round(zc.min() * 1e3, 2) if len(zc) else None,
        contact_z_hi_mm=round(zc.max() * 1e3, 2) if len(zc) else None,
        block_rise_mm=round((block_z.max() - block_z[0]) * 1e3, 3),
        table_contact_rows=int(sum(int(float(r["robot_table_contact"])) for r in rows)),
        first_update_ratio=round(float(ratio[0].max()), 4) if len(ratio) else None,
        max_update_ratio=round(float(ratio[1:].max()), 4) if len(ratio) > 1 else None,
    )


def run(out: Path, labels: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    runner = ForcedRunner()
    policy = FeedbackBaseline(runner.config, PUSH)
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "settings": {}}
    for label in labels:
        m, mu = SETTINGS[label]
        runner.forced = HiddenPhysics(block_mass=m, sliding_friction=mu)
        recs = []
        for seed in PUSH.seeds.development:
            tp = out / label / "trajectories" / f"seed_{seed}.csv"
            rec = run_episode(runner, policy, PUSH, seed, trajectory_path=tp)
            assert abs(runner.applied_physics().block_mass - m) < 1e-12
            forb = [e for e in rec.contact_events if e["forbidden"]]
            r = dict(seed=seed, success=rec.success, reason=rec.reason, detail=rec.detail,
                     pushes=rec.pushes_used, t=round(rec.end_time, 2),
                     final_mm=round(rec.final["block_target_distance"] * 1e3, 1),
                     judge_min_tcp_mm=round(rec.stats["min_tcp_z"] * 1e3, 2),
                     judge_max_joint_speed_ratio=round(rec.stats["max_joint_speed_ratio"], 4),
                     forbidden=[(e["kind"], e["body_a"], e["body_b"], e["phase"]) for e in forb])
            r.update(episode_metrics(tp))
            recs.append(r)
            print(label, seed, "OK " if r["success"] else "FAIL", r["reason"], "pushes", r["pushes"],
                  "minTCP", r["min_tcp_mm"], "judge minTCP", r["judge_min_tcp_mm"],
                  "contact z", r["contact_z_lo_mm"], "..", r["contact_z_hi_mm"],
                  "block rise", r["block_rise_mm"], "upd ratio", r["max_update_ratio"], r["forbidden"][:2], flush=True)
        summary["settings"][label] = dict(mass=m, friction=mu, records=recs)
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "stress_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


def load(folder: Path) -> dict:
    summary = json.loads((folder / "stress_summary.json").read_text(encoding="utf-8"))
    result = {}
    for label, s in summary["settings"].items():
        recs = []
        for r in s["records"]:
            r = dict(r)
            r.update(episode_metrics(folder / label / "trajectories" / f"seed_{r['seed']}.csv"))
            recs.append(r)
        result[label] = recs
    return result


def setting_line(recs: list[dict]) -> dict:
    ok = [r for r in recs if r["success"]]
    reasons: dict[str, int] = {}
    for r in recs:
        if not r["success"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    lo = [r["contact_z_lo_mm"] for r in recs if r["contact_z_lo_mm"] is not None]
    hi = [r["contact_z_hi_mm"] for r in recs if r["contact_z_hi_mm"] is not None]
    return dict(
        successes=f"{len(ok)}/{len(recs)}", failures=reasons or "none",
        mean_pushes_all=round(float(np.mean([r["pushes"] for r in recs])), 2),
        max_pushes=max(r["pushes"] for r in recs),
        mean_time_all_s=round(float(np.mean([r["t"] for r in recs])), 2),
        final_mm_range_successes=(min(r["final_mm"] for r in ok), max(r["final_mm"] for r in ok)) if ok else None,
        min_tcp_mm=min(r["min_tcp_mm"] for r in recs),
        judge_min_tcp_mm=min(r["judge_min_tcp_mm"] for r in recs) if "judge_min_tcp_mm" in recs[0] else "n/a",
        max_below_push_mm=max(r["below_push_mm"] for r in recs),
        contact_z_mm=(min(lo), max(hi)),
        max_block_rise_mm=max(r["block_rise_mm"] for r in recs),
        episodes_with_table_contact=sum(1 for r in recs if r["table_contact_rows"] > 0),
        forbidden_contacts=sum(len(r["forbidden"]) for r in recs),
        max_first_update_ratio=max(r["first_update_ratio"] for r in recs),
        max_update_ratio=max(r["max_update_ratio"] for r in recs),
        judge_max_joint_speed_ratio=(max(r["judge_max_joint_speed_ratio"] for r in recs)
                                     if "judge_max_joint_speed_ratio" in recs[0] else "n/a"),
    )


def analyze(folders: list[Path]) -> None:
    for folder in folders:
        print(f"## {folder}")
        for label, recs in load(folder).items():
            print(f"  {label}: {json.dumps(setting_line(recs))}")


def compare(before: Path, after: Path) -> None:
    b, a = load(before), load(after)
    report = {"before": str(before), "after": str(after), "settings": {}}
    for label in b:
        if label not in a:
            continue
        bl, al = setting_line(b[label]), setting_line(a[label])
        pairs = list(zip(b[label], a[label], strict=True))
        assert all(x["seed"] == y["seed"] for x, y in pairs)
        changed = [(x["seed"], x["pushes"], y["pushes"]) for x, y in pairs if x["pushes"] != y["pushes"]]
        lost = [x["seed"] for x, y in pairs if x["success"] and not y["success"]]
        gained = [x["seed"] for x, y in pairs if y["success"] and not x["success"]]
        report["settings"][label] = dict(before=bl, after=al, seeds_with_changed_pushes=changed,
                                         successes_lost=lost, successes_gained=gained)
        print(f"== {label} (m={SETTINGS[label][0]}, mu={SETTINGS[label][1]})")
        for key in bl:
            print(f"   {key:30s} before {str(bl[key]):28s} after {al[key]}")
        print(f"   pushes changed on {len(changed)} seeds {changed}; successes lost {lost}, gained {gained}")
    (after / "compare_to_before.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(Path(sys.argv[2]), sys.argv[3].split(",") if len(sys.argv) > 3 else list(SETTINGS))
    elif cmd == "analyze":
        analyze([Path(p) for p in sys.argv[2:]])
    elif cmd == "compare":
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
