"""Review check (not a project feature): unchanged Phase 2 baseline on the development seeds
under forced hidden physics; measures executed TCP height, sag and table contact."""
import csv, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from physical_ai_lab.scenario import ScenarioRunner, HiddenPhysics
from physical_ai_lab.evaluation import run_episode
from physical_ai_lab.baseline import FeedbackBaseline
from physical_ai_lab.push_config import load_push_config
from physical_ai_lab.kinematics import ArmKinematics

class ForcedRunner(ScenarioRunner):
    forced = None
    def reset(self, seed, randomize_physics=False, physics=None, **kw):
        return super().reset(seed, physics=self.forced, **kw)

settings = {"nominal": (0.25, 0.50), "heavy_grippy": (0.40, 0.80), "light_slippery": (0.10, 0.20)}
if len(sys.argv) > 2:
    settings = {k: v for k, v in settings.items() if k in sys.argv[2].split(",")}
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
push = load_push_config(); runner = ForcedRunner(); policy = FeedbackBaseline(runner.config, push)
kin = ArmKinematics()
summary = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "settings": {}}
for label, (m, mu) in settings.items():
    runner.forced = HiddenPhysics(block_mass=m, sliding_friction=mu)
    recs = []
    for seed in push.seeds.development:
        tp = out / label / "trajectories" / f"seed_{seed}.csv"
        rec = run_episode(runner, policy, push, seed, trajectory_path=tp)
        assert abs(runner.applied_physics().block_mass - m) < 1e-12
        rows = list(csv.DictReader(open(tp)))
        minz = min(float(r["tcp_z"]) for r in rows)
        stroke = [r for r in rows if r["phase"] == "stroke"]
        sag = max((kin.forward(np.array([float(r[f"q{i}_target"]) for i in range(1, 8)]))[0][2] - float(r["tcp_z"]))
                  for r in stroke) if stroke else float("nan")
        minz_stroke = min(float(r["tcp_z"]) for r in stroke) if stroke else float("nan")
        table_rows = sum(int(float(r["robot_table_contact"])) for r in rows)
        forb = [e for e in rec.contact_events if e["forbidden"]]
        recs.append(dict(seed=seed, success=rec.success, reason=rec.reason, detail=rec.detail,
                         pushes=rec.pushes_used, t=round(rec.end_time, 2),
                         final_mm=round(rec.final["block_target_distance"] * 1e3, 1),
                         min_tcp_mm=round(minz * 1e3, 2), min_tcp_stroke_mm=round(minz_stroke * 1e3, 2),
                         max_stroke_sag_mm=round(sag * 1e3, 2), table_contact_rows=table_rows,
                         forbidden=[(e["kind"], e["body_a"], e["body_b"], e["phase"]) for e in forb]))
        r = recs[-1]
        print(label, seed, "OK " if r["success"] else "FAIL", r["reason"], "pushes", r["pushes"],
              "minTCP", r["min_tcp_mm"], "stroke minTCP", r["min_tcp_stroke_mm"], "sag", r["max_stroke_sag_mm"],
              "table rows", table_rows, r["forbidden"][:2], flush=True)
    ok = sum(r["success"] for r in recs)
    reasons = {}
    for r in recs:
        if not r["success"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    s = dict(mass=m, friction=mu, successes=ok, episodes=len(recs), reasons=reasons,
             min_tcp_mm=min(r["min_tcp_mm"] for r in recs),
             max_stroke_sag_mm=max(r["max_stroke_sag_mm"] for r in recs if r["max_stroke_sag_mm"] == r["max_stroke_sag_mm"]),
             episodes_with_table_contact=sum(1 for r in recs if r["table_contact_rows"] > 0), records=recs)
    summary["settings"][label] = s
    print(f"== {label} (m={m}, mu={mu}): {ok}/{len(recs)} success, reasons {reasons}, min TCP {s['min_tcp_mm']} mm, "
          f"max stroke sag {s['max_stroke_sag_mm']} mm, episodes with robot-table contact {s['episodes_with_table_contact']}", flush=True)
summary["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
(out / "stress_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
