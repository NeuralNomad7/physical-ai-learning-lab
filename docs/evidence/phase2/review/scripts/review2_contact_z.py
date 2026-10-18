"""Review 2 probe: measured TCP height while the gripper touches the block, fix on vs off (one baseline push)."""
import sys
import numpy as np
from physical_ai_lab.config import load_scene_config
from physical_ai_lab.push_config import load_push_config
from physical_ai_lab.scenario import ScenarioRunner
from physical_ai_lab.kinematics import ArmKinematics
from physical_ai_lab.controller import PushSkill, EpisodeOver
from physical_ai_lab.baseline import FeedbackBaseline
from physical_ai_lab.evaluation import TRAJECTORY_COLUMNS, EpisodeJudge, EvaluatedPlant

cfg = load_scene_config(); push = load_push_config()
runner = ScenarioRunner(cfg)
c = TRAJECTORY_COLUMNS.index
h = push.controller.push_height
original = ArmKinematics.gravity_offset
seeds = range(21000, 21000 + int(sys.argv[1]) if len(sys.argv) > 1 else 21010)
for mode in ("fix", "nofix"):
    ArmKinematics.gravity_offset = original if mode == "fix" else (lambda self, q: np.zeros(len(q)))
    contact_lo, contact_hi, overall_lo = [], [], []
    for seed in seeds:
        skill = PushSkill(cfg, push, ArmKinematics(cfg))
        obs = runner.reset(seed)
        plan, _ = FeedbackBaseline(cfg, push, skill).choose(obs)
        judge = EpisodeJudge(runner, push); plant = EvaluatedPlant(runner, judge, push)
        plant.note("push_start", index=1, plan=plan.summary(), candidates=[])
        try:
            skill.execute(plant, obs, plan)
            ended = "-"
        except EpisodeOver:
            o = judge.outcome
            ended = f"{o.reason} at t={o.time:.2f} phase={judge.context['phase']} ({o.detail})"
        rows = plant.trajectory
        zc = np.array([r[c("tcp_z")] for r in rows if r[c("robot_block_contact")]])
        zall = min(r[c("tcp_z")] for r in rows)
        stroke = [r for r in rows if r[c("phase")] == "stroke"]
        contact_lo.append(zc.min() if len(zc) else np.nan); contact_hi.append(zc.max() if len(zc) else np.nan)
        overall_lo.append(zall)
        print(f"{mode} seed {seed} speed {plan.action.speed:.2f} stroke-start z {stroke[0][c('tcp_z')]*1e3:.2f} mm"
              f" | contact rows {len(zc)} z {zc.min()*1e3:.2f}..{zc.max()*1e3:.2f} mm"
              f" | min z all {zall*1e3:.2f} mm | ended {ended}")
    print(f"{mode} SUMMARY contact z {np.nanmin(contact_lo)*1e3:.2f}..{np.nanmax(contact_hi)*1e3:.2f} mm,"
          f" min z all {min(overall_lo)*1e3:.2f} mm (push height {h*1e3:.1f} mm)")
