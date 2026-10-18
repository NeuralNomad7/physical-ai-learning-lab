"""Probe: gravity offset vs the simulator's own bias forces, and static hold error in a raw sim."""
import itertools
import math

import mujoco
import numpy as np

from physical_ai_lab.config import load_scene_config
from physical_ai_lab.kinematics import ArmKinematics
from physical_ai_lab.push_config import load_push_config
from physical_ai_lab.scenario import ScenarioRunner

cfg = load_scene_config()
push = load_push_config()
kin = ArmKinematics(cfg)
ctl = push.controller
print("servo kp", kin.servo_kp)

# 1) offset at home vs the evaluator simulator's qfrc_bias at rest
runner = ScenarioRunner(cfg)
obs = runner.reset(21000)
d = runner.sim_data
m = runner.sim_model
print("sim qvel max at reset", float(np.max(np.abs(d.qvel))))
sim_bias = d.qfrc_bias[kin.arm_dadr] / kin.servo_kp
off = kin.gravity_offset(obs.arm_qpos)
print("home offset (mrad)", np.round(off * 1e3, 4))
print("sim bias/kp (mrad)", np.round(sim_bias * 1e3, 4))
print("max diff (rad)", float(np.max(np.abs(off - sim_bias))))

# 2) static hold in a raw sim built from the MJCF
model = mujoco.MjModel.from_xml_path(str(cfg.mjcf_path))
data = mujoco.MjData(model)
key = model.key(cfg.robot.home_keyframe).id
arm_act = np.array([model.actuator(n).id for n in cfg.robot.arm_actuators])
grip_act = model.actuator(cfg.robot.gripper_actuator).id
finger_qadr = np.array([model.joint(n).qposadr[0] for n in cfg.robot.finger_joints])
hand = model.body(cfg.robot.tcp_body).id
robot_root = model.body_rootid[hand]
tcp_off = np.array(cfg.robot.tcp_offset)


def tcp():
    r = data.xmat[hand].reshape(3, 3)
    return data.xpos[hand] + r @ tcp_off


def robot_contacts():
    n = 0
    for c in data.contact[: data.ncon]:
        for g in (c.geom1, c.geom2):
            if model.body_rootid[model.geom_bodyid[g]] == robot_root:
                n += 1
                break
    return n


def hold(q, ctrl_q, seconds=1.0):
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[kin.arm_qadr] = q
    data.qpos[finger_qadr] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[arm_act] = ctrl_q
    data.ctrl[grip_act] = cfg.robot.gripper_ctrl_closed
    mujoco.mj_forward(model, data)
    contacts = 0
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        contacts = max(contacts, robot_contacts())
    return tcp(), contacts, float(np.max(np.abs(data.qvel[kin.arm_dadr])))


rows = []
for (x, y), yaw in itertools.product(((0.40, -0.25), (0.40, 0.25), (0.65, -0.20), (0.65, 0.20), (0.72, 0.0),
                                      (0.30, 0.0)), (0.0, math.pi / 2, -math.pi / 4)):
    target = np.array([x, y, ctl.push_height])
    rest = kin.home_q.copy()
    rest[0] = math.atan2(y, x)
    res = None
    for hy in (yaw, yaw + math.pi):
        res = kin.solve(target, hy, kin.home_q, rest, joint_margin=ctl.joint_margin,
                        position_tolerance=ctl.ik_position_tolerance,
                        orientation_tolerance=ctl.ik_orientation_tolerance)
        if res.converged:
            break
    if not res.converged:
        print("no IK", x, y, yaw)
        continue
    q = res.q
    g = kin.gravity_offset(q)
    p_ff, c_ff, v_ff = hold(q, q + g)
    p_raw, c_raw, v_raw = hold(q, q)
    rows.append((x, y, yaw, (p_ff[2] - target[2]) * 1e3, np.linalg.norm(p_ff - target) * 1e3,
                 (p_raw[2] - target[2]) * 1e3, np.linalg.norm(p_raw - target) * 1e3, c_ff, c_raw,
                 float(np.max(np.abs(g))), v_ff))
print(f"{'x':>5} {'y':>6} {'yaw':>6} | {'ff dz':>7} {'ff |e|':>7} | {'raw dz':>7} {'raw |e|':>7} | cff craw | max|g| rad | ff qvel")
for r in rows:
    print(f"{r[0]:5.2f} {r[1]:6.2f} {math.degrees(r[2]):6.1f} | {r[3]:7.3f} {r[4]:7.3f} | {r[5]:7.3f} {r[6]:7.3f} | "
          f"{r[7]:3d} {r[8]:4d} | {r[9]:.5f} | {r[10]:.2e}")
ff = np.array([abs(r[3]) for r in rows])
raw = np.array([r[5] for r in rows])
print("max |ff dz| mm", ff.max(), "max ff |e| mm", max(r[4] for r in rows))
print("raw dz mm range", raw.min(), raw.max())
