# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Random action agent + PD/torque debug for Isaac Lab environments.

What this adds:
- Periodic debug print:
  q_rms, qd_rms, tau_rms (applied), sat ratio vs YOUR torque limits, clamp_gap_rms (computed-applied),
  tilt proxy from projected gravity, and top offending joints by saturation ratio.
- One-time dump of torque-related fields that actually exist on your robot.data.

How to use:
  python scripts/random_agent_dbg.py --task Igris_Flat_v1 --num_envs 10
"""

import argparse
import math

from isaaclab.app import AppLauncher

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="Random agent with PD debug for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, required=True, help="Gym task id, e.g. Igris_Flat_v1")

# Random action amplitude (IMPORTANT for PD stability check)
parser.add_argument("--action_amp", type=float, default=0.1, help="Action amplitude multiplier (start small).")

# Debug cadence
parser.add_argument("--dbg_every", type=int, default=120, help="Print debug every N env steps.")
parser.add_argument("--dbg_topk", type=int, default=6, help="Top-K offending joints to show.")

# append AppLauncher args (device, headless, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------- Isaac imports (after app launch) ----------------
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import IGRIS_C_walk.tasks  # noqa: F401  (ensures gym registry imports)

# ---------------- YOUR JOINT META (EDIT HERE) ----------------
# If you already import this from your asset cfg file, replace this block with:
#   from IGRIS_C_walk.tasks.manager_based.igris_c_walk.igris_c import _JOINT_META
_JOINT_META = {
    # LEFT LEG
    "Joint_Hip_Pitch_Left":  {"kp": 500.0, "kd": 35, "torque": 100.0, "vmax": 10000.0, "arm": None},
    "Joint_Hip_Roll_Left":   {"kp": 450.0, "kd": 35, "torque": 100.0,  "vmax": 10000.0, "arm": None},
    "Joint_Hip_Yaw_Left":    {"kp": 400.0,  "kd": 40, "torque": 100.0,  "vmax": 10000.0, "arm": None},
    "Joint_Knee_Pitch_Left": {"kp": 500.0, "kd": 40, "torque": 100.0, "vmax": 10000.0, "arm": None},
    "Joint_Ankle_Pitch_Left":{"kp": 50.0, "kd": 8, "torque": 70.0,  "vmax": 10000.0, "arm": None},
    "Joint_Ankle_Roll_Left": {"kp": 50.0, "kd": 0.8, "torque": 70.0,  "vmax": 10000.0, "arm": None},

    # RIGHT LEG
    "Joint_Hip_Pitch_Right":  {"kp": 500.0, "kd": 35, "torque": 100.0, "vmax": 10000.0, "arm": None},
    "Joint_Hip_Roll_Right":   {"kp": 450.0, "kd": 35, "torque": 100.0,  "vmax": 10000.0, "arm": None},
    "Joint_Hip_Yaw_Right":    {"kp": 400.0,  "kd": 40, "torque": 100.0,  "vmax": 10000.0, "arm": None},
    "Joint_Knee_Pitch_Right": {"kp": 500.0, "kd": 40, "torque": 100.0, "vmax": 10000.0, "arm": None},
    "Joint_Ankle_Pitch_Right":{"kp": 100.0, "kd": 8, "torque": 70.0,  "vmax": 10000.0, "arm": None},
    "Joint_Ankle_Roll_Right": {"kp": 10.0, "kd": 0.8, "torque": 70.0,  "vmax": 10000.0, "arm": None},
}

LEG_JOINTS = [
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Yaw_Left",
    "Joint_Knee_Pitch_Left",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Roll_Left",
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Right",
]


# ---------------- Debug helpers ----------------
def _safe_getattr(obj, name: str):
    return getattr(obj, name) if hasattr(obj, name) else None


def dump_torque_fields_once(env):
    robot = env.unwrapped.scene["robot"]
    data = robot.data
    print("\n[DBG] ===== robot.data torque/effort fields =====")
    for k in [
        "applied_torque",
        "computed_torque",
        "joint_effort",
        "joint_effort_limits",
        "joint_effort_target",
        "joint_pos_limits",
        "joint_vel_limits",
        "projected_gravity_b",
    ]:
        v = _safe_getattr(data, k)
        if v is None:
            print(f"  - {k:<28} (missing)")
            continue
        if torch.is_tensor(v):
            finite = torch.isfinite(v).all().item()
            vmin = v.min().item()
            vmax = v.max().item()
            print(f"  - {k:<28} shape={tuple(v.shape)} finite={finite} min={vmin:.3g} max={vmax:.3g}")
        else:
            print(f"  - {k:<28} type={type(v)}")
    print("[DBG] =========================================\n")


def build_effort_limits(device, joint_names_all, joint_meta, joints_of_interest):
    name_to_idx = {n: i for i, n in enumerate(joint_names_all)}
    missing = [j for j in joints_of_interest if j not in name_to_idx]
    if missing:
        raise KeyError(
            f"[DBG] These joints are not in robot.data.joint_names:\n  {missing}\n"
            f"Tip: print(robot.data.joint_names) and fix naming."
        )
    idx = torch.tensor([name_to_idx[j] for j in joints_of_interest], device=device, dtype=torch.long)

    # Use YOUR torque limits (from _JOINT_META), not robot.data.joint_effort_limits (often nonsense/unset)
    lim = torch.tensor([float(joint_meta[j]["torque"]) for j in joints_of_interest], device=device, dtype=torch.float32)
    # Avoid divide-by-zero
    lim = torch.clamp(lim, min=1e-6)
    return idx, lim


def dbg_pd(env, joint_meta, every=120, topk=6):
    robot = env.unwrapped.scene["robot"]
    data = robot.data

    # Isaac Lab increments common_step_counter in env (ManagerBased)
    step = int(getattr(env.unwrapped, "common_step_counter", 0))
    if step % every != 0:
        return

    # Required tensors
    q = data.joint_pos
    qd = data.joint_vel
    tau_applied = data.applied_torque     # post-limit/clamp torque
    tau_cmd = data.computed_torque        # PD computed torque (pre-limit)

    idx, lim = build_effort_limits(
        device=q.device,
        joint_names_all=data.joint_names,
        joint_meta=joint_meta,
        joints_of_interest=LEG_JOINTS,
    )

    qj = q[:, idx]
    qdj = qd[:, idx]
    tau_a = tau_applied[:, idx]
    tau_c = tau_cmd[:, idx]

    # Finite check
    finite = (
        torch.isfinite(qj).all()
        and torch.isfinite(qdj).all()
        and torch.isfinite(tau_a).all()
        and torch.isfinite(tau_c).all()
    )
    if not finite:
        print(f"[DBG step={step}] Non-finite detected in q/qd/tau (sim exploded).")
        return

    # RMS (across envs and joints)
    q_rms = torch.sqrt(torch.mean(qj**2)).item()
    qd_rms = torch.sqrt(torch.mean(qdj**2)).item()
    tau_rms = torch.sqrt(torch.mean(tau_a**2)).item()

    # Saturation vs YOUR torque limits
    ratio = torch.abs(tau_a) / lim.view(1, -1)        # (Nenv,12)
    sat = (ratio > 1.0).float()
    tau_sat = sat.mean().item()
    tau_sat_env_max = ratio.max(dim=1).values.mean().item()

    # Biggest offenders by mean ratio per joint
    mean_ratio_per_joint = ratio.mean(dim=0)          # (12,)
    k = min(topk, mean_ratio_per_joint.numel())
    topv, topi = torch.topk(mean_ratio_per_joint, k=k)
    offenders = ", ".join([f"{LEG_JOINTS[i]}:{topv[j].item():.2f}x" for j, i in enumerate(topi.tolist())])

    # Tilt proxy from projected gravity (if available)
    tilt_mean = float("nan")
    tilt_max = float("nan")
    if hasattr(data, "projected_gravity_b") and torch.is_tensor(data.projected_gravity_b):
        pg = data.projected_gravity_b
        tilt = torch.linalg.norm(pg[:, :2], dim=1)  # 0이면 upright
        tilt_mean = tilt.mean().item()
        tilt_max = tilt.max().item()

    # How much clamping happened (cmd vs applied)
    clamp_gap_rms = torch.sqrt(torch.mean((tau_c - tau_a) ** 2)).item()

    print(
        f"[DBG step={step}] q_rms={q_rms:.4f} qd_rms={qd_rms:.4f} "
        f"tau_rms={tau_rms:.2f} sat={tau_sat:.3f} env_max={tau_sat_env_max:.2f} "
        f"clamp_gap_rms={clamp_gap_rms:.2f} tilt_mean={tilt_mean:.3f} tilt_max={tilt_max:.3f} "
        f"top={offenders}"
    )


# ---------------- Main ----------------
def main():
    # create env cfg
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # make env
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: action_amp={args_cli.action_amp} dbg_every={args_cli.dbg_every}")

    # reset
    obs = env.reset()

    # dump once
    dump_torque_fields_once(env)

    # step loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Random actions in [-1,1], scaled
            actions = args_cli.action_amp * (2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0)

            # gymnasium API can return 4 or 5 values depending on wrapper
            out = env.step(actions)
            if isinstance(out, tuple) and len(out) == 5:
                obs, rew, terminated, truncated, info = out
            elif isinstance(out, tuple) and len(out) == 4:
                obs, rew, done, info = out
            else:
                # unexpected API
                pass

            # debug
            dbg_pd(env, _JOINT_META, every=args_cli.dbg_every, topk=args_cli.dbg_topk)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
