import math
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import IGRIS_C_walk.tasks  # noqa


def _pick_torque_tensor(robot):
    """Try to find a useful torque tensor field across Isaac Lab versions."""
    for name in ["applied_torque", "joint_torques", "computed_torque", "actuator_torques"]:
        if hasattr(robot.data, name):
            return getattr(robot.data, name), name
    return None, None


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    base_env = env.unwrapped
    num_envs = base_env.num_envs
    device = base_env.device
    dt = base_env.step_dt

    robot = base_env.scene["robot"]

    # -------------------------
    # 베이스 고정 (공중에서 관절 튜닝용)
    # -------------------------
    root_state_fixed = robot.data.root_state_w.clone()
    root_state_fixed[:, 7:13] = 0.0  # lin/ang vel 0

    # -------------------------
    # action 설정
    # -------------------------
    action_dim = env.action_space.shape[-1]  # ex) 12
    target_action_indices = [0]             # 0~(action_dim-1)

    # -------------------------
    # 사각파(스텝) 튜닝 파라미터
    # -------------------------
    amplitude = 0.4          # action magnitude
    period_s = 2.0           # seconds (ex: 2초 주기 -> 1초마다 +amp/-amp)
    settle_print_every_s = 0.25  # 로그 주기(초)

    # 로그 준비
    torque_tensor, torque_name = _pick_torque_tensor(robot)
    print(f"[INFO] action_dim={action_dim}, target_action_indices={target_action_indices}")
    print(f"[INFO] torque field: {torque_name if torque_name else 'N/A'}")
    print(f"[INFO] dt={dt:.6f}s, square period={period_s}s, amplitude={amplitude}")

    t = 0.0
    step_i = 0
    log_every_steps = max(1, int(settle_print_every_s / dt))

    # 참고: action index와 robot joint index는 다를 수 있음.
    # 여기서는 "관절 측정"을 action index로 찍지 않고, 관측에 들어가는 joint_pos(12개) 첫 12개를 기준으로 찍는 게 안전함.
    # (정확한 매핑이 필요하면 action term의 joint_ids를 출력해서 매핑을 만들어야 함)
    obs_joint_dim = min(12, robot.data.joint_pos.shape[1])

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros((num_envs, action_dim), device=device)

            # -------------------------
            # 사각파 생성: period_s 주기로 +amplitude / -amplitude
            # -------------------------
            phase = (t % period_s)
            value = amplitude if phase < (period_s / 2.0) else -amplitude

            for a_idx in target_action_indices:
                if 0 <= a_idx < action_dim:
                    actions[:, a_idx] = value

            # step
            env.step(actions)

            # 베이스 고정 (중력은 관절에만 작용)
            robot.write_root_state_to_sim(root_state_fixed)
            robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_vel_w))

            # -------------------------
            # 로깅 (주기적으로)
            # -------------------------
            if step_i % log_every_steps == 0:
                # 관절 상태
                q = robot.data.joint_pos[0, :obs_joint_dim].detach().cpu()
                qd = robot.data.joint_vel[0, :obs_joint_dim].detach().cpu()

                # 타깃 action index에 대응되는 "관측 joint_pos의 인덱스"를 같은 숫자로 가정하면 틀릴 수 있음.
                # 그래도 튜닝 감을 보기 위한 간단 로그로는, 우선 a_idx와 동일 인덱스가 존재할 때만 찍음.
                for a_idx in target_action_indices:
                    if a_idx < obs_joint_dim:
                        q_i = q[a_idx].item()
                        qd_i = qd[a_idx].item()

                        tau_i = None
                        if torque_tensor is not None and a_idx < torque_tensor.shape[1]:
                            tau_i = torque_tensor[0, a_idx].item()

                        if tau_i is None:
                            print(f"t={t:7.3f}  a[{a_idx}]={value:+.3f}  q={q_i:+.3f}  qd={qd_i:+.3f}")
                        else:
                            print(
                                f"t={t:7.3f}  a[{a_idx}]={value:+.3f}  q={q_i:+.3f}  qd={qd_i:+.3f}  tau={tau_i:+.3f}"
                            )
                    else:
                        print(f"t={t:7.3f}  a[{a_idx}]={value:+.3f}  (no joint index {a_idx} in logged slice)")

            # advance time
            t += dt
            step_i += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
