# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""Rough terrain locomotion environment config for IGRIS_C."""

from typing import Dict, Optional, Tuple
import torch

import isaaclab.terrains as terrain_gen
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ImuCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from IGRIS_C_walk.tasks.manager_based.igris_c_walk.mdp import terminations
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from .igris_c import igris_c_CFG


# -----------------------------
# IGRIS naming (YOU GAVE THESE)
# -----------------------------
IGRIS_JOINTS_12 = [
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

IGRIS_LINKS = [
    "Link_Hip_Pitch_Left",
    "Link_Hip_Roll_Left",
    "Link_Hip_Yaw_Left",
    "Link_Knee_Pitch_Left",
    "Link_Hip_Pitch_Right",
    "Link_Hip_Roll_Right",
    "Link_Hip_Yaw_Right",
    "Link_Knee_Pitch_Right",
]


# Feet/body names (from your logs)
IGRIS_FEET_BODIES = ["Link_Ankle_Roll_Left", "Link_Ankle_Roll_Right"]

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=IGRIS_JOINTS_12,
        scale=0.25,
        use_default_offset=True,
    )


def randomize_imu_mount(
    env: ManagerBasedEnv,
    env_ids: Optional[torch.Tensor],
    sensor_cfg: SceneEntityCfg,
    pos_range: Dict[str, Tuple[float, float]],
    rot_range: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    imu_sensor = env.scene.sensors[sensor_cfg.name]
    env_indices = env_ids if env_ids is not None else torch.arange(imu_sensor.num_instances, device=env.device)
    num_envs_to_update = len(env_indices)

    def sample_uniform(lo: float, hi: float) -> torch.Tensor:
        return (hi - lo) * torch.rand(num_envs_to_update, device=env.device) + lo

    position_offsets = torch.stack(
        [sample_uniform(*pos_range["x"]), sample_uniform(*pos_range["y"]), sample_uniform(*pos_range["z"])],
        dim=-1,
    )
    roll_offsets = sample_uniform(*rot_range["roll"])
    pitch_offsets = sample_uniform(*rot_range["pitch"])
    yaw_offsets = sample_uniform(*rot_range["yaw"])
    quaternion_offsets = quat_from_euler_xyz(roll_offsets, pitch_offsets, yaw_offsets)

    imu_sensor._offset_pos_b[env_indices] = position_offsets
    imu_sensor._offset_quat_b[env_indices] = quaternion_offsets

    mean_offset_cm = (position_offsets.norm(dim=-1).mean() * 100.0).item()
    mean_tilt_deg = (torch.rad2deg(torch.acos(quaternion_offsets[:, 0].clamp(-1.0, 1.0))).mean().item())

    return {"imu_offset_cm": mean_offset_cm, "imu_tilt_deg": mean_tilt_deg}


IGRIS_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
    },
)


def velocity_push_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    min_push: float,
    max_push: float,
    curriculum_start_step: int,
    curriculum_stop_step: int,
):
    if env.common_step_counter < curriculum_start_step:
        progress = 0.0
    else:
        curriculum_duration = curriculum_stop_step - curriculum_start_step
        progress = (env.common_step_counter - curriculum_start_step) / curriculum_duration
        progress = min(progress, 1.0)

    current_velocity = min_push + (max_push - min_push) * progress

    if hasattr(env.event_manager.cfg, "push_robot"):
        env.event_manager.cfg.push_robot.params["velocity_range"] = {
            "x": (-current_velocity, current_velocity),
            "y": (-current_velocity, current_velocity),
        }

    return {"push_velocity_progress": progress, "push_velocity_magnitude": current_velocity}


@configclass
class IGRISRewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=IGRIS_FEET_BODIES),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=IGRIS_FEET_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=IGRIS_FEET_BODIES),
        },
    )

    # Joint-limit & deviation penalties (ALL UPDATED TO YOUR JOINT NAMES)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Joint_Ankle_Pitch_Left",
                    "Joint_Ankle_Roll_Left",
                    "Joint_Ankle_Pitch_Right",
                    "Joint_Ankle_Roll_Right",
                ],
            )
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Joint_Hip_Yaw_Left",
                    "Joint_Hip_Roll_Left",
                    "Joint_Hip_Yaw_Right",
                    "Joint_Hip_Roll_Right",
                ],
            )
        },
    )

    joint_deviation_hip_pitch_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Joint_Hip_Pitch_Left",
                    "Joint_Knee_Pitch_Left",
                    "Joint_Hip_Pitch_Right",
                    "Joint_Knee_Pitch_Right",
                ],
            ),
        },
    )

    joint_deviation_ankles = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Joint_Ankle_Pitch_Left",
                    "Joint_Ankle_Roll_Left",
                    "Joint_Ankle_Pitch_Right",
                    "Joint_Ankle_Roll_Right",
                ],
            ),
        },
    )

    # torso_yaw 같은거 없는 모델이니까 제거(=None으로 꺼버림)
    joint_deviation_arms = None

    foot_impact_penalty = RewTerm(
        func=mdp.contact_forces,
        weight=-1.5e-3,
        params={
            "threshold": 800.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=IGRIS_FEET_BODIES),
        },
    )



@configclass
class IGRISObservations:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # 다리 관절만 관측
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=IGRIS_JOINTS_12)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=IGRIS_JOINTS_12)})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# @configclass
# class IGRISObservations:
#     @configclass
#     class CriticCfg(ObservationGroupCfg):
#         base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
#         base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
#         projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
#         velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
#         actions = ObsTerm(func=mdp.last_action)

#         height_scan = ObsTerm(
#             func=mdp.height_scan,
#             params={"sensor_cfg": SceneEntityCfg("height_scanner")},
#             noise=Unoise(n_min=-0.1, n_max=0.1),
#             clip=(-1.0, 1.0),
#         )

#         imu_projected_gravity = ObsTerm(
#             func=mdp.imu_projected_gravity,
#             params={"asset_cfg": SceneEntityCfg("imu")},
#             noise=Unoise(n_min=-0.05, n_max=0.05),
#         )
#         imu_ang_vel = ObsTerm(
#             func=mdp.imu_ang_vel,
#             params={"asset_cfg": SceneEntityCfg("imu")},
#             noise=Unoise(n_min=-0.1, n_max=0.1),
#         )
#         imu_lin_acc = ObsTerm(
#             func=mdp.imu_lin_acc,
#             params={"asset_cfg": SceneEntityCfg("imu")},
#             noise=Unoise(n_min=-0.1, n_max=0.1),
#         )

#         joint_torques = ObsTerm(
#             func=mdp.joint_effort,
#             params={"asset_cfg": SceneEntityCfg("robot")},
#             noise=Unoise(n_min=-0.0001, n_max=0.0001),
#         )

#         # body_poses: torso_dummy_1 같은거 제거하고, 확실히 있는 것만 사용
#         body_poses = ObsTerm(
#             func=mdp.body_pose_w,
#             params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"] + IGRIS_FEET_BODIES)},
#             noise=Unoise(n_min=-0.0001, n_max=0.0001),
#         )

#         joint_pos_accurate = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.0001, n_max=0.0001))
#         joint_vel_accurate = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.0001, n_max=0.0001))

#         base_pos = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.0001, n_max=0.0001))
#         root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001))
#         root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001))

#         def __post_init__(self):
#             self.enable_corruption = False

#     @configclass
#     class PolicyCfg(ObservationGroupCfg):
#         velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

#         # IMPORTANT: restrict to your 12 joints so joint_pos_rel/joint_vel_rel doesn't look for adult joints
#         joint_pos = ObsTerm(
#             func=mdp.joint_pos_rel,
#             params={"asset_cfg": SceneEntityCfg("robot", joint_names=IGRIS_JOINTS_12, preserve_order=True)},
#             noise=Unoise(n_min=-0.05, n_max=0.05),
#         )
#         joint_vel = ObsTerm(
#             func=mdp.joint_vel_rel,
#             params={"asset_cfg": SceneEntityCfg("robot", joint_names=IGRIS_JOINTS_12, preserve_order=True)},
#             noise=Unoise(n_min=-0.5, n_max=0.5),
#         )

#         imu_ang_vel = ObsTerm(
#             func=mdp.imu_ang_vel,
#             params={"asset_cfg": SceneEntityCfg("imu")},
#             noise=Unoise(n_min=-0.1, n_max=0.1),
#         )

#         actions = ObsTerm(func=mdp.last_action)

#         def __post_init__(self):
#             self.enable_corruption = True
#             self.concatenate_terms = True

#     critic: CriticCfg = CriticCfg()
#     policy: PolicyCfg = PolicyCfg()


@configclass
class IGRISCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    velocity_push_curriculum = CurrTerm(
        func=velocity_push_curriculum,
        params={
            "min_push": 0.01,
            "max_push": 2.0,
            "curriculum_start_step": 24 * 500,
            "curriculum_stop_step": 24 * 5500,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_height_low = DoneTerm(
        func=terminations.base_height_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.50,   # <- 네가 말한 값. 나중에 0.35~0.45로 조절할 수도 있음
        },
    )



@configclass
class IGRISRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    enable_randomization: bool = True
    rewards: IGRISRewards = IGRISRewards()
    observations: IGRISObservations = IGRISObservations()
    curriculum: IGRISCurriculumCfg = IGRISCurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Scene
        self.scene.robot = igris_c_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Terrains
        self.scene.terrain.terrain_generator = IGRIS_ROUGH_TERRAINS_CFG
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # IMU
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            update_period=0.0,
            debug_vis=True,
            gravity_bias=(0.0, 0.0, 0.0),
            offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.events.base_external_force_torque = None

        # Friction randomization: apply only on feet bodies we know exist
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=IGRIS_FEET_BODIES),
                "static_friction_range": (0.05, 4.0),
                "dynamic_friction_range": (0.05, 4.0),
                "restitution_range": (0.05, 0.5),
                "num_buckets": 64,
                "make_consistent": True,
            },
        )

        # Mass randomization: use a safe regex so it doesn't crash on missing adult link names
        self.events.add_limb_masses = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.7, 1.5),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )

        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=IGRIS_JOINTS_12),
                "stiffness_distribution_params": (0.5, 1.5),
                "damping_distribution_params": (0.5, 1.5),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        self.events.randomize_joint_properties = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=IGRIS_JOINTS_12),
                "friction_distribution_params": (0.0, 0.5),
                "armature_distribution_params": (0.5, 1.5),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # Reset joints by offset (works)
        self.events.reset_robot_joints.params["position_range"] = (-0.2, 0.2)
        self.events.reset_robot_joints.params["velocity_range"] = (-1.0, 1.0)
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset

        # Push
        self.events.push_robot.mode = "interval"
        self.events.push_robot.interval_range_s = (5.0, 15.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}

        # Base reset
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        }

        # IMU mount randomization
        self.events.randomize_imu_mount = EventTerm(
            func=randomize_imu_mount,
            mode="reset",
            params={
                "sensor_cfg": SceneEntityCfg("imu"),
                "pos_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
                "rot_range": {"roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            },
        )

        # If base mass/com randomization breaks on your asset, keep them off
        self.events.add_base_mass = None
        self.events.base_com = None

        # Rewards tuning + make sure torque/acc penalties reference your joints (NOT adult joints)
        #self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.05

        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "Joint_Hip_Pitch_Left", "Joint_Hip_Roll_Left", "Joint_Hip_Yaw_Left", "Joint_Knee_Pitch_Left",
                "Joint_Hip_Pitch_Right", "Joint_Hip_Roll_Right", "Joint_Hip_Yaw_Right", "Joint_Knee_Pitch_Right",
            ],
        )

        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=IGRIS_JOINTS_12)

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.2

        # # Terminations: adult 링크들 싹 제거하고, "base_link만" 일단 확실히 존재하는 걸로
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link"]

        if not self.enable_randomization:
            self._disable_randomization()

    def _disable_randomization(self):
        print("[INFO]: Disabling all domain randomization!\n" * 5, end="")

        self.events.physics_material = None
        self.events.add_limb_masses = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_properties = None
        self.events.randomize_imu_mount = None

        self.events.reset_robot_joints.params.update({"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)})
        self.events.reset_robot_joints.func = mdp.reset_joints_by_scale

        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        self.events.push_robot = None
        if hasattr(self.curriculum, "velocity_push_curriculum"):
            self.curriculum.velocity_push_curriculum = None

        self.observations.policy.enable_corruption = False

        if hasattr(self.rewards, "foot_impact_penalty"):
            self.rewards.foot_impact_penalty = None


@configclass
class IGRISRoughEnvCfg_PLAY(IGRISRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum.velocity_push_curriculum = None
