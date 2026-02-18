# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ImuCfg
try:
    from isaaclab.actuators import ImplicitActuatorCfg
except Exception:
    from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

#import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from . import mdp
from IGRIS_C_walk.tasks.manager_based.igris_c_walk.mdp import terminations
from .igris_c_isaaclab_cfg import IGRIS_C_CFG
##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##
LEG_JOINTS = [
     #"Joint_Waist_Yaw",
     #"Joint_Waist_Roll",
     #"Joint_Waist_Pitch",
     "Joint_Hip_Pitch_Left",
     "Joint_Hip_Roll_Left",
     "Joint_Hip_Yaw_Left",
     "Joint_Knee_Pitch_Left",
     "Joint_Ankle_Pitch_Left",
     "Joint_Ankle_Roll_Left",
     "Joint_Hip_Yaw_Right",
     "Joint_Hip_Roll_Right",
     "Joint_Hip_Pitch_Right",
     "Joint_Knee_Pitch_Right",
     "Joint_Ankle_Pitch_Right",
     "Joint_Ankle_Roll_Right",
]

# 발 링크가 따로 없어서 ankle roll 링크를 발로 취급
FOOT_BODIES_REGEX = "Link_Ankle_Roll_(Left|Right)"






@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # robot: ArticulationCfg = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot",
    #     spawn=sim_utils.UsdFileCfg(
    #         #usd_path="/home/moon/isaaclab/assets/robot_control/asset/urdf/igris_c_v2_pelvis/igris_c_v2_pelvis.usd",  # <- 네 USD 경로
    #         usd_path="/home/moon/isaaclab/assets/igris_c_v2.usd",
    #         activate_contact_sensors=True,
    #     ),


    #     actuators={
    #         "legs" : ImplicitActuatorCfg(
    #                     joint_names_expr=LEG_JOINTS,

    #                     stiffness={
    #                     "Joint_Hip_Pitch_(Left|Right)" : 600,
    #                     "Joint_Hip_Roll_(Left|Right)" : 400,
    #                     "Joint_Hip_Yaw_(Left|Right)" : 300,
    #                     "Joint_Knee_Pitch_(Left|Right)" : 600,
    #                     "Joint_Ankle_Pitch_(Left|Right)" : 40,
    #                     "Joint_Ankle_Roll_(Left|Right)" : 40,
    #                     },

    #                     damping={
    #                     "Joint_Hip_Pitch_(Left|Right)" : 30,
    #                     "Joint_Hip_Roll_(Left|Right)" :20,
    #                     "Joint_Hip_Yaw_(Left|Right)" : 15,
    #                     "Joint_Knee_Pitch_(Left|Right)" : 30,
    #                     "Joint_Ankle_Pitch_(Left|Right)" : 1,
    #                     "Joint_Ankle_Roll_(Left|Right)" : 1,#0.8
    #                     },

    #                     effort_limit={
    #                     "Joint_Hip_Pitch_(Left|Right)" : 300,
    #                     "Joint_Hip_Roll_(Left|Right)" : 100, #200
    #                     "Joint_Hip_Yaw_(Left|Right)" : 150,
    #                     "Joint_Knee_Pitch_(Left|Right)" : 300, #250
    #                     "Joint_Ankle_Pitch_(Left|Right)" : 40, #120,
    #                     "Joint_Ankle_Roll_(Left|Right)" : 40, #30,
    #                     },
    #                     armature=0.01,
    #         ),

    #     },
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 1.0),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    # )


    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.25,
        use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        #base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        #base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}, noise=Unoise(n_min=-1.5, n_max=1.5))
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True




    @configclass
    class CriticCfg(ObsGroup):
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        # Replaced with privileged observations without noise below
        # joint_pos = ObsTerm(
        #     func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        # )
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        # IMU observations
        # TODO : check imu_projected_gravity return type issue
        imu_projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        # Privileged Critic Observations
        # Joint dynamics information (privileged)
        joint_torques = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )

        # TODO : check body_incoming_wrench return type issue 
        # Contact forces on feet (privileged foot contact information)
        # feet_contact_forces = ObsTerm(
        #     func=mdp.body_incoming_wrench,
        #     scale=0.01,
        #     params={
        #         "asset_cfg": SceneEntityCfg(
        #             "robot", body_names=["left_foot_1", "right_foot_1"]
        #         )
        #     },
        # )

        # Body poses for important body parts (privileged state info)
        body_poses = ObsTerm(
            func=mdp.body_pose_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["base_link", "Link_Ankle_Roll_Left","Link_Ankle_Roll_Right"],
                )
            },
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )

        # Joint positions and velocities with less noise (privileged accurate state)
        joint_pos_accurate = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )
        joint_vel_accurate = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )

        # Base position (full pose information - privileged)
        base_pos = ObsTerm(
            func=mdp.base_pos_z, noise=Unoise(n_min=-0.0001, n_max=0.0001)
        )

        # Root state information (privileged)
        root_lin_vel_w = ObsTerm(
            func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001)
        )
        root_ang_vel_w = ObsTerm(
            func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001)
        )

        # No noise for the critic
        def __post_init__(self):
            self.enable_corruption = False



    
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()




@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


    # Joint position 랜덤화
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ============================================================
    # Mass Randomization
    # ============================================================

    # Base mass 랜덤화
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),  # ±5kg
            "operation": "add",
        },
    )

    # 다리 링크 mass 랜덤화
    randomize_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Link_(Hip|Knee|Ankle).*"),
            "mass_distribution_params": (0.8, 1.2),  # ±20%
            "operation": "scale",
        },
    )

    # ============================================================
    # Physics Randomization
    # ============================================================

    # # 외부 힘 (학습 중 계속 적용)
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 2.0),  # 5~10초마다
    #     params={
    #         "velocity_range": {
    #             "x": (-0.5, 0.5),
    #             "y": (-0.5, 0.5),
    #             "yaw": (-0.5, 0.5),
    #         },
    #     },
    # )

    # # 또는 외부 힘/토크 (더 물리적)
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(2.0, 4.0),  # 2~4초마다
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="Link_Torso"),
    #         "force_range": (-50.0, 50.0),   # ±50N
    #         "torque_range": (-10.0, 10.0),  # ±10Nm
    #     },
    # )

    # ============================================================
    # Actuator Randomization (모터 성능 변화)
    # ============================================================

    # Joint stiffness 랜덤화
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "stiffness_distribution_params": (0.8, 1.2),  # ±20%
            "damping_distribution_params": (0.8, 1.2),    # ±20%
            "operation": "scale",
        },
    )

    # ============================================================
    # Friction Randomization (지면 마찰)
    # ============================================================

    # 발바닥 마찰 계수 랜덤화
    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Link_Ankle_Roll_(Left|Right)"),
            "static_friction_range": (0.6, 1.2),   # 기본 0.8 → 0.6~1.2
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.1),       # 탄성 계수
            "num_buckets": 64,
        },
    )

    # ============================================================
    # Joint Friction/Damping Randomization
    # ============================================================

    # 관절 마찰 랜덤화
    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "friction_distribution_params": (0.0, 0.2),    # 관절 마찰
            "armature_distribution_params": (0.8, 1.2),    # 관절 armature
            "lower_limit_distribution_params": (0.0, 0.0), # 관절 한계는 고정
            "upper_limit_distribution_params": (0.0, 0.0),
            "operation": "add",
        },
    )

    # ============================================================
    # Gravity Randomization (선택사항 - 매우 강력)
    # ============================================================

    # # 중력 방향 약간 변화 (언덕 시뮬레이션)
    # randomize_gravity = EventTerm(
    #     func=mdp.randomize_physics_scene_gravity,
    #     mode="interval",
    #     interval_range_s=(10.0, 20.0),  # 10~20초마다
    #     params={
    #         "gravity_distribution_params": ([0.0, 0.0, -9.81], [0.2, 0.2, 0.5]),  # 약간의 변화
    #         "operation": "add",
    #     },
    # )





ANKLE_JOINTS = ["Joint_Ankle_Pitch_(Left|Right)", "Joint_Ankle_Roll_(Left|Right)"]
HIP_JOINTS   = ["Joint_Hip_Yaw_(Left|Right)", "Joint_Hip_Roll_(Left|Right)", "Joint_Hip_Pitch_(Left|Right)"]
KNEE_JOINTS  = ["Joint_Knee_Pitch_(Left|Right)"]

FOOT_BODIES = "Link_Ankle_Roll_(Left|Right)"   
FOOT_JOINTS = ["Joint_Ankle_Roll_(Left|Right)",
               "Joint_Ankle_Pitch_(Left|Right)",
]

BASE_CMD = "base_velocity"




@configclass
class RewardsCfg:
    # -- task
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
        },
    )


    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Joint_Hip_Roll_(Left|Right)",
                    "Joint_Hip_Pitch_(Left|Right)",
                    "Joint_Hip_Yaw_(Left|Right)",
                    "Joint_Knee_Pitch_(Left|Right)",
                ]
                
            )
        },
    )

    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7,
            params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Joint_Hip_Roll_(Left|Right)",
                    "Joint_Hip_Pitch_(Left|Right)",
                    "Joint_Hip_Yaw_(Left|Right)",
                    "Joint_Knee_Pitch_(Left|Right)",
                ]            
            )
        },
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)


    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=FOOT_JOINTS)},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Joint_Hip_Yaw_(Left|Right)",
                    "Joint_Hip_Roll_(Left|Right)"
                ]
            )
        },
    )

    joint_deviation_hip_pitch_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Joint_Hip_Pitch_(Left|Right)",
                    "Joint_Knee_Pitch_(Left|Right)",
                ],
            ),
        },
    )

    joint_deviation_ankles = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=FOOT_JOINTS
            ),
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)







@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )



    base_height_low = DoneTerm(
        func=terminations.base_height_below_threshold,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.50,   # <- 네가 말한 값. 나중에 0.35~0.45로 조절할 수도 있음
        },
    )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        super().__post_init__()
        self.scene.robot = IGRIS_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            update_period=0.0,
            debug_vis=True,
            gravity_bias=(0.0, 0.0, 0.0),
            offset=ImuCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)  # meters, quaternion
            ),
        )



        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

