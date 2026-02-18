# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import ImuCfg
from . import mdp
from IGRIS_C_walk.tasks.manager_based.igris_c_walk.mdp import terminations


try:
    from isaaclab.actuators import ImplicitActuatorCfg
except Exception:
    from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


##
# Scene definition
##



LEG_JOINTS = [
    #"Joint_Waist_Yaw",
    #"Joint_Waist_Roll",
    "Joint_Waist_Pitch",
    "Joint_Hip_Yaw_Left",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Pitch_Left",
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
FOOT_LINKS = "Link_Ankle_Roll_(Left|Right)"
FOOT_ROLL_LINKS = ["Link_Ankle_Roll_Left", "Link_Ankle_Roll_Right"]

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



@configclass
class IgrisCWalkSceneCfg(InteractiveSceneCfg):
    """Flat ground + IGRIS-C robot."""

    # 평지(plane)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )


    # robot
    #robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/moon/isaaclab/assets/igris/igris_c_v2.usd",  # <- 네 USD 경로
            activate_contact_sensors=True,
        ),


        actuators={
            "legs" : ImplicitActuatorCfg(
                    joint_names_expr=LEG_JOINTS,
                    stiffness={
                        "Joint_Hip_Yaw_(Left|Right)" : 400,
                        "Joint_Hip_Roll_(Left|Right)" : 400,
                        "Joint_Hip_Pitch_(Left|Right)" : 500,
                        "Joint_Knee_Pitch_(Left|Right)" : 500,
                        "Joint_Ankle_Pitch_(Left|Right)" : 100,
                        "Joint_Ankle_Roll_(Left|Right)" : 100,
                    },
                    damping={
                        "Joint_Hip_Yaw_(Left|Right)" : 32.0,
                        "Joint_Hip_Roll_(Left|Right)" : 32.0,
                        "Joint_Hip_Pitch_(Left|Right)" : 40.0,
                        "Joint_Knee_Pitch_(Left|Right)" : 40,
                        "Joint_Ankle_Pitch_(Left|Right)" : 2, #8
                        "Joint_Ankle_Roll_(Left|Right)" : 2,#0.8
                    },

                    effort_limit={
                        "Joint_Hip_Yaw_(Left|Right)" : 200,
                        "Joint_Hip_Roll_(Left|Right)" : 200, #200
                        "Joint_Hip_Pitch_(Left|Right)" : 200,
                        "Joint_Knee_Pitch_(Left|Right)" : 400, #250
                        "Joint_Ankle_Pitch_(Left|Right)" : 50, #120,
                        "Joint_Ankle_Roll_(Left|Right)" : 50, #30,
                    },
                    armature=0.01,
            ),

        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 접촉 센서 (발/베이스 접촉 보상/종료에 사용)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


    # 발 전체 접촉 센서 (발가락, 발바닥, 뒤꿈치)
    foot_contact_front = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Link_Ankle_Pitch_.*",  # 발 앞부분
        history_length=3,
        track_air_time=False,
    )

    foot_contact_back = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Link_Ankle_Roll_.*",   # 발 뒷부분
        history_length=3,
        track_air_time=False,
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 3.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.6, 1.0),      # flat에서는 앞으로 걷기 위주로
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
        ),
    )

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.25,
        use_default_offset=True,
    )


# @configclass
# class ObservationsCfg:
#     @configclass
#     class PolicyCfg(ObsGroup):
#         base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
#         base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
#         projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
#         velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

#         # 다리 관절만 관측
#         joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)})
#         actions = ObsTerm(func=mdp.last_action)

#         def __post_init__(self):
#             self.enable_corruption = True
#             self.concatenate_terms = True

#     policy: PolicyCfg = PolicyCfg()




@configclass
class ObservationsCfg:
    @configclass
    class CriticCfg(ObservationGroupCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        actions = ObsTerm(func=mdp.last_action)

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

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

        joint_torques = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )

        # body_poses: torso_dummy_1 같은거 제거하고, 확실히 있는 것만 사용
        body_poses = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link"] + FOOT_ROLL_LINKS)},
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )

        joint_pos_accurate = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.0001, n_max=0.0001))
        joint_vel_accurate = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.0001, n_max=0.0001))

        base_pos = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.0001, n_max=0.0001))
        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001))
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001))

        def __post_init__(self):
            self.enable_corruption = False

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # IMPORTANT: restrict to your 12 joints so joint_pos_rel/joint_vel_rel doesn't look for adult joints
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )

        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Configuration for randomization events."""

    # ============================================================
    # Reset Events (Episode 시작 시)
    # ============================================================

    # Base position/orientation 랜덤화
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),  # 높이는 고정
                "roll": (-0.1, 0.1),    # 약간의 roll
                "pitch": (-0.1, 0.1),   # 약간의 pitch
                "yaw": (-3.14, 3.14),   # 전방향
            },
            "velocity_range": {
                "x": (0.0, 0.5),      # 초기 전진 속도
                "y": (-0.2, 0.2),     # 약간의 측면 속도
                "z": (0.0, 0.0),      # 수직 속도 0
                "roll": (-0.1, 0.1),  # 약간의 회전
                "pitch": (-0.1, 0.1),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # Joint position 랜덤화
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.7, 1.3),   # 0.8~1.2 → 0.7~1.3 (더 넓게)
            "velocity_range": (-0.5, 0.5),  # 초기 관절 속도 추가
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





@configclass
class RewardsCfg:
    """
    Reward configuration for IGRIS_C humanoid robot

    Hardware specs:
    - Height: 1.5m
    - Mass: 58kg
    - Strong hip/knee (kp=600-1300), weak ankles (kp=2.5-71)
    - Strategy: Hip/Knee dominant locomotion
    """

    # ========================================================================
    # 1. Safety & Termination (최우선!)
    # ========================================================================
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0
    )

    # ========================================================================
    # 2. Task Achievement (목표 달성)
    # ========================================================================
    # G1 스타일: 낮은 weight로 명확한 신호만

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=5, 
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=5.0,  
        params={"command_name": "base_velocity", "std": 0.5}
    )

    # 3-1. 교대 보행 (G1보다 약간 강조)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link_Ankle_Roll_(Left|Right)"),
            "threshold": 0.35,
        },
    )

    # 3-2. 착지 안정성
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link_Ankle_Roll_(Left|Right)"),
            "asset_cfg": SceneEntityCfg("robot", body_names="Link_Ankle_Roll_(Left|Right)"),
        },
    )

    # ========================================================================
    # 4. Joint Control Strategy (하드웨어 특성 반영!)
    # ========================================================================





    # ankle_torque_penalty = RewTerm(
    #     func=mdp.joint_torques_l2,
    #     weight=-5e-6,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=["Joint_Ankle_Pitch_(Left|Right)", "Joint_Ankle_Roll_(Left|Right)"]
    #         )
    #     },
    # )

    dof_pos_limits_ankle = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,  
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Joint_Ankle_Pitch_(Left|Right)", "Joint_Ankle_Roll_(Left|Right)"]
            )
        },
    )

    # 4-6. 발목 평평하게 유지
    ankle_orientation_flat = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Joint_Ankle_Pitch_(Left|Right)", "Joint_Ankle_Roll_(Left|Right)"]
            ),
        },
    )


    # Hip Yaw/Roll: 외회전 방지
    joint_deviation_hip_yaw_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,  
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Joint_Hip_Yaw_(Left|Right)", "Joint_Hip_Roll_(Left|Right)"]
            )
        },
    )
    
    #pitch 방향 과도하게 금지
    joint_deviation_hip_pitch_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
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

    # 5-1. 자세 안정성 - G1보다 2.5배 강함
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,  # 🔥 G1(-1.0)의 2.5배 - 무겁고 커서 중요!
    )

    # 6-1. Action rate (급격한 변화 억제)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,  # G1(-0.005)의 2배 - 관성 고려
    )

    # 6-2. Joint acceleration
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-7,  # G1(-1.25e-7)의 2배
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
                                                                "Joint_Waist_Pitch",
                                                                "Joint_Hip_Yaw_Left",
                                                                "Joint_Hip_Roll_Left",
                                                                "Joint_Hip_Pitch_Left",
                                                                "Joint_Knee_Pitch_Left",
                                                                "Joint_Hip_Yaw_Right",
                                                                "Joint_Hip_Roll_Right",
                                                                "Joint_Hip_Pitch_Right",
                                                                "Joint_Knee_Pitch_Right",]
                                                        )},
    )

    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-5,  # G1(-1.5e-7)의 2배
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=LEG_JOINTS
            )
        },
    )




    foot_impact_penalty = RewTerm(
        func=mdp.contact_forces,
        weight=-1.5e-3,
        params={
            "threshold": 500.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_ROLL_LINKS),
        },
    )








@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # base_link가 바닥에 닿으면 종료
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



##
# Environment configuration
##


@configclass
class IgrisCWalkEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: IgrisCWalkSceneCfg = IgrisCWalkSceneCfg(num_envs=2048, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation



        self.scene.height_scanner = None
        # policy group in our igris cfg doesn't have height_scan; critic does.
        self.observations.critic.height_scan = None


        # IMU
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            update_period=0.0,
            debug_vis=True,
            gravity_bias=(0.0, 0.0, 0.0),
            offset=ImuCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )


        # 초기 무릎 각도를 강제로 설정
        #domain randomization이 안될 수 있음.
        self.scene.robot.init_state.joint_pos = {
            "Joint_Knee_Pitch_.*": 0.5,  # 무릎을 -0.5 rad (약 -28도)로
            "Joint_Hip_Pitch_.*": -0.2,   # 고관절도 약간 앞으로
            "Joint_Ankle_Pitch_.*": -0.2,  # 발목으로 균형
        }

class IgrisCWalkEnvCfg_PLAY(IgrisCWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.reset_base = None

