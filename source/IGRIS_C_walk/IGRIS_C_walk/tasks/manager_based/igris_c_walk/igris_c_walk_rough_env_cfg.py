# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns



from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR



from . import mdp
from IGRIS_C_walk.tasks.manager_based.igris_c_walk.mdp import terminations
from .LocomotionEnv_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
from IGRIS_C_walk.tasks.manager_based.igris_c_walk.igris_c_isaaclab_cfg import IGRIS_C_CFG

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
FOOT_BODIES = "Link_Ankle_Roll_(Left|Right)"

FOOT_JOINTS = ["Joint_Ankle_Roll_(Left|Right)",
               "Joint_Ankle_Pitch_(Left|Right)",
]
               



@configclass
class IGRISRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    #rewards: IGRISRewardsCfg = IGRISRewardsCfg()
    #rewards: RewardsCfg = RewardsCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        #self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.height_scanner.prim_path = "/World/envs/env_.*/Robot"
        # # Randomization
        # self.events.push_robot = None
        # self.events.add_base_mass = None
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        # self.events.reset_base.params = {
        #     "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        #     "velocity_range": {
        #         "x": (0.0, 0.0),
        #         "y": (0.0, 0.0),
        #         "z": (0.0, 0.0),
        #         "roll": (0.0, 0.0),
        #         "pitch": (0.0, 0.0),
        #         "yaw": (0.0, 0.0),
        #     },
        # }
        # self.events.base_com = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # # terminations
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class IGRISRoughEnvCfg_PLAY(IGRISRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
