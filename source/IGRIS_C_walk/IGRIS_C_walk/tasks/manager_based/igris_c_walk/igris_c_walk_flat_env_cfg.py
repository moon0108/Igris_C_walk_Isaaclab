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
from .igris_c_walk_rough_env_cfg import IGRISRoughEnvCfg




# 발 링크가 따로 없어서 ankle roll 링크를 발로 취급
FOOT_BODIES_REGEX = "Link_Ankle_Roll_(Left|Right)"

@configclass
class IGRISFlatEnvCfg(IGRISRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        #self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


        # Rewards
        # self.rewards.track_lin_vel_xy_exp.weight = 6
        # self.rewards.track_ang_vel_z_exp.weight = 2
        # self.rewards.feet_air_time.weight = 3
        # self.rewards.feet_slide.weight = -3

        # self.rewards.joint_deviation_hip_yaw_roll.weight = -0.3
        # self.rewards.ankle_torque_penalty.weight = -5e-6
        # self.rewards.dof_pos_limits.weight = -0.5

        # self.rewards.ankle_orientation_flat.weight = -0.4
        # self.rewards.flat_orientation_l2.weight = -1.5

        # self.rewards.action_rate_l2.weight = -0.02
        # self.rewards.joint_acc_l2.weight = -1e-5
        # self.rewards.dof_torques_l2.weight = -1e-5
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.2
        self.rewards.dof_acc_l2.weight = -0.005
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.flat_orientation_l2.weight = -1.0

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class IGRISFlatEnvCfg_PLAY(IGRISFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
