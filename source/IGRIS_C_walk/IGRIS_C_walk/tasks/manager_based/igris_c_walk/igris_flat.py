# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat environment config for IGRIS-C (legs-only)."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# import the IGRIS rough env cfg you made (module path = your project)
from .igris_rough import IGRISRoughEnvCfg


@configclass
class IGRISFlatEnvCfg(IGRISRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ---- Terrain: flat plane ----
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ---- No height scan on flat ----
        self.scene.height_scanner = None
        # policy group in our igris cfg doesn't have height_scan; critic does.
        #self.observations.critic.height_scan = None

        # ---- No terrain curriculum ----
        self.curriculum.terrain_levels = None

        # ---- Rewards (tuned like your adultFlat example, but with IGRIS joints) ----
        #self.rewards.track_ang_vel_z_exp.weight = 2.0
        #self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7

        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4

        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                # hips + knees only (8 dof)
                "Joint_Hip_Pitch_Left",
                "Joint_Hip_Roll_Left",
                "Joint_Hip_Yaw_Left",
                "Joint_Knee_Pitch_Left",
                "Joint_Hip_Pitch_Right",
                "Joint_Hip_Roll_Right",
                "Joint_Hip_Yaw_Right",
                "Joint_Knee_Pitch_Right",
            ],
        )

        # ---- Commands ----
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class IGRISFlatEnvCfg_PLAY(IGRISFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # no curriculum for play
        self.curriculum = None

        # smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable observation corruption for play
        self.observations.policy.enable_corruption = False

        # remove random pushing for play
        self.events.base_external_force_torque = None
        self.events.push_robot = None
