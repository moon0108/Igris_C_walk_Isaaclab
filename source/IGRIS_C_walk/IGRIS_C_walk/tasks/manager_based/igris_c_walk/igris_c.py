# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""IGRIS-C (legs-only) robot description in the same style as the provided dummy robot file."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DelayedPDActuatorCfg


try:
    from isaaclab.actuators import ImplicitActuatorCfg
except Exception:
    from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg


# -----------------------------------------------------------------------------
# Asset
# -----------------------------------------------------------------------------
USD_PATH = "/home/moon/isaaclab/assets/igris_c_v2.usd"

# -----------------------------------------------------------------------------
# Initial joint pose (radians)
# NOTE: These keys must match the USD joint names exactly.
# -----------------------------------------------------------------------------
_INIT_JOINT_POS = {
    "Joint_Hip_Pitch_Left":  -0.2,
    "Joint_Hip_Roll_Left":    0.0,
    "Joint_Hip_Yaw_Left":     0.0,
    "Joint_Knee_Pitch_Left":  0.5,
    "Joint_Ankle_Pitch_Left": -0.2,
    "Joint_Ankle_Roll_Left":  0.0,

    "Joint_Hip_Pitch_Right":  -0.2,
    "Joint_Hip_Roll_Right":    0.0,
    "Joint_Hip_Yaw_Right":     0.0,
    "Joint_Knee_Pitch_Right":  0.5,
    "Joint_Ankle_Pitch_Right": -0.2,
    "Joint_Ankle_Roll_Right":  0.0,
}

# -----------------------------------------------------------------------------
# Joint metadata (PD gains + limits)
# PD defaults from your IGRIS-C SDK snippet:
#   Hip_Pitch kp=500 kd=3.0
#   Hip_Roll  kp=200 kd=0.5
#   Hip_Yaw   kp=50  kd=0.5
#   Knee      kp=500 kd=3.0
#   Ankle     kp=300 kd=1.5
#
# Torque limits: your chosen per-joint caps (Nm)
# Velocity limits: kept high unless you have measured specs.
# Armature: left as None (use USD-authored inertia/armature), adjust if needed.
# -----------------------------------------------------------------------------
armma = 0.01
_JOINT_META = {
    # ────────────── LEFT LEG ──────────────
    "Joint_Hip_Pitch_Left": {
        "kp": 500.0, "kd": 40, "torque": 200.0, "vmax": 10000.0, "arm": armma,
    },
    "Joint_Hip_Roll_Left": {
        "kp": 400.0, "kd": 32, "torque": 200.0, "vmax": 10000.0, "arm": armma,
    },
    "Joint_Hip_Yaw_Left": {
        "kp": 400.0, "kd": 32, "torque": 200.0, "vmax": 10000.0, "arm": armma,
    },
    "Joint_Knee_Pitch_Left": {
        "kp": 500.0, "kd": 40, "torque": 400.0, "vmax": 10000.0, "arm": armma,
    },
    "Joint_Ankle_Pitch_Left": {
        "kp": 50.0, "kd": 2, "torque": 40.0, "vmax": 10000.0, "arm": armma,
    },
    "Joint_Ankle_Roll_Left": {
        "kp": 50.0, "kd": 2, "torque": 40.0, "vmax": 10000.0, "arm": armma,
    },

    # ────────────── RIGHT LEG ──────────────
    "Joint_Hip_Pitch_Right": {
        "kp": 500.0, "kd": 40, "torque": 200.0, "vmax": 10000.0, "arm": armma,
    },
    "Joint_Hip_Roll_Right": {
        "kp": 400.0, "kd": 40, "torque": 200.0, "vmax": 10000.0, "arm": armma,
    },
    "Joint_Hip_Yaw_Right": {
        "kp": 400.0, "kd": 40, "torque": 200.0, "vmax": 10000.0, "arm": armma,
    },
    "Joint_Knee_Pitch_Right": {
        "kp": 500.0, "kd": 40, "torque": 400.0, "vmax": 10000.0, "arm": armma,
    },
    "Joint_Ankle_Pitch_Right": {
        "kp": 50.0, "kd": 2, "torque": 40.0, "vmax": 10000.0, "arm": armma,
    },
    "Joint_Ankle_Roll_Right": {
        "kp": 50.0, "kd": 2, "torque": 40.0, "vmax": 10000.0, "arm": armma,
    },
}
KP_SCALE = 1.0
KD_SCALE = 1.0
TORQUE_SCALE = 1.0
# -----------------------------------------------------------------------------
# Actuators: one DelayedPDActuatorCfg per joint (same style as your dummy file)
# If sim dt is 0.005 seconds and decimation=4 -> control_dt = 0.02s.
# max_delay=4 => up to 80ms delay (4 * 20ms). Tune as needed.
# -----------------------------------------------------------------------------
_ACTUATORS = {
    jn: ImplicitActuatorCfg(
        joint_names_expr=[jn],
        effort_limit=meta["torque"] * TORQUE_SCALE,
        velocity_limit=meta["vmax"],
        stiffness={jn: meta["kp"] * KP_SCALE},
        damping={jn: meta["kd"] * KD_SCALE},
        armature=meta["arm"],
        #min_delay=0,
        #max_delay=4,
    )
    for jn, meta in _JOINT_META.items()
}
# -----------------------------------------------------------------------------
# Articulation config
# -----------------------------------------------------------------------------
igris_c_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10000.0,
            max_angular_velocity=10000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos=_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators=_ACTUATORS,
)
