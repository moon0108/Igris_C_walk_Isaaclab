# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause
"""
IGRIS_C configuration (auto-generated).

Inputs on real robot: q_d, qd_dot, tau_ff
Motor law: tau = Kp*(q_d - q) + Kd*(qd_dot - q_dot) + tau_ff

This cfg uses Isaac Lab's explicit PD actuator (IdealPDActuatorCfg) to match the same interface.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


# -----------------------------------------------------------------------------
# Joint names extracted from the provided URDF in the same asset bundle:
#   /mnt/data/igris_c_v2_pelvis_with_hands.urdf
#
# Note: The bundled USD is a binary "USD crate". If you want to *verify* the
# runtime DOF names directly from USD, run the debug snippet at the bottom.
# -----------------------------------------------------------------------------

# Body joint order (must match the Kp/Kd arrays below)
BODY_JOINT_ORDER = [
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

BODY_KP = [

    # Left leg (3..8)  (no scale)
    500, 200, 50, 500, 300.0, 300,

    # Right leg (9..14) (no scale)
    500, 200, 50, 500, 300.0, 300,
]

BODY_KD = [

    # Left leg (3..8) (no scale)
    3, 0.5, 0.5, 3.0, 1.5, 1.5,

    # Right leg (9..14) (no scale)
    3, 0.5, 0.5, 3.0, 1.5, 1.5,

]
TORQUE_LIMIT = dict(zip(BODY_JOINT_ORDER, [
    100.0, 70.0, 40.0, 100.0, 70.0, 70.0,
    100.0, 70.0, 40.0, 100.0, 70.0, 70.0,
]))

# Hand joints present in the asset (fingers). Included for completeness.
# HAND_JOINTS = [
#     "Joint_Hand_Left_Thumb_Proximal",
#     "Joint_Hand_Left_Thumb_Middle",
#     "Joint_Hand_Left_Thumb_Distal",
#     "Joint_Hand_Left_Index_Middle",
#     "Joint_Hand_Left_Index_Distal",
#     "Joint_Hand_Left_Middle_Middle",
#     "Joint_Hand_Left_Middle_Distal",
#     "Joint_Hand_Left_Ring_Middle",
#     "Joint_Hand_Left_Ring_Distal",
#     "Joint_Hand_Left_Little_Middle",
#     "Joint_Hand_Left_Little_Distal",
#     "Joint_Hand_Right_Thumb_Proximal",
#     "Joint_Hand_Right_Thumb_Middle",
#     "Joint_Hand_Right_Thumb_Distal",
#     "Joint_Hand_Right_Index_Middle",
#     "Joint_Hand_Right_Index_Distal",
#     "Joint_Hand_Right_Middle_Middle",
#     "Joint_Hand_Right_Middle_Distal",
#     "Joint_Hand_Right_Ring_Middle",
#     "Joint_Hand_Right_Ring_Distal",
#     "Joint_Hand_Right_Little_Middle",
#     "Joint_Hand_Right_Little_Distal"
# ]


assert len(BODY_JOINT_ORDER) == len(BODY_KP) == len(BODY_KD), "Body joint/Kp/Kd length mismatch!"


# Per-joint gain maps (exact joint-name keys)
BODY_STIFFNESS = dict(zip(BODY_JOINT_ORDER, BODY_KP))
BODY_DAMPING = dict(zip(BODY_JOINT_ORDER, BODY_KD))

# Upper-body joints (waist + arms + neck) used by IK
# UPPER_BODY_JOINTS = set(BODY_JOINT_ORDER[:3] + BODY_JOINT_ORDER[15:])

# High-PD scaling (tunable)
HIGH_PD_BODY_STIFFNESS_SCALE = 1.0 #5.0 
HIGH_PD_BODY_DAMPING_SCALE = 1.0 #2.5


def _scale_body_gains(gains: dict[str, float], scale: float) -> dict[str, float]:
    return {name: (value * scale if name in BODY_JOINT_ORDER else value) for name, value in gains.items()}


HIGH_PD_BODY_STIFFNESS = _scale_body_gains(BODY_STIFFNESS, HIGH_PD_BODY_STIFFNESS_SCALE)
HIGH_PD_BODY_DAMPING = _scale_body_gains(BODY_DAMPING, HIGH_PD_BODY_DAMPING_SCALE)

# Temporary hand gains (tunable). Rule-based regex map keeps it robust.
# - thumb: slightly stiffer
# - other fingers: moderate
# HAND_STIFFNESS = {
#     "Joint_Hand_.*Thumb_Proximal": 3000.0,
#     "Joint_Hand_.*Thumb_Middle":   2500.0,
#     "Joint_Hand_.*Thumb_Distal":   2000.0,
#     "Joint_Hand_.*_(Index|Middle|Ring|Little)_Middle": 2500.0,
#     "Joint_Hand_.*_(Index|Middle|Ring|Little)_Distal": 1800.0,
# }
# HAND_DAMPING = {
#     "Joint_Hand_.*Thumb_Proximal": 150.0,
#     "Joint_Hand_.*Thumb_Middle":   120.0,
#     "Joint_Hand_.*Thumb_Distal":   90.0,
#     "Joint_Hand_.*_(Index|Middle|Ring|Little)_Middle": 120.0,
#     "Joint_Hand_.*_(Index|Middle|Ring|Little)_Distal": 80.0,
# }

# -----------------------------------------------------------------------------
# Actuators
# -----------------------------------------------------------------------------

IGRIS_C_ACTUATORS = {
    # Body: exact-name match + per-joint gains
    "body_pd": ImplicitActuatorCfg(
        joint_names_expr=BODY_JOINT_ORDER,
        stiffness=BODY_STIFFNESS,
        damping=BODY_DAMPING,
        # Leave limits to USD (if authored). Override here if you have measured specs.
        armature=None,
        effort_limit_sim=TORQUE_LIMIT,
        velocity_limit_sim=10000.0,
        friction=None,
    ),
    # Hands: regex match + temporary gains
    # "hands_pd": ImplicitActuatorCfg(
    #     joint_names_expr=["Joint_Hand_.*"],
    #     stiffness=HAND_STIFFNESS,
    #     damping=HAND_DAMPING,
    #     effort_limit_sim=10000.0,
    #     velocity_limit_sim=10000.0,
    #     armature=0.01,
    #     friction=None,
    # ),
}

# -----------------------------------------------------------------------------
# Robot articulation
# -----------------------------------------------------------------------------



_INIT_JOINT_POS = {

     "Joint_Hip_Pitch_.*" : -0.2,
     "Joint_Hip_Roll_.*" : 0.0,
     "Joint_Hip_Yaw_.*" : 0.0,
     "Joint_Knee_Pitch_.*" : 0.5, 
     "Joint_Ankle_Pitch_.*" : -0.2,
     "Joint_Ankle_Roll_.*" : 0.0,

}

IGRIS_C_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # IMPORTANT: point this to your actual asset location
        # Example (local): "/absolute/path/to/igris_c_v2_pelvis_with_hands.usd"
        # Example (nucleus): f"{ISAAC_NUCLEUS_DIR}/Robots/IGRIS/igris_c_v2_pelvis_with_hands.usd"
        #usd_path="/home/moon/isaaclab/assets/robot_control/asset/urdf/igris_c_v2_pelvis/igris_c_v2_pelvis.usd",
        usd_path="/home/moon/isaaclab/assets/igris_c_v2.usd",
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
            #fix_root_link=False,  # set True for fixed-base manipulation 이 새기 땜에 로봇 body_link가 고정
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),   #z축 1.05 
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=_INIT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators=IGRIS_C_ACTUATORS,
    #prim_path="/World/envs/env_.*/Robot",
)

IGRIS_C_HIGH_PD_CFG = IGRIS_C_CFG.copy()
IGRIS_C_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
IGRIS_C_HIGH_PD_CFG.actuators["body_pd"].stiffness = HIGH_PD_BODY_STIFFNESS
IGRIS_C_HIGH_PD_CFG.actuators["body_pd"].damping = HIGH_PD_BODY_DAMPING
"""Configuration of IGRIS_C robot with stiffer upper-body PD control.
This is useful for upper-body task-space control using differential IK.
"""


# -----------------------------------------------------------------------------
# Debug: print DOF names at runtime (to validate they match the tokens above)
# -----------------------------------------------------------------------------
# In an Isaac Lab script after spawning the articulation:
#
#   print(robot.data.joint_names)
#   print(len(robot.data.joint_names))
#
# If any name differs, update BODY_JOINT_ORDER and/or HAND regex accordingly.
# -----------------------------------------------------------------------------
# Enforce: BODY_JOINT_ORDER must match the real-robot JointIndex(0..30) order.
# This prevents silent mismatch between real robot controller indexing and sim DOF order.
# -----------------------------------------------------------------------------


REAL_ROBOT_JOINTINDEX_BODY_ORDER = [
    # Waist (JointIndex: 0..2)
    # "Joint_Waist_Yaw",
    # "Joint_Waist_Roll",
    # "Joint_Waist_Pitch",

    # Left leg (JointIndex: 3..8)
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Yaw_Left",
    "Joint_Knee_Pitch_Left",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Roll_Left",

    # Right leg (JointIndex: 9..14)
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Right",

    # # Left arm (JointIndex: 15..21)
    # "Joint_Shoulder_Pitch_Left",
    # "Joint_Shoulder_Roll_Left",
    # "Joint_Shoulder_Yaw_Left",
    # "Joint_Elbow_Pitch_Left",
    # "Joint_Wrist_Yaw_Left",
    # "Joint_Wrist_Roll_Left",
    # "Joint_Wrist_Pitch_Left",

    # # Right arm (JointIndex: 22..28)
    # "Joint_Shoulder_Pitch_Right",
    # "Joint_Shoulder_Roll_Right",
    # "Joint_Shoulder_Yaw_Right",
    # "Joint_Elbow_Pitch_Right",
    # "Joint_Wrist_Yaw_Right",
    # "Joint_Wrist_Roll_Right",
    # "Joint_Wrist_Pitch_Right",

    # # Neck (JointIndex: 29..30)
    # "Joint_Neck_Yaw",
    # "Joint_Neck_Pitch",
]


def _raise_joint_order_mismatch(expected: list[str], actual: list[str]) -> None:
    if len(expected) != len(actual):
        raise ValueError(
            f"[IGRIS_C_CFG] BODY_JOINT_ORDER length mismatch: "
            f"expected {len(expected)} (JointIndex 0..30), got {len(actual)}"
        )
    bad = [i for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
    if not bad:
        return
    i0 = bad[0]
    ctx = slice(max(0, i0 - 3), min(len(expected), i0 + 4))
    raise ValueError(
        "[IGRIS_C_CFG] BODY_JOINT_ORDER must EXACTLY match real-robot JointIndex order.\n"
        f"  First mismatch at index {i0}:\n"
        f"    expected: {expected[i0]}\n"
        f"    actual  : {actual[i0]}\n"
        f"  Context expected[{ctx.start}:{ctx.stop}]: {expected[ctx]}\n"
        f"  Context actual  [{ctx.start}:{ctx.stop}]: {actual[ctx]}\n"
        "Fix: reorder BODY_JOINT_ORDER to match REAL_ROBOT_JOINTINDEX_BODY_ORDER."
    )


# Hard enforcement (do not allow accidental reorder)
_raise_joint_order_mismatch(REAL_ROBOT_JOINTINDEX_BODY_ORDER, BODY_JOINT_ORDER)

# Optional: also enforce group semantics (3/12/14/2)
assert BODY_JOINT_ORDER[:3] == REAL_ROBOT_JOINTINDEX_BODY_ORDER[:3], "[IGRIS_C_CFG] waist slice mismatch"
assert BODY_JOINT_ORDER[3:15] == REAL_ROBOT_JOINTINDEX_BODY_ORDER[3:15], "[IGRIS_C_CFG] leg slice mismatch"
assert BODY_JOINT_ORDER[15:29] == REAL_ROBOT_JOINTINDEX_BODY_ORDER[15:29], "[IGRIS_C_CFG] arm slice mismatch"
assert BODY_JOINT_ORDER[29:31] == REAL_ROBOT_JOINTINDEX_BODY_ORDER[29:31], "[IGRIS_C_CFG] neck slice mismatch"
