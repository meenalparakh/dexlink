from os.path import join

from dexlink.utils.retargeting.path_utils import get_external_data_dir

inspire_asset_dir = "inspire_hand/"
left_rel_urdf = join(inspire_asset_dir, "left_xyz_copy_abs_obj_kpts.urdf")
right_rel_urdf = join(inspire_asset_dir, "right_xyz_copy_abs_obj_kpts.urdf")

MIMIC_JOINTS = {
    # "TH_J3": ("TH_J2", 1.0),
    # "TH_J4": ("TH_J2", 1.13),
    # "FF_J2": ("FF_J1", 1.13),
    # "MF_J2": ("MF_J1", 1.13),
    # "RF_J2": ("RF_J1", 1.08),
    # "LF_J2": ("LF_J1", 1.13),
}
INSPIRE_LEFT_MIMIC_JOINTS = dict()
for name, (parent, ratio) in MIMIC_JOINTS.items():
    INSPIRE_LEFT_MIMIC_JOINTS[f"L_{name}"] = (f"L_{parent}", ratio)
INSPIRE_RIGHT_MIMIC_JOINTS = dict()
for name, (parent, ratio) in MIMIC_JOINTS.items():
    INSPIRE_RIGHT_MIMIC_JOINTS[f"R_{name}"] = (f"R_{parent}", ratio)

INSPIRE_LINK_NAMES = ["TH_Tip", "FF_Tip", "MF_Tip", "RF_Tip", "LF_Tip"]
INSPIRE_DEFAULT_QPOS = {
    "left": [
        0.25,
        0.11,
        1.0,
        0.76,
        0.18,
        0.58,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "right": [
        -0.35,
        0.1,
        1.05,
        -0.37,
        0.23,
        2.64,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
}

ALL_TACTILE_LINKS_LEFT = [
    "base_link",
    # "L_TH_L2",
    "L_TH_L3",
    "L_TH_L4",
    # "L_FF_L1",
    "L_FF_L2",
    # "L_MF_L1",
    "L_MF_L2",
    # "L_RF_L1",
    "L_RF_L2",
    # "L_LF_L1",
    "L_LF_L2",
]
ALL_TACTILE_LINKS_RIGHT = [
    "base_link",
    *(["R" + name[1:] for name in ALL_TACTILE_LINKS_LEFT[1:]]),
]

POSITION_CONTROLLED_JOINTS = [
    "forearm_tx_link_joint",
    "forearm_ty_link_joint",
    "forearm_tz_link_joint",
    "forearm_roll_link_joint",
    "forearm_pitch_link_joint",
    "forearm_yaw_link_joint",
    # thumb joints
    "TH_J1",
    "TH_J2",
    # finger proximal joints
    "FF_J1",
    "MF_J1",
    "RF_J1",
    "LF_J1",
]

VELOCITY_CONTROLLED_JOINTS = [
    "TH_J3",
    "TH_J4",
    # finger joints
    "FF_J2",
    "RF_J2",
    "MF_J2",
    "LF_J2",
]

FINGER_JOINT_MAPS = {
    "wrist": [
        "forearm_tx_link_joint",
        "forearm_ty_link_joint",
        "forearm_tz_link_joint",
        "forearm_roll_link_joint",
        "forearm_pitch_link_joint",
        "forearm_yaw_link_joint",
    ],
    "thumb": [
        "TH_J1",
        "TH_J2",
        "TH_J3",
        "TH_J4",
    ],
    "index": [
        "FF_J1",
        "FF_J2",
    ],
    "mid": [
        "MF_J1",
        "MF_J2",
    ],
    "ring": [
        "RF_J1",
        "RF_J2",
    ],
    "pinky": [
        "LF_J1",
        "LF_J2",
    ],
}

FINGER_JOINT_MAPS_LEFT = {
    finger: [f"L_{joint}" for joint in joints]
    for finger, joints in FINGER_JOINT_MAPS.items()
}

FINGER_JOINT_MAPS_RIGHT = {
    finger: [f"R_{joint}" for joint in joints]
    for finger, joints in FINGER_JOINT_MAPS.items()
}

OPEN_CONFIG = {
    joint: 0.0
    for joint in (
        "TH_J1",
        "TH_J2",
        "TH_J3",
        "TH_J4",
        "FF_J1",
        "FF_J2",
        "MF_J1",
        "MF_J2",
        "RF_J1",
        "RF_J2",
        "LF_J1",
        "LF_J2",
    )
}

HUMAN_ROBOT_LINK_MAP = {
    "wrist": "base_link",
    "thumb_tip": "{side_acro_cap}_TH_Tip",
    "index_tip": "{side_acro_cap}_FF_Tip",
    "middle_tip": "{side_acro_cap}_MF_Tip",
    "ring_tip": "{side_acro_cap}_RF_Tip",
    "little_tip": "{side_acro_cap}_LF_Tip",
    "thumb_base": "{side_acro_cap}_TH_L1",
    "index_base": "{side_acro_cap}_FF_L1",
    "middle_base": "{side_acro_cap}_MF_L1",
    "ring_base": "{side_acro_cap}_RF_L1",
    "little_base": "{side_acro_cap}_LF_L1",
    "thumb_proximal": "{side_acro_cap}_TH_L2",
    "index_proximal": "{side_acro_cap}_FF_L2",
    "middle_proximal": "{side_acro_cap}_MF_L2",
    "ring_proximal": "{side_acro_cap}_RF_L2",
    "little_proximal": "{side_acro_cap}_LF_L2",
}

HUMAN_ROBOT_KPT_MAP = {
    "wrist": "base_link",
    "thumb_tip": "{side_acro_cap}_TH_Tip",
    "index_tip": "{side_acro_cap}_FF_Tip",
    "middle_tip": "{side_acro_cap}_MF_Tip",
    "ring_tip": "{side_acro_cap}_RF_Tip",
    "little_tip": "{side_acro_cap}_LF_Tip",
    "thumb_distal_kpt1": "distal_{side_acro_cap}_TH_Tip_kpt1",
    "thumb_distal_kpt2": "distal_{side_acro_cap}_TH_Tip_kpt2",
    "index_distal_kpt1": "distal_{side_acro_cap}_FF_Tip_kpt1",
    "index_distal_kpt2": "distal_{side_acro_cap}_FF_Tip_kpt2",
    "middle_distal_kpt1": "distal_{side_acro_cap}_MF_Tip_kpt1",
    "middle_distal_kpt2": "distal_{side_acro_cap}_MF_Tip_kpt2",
    "ring_distal_kpt1": "distal_{side_acro_cap}_RF_Tip_kpt1",
    "ring_distal_kpt2": "distal_{side_acro_cap}_RF_Tip_kpt2",
    "little_distal_kpt1": "distal_{side_acro_cap}_LF_Tip_kpt1",
    "little_distal_kpt2": "distal_{side_acro_cap}_LF_Tip_kpt2",
    "thumb_base": "{side_acro_cap}_TH_L1",
    "index_base": "{side_acro_cap}_FF_L1",
    "middle_base": "{side_acro_cap}_MF_L1",
    "ring_base": "{side_acro_cap}_RF_L1",
    "little_base": "{side_acro_cap}_LF_L1",
    "thumb_proximal": "{side_acro_cap}_TH_L2",
    "index_proximal": "{side_acro_cap}_FF_L2",
    "middle_proximal": "{side_acro_cap}_MF_L2",
    "ring_proximal": "{side_acro_cap}_RF_L2",
    "little_proximal": "{side_acro_cap}_LF_L2",
    "thumb_proximal_kpt1": "proximal_{side_acro_cap}_TH_L3_kpt1",
    "thumb_proximal_kpt2": "proximal_{side_acro_cap}_TH_L3_kpt2",
    "index_proximal_kpt1": "proximal_{side_acro_cap}_FF_L2_kpt1",
    "index_proximal_kpt2": "proximal_{side_acro_cap}_FF_L2_kpt2",
    "middle_proximal_kpt1": "proximal_{side_acro_cap}_MF_L2_kpt1",
    "middle_proximal_kpt2": "proximal_{side_acro_cap}_MF_L2_kpt2",
    "ring_proximal_kpt1": "proximal_{side_acro_cap}_RF_L2_kpt1",
    "ring_proximal_kpt2": "proximal_{side_acro_cap}_RF_L2_kpt2",
    "little_proximal_kpt1": "proximal_{side_acro_cap}_LF_L2_kpt1",
    "little_proximal_kpt2": "proximal_{side_acro_cap}_LF_L2_kpt2",
    "index_palm_kpt1": "wrist_base_{side_acro_cap}_FF_L1_kpt1",
    "middle_palm_kpt1": "wrist_base_{side_acro_cap}_MF_L1_kpt1",
    "little_palm_kpt1": "wrist_base_{side_acro_cap}_LF_L1_kpt1",
}

INSPIRE_LEFT_CFG = {
    "urdf_path": str(get_external_data_dir() / "robots" / left_rel_urdf),
    "wrist_link_name": "base_link",
    "mimic_joint_map": INSPIRE_LEFT_MIMIC_JOINTS.copy(),
    "kpt_link_names": ["L_" + name for name in INSPIRE_LINK_NAMES],
    # NOTE: this gets reset by demo data
    "joint_limits": {
        "L_forearm_tx_link_joint": (-0.05, 0.3),
        "L_forearm_ty_link_joint": (-0.1, 0.2),
        "L_forearm_tz_link_joint": (1.0, 1.3),
        "L_forearm_roll_link_joint": (-0.25, 0.5),
        "L_forearm_pitch_link_joint": (0.2, 0.6),
        "L_forearm_yaw_link_joint": (-0.7, 0.4),
    },
    "default_qpos": INSPIRE_DEFAULT_QPOS["left"],
    "actuators": {
        "finger": dict(
            joint_exprs=[".*J1", ".*J2", ".*J3", ".*J4"],
            kp=10.0,
            kv=1.0,
            force_range=200.0,
        ),
        "wrist_rot": dict(
            joint_exprs=[r"[LR]_forearm_(roll|pitch|yaw)_link_joint"],
            kp=100.0,
            kv=8.0,
            force_range=200.0,
        ),
        "wrist_trans": dict(
            joint_exprs=[r"[LR]_forearm_t[xyz]_link_joint"],
            kp=100.0,
            kv=8.0,
            force_range=200.0,
        ),
    },
    "collision_groups": {7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 1, 23: 1},
    "collision_palm_name": "base_link",
    "all_tactile_links": ALL_TACTILE_LINKS_LEFT,
    "human_robot_link_map": {
        k: v.format(side_acro_cap="L") for k, v in HUMAN_ROBOT_LINK_MAP.items()
    },
    "human_robot_kpt_map": {
        k: v.format(side_acro_cap="L") for k, v in HUMAN_ROBOT_KPT_MAP.items()
    },
    "finger_joint_maps": FINGER_JOINT_MAPS_LEFT,
    "position_controlled_joints": [f"L_{j}" for j in POSITION_CONTROLLED_JOINTS],
    "velocity_controlled_joints": [f"L_{j}" for j in VELOCITY_CONTROLLED_JOINTS],
    "open_qpos": {f"L_{joint}": value for joint, value in OPEN_CONFIG.items()},
}


INSPIRE_RIGHT_CFG = {
    "urdf_path": str(get_external_data_dir() / "robots" / right_rel_urdf),
    "wrist_link_name": "base_link",
    "mimic_joint_map": INSPIRE_RIGHT_MIMIC_JOINTS.copy(),
    "kpt_link_names": ["R_" + name for name in INSPIRE_LINK_NAMES],
    # NOTE: this gets reset by demo data
    "joint_limits": {
        "R_forearm_tx_link_joint": (-0.4, -0.1),
        "R_forearm_ty_link_joint": (-0.1, 0.2),
        "R_forearm_tz_link_joint": (1.0, 1.3),
        "R_forearm_roll_link_joint": (-0.4, 0.2),
        "R_forearm_pitch_link_joint": (0.0, 0.6),
        "R_forearm_yaw_link_joint": (2.5, 3.1),
    },
    "default_qpos": INSPIRE_DEFAULT_QPOS["right"],
    "actuators": INSPIRE_LEFT_CFG["actuators"].copy(),
    "collision_groups": INSPIRE_LEFT_CFG["collision_groups"].copy(),
    "collision_palm_name": INSPIRE_LEFT_CFG["collision_palm_name"],
    "all_tactile_links": ALL_TACTILE_LINKS_RIGHT,
    "human_robot_link_map": {
        k: v.format(side_acro_cap="R") for k, v in HUMAN_ROBOT_LINK_MAP.items()
    },
    "human_robot_kpt_map": {
        k: v.format(side_acro_cap="R") for k, v in HUMAN_ROBOT_KPT_MAP.items()
    },
    "finger_joint_maps": FINGER_JOINT_MAPS_RIGHT,
    "position_controlled_joints": [f"R_{j}" for j in POSITION_CONTROLLED_JOINTS],
    "velocity_controlled_joints": [f"R_{j}" for j in VELOCITY_CONTROLLED_JOINTS],
    "open_qpos": {f"R_{joint}": value for joint, value in OPEN_CONFIG.items()},
}

INSPIRE_CFGs = dict(left=INSPIRE_LEFT_CFG, right=INSPIRE_RIGHT_CFG)
