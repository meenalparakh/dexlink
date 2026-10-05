from os.path import join

import numpy as np

from dexlink.utils.retargeting.path_utils import get_assets_dir

xhand_asset_dir = "xhand/"
left_rel_urdf = join(xhand_asset_dir, "xhand_left_6dof_kpts_abs_obj.urdf")
right_rel_urdf = join(xhand_asset_dir, "xhand_right_6dof_kpts_abs_obj.urdf")

# defined joints

ALL_TACTILE_LINKS = [
    # "hand_link",
    # "hand_thumb_rota_link1",
    "hand_thumb_rota_link2",
    "hand_index_rota_link2",
    "hand_mid_link2",
    "hand_ring_link2",
    "hand_pinky_link2",
]

# flat hand configuration for finger joints (excluding the wrist joints)
OPEN_CONFIG = {
    "hand_thumb_bend_joint": 0.0,
    "hand_thumb_rota_joint1": 0.0,
    "hand_thumb_rota_joint2": 0.0,
    "hand_index_bend_joint": 0.0,
    "hand_index_joint1": 0.0,
    "hand_index_joint2": 0.0,
    "hand_mid_joint1": 0.0,
    "hand_mid_joint2": 0.0,
    "hand_ring_joint1": 0.0,
    "hand_ring_joint2": 0.0,
    "hand_pinky_joint1": 0.0,
    "hand_pinky_joint2": 0.0,
}

ALL_TACTILE_LINKS_LEFT = ["left_" + link for link in ALL_TACTILE_LINKS]
ALL_TACTILE_LINKS_RIGHT = ["right_" + link for link in ALL_TACTILE_LINKS]

POSITION_CONTROLLED_JOINTS = [
    "forearm_tx_link_joint",
    "forearm_ty_link_joint",
    "forearm_tz_link_joint",
    "forearm_roll_link_joint",
    "forearm_pitch_link_joint",
    "forearm_yaw_link_joint",
    # thumb joints
    "hand_thumb_bend_joint",
    "hand_thumb_rota_joint1",
    # index joints
    "hand_index_bend_joint",
    "hand_index_joint1",
    # other fingers
    "hand_mid_joint1",
    "hand_ring_joint1",
    "hand_pinky_joint1",
]

VELOCITY_CONTROLLED_JOINTS = [
    "hand_thumb_rota_joint2",
    # finger joints
    "hand_index_joint2",
    "hand_mid_joint2",
    "hand_ring_joint2",
    "hand_pinky_joint2",
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
        "hand_thumb_bend_joint",
        "hand_thumb_rota_joint1",
        "hand_thumb_rota_joint2",
    ],
    "index": [
        "hand_index_bend_joint",
        "hand_index_joint1",
        "hand_index_joint2",
    ],
    "mid": [
        "hand_mid_joint1",
        "hand_mid_joint2",
    ],
    "ring": [
        "hand_ring_joint1",
        "hand_ring_joint2",
    ],
    "pinky": [
        "hand_pinky_joint1",
        "hand_pinky_joint2",
    ],
}

FINGER_JOINT_MAPS_LEFT = {
    finger: [
        ("L_" + joint) if joint.startswith("forearm_") else ("left_" + joint)
        for joint in joints
    ]
    for finger, joints in FINGER_JOINT_MAPS.items()
}

FINGER_JOINT_MAPS_RIGHT = {
    finger: [
        ("R_" + joint) if joint.startswith("forearm_") else ("right_" + joint)
        for joint in joints
    ]
    for finger, joints in FINGER_JOINT_MAPS.items()
}

HUMAN_ROBOT_LINK_MAP = {
    "wrist": "{side_acro_cap}_hand_link",
    "thumb_tip": "{side_acro_cap}_hand_thumb_rota_tip",
    "index_tip": "{side_acro_cap}_hand_index_rota_tip",
    "middle_tip": "{side_acro_cap}_hand_mid_tip",
    "ring_tip": "{side_acro_cap}_hand_ring_tip",
    "little_tip": "{side_acro_cap}_hand_pinky_tip",
    "thumb_base": "{side_acro_cap}_hand_thumb_rota_link1",
    "index_base": "{side_acro_cap}_hand_index_rota_link1",
    "middle_base": "{side_acro_cap}_hand_mid_link1",
    "ring_base": "{side_acro_cap}_hand_ring_link1",
    "little_base": "{side_acro_cap}_hand_pinky_link1",
    "thumb_proximal": "{side_acro_cap}_hand_thumb_rota_link2",
    "index_proximal": "{side_acro_cap}_hand_index_rota_link2",
    "middle_proximal": "{side_acro_cap}_hand_mid_link2",
    "ring_proximal": "{side_acro_cap}_hand_ring_link2",
    "little_proximal": "{side_acro_cap}_hand_pinky_link2",
    # "thumb_distal": "{side_acro_cap}_hand_thumb_bend_link",
}


HUMAN_ROBOT_KPT_MAP = {
    # as opposed to the link map (which is mainly used for broad hand cfg),
    # kpt map is used for object-distance based costs, and are more for the
    # parts that interact with the object.
    "wrist": "{side_acro_cap}_hand_link",
    "thumb_tip": "{side_acro_cap}_hand_thumb_rota_tip",
    "index_tip": "{side_acro_cap}_hand_index_rota_tip",
    "middle_tip": "{side_acro_cap}_hand_mid_tip",
    "ring_tip": "{side_acro_cap}_hand_ring_tip",
    "little_tip": "{side_acro_cap}_hand_pinky_tip",
    # 2 kpts each on proximal and distal finger links
    "thumb_distal_kpt1": "distal_{side_acro_cap}_hand_thumb_rota_tip_kpt1",
    "thumb_distal_kpt2": "distal_{side_acro_cap}_hand_thumb_rota_tip_kpt2",
    "index_distal_kpt1": "distal_{side_acro_cap}_hand_index_rota_tip_kpt1",
    "index_distal_kpt2": "distal_{side_acro_cap}_hand_index_rota_tip_kpt2",
    "middle_distal_kpt1": "distal_{side_acro_cap}_hand_mid_tip_kpt1",
    "middle_distal_kpt2": "distal_{side_acro_cap}_hand_mid_tip_kpt2",
    "ring_distal_kpt1": "distal_{side_acro_cap}_hand_ring_tip_kpt1",
    "ring_distal_kpt2": "distal_{side_acro_cap}_hand_ring_tip_kpt2",
    "little_distal_kpt1": "distal_{side_acro_cap}_hand_pinky_tip_kpt1",
    "thumb_base": "{side_acro_cap}_hand_thumb_rota_link1",
    "index_base": "{side_acro_cap}_hand_index_rota_link1",
    "middle_base": "{side_acro_cap}_hand_mid_link1",
    "ring_base": "{side_acro_cap}_hand_ring_link1",
    "little_base": "{side_acro_cap}_hand_pinky_link1",
    "thumb_proximal": "{side_acro_cap}_hand_thumb_rota_link2",
    "index_proximal": "{side_acro_cap}_hand_index_rota_link2",
    "middle_proximal": "{side_acro_cap}_hand_mid_link2",
    "ring_proximal": "{side_acro_cap}_hand_ring_link2",
    "little_proximal": "{side_acro_cap}_hand_pinky_link2",
    "little_distal_kpt2": "distal_{side_acro_cap}_hand_pinky_tip_kpt2",
    "thumb_proximal_kpt1": "proximal_{side_acro_cap}_hand_thumb_rota_link2_kpt1",
    "thumb_proximal_kpt2": "proximal_{side_acro_cap}_hand_thumb_rota_link2_kpt2",
    "index_proximal_kpt1": "proximal_{side_acro_cap}_hand_index_rota_link2_kpt1",
    "index_proximal_kpt2": "proximal_{side_acro_cap}_hand_index_rota_link2_kpt2",
    "middle_proximal_kpt1": "proximal_{side_acro_cap}_hand_mid_link2_kpt1",
    "middle_proximal_kpt2": "proximal_{side_acro_cap}_hand_mid_link2_kpt2",
    "ring_proximal_kpt1": "proximal_{side_acro_cap}_hand_ring_link2_kpt1",
    "ring_proximal_kpt2": "proximal_{side_acro_cap}_hand_ring_link2_kpt2",
    "little_proximal_kpt1": "proximal_{side_acro_cap}_hand_pinky_link2_kpt1",
    "little_proximal_kpt2": "proximal_{side_acro_cap}_hand_pinky_link2_kpt2",
    # 3 on palm.
    "index_palm_kpt1": "wrist_base_{side_acro_cap}_hand_index_bend_link_kpt1",
    "middle_palm_kpt1": "wrist_base_{side_acro_cap}_hand_mid_link1_kpt1",
    "little_palm_kpt1": "wrist_base_{side_acro_cap}_hand_pinky_link1_kpt1",
}


XHAND_LEFT_CFG = {
    "urdf_path": str(get_assets_dir() / "robots" / left_rel_urdf),
    "wrist_link_name": "left_hand_link",
    "kpt_link_names": [
        "left_hand_thumb_rota_tip",
        "left_hand_index_rota_tip",
        "left_hand_mid_tip",
        "left_hand_ring_tip",
        "left_hand_pinky_tip",
    ],
    "actuators": {
        "finger": dict(
            joint_exprs=[".*joint.*"],
            kp=20.0,
            kv=1.5,
            force_range=100.0,
        ),
        "wrist_rot": dict(
            joint_exprs=[r"[LR]_forearm_(roll|pitch|yaw)_link_joint"],
            kp=60.0,
            kv=3.0,
            force_range=100,
        ),
        "wrist_trans": dict(
            joint_exprs=[r"[LR]_forearm_t[xyz]_link_joint"],
            kp=350.0,
            kv=15.0,
            force_range=100,
        ),
    },
    "collision_groups": {7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 1, 19: 2},
    "collision_palm_name": "left_hand_link",
    "all_tactile_links": ALL_TACTILE_LINKS_LEFT,
    "human_robot_link_map": {
        k: v.format(side_acro_cap="left") for k, v in HUMAN_ROBOT_LINK_MAP.items()
    },
    "human_robot_kpt_map": {
        k: v.format(side_acro_cap="left") for k, v in HUMAN_ROBOT_KPT_MAP.items()
    },
    "finger_joint_maps": FINGER_JOINT_MAPS_LEFT,
    "position_controlled_joints": ["left_" + j for j in POSITION_CONTROLLED_JOINTS],
    "velocity_controlled_joints": ["left_" + j for j in VELOCITY_CONTROLLED_JOINTS],
    "open_qpos": {"left_" + joint: value for joint, value in OPEN_CONFIG.items()},
    "open_qpos_hull": {
        **{"left_" + joint: value for joint, value in OPEN_CONFIG.items()},
        "left_hand_thumb_bend_joint": 0.5 * np.pi,
    },
}

XHAND_RIGHT_CFG = {
    "urdf_path": str(get_assets_dir() / "robots" / right_rel_urdf),
    "wrist_link_name": "right_hand_link",
    "kpt_link_names": [
        "right_hand_thumb_rota_tip",
        "right_hand_index_rota_tip",
        "right_hand_mid_tip",
        "right_hand_ring_tip",
        "right_hand_pinky_tip",
    ],
    "actuators": XHAND_LEFT_CFG["actuators"].copy(),
    "collision_groups": {7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 1, 19: 2},
    "collision_palm_name": "right_hand_link",
    "all_tactile_links": ALL_TACTILE_LINKS_RIGHT,
    "human_robot_link_map": {
        k: v.format(side_acro_cap="right") for k, v in HUMAN_ROBOT_LINK_MAP.items()
    },
    "human_robot_kpt_map": {
        k: v.format(side_acro_cap="right") for k, v in HUMAN_ROBOT_KPT_MAP.items()
    },
    "finger_joint_maps": FINGER_JOINT_MAPS_RIGHT,
    "position_controlled_joints": ["right_" + j for j in POSITION_CONTROLLED_JOINTS],
    "velocity_controlled_joints": ["right_" + j for j in VELOCITY_CONTROLLED_JOINTS],
    "open_qpos": {"right_" + joint: value for joint, value in OPEN_CONFIG.items()},
    "open_qpos_hull": {
        **{"right_" + joint: value for joint, value in OPEN_CONFIG.items()},
        "right_hand_thumb_bend_joint": 0.5 * np.pi,
    },
}

XHAND_CFGs = dict(
    left=XHAND_LEFT_CFG,
    right=XHAND_RIGHT_CFG,
)
