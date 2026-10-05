from os.path import join

from dexlink.utils.retargeting.path_utils import get_external_data_dir

# xhand_asset_dir = "franka_xhand_real"
# left_rel_urdf = join(xhand_asset_dir, "franka_panda_xhand_v3_obj.urdf")
# right_rel_urdf = join(xhand_asset_dir, "franka_panda_xhand_v3_obj.urdf")

# !!!!!!!!!!!!!!!!!!!!!!!!! old franka xhand (the mount was not accounted for)
xhand_asset_dir = "frankaX_hand/"
left_rel_urdf = join(xhand_asset_dir, "panda_xhand_obj.urdf")
# left hand does not exist yet. downstream code will not be using it.

right_rel_urdf = join(xhand_asset_dir, "panda_xhand_obj.urdf")

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


POSITION_CONTROLLED_JOINTS_PANDA = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

POSITION_CONTROLLED_JOINTS = [
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


XHAND_LEFT_CFG = {
    "urdf_path": str(get_external_data_dir() / "robots" / left_rel_urdf),
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
        "panda_1": dict(
            joint_exprs=["panda_joint1|panda_joint2"],
            kp=200.0,
            kv=20.0,
            force_range=200.0,
        ),
        "panda_2": dict(
            joint_exprs=["panda_joint3|panda_joint4"],
            kp=200.0,
            kv=20.0,
            force_range=200.0,
        ),
        "panda_3": dict(
            joint_exprs=["panda_joint5|panda_joint6|panda_joint7"],
            kp=200.0,
            kv=20.0,
            force_range=200.0,
        ),
    },
    "collision_groups": {7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 1, 19: 2},
    "collision_palm_name": "left_hand_link",
    "all_tactile_links": ALL_TACTILE_LINKS_LEFT,
    "human_robot_link_map": {
        k: v.format(side_acro_cap="left") for k, v in HUMAN_ROBOT_LINK_MAP.items()
    },
    "position_controlled_joints": POSITION_CONTROLLED_JOINTS_PANDA
    + ["left_" + j for j in POSITION_CONTROLLED_JOINTS],
    "velocity_controlled_joints": ["left_" + j for j in VELOCITY_CONTROLLED_JOINTS],
    "open_qpos": {"left_" + joint: value for joint, value in OPEN_CONFIG.items()},
}

XHAND_RIGHT_CFG = {
    "urdf_path": str(get_external_data_dir() / "robots" / right_rel_urdf),
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
    "position_controlled_joints": POSITION_CONTROLLED_JOINTS_PANDA
    + ["right_" + j for j in POSITION_CONTROLLED_JOINTS],
    "velocity_controlled_joints": ["right_" + j for j in VELOCITY_CONTROLLED_JOINTS],
    "open_qpos": {"right_" + joint: value for joint, value in OPEN_CONFIG.items()},
}

XHAND_CFGs = dict(
    left=XHAND_LEFT_CFG,
    right=XHAND_RIGHT_CFG,
)
