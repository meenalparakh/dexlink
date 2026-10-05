from os.path import join

from dexlink.utils.retargeting.path_utils import get_external_data_dir

shadow_hand_asset_dir = "shadow_hand/"
left_rel_urdf = join(shadow_hand_asset_dir, "shadow_hand_left_6dof_kpts.urdf")
right_rel_urdf = join(shadow_hand_asset_dir, "shadow_hand_right_6dof_kpts.urdf")

FINGER_JOINT_MAPS = {
    "wrist": [
        "forearm_tx_link_joint",
        "forearm_ty_link_joint",
        "forearm_tz_link_joint",
        "forearm_roll_link_joint",
        "forearm_pitch_link_joint",
        "forearm_yaw_link_joint",
        "forearm_joint",
        "wrist_joint",
    ],
    "thumb": [
        "thumb_joint1",
        "thumb_joint2",
        "thumb_joint3",
        "thumb_joint4",
        "thumb_joint5",
    ],
    "index": [
        "index_finger_joint1",
        "index_finger_joint2",
        "index_finger_joint3",
        "index_finger_joint4",
    ],
    "middle": [
        "middle_finger_joint1",
        "middle_finger_joint2",
        "middle_finger_joint3",
        "middle_finger_joint4",
    ],
    "ring": [
        "ring_finger_joint1",
        "ring_finger_joint2",
        "ring_finger_joint3",
        "ring_finger_joint4",
    ],
    "little": [
        "little_finger_joint1",
        "little_finger_joint2",
        "little_finger_joint3",
        "little_finger_joint4",
        "little_finger_joint5",
    ],
}

POSITION_CONTROLLED_JOINTS = [
    *FINGER_JOINT_MAPS["wrist"],
    *FINGER_JOINT_MAPS["thumb"],
    *FINGER_JOINT_MAPS["index"],
    *FINGER_JOINT_MAPS["middle"],
    *FINGER_JOINT_MAPS["ring"],
    *FINGER_JOINT_MAPS["little"],
]

OPEN_CONFIG = {joint: 0.0 for joint in POSITION_CONTROLLED_JOINTS}

HUMAN_ROBOT_LINK_MAP = {
    "wrist": "{side_prefix}_wrist",
    "thumb_tip": "{side_prefix}_thumb_tip",
    "index_tip": "{side_prefix}_index_tip",
    "middle_tip": "{side_prefix}_middle_tip",
    "ring_tip": "{side_prefix}_ring_tip",
    "little_tip": "{side_prefix}_little_tip",
    "thumb_base": "{side_prefix}_thumb_base",
    "index_base": "{side_prefix}_index_base",
    "middle_base": "{side_prefix}_middle_base",
    "ring_base": "{side_prefix}_ring_base",
    "little_base": "{side_prefix}_little_base",
    "thumb_proximal": "{side_prefix}_thumb_proximal",
    "index_proximal": "{side_prefix}_index_proximal",
    "middle_proximal": "{side_prefix}_middle_proximal",
    "ring_proximal": "{side_prefix}_ring_proximal",
    "little_proximal": "{side_prefix}_little_proximal",
    "thumb_distal": "{side_prefix}_thumb_distal",
    "index_distal": "{side_prefix}_index_distal",
    "middle_distal": "{side_prefix}_middle_distal",
    "ring_distal": "{side_prefix}_ring_distal",
    "little_distal": "{side_prefix}_little_distal",
}

HUMAN_ROBOT_KPT_MAP = HUMAN_ROBOT_LINK_MAP.copy()

ALL_TACTILE_LINKS = [
    "{side_prefix}_body_thumb_distal",
    "{side_prefix}_body_index_finger_distal",
    "{side_prefix}_body_middle_finger_distal",
    "{side_prefix}_body_ring_finger_distal",
    "{side_prefix}_body_little_finger_distal",
    "{side_prefix}_thumb_tip",
    "{side_prefix}_index_tip",
    "{side_prefix}_middle_tip",
    "{side_prefix}_ring_tip",
    "{side_prefix}_little_tip",
]


def _prefix_joint(side_prefix: str, side_acro: str, joint: str) -> str:
    if joint.startswith(
        (
            "forearm_tx_",
            "forearm_ty_",
            "forearm_tz_",
            "forearm_roll_",
            "forearm_pitch_",
            "forearm_yaw_",
        )
    ):
        return f"{side_acro}_{joint}"
    return f"{side_prefix}_{joint}"


def _make_cfg(side: str) -> dict:
    if side == "left":
        side_prefix = "lh"
        side_acro = "L"
        rel_urdf = left_rel_urdf
    elif side == "right":
        side_prefix = "rh"
        side_acro = "R"
        rel_urdf = right_rel_urdf
    else:
        raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")

    position_controlled_joints = [
        _prefix_joint(side_prefix, side_acro, joint)
        for joint in POSITION_CONTROLLED_JOINTS
    ]

    return {
        "urdf_path": str(get_external_data_dir() / "robots" / rel_urdf),
        "wrist_link_name": f"{side_prefix}_body_palm",
        "kpt_link_names": [
            f"{side_prefix}_thumb_tip",
            f"{side_prefix}_index_tip",
            f"{side_prefix}_middle_tip",
            f"{side_prefix}_ring_tip",
            f"{side_prefix}_little_tip",
        ],
        "actuators": {
            "finger": dict(
                joint_exprs=[rf"{side_prefix}_.+_joint[1-5]$"],
                kp=20.0,
                kv=2.0,
                force_range=100.0,
            ),
            "wrist_rot": dict(
                joint_exprs=[
                    rf"{side_acro}_forearm_(roll|pitch|yaw)_link_joint",
                    rf"{side_prefix}_(forearm|wrist)_joint",
                ],
                kp=60.0,
                kv=3.0,
                force_range=100.0,
            ),
            "wrist_trans": dict(
                joint_exprs=[rf"{side_acro}_forearm_t[xyz]_link_joint"],
                kp=350.0,
                kv=15.0,
                force_range=100.0,
            ),
        },
        "collision_groups": {},
        "collision_palm_name": f"{side_prefix}_body_palm",
        "all_tactile_links": [
            link.format(side_prefix=side_prefix) for link in ALL_TACTILE_LINKS
        ],
        "human_robot_link_map": {
            k: v.format(side_prefix=side_prefix)
            for k, v in HUMAN_ROBOT_LINK_MAP.items()
        },
        "human_robot_kpt_map": {
            k: v.format(side_prefix=side_prefix)
            for k, v in HUMAN_ROBOT_KPT_MAP.items()
        },
        "finger_joint_maps": {
            finger: [
                _prefix_joint(side_prefix, side_acro, joint) for joint in joints
            ]
            for finger, joints in FINGER_JOINT_MAPS.items()
        },
        "position_controlled_joints": position_controlled_joints,
        "velocity_controlled_joints": [],
        "open_qpos": {
            _prefix_joint(side_prefix, side_acro, joint): value
            for joint, value in OPEN_CONFIG.items()
        },
        "open_qpos_hull": {
            _prefix_joint(side_prefix, side_acro, joint): value
            for joint, value in OPEN_CONFIG.items()
        },
    }


SHADOW_HAND_LEFT_CFG = _make_cfg("left")
SHADOW_HAND_RIGHT_CFG = _make_cfg("right")

SHADOW_HAND_CFGs = dict(
    left=SHADOW_HAND_LEFT_CFG,
    right=SHADOW_HAND_RIGHT_CFG,
)
