#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from urdfpy import URDF

from dexlink.visualizations.retargeting.vis_robot_retgt_motion import (
    ROBOT_URDFS,
    _pose_dict_for_object_motion_frame,
    _pose_dict_for_robot_motion_frame,
    _resolve_active_hands,
    _resolve_robot_base_pose,
    _resolve_robot_model,
)
from dexlink.utils.retargeting.path_utils import process_urdf
from dexlink.utils.retargeting.vis_server import VisAnimator


@dataclass(frozen=True)
class SequenceMotionVisArgs:
    sequence_motion: Path
    port: int
    mesh_type: str
    robot_model: str


def _unwrap_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _load_sequence_motion(path: Path) -> dict[str, np.ndarray]:
    if path.suffix != ".npz":
        raise ValueError(f"Expected --sequence-motion to be a .npz file, got {path}.")
    with np.load(path, allow_pickle=True) as data:
        motion = {key: data[key] for key in data.files}

    required_keys = (
        "asset_pos",
        "asset_quat",
        "asset_arti_vals",
        "asset_arti_names",
        "traj_object_urdf_path",
        "left_joint_names",
        "left_joint_vals",
        "right_joint_names",
        "right_joint_vals",
        "is_active_left",
        "is_active_right",
    )
    missing = [key for key in required_keys if key not in motion]
    if missing:
        raise ValueError(
            f"Expected sequence_motion.npz keys {required_keys}, missing {missing}. "
            f"Available keys: {sorted(motion)}"
        )

    asset_pos = np.asarray(motion["asset_pos"], dtype=np.float32)
    asset_quat = np.asarray(motion["asset_quat"], dtype=np.float32)
    asset_arti_vals = np.asarray(motion["asset_arti_vals"], dtype=np.float32)
    if asset_pos.ndim != 2 or asset_pos.shape[1] != 3:
        raise ValueError(
            f"Expected asset_pos to have shape (T, 3), got {asset_pos.shape}."
        )
    if asset_quat.shape != (asset_pos.shape[0], 4):
        raise ValueError(
            "Expected asset_quat to have shape (T, 4) matching asset_pos, "
            f"got {asset_quat.shape} and {asset_pos.shape}."
        )
    if asset_arti_vals.ndim != 2 or asset_arti_vals.shape[0] != asset_pos.shape[0]:
        raise ValueError(
            "Expected asset_arti_vals to have shape (T, n_obj_joints) matching "
            f"asset_pos, got {asset_arti_vals.shape} and {asset_pos.shape}."
        )

    motion["asset_pos"] = asset_pos
    motion["asset_quat"] = asset_quat
    motion["asset_arti_vals"] = asset_arti_vals
    motion["asset_arti_names"] = np.asarray(motion["asset_arti_names"], dtype=object)
    motion["left_joint_names"] = np.asarray(motion["left_joint_names"], dtype=object)
    motion["right_joint_names"] = np.asarray(motion["right_joint_names"], dtype=object)
    motion["left_joint_vals"] = np.asarray(motion["left_joint_vals"], dtype=np.float32)
    motion["right_joint_vals"] = np.asarray(
        motion["right_joint_vals"], dtype=np.float32
    )

    for side in ("left", "right"):
        names = motion[f"{side}_joint_names"]
        vals = motion[f"{side}_joint_vals"]
        if vals.ndim != 2:
            raise ValueError(
                f"Expected {side}_joint_vals to have shape (T, nq), got {vals.shape}."
            )
        if vals.shape[0] != asset_pos.shape[0]:
            raise ValueError(
                f"Expected {side}_joint_vals first dimension to match asset_pos T="
                f"{asset_pos.shape[0]}, got {vals.shape[0]}."
            )
        if vals.shape[1] != len(names):
            raise ValueError(
                f"Expected {side}_joint_vals width to match {side}_joint_names length, "
                f"got {vals.shape[1]} and {len(names)}."
            )

    return motion


def _sequence_motion_to_vis_motion(motion: dict[str, np.ndarray]) -> dict:
    return {
        **motion,
        "obj_pos": motion["asset_pos"],
        "obj_quat": motion["asset_quat"],
        "obj_arti": motion["asset_arti_vals"],
        "joint_names": motion["asset_arti_names"],
        "left_jnames": motion["left_joint_names"],
        "left_jvals": motion["left_joint_vals"],
        "right_jnames": motion["right_joint_names"],
        "right_jvals": motion["right_joint_vals"],
    }


def _parse_args() -> SequenceMotionVisArgs:
    parser = argparse.ArgumentParser(
        description="Visualize the last frame of a sequence_motion.npz trajectory."
    )
    parser.add_argument(
        "--sequence-motion",
        type=Path,
        required=True,
        help="Path to sequence_motion.npz produced by the sequence trajopt exporter.",
    )
    parser.add_argument("--port", type=int, default=6000, help="MeshCat ZMQ port.")
    parser.add_argument(
        "--mesh_type",
        type=str,
        default="visual",
        choices=["visual", "collision"],
        help="URDF mesh type used for MeshCat rendering.",
    )
    parser.add_argument(
        "--robot-model",
        type=str,
        default="auto",
        choices=["auto", "inspire", "xhand", "franka_xhand"],
        help="Robot URDF family to render. 'auto' uses sequence metadata.",
    )
    args = parser.parse_args()
    return SequenceMotionVisArgs(
        sequence_motion=args.sequence_motion,
        port=args.port,
        mesh_type=args.mesh_type,
        robot_model=args.robot_model,
    )


def main() -> None:
    args = _parse_args()
    sequence_motion = _load_sequence_motion(args.sequence_motion)
    motion = _sequence_motion_to_vis_motion(sequence_motion)
    frame_idx = int(motion["asset_pos"].shape[0] - 1)

    robot_model = _resolve_robot_model(args.robot_model, None, motion)
    robot_base_pose = _resolve_robot_base_pose(None, motion)
    active_hands = _resolve_active_hands(None, motion)
    object_urdf_path = Path(str(_unwrap_scalar(motion["traj_object_urdf_path"])))

    left_urdf = (
        URDF.load(str(ROBOT_URDFS[robot_model]["left"]))
        if active_hands["left"]
        else None
    )
    right_urdf = (
        URDF.load(str(ROBOT_URDFS[robot_model]["right"]))
        if active_hands["right"]
        else None
    )
    object_urdf = URDF.load(str(process_urdf(object_urdf_path)))

    print(f"Loaded sequence motion: {args.sequence_motion}")
    print(f"Visualizing last frame index: {frame_idx}")
    print(f"Using robot model: {robot_model}")
    print(f"Active hands: {active_hands}")
    print(f"Target object URDF: {object_urdf_path}")

    vis = VisAnimator(port_vis=args.port)
    if left_urdf is not None:
        vis.initialize_urdf(
            urdf_inst=left_urdf,
            name="robot/left_hand",
            mesh_type=args.mesh_type,
            color="cool",
            alpha=0.8,
        )
    if right_urdf is not None:
        vis.initialize_urdf(
            urdf_inst=right_urdf,
            name="robot/right_hand",
            mesh_type=args.mesh_type,
            color="cool",
            alpha=0.8,
        )
    vis.initialize_urdf(
        urdf_inst=object_urdf,
        name="target/object",
        mesh_type=args.mesh_type,
        color="orange",
        alpha=0.8,
    )
    vis.initialize()

    urdf_dict = {
        "target/object": _pose_dict_for_object_motion_frame(motion, frame_idx),
    }
    for side in ("left", "right"):
        if not active_hands[side]:
            continue
        joint_names = motion[f"{side}_jnames"]
        joint_vals = motion[f"{side}_jvals"]
        if len(joint_names) == 0:
            raise ValueError(f"{side} hand is active but {side}_joint_names is empty.")
        urdf_dict[f"robot/{side}_hand"] = _pose_dict_for_robot_motion_frame(
            joint_names,
            joint_vals[frame_idx],
            base_pose=robot_base_pose,
        )

    vis.add_frame(pts_dict={}, urdf_dict=urdf_dict)
    vis.play()
    print("Last frame loaded in MeshCat. Keep this process running while viewing.")


if __name__ == "__main__":
    main()
