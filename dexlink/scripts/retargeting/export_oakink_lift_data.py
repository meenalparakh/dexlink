#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.src.retargeting.data.hand_cfgs.franka_xhand import XHAND_CFGs as FRANKA_XHAND_CFGs
from dexlink.src.retargeting.data.hand_cfgs.inspire import INSPIRE_CFGs
from dexlink.src.retargeting.data.hand_cfgs.xhand import XHAND_CFGs
from dexlink.src.retargeting.optimization.oakink_processor import OakInkProcessor
from dexlink.utils.retargeting.path_utils import process_urdf
from dexlink.utils.retargeting.recompute_inertia_from_density import (
    recompute_urdf_inertia_from_density,
)
from dexlink.utils.retargeting.urdf_utils import create_urdf_from_obj

REFERENCE_KEYPOINT_NAMES = (
    "wrist",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "little_tip",
    "thumb_base",
    "index_base",
    "middle_base",
    "ring_base",
    "little_base",
    "thumb_proximal",
    "index_proximal",
    "middle_proximal",
    "ring_proximal",
    "little_proximal",
)
MANO_KEYPOINT_INDICES = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "little_tip": 20,
    "thumb_base": 2,
    "index_base": 5,
    "middle_base": 9,
    "ring_base": 13,
    "little_base": 17,
    "thumb_proximal": 3,
    "index_proximal": 6,
    "middle_proximal": 10,
    "ring_proximal": 14,
    "little_proximal": 18,
}


def _load_motion_npz(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _unwrap_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _required_motion_field(motion: dict, key: str):
    if key not in motion:
        raise ValueError(f"Motion npz is missing required field {key!r}.")
    return _unwrap_scalar(motion[key])


def _export_simple_urdf(
    demo: OakInkProcessor,
    export_dir: Path,
    obj_type: str,
    density: float,
) -> tuple[str, str]:
    obj_dir = export_dir / obj_type
    obj_dir.mkdir(parents=True, exist_ok=True)
    obj_path = Path(demo.export_object_obj(output_dir=obj_dir, obj_type=obj_type))
    urdf_path = Path(create_urdf_from_obj(str(obj_path), str(obj_dir), scale=1.0))
    recompute_urdf_inertia_from_density(
        input_urdf=urdf_path,
        output_urdf=urdf_path,
        density=density,
    )
    return str(obj_path), str(urdf_path)


def _resample_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, target_len, axis=0)
    indices = np.linspace(0, arr.shape[0] - 1, target_len)
    nearest = np.rint(indices).astype(int)
    return arr[nearest]


def _default_output_path(motion_npy_path: Path) -> Path:
    motion_npy_path = motion_npy_path.resolve()
    stem = motion_npy_path.stem
    if stem.endswith(".npz"):
        stem = Path(stem).stem
    return motion_npy_path.parent / f"{stem}_oakink_lift_exported.npy"


def _normalize_robot_model(robot_model: str) -> str:
    value = str(robot_model).lower()
    aliases = {
        "inspire": "inspire",
        "inspire_hand": "inspire",
        "xhand": "xhand",
        "franka": "franka_xhand",
        "franka_xhand": "franka_xhand",
        "franka-xhand": "franka_xhand",
        "panda_xhand": "franka_xhand",
    }
    if value not in aliases:
        raise ValueError(
            "Unsupported robot_model "
            f"{robot_model!r}; use 'inspire', 'xhand', or 'franka_xhand'."
        )
    return aliases[value]


def _robot_hand_cfg(robot_model: str, side: str) -> dict:
    model = _normalize_robot_model(robot_model)
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported side {side!r}; use 'left' or 'right'.")
    if model == "inspire":
        cfgs = INSPIRE_CFGs
    elif model == "xhand":
        cfgs = XHAND_CFGs
    else:
        cfgs = FRANKA_XHAND_CFGs
    return cfgs[side]


def _resolve_robot_model(motion: dict, robot_model_arg: str | None) -> str:
    if robot_model_arg is not None:
        return str(robot_model_arg)
    if "traj_robot_model" in motion:
        return str(_required_motion_field(motion, "traj_robot_model"))
    raise ValueError(
        "Robot model was not found in the motion payload. Pass it explicitly with "
        "`--robot-model`."
    )


def _link_positions_from_joint_motion(
    *,
    robot_model: str,
    side: str,
    config_side: str | None = None,
    joint_names: np.ndarray,
    joint_vals: np.ndarray,
    robot_base_pose: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if config_side is None:
        config_side = side
    cfg = _robot_hand_cfg(robot_model, config_side)
    urdf_inst = URDF.load(str(process_urdf(cfg["urdf_path"])))
    ordered_joint_names = [
        joint.name for joint in urdf_inst.joints if joint.joint_type != "fixed"
    ]
    source_joint_names = [
        str(name) for name in np.asarray(joint_names, dtype=object).tolist()
    ]
    missing_joints = [
        name for name in ordered_joint_names if name not in source_joint_names
    ]
    if missing_joints:
        raise ValueError(
            f"Retargeted motion missing {side} robot joints required for "
            f"{config_side} config FK: {missing_joints}"
        )

    joint_vals = np.asarray(joint_vals, dtype=np.float32)
    if joint_vals.ndim != 2:
        raise ValueError(
            f"Expected {side} joint values to have shape (T, nq), got {joint_vals.shape}."
        )

    joint_index_map = {name: idx for idx, name in enumerate(source_joint_names)}
    ordered_joint_vals = joint_vals[
        :, [joint_index_map[name] for name in ordered_joint_names]
    ]
    cfgs = {
        name: ordered_joint_vals[:, idx] for idx, name in enumerate(ordered_joint_names)
    }
    link_fk = urdf_inst.link_fk_batch(cfgs=cfgs)
    if robot_base_pose is not None:
        robot_base_pose = np.asarray(robot_base_pose, dtype=np.float32)
        if robot_base_pose.shape != (4, 4):
            raise ValueError(
                f"Expected robot_base_pose to have shape (4, 4), got {robot_base_pose.shape}."
            )
    return {
        str(link.name): (
            poses[:, :3, 3]
            if robot_base_pose is None
            else (robot_base_pose[None, :, :] @ poses)[:, :3, 3]
        ).astype(np.float32, copy=False)
        for link, poses in link_fk.items()
    }


def _robot_reference_keypoints(
    *,
    robot_model: str,
    side: str,
    config_side: str | None = None,
    joint_names: np.ndarray,
    joint_vals: np.ndarray,
    robot_base_pose: np.ndarray | None,
) -> np.ndarray:
    if config_side is None:
        config_side = side
    cfg = _robot_hand_cfg(robot_model, config_side)
    link_pos = _link_positions_from_joint_motion(
        robot_model=robot_model,
        side=side,
        config_side=config_side,
        joint_names=joint_names,
        joint_vals=joint_vals,
        robot_base_pose=robot_base_pose,
    )
    human_robot_link_map = cfg["human_robot_link_map"]
    missing_links = [
        human_robot_link_map[name]
        for name in REFERENCE_KEYPOINT_NAMES
        if human_robot_link_map[name] not in link_pos
    ]
    if missing_links:
        raise ValueError(
            f"Robot FK for {side} is missing mapped links required for reference keypoints: {missing_links}"
        )
    return np.stack(
        [link_pos[human_robot_link_map[name]] for name in REFERENCE_KEYPOINT_NAMES],
        axis=1,
    ).astype(np.float32)


def _lift_offsets(
    asset_pos: np.ndarray, n_reach_frames: int, n_lift_frames: int
) -> np.ndarray:
    asset_pos = np.asarray(asset_pos, dtype=np.float32)
    n_frames = asset_pos.shape[0]
    if n_reach_frames < 0 or n_reach_frames > n_frames:
        raise ValueError(
            f"n_reach_frames must be in [0, {n_frames}], got {n_reach_frames}."
        )
    if n_lift_frames < 0 or n_reach_frames + n_lift_frames != n_frames:
        raise ValueError(
            "Expected n_reach_frames + n_lift_frames to match total motion length, "
            f"got {n_reach_frames} + {n_lift_frames} vs {n_frames}."
        )

    offsets = np.zeros((n_frames, 1, 3), dtype=np.float32)
    if n_lift_frames == 0 or n_reach_frames == 0:
        return offsets
    z_delta = asset_pos[n_reach_frames:, 2] - asset_pos[n_reach_frames - 1, 2]
    offsets[n_reach_frames:, 0, 2] = z_delta.astype(np.float32, copy=False)
    return offsets


def _build_reference_hand_sequence(
    *,
    side: str,
    active_side: str,
    n_frames: int,
    n_reach_frames: int,
    lift_offsets: np.ndarray,
    mano_joints: np.ndarray,
    mano_verts: np.ndarray,
    robot_reference_keypoints: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_joints = _resample_to_length(
        np.asarray(mano_joints, dtype=np.float32), n_frames
    ).copy()
    ref_verts = _resample_to_length(
        np.asarray(mano_verts, dtype=np.float32), n_frames
    ).copy()
    ref_keypoints = ref_joints[
        :, [MANO_KEYPOINT_INDICES[name] for name in REFERENCE_KEYPOINT_NAMES], :
    ].copy()

    if side == active_side:
        if robot_reference_keypoints is None:
            raise ValueError(
                f"Missing robot reference keypoints for active side {side!r}."
            )
        ref_keypoints[:n_reach_frames] = robot_reference_keypoints[:n_reach_frames]
        for keypoint_idx, name in enumerate(REFERENCE_KEYPOINT_NAMES):
            mano_idx = MANO_KEYPOINT_INDICES[name]
            ref_joints[:n_reach_frames, mano_idx, :] = robot_reference_keypoints[
                :n_reach_frames, keypoint_idx, :
            ]
        ref_joints += lift_offsets
        ref_verts += lift_offsets
        ref_keypoints += lift_offsets

    return (
        ref_joints.astype(np.float32),
        ref_verts.astype(np.float32),
        ref_keypoints.astype(np.float32),
    )


def _optional_motion_array(
    motion: dict,
    key: str,
    *,
    dtype,
) -> np.ndarray | None:
    if key not in motion:
        return None
    value = motion[key]
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def build_payload(args) -> dict:
    motion = _load_motion_npz(args.motion_npy_path)
    demo_path = _required_motion_field(motion, "oakink_demo_path")
    run_dir = Path(_required_motion_field(motion, "oakink_run_dir"))
    target_demo_path = _unwrap_scalar(motion.get("oakink_target_demo_path"))
    materialized_target_obj_path = _unwrap_scalar(motion.get("oakink_target_obj_path"))
    requested_target_obj_path = _unwrap_scalar(
        motion.get("oakink_requested_target_obj_path")
    )
    target_transform_info = _unwrap_scalar(motion.get("oakink_target_transform_info"))
    target_scale_raw = motion.get("oakink_target_scale")
    target_scale = (
        1.0
        if target_scale_raw is None
        else float(np.asarray(target_scale_raw, dtype=np.float32).reshape(-1)[0])
    )
    hand_side = str(_required_motion_field(motion, "oakink_hand_side"))
    start_timestep = int(_required_motion_field(motion, "oakink_start_timestep"))
    end_timestep = int(_required_motion_field(motion, "oakink_end_timestep"))
    skip = int(_required_motion_field(motion, "oakink_skip"))
    robot_model = _resolve_robot_model(motion, args.robot_model)

    if materialized_target_obj_path in (None, "", "None"):
        target_obj_path = requested_target_obj_path
        resolved_target_transform_info = target_transform_info
        resolved_target_scale = target_scale
    else:
        target_obj_path = materialized_target_obj_path
        resolved_target_transform_info = None
        resolved_target_scale = 1.0

    demo = OakInkProcessor(
        original_demo_path=str(demo_path),
        run_dir=run_dir,
        target_demo_path=(
            None if target_demo_path in (None, "", "None") else str(target_demo_path)
        ),
        target_obj_path=(
            None if target_obj_path in (None, "", "None") else str(target_obj_path)
        ),
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        skip=skip,
        oakink_hand_side=hand_side,
        target_transform_info=resolved_target_transform_info,
        target_scale=resolved_target_scale,
    )
    n_frames = int(motion["asset_pos"].shape[0])
    n_reach_frames = int(np.asarray(motion["n_reach_frames"]).reshape(-1)[0])
    n_lift_frames = int(np.asarray(motion["n_lift_frames"]).reshape(-1)[0])
    export_dir = (
        Path(args.output_path).resolve().parent
        / f"{Path(args.output_path).stem}_assets"
    )

    original_obj_path, original_urdf_path = _export_simple_urdf(
        demo, export_dir, obj_type="original", density=args.density
    )
    target_obj_path, target_urdf_path = _export_simple_urdf(
        demo, export_dir, obj_type="target", density=args.density
    )

    ref_motion = demo.object_reference_motion
    active_side = str(motion["traj_hand_side"].item())
    lift_offsets = _lift_offsets(motion["asset_pos"], n_reach_frames, n_lift_frames)
    traj_robot_base_pose = _optional_motion_array(
        motion,
        "traj_robot_base_pose",
        dtype=np.float32,
    )
    traj_object_raised_by = _optional_motion_array(
        motion,
        "traj_object_raised_by",
        dtype=np.float32,
    )
    robot_reference_keypoints = {
        active_side: _robot_reference_keypoints(
            robot_model=robot_model,
            side=active_side,
            joint_names=motion[f"{active_side}_joint_names"],
            joint_vals=motion[f"{active_side}_joint_vals"],
            robot_base_pose=traj_robot_base_pose,
        )
    }
    left_mano_joints, left_mano_verts, left_reference_keypoints = (
        _build_reference_hand_sequence(
            side="left",
            active_side=active_side,
            n_frames=n_frames,
            n_reach_frames=n_reach_frames,
            lift_offsets=lift_offsets,
            mano_joints=demo.data["left_joints"],
            mano_verts=demo.left_hand_verts,
            robot_reference_keypoints=robot_reference_keypoints.get("left"),
        )
    )
    right_mano_joints, right_mano_verts, right_reference_keypoints = (
        _build_reference_hand_sequence(
            side="right",
            active_side=active_side,
            n_frames=n_frames,
            n_reach_frames=n_reach_frames,
            lift_offsets=lift_offsets,
            mano_joints=demo.data["right_joints"],
            mano_verts=demo.right_hand_verts,
            robot_reference_keypoints=robot_reference_keypoints.get("right"),
        )
    )
    payload = {
        "demo_kind": "lift",
        "side": str(motion["traj_hand_side"].item()),
        "robot_model": robot_model,
        "original_obj_name": demo.obj_name,
        "target_obj_name": demo.target_obj_name,
        "original_obj_path": original_obj_path,
        "target_obj_path": target_obj_path,
        "original_urdf_path": original_urdf_path,
        "target_urdf_path": target_urdf_path,
        "left_mano_verts": left_mano_verts,
        "right_mano_verts": right_mano_verts,
        "left_mano_joints": left_mano_joints,
        "right_mano_joints": right_mano_joints,
        "reference_keypoint_names": np.asarray(REFERENCE_KEYPOINT_NAMES, dtype=object),
        "left_reference_keypoint_pos": left_reference_keypoints,
        "right_reference_keypoint_pos": right_reference_keypoints,
        "reference_obj_pos": _resample_to_length(
            ref_motion["obj_pos"], n_frames
        ).astype(np.float32),
        "reference_obj_quat": _resample_to_length(
            ref_motion["obj_quat"], n_frames
        ).astype(np.float32),
        "reference_obj_rot": _resample_to_length(
            ref_motion["obj_rot"], n_frames
        ).astype(np.float32),
        "reference_obj_arti": _resample_to_length(
            ref_motion["obj_arti"], n_frames
        ).astype(np.float32),
        "reference_joint_names": np.asarray(ref_motion["joint_names"], dtype=object),
        "left_jnames": np.asarray(motion["left_joint_names"], dtype=object),
        "left_jvals": motion["left_joint_vals"].astype(np.float32),
        "right_jnames": np.asarray(motion["right_joint_names"], dtype=object),
        "right_jvals": motion["right_joint_vals"].astype(np.float32),
        "retargeted_obj_pos": motion["asset_pos"].astype(np.float32),
        "retargeted_obj_quat": motion["asset_quat"].astype(np.float32),
        "retargeted_obj_rot": R.from_quat(
            np.roll(motion["asset_quat"].astype(np.float32), -1, axis=-1)
        )
        .as_rotvec()
        .astype(np.float32),
        "retargeted_obj_arti": motion["asset_arti_vals"].astype(np.float32),
        "retargeted_joint_names": np.asarray(motion["asset_arti_names"], dtype=object),
        "n_reach_frames": motion.get("n_reach_frames", np.array([0], dtype=np.int64)),
        "n_lift_frames": motion.get("n_lift_frames", np.array([0], dtype=np.int64)),
        "is_active_left": motion.get(
            "is_active_left", np.array([hand_side == "left"], dtype=bool)
        ),
        "is_active_right": motion.get(
            "is_active_right", np.array([hand_side == "right"], dtype=bool)
        ),
    }
    if traj_robot_base_pose is not None:
        payload["traj_robot_base_pose"] = traj_robot_base_pose
    if traj_object_raised_by is not None:
        payload["traj_object_raised_by"] = traj_object_raised_by
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export OakInk lift data and optimized robot motion into a Genesis-ready payload."
    )
    parser.add_argument("--motion-npy-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument(
        "--robot-model",
        type=str,
        default=None,
        help="Optional explicit robot model override. Use 'inspire', 'xhand', or 'franka_xhand'.",
    )
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = _default_output_path(args.motion_npy_path)
    return args


def main():
    args = parse_args()
    payload = build_payload(args)
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, payload, allow_pickle=True)
    print(f"Saved OakInk lift payload to {output_path}")


if __name__ == "__main__":
    main()
