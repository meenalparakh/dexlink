from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexlink.scripts.retargeting.export_oakink_lift_data import (
    REFERENCE_KEYPOINT_NAMES,
    _robot_reference_keypoints,
)
from dexlink.src.retargeting.data.process_arctic import ArcticProcessor
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE, ArcticClipSpec, get_clip
from dexlink.src.reference.arctic.runtime import (
    materialize_runtime_urdfs,
    processor_key,
    register_processor,
)


@dataclass(frozen=True)
class AssembleArgs:
    clip: str
    workspace: Path
    motion_output: Path | None
    sequence_output: Path | None
    overwrite: bool


def parse_args(argv: list[str] | None = None) -> AssembleArgs:
    parser = argparse.ArgumentParser(
        description="Assemble dense bimanual ARCTIC retargets and their RL payload."
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--motion-output", type=Path, default=None)
    parser.add_argument("--sequence-output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    raw = parser.parse_args(argv)
    return AssembleArgs(
        clip=str(raw.clip),
        workspace=raw.workspace.resolve(),
        motion_output=(
            None if raw.motion_output is None else raw.motion_output.resolve()
        ),
        sequence_output=(
            None if raw.sequence_output is None else raw.sequence_output.resolve()
        ),
        overwrite=bool(raw.overwrite),
    )


def frame_motion_path(
    workspace: Path, clip: ArcticClipSpec, timestep: int, side: str
) -> Path:
    return (
        workspace
        / "retargeted"
        / clip.retargeting_clip_id
        / f"t{timestep:04d}"
        / side
        / "sequence_motion_lineseq.npz"
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dense retarget output is missing: {path}. Run the generated retarget "
            "commands before assembly."
        )
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _scalar(payload: dict[str, np.ndarray], key: str) -> object:
    if key not in payload:
        raise KeyError(f"Retarget motion is missing required field {key!r}.")
    value = np.asarray(payload[key])
    if value.shape != ():
        raise ValueError(f"Expected scalar field {key!r}, got shape {value.shape}.")
    return value.item()


def _active_hand_row(
    payload: dict[str, np.ndarray], *, side: str, timestep: int, path: Path
) -> tuple[np.ndarray, np.ndarray]:
    if _scalar(payload, "traj_hand_side") != side:
        raise ValueError(
            f"Expected {path} to contain active side {side!r}, got "
            f"{_scalar(payload, 'traj_hand_side')!r}."
        )
    if int(_scalar(payload, "arctic_timestep")) != timestep:
        raise ValueError(
            f"Expected {path} timestep metadata {timestep}, got "
            f"{_scalar(payload, 'arctic_timestep')}."
        )
    names = np.asarray(payload[f"{side}_joint_names"], dtype=object).reshape(-1)
    values = np.asarray(payload[f"{side}_joint_vals"], dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] != len(names):
        raise ValueError(
            f"Expected {side} joint values shape (1, {len(names)}) in {path}, "
            f"got {values.shape}."
        )
    return names, values[0]


def _unwrap_root_rotations(values: np.ndarray, names: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    rotation_indices = [
        idx
        for idx, name in enumerate(names.tolist())
        if any(token in str(name) for token in ("_roll_", "_pitch_", "_yaw_"))
    ]
    if len(rotation_indices) != 3:
        raise ValueError(
            "Expected exactly three roll/pitch/yaw forearm root joints, got "
            f"indices {rotation_indices} from {names.tolist()}."
        )
    out[:, rotation_indices] = np.unwrap(
        out[:, rotation_indices].astype(np.float64), axis=0
    ).astype(np.float32)
    return out


def load_dense_hands(
    clip: ArcticClipSpec, workspace: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    names_by_side: dict[str, np.ndarray] = {}
    rows_by_side: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    for timestep in range(clip.start_frame, clip.end_frame):
        for side in ("left", "right"):
            path = frame_motion_path(workspace, clip, timestep, side)
            payload = _load_npz(path)
            names, row = _active_hand_row(
                payload, side=side, timestep=timestep, path=path
            )
            previous_names = names_by_side.setdefault(side, names)
            if not np.array_equal(names, previous_names):
                raise ValueError(
                    f"Joint names changed at {path}: expected "
                    f"{previous_names.tolist()}, got {names.tolist()}."
                )
            rows_by_side[side].append(row)
    values_by_side = {
        side: _unwrap_root_rotations(np.stack(rows, axis=0), names_by_side[side])
        for side, rows in rows_by_side.items()
    }
    return names_by_side, values_by_side


def _load_arctic_demo(clip: ArcticClipSpec, workspace: Path) -> ArcticProcessor:
    target_object = register_processor(clip, workspace)
    object_urdf, _ = materialize_runtime_urdfs(clip, workspace)
    return ArcticProcessor(
        original_demo_path=str(clip.processed_demo_path()),
        target_object=target_object,
        start_timestep=clip.start_frame,
        end_timestep=clip.end_frame,
        skip=1,
        reference_urdf=str(object_urdf),
        target_urdf=str(object_urdf),
        run_dir=clip.trajectory_dir(workspace) / "processor",
    )


def _human_keypoints(demo: ArcticProcessor, side: str) -> np.ndarray:
    relevant = getattr(demo, f"relevant_{side}_joint_pos")
    missing = [name for name in REFERENCE_KEYPOINT_NAMES if name not in relevant]
    if missing:
        raise ValueError(f"ARCTIC {side} hand is missing keypoints {missing}.")
    return np.stack(
        [
            np.asarray(relevant[name], dtype=np.float32)
            for name in REFERENCE_KEYPOINT_NAMES
        ],
        axis=1,
    )


def build_motion_payload(
    clip: ArcticClipSpec,
    workspace: Path,
    names_by_side: dict[str, np.ndarray],
    values_by_side: dict[str, np.ndarray],
    demo: ArcticProcessor,
) -> dict[str, np.ndarray]:
    motion = demo.object_reference_motion
    asset_pos = np.asarray(motion["obj_pos"], dtype=np.float32)
    asset_quat = np.asarray(motion["obj_quat"], dtype=np.float32)
    asset_arti = np.asarray(motion["obj_arti"], dtype=np.float32)
    n_frames = clip.frame_count
    if asset_pos.shape != (n_frames, 3):
        raise ValueError(
            f"Expected ARCTIC object positions shape {(n_frames, 3)}, got {asset_pos.shape}."
        )
    if asset_quat.shape != (n_frames, 4):
        raise ValueError(
            f"Expected ARCTIC object quaternions shape {(n_frames, 4)}, got {asset_quat.shape}."
        )
    if asset_arti.ndim != 2 or asset_arti.shape[0] != n_frames:
        raise ValueError(
            f"Expected ARCTIC articulation shape ({n_frames}, A), got {asset_arti.shape}."
        )
    joint_names = np.asarray(motion["joint_names"], dtype=object).reshape(-1)
    if asset_arti.shape[1] != len(joint_names):
        raise ValueError(
            f"Articulation width {asset_arti.shape[1]} does not match names "
            f"{joint_names.tolist()}."
        )

    human_hand_data: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        joints = np.asarray(demo.world_coord[f"joints.{side}"], dtype=np.float32)
        verts = np.asarray(demo.world_coord[f"verts.{side}"], dtype=np.float32)
        if joints.shape != (n_frames, 21, 3):
            raise ValueError(
                f"Expected ARCTIC {side} joints shape {(n_frames, 21, 3)}, "
                f"got {joints.shape}."
            )
        if verts.ndim != 3 or verts.shape[0] != n_frames or verts.shape[2] != 3:
            raise ValueError(
                f"Expected ARCTIC {side} vertices shape ({n_frames}, V, 3), "
                f"got {verts.shape}."
            )
        human_hand_data[f"{side}_human_hand_joints"] = joints
        human_hand_data[f"{side}_human_hand_verts"] = verts

    support_urdf = clip.support_block_urdf_path(workspace).resolve()
    if not support_urdf.is_file():
        raise FileNotFoundError(
            f"Prepared Arctic support-block URDF is missing: {support_urdf}. "
            "Run dexlink.src.reference.arctic.prepare first."
        )
    scene_object_pos = np.zeros((n_frames, 2, 3), dtype=np.float32)
    scene_object_pos[:, 0, :] = asset_pos
    scene_object_pos[:, 1, :] = np.asarray(clip.support_block_pos, dtype=np.float32)
    scene_object_quat = np.zeros((n_frames, 2, 4), dtype=np.float32)
    scene_object_quat[:, 0, :] = asset_quat
    scene_object_quat[:, 1, 0] = 1.0
    scene_object_rot = (
        R.from_quat(np.roll(scene_object_quat.reshape(-1, 4), -1, axis=-1))
        .as_rotvec()
        .reshape(n_frames, 2, 3)
        .astype(np.float32)
    )
    scene_object_arti_vals = np.zeros(
        (n_frames, 2, asset_arti.shape[1]), dtype=np.float32
    )
    scene_object_arti_vals[:, 0, :] = asset_arti
    scene_object_arti_valid_mask = np.zeros((2, asset_arti.shape[1]), dtype=bool)
    scene_object_arti_valid_mask[0, :] = True
    scene_object_arti_names = np.empty(2, dtype=object)
    scene_object_arti_names[0] = joint_names.copy()
    scene_object_arti_names[1] = np.asarray([], dtype=object)
    scene_object_active_mask = np.zeros((n_frames, 2), dtype=bool)
    scene_object_active_mask[:, 0] = True

    payload: dict[str, np.ndarray] = {
        "asset_pos": asset_pos,
        "asset_quat": asset_quat,
        "asset_arti_vals": asset_arti,
        "asset_arti_names": joint_names,
        "left_joint_names": names_by_side["left"],
        "left_joint_vals": values_by_side["left"],
        "right_joint_names": names_by_side["right"],
        "right_joint_vals": values_by_side["right"],
        "traj_hand_side": np.array("right", dtype=object),
        "traj_hand_name": np.array("xhand", dtype=object),
        "traj_robot_model": np.array("xhand", dtype=object),
        "traj_object_name": np.array(processor_key(clip), dtype=object),
        "traj_object_urdf_path": np.array(
            str(clip.object_urdf_path(workspace)), dtype=object
        ),
        "exact_timesteps": np.arange(clip.start_frame, clip.end_frame, dtype=np.int64),
        "is_active_left": np.array([True], dtype=bool),
        "is_active_right": np.array([True], dtype=bool),
        "source_dataset": np.array("arctic", dtype=object),
        "arctic_clip_id": np.array(clip.clip_id, dtype=object),
        "arctic_dexmachina_clip": np.array(clip.dexmachina_clip, dtype=object),
        "arctic_demo_path": np.array(str(clip.processed_demo_path()), dtype=object),
        "arctic_start_timestep": np.array(clip.start_frame, dtype=np.int64),
        "arctic_end_timestep": np.array(clip.end_frame, dtype=np.int64),
        "arctic_skip": np.array(1, dtype=np.int64),
        "phase_names": np.asarray(["dense_mocap"], dtype=object),
        "phase_lengths": np.asarray([n_frames], dtype=np.int64),
        "n_reach_frames": np.array([0], dtype=np.int64),
        "n_lift_frames": np.array([0], dtype=np.int64),
        "human_skeleton_keypoint_names": np.asarray(
            REFERENCE_KEYPOINT_NAMES, dtype=object
        ),
        "left_human_skeleton_keypoint_pos": _human_keypoints(demo, "left"),
        "right_human_skeleton_keypoint_pos": _human_keypoints(demo, "right"),
        "left_human_skeleton_keypoint_valid": np.ones(n_frames, dtype=bool),
        "right_human_skeleton_keypoint_valid": np.ones(n_frames, dtype=bool),
        "target_bps_sampling_link_names": np.asarray(["bottom", "top"], dtype=object),
        "is_multi_object_sequence": np.array(True, dtype=bool),
        "scene_object_ids": np.asarray(
            ["object", "arctic_support_block"], dtype=object
        ),
        "scene_object_names": np.asarray(
            [processor_key(clip), "arctic_support_block"], dtype=object
        ),
        "scene_object_urdf_paths": np.asarray(
            [str(clip.object_urdf_path(workspace)), str(support_urdf)], dtype=object
        ),
        "scene_object_pos": scene_object_pos,
        "scene_object_quat": scene_object_quat,
        "scene_object_rot": scene_object_rot,
        "scene_object_arti_vals": scene_object_arti_vals,
        "scene_object_arti_valid_mask": scene_object_arti_valid_mask,
        "scene_object_arti_names": scene_object_arti_names,
        "scene_object_active_mask": scene_object_active_mask,
        "scene_object_static": np.asarray([False, True], dtype=bool),
        "scene_object_friction": np.asarray(
            [1.0, clip.support_block_friction], dtype=np.float32
        ),
        "scene_active_object_index": np.zeros(n_frames, dtype=np.int64),
        "scene_canonical_object_id": np.array("object", dtype=object),
        "scene_uses_block": np.array(True, dtype=bool),
        "scene_object_support_z_offsets": np.zeros(2, dtype=np.float32),
        **human_hand_data,
    }
    return payload


def build_sequence_payload(motion: dict[str, np.ndarray]) -> dict[str, object]:
    n_frames = int(motion["asset_pos"].shape[0])
    object_urdf = str(np.asarray(motion["traj_object_urdf_path"]).item())
    object_name = str(np.asarray(motion["traj_object_name"]).item())
    object_quat = np.asarray(motion["asset_quat"], dtype=np.float32)
    object_rot = (
        R.from_quat(np.roll(object_quat, -1, axis=-1)).as_rotvec().astype(np.float32)
    )
    for side in ("left", "right"):
        joints = np.asarray(motion[f"{side}_human_hand_joints"], dtype=np.float32)
        verts = np.asarray(motion[f"{side}_human_hand_verts"], dtype=np.float32)
        if joints.shape != (n_frames, 21, 3):
            raise ValueError(
                f"Expected assembled {side} MANO joints shape {(n_frames, 21, 3)}, "
                f"got {joints.shape}."
            )
        if verts.ndim != 3 or verts.shape[0] != n_frames or verts.shape[2] != 3:
            raise ValueError(
                f"Expected assembled {side} MANO vertices shape ({n_frames}, V, 3), "
                f"got {verts.shape}."
            )

    robot_reference = {
        side: _robot_reference_keypoints(
            robot_model="xhand",
            side=side,
            config_side=side,
            joint_names=motion[f"{side}_joint_names"],
            joint_vals=motion[f"{side}_joint_vals"],
            robot_base_pose=None,
        )
        for side in ("left", "right")
    }
    human_reference = {
        side: np.asarray(
            motion[f"{side}_human_skeleton_keypoint_pos"], dtype=np.float32
        )
        for side in ("left", "right")
    }
    payload = {
        "demo_kind": "sequence",
        "side": "right",
        "active_sides": np.asarray(["left", "right"], dtype=object),
        "robot_model": "xhand",
        "original_obj_name": object_name,
        "target_obj_name": object_name,
        "original_urdf_path": object_urdf,
        "target_urdf_path": object_urdf,
        "target_articulated_urdf_path": object_urdf,
        "target_articulation_link_names": np.asarray(["bottom", "top"], dtype=object),
        "left_mano_verts": motion["left_human_hand_verts"].copy(),
        "right_mano_verts": motion["right_human_hand_verts"].copy(),
        "left_mano_joints": motion["left_human_hand_joints"].copy(),
        "right_mano_joints": motion["right_human_hand_joints"].copy(),
        "reference_keypoint_names": np.asarray(REFERENCE_KEYPOINT_NAMES, dtype=object),
        "left_reference_keypoint_pos": robot_reference["left"],
        "right_reference_keypoint_pos": robot_reference["right"],
        "left_human_reference_keypoint_pos": human_reference["left"],
        "right_human_reference_keypoint_pos": human_reference["right"],
        "left_human_reference_keypoint_valid": np.ones(n_frames, dtype=bool),
        "right_human_reference_keypoint_valid": np.ones(n_frames, dtype=bool),
        "reference_obj_pos": motion["asset_pos"].copy(),
        "reference_obj_quat": object_quat.copy(),
        "reference_obj_rot": object_rot.copy(),
        "reference_obj_arti": motion["asset_arti_vals"].copy(),
        "reference_joint_names": motion["asset_arti_names"].copy(),
        "left_jnames": motion["left_joint_names"].copy(),
        "left_jvals": motion["left_joint_vals"].copy(),
        "right_jnames": motion["right_joint_names"].copy(),
        "right_jvals": motion["right_joint_vals"].copy(),
        "retargeted_obj_pos": motion["asset_pos"].copy(),
        "retargeted_obj_quat": object_quat.copy(),
        "retargeted_obj_rot": object_rot.copy(),
        "retargeted_obj_arti": motion["asset_arti_vals"].copy(),
        "retargeted_joint_names": motion["asset_arti_names"].copy(),
        "phase_names": motion["phase_names"].copy(),
        "phase_lengths": motion["phase_lengths"].copy(),
        "n_reach_frames": motion["n_reach_frames"].copy(),
        "n_lift_frames": motion["n_lift_frames"].copy(),
        "is_active_left": np.array([True], dtype=bool),
        "is_active_right": np.array([True], dtype=bool),
        "source_dataset": motion["source_dataset"].copy(),
        "arctic_clip_id": motion["arctic_clip_id"].copy(),
        "arctic_dexmachina_clip": motion["arctic_dexmachina_clip"].copy(),
        "arctic_demo_path": motion["arctic_demo_path"].copy(),
        "arctic_start_timestep": motion["arctic_start_timestep"].copy(),
        "arctic_end_timestep": motion["arctic_end_timestep"].copy(),
        "arctic_skip": motion["arctic_skip"].copy(),
    }
    scene_keys = (
        "is_multi_object_sequence",
        "scene_object_ids",
        "scene_object_names",
        "scene_object_urdf_paths",
        "scene_object_pos",
        "scene_object_quat",
        "scene_object_rot",
        "scene_object_arti_vals",
        "scene_object_arti_valid_mask",
        "scene_object_arti_names",
        "scene_object_active_mask",
        "scene_object_static",
        "scene_object_friction",
        "scene_active_object_index",
        "scene_canonical_object_id",
        "scene_uses_block",
        "scene_object_support_z_offsets",
    )
    payload.update({key: motion[key].copy() for key in scene_keys})
    return payload


def _check_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clip = get_clip(args.clip)
    motion_output = args.motion_output or clip.dense_motion_path(args.workspace)
    sequence_output = args.sequence_output or clip.exported_sequence_path(
        args.workspace
    )
    _check_output(motion_output, args.overwrite)
    _check_output(sequence_output, args.overwrite)

    names_by_side, values_by_side = load_dense_hands(clip, args.workspace)
    demo = _load_arctic_demo(clip, args.workspace)
    motion = build_motion_payload(
        clip, args.workspace, names_by_side, values_by_side, demo
    )
    np.savez_compressed(motion_output, **motion)
    np.save(sequence_output, build_sequence_payload(motion), allow_pickle=True)
    print(f"Wrote dense motion: {motion_output}")
    print(f"Wrote multi-object RL sequence: {sequence_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
