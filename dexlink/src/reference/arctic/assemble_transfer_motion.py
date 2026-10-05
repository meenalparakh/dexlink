from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from urdfpy import URDF

from dexlink.src.retargeting.data.process_arctic import ArcticProcessor, get_processor_cfg
from dexlink.utils.retargeting.urdf_utils import create_urdf_from_obj
from dexlink.src.reference.arctic.assemble_dense_motion import (
    _active_hand_row,
    _check_output,
    _load_npz,
    _unwrap_root_rotations,
    build_motion_payload,
    build_sequence_payload,
)
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE
from dexlink.src.reference.arctic.transfer_config import ArcticTransferSpec, get_transfer
from dexlink.src.reference.v2_tasks.common import write_box_obj


@dataclass(frozen=True)
class AssembleTransferArgs:
    transfer: str
    workspace: Path
    motion_output: Path | None
    sequence_output: Path | None
    overwrite: bool


def parse_args(argv: list[str] | None = None) -> AssembleTransferArgs:
    parser = argparse.ArgumentParser(
        description="Assemble one dense ARCTIC geometry-transfer reference."
    )
    parser.add_argument("--transfer", required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--motion-output", type=Path)
    parser.add_argument("--sequence-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    raw = parser.parse_args(argv)
    return AssembleTransferArgs(
        transfer=str(raw.transfer),
        workspace=raw.workspace.resolve(),
        motion_output=(
            None if raw.motion_output is None else raw.motion_output.resolve()
        ),
        sequence_output=(
            None if raw.sequence_output is None else raw.sequence_output.resolve()
        ),
        overwrite=bool(raw.overwrite),
    )


def _default_outputs(transfer: ArcticTransferSpec, workspace: Path) -> tuple[Path, Path]:
    root = workspace / "transfer_trajectories" / transfer.transfer_id
    return (
        root / "dense_robot_motion.npz",
        root / "exports" / f"{transfer.transfer_id}_oakink_sequence.npy",
    )


def load_transfer_hands(
    transfer: ArcticTransferSpec, workspace: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str, Path]:
    names_by_side: dict[str, np.ndarray] = {}
    rows_by_side: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    target_name: str | None = None
    target_urdf: Path | None = None
    clip = transfer.source_clip
    for timestep in range(clip.start_frame, clip.end_frame):
        for side in ("left", "right"):
            path = transfer.frame_motion_path(workspace, timestep, side)
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
            frame_target_name = str(np.asarray(payload["traj_object_name"]).item())
            frame_target_urdf = Path(
                str(np.asarray(payload["traj_object_urdf_path"]).item())
            ).resolve()
            if target_name is None:
                target_name = frame_target_name
                target_urdf = frame_target_urdf
            elif frame_target_name != target_name or frame_target_urdf != target_urdf:
                raise ValueError(
                    "Transfer target metadata changed across retargeted frames: "
                    f"expected name={target_name!r}, urdf={target_urdf}, got "
                    f"name={frame_target_name!r}, urdf={frame_target_urdf} at {path}."
                )
            rows_by_side[side].append(row)
    if target_name is None or target_urdf is None:
        raise ValueError(f"Transfer {transfer.transfer_id!r} contains no hand frames.")
    if target_name != transfer.target_object:
        raise ValueError(
            f"Expected transfer target {transfer.target_object!r}, got {target_name!r}."
        )
    if not target_urdf.is_file():
        raise FileNotFoundError(f"Transfer target URDF does not exist: {target_urdf}.")
    values_by_side = {
        side: _unwrap_root_rotations(np.stack(rows, axis=0), names_by_side[side])
        for side, rows in rows_by_side.items()
    }
    return names_by_side, values_by_side, target_name, target_urdf


def _load_transfer_demo(
    transfer: ArcticTransferSpec, workspace: Path
) -> ArcticProcessor:
    clip = transfer.source_clip
    processor_cfg = get_processor_cfg(transfer.target_object)
    demo_class = processor_cfg["processor_cls"]
    demo = demo_class(
        original_demo_path=str(clip.processed_demo_path()),
        target_object=transfer.target_object,
        start_timestep=clip.start_frame,
        end_timestep=clip.end_frame,
        skip=1,
        target_urdf=processor_cfg["target_urdf_path"],
        reference_urdf=processor_cfg["reference_urdf"],
        run_dir=workspace / "transfer_trajectories" / transfer.transfer_id / "processor",
    )
    return demo


def _materialize_transfer_support(
    transfer: ArcticTransferSpec, workspace: Path
) -> Path:
    asset_dir = transfer.support_block_asset_dir(workspace)
    obj_path = write_box_obj(
        asset_dir / "support_block.obj", transfer.support_block_size
    )
    urdf_path = Path(
        create_urdf_from_obj(
            obj_path=str(obj_path.resolve()),
            save_dir=str(asset_dir),
            scale=1.0,
        )
    ).resolve()
    expected = transfer.support_block_urdf_path(workspace).resolve()
    if urdf_path != expected or not urdf_path.is_file():
        raise FileNotFoundError(
            f"Expected transfer support URDF at {expected}, got {urdf_path}."
        )
    return urdf_path


def _target_first_frame_bounds(
    motion: dict[str, np.ndarray], target_urdf: Path
) -> np.ndarray:
    object_index = 0
    position = np.asarray(motion["scene_object_pos"][0, object_index], dtype=np.float64)
    quat_wxyz = np.asarray(
        motion["scene_object_quat"][0, object_index], dtype=np.float64
    )
    if position.shape != (3,) or quat_wxyz.shape != (4,):
        raise ValueError(
            "Expected first-frame target position/quaternion shapes (3,) and (4,), "
            f"got {position.shape} and {quat_wxyz.shape}."
        )

    names = np.asarray(motion["scene_object_arti_names"][object_index], dtype=object)
    values = np.asarray(
        motion["scene_object_arti_vals"][0, object_index], dtype=np.float64
    )
    valid = np.asarray(
        motion["scene_object_arti_valid_mask"][object_index], dtype=bool
    )
    if names.shape != values.shape or names.shape != valid.shape:
        raise ValueError(
            "Expected matching target articulation name/value/valid shapes, got "
            f"{names.shape}, {values.shape}, and {valid.shape}."
        )
    joint_config = {
        str(name): float(value)
        for name, value, is_valid in zip(names, values, valid, strict=True)
        if is_valid
    }

    object_pose = np.eye(4, dtype=np.float64)
    object_pose[:3, :3] = Rotation.from_quat(
        quat_wxyz[[1, 2, 3, 0]]
    ).as_matrix()
    object_pose[:3, 3] = position

    urdf = URDF.load(str(target_urdf))
    mesh_fk = urdf.visual_trimesh_fk(cfg=joint_config)
    if not mesh_fk:
        mesh_fk = urdf.collision_trimesh_fk(cfg=joint_config)
    if not mesh_fk:
        raise ValueError(f"Target URDF has no visual or collision meshes: {target_urdf}.")

    world_vertices = []
    for mesh, link_pose in mesh_fk.items():
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            raise ValueError(
                f"Expected target mesh vertices with shape (V, 3), got {vertices.shape}."
            )
        vertices_h = np.concatenate(
            [vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1
        )
        world_vertices.append(
            (object_pose @ np.asarray(link_pose) @ vertices_h.T).T[:, :3]
        )
    points = np.concatenate(world_vertices, axis=0)
    bounds = np.stack([points.min(axis=0), points.max(axis=0)], axis=0)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError(f"Invalid first-frame target bounds: {bounds}.")
    return bounds


def _apply_transfer_scene_metadata(
    motion: dict[str, np.ndarray],
    transfer: ArcticTransferSpec,
    workspace: Path,
    target_name: str,
    target_urdf: Path,
) -> None:
    support_urdf = _materialize_transfer_support(transfer, workspace)
    target_bounds = _target_first_frame_bounds(motion, target_urdf)
    support_size = np.asarray(transfer.support_block_size, dtype=np.float32)
    support_xy_offset = np.asarray(
        transfer.support_block_xy_offset, dtype=np.float32
    )
    support_position = np.asarray(
        [
            0.5 * (target_bounds[0, 0] + target_bounds[1, 0])
            + support_xy_offset[0],
            0.5 * (target_bounds[0, 1] + target_bounds[1, 1])
            + support_xy_offset[1],
            target_bounds[0, 2] - 0.5 * support_size[2],
        ],
        dtype=np.float32,
    )

    n_frames = transfer.frame_count
    motion["traj_object_name"] = np.array(target_name, dtype=object)
    motion["traj_object_urdf_path"] = np.array(str(target_urdf), dtype=object)
    motion["arctic_clip_id"] = np.array(transfer.transfer_id, dtype=object)
    motion["arctic_source_clip_id"] = np.array(
        transfer.source_clip_id, dtype=object
    )
    motion["arctic_transfer_id"] = np.array(transfer.transfer_id, dtype=object)
    motion["scene_object_names"] = np.asarray(
        [target_name, "arctic_support_block"], dtype=object
    )
    motion["scene_object_urdf_paths"] = np.asarray(
        [str(target_urdf), str(support_urdf)], dtype=object
    )
    motion["scene_object_pos"][:, 1, :] = support_position
    motion["scene_object_quat"][:, 1, :] = np.asarray(
        [1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    motion["scene_object_friction"][1] = transfer.support_clip.support_block_friction
    motion["scene_transfer_support_block_size"] = support_size
    motion["scene_transfer_support_block_pos"] = support_position
    motion["scene_transfer_support_block_xy_offset"] = support_xy_offset
    if motion["scene_object_pos"].shape != (n_frames, 2, 3):
        raise ValueError(
            "Expected transfer scene-object positions shape "
            f"{(n_frames, 2, 3)}, got {motion['scene_object_pos'].shape}."
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    transfer = get_transfer(args.transfer)
    default_motion, default_sequence = _default_outputs(transfer, args.workspace)
    motion_output = args.motion_output or default_motion
    sequence_output = args.sequence_output or default_sequence
    _check_output(motion_output, args.overwrite)
    _check_output(sequence_output, args.overwrite)

    names, values, target_name, target_urdf = load_transfer_hands(
        transfer, args.workspace
    )
    demo = _load_transfer_demo(transfer, args.workspace)
    motion = build_motion_payload(
        transfer.source_clip,
        args.workspace,
        names,
        values,
        demo,
    )
    _apply_transfer_scene_metadata(
        motion, transfer, args.workspace, target_name, target_urdf
    )
    sequence = build_sequence_payload(motion)
    sequence["arctic_source_clip_id"] = motion["arctic_source_clip_id"].copy()
    sequence["arctic_transfer_id"] = motion["arctic_transfer_id"].copy()

    np.savez_compressed(motion_output, **motion)
    np.save(sequence_output, sequence, allow_pickle=True)
    print(f"Wrote transfer motion: {motion_output}")
    print(f"Wrote transfer RL sequence: {sequence_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
