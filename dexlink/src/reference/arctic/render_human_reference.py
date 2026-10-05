#!/usr/bin/env python
"""Render the source ARCTIC human hands and object for a benchmark interval."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from urdfpy import URDF

from dexlink.utils.retargeting.path_utils import get_external_data_dir, process_urdf
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE, ArcticClipSpec, get_clip
from dexlink.src.reference.arctic.transfer_config import get_transfer


@dataclass(frozen=True)
class RenderHumanReferenceArgs:
    clip: str | None
    transfer: str | None
    workspace: Path
    output: Path
    width: int
    height: int
    fps: int


def parse_args(argv: list[str] | None = None) -> RenderHumanReferenceArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    benchmark = parser.add_mutually_exclusive_group(required=True)
    benchmark.add_argument("--clip")
    benchmark.add_argument("--transfer")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    raw = parser.parse_args(argv)
    if raw.width <= 0 or raw.height <= 0 or raw.fps <= 0:
        raise ValueError(
            "Expected positive --width, --height, and --fps; got "
            f"{raw.width}, {raw.height}, and {raw.fps}."
        )
    return RenderHumanReferenceArgs(
        clip=None if raw.clip is None else str(raw.clip),
        transfer=None if raw.transfer is None else str(raw.transfer),
        workspace=raw.workspace.resolve(),
        output=raw.output.resolve(),
        width=int(raw.width),
        height=int(raw.height),
        fps=int(raw.fps),
    )


def _source_clip(args: RenderHumanReferenceArgs) -> ArcticClipSpec:
    if args.clip is not None:
        return get_clip(args.clip)
    if args.transfer is None:
        raise ValueError("Expected exactly one clip or transfer.")
    return get_transfer(args.transfer).source_clip


def _load_human_reference(
    clip: ArcticClipSpec, workspace: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], URDF]:
    payload = np.load(clip.processed_demo_path(), allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected an ARCTIC dictionary payload in {clip.processed_demo_path()}, "
            f"got {type(payload)!r}."
        )
    world_coord = payload.get("world_coord")
    params = payload.get("params")
    if not isinstance(world_coord, dict) or not isinstance(params, dict):
        raise ValueError(
            f"Expected world_coord and params dictionaries in {clip.processed_demo_path()}."
        )
    frame_slice = slice(clip.start_frame, clip.end_frame)
    left = np.asarray(world_coord["verts.left"][frame_slice], dtype=np.float32)
    right = np.asarray(world_coord["verts.right"][frame_slice], dtype=np.float32)
    object_urdf = URDF.load(
        str(process_urdf(clip.object_urdf_path(workspace), append_jobname=True))
    )
    object_joint_names = tuple(
        joint.name for joint in object_urdf.joints if joint.joint_type != "fixed"
    )
    object_articulation = np.asarray(params["obj_arti"][frame_slice], dtype=np.float32)
    if object_articulation.ndim == 1:
        object_articulation = object_articulation[:, None]
    object_rotation = np.asarray(params["obj_rot"][frame_slice], dtype=np.float32)
    motion = {
        "obj_pos": np.asarray(params["obj_trans"][frame_slice], dtype=np.float32)
        / 1000.0,
        "obj_quat": np.roll(Rotation.from_rotvec(object_rotation).as_quat(), 1, axis=-1),
        "obj_arti": object_articulation,
        "joint_names": np.asarray(object_joint_names, dtype=object),
    }
    expected_hand_shape = (clip.frame_count, 778, 3)
    if left.shape != expected_hand_shape or right.shape != expected_hand_shape:
        raise ValueError(
            f"Expected MANO hand vertices shaped {expected_hand_shape}; "
            f"got left={left.shape}, right={right.shape}."
        )
    if np.asarray(motion["obj_pos"]).shape != (clip.frame_count, 3):
        raise ValueError(
            "Expected object positions shaped "
            f"{(clip.frame_count, 3)}, got {np.asarray(motion['obj_pos']).shape}."
        )
    expected_articulation_shape = (clip.frame_count, len(object_joint_names))
    if motion["obj_arti"].shape != expected_articulation_shape:
        raise ValueError(
            f"Expected object articulation shaped {expected_articulation_shape}, "
            f"got {motion['obj_arti'].shape}."
        )
    return left, right, motion, object_urdf


def _load_mano_faces(side: str) -> np.ndarray:
    if side not in {"left", "right"}:
        raise ValueError(f"Expected MANO side 'left' or 'right', got {side!r}.")
    path = get_external_data_dir() / "mano" / "arctic_faces" / f"{side}_faces.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing precomputed ARCTIC MANO topology at {path}. "
            "Copy left_faces.npy and right_faces.npy from the ARCTIC data root."
        )
    faces = np.asarray(np.load(path), dtype=np.int64)
    if faces.shape != (1538, 3) or faces.min() < 0 or faces.max() >= 778:
        raise ValueError(
            f"Expected MANO faces shaped (1538, 3) over 778 vertices, got "
            f"shape={faces.shape}, min={faces.min()}, max={faces.max()}."
        )
    return faces


def _pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64)
    if position.shape != (3,) or quaternion_wxyz.shape != (4,):
        raise ValueError(
            "Expected position (3,) and quaternion (4,), got "
            f"{position.shape} and {quaternion_wxyz.shape}."
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(
        np.roll(quaternion_wxyz, -1)
    ).as_matrix()
    transform[:3, 3] = position
    return transform


def _object_meshes(
    object_urdf: URDF, motion: dict[str, np.ndarray], frame_index: int
) -> list[trimesh.Trimesh]:
    joint_names = tuple(str(value) for value in motion["joint_names"].tolist())
    joint_values = np.asarray(motion["obj_arti"][frame_index], dtype=np.float64).reshape(-1)
    if joint_values.shape != (len(joint_names),):
        raise ValueError(
            f"Expected {len(joint_names)} object joint values, got {joint_values.shape}."
        )
    joint_cfg = {
        name: float(value) for name, value in zip(joint_names, joint_values, strict=True)
    }
    root_pose = _pose_matrix(
        motion["obj_pos"][frame_index], motion["obj_quat"][frame_index]
    )
    meshes = []
    for mesh, link_pose in object_urdf.visual_trimesh_fk(cfg=joint_cfg).items():
        posed = mesh.copy()
        posed.apply_transform(root_pose @ np.asarray(link_pose, dtype=np.float64))
        meshes.append(posed)
    if not meshes:
        raise ValueError("The ARCTIC object URDF contains no visual meshes.")
    return meshes


def _look_at(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(
        camera_position, dtype=np.float64
    )
    norm = float(np.linalg.norm(forward))
    if norm <= 1.0e-8:
        raise ValueError("Camera position must differ from the target.")
    forward /= norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = -forward
    x_axis = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = camera_position
    return pose


def _camera_pose(
    object_urdf: URDF,
    motion: dict[str, np.ndarray],
) -> np.ndarray:
    # Keep the view centered on the manipulated object. ARCTIC records both
    # hands even when one begins far outside the interaction region, so fitting
    # the camera to all hand vertices makes the actual interaction unreadable.
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    frame_count = int(np.asarray(motion["obj_pos"]).shape[0])
    sample_indices = np.unique(
        np.linspace(0, frame_count - 1, min(frame_count, 25)).round().astype(int)
    )
    for frame_index in sample_indices.tolist():
        for mesh in _object_meshes(object_urdf, motion, frame_index):
            bounds_min = np.minimum(bounds_min, mesh.bounds[0])
            bounds_max = np.maximum(bounds_max, mesh.bounds[1])
    center = 0.5 * (bounds_min + bounds_max)
    extent = bounds_max - bounds_min
    radius = max(float(np.linalg.norm(extent)), 0.25)
    camera_position = center + radius * np.array([1.15, -1.65, 0.85])
    return _look_at(camera_position, center)


def _colored_hand_mesh(
    vertices: np.ndarray, faces: np.ndarray, rgba: tuple[int, int, int, int]
) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = np.tile(
        np.asarray(rgba, dtype=np.uint8)[None, :], (faces.shape[0], 1)
    )
    return mesh


def render(args: RenderHumanReferenceArgs) -> Path:
    import pyrender

    clip = _source_clip(args)
    left, right, motion, object_urdf = _load_human_reference(clip, args.workspace)
    left_faces = _load_mano_faces("left")
    right_faces = _load_mano_faces("right")
    camera_pose = _camera_pose(object_urdf, motion)
    renderer = pyrender.OffscreenRenderer(
        viewport_width=args.width, viewport_height=args.height
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(
            args.output, fps=args.fps, codec="libx264", macro_block_size=None
        ) as writer:
            for frame_index in range(clip.frame_count):
                scene = pyrender.Scene(
                    bg_color=np.array([245, 245, 245, 255], dtype=np.uint8),
                    ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
                )
                meshes = [
                    *_object_meshes(object_urdf, motion, frame_index),
                    _colored_hand_mesh(
                        left[frame_index], left_faces, (46, 114, 184, 255)
                    ),
                    _colored_hand_mesh(
                        right[frame_index], right_faces, (221, 91, 68, 255)
                    ),
                ]
                for mesh in meshes:
                    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
                scene.add(
                    pyrender.PerspectiveCamera(yfov=np.deg2rad(42.0)),
                    pose=camera_pose,
                )
                scene.add(
                    pyrender.DirectionalLight(color=np.ones(3), intensity=2.4),
                    pose=camera_pose,
                )
                color, _ = renderer.render(
                    scene,
                    flags=pyrender.RenderFlags.RGBA
                    | pyrender.RenderFlags.SKIP_CULL_FACES,
                )
                writer.append_data(color[:, :, :3])
    finally:
        renderer.delete()
    if not args.output.is_file():
        raise FileNotFoundError(f"Failed to render human reference to {args.output}.")
    print(f"[arctic-human-reference] {clip.clip_id} -> {args.output}")
    return args.output


def main(argv: list[str] | None = None) -> int:
    render(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
