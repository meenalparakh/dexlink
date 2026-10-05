from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from dexlink.utils.retargeting.render_utils import TrimeshInstance, render_centered_four_views


def _load_trimesh(mesh_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(
            f"Expected {mesh_path!r} to load as trimesh.Trimesh, got {type(loaded)!r}."
        )
    return loaded


def _parse_pose_values(pose_values: list[str] | None) -> list[np.ndarray]:
    if pose_values is None:
        return []

    poses = []
    for pose_str in pose_values:
        pose_entries = np.fromstring(pose_str, sep=" ", dtype=np.float32)
        if pose_entries.shape != (16,):
            raise ValueError(
                "Each --pose must contain exactly 16 whitespace-separated numbers "
                f"for a row-major 4x4 transform. Got {pose_entries.shape[0]} values."
            )
        poses.append(pose_entries.reshape(4, 4))
    return poses


def _make_default_scene() -> list[TrimeshInstance]:
    cube_size_m = 0.10
    z_offsets_m = (0.0, cube_size_m, 2.0 * cube_size_m)
    colors_rgba = (
        np.array([255, 0, 0, 255], dtype=np.uint8),
        np.array([0, 255, 0, 255], dtype=np.uint8),
        np.array([0, 0, 255, 255], dtype=np.uint8),
    )

    mesh_instances = []
    for z_offset_m, color_rgba in zip(z_offsets_m, colors_rgba):
        mesh = trimesh.creation.box(extents=(cube_size_m, cube_size_m, cube_size_m))
        mesh.visual.face_colors = np.tile(color_rgba[None, :], (len(mesh.faces), 1))

        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = z_offset_m
        mesh_instances.append(TrimeshInstance(mesh=mesh, pose=pose))

    return mesh_instances


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render a set of posed trimesh meshes from left/right/front/back views "
            "into a single tiled image."
        )
    )
    parser.add_argument(
        "--mesh",
        action="append",
        default=None,
        help="Path to a mesh file. Repeat this flag to add multiple meshes.",
    )
    parser.add_argument(
        "--pose",
        action="append",
        default=None,
        help=(
            "Optional row-major 4x4 transform for a mesh as 16 whitespace-separated "
            "numbers. If provided, repeat once per --mesh in matching order."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/tmp/render.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width of each sub-image in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of each sub-image in pixels.",
    )
    args = parser.parse_args()

    mesh_paths: list[str] = [] if args.mesh is None else args.mesh
    poses = _parse_pose_values(args.pose)
    if len(poses) not in {0, len(mesh_paths)}:
        raise ValueError(
            f"Expected either 0 or {len(mesh_paths)} poses, got {len(poses)}."
        )

    if len(mesh_paths) == 0:
        mesh_instances = _make_default_scene()
    else:
        mesh_instances = []
        for idx, mesh_path in enumerate(mesh_paths):
            pose = np.eye(4, dtype=np.float32) if len(poses) == 0 else poses[idx]
            mesh_instances.append(
                TrimeshInstance(mesh=_load_trimesh(mesh_path), pose=pose)
            )

    rendered_image = render_centered_four_views(
        mesh_instances=mesh_instances,
        image_width=args.width,
        image_height=args.height,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_image.save(out_path)
    print(f"Saved render to {out_path.resolve()}")


if __name__ == "__main__":
    main()
