from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# Default to EGL for headless cluster rendering. Users can override this with the
# PYOPENGL_PLATFORM environment variable or --pyopengl-platform.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import seaborn as sns
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.visualizations.retargeting.render_oakink_last_frame_views import (
    _copy_with_color,
    _object_pose_dict_for_frame,
    _pose_from_pos_rot,
    _resolve_active_hands,
    _resolve_robot_base_pose,
    _resolve_robot_model,
    _resolve_robot_urdf_path,
)
from dexlink.utils.retargeting.path_utils import get_logs_dir, process_urdf
from dexlink.utils.retargeting.render_utils import TrimeshInstance, render_centered_four_views

# NumPy 2.x compatibility for urdfpy.
if not hasattr(np, "product"):
    np.product = np.prod


@dataclass(frozen=True)
class RenderSequenceMotionArgs:
    categories: list[str]
    motion_paths: list[Path]
    root_dir: Path
    out_name: str
    mesh_type: str
    robot_model: str
    width: int
    height: int
    camera_distance: float
    camera_elevation: float
    pyopengl_platform: str | None
    skip_existing: bool


def _unwrap_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _load_sequence_motion(path: Path) -> dict[str, np.ndarray]:
    if path.suffix != ".npz":
        raise ValueError(f"Expected an .npz motion file, got {path}.")
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
            f"{path}: expected sequence_motion.npz keys {required_keys}, "
            f"missing {missing}. Available keys: {sorted(motion)}"
        )

    asset_pos = np.asarray(motion["asset_pos"], dtype=np.float32)
    asset_quat = np.asarray(motion["asset_quat"], dtype=np.float32)
    asset_arti_vals = np.asarray(motion["asset_arti_vals"], dtype=np.float32)
    asset_arti_names = np.asarray(motion["asset_arti_names"], dtype=object)
    left_joint_names = np.asarray(motion["left_joint_names"], dtype=object)
    right_joint_names = np.asarray(motion["right_joint_names"], dtype=object)
    left_joint_vals = np.asarray(motion["left_joint_vals"], dtype=np.float32)
    right_joint_vals = np.asarray(motion["right_joint_vals"], dtype=np.float32)

    if asset_pos.ndim != 2 or asset_pos.shape[1] != 3:
        raise ValueError(
            f"{path}: expected asset_pos shape (T, 3), got {asset_pos.shape}."
        )
    if asset_quat.shape != (asset_pos.shape[0], 4):
        raise ValueError(
            f"{path}: expected asset_quat shape (T, 4) matching asset_pos, "
            f"got {asset_quat.shape} and {asset_pos.shape}."
        )
    if asset_arti_vals.ndim != 2 or asset_arti_vals.shape[0] != asset_pos.shape[0]:
        raise ValueError(
            f"{path}: expected asset_arti_vals shape (T, n_obj_joints) matching "
            f"asset_pos, got {asset_arti_vals.shape} and {asset_pos.shape}."
        )
    if asset_arti_vals.shape[1] != len(asset_arti_names):
        raise ValueError(
            f"{path}: expected asset_arti_vals width to match asset_arti_names length, "
            f"got {asset_arti_vals.shape[1]} and {len(asset_arti_names)}."
        )
    if "scene_object_pos" in motion:
        scene_required_keys = (
            "scene_object_ids",
            "scene_object_urdf_paths",
            "scene_object_pos",
            "scene_object_quat",
            "scene_object_arti_vals",
            "scene_object_arti_valid_mask",
            "scene_object_arti_names",
        )
        scene_missing = [key for key in scene_required_keys if key not in motion]
        if scene_missing:
            raise ValueError(
                f"{path}: scene_object_pos is present, so expected multi-object keys "
                f"{scene_required_keys}; missing {scene_missing}."
            )
        scene_object_ids = np.asarray(motion["scene_object_ids"], dtype=object)
        scene_object_urdf_paths = np.asarray(
            motion["scene_object_urdf_paths"], dtype=object
        )
        scene_object_pos = np.asarray(motion["scene_object_pos"], dtype=np.float32)
        scene_object_quat = np.asarray(motion["scene_object_quat"], dtype=np.float32)
        scene_object_arti_vals = np.asarray(
            motion["scene_object_arti_vals"], dtype=np.float32
        )
        scene_object_arti_valid_mask = np.asarray(
            motion["scene_object_arti_valid_mask"], dtype=bool
        )
        scene_object_arti_names = np.asarray(
            motion["scene_object_arti_names"], dtype=object
        )
        if scene_object_pos.ndim != 3 or scene_object_pos.shape[2] != 3:
            raise ValueError(
                f"{path}: expected scene_object_pos shape (T, N, 3), "
                f"got {scene_object_pos.shape}."
            )
        if scene_object_pos.shape[0] != asset_pos.shape[0]:
            raise ValueError(
                f"{path}: expected scene_object_pos T to match asset_pos T="
                f"{asset_pos.shape[0]}, got {scene_object_pos.shape[0]}."
            )
        if scene_object_quat.shape != (
            scene_object_pos.shape[0],
            scene_object_pos.shape[1],
            4,
        ):
            raise ValueError(
                f"{path}: expected scene_object_quat shape (T, N, 4) matching "
                f"scene_object_pos, got {scene_object_quat.shape}."
            )
        if scene_object_arti_vals.ndim != 3:
            raise ValueError(
                f"{path}: expected scene_object_arti_vals shape (T, N, Amax), "
                f"got {scene_object_arti_vals.shape}."
            )
        if scene_object_arti_vals.shape[:2] != scene_object_pos.shape[:2]:
            raise ValueError(
                f"{path}: expected scene_object_arti_vals first two dimensions to "
                f"match scene_object_pos, got {scene_object_arti_vals.shape} and "
                f"{scene_object_pos.shape}."
            )
        if scene_object_arti_valid_mask.shape != scene_object_arti_vals.shape[1:]:
            raise ValueError(
                f"{path}: expected scene_object_arti_valid_mask shape (N, Amax), "
                f"got {scene_object_arti_valid_mask.shape} for "
                f"scene_object_arti_vals {scene_object_arti_vals.shape}."
            )
        n_objects = int(scene_object_pos.shape[1])
        if len(scene_object_ids) != n_objects:
            raise ValueError(
                f"{path}: expected {n_objects} scene_object_ids, got "
                f"{len(scene_object_ids)}."
            )
        if len(scene_object_urdf_paths) != n_objects:
            raise ValueError(
                f"{path}: expected {n_objects} scene_object_urdf_paths, got "
                f"{len(scene_object_urdf_paths)}."
            )
        if len(scene_object_arti_names) != n_objects:
            raise ValueError(
                f"{path}: expected {n_objects} scene_object_arti_names entries, got "
                f"{len(scene_object_arti_names)}."
            )
        motion["scene_object_ids"] = scene_object_ids
        motion["scene_object_urdf_paths"] = scene_object_urdf_paths
        motion["scene_object_pos"] = scene_object_pos
        motion["scene_object_quat"] = scene_object_quat
        motion["scene_object_arti_vals"] = scene_object_arti_vals
        motion["scene_object_arti_valid_mask"] = scene_object_arti_valid_mask
        motion["scene_object_arti_names"] = scene_object_arti_names

    for side, names, vals in (
        ("left", left_joint_names, left_joint_vals),
        ("right", right_joint_names, right_joint_vals),
    ):
        if vals.ndim != 2:
            raise ValueError(
                f"{path}: expected {side}_joint_vals shape (T, nq), got {vals.shape}."
            )
        if vals.shape[0] != asset_pos.shape[0]:
            raise ValueError(
                f"{path}: expected {side}_joint_vals first dimension to match "
                f"asset_pos T={asset_pos.shape[0]}, got {vals.shape[0]}."
            )
        if vals.shape[1] != len(names):
            raise ValueError(
                f"{path}: expected {side}_joint_vals width to match "
                f"{side}_joint_names length, got {vals.shape[1]} and {len(names)}."
            )

    return {
        **motion,
        "obj_pos": asset_pos,
        "obj_quat": asset_quat,
        "obj_arti": asset_arti_vals,
        "joint_names": asset_arti_names,
        "left_jnames": left_joint_names,
        "left_jvals": left_joint_vals,
        "right_jnames": right_joint_names,
        "right_jvals": right_joint_vals,
    }


DEFAULT_CATEGORIES = [
    "bottle",
    "bowl",
    "cup",
    "drawer",
    "eraser",
    "fruit",
    "headphones",
    "marker",
    "mouse",
    "mug",
    "notebook",
    "screwdriver",
    "spoon",
    "stapler",
    "toothbrush",
    "trigger_sprayer",
]


def _iter_sequence_motion_paths(root_dir: Path, categories: list[str]) -> list[Path]:
    paths: list[Path] = []
    if categories[0] == "all":
        categories = DEFAULT_CATEGORIES

    for category in categories:
        category_dir = root_dir / category
        if not category_dir.is_dir():
            print(f"Missing category directory: {category_dir}")
            continue
        paths.extend(sorted(category_dir.glob("*/sequence_motion.npz")))
    return paths


def _scene_object_joint_dict_for_frame(
    motion: dict[str, np.ndarray], frame_idx: int, object_idx: int
) -> dict[str, float]:
    raw_names = motion["scene_object_arti_names"][object_idx]
    if isinstance(raw_names, np.ndarray):
        names = [str(name) for name in raw_names.tolist()]
    else:
        names = [str(name) for name in list(raw_names)]
    valid_mask = np.asarray(
        motion["scene_object_arti_valid_mask"][object_idx], dtype=bool
    )
    vals = np.asarray(
        motion["scene_object_arti_vals"][frame_idx, object_idx], dtype=np.float32
    )
    if int(valid_mask.sum()) != len(names):
        raise ValueError(
            "scene_object_arti_valid_mask true count must match "
            f"scene_object_arti_names length for object index {object_idx}; "
            f"got {int(valid_mask.sum())} and {len(names)}."
        )
    return {
        name: float(value)
        for name, value in zip(names, vals[valid_mask].reshape(-1).tolist())
    }


def _scene_object_pose_for_frame(
    motion: dict[str, np.ndarray], frame_idx: int, object_idx: int
) -> np.ndarray:
    pos = np.asarray(
        motion["scene_object_pos"][frame_idx, object_idx], dtype=np.float32
    )
    quat_wxyz = np.asarray(
        motion["scene_object_quat"][frame_idx, object_idx], dtype=np.float32
    )
    rot = R.from_quat(np.roll(quat_wxyz, -1)).as_matrix().astype(np.float32)
    return _pose_from_pos_rot(pos, rot)


def _palette_rgba(palette: str, n_colors: int) -> list[np.ndarray]:
    if n_colors <= 0:
        raise ValueError(f"Expected n_colors > 0, got {n_colors}.")
    colors = sns.color_palette(palette=palette, n_colors=n_colors)
    return [
        np.array([round(channel * 255.0) for channel in color] + [255], dtype=np.uint8)
        for color in colors
    ]


def _single_palette_rgba(palette: str, color_index: int) -> np.ndarray:
    colors = sns.color_palette(palette=palette)
    if color_index < 0 or color_index >= len(colors):
        raise ValueError(
            f"Color index {color_index} out of bounds for {palette!r} palette "
            f"with {len(colors)} colors."
        )
    color = colors[color_index]
    return np.array(
        [round(channel * 255.0) for channel in color] + [255],
        dtype=np.uint8,
    )


def _fk_mesh_instances_with_link_palette(
    *,
    urdf_inst: URDF,
    joint_dict: dict[str, float],
    base_pose: np.ndarray,
    mesh_type: str,
    palette: str,
) -> list[TrimeshInstance]:
    if mesh_type == "visual":
        fk = (
            urdf_inst.visual_trimesh_fk()
            if len(joint_dict) == 0
            else urdf_inst.visual_trimesh_fk(joint_dict)
        )
    elif mesh_type == "collision":
        fk = (
            urdf_inst.collision_trimesh_fk()
            if len(joint_dict) == 0
            else urdf_inst.collision_trimesh_fk(cfg=joint_dict)
        )
    else:
        raise ValueError(f"Unsupported mesh_type {mesh_type!r}.")

    colors = _palette_rgba(palette=palette, n_colors=len(fk))
    mesh_instances: list[TrimeshInstance] = []
    for color_rgba, (mesh, local_pose) in zip(colors, fk.items()):
        mesh_instances.append(
            TrimeshInstance(
                mesh=_copy_with_color(mesh, color_rgba),
                pose=base_pose @ np.asarray(local_pose, dtype=np.float32),
            )
        )
    return mesh_instances


def _fk_mesh_instances_with_color(
    *,
    urdf_inst: URDF,
    joint_dict: dict[str, float],
    base_pose: np.ndarray,
    mesh_type: str,
    color_rgba: np.ndarray,
) -> list[TrimeshInstance]:
    if mesh_type == "visual":
        fk = (
            urdf_inst.visual_trimesh_fk()
            if len(joint_dict) == 0
            else urdf_inst.visual_trimesh_fk(joint_dict)
        )
    elif mesh_type == "collision":
        fk = (
            urdf_inst.collision_trimesh_fk()
            if len(joint_dict) == 0
            else urdf_inst.collision_trimesh_fk(cfg=joint_dict)
        )
    else:
        raise ValueError(f"Unsupported mesh_type {mesh_type!r}.")

    return [
        TrimeshInstance(
            mesh=_copy_with_color(mesh, color_rgba),
            pose=base_pose @ np.asarray(local_pose, dtype=np.float32),
        )
        for mesh, local_pose in fk.items()
    ]


def _build_scene_for_sequence_motion(
    *,
    sequence_motion_path: Path,
    mesh_type: str,
    robot_model_arg: str,
) -> tuple[list[TrimeshInstance], str]:
    motion = _load_sequence_motion(sequence_motion_path)
    n_frames = int(np.asarray(motion["obj_pos"]).shape[0])
    if n_frames <= 0:
        raise ValueError(f"{sequence_motion_path}: expected at least one frame.")
    frame_idx = n_frames - 1

    active_hands = _resolve_active_hands(None, motion)
    robot_model = _resolve_robot_model(None, motion, robot_model_arg)
    robot_base_pose = _resolve_robot_base_pose(None, motion)
    mesh_instances: list[TrimeshInstance] = []
    for side in ("left", "right"):
        if not active_hands[side]:
            continue
        joint_names = motion[f"{side}_jnames"]
        joint_vals = motion[f"{side}_jvals"]
        if len(joint_names) == 0:
            raise ValueError(
                f"{sequence_motion_path}: {side} hand is active but "
                f"{side}_joint_names is empty."
            )
        joint_dict = {
            str(name): float(value)
            for name, value in zip(joint_names.tolist(), joint_vals[frame_idx].tolist())
        }
        robot_urdf = URDF.load(str(_resolve_robot_urdf_path(robot_model, side)))
        mesh_instances.extend(
            _fk_mesh_instances_with_link_palette(
                urdf_inst=robot_urdf,
                joint_dict=joint_dict,
                base_pose=robot_base_pose,
                mesh_type=mesh_type,
                palette="Purples_d",
            )
        )

    if "scene_object_pos" in motion:
        object_colors = _palette_rgba(
            palette="YlGnBu",
            n_colors=int(motion["scene_object_pos"].shape[1]),
        )
        for object_idx, object_color in enumerate(object_colors):
            object_urdf_path = Path(
                str(_unwrap_scalar(motion["scene_object_urdf_paths"][object_idx]))
            )
            object_urdf = URDF.load(str(process_urdf(object_urdf_path)))
            mesh_instances.extend(
                _fk_mesh_instances_with_color(
                    urdf_inst=object_urdf,
                    joint_dict=_scene_object_joint_dict_for_frame(
                        motion, frame_idx, object_idx
                    ),
                    base_pose=_scene_object_pose_for_frame(
                        motion, frame_idx, object_idx
                    ),
                    mesh_type=mesh_type,
                    color_rgba=object_color,
                )
            )
    else:
        object_pose, object_joint_dict = _object_pose_dict_for_frame(motion, frame_idx)
        object_urdf_path = Path(str(_unwrap_scalar(motion["traj_object_urdf_path"])))
        object_urdf = URDF.load(str(process_urdf(object_urdf_path)))
        mesh_instances.extend(
            _fk_mesh_instances_with_color(
                urdf_inst=object_urdf,
                joint_dict=object_joint_dict,
                base_pose=object_pose,
                mesh_type=mesh_type,
                color_rgba=_single_palette_rgba(palette="YlGnBu", color_index=1),
            )
        )
    if len(mesh_instances) == 0:
        raise ValueError(f"No meshes were generated for {sequence_motion_path}.")

    label = (
        f"{sequence_motion_path}: frame={frame_idx} "
        f"robot_model={robot_model} mesh_type={mesh_type}"
    )
    return mesh_instances, label


def _parse_args() -> RenderSequenceMotionArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Render last-frame four-view images directly from "
            "logs/experiment_category_retargetings/<category>/<run_dir>/sequence_motion.npz."
        )
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help=(
            "Experiment category under logs/experiment_category_retargetings. "
            "Repeat for multiple categories."
        ),
    )
    parser.add_argument(
        "--motion-path",
        action="append",
        type=Path,
        default=[],
        help=(
            "Direct path to a sequence motion .npz file. Repeat to render multiple "
            "files. When provided, --category is not required."
        ),
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=get_logs_dir() / "experiment_category_retargetings",
        help="Root directory containing category subdirectories.",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="sequence_motion_last_frame_four_views.png",
        help="Output filename saved in each run directory.",
    )
    parser.add_argument(
        "--mesh-type",
        type=str,
        default="visual",
        choices=["visual", "collision"],
        help="URDF mesh type used for rendering.",
    )
    parser.add_argument(
        "--robot-model",
        type=str,
        default="auto",
        choices=["auto", "inspire", "xhand", "franka_xhand"],
        help="Robot URDF family to render. 'auto' uses sequence metadata.",
    )
    parser.add_argument("--width", type=int, default=640, help="View width in pixels.")
    parser.add_argument(
        "--height", type=int, default=480, help="View height in pixels."
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=0.5,
        help="Camera distance used for each rendered view in meters.",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=0.25,
        help="Camera elevation used for each rendered view in meters.",
    )
    parser.add_argument(
        "--pyopengl-platform",
        type=str,
        default=None,
        choices=["egl", "osmesa", "pyglet"],
        help=(
            "PyOpenGL backend for pyrender. Defaults to PYOPENGL_PLATFORM if set, "
            "else egl for headless cluster rendering."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run directory when the output image already exists.",
    )
    args = parser.parse_args()
    if len(args.motion_path) == 0 and len(args.category) == 0:
        raise ValueError("Provide at least one --motion-path or --category.")
    return RenderSequenceMotionArgs(
        categories=args.category,
        motion_paths=[path.resolve() for path in args.motion_path],
        root_dir=args.root_dir.resolve(),
        out_name=args.out_name,
        mesh_type=args.mesh_type,
        robot_model=args.robot_model,
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        camera_elevation=args.camera_elevation,
        pyopengl_platform=args.pyopengl_platform,
        skip_existing=args.skip_existing,
    )


def main() -> None:
    args = _parse_args()
    if args.pyopengl_platform is not None:
        os.environ["PYOPENGL_PLATFORM"] = args.pyopengl_platform

    print(f"Using PYOPENGL_PLATFORM={os.environ['PYOPENGL_PLATFORM']}")
    sequence_motion_paths = (
        args.motion_paths
        if len(args.motion_paths) > 0
        else _iter_sequence_motion_paths(args.root_dir, args.categories)
    )
    if len(sequence_motion_paths) == 0:
        raise FileNotFoundError(
            f"No sequence_motion.npz files found under {args.root_dir} for "
            f"categories {args.categories}."
        )

    print(f"Found {len(sequence_motion_paths)} sequence_motion.npz files.")
    rendered = 0
    skipped = 0
    for sequence_motion_path in sequence_motion_paths:
        out_path = sequence_motion_path.with_name(args.out_name)
        if args.skip_existing and out_path.exists():
            skipped += 1
            print(f"Skipping existing render: {out_path}")
            continue

        mesh_instances, label = _build_scene_for_sequence_motion(
            sequence_motion_path=sequence_motion_path,
            mesh_type=args.mesh_type,
            robot_model_arg=args.robot_model,
        )
        image = render_centered_four_views(
            mesh_instances=mesh_instances,
            image_width=args.width,
            image_height=args.height,
            camera_distance_m=args.camera_distance,
            camera_elevation_m=args.camera_elevation,
            grid_columns=4,
        )
        image.save(out_path)
        rendered += 1
        print(f"Saved {label} to {out_path}")

    print(f"Done. rendered={rendered}, skipped={skipped}")


if __name__ == "__main__":
    main()
