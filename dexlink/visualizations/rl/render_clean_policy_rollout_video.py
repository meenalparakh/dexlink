#!/usr/bin/env python
"""Render one recorded policy rollout as a clean, simulator-free mesh video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dexlink_matplotlib")

import imageio.v2 as imageio
import numpy as np
import trimesh
from urdfpy import URDF

from dexlink.visualizations.rl.rollout_rendering import (
    CameraSpec,
    OBJECT_COLOR_RGBA,
    fk_mesh_instances_with_color,
    fk_robot_link_instances,
    hand_config_source_by_side,
    load_episode_from_eval,
    load_exported_npy_payload,
    localize_path,
    look_at,
    render_panel,
    resolve_visualized_joint_motion,
    rollout_motion_from_reference,
    scene_object_joint_dict_for_frame,
    scene_object_pose_for_frame,
    transformed_meshes,
    unwrap_scalar,
)
from dexlink.visualizations.retargeting.render_oakink_last_frame_views import (
    ROBOT_URDFS,
    _joint_dict_for_frame,
    _resolve_active_hands,
    _resolve_robot_base_pose,
    _resolve_robot_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAMERA = CameraSpec(direction=(-1.0, 1.0, 0.30), margin=1.22)
OBJECT_COLORS: dict[str, np.ndarray] = {
    "red": np.array([198, 65, 61, 255], dtype=np.uint8),
    "green": np.array([73, 142, 91, 255], dtype=np.uint8),
    "blue": np.array([70, 105, 184, 255], dtype=np.uint8),
    "peg": np.array([213, 94, 62, 255], dtype=np.uint8),
    "post": np.array([148, 151, 157, 255], dtype=np.uint8),
    "block": np.array([174, 177, 181, 255], dtype=np.uint8),
    "table": np.array([137, 143, 151, 255], dtype=np.uint8),
}
CameraMode = Literal["tracked", "fixed"]


@dataclass(frozen=True)
class RenderVideoArgs:
    eval_dir: Path
    output_path: Path
    metadata_path: Path
    cache_dir: Path
    task_id: str
    episode_index: int
    width: int
    height: int
    frame_stride: int
    fps: float
    camera_smoothing_window: int
    camera_mode: CameraMode


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    object_index: int
    urdf: URDF
    color_rgba: np.ndarray


@dataclass(frozen=True)
class HandModel:
    side: str
    urdf: URDF
    joint_names: tuple[str, ...]
    joint_values: np.ndarray


def parse_args(argv: list[str] | None = None) -> RenderVideoArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--camera-smoothing-window", type=int, default=7)
    parser.add_argument(
        "--camera-mode",
        choices=("tracked", "fixed"),
        default="tracked",
        help="Use the moving close-up camera or one fixed third-person camera.",
    )
    parsed = parser.parse_args(argv)
    args = RenderVideoArgs(
        eval_dir=parsed.eval_dir,
        output_path=parsed.output_path,
        metadata_path=parsed.metadata_path,
        cache_dir=parsed.cache_dir,
        task_id=str(parsed.task_id),
        episode_index=int(parsed.episode_index),
        width=int(parsed.width),
        height=int(parsed.height),
        frame_stride=int(parsed.frame_stride),
        fps=float(parsed.fps),
        camera_smoothing_window=int(parsed.camera_smoothing_window),
        camera_mode=parsed.camera_mode,
    )
    if args.episode_index < 0:
        raise ValueError(f"Expected --episode-index >= 0, got {args.episode_index}.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError(
            f"Expected positive video dimensions, got {args.width}x{args.height}."
        )
    if args.frame_stride <= 0:
        raise ValueError(f"Expected --frame-stride > 0, got {args.frame_stride}.")
    if args.fps <= 0.0:
        raise ValueError(f"Expected --fps > 0, got {args.fps}.")
    if args.camera_smoothing_window <= 0 or args.camera_smoothing_window % 2 == 0:
        raise ValueError(
            "Expected --camera-smoothing-window to be a positive odd integer, got "
            f"{args.camera_smoothing_window}."
        )
    return args


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, got {type(value).__name__}."
        )
    return value


def _reference_motion_path(eval_dir: Path) -> Path:
    config_path = eval_dir / "final_eval_exp_cfg.json"
    config = _load_json_object(config_path)
    try:
        raw_path = config["env"]["task_cfg"]["oakink_exported_path"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"Expected env.task_cfg.oakink_exported_path in {config_path}."
        ) from exc
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(
            f"Expected a non-empty reference-motion path in {config_path}, got "
            f"{raw_path!r}."
        )
    path = localize_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference motion from {config_path} does not exist: {path}"
        )
    return path


def _metric_config(eval_dir: Path) -> dict[str, Any]:
    config_path = eval_dir / "final_eval_exp_cfg.json"
    config = _load_json_object(config_path)
    try:
        value = config["env"]["task_cfg"]["metric_cfg"]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"Expected env.task_cfg.metric_cfg in {config_path}.") from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected metric_cfg to be a dict in {config_path}, got "
            f"{type(value).__name__}."
        )
    return value


def _semantic_object_ids(
    *,
    reference_motion: dict[str, object],
    metric_config: dict[str, Any],
    task_id: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    object_ids = tuple(
        str(unwrap_scalar(value))
        for value in np.asarray(reference_motion["scene_object_ids"], dtype=object)
    )
    static = np.asarray(reference_motion["scene_object_static"], dtype=bool)
    if static.shape != (len(object_ids),):
        raise ValueError(
            f"Expected scene_object_static shape {(len(object_ids),)}, got "
            f"{static.shape}."
        )

    semantic_ids = {
        object_id
        for object_id, is_static in zip(object_ids, static, strict=True)
        if not is_static
    }
    for key, value in metric_config.items():
        if key.endswith("_object_id") and isinstance(value, str) and value:
            semantic_ids.add(value)
        elif key.endswith("_object_ids") and isinstance(value, (list, tuple)):
            if not all(isinstance(item, str) and item for item in value):
                raise ValueError(
                    f"Expected non-empty string values for metric field {key!r}, "
                    f"got {value!r}."
                )
            semantic_ids.update(value)
    if _is_hammer_strike_task(task_id):
        semantic_ids.add("block")
    if "cube_strategy" in task_id:
        semantic_ids.add("table")
    if task_id.startswith("v3_05_hammer_to_wineglass_stem_"):
        semantic_ids.add("wineglass_support_block")

    unknown = sorted(semantic_ids - set(object_ids))
    if unknown:
        raise ValueError(
            f"Metric/task semantics request missing scene objects {unknown}; "
            f"available object ids are {object_ids}."
        )
    included = tuple(object_id for object_id in object_ids if object_id in semantic_ids)
    excluded = tuple(
        object_id for object_id in object_ids if object_id not in semantic_ids
    )
    if not included:
        raise ValueError(
            f"No semantic scene objects selected from available ids {object_ids}."
        )
    return included, excluded


def _camera_for_task(task_id: str) -> CameraSpec:
    if _is_hammer_strike_task(task_id):
        return CameraSpec(direction=(0.0, -1.0, 0.25), margin=1.20)
    if "cube_strategy" in task_id:
        return CameraSpec(direction=(1.0, 0.0, 0.45), margin=1.22)
    if "mounting" in task_id:
        return CameraSpec(direction=(0.0, -1.0, 0.38), margin=1.24)
    if "stack_rgb_blocks" in task_id:
        return CameraSpec(direction=(-1.0, 1.0, 0.55), margin=1.24)
    if "banana_handover" in task_id:
        return CameraSpec(direction=(0.0, -1.0, 0.35), margin=1.24)
    return DEFAULT_CAMERA


def _is_hammer_strike_task(task_id: str) -> bool:
    return task_id.startswith("v3_controllability_hammer2_")


def _color_for_object(object_id: str) -> np.ndarray:
    return OBJECT_COLORS.get(object_id, OBJECT_COLOR_RGBA).copy()


def _localize_rollout_path(
    path: str | Path, *, relative_to: Path | None = None
) -> Path:
    raw_path = Path(path)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        if relative_to is not None:
            candidates.append(relative_to / raw_path)
        candidates.append(PROJECT_ROOT / raw_path)

    unique_candidates = list(
        dict.fromkeys(candidate.resolve() for candidate in candidates)
    )
    for candidate in unique_candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve rollout asset path {str(path)!r}; checked "
        f"{[str(candidate) for candidate in unique_candidates]}."
    )


def _process_rollout_urdf_cached(urdf_path: Path, cache_root: Path) -> Path:
    source_path = _localize_rollout_path(urdf_path)
    tree = ET.parse(source_path)
    geometry_nodes = list(tree.getroot().iter("geometry"))
    if not geometry_nodes:
        raise ValueError(f"URDF has no visual/collision geometry nodes: {source_path}")
    supported_geometry_tags = {"box", "cylinder", "mesh", "sphere"}
    unsupported_geometry = [
        tuple(child.tag for child in geometry_node)
        for geometry_node in geometry_nodes
        if not any(child.tag in supported_geometry_tags for child in geometry_node)
    ]
    if unsupported_geometry:
        raise ValueError(
            f"URDF contains unsupported geometry nodes {unsupported_geometry}: "
            f"{source_path}"
        )

    # Make file-backed geometry absolute; primitive geometry remains unchanged.
    mesh_nodes = list(tree.getroot().iter("mesh"))
    for mesh_node in mesh_nodes:
        filename = mesh_node.get("filename")
        if not filename:
            raise ValueError(f"URDF mesh node has no filename in {source_path}.")
        mesh_path = _localize_rollout_path(filename, relative_to=source_path.parent)
        mesh_node.set("filename", str(mesh_path))

    digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:16]
    output_path = cache_root / f"{source_path.stem}_{digest}.urdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _load_scene_objects(
    *,
    reference_motion: dict[str, object],
    included_object_ids: tuple[str, ...],
    cache_dir: Path,
) -> tuple[SceneObject, ...]:
    all_object_ids = tuple(
        str(unwrap_scalar(value))
        for value in np.asarray(reference_motion["scene_object_ids"], dtype=object)
    )
    urdf_paths = np.asarray(reference_motion["scene_object_urdf_paths"], dtype=object)
    if urdf_paths.shape != (len(all_object_ids),):
        raise ValueError(
            f"Expected scene_object_urdf_paths shape {(len(all_object_ids),)}, got "
            f"{urdf_paths.shape}."
        )
    objects = []
    for object_id in included_object_ids:
        object_index = all_object_ids.index(object_id)
        urdf_path = Path(str(unwrap_scalar(urdf_paths[object_index])))
        processed_path = _process_rollout_urdf_cached(urdf_path, cache_dir / "urdf")
        objects.append(
            SceneObject(
                object_id=object_id,
                object_index=object_index,
                urdf=URDF.load(str(processed_path)),
                color_rgba=_color_for_object(object_id),
            )
        )
    return tuple(objects)


def _load_hand_models(
    *, motion: dict[str, object], cache_dir: Path
) -> tuple[tuple[HandModel, ...], np.ndarray]:
    active_hands = _resolve_active_hands(None, motion)
    robot_model = _resolve_robot_model(None, motion, "auto")
    if robot_model != "xhand":
        raise ValueError(
            f"Clean revision videos require robot_model='xhand', got {robot_model!r}."
        )
    hand_config_source = hand_config_source_by_side(motion)
    models = []
    for side in ("left", "right"):
        if not active_hands[side]:
            continue
        config_side = hand_config_source.get(side, side)
        relative_urdf_path = ROBOT_URDFS[robot_model][config_side]
        if relative_urdf_path is None:
            raise ValueError(
                f"Missing {robot_model!r} robot URDF for hand side {config_side!r}."
            )
        processed_path = _process_rollout_urdf_cached(
            Path(relative_urdf_path), cache_dir / "urdf"
        )
        joint_names_raw, joint_values_raw = resolve_visualized_joint_motion(
            motion, side=side
        )
        if joint_values_raw is None:
            raise ValueError(f"Motion is missing joint values for active {side} hand.")
        joint_names = tuple(str(value) for value in joint_names_raw)
        joint_values = np.asarray(joint_values_raw, dtype=np.float32)
        if joint_values.ndim != 2 or joint_values.shape[1] != len(joint_names):
            raise ValueError(
                f"Expected {side} joint values shape (T, {len(joint_names)}), got "
                f"{joint_values.shape}."
            )
        models.append(
            HandModel(
                side=side,
                urdf=URDF.load(str(processed_path)),
                joint_names=joint_names,
                joint_values=joint_values,
            )
        )
    if not models:
        raise ValueError("Motion contains no active XHand model to render.")
    return tuple(models), _resolve_robot_base_pose(None, motion)


def _frame_meshes(
    *,
    motion: dict[str, object],
    frame_index: int,
    hand_models: tuple[HandModel, ...],
    robot_base_pose: np.ndarray,
    scene_objects: tuple[SceneObject, ...],
) -> list[trimesh.Trimesh]:
    # Reconstruct robot visuals with the established cividis per-link palette.
    hand_instances = []
    for hand_model in hand_models:
        if frame_index >= hand_model.joint_values.shape[0]:
            raise ValueError(
                f"{hand_model.side} hand has {hand_model.joint_values.shape[0]} "
                f"frames, cannot render frame {frame_index}."
            )
        hand_instances.extend(
            fk_robot_link_instances(
                urdf_inst=hand_model.urdf,
                joint_dict=_joint_dict_for_frame(
                    hand_model.joint_names,
                    hand_model.joint_values[frame_index],
                ),
                base_pose=robot_base_pose,
                mesh_type="visual",
            )
        )
    meshes = transformed_meshes(hand_instances)

    # Render only task-semantic objects, using consistent paper colors and lights.
    for scene_object in scene_objects:
        instances = fk_mesh_instances_with_color(
            urdf_inst=scene_object.urdf,
            joint_dict=scene_object_joint_dict_for_frame(
                motion, frame_index, scene_object.object_index
            ),
            base_pose=scene_object_pose_for_frame(
                motion, frame_index, scene_object.object_index
            ),
            mesh_type="visual",
            color_rgba=scene_object.color_rgba,
        )
        if not instances:
            raise ValueError(
                f"No visual meshes found for scene object {scene_object.object_id!r}."
            )
        meshes.extend(transformed_meshes(instances))
    return meshes


def _frame_indices(episode: dict[str, np.ndarray], stride: int) -> np.ndarray:
    done = np.asarray(episode["done"], dtype=bool)
    if done.ndim != 1:
        raise ValueError(f"Expected episode done shape (T,), got {done.shape}.")
    visible = np.flatnonzero(~done)
    if visible.size == 0:
        raise ValueError("Recorded episode has no visible pre-terminal frames.")
    indices = visible[::stride]
    if indices[-1] != visible[-1]:
        indices = np.append(indices, visible[-1])
    return indices.astype(np.int64, copy=False)


def _mesh_bounds(meshes: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.stack([mesh.bounds for mesh in meshes], axis=0).astype(np.float32)
    if bounds.ndim != 3 or bounds.shape[1:] != (2, 3):
        raise ValueError(
            f"Expected stacked mesh bounds shape (M, 2, 3), got {bounds.shape}."
        )
    bounds_min = np.min(bounds[:, 0], axis=0)
    bounds_max = np.max(bounds[:, 1], axis=0)
    if not np.all(np.isfinite(bounds_min)) or not np.all(np.isfinite(bounds_max)):
        raise ValueError("Failed to compute finite rollout render bounds.")
    return bounds_min, bounds_max


def _smooth_rows(values: np.ndarray, window: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected smoothing input shape (T, D), got {values.shape}.")
    if window <= 0 or window % 2 == 0:
        raise ValueError(f"Expected a positive odd smoothing window, got {window}.")
    if len(values) == 1 or window == 1:
        return values.copy()
    effective_window = min(window, 2 * ((len(values) - 1) // 2) + 1)
    if effective_window == 1:
        return values.copy()
    radius = effective_window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.full(effective_window, 1.0 / effective_window, dtype=np.float32)
    return np.stack(
        [
            np.convolve(padded[:, dimension], kernel, mode="valid")
            for dimension in range(values.shape[1])
        ],
        axis=1,
    ).astype(np.float32)


def _tracked_camera_sequence(
    *,
    mesh_bounds: list[tuple[np.ndarray, np.ndarray]],
    camera: CameraSpec,
    width: int,
    height: int,
    smoothing_window: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    if not mesh_bounds:
        raise ValueError(
            "Expected at least one rollout frame bound for camera tracking."
        )
    direction = np.asarray(camera.direction, dtype=np.float32)
    if direction.shape != (3,) or np.linalg.norm(direction) <= 1.0e-8:
        raise ValueError(f"Expected a nonzero camera direction, got {direction}.")
    direction /= np.linalg.norm(direction)
    raw_centers = np.stack(
        [0.5 * (bounds_min + bounds_max) for bounds_min, bounds_max in mesh_bounds],
        axis=0,
    ).astype(np.float32)
    centers = _smooth_rows(raw_centers, smoothing_window)
    aspect = width / height
    camera_poses: list[np.ndarray] = []
    required_ymag = []
    for center, (bounds_min, bounds_max) in zip(centers, mesh_bounds, strict=True):
        extent = np.maximum(bounds_max - bounds_min, 1.0e-4)
        radius = 0.5 * float(np.linalg.norm(extent))
        camera_pose = look_at(
            center + max(2.8 * radius, 0.35) * direction,
            center,
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        corners = np.array(
            [
                [x, y, z]
                for x in (bounds_min[0], bounds_max[0])
                for y in (bounds_min[1], bounds_max[1])
                for z in (bounds_min[2], bounds_max[2])
            ],
            dtype=np.float32,
        )
        projected = trimesh.transform_points(corners, np.linalg.inv(camera_pose))
        half_width = float(np.max(np.abs(projected[:, 0])))
        half_height = float(np.max(np.abs(projected[:, 1])))
        required_ymag.append(
            max(
                camera.margin * half_height,
                camera.margin * half_width / aspect,
                0.025,
            )
        )
        camera_poses.append(camera_pose)
    required = np.asarray(required_ymag, dtype=np.float32)
    smoothed = _smooth_rows(required[:, None], smoothing_window)[:, 0]
    ymags = np.maximum(required, smoothed)
    return camera_poses, ymags * aspect, ymags, centers


def _fixed_camera_sequence(
    *,
    mesh_bounds: list[tuple[np.ndarray, np.ndarray]],
    camera: CameraSpec,
    width: int,
    height: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    if not mesh_bounds:
        raise ValueError("Expected at least one rollout frame bound for fixed camera.")
    bounds_min = np.min(
        np.stack([frame_min for frame_min, _ in mesh_bounds], axis=0), axis=0
    )
    bounds_max = np.max(
        np.stack([frame_max for _, frame_max in mesh_bounds], axis=0), axis=0
    )
    poses, xmags, ymags, centers = _tracked_camera_sequence(
        mesh_bounds=[(bounds_min, bounds_max)],
        camera=camera,
        width=width,
        height=height,
        smoothing_window=1,
    )
    frame_count = len(mesh_bounds)
    return (
        [poses[0].copy() for _ in range(frame_count)],
        np.repeat(xmags, frame_count),
        np.repeat(ymags, frame_count),
        np.repeat(centers, frame_count, axis=0),
    )


def _camera_scene_objects(
    *, scene_objects: tuple[SceneObject, ...], task_id: str, camera_mode: CameraMode
) -> tuple[SceneObject, ...]:
    if camera_mode == "fixed":
        return scene_objects
    if camera_mode != "tracked":
        raise ValueError(
            f"Expected camera mode 'tracked' or 'fixed', got {camera_mode!r}."
        )
    if "cube_strategy" not in task_id:
        return scene_objects
    return tuple(
        scene_object
        for scene_object in scene_objects
        if scene_object.object_id != "table"
    )


def render_video(args: RenderVideoArgs) -> dict[str, Any]:
    evaluation_metadata = _load_json_object(args.eval_dir / "metadata.json")
    reference_path = _reference_motion_path(args.eval_dir)
    metric_config = _metric_config(args.eval_dir)
    episode = load_episode_from_eval(args.eval_dir, args.episode_index)
    reference_motion = load_exported_npy_payload(reference_path)
    rollout_motion = rollout_motion_from_reference(reference_motion, episode)
    included_object_ids, excluded_object_ids = _semantic_object_ids(
        reference_motion=reference_motion,
        metric_config=metric_config,
        task_id=args.task_id,
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    hand_models, robot_base_pose = _load_hand_models(
        motion=rollout_motion, cache_dir=args.cache_dir
    )
    scene_objects = _load_scene_objects(
        reference_motion=reference_motion,
        included_object_ids=included_object_ids,
        cache_dir=args.cache_dir,
    )
    frame_indices = _frame_indices(episode, args.frame_stride)
    camera_scene_objects = _camera_scene_objects(
        scene_objects=scene_objects,
        task_id=args.task_id,
        camera_mode=args.camera_mode,
    )
    frame_bounds = []
    for frame_index in frame_indices.tolist():
        meshes = _frame_meshes(
            motion=rollout_motion,
            frame_index=frame_index,
            hand_models=hand_models,
            robot_base_pose=robot_base_pose,
            scene_objects=camera_scene_objects,
        )
        frame_bounds.append(_mesh_bounds(meshes))
    camera = _camera_for_task(args.task_id)
    if args.camera_mode == "tracked":
        camera_poses, xmags, ymags, camera_centers = _tracked_camera_sequence(
            mesh_bounds=frame_bounds,
            camera=camera,
            width=args.width,
            height=args.height,
            smoothing_window=args.camera_smoothing_window,
        )
    elif args.camera_mode == "fixed":
        camera_poses, xmags, ymags, camera_centers = _fixed_camera_sequence(
            mesh_bounds=frame_bounds,
            camera=camera,
            width=args.width,
            height=args.height,
        )
    else:
        raise ValueError(
            f"Expected camera mode 'tracked' or 'fixed', got {args.camera_mode!r}."
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        args.output_path,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    ) as writer:
        for output_index, frame_index in enumerate(frame_indices.tolist()):
            meshes = _frame_meshes(
                motion=rollout_motion,
                frame_index=frame_index,
                hand_models=hand_models,
                robot_base_pose=robot_base_pose,
                scene_objects=scene_objects,
            )
            image = render_panel(
                meshes,
                camera_pose=camera_poses[output_index],
                xmag=float(xmags[output_index]),
                ymag=float(ymags[output_index]),
                width=args.width,
                height=args.height,
            )
            writer.append_data(np.asarray(image, dtype=np.uint8))
            rendered_count = output_index + 1
            if rendered_count % 50 == 0 or rendered_count == len(frame_indices):
                print(
                    f"[clean-video] rendered {rendered_count}/{len(frame_indices)} frames",
                    flush=True,
                )
    if not args.output_path.is_file() or args.output_path.stat().st_size == 0:
        raise FileNotFoundError(f"Video encoder did not write {args.output_path}.")

    metadata = {
        "renderer": "pyrender visual-URDF reconstruction from recorded policy state",
        "eval_dir": str(args.eval_dir),
        "reference_motion": str(reference_path),
        "task_id": args.task_id,
        "episode_index": args.episode_index,
        "episode_selection": "fixed episode index 0; no success cherry-picking",
        "run_dir": evaluation_metadata.get("run_dir"),
        "checkpoint_label": evaluation_metadata.get("checkpoint_label"),
        "policy_checkpoint": evaluation_metadata.get("policy_checkpoint"),
        "included_scene_objects": list(included_object_ids),
        "excluded_scene_objects": list(excluded_object_ids),
        "robot_coloring": "cividis per-link XHand colors",
        "object_coloring": {
            scene_object.object_id: scene_object.color_rgba.tolist()
            for scene_object in scene_objects
        },
        "background": "neutral white; no simulator scene, viewer, or table background",
        "camera": asdict(camera),
        "camera_mode": args.camera_mode,
        "camera_tracking": (
            "fixed orientation with smoothed action-centered translation and zoom"
            if args.camera_mode == "tracked"
            else "fixed third-person pose and zoom over the complete rollout"
        ),
        "camera_framing_scene_objects": [
            scene_object.object_id for scene_object in camera_scene_objects
        ],
        "camera_smoothing_window": args.camera_smoothing_window,
        "camera_centers": camera_centers.tolist(),
        "camera_ymag": ymags.tolist(),
        "rollout_frame_indices": frame_indices.tolist(),
        "frame_stride": args.frame_stride,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "output_path": str(args.output_path),
    }
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    render_video(args)
    print(f"Wrote clean rollout video to {args.output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
