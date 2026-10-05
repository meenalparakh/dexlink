#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from dexlink.scripts.retargeting.export_oakink_lift_data import (
    MANO_KEYPOINT_INDICES,
    REFERENCE_KEYPOINT_NAMES,
    _load_motion_npz,
    _optional_motion_array,
    _required_motion_field,
    _resample_to_length,
    _resolve_robot_model,
    _robot_reference_keypoints,
    _unwrap_scalar,
)
from dexlink.src.retargeting.optimization.oakink_processor import OakInkProcessor
from dexlink.utils.retargeting.recompute_inertia_from_density import (
    recompute_urdf_inertia_from_density,
)
from dexlink.utils.retargeting.urdf_utils import create_urdf_from_obj, load_urdf_combined_mesh

DEXLINK_ROOT = Path(__file__).resolve().parents[3]


def _default_output_path(motion_npy_path: Path) -> Path:
    motion_npy_path = motion_npy_path.resolve()
    stem = motion_npy_path.stem
    if stem.endswith(".npz"):
        stem = Path(stem).stem
    return motion_npy_path.parent / f"{stem}_oakink_sequence_exported.npy"


def _hand_config_source_by_side(motion: dict[str, np.ndarray]) -> dict[str, str]:
    raw = motion.get("hand_config_source_by_side", {})
    if isinstance(raw, np.ndarray):
        if raw.shape != ():
            raise ValueError(
                "Expected hand_config_source_by_side to be a scalar object array, "
                f"got shape {raw.shape}."
            )
        raw = raw.item()
    if raw in (None, "", "None"):
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            "Expected hand_config_source_by_side to be a dict, "
            f"got {type(raw).__name__}."
        )
    source_by_side: dict[str, str] = {}
    for role_side, source_side in raw.items():
        role_side = str(role_side)
        source_side = str(source_side)
        if role_side not in ("left", "right") or source_side not in ("left", "right"):
            raise ValueError(
                "Expected hand_config_source_by_side to map left/right role sides "
                f"to left/right config sides, got {raw!r}."
            )
        source_by_side[role_side] = source_side
    return source_by_side


def _resolve_urdf_mesh_path(urdf_dir: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        filename = filename[len("package://") :]
    mesh_path = Path(filename)
    if not mesh_path.is_absolute():
        mesh_path = (urdf_dir / mesh_path).resolve()
    return mesh_path


def _mesh_tag_scale(mesh: ET.Element) -> np.ndarray:
    scale_text = mesh.attrib.get("scale", "1 1 1")
    scale = np.asarray([float(value) for value in scale_text.split()], dtype=np.float64)
    if scale.shape != (3,):
        raise ValueError(f"Expected mesh scale to have 3 values, got {scale_text!r}.")
    if np.any(scale <= 0.0):
        raise ValueError(f"Expected positive mesh scale values, got {scale_text!r}.")
    return scale


def _load_mesh_volume(
    mesh_path: Path,
    scale: np.ndarray,
    *,
    allow_convex_hull: bool = False,
) -> float:
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a Trimesh from {mesh_path}, got {type(mesh)}.")
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    volume = float(abs(mesh.volume))
    if volume <= 0.0 and allow_convex_hull:
        volume = float(abs(mesh.convex_hull.volume))
    if volume <= 0.0:
        raise ValueError(f"Mesh {mesh_path} produced non-positive volume {volume}.")
    return volume


def _urdf_collision_volume(
    urdf_path: str | Path, *, allow_convex_hull: bool = False
) -> float:
    urdf_path = Path(urdf_path).resolve()
    urdf_dir = urdf_path.parent
    root = ET.parse(urdf_path).getroot()
    volumes: list[float] = []
    for mesh in root.findall(".//collision/geometry/mesh"):
        filename = mesh.attrib.get("filename")
        if filename is None:
            raise ValueError(f"Collision mesh in {urdf_path} is missing filename.")
        volumes.append(
            _load_mesh_volume(
                _resolve_urdf_mesh_path(urdf_dir, filename),
                scale=_mesh_tag_scale(mesh),
                allow_convex_hull=allow_convex_hull,
            )
        )
    if not volumes:
        raise ValueError(f"Expected at least one collision mesh in {urdf_path}.")
    return float(sum(volumes))


def _export_simple_urdf_from_mesh(
    *,
    mesh,
    export_dir: Path,
    obj_name: str,
    density: float,
    recompute_inertia: bool = True,
) -> tuple[str, str]:
    export_dir.mkdir(parents=True, exist_ok=True)
    obj_path = export_dir / f"{obj_name}.obj"
    mesh.export(obj_path)
    urdf_path = Path(create_urdf_from_obj(str(obj_path), str(export_dir), scale=1.0))
    if recompute_inertia:
        recompute_urdf_inertia_from_density(
            input_urdf=urdf_path,
            output_urdf=urdf_path,
            density=density,
        )
    return str(obj_path), str(urdf_path)


def _export_simple_urdf_from_urdf(
    *,
    source_urdf_path: str | Path,
    export_dir: Path,
    obj_name: str,
    density: float,
    recompute_inertia: bool = True,
) -> tuple[str, str]:
    mesh = load_urdf_combined_mesh(source_urdf_path, mesh_type="visual")
    return _export_simple_urdf_from_mesh(
        mesh=mesh,
        export_dir=export_dir,
        obj_name=obj_name,
        density=density,
        recompute_inertia=recompute_inertia,
    )


def _export_simple_urdf_from_processor(
    demo: OakInkProcessor,
    export_dir: Path,
    obj_type: str,
    density: float,
    recompute_inertia: bool = True,
) -> tuple[str, str]:
    obj_dir = export_dir / obj_type
    obj_dir.mkdir(parents=True, exist_ok=True)
    obj_path = Path(demo.export_object_obj(output_dir=obj_dir, obj_type=obj_type))
    urdf_path = Path(create_urdf_from_obj(str(obj_path), str(obj_dir), scale=1.0))
    if recompute_inertia:
        recompute_urdf_inertia_from_density(
            input_urdf=urdf_path,
            output_urdf=urdf_path,
            density=density,
        )
    return str(obj_path), str(urdf_path)


def _motion_density_range(motion: dict, default_density: float) -> np.ndarray:
    if "object_density_range" not in motion:
        if default_density <= 0.0:
            raise ValueError(f"Expected positive --density, got {default_density}.")
        density_range = np.asarray([default_density, default_density], dtype=np.float32)
    else:
        density_range = np.asarray(
            _unwrap_scalar(motion["object_density_range"]),
            dtype=np.float32,
        ).reshape(-1)
    if density_range.shape != (2,):
        raise ValueError(
            f"Expected object_density_range to have shape (2,), got {density_range.shape}."
        )
    if density_range[0] <= 0.0 or density_range[1] <= 0.0:
        raise ValueError(
            f"Expected positive object_density_range, got {density_range}."
        )
    if density_range[0] > density_range[1]:
        raise ValueError(f"Expected ordered object_density_range, got {density_range}.")
    return density_range


def _resolve_stored_path(path_value: object) -> str | None:
    if path_value in (None, "", "None"):
        return None
    path = Path(str(path_value))
    return str(path if path.is_absolute() else DEXLINK_ROOT / path)


def _articulated_urdf_from_retarget_manifest(motion: dict) -> str | None:
    run_dir_raw = _resolve_stored_path(_unwrap_scalar(motion.get("oakink_run_dir")))
    if run_dir_raw in (None, "", "None"):
        return None
    manifest_path = Path(str(run_dir_raw)) / "retarget_job_manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    requested_target_mesh = _resolve_stored_path(
        manifest.get("requested_target_mesh_path")
    )
    if requested_target_mesh in (None, "", "None"):
        return None
    requested_path = Path(str(requested_target_mesh))
    if requested_path.suffix.lower() != ".urdf":
        return None
    transform_dir = _resolve_stored_path(manifest.get("transform_dir"))
    target_id = str(manifest.get("target_id", "")).strip()
    if transform_dir not in (None, "", "None") and target_id:
        scaled_path = Path(str(transform_dir)) / f"{target_id}_scaled.urdf"
        if scaled_path.exists():
            return str(scaled_path.resolve())
        urdfpy_path = Path(str(transform_dir)) / f"{target_id}_urdfpy.urdf"
        if urdfpy_path.exists():
            return str(urdfpy_path.resolve())
    return str(requested_path.resolve())


def _build_reference_hand_sequence(
    *,
    side: str,
    is_active: bool,
    n_frames: int,
    mano_joints: np.ndarray,
    mano_verts: np.ndarray,
    robot_reference_keypoints: np.ndarray | None,
    human_reference_keypoints: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_joints = _resample_to_length(
        np.asarray(mano_joints, dtype=np.float32), n_frames
    ).copy()
    ref_verts = _resample_to_length(
        np.asarray(mano_verts, dtype=np.float32), n_frames
    ).copy()
    ref_keypoints = ref_joints[
        :, [MANO_KEYPOINT_INDICES[name] for name in REFERENCE_KEYPOINT_NAMES], :
    ].copy()
    if human_reference_keypoints is None:
        human_ref_keypoints = ref_keypoints.copy()
    else:
        human_ref_keypoints = np.asarray(human_reference_keypoints, dtype=np.float32)
        if human_ref_keypoints.shape != (n_frames, len(REFERENCE_KEYPOINT_NAMES), 3):
            raise ValueError(
                "Expected explicit human reference keypoints to have shape "
                f"{(n_frames, len(REFERENCE_KEYPOINT_NAMES), 3)}, got "
                f"{human_ref_keypoints.shape}."
            )

    if is_active:
        if robot_reference_keypoints is None:
            raise ValueError(
                f"Missing robot reference keypoints for active side {side!r}."
            )
        ref_keypoints[:] = robot_reference_keypoints
        for keypoint_idx, name in enumerate(REFERENCE_KEYPOINT_NAMES):
            mano_idx = MANO_KEYPOINT_INDICES[name]
            ref_joints[:, mano_idx, :] = robot_reference_keypoints[:, keypoint_idx, :]

    return (
        ref_joints.astype(np.float32),
        ref_verts.astype(np.float32),
        ref_keypoints.astype(np.float32),
        human_ref_keypoints.astype(np.float32),
    )


def _motion_keypoints_for_reference_names(
    motion: dict,
    *,
    side: str,
    owner: str,
    n_frames: int,
) -> np.ndarray | None:
    names_key = "human_skeleton_keypoint_names"
    pos_key = f"{side}_{owner}_skeleton_keypoint_pos"
    valid_key = f"{side}_{owner}_skeleton_keypoint_valid"
    if pos_key not in motion:
        return None
    if names_key not in motion:
        raise ValueError(f"{pos_key} exists, but {names_key} is missing.")
    if valid_key in motion:
        valid = np.asarray(motion[valid_key], dtype=bool).reshape(-1)
        if valid.shape != (n_frames,):
            raise ValueError(
                f"Expected {valid_key} shape {(n_frames,)}, got {valid.shape}."
            )
    names = [str(name) for name in np.asarray(motion[names_key]).reshape(-1).tolist()]
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in REFERENCE_KEYPOINT_NAMES if name not in name_to_idx]
    if missing:
        raise ValueError(f"{names_key} is missing reference names: {missing}.")
    values = np.asarray(motion[pos_key], dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != n_frames or values.shape[2] != 3:
        raise ValueError(f"Expected {pos_key} shape (T, K, 3), got {values.shape}.")
    return values[:, [name_to_idx[name] for name in REFERENCE_KEYPOINT_NAMES], :]


def _copy_multi_object_scene_payload(motion: dict, n_frames: int) -> dict:
    if "scene_object_pos" not in motion:
        return {}

    required_keys = (
        "scene_object_ids",
        "scene_object_names",
        "scene_object_urdf_paths",
        "scene_object_pos",
        "scene_object_quat",
        "scene_object_arti_vals",
        "scene_object_arti_valid_mask",
        "scene_object_arti_names",
        "scene_object_active_mask",
        "scene_object_static",
    )
    missing = [key for key in required_keys if key not in motion]
    if missing:
        raise ValueError(
            "Motion npz contains scene_object_pos, so expected full multi-object "
            f"scene keys {required_keys}; missing {missing}."
        )

    scene_object_ids = np.asarray(motion["scene_object_ids"], dtype=object)
    scene_object_names = np.asarray(motion["scene_object_names"], dtype=object)
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
    scene_object_active_mask = np.asarray(
        motion["scene_object_active_mask"], dtype=bool
    )
    scene_object_static = np.asarray(motion["scene_object_static"], dtype=bool)
    scene_object_friction = np.asarray(
        motion.get(
            "scene_object_friction", np.ones_like(scene_object_static, dtype=np.float32)
        ),
        dtype=np.float32,
    )

    if scene_object_pos.ndim != 3 or scene_object_pos.shape[2] != 3:
        raise ValueError(
            "Expected scene_object_pos to have shape (T, N, 3), got "
            f"{scene_object_pos.shape}."
        )
    if scene_object_pos.shape[0] != n_frames:
        raise ValueError(
            "Expected scene_object_pos frame count to match asset_pos frame count "
            f"{n_frames}, got {scene_object_pos.shape[0]}."
        )
    if scene_object_quat.shape != (
        scene_object_pos.shape[0],
        scene_object_pos.shape[1],
        4,
    ):
        raise ValueError(
            "Expected scene_object_quat to have shape (T, N, 4) matching "
            f"scene_object_pos, got {scene_object_quat.shape}."
        )
    if scene_object_arti_vals.ndim != 3:
        raise ValueError(
            "Expected scene_object_arti_vals to have shape (T, N, Amax), got "
            f"{scene_object_arti_vals.shape}."
        )
    if scene_object_arti_vals.shape[:2] != scene_object_pos.shape[:2]:
        raise ValueError(
            "Expected scene_object_arti_vals first two dimensions to match "
            f"scene_object_pos, got {scene_object_arti_vals.shape} and "
            f"{scene_object_pos.shape}."
        )
    if scene_object_arti_valid_mask.shape != scene_object_arti_vals.shape[1:]:
        raise ValueError(
            "Expected scene_object_arti_valid_mask shape (N, Amax), got "
            f"{scene_object_arti_valid_mask.shape} for values "
            f"{scene_object_arti_vals.shape}."
        )
    if scene_object_active_mask.shape != scene_object_pos.shape[:2]:
        raise ValueError(
            "Expected scene_object_active_mask shape (T, N), got "
            f"{scene_object_active_mask.shape} for scene_object_pos "
            f"{scene_object_pos.shape}."
        )
    n_objects = int(scene_object_pos.shape[1])
    if scene_object_static.shape != (n_objects,):
        raise ValueError(
            f"Expected scene_object_static shape {(n_objects,)}, got "
            f"{scene_object_static.shape}."
        )
    if scene_object_friction.shape != (n_objects,):
        raise ValueError(
            f"Expected scene_object_friction shape {(n_objects,)}, got "
            f"{scene_object_friction.shape}."
        )
    if np.any(scene_object_friction <= 0.0):
        raise ValueError(
            f"Expected positive scene_object_friction, got {scene_object_friction}."
        )
    for key, value in (
        ("scene_object_ids", scene_object_ids),
        ("scene_object_names", scene_object_names),
        ("scene_object_urdf_paths", scene_object_urdf_paths),
        ("scene_object_arti_names", scene_object_arti_names),
    ):
        if len(value) != n_objects:
            raise ValueError(
                f"Expected {key} length to match scene object count {n_objects}, "
                f"got {len(value)}."
            )

    scene_object_rot = (
        R.from_quat(np.roll(scene_object_quat.reshape(-1, 4), -1, axis=-1))
        .as_rotvec()
        .reshape(scene_object_quat.shape[0], scene_object_quat.shape[1], 3)
        .astype(np.float32)
    )
    scene_payload = {
        "scene_object_ids": scene_object_ids.copy(),
        "scene_object_names": scene_object_names.copy(),
        "scene_object_urdf_paths": scene_object_urdf_paths.copy(),
        "scene_object_pos": scene_object_pos.copy(),
        "scene_object_quat": scene_object_quat.copy(),
        "scene_object_rot": scene_object_rot,
        "scene_object_arti_vals": scene_object_arti_vals.copy(),
        "scene_object_arti_valid_mask": scene_object_arti_valid_mask.copy(),
        "scene_object_arti_names": scene_object_arti_names.copy(),
        "scene_object_active_mask": scene_object_active_mask.copy(),
        "scene_object_static": scene_object_static.copy(),
        "scene_object_friction": scene_object_friction.copy(),
    }
    for key in (
        "scene_table_height",
        "scene_table_margin",
        "scene_table_size",
        "scene_uses_block",
        "scene_object_support_z_offsets",
        "scene_support_block_ids",
        "scene_support_block_pos",
        "scene_support_block_size",
    ):
        if key in motion:
            scene_payload[key] = np.asarray(motion[key]).copy()
    return scene_payload


def build_payload(args) -> dict:
    motion = _load_motion_npz(args.motion_npy_path)
    n_frames = int(np.asarray(motion["asset_pos"]).shape[0])
    processor_run_dir = getattr(args, "processor_run_dir", None)
    robot_model = _resolve_robot_model(motion, args.robot_model)
    geometry_only = bool(getattr(args, "geometry_only", False))
    export_dir = (
        Path(args.output_path).resolve().parent
        / f"{Path(args.output_path).stem}_assets"
    )
    traj_object_urdf_path = _resolve_stored_path(
        _unwrap_scalar(motion.get("traj_object_urdf_path"))
    )
    if traj_object_urdf_path is None:
        raise ValueError(
            "Motion npz is missing required field 'traj_object_urdf_path'."
        )

    if "oakink_demo_path" in motion:
        demo_path = _resolve_stored_path(
            _required_motion_field(motion, "oakink_demo_path")
        )
        if demo_path is None:
            raise ValueError("Motion metadata field 'oakink_demo_path' is empty.")
        run_dir = (
            Path(processor_run_dir).resolve()
            if processor_run_dir is not None
            else Path(
                str(
                    _resolve_stored_path(
                        _required_motion_field(motion, "oakink_run_dir")
                    )
                )
            )
        )
        target_demo_path = _resolve_stored_path(
            _unwrap_scalar(motion.get("oakink_target_demo_path"))
        )
        materialized_target_obj_path = _resolve_stored_path(
            _unwrap_scalar(motion.get("oakink_target_obj_path"))
        )
        requested_target_obj_path = _resolve_stored_path(
            _unwrap_scalar(motion.get("oakink_requested_target_obj_path"))
        )
        target_transform_info = _resolve_stored_path(
            _unwrap_scalar(motion.get("oakink_target_transform_info"))
        )
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
                None
                if target_demo_path in (None, "", "None")
                else str(target_demo_path)
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
        original_obj_path, original_urdf_path = _export_simple_urdf_from_processor(
            demo,
            export_dir,
            obj_type="original",
            density=args.density,
            recompute_inertia=not geometry_only,
        )
        target_obj_path, target_urdf_path = _export_simple_urdf_from_processor(
            demo,
            export_dir,
            obj_type="target",
            density=args.density,
            recompute_inertia=not geometry_only,
        )
        original_obj_name = demo.obj_name
        target_obj_name = demo.target_obj_name
        mano_data = demo.data
        left_mano_verts_raw = demo.left_hand_verts
        right_mano_verts_raw = demo.right_hand_verts
    else:
        original_obj_name = str(
            _unwrap_scalar(motion.get("traj_object_name", "sequence_object"))
        )
        target_obj_name = original_obj_name
        original_obj_path, original_urdf_path = _export_simple_urdf_from_urdf(
            source_urdf_path=traj_object_urdf_path,
            export_dir=export_dir / "original",
            obj_name=f"{original_obj_name}_original",
            density=args.density,
            recompute_inertia=not geometry_only,
        )
        target_obj_path, target_urdf_path = _export_simple_urdf_from_urdf(
            source_urdf_path=traj_object_urdf_path,
            export_dir=export_dir / "target",
            obj_name=f"{target_obj_name}_target",
            density=args.density,
            recompute_inertia=not geometry_only,
        )
        mano_data = {
            "left_joints": np.zeros((n_frames, 21, 3), dtype=np.float32),
            "right_joints": np.zeros((n_frames, 21, 3), dtype=np.float32),
        }
        left_mano_verts_raw = np.zeros((n_frames, 21, 3), dtype=np.float32)
        right_mano_verts_raw = np.zeros((n_frames, 21, 3), dtype=np.float32)

    target_volume = _urdf_collision_volume(
        target_urdf_path, allow_convex_hull=geometry_only
    )
    density_range = _motion_density_range(motion, default_density=float(args.density))
    mass_range = density_range * np.float32(target_volume)

    active_side = str(motion["traj_hand_side"].item())
    active_sides = tuple(
        side
        for side in ("left", "right")
        if bool(
            np.asarray(
                motion.get(
                    f"is_active_{side}", np.array([side == active_side], dtype=bool)
                ),
                dtype=bool,
            ).reshape(-1)[0]
        )
    )
    if len(active_sides) == 0:
        raise ValueError("Motion payload has no active hand side.")
    traj_robot_base_pose = _optional_motion_array(
        motion,
        "traj_robot_base_pose",
        dtype=np.float32,
    )
    hand_config_source_by_side = _hand_config_source_by_side(motion)
    robot_reference_keypoints = {
        side: (
            explicit_keypoints
            if (
                explicit_keypoints := _motion_keypoints_for_reference_names(
                    motion,
                    side=side,
                    owner="robot",
                    n_frames=n_frames,
                )
            )
            is not None
            else _robot_reference_keypoints(
                robot_model=robot_model,
                side=side,
                config_side=hand_config_source_by_side.get(side, side),
                joint_names=motion[f"{side}_joint_names"],
                joint_vals=motion[f"{side}_joint_vals"],
                robot_base_pose=traj_robot_base_pose,
            )
        )
        for side in active_sides
    }
    human_reference_keypoints = {
        side: _motion_keypoints_for_reference_names(
            motion,
            side=side,
            owner="human",
            n_frames=n_frames,
        )
        for side in ("left", "right")
    }
    human_reference_valid = {
        side: (
            np.asarray(motion[f"{side}_human_skeleton_keypoint_valid"], dtype=bool)
            .reshape(-1)
            .copy()
            if f"{side}_human_skeleton_keypoint_valid" in motion
            else np.ones((n_frames,), dtype=bool)
        )
        for side in ("left", "right")
    }
    (
        left_mano_joints,
        left_mano_verts,
        left_reference_keypoints,
        left_human_reference_keypoints,
    ) = _build_reference_hand_sequence(
        side="left",
        is_active="left" in active_sides,
        n_frames=n_frames,
        mano_joints=mano_data["left_joints"],
        mano_verts=left_mano_verts_raw,
        robot_reference_keypoints=robot_reference_keypoints.get("left"),
        human_reference_keypoints=human_reference_keypoints["left"],
    )
    (
        right_mano_joints,
        right_mano_verts,
        right_reference_keypoints,
        right_human_reference_keypoints,
    ) = _build_reference_hand_sequence(
        side="right",
        is_active="right" in active_sides,
        n_frames=n_frames,
        mano_joints=mano_data["right_joints"],
        mano_verts=right_mano_verts_raw,
        robot_reference_keypoints=robot_reference_keypoints.get("right"),
        human_reference_keypoints=human_reference_keypoints["right"],
    )

    asset_pos = motion["asset_pos"].astype(np.float32)
    asset_quat = motion["asset_quat"].astype(np.float32)
    asset_rot = (
        R.from_quat(np.roll(asset_quat, -1, axis=-1)).as_rotvec().astype(np.float32)
    )
    asset_arti = motion["asset_arti_vals"].astype(np.float32)
    asset_joint_names = np.asarray(motion["asset_arti_names"], dtype=object)
    target_articulated_urdf_path = None
    if asset_joint_names.size > 0:
        metadata_articulated_urdf_path = _resolve_stored_path(
            _unwrap_scalar(motion.get("oakink_target_articulated_urdf_path"))
        )
        if metadata_articulated_urdf_path not in (None, "", "None"):
            target_articulated_urdf_path = str(metadata_articulated_urdf_path)
        else:
            manifest_articulated_urdf_path = _articulated_urdf_from_retarget_manifest(
                motion
            )
            if manifest_articulated_urdf_path not in (None, "", "None"):
                target_articulated_urdf_path = str(manifest_articulated_urdf_path)
        if target_articulated_urdf_path is None and traj_object_urdf_path not in (
            None,
            "",
            "None",
        ):
            target_articulated_urdf_path = str(traj_object_urdf_path)
    target_articulation_link_names = np.asarray(
        motion.get("target_bps_sampling_link_names", np.asarray([], dtype=object)),
        dtype=object,
    )

    payload = {
        "demo_kind": "sequence",
        "side": str(motion["traj_hand_side"].item()),
        "active_sides": np.asarray(active_sides, dtype=object),
        "robot_model": robot_model,
        "original_obj_name": original_obj_name,
        "target_obj_name": target_obj_name,
        "original_obj_path": original_obj_path,
        "target_obj_path": target_obj_path,
        "original_urdf_path": original_urdf_path,
        "target_urdf_path": target_urdf_path,
        "target_articulated_urdf_path": target_articulated_urdf_path,
        "target_articulation_link_names": target_articulation_link_names.copy(),
        "target_object_volume": np.array(target_volume, dtype=np.float32),
        "object_density_range": density_range.astype(np.float32),
        "object_mass_range": mass_range.astype(np.float32),
        "left_mano_verts": left_mano_verts,
        "right_mano_verts": right_mano_verts,
        "left_mano_joints": left_mano_joints,
        "right_mano_joints": right_mano_joints,
        "reference_keypoint_names": np.asarray(REFERENCE_KEYPOINT_NAMES, dtype=object),
        "left_reference_keypoint_pos": left_reference_keypoints,
        "right_reference_keypoint_pos": right_reference_keypoints,
        "left_human_reference_keypoint_pos": left_human_reference_keypoints,
        "right_human_reference_keypoint_pos": right_human_reference_keypoints,
        "left_human_reference_keypoint_valid": human_reference_valid["left"],
        "right_human_reference_keypoint_valid": human_reference_valid["right"],
        "reference_obj_pos": asset_pos.copy(),
        "reference_obj_quat": asset_quat.copy(),
        "reference_obj_rot": asset_rot.copy(),
        "reference_obj_arti": asset_arti.copy(),
        "reference_joint_names": asset_joint_names.copy(),
        "left_jnames": np.asarray(motion["left_joint_names"], dtype=object),
        "left_jvals": motion["left_joint_vals"].astype(np.float32),
        "right_jnames": np.asarray(motion["right_joint_names"], dtype=object),
        "right_jvals": motion["right_joint_vals"].astype(np.float32),
        "retargeted_obj_pos": asset_pos.copy(),
        "retargeted_obj_quat": asset_quat.copy(),
        "retargeted_obj_rot": asset_rot.copy(),
        "retargeted_obj_arti": asset_arti.copy(),
        "retargeted_joint_names": asset_joint_names.copy(),
        "phase_names": np.asarray(motion.get("phase_names", []), dtype=object),
        "phase_lengths": np.asarray(motion.get("phase_lengths", []), dtype=np.int64),
        "n_reach_frames": np.array([0], dtype=np.int64),
        "n_lift_frames": np.array([0], dtype=np.int64),
        "is_active_left": motion.get(
            "is_active_left", np.array([active_side == "left"], dtype=bool)
        ),
        "is_active_right": motion.get(
            "is_active_right", np.array([active_side == "right"], dtype=bool)
        ),
    }
    if "object_static" in motion:
        payload["object_static"] = np.asarray(motion["object_static"], dtype=bool)
    payload.update(_copy_multi_object_scene_payload(motion, n_frames))
    if traj_robot_base_pose is not None:
        payload["traj_robot_base_pose"] = traj_robot_base_pose
    for key in (
        "sequence_spec_path",
        "experiment_id",
        "sequence_id",
        "category",
        "hand_config_source_by_side",
    ):
        if key in motion:
            payload[key] = motion[key]
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a generic primitive-built sequence motion into an OakInk-style payload."
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
    parser.add_argument(
        "--processor-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional local run directory for OakInkProcessor scratch/export files. "
            "Use this when motion metadata points at a cluster-only run_dir."
        ),
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
    print(f"Saved OakInk sequence payload to {output_path}")


if __name__ == "__main__":
    main()
