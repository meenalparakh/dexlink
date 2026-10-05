#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from dexlink.src.reference.category_paths import (
    category_prepared_target_dir,
    category_retarget_run_dir,
    category_transform_dir,
    normalize_retarget_hand_type,
)
from dexlink.src.reference.task_io import load_task_spec
from dexlink.src.retargeting.optimization.sequence_primitives import (
    TemplateState,
    _load_object_urdf,
    articulate_object,
    build_motion_payload,
    combine_bimanual_template_states,
    derive_open_reach_template,
    derive_template_from_transforms,
    hand_transition_via_clearance,
    hold_template,
    interpolate_articulation_link_motion,
    interpolate_pair_templates,
    interpolate_templates,
    load_template_state_from_motion,
    move_pair_to_pose,
    object_highest_z,
    object_lowest_z,
    place_object,
    release_or_open,
    set_object_articulation_model,
)
from dexlink.src.retargeting.optimization.target_transform_utils import load_target_transform
from dexlink.src.retargeting.optimization.template_mirror import (
    mirror_template_state,
    set_template_active_side,
)
from dexlink.utils.retargeting.urdf_utils import create_urdf_from_obj
from dexlink.utils.retargeting.path_utils import (
    get_external_data_dir,
    get_logs_dir,
    get_src,
)
from dexlink.src.reference.common.exp_info import (
    OBJECT_CATEGORIES,
    get_target_object,
)


@dataclass(frozen=True)
class TemplateEntry:
    state: TemplateState
    object_id: str


TEMPLATE_MOTION_FILENAME = "sequence_motion_lineseq.npz"
DEFAULT_SCENE_TABLE_HEIGHT = 0.75
DEFAULT_SCENE_TABLE_MARGIN = 0.001
def _object_static_from_cfg(scene_object: dict) -> bool:
    return bool(scene_object.get("static", scene_object.get("fixed", False)))


def _object_friction_from_cfg(scene_object: dict) -> float:
    friction = float(scene_object.get("friction", 1.0))
    if friction <= 0.0:
        raise ValueError(
            f"Scene object {scene_object.get('id')!r} friction must be positive, "
            f"got {friction}."
        )
    return friction


def _spec_object_static(spec: dict) -> bool:
    return bool(spec.get("object_static", spec.get("static", spec.get("fixed", False))))


def _validate_hand_config_source_by_side(
    source_by_side: object,
    *,
    context: str,
) -> dict[str, str]:
    if source_by_side in (None, ""):
        return {}
    if not isinstance(source_by_side, dict):
        raise ValueError(
            f"Expected {context} to be a dict mapping role side to config side, "
            f"got {type(source_by_side).__name__}."
        )
    validated: dict[str, str] = {}
    for role_side, source_side in source_by_side.items():
        role_side = str(role_side)
        source_side = str(source_side)
        if role_side not in ("left", "right") or source_side not in ("left", "right"):
            raise ValueError(
                f"Expected {context} to map left/right role sides to left/right "
                f"config sides, got {source_by_side!r}."
            )
        validated[role_side] = source_side
    return validated


def _hand_config_source_by_side_for_payload(
    spec: dict,
    canonical_template: TemplateState,
) -> dict[str, str]:
    template_source_by_side = _validate_hand_config_source_by_side(
        canonical_template.metadata.get("hand_config_source_by_side", {}),
        context="canonical template hand_config_source_by_side metadata",
    )
    spec_source_by_side = _validate_hand_config_source_by_side(
        spec.get("hand_config_source_by_side", {}),
        context="spec hand_config_source_by_side",
    )
    conflicts = {
        side: (template_source_by_side[side], spec_source_by_side[side])
        for side in template_source_by_side.keys() & spec_source_by_side.keys()
        if template_source_by_side[side] != spec_source_by_side[side]
    }
    if conflicts:
        raise ValueError(
            "Spec hand_config_source_by_side conflicts with canonical template "
            f"metadata: {conflicts}."
        )
    return {**template_source_by_side, **spec_source_by_side}


def _load_spec(path: Path) -> dict:
    return load_task_spec(path)


def _default_retarget_dir(root: Path, experiment_id: str, job_id: str) -> Path:
    return get_logs_dir() / "sequence_retargetings" / experiment_id / job_id


def _rooted_path(root: Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "outputs":
        return get_logs_dir().joinpath(*path.parts[1:])
    if path.parts and path.parts[0] == "external_data":
        return get_external_data_dir().joinpath(*path.parts[1:])
    return root / path


def _project_local_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    resolved = _rooted_path(root, path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Expected a valid absolute or repository-relative path, got "
            f"{raw_path!r} (resolved to {resolved})."
        )
    return resolved


def _project_local_path_if_exists(root: Path, raw_path: str | Path) -> Path | None:
    try:
        return _project_local_path(root, raw_path)
    except FileNotFoundError:
        return None


def _category_retarget_motion_path(
    root: Path,
    source_target: dict,
    retarget_hand_type: str,
) -> Path:
    category = str(source_target["category"])
    source_index = int(source_target["source_index"])
    target_index = int(source_target["target_index"])
    articulation_sample_id = source_target.get("articulation_sample_id")
    if articulation_sample_id is not None:
        articulation_sample_id = str(articulation_sample_id)
    if "retarget_run_dir" in source_target:
        run_dir = _rooted_path(root, str(source_target["retarget_run_dir"]))
    else:
        run_dir = category_retarget_run_dir(
            root,
            normalize_retarget_hand_type(retarget_hand_type),
            category,
            source_index,
            target_index,
            articulation_sample_id,
        )
    motion_path = run_dir / TEMPLATE_MOTION_FILENAME
    if not motion_path.exists():
        raise FileNotFoundError(
            f"Expected category retargeted motion at {motion_path}."
        )
    return motion_path


def _source_target_transform_dir(root: Path, source_target: dict) -> Path:
    category = str(source_target["category"])
    source_index = int(source_target["source_index"])
    target_index = int(source_target["target_index"])
    return category_transform_dir(
        root,
        category,
        source_index,
        target_index,
    )


def _source_target_articulated_urdf_path(
    root: Path, source_target: dict
) -> Path | None:
    category = str(source_target["category"])
    target_index = int(source_target["target_index"])
    target = get_target_object(category, target_index)
    if target.mesh_path is None or Path(target.mesh_path).suffix.lower() != ".urdf":
        return None
    prepared_urdf = (
        category_prepared_target_dir(
            root,
            category,
            int(source_target["source_index"]),
            target_index,
        )
        / f"{target.target_id}_scaled.urdf"
    )
    if prepared_urdf.exists():
        return prepared_urdf.resolve()
    original_urdf = _rooted_path(root, target.mesh_path).resolve()
    if not original_urdf.exists():
        raise FileNotFoundError(f"Expected articulated target URDF at {original_urdf}.")
    scale = _source_target_scale(root, source_target)
    if not np.isclose(scale, 1.0):
        raise FileNotFoundError(
            f"Expected scaled articulated target URDF at {prepared_urdf}. Run "
            "the category retargeting stage before reference construction."
        )
    return original_urdf


def _source_target_transform_path(root: Path, source_target: dict) -> Path | None:
    transform_path = _source_target_transform_dir(root, source_target) / "rotation.json"
    if transform_path.exists():
        return transform_path.resolve()
    return None


def _source_target_scale(root: Path, source_target: dict) -> float:
    scale_path = _source_target_transform_dir(root, source_target) / "scale.txt"
    if scale_path.exists():
        scale = float(scale_path.read_text(encoding="utf-8").strip())
    else:
        target = get_target_object(
            str(source_target["category"]), int(source_target["target_index"])
        )
        scale = float(target.scale)
    if scale <= 0.0:
        raise ValueError(f"Expected positive source-target scale, got {scale}.")
    return scale


def _resolve_retarget_motion_path(root: Path, spec: dict, job_id: str) -> Path:
    experiment_id = str(spec["experiment_id"])
    for job in spec.get("retarget_jobs", []):
        if str(job["job_id"]) != job_id:
            continue
        run_dir = Path(
            job.get("run_dir", _default_retarget_dir(root, experiment_id, job_id))
        )
        motion_path = run_dir / TEMPLATE_MOTION_FILENAME
        if not motion_path.exists():
            raise FileNotFoundError(
                f"Expected retargeted motion for job {job_id!r} at {motion_path}."
            )
        return motion_path
    raise ValueError(f"Unknown retarget_job_id {job_id!r}.")


def _scene_object_source_target(spec: dict, object_id: str) -> dict:
    matches = [
        scene_object
        for scene_object in spec.get("scene_objects", [])
        if str(scene_object["id"]) == object_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one scene object with id {object_id!r}, found {len(matches)}."
        )
    if "source_target" not in matches[0]:
        raise ValueError(
            f"Scene object {object_id!r} must define `source_target` for motion_path='auto'."
        )
    return dict(matches[0]["source_target"])


def _resolve_keyframe_motion_path(
    root: Path,
    spec: dict,
    keyframe_cfg: dict,
    retarget_hand_type: str,
) -> Path:
    from_retarget = keyframe_cfg["from_retarget"]
    motion_path = from_retarget.get("motion_path", "auto")
    if str(motion_path) != "auto":
        return _project_local_path(root, str(motion_path))
    if "retarget_job_id" in from_retarget:
        return _resolve_retarget_motion_path(
            root,
            spec,
            str(from_retarget["retarget_job_id"]),
        )
    if "source_target" in keyframe_cfg:
        source_target = dict(keyframe_cfg["source_target"])
        if "articulation_sample_id" in from_retarget:
            source_target["articulation_sample_id"] = str(
                from_retarget["articulation_sample_id"]
            )
        return _category_retarget_motion_path(
            root,
            source_target,
            retarget_hand_type,
        )
    if "object_id" in keyframe_cfg and "scene_objects" in spec:
        source_target = _scene_object_source_target(
            spec, str(keyframe_cfg["object_id"])
        )
        if "articulation_sample_id" in from_retarget:
            source_target["articulation_sample_id"] = str(
                from_retarget["articulation_sample_id"]
            )
        return _category_retarget_motion_path(
            root,
            source_target,
            retarget_hand_type,
        )
    if "source_target" in spec:
        source_target = dict(spec["source_target"])
        if "articulation_sample_id" in from_retarget:
            source_target["articulation_sample_id"] = str(
                from_retarget["articulation_sample_id"]
            )
        return _category_retarget_motion_path(
            root,
            source_target,
            retarget_hand_type,
        )
    raise ValueError(
        f"Keyframe {keyframe_cfg.get('id')} uses motion_path='auto' but no source_target or retarget_job_id was provided."
    )


def _template_id(template_cfg: dict) -> str:
    if "template_id" in template_cfg:
        return str(template_cfg["template_id"])
    if "id" in template_cfg:
        return str(template_cfg["id"])
    raise ValueError(
        f"Template config must define `template_id` or `id`: {template_cfg}"
    )


def _pose_rotation_from_cfg(pose_cfg: dict) -> R:
    if "quat_wxyz" in pose_cfg:
        quat_wxyz = np.asarray(pose_cfg["quat_wxyz"], dtype=np.float64).reshape(4)
        return R.from_quat(np.roll(quat_wxyz, -1))
    if "object_quat_wxyz" in pose_cfg:
        quat_wxyz = np.asarray(pose_cfg["object_quat_wxyz"], dtype=np.float64).reshape(
            4
        )
        return R.from_quat(np.roll(quat_wxyz, -1))
    if "euler_xyz_degrees" in pose_cfg:
        return R.from_euler(
            "xyz",
            np.asarray(pose_cfg["euler_xyz_degrees"], dtype=np.float64).reshape(3),
            degrees=True,
        )
    if "euler_xyz_radians" in pose_cfg:
        return R.from_euler(
            "xyz",
            np.asarray(pose_cfg["euler_xyz_radians"], dtype=np.float64).reshape(3),
            degrees=False,
        )
    if "euler_degrees" in pose_cfg:
        return R.from_euler(
            str(pose_cfg.get("euler_order", "xyz")),
            np.asarray(pose_cfg["euler_degrees"], dtype=np.float64).reshape(3),
            degrees=True,
        )
    if "euler_radians" in pose_cfg:
        return R.from_euler(
            str(pose_cfg.get("euler_order", "xyz")),
            np.asarray(pose_cfg["euler_radians"], dtype=np.float64).reshape(3),
            degrees=False,
        )
    return R.identity()


def _pose_translation_from_cfg(pose_cfg: dict) -> np.ndarray:
    if "translation" in pose_cfg:
        return np.asarray(pose_cfg["translation"], dtype=np.float32).reshape(3)
    if "object_pos" in pose_cfg:
        return np.asarray(pose_cfg["object_pos"], dtype=np.float32).reshape(3)
    return np.zeros(3, dtype=np.float32)


def _apply_pose_cfg(
    state: TemplateState,
    *,
    template_id: str,
    pose_cfg: dict,
) -> TemplateState:
    from dexlink.src.retargeting.optimization.sequence_primitives import apply_pair_rigid_transform

    rotation = _pose_rotation_from_cfg(pose_cfg)
    translation = _pose_translation_from_cfg(pose_cfg)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = translation
    return apply_pair_rigid_transform(
        state,
        template_id=template_id,
        transform=transform,
    )


TEMPLATE_POSE_KEYS = (
    "template_pose",
    "template_usage_pose",
    "use_template_pose",
    "motion_pose",
    "lie_down_pose",
    "object_lie_down_pose",
)


def _template_pose_from_container(container: dict, context: str) -> dict | None:
    pose_keys = [key for key in TEMPLATE_POSE_KEYS if key in container]
    if len(pose_keys) > 1:
        raise ValueError(
            f"{context} defines multiple template pose keys {pose_keys}. "
            f"Use exactly one of {TEMPLATE_POSE_KEYS}."
        )
    if len(pose_keys) == 1:
        return dict(container[pose_keys[0]])
    return None


def _template_usage_pose_cfg(
    spec: dict, template_cfg: dict, object_id: str
) -> dict | None:
    template_pose = _template_pose_from_container(
        template_cfg, f"template `{_template_id(template_cfg)}`"
    )
    if template_pose is not None:
        return template_pose
    if "scene_objects" in spec:
        matches = [
            scene_object
            for scene_object in spec["scene_objects"]
            if str(scene_object["id"]) == object_id
        ]
        if len(matches) == 1:
            object_pose = _template_pose_from_container(
                matches[0], f"scene object `{object_id}`"
            )
            if object_pose is not None:
                return object_pose
    spec_pose = _template_pose_from_container(spec, "trajectory spec")
    if spec_pose is not None:
        return spec_pose
    return None


def _apply_template_usage_pose(
    state: TemplateState,
    spec: dict,
    template_cfg: dict,
    object_id: str,
) -> TemplateState:
    pose_cfg = _template_usage_pose_cfg(spec, template_cfg, object_id)
    if pose_cfg is None:
        return state
    return _apply_pose_cfg(
        state,
        template_id=state.template_id,
        pose_cfg=pose_cfg,
    )


def _target_vertex_mean_from_path(path: Path) -> np.ndarray:
    mesh = trimesh.load(path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(
            f"Expected target mesh vertices with shape (N, 3), got {vertices.shape} "
            f"from {path}."
        )
    return vertices.mean(axis=0)


def _target_visual_vertex_mean(state: TemplateState, root: Path) -> np.ndarray:
    requested_target_obj_path = state.metadata.get("oakink_requested_target_obj_path")
    if requested_target_obj_path is None:
        raise ValueError(
            "URDF target retarget metadata must define `oakink_requested_target_obj_path` "
            "to reconstruct the optimization-time base transform."
        )
    target_obj_path = _project_local_path_if_exists(
        root, str(requested_target_obj_path)
    )
    if target_obj_path is None:
        target_transform_info = state.metadata.get("oakink_target_transform_info")
        if target_transform_info is None:
            raise FileNotFoundError(
                f"Expected target visual mesh to exist: {requested_target_obj_path}"
            )
        target_transform_path = _project_local_path(root, str(target_transform_info))
        manifest_path = target_transform_path.parent / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Expected target visual mesh {requested_target_obj_path} or "
                f"transform manifest {manifest_path} to exist."
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if "target_visualization_obj_path" not in manifest:
            raise ValueError(
                f"Transform manifest {manifest_path} must define "
                "`target_visualization_obj_path` when the requested visual mesh "
                f"is missing: {requested_target_obj_path}"
            )
        target_obj_path = _project_local_path(
            root, str(manifest["target_visualization_obj_path"])
        )
    return _target_vertex_mean_from_path(target_obj_path)


def _target_bps_vertex_mean(state: TemplateState, root: Path) -> np.ndarray:
    requested_target_bps_obj_path = state.metadata.get(
        "oakink_requested_target_bps_obj_path"
    )
    if requested_target_bps_obj_path in (None, "", "None"):
        raise ValueError(
            "URDF selected-link retarget metadata must define "
            "`oakink_requested_target_bps_obj_path` when target_pose_frame is "
            "'bps_sampling'."
        )
    target_bps_obj_path = _project_local_path(root, str(requested_target_bps_obj_path))
    return _target_vertex_mean_from_path(target_bps_obj_path)


def _target_reconstruction_vertex_mean(
    state: TemplateState,
    root: Path,
    source_target: dict,
) -> tuple[np.ndarray, str]:
    target_pose_frame = str(source_target.get("target_pose_frame", "visual"))
    if target_pose_frame == "visual":
        return _target_visual_vertex_mean(state, root), target_pose_frame
    if target_pose_frame == "bps_sampling":
        return _target_bps_vertex_mean(state, root), target_pose_frame
    raise ValueError(
        "Expected source_target.target_pose_frame to be 'visual' or "
        f"'bps_sampling', got {target_pose_frame!r}."
    )


def _apply_optimization_target_pose(
    state: TemplateState,
    root: Path,
    source_target: dict | None,
) -> TemplateState:
    if source_target is None:
        return state
    category = str(source_target["category"])
    target_index = int(source_target["target_index"])
    target = get_target_object(category, target_index)
    if target.mesh_path is None or Path(target.mesh_path).suffix.lower() != ".urdf":
        return state
    target_transform_info = state.metadata.get("oakink_target_transform_info")
    if target_transform_info in (None, "", "None"):
        target_transform_path = _source_target_transform_path(root, source_target)
    else:
        target_transform_path = _project_local_path(root, str(target_transform_info))
    if target_transform_path is None:
        return state
    target_rotation = load_target_transform(target_transform_path)
    target_center, target_pose_frame = _target_reconstruction_vertex_mean(
        state, root, source_target
    )
    target_center = target_center * _source_target_scale(root, source_target)
    local_translation = -(target_rotation @ target_center)
    current_rotation = R.from_quat(np.roll(state.object_quat_wxyz, -1))
    current_position = np.asarray(state.object_pos, dtype=np.float64).reshape(3)
    object_rotation = current_rotation * R.from_matrix(target_rotation)
    object_position = (
        current_position + current_rotation.as_matrix() @ local_translation
    )
    quat_wxyz = np.roll(R.from_matrix(target_rotation).as_quat(), 1)
    metadata = dict(state.metadata)
    metadata["retarget_target_transform_path"] = str(target_transform_path)
    metadata["retarget_target_transform_quat_wxyz"] = quat_wxyz.tolist()
    metadata["retarget_target_transform_translation"] = local_translation.tolist()
    metadata["retarget_target_visual_center"] = target_center.tolist()
    metadata["retarget_target_pose_frame"] = target_pose_frame
    metadata["retarget_target_transform_applied"] = True
    return replace(
        state,
        object_pos=object_position.astype(np.float32),
        object_quat_wxyz=np.roll(object_rotation.as_quat(), 1).astype(np.float32),
        metadata=metadata,
    )


def _source_target_for_template(
    spec: dict,
    template_cfg: dict,
    object_id: str,
) -> dict | None:
    def with_from_retarget_articulation_sample(source_target: dict) -> dict:
        from_retarget = template_cfg.get("from_retarget")
        if (
            isinstance(from_retarget, dict)
            and "articulation_sample_id" in from_retarget
        ):
            source_target["articulation_sample_id"] = str(
                from_retarget["articulation_sample_id"]
            )
        return source_target

    if "source_target" in template_cfg:
        return with_from_retarget_articulation_sample(
            dict(template_cfg["source_target"])
        )
    if "scene_objects" in spec:
        matches = [
            scene_object
            for scene_object in spec["scene_objects"]
            if str(scene_object["id"]) == object_id
        ]
        if len(matches) == 1 and "source_target" in matches[0]:
            return with_from_retarget_articulation_sample(
                dict(matches[0]["source_target"])
            )
    if "source_target" in spec:
        return with_from_retarget_articulation_sample(dict(spec["source_target"]))
    return None


def _attach_source_target_metadata(
    state: TemplateState,
    root: Path,
    spec: dict,
    template_cfg: dict,
    object_id: str,
) -> TemplateState:
    source_target = _source_target_for_template(spec, template_cfg, object_id)
    if source_target is None:
        return state
    category = str(source_target["category"])
    target_index = int(source_target["target_index"])
    target = get_target_object(category, target_index)
    articulated_urdf_path = _source_target_articulated_urdf_path(root, source_target)
    if articulated_urdf_path is not None:
        state = set_object_articulation_model(
            state,
            object_urdf_path=str(articulated_urdf_path),
        )
    if "articulation_sample_id" in source_target:
        sample_id = str(source_target["articulation_sample_id"])
        matches = [
            sample
            for sample in target.articulation_retarget_samples
            if sample.sample_id == sample_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one articulation sample {sample_id!r} for "
                f"{category}/{target.target_id}, found {len(matches)}."
            )
        sample_joint_values = matches[0].joint_values
        name_to_idx = {name: idx for idx, name in enumerate(state.object_arti_names)}
        unknown = sorted(set(sample_joint_values) - set(name_to_idx))
        if unknown:
            raise ValueError(
                f"Articulation sample {sample_id!r} defines unknown joints {unknown}; "
                f"state exposes {state.object_arti_names}."
            )
        object_arti_vals = state.object_arti_vals.copy()
        for joint_name, value in sample_joint_values.items():
            object_arti_vals[name_to_idx[joint_name]] = float(value)
        state = replace(state, object_arti_vals=object_arti_vals)
    metadata = dict(state.metadata)
    if target.bps_sampling_link_names:
        metadata["target_bps_sampling_link_names"] = list(
            target.bps_sampling_link_names
        )
    if "articulation_sample_id" in source_target:
        metadata["articulation_retarget_mode"] = target.articulation_retarget_mode
        metadata["articulation_sample_id"] = str(
            source_target["articulation_sample_id"]
        )
    return replace(state, metadata=metadata)


def _object_pose_changes(start_state: TemplateState, end_state: TemplateState) -> bool:
    pos_delta = float(np.linalg.norm(start_state.object_pos - end_state.object_pos))
    quat_dot = float(
        abs(
            np.dot(
                start_state.object_quat_wxyz.astype(np.float64),
                end_state.object_quat_wxyz.astype(np.float64),
            )
        )
    )
    return pos_delta > 1e-6 or (1.0 - quat_dot) > 1e-6


def _active_non_root_joints_match(
    start_state: TemplateState, end_state: TemplateState
) -> bool:
    root_suffixes = (
        "forearm_tx_link_joint",
        "forearm_ty_link_joint",
        "forearm_tz_link_joint",
        "forearm_roll_link_joint",
        "forearm_pitch_link_joint",
        "forearm_yaw_link_joint",
    )
    mask = np.asarray(
        [
            not any(name.endswith(suffix) for suffix in root_suffixes)
            for name in start_state.active_joint_names
        ],
        dtype=bool,
    )
    return bool(
        np.allclose(
            start_state.active_joint_vals[mask],
            end_state.active_joint_vals[mask],
            atol=1e-6,
        )
    )


def _active_root_joints_match(
    start_state: TemplateState, end_state: TemplateState
) -> bool:
    root_suffixes = (
        "forearm_tx_link_joint",
        "forearm_ty_link_joint",
        "forearm_tz_link_joint",
        "forearm_roll_link_joint",
        "forearm_pitch_link_joint",
        "forearm_yaw_link_joint",
    )
    mask = np.asarray(
        [
            any(name.endswith(suffix) for suffix in root_suffixes)
            for name in start_state.active_joint_names
        ],
        dtype=bool,
    )
    return bool(
        np.allclose(
            start_state.active_joint_vals[mask],
            end_state.active_joint_vals[mask],
            atol=1e-6,
        )
    )


def _looks_like_hand_release(
    start_state: TemplateState, end_state: TemplateState
) -> bool:
    active_delta = float(
        np.linalg.norm(start_state.active_joint_vals - end_state.active_joint_vals)
    )
    return (
        not _object_pose_changes(start_state, end_state)
        and _active_root_joints_match(start_state, end_state)
        and active_delta > 1e-6
    )


def _object_articulation_changes(
    start_state: TemplateState, end_state: TemplateState
) -> bool:
    return not np.allclose(
        start_state.object_arti_vals,
        end_state.object_arti_vals,
        atol=1e-6,
    )


def _uses_per_articulation_retarget(
    start_state: TemplateState, end_state: TemplateState
) -> bool:
    return (
        str(start_state.metadata.get("articulation_retarget_mode", ""))
        == "per_articulation_retarget"
        or str(end_state.metadata.get("articulation_retarget_mode", ""))
        == "per_articulation_retarget"
    )


def _interpolate_template_piece(
    start_state: TemplateState,
    end_state: TemplateState,
    *,
    frames: int,
    articulation_link_name: str | None = None,
) -> dict[str, np.ndarray]:
    if _object_articulation_changes(start_state, end_state):
        if _uses_per_articulation_retarget(start_state, end_state):
            return interpolate_templates(start_state, end_state, frames=frames)
        return interpolate_articulation_link_motion(
            start_state,
            end_state,
            frames=frames,
            link_name=articulation_link_name,
        )
    if _object_pose_changes(start_state, end_state) and _active_non_root_joints_match(
        start_state, end_state
    ):
        return interpolate_pair_templates(start_state, end_state, frames=frames)
    return interpolate_templates(start_state, end_state, frames=frames)


def _derive_bimanual_open_reach_template(
    target_template: TemplateState,
    *,
    template_id: str,
    backoff_translation: np.ndarray,
    backoff_distance: float | None,
    height_offset: float,
    frozen_sides: tuple[str, ...] = (),
) -> TemplateState:
    if set(target_template.active_sides) != {"left", "right"}:
        return derive_open_reach_template(
            target_template,
            template_id=template_id,
            backoff_translation=backoff_translation,
            backoff_distance=backoff_distance,
            height_offset=height_offset,
        )
    for side in frozen_sides:
        if side not in ("left", "right"):
            raise ValueError(
                f"Expected frozen reach side 'left' or 'right', got {side!r}."
            )
    left_start = (
        replace(target_template, active_side="left")
        if "left" in frozen_sides
        else derive_open_reach_template(
            replace(target_template, active_side="left"),
            template_id=f"{template_id}_left",
            backoff_translation=backoff_translation,
            backoff_distance=backoff_distance,
            height_offset=height_offset,
        )
    )
    right_start = (
        replace(target_template, active_side="right")
        if "right" in frozen_sides
        else derive_open_reach_template(
            replace(target_template, active_side="right"),
            template_id=f"{template_id}_right",
            backoff_translation=backoff_translation,
            backoff_distance=backoff_distance,
            height_offset=height_offset,
        )
    )
    start_template = combine_bimanual_template_states(
        template_id=template_id,
        left_state=left_start,
        right_state=right_start,
    )
    return set_template_active_side(
        start_template,
        active_side=target_template.active_side,
        template_id=template_id,
    )


def _relabel_template_hand_role(
    state: TemplateState,
    *,
    role_side: str,
    source_side: str,
) -> TemplateState:
    if role_side not in ("left", "right") or source_side not in ("left", "right"):
        raise ValueError(
            "Expected role_side and source_side to be 'left' or 'right', "
            f"got role_side={role_side!r}, source_side={source_side!r}."
        )
    updates = {
        f"{role_side}_joint_names": getattr(state, f"{source_side}_joint_names"),
        f"{role_side}_joint_vals": getattr(state, f"{source_side}_joint_vals"),
    }
    for kind in ("skeleton", "bps"):
        updates[f"{role_side}_human_{kind}_keypoint_pos"] = getattr(
            state,
            f"{source_side}_human_{kind}_keypoint_pos",
        )
        updates[f"{role_side}_human_{kind}_keypoint_valid"] = getattr(
            state,
            f"{source_side}_human_{kind}_keypoint_valid",
        )
    metadata = dict(state.metadata)
    source_by_side = dict(metadata.get("hand_config_source_by_side", {}))
    source_by_side[role_side] = source_side
    metadata["hand_config_source_by_side"] = source_by_side
    return replace(state, active_side=role_side, metadata=metadata, **updates)


def _validate_template_source_hand_side(
    state: TemplateState,
    *,
    source_side: str,
    template_id: str,
) -> TemplateState:
    if source_side not in ("left", "right"):
        raise ValueError(
            f"Expected source_side 'left' or 'right', got {source_side!r}."
        )
    if source_side != state.active_side:
        raise ValueError(
            f"Template {template_id!r} requests source_hand_side={source_side!r}, "
            f"but retarget motion is active on {state.active_side!r}. Regenerate "
            "the retargeting for the requested source hand; cross-hand copying is "
            "not a valid substitute for a solved retargeting."
        )
    return state


def _combine_mixed_object_bimanual_template_states(
    *,
    template_id: str,
    left_state: TemplateState,
    right_state: TemplateState,
    active_side: str,
    left_object_id: str,
    right_object_id: str,
) -> TemplateState:
    if left_state.active_side != "left":
        raise ValueError(
            f"Expected mixed bimanual left role to have active_side='left', "
            f"got {left_state.active_side!r}."
        )
    if right_state.active_side != "right":
        raise ValueError(
            f"Expected mixed bimanual right role to have active_side='right', "
            f"got {right_state.active_side!r}."
        )
    if active_side not in ("left", "right"):
        raise ValueError(
            "Expected mixed bimanual active_side 'left' or 'right', "
            f"got {active_side!r}."
        )
    if left_state.robot_model != right_state.robot_model:
        raise ValueError(
            "Cannot combine mixed bimanual templates with different robot models: "
            f"{left_state.robot_model!r} vs {right_state.robot_model!r}."
        )
    object_state = left_state if active_side == "left" else right_state
    source_by_side = {
        **dict(left_state.metadata.get("hand_config_source_by_side", {})),
        **dict(right_state.metadata.get("hand_config_source_by_side", {})),
    }
    metadata = dict(object_state.metadata)
    metadata.update(
        {
            "active_sides": ("left", "right"),
            "mixed_object_bimanual_template": True,
            "left_template_id": left_state.template_id,
            "right_template_id": right_state.template_id,
            "left_object_id": str(left_object_id),
            "right_object_id": str(right_object_id),
            "hand_config_source_by_side": source_by_side,
        }
    )
    return TemplateState(
        template_id=template_id,
        active_side=active_side,
        robot_model=object_state.robot_model,
        left_joint_names=left_state.left_joint_names,
        left_joint_vals=left_state.left_joint_vals,
        right_joint_names=right_state.right_joint_names,
        right_joint_vals=right_state.right_joint_vals,
        object_name=object_state.object_name,
        object_urdf_path=object_state.object_urdf_path,
        object_pos=object_state.object_pos,
        object_quat_wxyz=object_state.object_quat_wxyz,
        object_arti_vals=object_state.object_arti_vals,
        object_arti_names=object_state.object_arti_names,
        metadata=metadata,
        human_skeleton_keypoint_names=object_state.human_skeleton_keypoint_names,
        human_bps_keypoint_names=object_state.human_bps_keypoint_names,
        left_human_skeleton_keypoint_pos=left_state.left_human_skeleton_keypoint_pos,
        right_human_skeleton_keypoint_pos=(
            right_state.right_human_skeleton_keypoint_pos
        ),
        left_human_bps_keypoint_pos=left_state.left_human_bps_keypoint_pos,
        right_human_bps_keypoint_pos=right_state.right_human_bps_keypoint_pos,
        left_human_skeleton_keypoint_valid=(
            left_state.left_human_skeleton_keypoint_valid
        ),
        right_human_skeleton_keypoint_valid=(
            right_state.right_human_skeleton_keypoint_valid
        ),
        left_human_bps_keypoint_valid=left_state.left_human_bps_keypoint_valid,
        right_human_bps_keypoint_valid=right_state.right_human_bps_keypoint_valid,
    )


def _canonicalize_bimanual_object_state(
    side_states: dict[str, TemplateState],
    *,
    canonical_side: str,
) -> dict[str, TemplateState]:
    if canonical_side not in ("left", "right"):
        raise ValueError(
            "Expected canonical_object_side to be 'left' or 'right', "
            f"got {canonical_side!r}."
        )
    canonical_state = side_states[canonical_side]
    return {
        side: replace(
            state,
            object_name=canonical_state.object_name,
            object_urdf_path=canonical_state.object_urdf_path,
        )
        for side, state in side_states.items()
    }


def _build_template_entries(
    root: Path,
    spec: dict,
    retarget_hand_type: str,
) -> dict[str, TemplateEntry]:
    templates: dict[str, TemplateEntry] = {}
    template_cfgs = spec.get("templates", spec.get("keyframes", []))

    def resolve_transform_refs(transforms: list[dict]) -> list[dict]:
        resolved = []
        for transform in transforms:
            transform = dict(transform)
            if str(transform.get("type", "")) == "align_pair_lowest_z":
                target_lowest_z_offset = float(
                    transform.pop("target_lowest_z_offset", 0.0)
                )
                reference_template = transform.pop(
                    "target_lowest_z_from_template", None
                )
                reference_template_top = transform.pop(
                    "target_lowest_z_from_template_top", None
                )
                if (
                    reference_template is not None
                    and reference_template_top is not None
                ):
                    raise ValueError(
                        "align_pair_lowest_z must define only one template z reference."
                    )
                if reference_template is not None:
                    reference_template = str(reference_template)
                    if reference_template not in templates:
                        raise ValueError(
                            "align_pair_lowest_z references unknown template "
                            f"{reference_template!r}."
                        )
                    if "target_lowest_z" in transform:
                        raise ValueError(
                            "align_pair_lowest_z must define only one of "
                            "target_lowest_z or target_lowest_z_from_template."
                        )
                    transform["target_lowest_z"] = (
                        object_lowest_z(templates[reference_template].state)
                        + target_lowest_z_offset
                    )
                if reference_template_top is not None:
                    reference_template_top = str(reference_template_top)
                    if reference_template_top not in templates:
                        raise ValueError(
                            "align_pair_lowest_z references unknown template "
                            f"{reference_template_top!r}."
                        )
                    if "target_lowest_z" in transform:
                        raise ValueError(
                            "align_pair_lowest_z must define only one of "
                            "target_lowest_z or target_lowest_z_from_template_top."
                        )
                    transform["target_lowest_z"] = (
                        object_highest_z(templates[reference_template_top].state)
                        + target_lowest_z_offset
                    )
                if (
                    reference_template is None
                    and reference_template_top is None
                    and target_lowest_z_offset != 0.0
                ):
                    if "target_lowest_z" not in transform:
                        raise ValueError(
                            "align_pair_lowest_z with target_lowest_z_offset must "
                            "define target_lowest_z or a template z reference."
                        )
                    transform["target_lowest_z"] = (
                        float(transform["target_lowest_z"]) + target_lowest_z_offset
                    )
            resolved.append(transform)
        return resolved

    for template_cfg in template_cfgs:
        template_id = _template_id(template_cfg)
        if "bimanual_from_retarget" in template_cfg:
            side_cfgs = dict(template_cfg["bimanual_from_retarget"])
            missing_sides = [
                side for side in ("left", "right") if side not in side_cfgs
            ]
            if missing_sides:
                raise ValueError(
                    f"Bimanual template {template_id!r} must define left and right "
                    f"retarget configs; missing {missing_sides}."
                )
            side_states: dict[str, TemplateState] = {}
            object_id = str(template_cfg.get("object_id", "asset"))
            for side in ("left", "right"):
                side_cfg = dict(side_cfgs[side])
                side_cfg["id"] = f"{template_id}_{side}"
                side_object_id = str(side_cfg.get("object_id", object_id))
                side_cfg["object_id"] = side_object_id
                source_template_id = side_cfg.get(
                    "from_template", side_cfg.get("from_keyframe")
                )
                if source_template_id is not None:
                    source_template_id = str(source_template_id)
                    if source_template_id not in templates:
                        raise ValueError(
                            f"Bimanual template {template_id!r} side {side!r} "
                            f"references unknown from_template {source_template_id!r}."
                        )
                    source_entry = templates[source_template_id]
                    if source_entry.object_id != side_object_id:
                        raise ValueError(
                            f"Bimanual template {template_id!r} side {side!r} "
                            f"uses object_id={side_object_id!r}, but "
                            f"from_template {source_template_id!r} belongs to "
                            f"{source_entry.object_id!r}."
                        )
                    state = replace(
                        source_entry.state,
                        template_id=f"{template_id}_{side}",
                    )
                elif "from_retarget" in side_cfg:
                    if "source_target" not in side_cfg:
                        raise ValueError(
                            f"Bimanual template {template_id!r} side {side!r} "
                            "must define source_target when using from_retarget."
                        )
                    motion_path = _resolve_keyframe_motion_path(
                        root,
                        spec,
                        side_cfg,
                        retarget_hand_type,
                    )
                    state = load_template_state_from_motion(
                        motion_path,
                        template_id=f"{template_id}_{side}",
                        frame=side_cfg["from_retarget"].get(
                            "frame", side_cfg.get("frame", "last")
                        ),
                    )
                    source_target = _source_target_for_template(
                        spec,
                        side_cfg,
                        side_object_id,
                    )
                    state = _apply_optimization_target_pose(state, root, source_target)
                    state = _apply_template_usage_pose(
                        state,
                        spec,
                        side_cfg,
                        side_object_id,
                    )
                    state = _attach_source_target_metadata(
                        state,
                        root,
                        spec,
                        side_cfg,
                        side_object_id,
                    )
                else:
                    raise ValueError(
                        f"Bimanual template {template_id!r} side {side!r} must define "
                        "from_retarget or from_template."
                    )
                source_side = str(side_cfg.get("source_hand_side", state.active_side))
                state = _validate_template_source_hand_side(
                    state,
                    source_side=source_side,
                    template_id=f"{template_id}_{side}",
                )
                if source_side != side:
                    state = _relabel_template_hand_role(
                        state,
                        role_side=side,
                        source_side=source_side,
                    )
                side_states[side] = state
            canonical_object_side = template_cfg.get("canonical_object_side", None)
            if canonical_object_side is not None:
                side_states = _canonicalize_bimanual_object_state(
                    side_states,
                    canonical_side=str(canonical_object_side),
                )
            if bool(template_cfg.get("allow_mixed_objects", False)):
                state = _combine_mixed_object_bimanual_template_states(
                    template_id=template_id,
                    left_state=side_states["left"],
                    right_state=side_states["right"],
                    active_side=str(template_cfg.get("active_side", "right")),
                    left_object_id=str(side_cfgs["left"].get("object_id", object_id)),
                    right_object_id=str(side_cfgs["right"].get("object_id", object_id)),
                )
            else:
                state = combine_bimanual_template_states(
                    template_id=template_id,
                    left_state=side_states["left"],
                    right_state=side_states["right"],
                )
            state = derive_template_from_transforms(
                state,
                template_id=template_id,
                transforms=resolve_transform_refs(
                    list(template_cfg.get("transforms", []))
                ),
            )
            if "active_side" in template_cfg and not bool(
                template_cfg.get("allow_mixed_objects", False)
            ):
                state = set_template_active_side(
                    state,
                    active_side=str(template_cfg["active_side"]),
                    template_id=template_id,
                )
            templates[template_id] = TemplateEntry(
                state=state,
                object_id=object_id,
            )
        elif "mirror_from_template" in template_cfg:
            source_template_id = str(template_cfg["mirror_from_template"])
            if source_template_id not in templates:
                raise ValueError(
                    f"Mirror template {template_id!r} references unknown "
                    f"mirror_from_template {source_template_id!r}."
                )
            source_entry = templates[source_template_id]
            object_id = str(template_cfg.get("object_id", source_entry.object_id))
            if object_id != source_entry.object_id:
                raise ValueError(
                    f"Mirror template {template_id!r} has object_id {object_id!r}, "
                    f"but source template {source_template_id!r} uses "
                    f"{source_entry.object_id!r}."
                )
            state = mirror_template_state(
                source_entry.state,
                template_id=template_id,
                source_side=str(
                    template_cfg.get("source_side", source_entry.state.active_side)
                ),
                target_side=str(template_cfg["target_side"]),
                mirror_axis=str(template_cfg.get("mirror_axis", "x")),
                mirror_frame=str(template_cfg.get("mirror_frame", "object")),
            )
            state = derive_template_from_transforms(
                state,
                template_id=template_id,
                transforms=resolve_transform_refs(
                    list(template_cfg.get("transforms", []))
                ),
            )
            if "active_side" in template_cfg:
                state = set_template_active_side(
                    state,
                    active_side=str(template_cfg["active_side"]),
                    template_id=template_id,
                )
            templates[template_id] = TemplateEntry(
                state=state,
                object_id=object_id,
            )
        elif "from_retarget" in template_cfg:
            motion_path = _resolve_keyframe_motion_path(
                root,
                spec,
                template_cfg,
                retarget_hand_type,
            )
            object_id = str(template_cfg.get("object_id", "asset"))
            state = load_template_state_from_motion(
                motion_path,
                template_id=template_id,
                frame=template_cfg["from_retarget"].get(
                    "frame",
                    template_cfg.get("frame", "last"),
                ),
            )
            source_side = str(template_cfg.get("source_hand_side", state.active_side))
            state = _validate_template_source_hand_side(
                state,
                source_side=source_side,
                template_id=template_id,
            )
            source_target = _source_target_for_template(spec, template_cfg, object_id)
            state = _apply_optimization_target_pose(state, root, source_target)
            state = _attach_source_target_metadata(
                state, root, spec, template_cfg, object_id
            )
            state = _apply_template_usage_pose(state, spec, template_cfg, object_id)
            state = derive_template_from_transforms(
                state,
                template_id=template_id,
                transforms=resolve_transform_refs(
                    list(template_cfg.get("transforms", []))
                ),
            )
            templates[template_id] = TemplateEntry(
                state=state,
                object_id=object_id,
            )
        elif "retarget_job_id" in template_cfg:
            motion_path = _resolve_retarget_motion_path(
                root, spec, str(template_cfg["retarget_job_id"])
            )
            object_id = str(template_cfg.get("object_id", "asset"))
            state = load_template_state_from_motion(
                motion_path,
                template_id=template_id,
                frame=template_cfg.get("frame", "last"),
            )
            source_target = _source_target_for_template(spec, template_cfg, object_id)
            state = _apply_optimization_target_pose(state, root, source_target)
            state = _attach_source_target_metadata(
                state, root, spec, template_cfg, object_id
            )
            state = _apply_template_usage_pose(state, spec, template_cfg, object_id)
            state = derive_template_from_transforms(
                state,
                template_id=template_id,
                transforms=resolve_transform_refs(
                    list(template_cfg.get("transforms", []))
                ),
            )
            templates[template_id] = TemplateEntry(
                state=state,
                object_id=object_id,
            )
        elif "base_template_id" in template_cfg or "derive_from" in template_cfg:
            base_template_id = str(
                template_cfg.get("base_template_id", template_cfg.get("derive_from"))
            )
            if base_template_id not in templates:
                raise ValueError(
                    f"Base template {base_template_id!r} must be defined before {template_id!r}."
                )
            base_entry = templates[base_template_id]
            object_id = str(template_cfg.get("object_id", base_entry.object_id))
            if object_id != base_entry.object_id:
                raise ValueError(
                    f"Derived template {template_id!r} has object_id {object_id!r}, "
                    f"but its base template {base_template_id!r} uses {base_entry.object_id!r}."
                )
            state = derive_template_from_transforms(
                base_entry.state,
                template_id=template_id,
                transforms=resolve_transform_refs(
                    list(template_cfg.get("transforms", []))
                ),
            )
            if "active_side" in template_cfg:
                state = set_template_active_side(
                    state,
                    active_side=str(template_cfg["active_side"]),
                    template_id=template_id,
                )
            templates[template_id] = TemplateEntry(state=state, object_id=object_id)
        else:
            raise ValueError(
                f"Template {template_id!r} must define from_retarget, retarget_job_id, "
                "mirror_from_template, base_template_id, or derive_from."
            )
    return templates


def _build_templates(root: Path, spec: dict) -> dict[str, TemplateState]:
    return {
        template_id: entry.state
        for template_id, entry in _build_template_entries(root, spec).items()
    }


def _primitive_piece(
    primitive: dict, templates: dict[str, TemplateState]
) -> tuple[dict[str, np.ndarray], TemplateState]:
    primitive_type = str(primitive["type"])
    frames = int(primitive["frames"])

    if primitive_type in ("hold_template", "hold_keyframe"):
        template = templates[
            str(primitive.get("template_id", primitive.get("keyframe")))
        ]
        return hold_template(template, frames=frames), template
    if primitive_type in ("interpolate_templates", "interpolate_keyframes"):
        start_template = templates[
            str(primitive.get("start_template_id", primitive.get("start")))
        ]
        end_template = templates[
            str(primitive.get("end_template_id", primitive.get("end")))
        ]
        if _looks_like_hand_release(start_template, end_template):
            return (
                release_or_open(
                    start_template,
                    frames=frames,
                    clearance_translation=np.asarray(
                        primitive.get("clearance_translation", [0.0, 0.0, 0.08]),
                        dtype=np.float32,
                    ),
                ),
                start_template,
            )
        return (
            _interpolate_template_piece(
                start_template,
                end_template,
                frames=frames,
                articulation_link_name=primitive.get("articulation_link_name"),
            ),
            start_template,
        )
    if primitive_type in ("reach_to_template", "reach_to_keyframe"):
        target_template = templates[
            str(primitive.get("target_template_id", primitive.get("target")))
        ]
        start_template = _derive_bimanual_open_reach_template(
            target_template,
            template_id=f"{target_template.template_id}_reach_start",
            backoff_translation=np.asarray(
                primitive.get("backoff_translation", [-0.08, 0.0, 0.0]),
                dtype=np.float32,
            ),
            backoff_distance=(
                None
                if "backoff_distance" not in primitive
                else float(primitive["backoff_distance"])
            ),
            height_offset=float(primitive.get("height_offset", 0.06)),
            frozen_sides=tuple(str(side) for side in primitive.get("frozen_sides", ())),
        )
        return (
            interpolate_templates(start_template, target_template, frames=frames),
            target_template,
        )
    if primitive_type == "move_pair_to_pose":
        start_template = templates[str(primitive["start"])]
        return (
            move_pair_to_pose(
                start_template,
                frames=frames,
                object_pos=np.asarray(primitive["object_pos"], dtype=np.float32),
                object_quat_wxyz=(
                    None
                    if "object_quat_wxyz" not in primitive
                    else np.asarray(primitive["object_quat_wxyz"], dtype=np.float32)
                ),
            ),
            start_template,
        )
    if primitive_type == "place_object":
        start_template = templates[str(primitive["start"])]
        return (
            place_object(
                start_template,
                frames=frames,
                object_pos=np.asarray(primitive["object_pos"], dtype=np.float32),
                object_quat_wxyz=(
                    None
                    if "object_quat_wxyz" not in primitive
                    else np.asarray(primitive["object_quat_wxyz"], dtype=np.float32)
                ),
            ),
            start_template,
        )
    if primitive_type == "release_or_open":
        start_template = templates[str(primitive["start"])]
        return (
            release_or_open(
                start_template,
                frames=frames,
                clearance_translation=np.asarray(
                    primitive.get("clearance_translation", [0.0, 0.0, 0.08]),
                    dtype=np.float32,
                ),
            ),
            start_template,
        )
    if primitive_type == "articulate_object":
        start_template = templates[str(primitive["start"])]
        return (
            articulate_object(
                start_template,
                frames=frames,
                values=(
                    None
                    if "values" not in primitive
                    else np.asarray(primitive["values"], dtype=np.float32)
                ),
                joint_values=primitive.get("joint_values"),
                link_name=primitive.get("articulation_link_name"),
            ),
            start_template,
        )
    raise ValueError(f"Unsupported primitive type {primitive_type!r}.")


def _primitive_template_ids(primitive: dict) -> list[str]:
    primitive_type = str(primitive["type"])
    if primitive_type in ("hold_template", "hold_keyframe"):
        return [str(primitive.get("template_id", primitive.get("keyframe")))]
    if primitive_type in ("interpolate_templates", "interpolate_keyframes"):
        return [
            str(primitive.get("start_template_id", primitive.get("start"))),
            str(primitive.get("end_template_id", primitive.get("end"))),
        ]
    if primitive_type in ("reach_to_template", "reach_to_keyframe"):
        return [str(primitive.get("target_template_id", primitive.get("target")))]
    if primitive_type in {
        "move_pair_to_pose",
        "place_object",
        "release_or_open",
        "articulate_object",
    }:
        return [str(primitive["start"])]
    raise ValueError(f"Unsupported primitive type {primitive_type!r}.")


def _validate_scene_objects(spec: dict) -> list[str]:
    scene_objects = spec.get("scene_objects", [])
    if not scene_objects:
        raise ValueError("Multi-object specs must define non-empty `scene_objects`.")
    object_ids = [str(scene_object["id"]) for scene_object in scene_objects]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError(f"Scene object ids must be unique, got {object_ids}.")
    return object_ids


def _support_scene_object_cfgs(spec: dict) -> list[dict]:
    cfgs = [
        dict(scene_object) for scene_object in spec.get("support_scene_objects", [])
    ]
    object_ids = [str(scene_object["id"]) for scene_object in cfgs]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError(f"Support scene object ids must be unique, got {object_ids}.")
    scene_object_ids = {
        str(scene_object["id"]) for scene_object in spec.get("scene_objects", [])
    }
    overlap = sorted(scene_object_ids & set(object_ids))
    if overlap:
        raise ValueError(
            f"Support scene object ids overlap with active scene objects: {overlap}."
        )
    return cfgs


def _static_scene_object_state(
    root: Path, spec: dict, scene_object: dict
) -> TemplateState:
    object_id = str(scene_object["id"])
    if "mesh_path" not in scene_object and "urdf_path" not in scene_object:
        raise ValueError(
            f"Static scene object {object_id!r} must define `mesh_path` or `urdf_path` "
            "when it has no template."
        )
    asset_path = _rooted_path(
        root, str(scene_object.get("urdf_path", scene_object.get("mesh_path")))
    ).resolve()
    if not asset_path.exists():
        raise FileNotFoundError(f"Static scene object asset not found: {asset_path}")
    scale = float(scene_object.get("scale", 1.0))
    if scale <= 0.0:
        raise ValueError(
            f"Static scene object {object_id!r} scale must be positive, got {scale}."
        )
    if asset_path.suffix.lower() == ".urdf":
        if not np.isclose(scale, 1.0):
            raise ValueError(
                f"Static scene object {object_id!r} uses a URDF asset and must use "
                f"scale=1.0, got {scale}."
            )
        urdf_path = asset_path
    else:
        urdf_dir = (
            _output_motion_path(root, spec).parent / "static_scene_objects" / object_id
        )
        urdf_dir.mkdir(parents=True, exist_ok=True)
        urdf_path = Path(
            create_urdf_from_obj(str(asset_path), str(urdf_dir), scale=scale)
        )
    pose_cfg = dict(
        scene_object.get(
            "template_pose",
            {
                "object_pos": scene_object.get("pos", [0.0, 0.0, 0.0]),
                "object_quat_wxyz": scene_object.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
            },
        )
    )
    object_pos = np.asarray(
        pose_cfg.get("object_pos", pose_cfg.get("pos")), dtype=np.float32
    ).reshape(3)
    object_quat_wxyz = np.asarray(
        pose_cfg.get(
            "object_quat_wxyz",
            pose_cfg.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
        ),
        dtype=np.float32,
    ).reshape(4)
    return TemplateState(
        template_id=f"{object_id}_static",
        active_side=str(spec.get("hand_side", "right")),
        robot_model=str(spec.get("robot_model", "xhand")),
        left_joint_names=[],
        left_joint_vals=np.zeros((0,), dtype=np.float32),
        right_joint_names=[],
        right_joint_vals=np.zeros((0,), dtype=np.float32),
        object_name=object_id,
        object_urdf_path=str(urdf_path),
        object_pos=object_pos,
        object_quat_wxyz=object_quat_wxyz,
        object_arti_vals=np.zeros((0,), dtype=np.float32),
        object_arti_names=[],
        metadata={
            "static_scene_object": True,
            "object_friction": _object_friction_from_cfg(scene_object),
        },
    )


def _support_scene_object_states(root: Path, spec: dict) -> dict[str, TemplateState]:
    states: dict[str, TemplateState] = {}
    for scene_object in _support_scene_object_cfgs(spec):
        object_id = str(scene_object["id"])
        if not _object_static_from_cfg(scene_object):
            raise ValueError(
                f"Support scene object {object_id!r} must be static/fixed."
            )
        states[object_id] = _static_scene_object_state(root, spec, scene_object)
    return states


def _initial_scene_states(
    root: Path,
    spec: dict,
    object_ids: list[str],
    entries: dict[str, TemplateEntry],
) -> dict[str, TemplateState]:
    states: dict[str, TemplateState] = {}
    scene_objects = {
        str(scene_object["id"]): dict(scene_object)
        for scene_object in spec.get("scene_objects", [])
    }
    for object_id in object_ids:
        for entry in entries.values():
            if entry.object_id == object_id:
                states[object_id] = entry.state
                break
        if object_id not in states:
            scene_object = scene_objects[object_id]
            if _object_static_from_cfg(scene_object) and (
                "mesh_path" in scene_object or "urdf_path" in scene_object
            ):
                states[object_id] = _static_scene_object_state(root, spec, scene_object)
            else:
                raise ValueError(
                    f"Scene object {object_id!r} has no associated keyframe/template."
                )
    return states


def _repeat_current_object(
    state: TemplateState, frames: int, max_arti: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arti = np.zeros((frames, max_arti), dtype=np.float32)
    n_arti = state.object_arti_vals.shape[0]
    if n_arti:
        arti[:, :n_arti] = np.repeat(state.object_arti_vals[None, :], frames, axis=0)
    return (
        np.repeat(state.object_pos[None, :], frames, axis=0).astype(np.float32),
        np.repeat(state.object_quat_wxyz[None, :], frames, axis=0).astype(np.float32),
        arti,
    )


def _pad_piece_arti(asset_arti_vals: np.ndarray, max_arti: int) -> np.ndarray:
    asset_arti_vals = np.asarray(asset_arti_vals, dtype=np.float32)
    if asset_arti_vals.shape[1] > max_arti:
        raise ValueError(
            f"Active object has {asset_arti_vals.shape[1]} articulation values, "
            f"but scene max is {max_arti}."
        )
    padded = np.zeros((asset_arti_vals.shape[0], max_arti), dtype=np.float32)
    padded[:, : asset_arti_vals.shape[1]] = asset_arti_vals
    return padded


def _scene_rotvec(scene_quat: np.ndarray) -> np.ndarray:
    flat_quat = scene_quat.reshape(-1, 4)
    flat_rot = R.from_quat(np.roll(flat_quat, -1, axis=-1)).as_rotvec()
    return flat_rot.reshape(scene_quat.shape[0], scene_quat.shape[1], 3).astype(
        np.float32
    )


def _object_transform_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.from_quat(np.roll(quat_wxyz, -1)).as_matrix()
    pose[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return pose


def _object_lowest_z(state: TemplateState) -> float:
    urdf_inst = _load_object_urdf(state.object_urdf_path)
    joint_dict = dict(
        zip(
            state.object_arti_names,
            np.asarray(state.object_arti_vals, dtype=np.float32).reshape(-1),
        )
    )
    fk = urdf_inst.visual_trimesh_fk(joint_dict)
    if len(fk) == 0:
        fk = urdf_inst.collision_trimesh_fk(joint_dict)
    if len(fk) == 0:
        raise ValueError(
            f"Object {state.object_name!r} has no visual or collision meshes."
        )
    mesh_pieces = []
    for mesh, pose in fk.items():
        transformed = mesh.copy()
        transformed.apply_transform(pose)
        mesh_pieces.append(transformed)
    mesh = trimesh.util.concatenate(mesh_pieces)
    mesh.apply_transform(
        _object_transform_matrix(state.object_pos, state.object_quat_wxyz)
    )
    return float(mesh.bounds[0, 2])


def _shift_joint_vals_z(
    joint_names: list[str],
    joint_vals: np.ndarray,
    dz: float,
) -> np.ndarray:
    shifted = np.asarray(joint_vals, dtype=np.float32).copy()
    matches = [
        idx
        for idx, name in enumerate(joint_names)
        if str(name).endswith("forearm_tz_link_joint")
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Expected at most one forearm_tz_link_joint in {joint_names}, got {matches}."
        )
    if len(matches) == 1:
        shifted[matches[0]] += float(dz)
    return shifted


def _shift_template_state_z(state: TemplateState, dz: float) -> TemplateState:
    return _shift_template_state_scene_z(
        state,
        object_dz=float(dz),
        left_dz=float(dz),
        right_dz=float(dz),
    )


def _shift_template_state_scene_z(
    state: TemplateState,
    *,
    object_dz: float,
    left_dz: float,
    right_dz: float,
) -> TemplateState:
    metadata = dict(state.metadata)
    metadata["scene_table_height_equalization_z"] = float(object_dz)
    metadata["scene_table_height_equalization_left_z"] = float(left_dz)
    metadata["scene_table_height_equalization_right_z"] = float(right_dz)
    object_pos = np.asarray(state.object_pos, dtype=np.float32).copy()
    object_pos[2] += float(object_dz)
    keypoint_updates = {}
    for side in ("left", "right"):
        side_dz = left_dz if side == "left" else right_dz
        for kind in ("skeleton", "bps"):
            attr = f"{side}_human_{kind}_keypoint_pos"
            points = getattr(state, attr)
            if points is not None:
                shifted_points = np.asarray(points, dtype=np.float32).copy()
                shifted_points[:, 2] += float(side_dz)
                keypoint_updates[attr] = shifted_points
    return replace(
        state,
        left_joint_vals=_shift_joint_vals_z(
            state.left_joint_names,
            state.left_joint_vals,
            left_dz,
        ),
        right_joint_vals=_shift_joint_vals_z(
            state.right_joint_names,
            state.right_joint_vals,
            right_dz,
        ),
        object_pos=object_pos,
        metadata=metadata,
        **keypoint_updates,
    )


def _shift_template_entry_for_scene_z(
    entry: TemplateEntry,
    z_offsets: dict[str, float],
) -> TemplateEntry:
    object_dz = z_offsets[entry.object_id]
    metadata = dict(entry.state.metadata)
    if bool(metadata.get("mixed_object_bimanual_template", False)):
        left_object_id = str(metadata["left_object_id"])
        right_object_id = str(metadata["right_object_id"])
        left_dz = z_offsets[left_object_id]
        right_dz = z_offsets[right_object_id]
    else:
        left_dz = object_dz
        right_dz = object_dz
    return TemplateEntry(
        state=_shift_template_state_scene_z(
            entry.state,
            object_dz=object_dz,
            left_dz=left_dz,
            right_dz=right_dz,
        ),
        object_id=entry.object_id,
    )


def _scene_support_cfg(spec: dict) -> tuple[float, float]:
    sequence_cfg = _sequence_cfg(spec)
    scene_cfg = dict(spec.get("scene", sequence_cfg.get("scene", {})))
    table_height = float(scene_cfg.get("table_height", DEFAULT_SCENE_TABLE_HEIGHT))
    table_margin = float(scene_cfg.get("table_margin", DEFAULT_SCENE_TABLE_MARGIN))
    if table_height < 0.0:
        raise ValueError(
            f"scene.table_height must be non-negative, got {table_height}."
        )
    if table_margin < 0.0:
        raise ValueError(
            f"scene.table_margin must be non-negative, got {table_margin}."
        )
    return table_height, table_margin


def _scene_table_size(spec: dict) -> np.ndarray:
    sequence_cfg = _sequence_cfg(spec)
    scene_cfg = dict(spec.get("scene", sequence_cfg.get("scene", {})))
    size = np.asarray(scene_cfg.get("table_size", [0.8, 1.2]), dtype=np.float32)
    if size.shape != (2,):
        raise ValueError(f"scene.table_size must have shape (2,), got {size.shape}.")
    if np.any(size <= 0.0):
        raise ValueError(f"scene.table_size values must be positive, got {size}.")
    return size


def _scene_support_blocks(spec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequence_cfg = _sequence_cfg(spec)
    scene_cfg = dict(spec.get("scene", sequence_cfg.get("scene", {})))
    block_cfgs = list(scene_cfg.get("support_blocks", []))
    block_ids: list[str] = []
    block_pos: list[np.ndarray] = []
    block_size: list[np.ndarray] = []
    for idx, block_cfg in enumerate(block_cfgs):
        block = dict(block_cfg)
        block_id = str(block.get("id", f"support_block_{idx}"))
        center = np.asarray(block["center"], dtype=np.float32).reshape(3)
        size = np.asarray(block["size"], dtype=np.float32).reshape(3)
        if np.any(size <= 0.0):
            raise ValueError(
                f"Support block {block_id!r} must have positive size, got {size}."
            )
        block_ids.append(block_id)
        block_pos.append(center)
        block_size.append(size)
    if len(block_ids) == 0:
        return (
            np.asarray([], dtype=object),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    if len(block_ids) != len(set(block_ids)):
        raise ValueError(f"Support block ids must be unique, got {block_ids}.")
    return (
        np.asarray(block_ids, dtype=object),
        np.stack(block_pos, axis=0).astype(np.float32),
        np.stack(block_size, axis=0).astype(np.float32),
    )


def _build_single_object_scene_payload(
    spec: dict,
    canonical_template: TemplateState,
    sequence_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    root = get_src()
    support_states = _support_scene_object_states(root, spec)
    support_block_ids, support_block_pos, support_block_size = _scene_support_blocks(
        spec
    )
    if support_block_ids.shape[0] == 0 and len(support_states) == 0:
        return None

    n_frames = int(np.asarray(sequence_arrays["asset_pos"]).shape[0])
    scene_table_height, scene_table_margin = _scene_support_cfg(spec)
    arti_vals = np.asarray(sequence_arrays["asset_arti_vals"], dtype=np.float32)
    if arti_vals.ndim != 2:
        raise ValueError(
            "Expected single-object asset_arti_vals shape (T, A), "
            f"got {arti_vals.shape}."
        )
    if arti_vals.shape[0] != n_frames:
        raise ValueError(
            "Expected asset_arti_vals frame count to match asset_pos frame count "
            f"{n_frames}, got {arti_vals.shape[0]}."
        )

    object_id = str(spec.get("scene_object_id", spec.get("category", "asset")))
    object_quat = np.asarray(sequence_arrays["asset_quat"], dtype=np.float32)
    object_ids = [object_id, *support_states.keys()]
    object_names = [canonical_template.object_name] + [
        state.object_name for state in support_states.values()
    ]
    object_urdf_paths = [canonical_template.object_urdf_path] + [
        state.object_urdf_path for state in support_states.values()
    ]
    n_objects = len(object_ids)
    max_arti = int(arti_vals.shape[1])
    scene_object_pos = np.zeros((n_frames, n_objects, 3), dtype=np.float32)
    scene_object_quat = np.zeros((n_frames, n_objects, 4), dtype=np.float32)
    scene_object_arti_vals = np.zeros((n_frames, n_objects, max_arti), dtype=np.float32)
    scene_object_pos[:, 0, :] = np.asarray(
        sequence_arrays["asset_pos"], dtype=np.float32
    )
    scene_object_quat[:, 0, :] = object_quat
    scene_object_arti_vals[:, 0, :] = arti_vals
    for support_idx, state in enumerate(support_states.values(), start=1):
        scene_object_pos[:, support_idx, :] = state.object_pos
        scene_object_quat[:, support_idx, :] = state.object_quat_wxyz
    scene_object_arti_valid_mask = np.zeros((n_objects, max_arti), dtype=bool)
    scene_object_arti_valid_mask[0, :] = True
    scene_object_active_mask = np.zeros((n_frames, n_objects), dtype=bool)
    scene_object_active_mask[:, 0] = True
    return {
        "is_multi_object_sequence": np.array(True, dtype=bool),
        "scene_object_ids": np.asarray(object_ids, dtype=object),
        "scene_object_names": np.asarray(object_names, dtype=object),
        "scene_object_urdf_paths": np.asarray(object_urdf_paths, dtype=object),
        "scene_object_pos": scene_object_pos,
        "scene_object_quat": scene_object_quat,
        "scene_object_rot": _scene_rotvec(scene_object_quat),
        "scene_object_arti_vals": scene_object_arti_vals,
        "scene_object_arti_valid_mask": scene_object_arti_valid_mask,
        "scene_object_arti_names": np.asarray(
            [canonical_template.object_arti_names]
            + [state.object_arti_names for state in support_states.values()],
            dtype=object,
        ),
        "scene_object_active_mask": scene_object_active_mask,
        "scene_object_static": np.asarray(
            [_spec_object_static(spec)] + [True] * len(support_states),
            dtype=bool,
        ),
        "scene_object_friction": np.asarray(
            [1.0]
            + [
                float(state.metadata["object_friction"])
                for state in support_states.values()
            ],
            dtype=np.float32,
        ),
        "scene_active_object_index": np.zeros(n_frames, dtype=np.int64),
        "scene_canonical_object_id": np.array(object_id, dtype=object),
        "scene_table_height": np.array(scene_table_height, dtype=np.float32),
        "scene_table_margin": np.array(scene_table_margin, dtype=np.float32),
        "scene_table_size": _scene_table_size(spec),
        "scene_uses_block": np.array(False, dtype=bool),
        "scene_object_support_z_offsets": np.zeros(n_objects, dtype=np.float32),
        "scene_support_block_ids": support_block_ids,
        "scene_support_block_pos": support_block_pos,
        "scene_support_block_size": support_block_size,
    }


def _equalize_multi_object_scene_heights(
    *,
    root: Path,
    spec: dict,
    object_ids: list[str],
    entries: dict[str, TemplateEntry],
) -> tuple[dict[str, TemplateEntry], dict[str, float], float, float]:
    table_height, table_margin = _scene_support_cfg(spec)
    current_states = _initial_scene_states(root, spec, object_ids, entries)
    scene_object_cfgs = {
        str(scene_object["id"]): dict(scene_object)
        for scene_object in spec.get("scene_objects", [])
    }
    z_offsets = {}
    for object_id, state in current_states.items():
        scene_object = scene_object_cfgs[object_id]
        lowest_z_above_table = float(scene_object.get("lowest_z_above_table", 0.0))
        if lowest_z_above_table < 0.0:
            raise ValueError(
                f"Expected scene object {object_id!r} lowest_z_above_table to be "
                f"non-negative, got {lowest_z_above_table}."
            )
        z_offsets[object_id] = (
            table_height + table_margin + lowest_z_above_table - _object_lowest_z(state)
        )
    for object_id in object_ids:
        scene_object = scene_object_cfgs[object_id]
        if "z_offset_reference_object" not in scene_object:
            continue
        reference_object_id = str(scene_object["z_offset_reference_object"])
        if reference_object_id not in z_offsets:
            raise ValueError(
                f"Scene object {object_id!r} references unknown "
                f"z_offset_reference_object {reference_object_id!r}; "
                f"expected one of {object_ids}."
            )
        z_offsets[object_id] = z_offsets[reference_object_id]
    shifted_entries = {
        template_id: _shift_template_entry_for_scene_z(entry, z_offsets)
        for template_id, entry in entries.items()
    }
    entry_object_ids = {entry.object_id for entry in entries.values()}
    for object_id, state in current_states.items():
        if object_id not in entry_object_ids:
            shifted_entries[f"__scene_static_{object_id}"] = TemplateEntry(
                state=_shift_template_state_z(state, z_offsets[object_id]),
                object_id=object_id,
            )
    return shifted_entries, z_offsets, table_height, table_margin


def _template_from_piece_frame(
    base_state: TemplateState,
    piece: dict[str, np.ndarray],
    *,
    template_id: str,
    frame_idx: int,
) -> TemplateState:
    keypoint_kwargs = {}
    for side in ("left", "right"):
        for kind in ("skeleton", "bps"):
            pos_key = f"{side}_human_{kind}_keypoint_pos"
            valid_key = f"{side}_human_{kind}_keypoint_valid"
            keypoint_kwargs[pos_key] = (
                np.asarray(piece[pos_key][frame_idx], dtype=np.float32)
                if pos_key in piece
                else getattr(base_state, pos_key)
            )
            keypoint_kwargs[valid_key] = (
                bool(np.asarray(piece[valid_key], dtype=bool)[frame_idx])
                if valid_key in piece
                else bool(getattr(base_state, valid_key))
            )
    return TemplateState(
        template_id=template_id,
        active_side=base_state.active_side,
        robot_model=base_state.robot_model,
        left_joint_names=base_state.left_joint_names,
        left_joint_vals=np.asarray(
            piece["left_joint_vals"][frame_idx], dtype=np.float32
        ),
        right_joint_names=base_state.right_joint_names,
        right_joint_vals=np.asarray(
            piece["right_joint_vals"][frame_idx], dtype=np.float32
        ),
        object_name=base_state.object_name,
        object_urdf_path=base_state.object_urdf_path,
        object_pos=np.asarray(piece["asset_pos"][frame_idx], dtype=np.float32),
        object_quat_wxyz=np.asarray(piece["asset_quat"][frame_idx], dtype=np.float32),
        object_arti_vals=np.asarray(
            piece["asset_arti_vals"][frame_idx], dtype=np.float32
        ),
        object_arti_names=base_state.object_arti_names,
        metadata=base_state.metadata,
        human_skeleton_keypoint_names=base_state.human_skeleton_keypoint_names,
        human_bps_keypoint_names=base_state.human_bps_keypoint_names,
        **keypoint_kwargs,
    )


def _build_multi_object_sequence(
    root: Path, spec: dict, entries: dict[str, TemplateEntry]
) -> tuple[
    dict[str, np.ndarray], list[str], list[int], TemplateState, dict[str, np.ndarray]
]:
    object_ids = _validate_scene_objects(spec)
    support_states = _support_scene_object_states(root, spec)
    support_object_ids = list(support_states.keys())
    all_object_ids = [*object_ids, *support_object_ids]
    entries, scene_z_offsets, scene_table_height, scene_table_margin = (
        _equalize_multi_object_scene_heights(
            root=root,
            spec=spec,
            object_ids=object_ids,
            entries=entries,
        )
    )
    object_idx = {object_id: idx for idx, object_id in enumerate(all_object_ids)}
    current_states = _initial_scene_states(root, spec, object_ids, entries)
    current_states.update(support_states)
    max_arti = max(
        current_states[object_id].object_arti_vals.shape[0]
        for object_id in all_object_ids
    )

    templates = {template_id: entry.state for template_id, entry in entries.items()}
    sequence_cfg = _sequence_cfg(spec)
    primitives = sequence_cfg.get("primitives", sequence_cfg.get("phases"))
    if primitives is None:
        raise ValueError("Sequence config must define `primitives` or `phases`.")

    phase_names: list[str] = []
    phase_lengths: list[int] = []
    hand_pieces: dict[str, list[np.ndarray]] = {}
    scene_pos_pieces: list[np.ndarray] = []
    scene_quat_pieces: list[np.ndarray] = []
    scene_arti_pieces: list[np.ndarray] = []
    active_mask_pieces: list[np.ndarray] = []
    active_index_pieces: list[np.ndarray] = []

    canonical_object_id = object_ids[0]
    canonical_template = current_states[canonical_object_id]
    current_hand_state: TemplateState | None = None
    previous_active_object: str | None = None
    switch_transition_frames = int(
        sequence_cfg.get("object_switch_transition_frames", 12)
    )
    switch_clearance_translation = np.asarray(
        sequence_cfg.get("object_switch_clearance_translation", [0.0, 0.0, 0.08]),
        dtype=np.float32,
    )
    switch_open_fraction = float(sequence_cfg.get("object_switch_open_fraction", 0.7))

    def append_scene_piece(
        *,
        piece: dict[str, np.ndarray],
        frames: int,
        active_object: str | None,
        phase_name: str,
    ) -> None:
        _append_hand_piece(hand_pieces, piece)
        phase_names.append(phase_name)
        phase_lengths.append(frames)

        scene_pos = np.zeros((frames, len(all_object_ids), 3), dtype=np.float32)
        scene_quat = np.zeros((frames, len(all_object_ids), 4), dtype=np.float32)
        scene_arti = np.zeros((frames, len(all_object_ids), max_arti), dtype=np.float32)
        active_mask = np.zeros((frames, len(all_object_ids)), dtype=bool)
        active_index_value = -1 if active_object is None else object_idx[active_object]
        active_index = np.full(frames, active_index_value, dtype=np.int64)

        for object_id in all_object_ids:
            idx = object_idx[object_id]
            if object_id == active_object:
                scene_pos[:, idx, :] = piece["asset_pos"]
                scene_quat[:, idx, :] = piece["asset_quat"]
                scene_arti[:, idx, :] = _pad_piece_arti(
                    piece["asset_arti_vals"], max_arti
                )
                active_mask[:, idx] = True
            else:
                pos, quat, arti = _repeat_current_object(
                    current_states[object_id], frames, max_arti
                )
                scene_pos[:, idx, :] = pos
                scene_quat[:, idx, :] = quat
                scene_arti[:, idx, :] = arti

        scene_pos_pieces.append(scene_pos)
        scene_quat_pieces.append(scene_quat)
        scene_arti_pieces.append(scene_arti)
        active_mask_pieces.append(active_mask)
        active_index_pieces.append(active_index)

    for primitive in primitives:
        active_object = str(primitive.get("active_object", ""))
        if active_object not in object_ids:
            raise ValueError(
                f"Primitive {primitive.get('name', primitive.get('type'))!r} must define "
                f"`active_object` from {object_ids}; got {active_object!r}."
            )
        for template_id in _primitive_template_ids(primitive):
            if template_id not in entries:
                raise ValueError(
                    f"Primitive references unknown template {template_id!r}."
                )
            template_object_id = entries[template_id].object_id
            if template_object_id != active_object:
                raise ValueError(
                    f"Primitive {primitive.get('name', primitive.get('type'))!r} "
                    f"uses active_object={active_object!r} but template {template_id!r} "
                    f"belongs to {template_object_id!r}."
                )

        piece, piece_template = _primitive_piece(primitive, templates)
        frames = int(primitive["frames"])
        if piece["asset_pos"].shape[0] != frames:
            raise ValueError(
                f"Primitive {primitive.get('name', primitive.get('type'))!r} declared "
                f"{frames} frames but produced {piece['asset_pos'].shape[0]}."
            )

        if (
            current_hand_state is not None
            and previous_active_object is not None
            and active_object != previous_active_object
            and switch_transition_frames > 0
        ):
            next_start_state = _template_from_piece_frame(
                piece_template,
                piece,
                template_id=f"{piece_template.template_id}_switch_start",
                frame_idx=0,
            )
            transition_piece = hand_transition_via_clearance(
                current_hand_state,
                next_start_state,
                frames=switch_transition_frames,
                clearance_translation=switch_clearance_translation,
                open_fraction=switch_open_fraction,
                object_state=current_hand_state,
            )
            append_scene_piece(
                piece=transition_piece,
                frames=switch_transition_frames,
                active_object=None,
                phase_name=f"switch_{previous_active_object}_to_{active_object}",
            )
            current_hand_state = _template_from_piece_frame(
                current_hand_state,
                transition_piece,
                template_id=f"{current_hand_state.template_id}_switch_end",
                frame_idx=-1,
            )

        append_scene_piece(
            piece=piece,
            frames=frames,
            active_object=active_object,
            phase_name=str(
                primitive.get("phase_name", primitive.get("name", primitive["type"]))
            ),
        )

        active_state = current_states[active_object]
        current_states[active_object] = _template_from_piece_frame(
            active_state,
            piece,
            template_id=f"{active_state.template_id}_scene_current",
            frame_idx=-1,
        )
        current_hand_state = current_states[active_object]
        previous_active_object = active_object

    sequence_arrays = _concat_piecewise(hand_pieces)
    scene_pos = np.concatenate(scene_pos_pieces, axis=0)
    scene_quat = np.concatenate(scene_quat_pieces, axis=0)
    scene_arti_vals = np.concatenate(scene_arti_pieces, axis=0)
    scene_active_object_index = np.concatenate(active_index_pieces, axis=0)

    canonical_idx = object_idx[canonical_object_id]
    sequence_arrays["asset_pos"] = scene_pos[:, canonical_idx, :]
    sequence_arrays["asset_quat"] = scene_quat[:, canonical_idx, :]
    canonical_arti_dim = canonical_template.object_arti_vals.shape[0]
    sequence_arrays["asset_arti_vals"] = scene_arti_vals[
        :, canonical_idx, :canonical_arti_dim
    ]

    scene_arti_valid_mask = np.zeros((len(all_object_ids), max_arti), dtype=bool)
    for object_id in all_object_ids:
        idx = object_idx[object_id]
        n_arti = current_states[object_id].object_arti_vals.shape[0]
        scene_arti_valid_mask[idx, :n_arti] = True

    scene_payload = {
        "is_multi_object_sequence": np.array(True, dtype=bool),
        "scene_object_ids": np.asarray(all_object_ids, dtype=object),
        "scene_object_names": np.asarray(
            [current_states[object_id].object_name for object_id in all_object_ids],
            dtype=object,
        ),
        "scene_object_urdf_paths": np.asarray(
            [
                current_states[object_id].object_urdf_path
                for object_id in all_object_ids
            ],
            dtype=object,
        ),
        "scene_object_pos": scene_pos,
        "scene_object_quat": scene_quat,
        "scene_object_rot": _scene_rotvec(scene_quat),
        "scene_object_arti_vals": scene_arti_vals,
        "scene_object_arti_valid_mask": scene_arti_valid_mask,
        "scene_object_arti_names": np.asarray(
            [
                current_states[object_id].object_arti_names
                for object_id in all_object_ids
            ],
            dtype=object,
        ),
        "scene_object_active_mask": np.concatenate(active_mask_pieces, axis=0),
        "scene_object_static": np.asarray(
            [
                _object_static_from_cfg(scene_object)
                for scene_object in spec["scene_objects"]
            ]
            + [True] * len(support_states),
            dtype=bool,
        ),
        "scene_object_friction": np.asarray(
            [
                _object_friction_from_cfg(scene_object)
                for scene_object in spec["scene_objects"]
            ]
            + [
                float(state.metadata["object_friction"])
                for state in support_states.values()
            ],
            dtype=np.float32,
        ),
        "scene_active_object_index": scene_active_object_index,
        "scene_canonical_object_id": np.array(canonical_object_id, dtype=object),
        "scene_table_height": np.array(scene_table_height, dtype=np.float32),
        "scene_table_margin": np.array(scene_table_margin, dtype=np.float32),
        "scene_table_size": _scene_table_size(spec),
        "scene_uses_block": np.array(False, dtype=bool),
        "scene_object_support_z_offsets": np.asarray(
            [scene_z_offsets[object_id] for object_id in object_ids]
            + [0.0] * len(support_states),
            dtype=np.float32,
        ),
    }
    support_block_ids, support_block_pos, support_block_size = _scene_support_blocks(
        spec
    )
    if support_block_ids.shape[0] > 0:
        scene_payload.update(
            {
                "scene_support_block_ids": support_block_ids,
                "scene_support_block_pos": support_block_pos,
                "scene_support_block_size": support_block_size,
            }
        )

    return (
        sequence_arrays,
        phase_names,
        phase_lengths,
        canonical_template,
        scene_payload,
    )


def _append_piece(
    accumulator: dict[str, list[np.ndarray]], piece: dict[str, np.ndarray]
) -> None:
    for key, value in piece.items():
        accumulator.setdefault(key, []).append(np.asarray(value))


def _append_hand_piece(
    accumulator: dict[str, list[np.ndarray]], piece: dict[str, np.ndarray]
) -> None:
    hand_keys = {
        key
        for key in piece
        if key in {"left_joint_vals", "right_joint_vals"}
        or ("_human_" in key and "_keypoint_" in key)
    }
    required = {"left_joint_vals", "right_joint_vals"}
    missing = sorted(required - hand_keys)
    if missing:
        raise ValueError(f"Hand sequence piece is missing required arrays: {missing}.")
    existing = set(accumulator)
    if existing and hand_keys != existing:
        raise ValueError(
            "Hand sequence pieces have inconsistent human-reference arrays: "
            f"expected {sorted(existing)}, got {sorted(hand_keys)}."
        )
    _append_piece(accumulator, {key: piece[key] for key in sorted(hand_keys)})


def _concat_piecewise(
    accumulator: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    return {key: np.concatenate(values, axis=0) for key, values in accumulator.items()}


def _build_sequence(
    spec: dict, templates: dict[str, TemplateState]
) -> tuple[dict[str, np.ndarray], list[str], list[int], TemplateState | None]:
    sequence_cfg = _sequence_cfg(spec)
    primitives = sequence_cfg.get("primitives", sequence_cfg.get("phases"))
    if primitives is None:
        raise ValueError("Sequence config must define `primitives` or `phases`.")
    phase_names: list[str] = []
    phase_lengths: list[int] = []
    pieces: dict[str, list[np.ndarray]] = {}
    canonical_template: TemplateState | None = None

    for primitive in primitives:
        frames = int(primitive["frames"])
        phase_name = str(
            primitive.get("phase_name", primitive.get("name", primitive["type"]))
        )
        piece, piece_template = _primitive_piece(primitive, templates)
        canonical_template = (
            piece_template if canonical_template is None else canonical_template
        )

        _append_piece(pieces, piece)
        phase_names.append(phase_name)
        phase_lengths.append(frames)

    return _concat_piecewise(pieces), phase_names, phase_lengths, canonical_template


def _sequence_cfg(spec: dict) -> dict:
    sequence_cfg = spec.get("sequence", spec.get("trajectory"))
    if sequence_cfg is None:
        raise ValueError("Spec must define `sequence` or `trajectory`.")
    return sequence_cfg


def _sequence_id(spec: dict) -> str:
    sequence_cfg = _sequence_cfg(spec)
    if "sequence_id" in sequence_cfg:
        return str(sequence_cfg["sequence_id"])
    if "id" in sequence_cfg:
        return str(sequence_cfg["id"])
    raise ValueError("Sequence config must define `sequence_id` or `id`.")


def _payload_active_sides(
    spec: dict,
    canonical_template: TemplateState,
) -> tuple[str, ...]:
    raw = spec.get("active_sides", canonical_template.active_sides)
    if isinstance(raw, str):
        raise ValueError(
            "Expected spec active_sides to be a list/tuple of 'left'/'right', "
            f"got string {raw!r}."
        )
    active_sides = tuple(str(side) for side in raw)
    if len(active_sides) == 0 or any(
        side not in ("left", "right") for side in active_sides
    ):
        raise ValueError(
            "Expected spec active_sides to contain 'left' and/or 'right', "
            f"got {raw!r}."
        )
    if canonical_template.active_side not in active_sides:
        raise ValueError(
            f"Spec active_sides {active_sides} must include canonical active side "
            f"{canonical_template.active_side!r}."
        )
    return active_sides


def _density_range_for_source_target(source_target: dict) -> tuple[float, float]:
    category = str(source_target["category"])
    if category not in OBJECT_CATEGORIES:
        raise KeyError(f"Expected object category metadata for {category!r}.")
    return OBJECT_CATEGORIES[category].density_range


def _density_range_for_spec(spec: dict) -> tuple[float, float]:
    if "source_target" in spec:
        return _density_range_for_source_target(dict(spec["source_target"]))
    if "scene_objects" in spec:
        scene_objects = spec["scene_objects"]
        if not scene_objects:
            raise ValueError("Multi-object spec defines empty `scene_objects`.")
        first_scene_object = dict(scene_objects[0])
        if "source_target" not in first_scene_object:
            raise ValueError(
                "First scene object must define `source_target` to attach the "
                "canonical object density range."
            )
        return _density_range_for_source_target(
            dict(first_scene_object["source_target"])
        )
    if "category" in spec and str(spec["category"]) in OBJECT_CATEGORIES:
        return OBJECT_CATEGORIES[str(spec["category"])].density_range
    raise ValueError(
        "Spec must define `source_target` or a known `category` to attach an "
        "object density range."
    )


def _output_motion_path(root: Path, spec: dict) -> Path:
    sequence_cfg = _sequence_cfg(spec)
    if "output_motion_path" not in sequence_cfg:
        raise ValueError("Sequence config must define `output_motion_path`.")
    return _rooted_path(root, str(sequence_cfg["output_motion_path"])).resolve()


def _exported_rl_path(root: Path, spec: dict) -> Path:
    sequence_cfg = _sequence_cfg(spec)
    if "exported_rl_path" not in sequence_cfg:
        raise ValueError(
            "Sequence config must define `exported_rl_path` when --export-oakink is used "
            "without --export-output-path."
        )
    return _rooted_path(root, str(sequence_cfg["exported_rl_path"])).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a concatenative primitive motion sequence from a YAML spec."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument(
        "--retarget-hand-type",
        type=str,
        default="xhand",
        help=(
            "Robot hand hierarchy to use when resolving category retargeting "
            "motion_path='auto'. Use 'xhand' or 'inspire_hand'."
        ),
    )
    parser.add_argument(
        "--export-oakink",
        action="store_true",
        help=(
            "Also export the built sequence motion to an OakInk-style .npy payload. "
            "Uses sequence.exported_rl_path unless --export-output-path is provided."
        ),
    )
    parser.add_argument("--export-output-path", type=Path, default=None)
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help=(
            "Nominal object density passed to the RL exporter. Defaults to the "
            "midpoint of the task category's declared density range."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    spec_path = args.spec.resolve()
    spec = _load_spec(spec_path)
    retarget_hand_type = normalize_retarget_hand_type(args.retarget_hand_type)
    template_entries = _build_template_entries(root, spec, retarget_hand_type)
    if "scene_objects" in spec:
        (
            sequence_arrays,
            phase_names,
            phase_lengths,
            canonical_template,
            scene_payload,
        ) = _build_multi_object_sequence(root, spec, template_entries)
    else:
        templates = {
            template_id: entry.state for template_id, entry in template_entries.items()
        }
        sequence_arrays, phase_names, phase_lengths, canonical_template = (
            _build_sequence(spec, templates)
        )
        scene_payload = _build_single_object_scene_payload(
            spec,
            canonical_template,
            sequence_arrays,
        )
    if canonical_template is None:
        raise ValueError("Sequence must contain at least one primitive.")

    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else _output_motion_path(root, spec)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keypoint_payload = {}
    if canonical_template.human_skeleton_keypoint_names is not None:
        keypoint_payload["human_skeleton_keypoint_names"] = np.asarray(
            canonical_template.human_skeleton_keypoint_names,
            dtype=object,
        )
    if canonical_template.human_bps_keypoint_names is not None:
        keypoint_payload["human_bps_keypoint_names"] = np.asarray(
            canonical_template.human_bps_keypoint_names,
            dtype=object,
        )
    for key, value in sequence_arrays.items():
        if "_human_" in key and "_keypoint_" in key:
            keypoint_payload[key] = value
    density_range = _density_range_for_spec(spec)
    metadata = {
        "sequence_spec_path": str(spec_path),
        "experiment_id": str(spec["experiment_id"]),
        "sequence_id": _sequence_id(spec),
        "category": str(spec.get("category", "")),
        "object_static": _spec_object_static(spec),
        "object_density_range": density_range,
        "is_multi_object_sequence": "scene_objects" in spec,
        **{
            key: value
            for key, value in canonical_template.metadata.items()
            if key
            in {
                "oakink_demo_path",
                "oakink_target_demo_path",
                "oakink_target_obj_path",
                "oakink_requested_target_obj_path",
                "oakink_target_transform_info",
                "oakink_target_scale",
                "oakink_run_dir",
                "oakink_hand_side",
                "oakink_start_timestep",
                "oakink_end_timestep",
                "oakink_skip",
                "traj_robot_model",
                "target_bps_sampling_link_names",
                "retarget_target_transform_path",
                "retarget_target_transform_quat_wxyz",
                "retarget_target_transform_translation",
                "retarget_target_visual_center",
                "retarget_target_pose_frame",
                "retarget_target_transform_applied",
                "mixed_object_bimanual_template",
            }
        },
    }
    hand_config_source_by_side = _hand_config_source_by_side_for_payload(
        spec,
        canonical_template,
    )
    if hand_config_source_by_side:
        metadata["hand_config_source_by_side"] = hand_config_source_by_side

    payload = build_motion_payload(
        active_side=canonical_template.active_side,
        active_sides=_payload_active_sides(spec, canonical_template),
        robot_model=canonical_template.robot_model,
        left_joint_names=canonical_template.left_joint_names,
        right_joint_names=canonical_template.right_joint_names,
        object_name=canonical_template.object_name,
        object_urdf_path=canonical_template.object_urdf_path,
        object_arti_names=canonical_template.object_arti_names,
        left_joint_vals=sequence_arrays["left_joint_vals"],
        right_joint_vals=sequence_arrays["right_joint_vals"],
        asset_pos=sequence_arrays["asset_pos"],
        asset_quat=sequence_arrays["asset_quat"],
        asset_arti_vals=sequence_arrays["asset_arti_vals"],
        phase_names=phase_names,
        phase_lengths=phase_lengths,
        metadata=metadata,
        scene_payload=scene_payload,
        extra_payload=keypoint_payload,
    )
    np.savez(output_path, **payload)
    print(f"Saved sequence motion to {output_path}")

    if args.export_oakink:
        from dexlink.scripts.retargeting.export_oakink_sequence_data import build_payload

        export_density = (
            float(args.density)
            if args.density is not None
            else float(sum(density_range) / 2.0)
        )
        if export_density <= 0.0:
            raise ValueError(f"Expected positive object density, got {export_density}.")

        export_output_path = (
            args.export_output_path.resolve()
            if args.export_output_path is not None
            else _exported_rl_path(root, spec)
        )
        export_args = argparse.Namespace(
            motion_npy_path=output_path,
            output_path=export_output_path,
            density=export_density,
            robot_model=None,
            processor_run_dir=output_path.parent / "oakink_processor_run",
        )
        export_payload = build_payload(export_args)
        export_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(export_output_path, export_payload, allow_pickle=True)
        print(f"Saved OakInk sequence payload to {export_output_path}")

    if args.write_manifest:
        manifest_path = output_path.with_suffix(".manifest.json")
        manifest = {
            "spec_path": str(spec_path),
            "output_motion_path": str(output_path),
            "phase_names": phase_names,
            "phase_lengths": phase_lengths,
            "templates": [str(key) for key in template_entries.keys()],
            "scene_objects": [
                str(scene_object["id"])
                for scene_object in spec.get("scene_objects", [])
            ],
            "object_density_range": list(density_range),
            "export_density": (
                float(args.density)
                if args.density is not None
                else float(sum(density_range) / 2.0)
            ),
        }
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved sequence manifest to {manifest_path}")


if __name__ == "__main__":
    main()
