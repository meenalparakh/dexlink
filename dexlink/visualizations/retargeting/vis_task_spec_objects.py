#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.scripts.retargeting.build_sequence_from_spec import (
    _normalize_retarget_hand_type,
    _output_motion_path,
    _source_target_articulated_urdf_path,
)
from dexlink.src.retargeting.optimization.target_transform_utils import (
    apply_target_mesh_transform,
    load_target_transform,
)
from dexlink.utils.retargeting.path_utils import process_urdf
from dexlink.utils.retargeting.vis_server import VizServer
from dexlink.src.reference.common.exp_info import get_target_object

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class TaskSpecObjectVisArgs:
    specs: list[Path]
    spec_dir: Path | None
    spec_glob: str
    mode: str
    retarget_hand_type: str
    port: int
    mesh_type: str
    frame_stride: int
    frame_indices: list[int] | None
    phase_boundaries: bool
    sleep: float
    pause: bool
    keep_open: bool


@dataclass
class JsonObjectState:
    object_id: str
    name: str
    asset_path: Path
    asset_scale: float
    asset_transform: np.ndarray | None
    pos: np.ndarray
    quat_wxyz: np.ndarray
    joint_names: list[str]
    joint_values: np.ndarray


def _rooted_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pose_matrix(
    pos: np.ndarray | list[float],
    quat_wxyz: np.ndarray | list[float],
) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.from_quat(np.roll(quat_wxyz, -1)).as_matrix()
    pose[:3, 3] = pos
    return pose


def _state_joint_dict(state: JsonObjectState) -> dict[str, float]:
    names = list(state.joint_names)
    vals = np.asarray(state.joint_values, dtype=np.float64).reshape(-1)
    if len(names) != vals.shape[0]:
        raise ValueError(
            f"Object {state.object_id!r} has {len(names)} object joints but "
            f"{vals.shape[0]} articulation values."
        )
    return {str(name): float(value) for name, value in zip(names, vals)}


def _urdf_joint_names(path: Path) -> list[str]:
    urdf = URDF.load(str(process_urdf(path)))
    return [joint.name for joint in urdf.joints if joint.joint_type != "fixed"]


def _source_target_from_cfg(
    spec: dict[str, Any],
    cfg: dict[str, Any],
    object_id: str,
) -> dict[str, Any]:
    if "source_target" in cfg:
        return dict(cfg["source_target"])
    for scene_object in spec.get("scene_objects", []):
        if str(scene_object["id"]) == object_id and "source_target" in scene_object:
            return dict(scene_object["source_target"])
    if "source_target" in spec:
        return dict(spec["source_target"])
    raise ValueError(
        f"Cannot resolve object asset for {object_id!r}; expected source_target "
        "on the keyframe, scene object, or spec."
    )


def _scene_object_cfgs(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(scene_object["id"]): dict(scene_object)
        for scene_object in spec.get("scene_objects", [])
    }


def _default_object_id(spec: dict[str, Any]) -> str:
    scene_objects = list(spec.get("scene_objects", []))
    if scene_objects:
        return str(scene_objects[0]["id"])
    return str(spec.get("scene_object_id", spec.get("category", "asset")))


def _initial_json_state(
    spec: dict[str, Any],
    cfg: dict[str, Any],
    object_id: str,
) -> JsonObjectState:
    scene_object_cfg = _scene_object_cfgs(spec).get(object_id, {})
    if "mesh_path" in scene_object_cfg or "urdf_path" in scene_object_cfg:
        asset_key = "urdf_path" if "urdf_path" in scene_object_cfg else "mesh_path"
        asset_path = _rooted_path(scene_object_cfg[asset_key])
        asset_scale = float(scene_object_cfg.get("scale", 1.0))
        asset_transform = None
        if asset_path.suffix.lower() == ".urdf" and not np.isclose(asset_scale, 1.0):
            raise ValueError(
                f"Direct JSON visualization expects URDF scene object {object_id!r} "
                f"to use scale=1.0, got {asset_scale}."
            )
        name = object_id
    else:
        source_target = _source_target_from_cfg(spec, cfg, object_id)
        target = get_target_object(
            str(source_target["category"]),
            int(source_target["target_index"]),
        )
        if target.mesh_path is None:
            raise ValueError(
                f"Direct JSON visualization needs target.mesh_path for {object_id!r}; "
                f"target {source_target} provides only target_demo_path={target.target_demo_path!r}."
            )
        articulated_urdf_path = _source_target_articulated_urdf_path(
            REPO_ROOT, source_target
        )
        if articulated_urdf_path is None:
            asset_path = _rooted_path(target.mesh_path)
            asset_scale = float(target.scale)
            asset_transform = (
                None
                if target.transform_path is None
                else load_target_transform(_rooted_path(target.transform_path))
            )
        else:
            asset_path = articulated_urdf_path
            asset_scale = 1.0
            asset_transform = None
        name = str(target.target_id)
    if not asset_path.exists():
        raise FileNotFoundError(
            f"Object mesh/URDF not found for {object_id!r}: {asset_path}"
        )
    joint_names = (
        _urdf_joint_names(asset_path) if asset_path.suffix.lower() == ".urdf" else []
    )
    pos = np.zeros(3, dtype=np.float64)
    quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    pose_cfg = scene_object_cfg.get("template_pose")
    if pose_cfg is not None:
        pose_cfg = dict(pose_cfg)
        if "object_pos" in pose_cfg:
            pos = np.asarray(pose_cfg["object_pos"], dtype=np.float64).reshape(3)
        if "object_quat_wxyz" in pose_cfg:
            quat = np.asarray(pose_cfg["object_quat_wxyz"], dtype=np.float64).reshape(4)
    return JsonObjectState(
        object_id=object_id,
        name=name,
        asset_path=asset_path,
        asset_scale=asset_scale,
        asset_transform=asset_transform,
        pos=pos,
        quat_wxyz=quat,
        joint_names=joint_names,
        joint_values=np.zeros(len(joint_names), dtype=np.float64),
    )


def _copy_state(
    state: JsonObjectState, *, object_id: str | None = None
) -> JsonObjectState:
    return JsonObjectState(
        object_id=state.object_id if object_id is None else object_id,
        name=state.name,
        asset_path=state.asset_path,
        asset_scale=state.asset_scale,
        asset_transform=(
            None if state.asset_transform is None else state.asset_transform.copy()
        ),
        pos=state.pos.copy(),
        quat_wxyz=state.quat_wxyz.copy(),
        joint_names=list(state.joint_names),
        joint_values=state.joint_values.copy(),
    )


def _state_z_extent(state: JsonObjectState) -> tuple[float, float]:
    object_pose = _pose_matrix(state.pos, state.quat_wxyz)
    suffix = state.asset_path.suffix.lower()
    if suffix == ".urdf":
        urdf = URDF.load(str(process_urdf(state.asset_path)))
        fk = urdf.visual_trimesh_fk(_state_joint_dict(state))
        if len(fk) == 0:
            fk = urdf.collision_trimesh_fk(_state_joint_dict(state))
        if len(fk) == 0:
            raise ValueError(
                f"Object {state.object_id!r} has no visual or collision meshes."
            )
        mins = []
        maxs = []
        for mesh, link_pose in fk.items():
            verts = np.asarray(mesh.vertices, dtype=np.float64)
            verts_h = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
            posed = (object_pose @ link_pose @ verts_h.T).T[:, :3]
            mins.append(float(posed[:, 2].min()))
            maxs.append(float(posed[:, 2].max()))
        return min(mins), max(maxs)
    mesh = trimesh.load(state.asset_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"Expected Trimesh for {state.asset_path}, got {type(mesh).__name__}."
        )
    mesh = apply_target_mesh_transform(
        mesh,
        transform=state.asset_transform,
        scale=state.asset_scale,
    )
    mesh.apply_transform(object_pose)
    return float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])


def _scene_object_joint_dict(
    motion: dict[str, np.ndarray],
    *,
    frame_idx: int,
    object_idx: int,
) -> dict[str, float]:
    names = [str(name) for name in list(motion["scene_object_arti_names"][object_idx])]
    valid_mask = np.asarray(
        motion["scene_object_arti_valid_mask"][object_idx], dtype=bool
    )
    vals = np.asarray(
        motion["scene_object_arti_vals"][frame_idx, object_idx], dtype=np.float64
    )
    if valid_mask.ndim != 1 or vals.ndim != 1 or valid_mask.shape != vals.shape:
        raise ValueError(
            "Expected scene object articulation mask/value vectors to have matching "
            f"shape, got {valid_mask.shape} and {vals.shape}."
        )
    if int(valid_mask.sum()) != len(names):
        raise ValueError(
            "Expected scene_object_arti_names length to match the valid "
            f"articulation count for object index {object_idx}, got {len(names)} "
            f"names and {int(valid_mask.sum())} valid values."
        )
    return {
        name: float(value)
        for name, value in zip(names, vals[valid_mask].reshape(-1).tolist())
    }


def _support_pose_cfg(scene_object: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pose_cfg = dict(
        scene_object.get(
            "template_pose",
            {
                "object_pos": scene_object.get("pos", [0.0, 0.0, 0.0]),
                "object_quat_wxyz": scene_object.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
            },
        )
    )
    pos = np.asarray(
        pose_cfg.get("object_pos", pose_cfg.get("pos")), dtype=np.float64
    ).reshape(3)
    quat = np.asarray(
        pose_cfg.get(
            "object_quat_wxyz",
            pose_cfg.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
        ),
        dtype=np.float64,
    ).reshape(4)
    return pos, quat


def _render_support_objects(vis: VizServer, spec: dict[str, Any]) -> None:
    for scene_object in spec.get("support_scene_objects", []):
        object_id = str(scene_object["id"])
        if "mesh_path" not in scene_object:
            raise ValueError(
                f"support_scene_objects entry {object_id!r} must define mesh_path."
            )
        pos, quat = _support_pose_cfg(dict(scene_object))
        vis.view_mesh(
            name=f"support/{object_id}",
            obj_path=str(_rooted_path(scene_object["mesh_path"])),
            pose=_pose_matrix(pos, quat),
            color=0x888888,
            opacity=0.55,
        )


def _render_json_states(
    vis: VizServer,
    *,
    states: dict[str, JsonObjectState],
    urdf_cache: dict[str, URDF],
    mesh_type: str,
) -> None:
    for object_idx, (object_id, state) in enumerate(states.items()):
        color = 0xD77A22 if object_idx == 0 else 0x4B8BBE
        pose = _pose_matrix(state.pos, state.quat_wxyz)
        if state.asset_path.suffix.lower() == ".urdf":
            urdf_path = str(process_urdf(state.asset_path))
            if urdf_path not in urdf_cache:
                urdf_cache[urdf_path] = URDF.load(urdf_path)
            vis.view_robot(
                urdf_inst=urdf_cache[urdf_path],
                joint_names=list(state.joint_names),
                joint_vals=np.asarray(state.joint_values, dtype=np.float64),
                joint_dict=_state_joint_dict(state),
                name=f"object/{object_id}",
                color=color,
                alpha=0.8,
                mesh_type=mesh_type,
                base_pose=pose,
            )
        else:
            mesh = trimesh.load(state.asset_path, force="mesh", process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(
                    f"Expected Trimesh for {state.asset_path}, got {type(mesh).__name__}."
                )
            mesh = apply_target_mesh_transform(
                mesh,
                transform=state.asset_transform,
                scale=state.asset_scale,
            )
            vis.view_trimesh(
                name=f"object/{object_id}",
                mesh=mesh,
                pose=pose,
                color=color,
                opacity=0.8,
            )


def _render_keyframe(
    vis: VizServer,
    *,
    spec: dict[str, Any],
    states: dict[str, JsonObjectState],
    urdf_cache: dict[str, URDF],
    mesh_type: str,
    label: str,
) -> None:
    vis.clear()
    vis.view_ground("z0_plane", zval=0.0)
    _render_support_objects(vis, spec)
    _render_json_states(
        vis,
        states=states,
        urdf_cache=urdf_cache,
        mesh_type=mesh_type,
    )
    print(f"Rendered {label}: objects={list(states)}")


def _apply_json_transform(
    state: JsonObjectState,
    transform: dict[str, Any],
    templates: dict[str, JsonObjectState],
) -> JsonObjectState:
    state = _copy_state(state)
    transform_type = str(transform["type"])
    if transform_type == "set_pair_pose":
        state.pos = np.asarray(transform["object_pos"], dtype=np.float64).reshape(3)
        if "object_quat_wxyz" in transform:
            state.quat_wxyz = np.asarray(
                transform["object_quat_wxyz"], dtype=np.float64
            ).reshape(4)
        return state
    if transform_type == "translate_pair":
        state.pos = state.pos + np.asarray(
            transform["delta"], dtype=np.float64
        ).reshape(3)
        return state
    if transform_type == "rotate_pair":
        pivot = str(transform.get("pivot", "object_origin"))
        if pivot != "object_origin":
            raise ValueError(
                f"Direct JSON visualizer supports only object_origin pivot, got {pivot!r}."
            )
        axis = np.asarray(transform["axis"], dtype=np.float64).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 0.0:
            raise ValueError(f"Rotation axis must be non-zero, got {axis}.")
        delta = R.from_rotvec(
            np.deg2rad(float(transform["degrees"])) * axis / axis_norm
        )
        current = R.from_quat(np.roll(state.quat_wxyz, -1))
        state.quat_wxyz = np.roll((delta * current).as_quat(), 1)
        return state
    if transform_type == "align_pair_lowest_z":
        offset = float(transform.get("target_lowest_z_offset", 0.0))
        if "target_lowest_z" in transform:
            target_lowest_z = float(transform["target_lowest_z"])
        elif "target_lowest_z_from_template" in transform:
            ref = str(transform["target_lowest_z_from_template"])
            target_lowest_z = _state_z_extent(templates[ref])[0]
        elif "target_lowest_z_from_template_top" in transform:
            ref = str(transform["target_lowest_z_from_template_top"])
            target_lowest_z = _state_z_extent(templates[ref])[1]
        else:
            raise ValueError(
                f"align_pair_lowest_z needs a target z reference: {transform}."
            )
        lowest_z = _state_z_extent(state)[0]
        state.pos[2] += target_lowest_z + offset - lowest_z
        return state
    if transform_type == "set_articulation":
        values = np.asarray(transform["values"], dtype=np.float64).reshape(-1)
        if values.shape != state.joint_values.shape:
            raise ValueError(
                f"set_articulation for {state.object_id!r} expected values shape "
                f"{state.joint_values.shape}, got {values.shape}."
            )
        state.joint_values = values
        return state
    if transform_type == "open_hand":
        return state
    raise ValueError(f"Unsupported direct JSON transform type {transform_type!r}.")


def _visualize_keyframes(
    *,
    spec_path: Path,
    spec: dict[str, Any],
    args: TaskSpecObjectVisArgs,
    vis: VizServer,
) -> None:
    templates: dict[str, JsonObjectState] = {}
    current_states: dict[str, JsonObjectState] = {}
    for object_id, scene_object in _scene_object_cfgs(spec).items():
        if bool(scene_object.get("static", scene_object.get("fixed", False))) and (
            "mesh_path" in scene_object or "urdf_path" in scene_object
        ):
            current_states[object_id] = _initial_json_state(
                spec, scene_object, object_id
            )
    urdf_cache: dict[str, URDF] = {}
    for keyframe in spec.get("templates", spec.get("keyframes", [])):
        template_id = str(keyframe.get("template_id", keyframe.get("id")))
        object_id = str(keyframe.get("object_id", _default_object_id(spec)))
        base_template = keyframe.get("base_template_id", keyframe.get("derive_from"))
        if base_template is not None:
            state = _copy_state(templates[str(base_template)], object_id=object_id)
        elif object_id in current_states:
            state = _copy_state(current_states[object_id])
        else:
            state = _initial_json_state(spec, dict(keyframe), object_id)
        for transform in keyframe.get("transforms", []):
            state = _apply_json_transform(state, dict(transform), templates)
        templates[template_id] = state
        current_states[object_id] = state
        _render_keyframe(
            vis,
            spec=spec,
            states=current_states,
            urdf_cache=urdf_cache,
            mesh_type=args.mesh_type,
            label=f"{spec_path.name}:{template_id}",
        )
        _wait_between_frames(args)


def _load_sequence_motion(path: Path) -> dict[str, np.ndarray]:
    if path.suffix != ".npz":
        raise ValueError(f"Expected a .npz sequence motion path, got {path}.")
    with np.load(path, allow_pickle=True) as data:
        motion = {key: data[key] for key in data.files}
    if "scene_object_pos" in motion:
        required = (
            "scene_object_ids",
            "scene_object_names",
            "scene_object_urdf_paths",
            "scene_object_pos",
            "scene_object_quat",
            "scene_object_arti_vals",
            "scene_object_arti_valid_mask",
            "scene_object_arti_names",
        )
    else:
        required = (
            "traj_object_urdf_path",
            "asset_pos",
            "asset_quat",
            "asset_arti_vals",
            "asset_arti_names",
        )
    missing = [key for key in required if key not in motion]
    if missing:
        raise ValueError(
            f"{path} is missing required object motion keys {missing}. "
            f"Available keys: {sorted(motion)}"
        )
    return motion


def _sequence_frame_indices(
    motion: dict[str, np.ndarray],
    args: TaskSpecObjectVisArgs,
) -> list[int]:
    n_frames = int(
        np.asarray(
            motion["scene_object_pos"]
            if "scene_object_pos" in motion
            else motion["asset_pos"]
        ).shape[0]
    )
    if args.frame_indices is not None:
        indices = args.frame_indices
    elif args.phase_boundaries and "phase_lengths" in motion:
        lengths = np.asarray(motion["phase_lengths"], dtype=np.int64).reshape(-1)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        ends = np.cumsum(lengths) - 1
        indices = sorted(set(starts.tolist() + ends.tolist()))
    else:
        indices = list(range(0, n_frames, args.frame_stride))
        if indices[-1] != n_frames - 1:
            indices.append(n_frames - 1)
    invalid = [idx for idx in indices if idx < 0 or idx >= n_frames]
    if invalid:
        raise ValueError(
            f"Frame indices {invalid} are outside valid range [0, {n_frames - 1}]."
        )
    return indices


def _render_sequence_frame(
    vis: VizServer,
    *,
    spec: dict[str, Any],
    motion: dict[str, np.ndarray],
    frame_idx: int,
    mesh_type: str,
    urdf_cache: dict[str, URDF],
) -> None:
    vis.clear()
    vis.view_ground("z0_plane", zval=0.0)
    rendered_object_ids: set[str] = set()
    if "scene_object_pos" in motion:
        object_ids = [str(value) for value in list(motion["scene_object_ids"])]
        rendered_object_ids.update(object_ids)
        urdf_paths = [str(value) for value in list(motion["scene_object_urdf_paths"])]
        if len(object_ids) != len(urdf_paths):
            raise ValueError(
                "scene_object_ids and scene_object_urdf_paths must have matching "
                f"lengths, got {len(object_ids)} and {len(urdf_paths)}."
            )
        for object_idx, (object_id, urdf_path) in enumerate(
            zip(object_ids, urdf_paths)
        ):
            processed_urdf = str(process_urdf(_rooted_path(urdf_path)))
            if processed_urdf not in urdf_cache:
                urdf_cache[processed_urdf] = URDF.load(processed_urdf)
            vis.view_robot(
                urdf_inst=urdf_cache[processed_urdf],
                joint_names=[
                    str(v) for v in list(motion["scene_object_arti_names"][object_idx])
                ],
                joint_vals=np.asarray(
                    motion["scene_object_arti_vals"][frame_idx, object_idx],
                    dtype=np.float64,
                ),
                joint_dict=_scene_object_joint_dict(
                    motion,
                    frame_idx=frame_idx,
                    object_idx=object_idx,
                ),
                name=f"object/{object_id}",
                color=0xD77A22,
                alpha=0.8,
                mesh_type=mesh_type,
                base_pose=_pose_matrix(
                    motion["scene_object_pos"][frame_idx, object_idx],
                    motion["scene_object_quat"][frame_idx, object_idx],
                ),
            )
    else:
        processed_urdf = str(
            process_urdf(_rooted_path(str(motion["traj_object_urdf_path"].item())))
        )
        if processed_urdf not in urdf_cache:
            urdf_cache[processed_urdf] = URDF.load(processed_urdf)
        vis.view_robot(
            urdf_inst=urdf_cache[processed_urdf],
            joint_names=[str(v) for v in list(motion["asset_arti_names"])],
            joint_vals=np.asarray(
                motion["asset_arti_vals"][frame_idx], dtype=np.float64
            ),
            name="object/asset",
            color=0xD77A22,
            alpha=0.8,
            mesh_type=mesh_type,
            base_pose=_pose_matrix(
                motion["asset_pos"][frame_idx], motion["asset_quat"][frame_idx]
            ),
        )
    _render_support_objects(
        vis,
        {
            **spec,
            "support_scene_objects": [
                support_object
                for support_object in spec.get("support_scene_objects", [])
                if str(support_object["id"]) not in rendered_object_ids
            ],
        },
    )
    print(f"Rendered sequence frame {frame_idx}")


def _visualize_sequence_frames(
    *,
    spec_path: Path,
    spec: dict[str, Any],
    args: TaskSpecObjectVisArgs,
    vis: VizServer,
) -> None:
    motion_path = _output_motion_path(REPO_ROOT, spec)
    if not motion_path.exists():
        raise FileNotFoundError(
            f"{spec_path}: sequence motion does not exist at {motion_path}."
        )
    motion = _load_sequence_motion(motion_path)
    urdf_cache: dict[str, URDF] = {}
    frame_indices = _sequence_frame_indices(motion, args)
    print(f"Loaded {motion_path}; rendering frames {frame_indices}")
    for frame_idx in frame_indices:
        _render_sequence_frame(
            vis,
            spec=spec,
            motion=motion,
            frame_idx=frame_idx,
            mesh_type=args.mesh_type,
            urdf_cache=urdf_cache,
        )
        _wait_between_frames(args)


def _wait_between_frames(args: TaskSpecObjectVisArgs) -> None:
    if args.pause:
        input("Press Enter for next object configuration...")
    elif args.sleep > 0.0:
        time.sleep(args.sleep)


def _spec_paths(args: TaskSpecObjectVisArgs) -> list[Path]:
    paths = [path.resolve() for path in args.specs]
    if args.spec_dir is not None:
        paths.extend(sorted(args.spec_dir.resolve().glob(args.spec_glob)))
    if not paths:
        raise ValueError("Pass --spec or --spec-dir to select task specs.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Spec files not found: {missing}")
    return paths


def _parse_frame_indices(raw: str | None) -> list[int] | None:
    if raw in (None, ""):
        return None
    return [int(part) for part in raw.split(",")]


def _parse_args() -> TaskSpecObjectVisArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize object-only configurations from generated trajectory specs "
            "or their built sequence_motion.npz files."
        )
    )
    parser.add_argument("--spec", type=Path, action="append", default=[])
    parser.add_argument("--spec-dir", type=Path, default=None)
    parser.add_argument("--spec-glob", type=str, default="v2_03_*.json")
    parser.add_argument(
        "--mode",
        choices=("auto", "keyframes", "sequence"),
        default="auto",
        help=(
            "auto/keyframes read only the JSON spec and do not load retargeted "
            "hands or built sequence payloads. sequence reads trajectory.output_motion_path."
        ),
    )
    parser.add_argument(
        "--retarget-hand-type",
        type=str,
        default="xhand",
        choices=("xhand", "inspire", "inspire_hand"),
    )
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument(
        "--mesh-type",
        type=str,
        default="visual",
        choices=("visual", "collision"),
    )
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument(
        "--frame-indices",
        type=str,
        default=None,
        help="Comma-separated frame indices for sequence mode.",
    )
    parser.add_argument(
        "--phase-boundaries",
        action="store_true",
        help="In sequence mode, render start/end of each trajectory phase.",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--no-keep-open", action="store_true")
    args = parser.parse_args()
    if int(args.frame_stride) <= 0:
        raise ValueError(f"--frame-stride must be positive, got {args.frame_stride}.")
    return TaskSpecObjectVisArgs(
        specs=list(args.spec),
        spec_dir=args.spec_dir,
        spec_glob=args.spec_glob,
        mode=args.mode,
        retarget_hand_type=_normalize_retarget_hand_type(args.retarget_hand_type),
        port=int(args.port),
        mesh_type=args.mesh_type,
        frame_stride=int(args.frame_stride),
        frame_indices=_parse_frame_indices(args.frame_indices),
        phase_boundaries=bool(args.phase_boundaries),
        sleep=float(args.sleep),
        pause=not bool(args.no_pause),
        keep_open=not bool(args.no_keep_open),
    )


def main() -> None:
    args = _parse_args()
    vis = VizServer(port_vis=args.port)
    print(f"MeshCat ZMQ URL: {vis.zmq_url}")
    for spec_path in _spec_paths(args):
        spec = _load_spec(spec_path)
        mode = args.mode
        if mode == "auto":
            mode = "keyframes"
        print(f"Visualizing {spec_path} in {mode} mode.")
        if mode == "sequence":
            _visualize_sequence_frames(
                spec_path=spec_path,
                spec=spec,
                args=args,
                vis=vis,
            )
        elif mode == "keyframes":
            _visualize_keyframes(
                spec_path=spec_path,
                spec=spec,
                args=args,
                vis=vis,
            )
        else:
            raise ValueError(f"Unsupported mode {mode!r}.")
    if args.keep_open:
        input("Object configurations loaded in MeshCat. Press Enter to exit...")


if __name__ == "__main__":
    main()
