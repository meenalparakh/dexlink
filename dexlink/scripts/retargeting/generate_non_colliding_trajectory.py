#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from urdfpy import URDF

from dexlink.src.retargeting.data.hand_cfgs.franka_xhand import XHAND_CFGs as FRANKA_XHAND_CFGs
from dexlink.src.retargeting.data.hand_cfgs.inspire import INSPIRE_CFGs
from dexlink.src.retargeting.data.hand_cfgs.xhand import XHAND_CFGs
from dexlink.utils.retargeting.path_utils import process_urdf

ROOT_JOINT_TOKENS = (
    "forearm_tx_link_joint",
    "forearm_ty_link_joint",
    "forearm_tz_link_joint",
    "forearm_roll_link_joint",
    "forearm_pitch_link_joint",
    "forearm_yaw_link_joint",
)

DEXLINK_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RefinementConfig:
    input_path: Path
    output_path: Path
    hand: str
    line_search_samples: int
    line_search_passes: int
    joint_radius: float
    root_translation_radius: float
    root_rotation_radius: float
    collision_threshold: float
    include_target_object: bool
    include_scene_objects: bool
    include_inactive_scene_objects: bool
    refine_root_joints: bool
    fail_on_unresolved: bool


@dataclass(frozen=True)
class HandModel:
    side: str
    urdf: URDF
    joint_names: list[str]
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    movable_indices: np.ndarray


@dataclass(frozen=True)
class RefinementStats:
    frames: int
    changed: int
    initially_colliding: int
    unresolved: int
    mean_l1_delta: float
    max_l1_delta: float


def _as_payload(raw: object, input_path: Path) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, np.ndarray) and raw.shape == ():
        payload = raw.item()
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Expected {input_path} to contain an exported payload dict.")


def _load_payload(input_path: Path) -> dict:
    raw = np.load(input_path, allow_pickle=True)
    payload = _as_payload(raw, input_path)
    if str(payload.get("demo_kind", "sequence")).lower() != "sequence":
        raise ValueError(
            f"Expected exported sequence payload, got demo_kind={payload.get('demo_kind')!r}."
        )
    return dict(payload)


def _payload_bool(payload: dict, key: str) -> bool:
    if key not in payload:
        raise ValueError(f"Expected exported payload to contain {key!r}.")
    value = payload[key]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                f"Expected {key!r} to contain one bool, got {value.shape}."
            )
        value = value.reshape(-1)[0]
    return bool(value)


def _normalize_robot_model(robot_model: object) -> str:
    if isinstance(robot_model, np.ndarray):
        if robot_model.shape != ():
            raise ValueError(
                f"Expected scalar robot_model metadata, got shape {robot_model.shape}."
            )
        robot_model = robot_model.item()
    aliases = {
        "inspire": "inspire",
        "inspire_hand": "inspire",
        "xhand": "xhand",
        "franka": "franka_xhand",
        "franka_xhand": "franka_xhand",
        "franka-xhand": "franka_xhand",
        "panda_xhand": "franka_xhand",
    }
    value = str(robot_model).lower()
    if value not in aliases:
        raise ValueError(
            f"Unsupported robot_model {robot_model!r}; expected one of {sorted(aliases)}."
        )
    return aliases[value]


def _robot_urdf_path(robot_model: str, side: str) -> Path:
    if side not in ("left", "right"):
        raise ValueError(f"Expected hand side 'left' or 'right', got {side!r}.")
    if robot_model == "inspire":
        return Path(INSPIRE_CFGs[side]["urdf_path"])
    if robot_model == "xhand":
        return Path(XHAND_CFGs[side]["urdf_path"])
    if robot_model == "franka_xhand":
        return Path(FRANKA_XHAND_CFGs[side]["urdf_path"])
    raise ValueError(f"Unsupported robot_model {robot_model!r}.")


def _joint_limits(urdf: URDF) -> tuple[list[str], np.ndarray, np.ndarray]:
    joints = [joint for joint in urdf.joints if joint.joint_type != "fixed"]
    joint_names = [str(joint.name) for joint in joints]
    lower_limits = np.asarray([joint.limit.lower for joint in joints], dtype=np.float64)
    upper_limits = np.asarray([joint.limit.upper for joint in joints], dtype=np.float64)
    if np.any(~np.isfinite(lower_limits)) or np.any(~np.isfinite(upper_limits)):
        bad = [
            name
            for name, lo, hi in zip(joint_names, lower_limits, upper_limits)
            if not np.isfinite(lo) or not np.isfinite(hi)
        ]
        raise ValueError(f"Expected finite URDF limits for all joints; invalid {bad}.")
    return joint_names, lower_limits, upper_limits


def _root_joint_mask(joint_names: list[str]) -> np.ndarray:
    return np.asarray(
        [
            any(str(name).endswith(token) for token in ROOT_JOINT_TOKENS)
            for name in joint_names
        ],
        dtype=bool,
    )


def _build_hand_model(
    *,
    side: str,
    robot_model: str,
    payload_joint_names: list[str],
    refine_root_joints: bool,
) -> HandModel:
    urdf = URDF.load(str(process_urdf(_robot_urdf_path(robot_model, side))))
    joint_names, lower_limits, upper_limits = _joint_limits(urdf)
    if payload_joint_names != joint_names:
        raise ValueError(
            f"Expected {side}_jnames to exactly match {robot_model} URDF joints.\n"
            f"payload={payload_joint_names}\n"
            f"urdf={joint_names}"
        )
    movable_mask = np.ones(len(joint_names), dtype=bool)
    if not refine_root_joints:
        movable_mask &= ~_root_joint_mask(joint_names)
    movable_indices = np.flatnonzero(movable_mask)
    if movable_indices.size == 0:
        raise ValueError(
            f"No movable joints selected for {side} hand; pass --refine-root-joints "
            "or use a hand with non-root joints."
        )
    return HandModel(
        side=side,
        urdf=urdf,
        joint_names=joint_names,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        movable_indices=movable_indices,
    )


def _selected_hands(payload: dict, requested_hand: str) -> list[str]:
    if requested_hand != "auto":
        if requested_hand not in ("left", "right", "both"):
            raise ValueError(
                f"Expected --hand to be 'auto', 'left', 'right', or 'both', got {requested_hand!r}."
            )
        return ["left", "right"] if requested_hand == "both" else [requested_hand]

    selected = []
    if _payload_bool(payload, "is_active_left"):
        selected.append("left")
    if _payload_bool(payload, "is_active_right"):
        selected.append("right")
    if not selected:
        side = str(payload.get("side", "")).lower()
        if side not in ("left", "right"):
            raise ValueError(
                "No active hand flags were set and payload side metadata is not "
                f"'left' or 'right': {side!r}."
            )
        selected.append(side)
    return selected


def _joint_names_from_payload(payload: dict, side: str) -> list[str]:
    key = f"{side}_jnames"
    if key not in payload:
        raise ValueError(f"Expected exported payload to contain {key!r}.")
    return [str(name) for name in np.asarray(payload[key], dtype=object).tolist()]


def _joint_values_from_payload(payload: dict, side: str) -> np.ndarray:
    key = f"{side}_jvals"
    if key not in payload:
        raise ValueError(f"Expected exported payload to contain {key!r}.")
    values = np.asarray(payload[key], dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected {key!r} to have shape (T, J), got {values.shape}.")
    return values


def _pose_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_quat(np.roll(quat_wxyz, -1)).as_matrix()
    transform[:3, 3] = pos
    return transform


def _joint_dict(names: list[str], values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape != (len(names),):
        raise ValueError(
            f"Expected {len(names)} values for joints {names}, got {values.shape}."
        )
    return {name: float(value) for name, value in zip(names, values.tolist())}


def _object_mesh_entries(
    *,
    urdf: URDF,
    object_pose: np.ndarray,
    joint_names: list[str],
    joint_values: np.ndarray,
    prefix: str,
) -> list[tuple[str, trimesh.Trimesh, np.ndarray]]:
    cfg = _joint_dict(joint_names, joint_values) if joint_names else None
    fk = urdf.collision_trimesh_fk(cfg=cfg)
    if not fk:
        raise ValueError(f"Expected at least one collision mesh for object {prefix!r}.")
    return [
        (
            f"{prefix}_{mesh_idx}",
            mesh,
            object_pose @ np.asarray(local_pose, dtype=np.float64),
        )
        for mesh_idx, (mesh, local_pose) in enumerate(fk.items())
    ]


def _load_urdf(path_value: object, field_name: str) -> URDF:
    path_value = _unwrap_scalar(path_value, field_name)
    if path_value in (None, "", "None"):
        raise ValueError(f"Expected non-empty {field_name!r} URDF path.")
    path = _resolve_stored_path(Path(str(path_value)))
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {field_name!r} URDF to exist after localizing {path_value!r}: {path}"
        )
    return URDF.load(str(process_urdf(path)))


def _resolve_stored_path(path: Path) -> Path:
    return path if path.is_absolute() else DEXLINK_ROOT / path


def _unwrap_scalar(value: object, field_name: str) -> object:
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"Expected scalar {field_name!r}, got {value.shape}.")
        return value.item()
    return value


def _target_object_urdf(payload: dict) -> URDF:
    articulated = _unwrap_scalar(
        payload.get("target_articulated_urdf_path", None),
        "target_articulated_urdf_path",
    )
    if articulated not in (None, "", "None"):
        return _load_urdf(articulated, "target_articulated_urdf_path")
    if "target_urdf_path" not in payload:
        raise ValueError(
            "Expected payload to contain target_urdf_path when including target object."
        )
    return _load_urdf(payload["target_urdf_path"], "target_urdf_path")


def _frame_target_obstacles(
    payload: dict,
    frame_idx: int,
    target_urdf: URDF,
) -> list[tuple[str, trimesh.Trimesh, np.ndarray]]:
    required = (
        "retargeted_obj_pos",
        "retargeted_obj_quat",
        "retargeted_obj_arti",
        "retargeted_joint_names",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Target collision checking requires payload keys {missing}.")
    return _object_mesh_entries(
        urdf=target_urdf,
        object_pose=_pose_matrix(
            np.asarray(payload["retargeted_obj_pos"], dtype=np.float64)[frame_idx],
            np.asarray(payload["retargeted_obj_quat"], dtype=np.float64)[frame_idx],
        ),
        joint_names=[
            str(name)
            for name in np.asarray(payload["retargeted_joint_names"], dtype=object)
            .reshape(-1)
            .tolist()
        ],
        joint_values=np.asarray(payload["retargeted_obj_arti"], dtype=np.float64)[
            frame_idx
        ],
        prefix=f"target_object_{frame_idx}",
    )


def _scene_object_joint_names(payload: dict, object_idx: int) -> list[str]:
    names = np.asarray(payload["scene_object_arti_names"], dtype=object)[object_idx]
    valid_mask = np.asarray(payload["scene_object_arti_valid_mask"], dtype=bool)[
        object_idx
    ]
    names_list = [str(name) for name in np.asarray(names, dtype=object).tolist()]
    if int(valid_mask.sum()) != len(names_list):
        raise ValueError(
            "scene_object_arti_valid_mask true count must match "
            f"scene_object_arti_names length for object index {object_idx}; "
            f"got {int(valid_mask.sum())} and {len(names_list)}."
        )
    return names_list


def _frame_scene_obstacles(
    payload: dict,
    frame_idx: int,
    scene_urdfs: list[URDF],
    include_inactive_scene_objects: bool,
) -> list[tuple[str, trimesh.Trimesh, np.ndarray]]:
    required = (
        "scene_object_pos",
        "scene_object_quat",
        "scene_object_arti_vals",
        "scene_object_arti_valid_mask",
        "scene_object_arti_names",
        "scene_object_active_mask",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Scene collision checking requires payload keys {missing}.")

    scene_pos = np.asarray(payload["scene_object_pos"], dtype=np.float64)
    scene_quat = np.asarray(payload["scene_object_quat"], dtype=np.float64)
    scene_arti_vals = np.asarray(payload["scene_object_arti_vals"], dtype=np.float64)
    scene_valid_mask = np.asarray(payload["scene_object_arti_valid_mask"], dtype=bool)
    scene_active_mask = np.asarray(payload["scene_object_active_mask"], dtype=bool)
    if scene_pos.ndim != 3 or scene_pos.shape[2] != 3:
        raise ValueError(
            f"Expected scene_object_pos shape (T, N, 3), got {scene_pos.shape}."
        )
    if len(scene_urdfs) != scene_pos.shape[1]:
        raise ValueError(
            "Expected scene_object_urdf_paths length to match scene object count "
            f"{scene_pos.shape[1]}, got {len(scene_urdfs)}."
        )

    obstacles: list[tuple[str, trimesh.Trimesh, np.ndarray]] = []
    for object_idx, urdf in enumerate(scene_urdfs):
        if not include_inactive_scene_objects and not bool(
            scene_active_mask[frame_idx, object_idx]
        ):
            continue
        joint_names = _scene_object_joint_names(payload, object_idx)
        joint_values = scene_arti_vals[frame_idx, object_idx][
            scene_valid_mask[object_idx]
        ]
        obstacles.extend(
            _object_mesh_entries(
                urdf=urdf,
                object_pose=_pose_matrix(
                    scene_pos[frame_idx, object_idx],
                    scene_quat[frame_idx, object_idx],
                ),
                joint_names=joint_names,
                joint_values=joint_values,
                prefix=f"scene_object_{object_idx}_{frame_idx}",
            )
        )
    return obstacles


def _build_collision_manager(
    obstacles: list[tuple[str, trimesh.Trimesh, np.ndarray]],
) -> trimesh.collision.CollisionManager:
    if not obstacles:
        raise ValueError("Expected at least one object collision obstacle.")
    manager = trimesh.collision.CollisionManager()
    for name, mesh, pose in obstacles:
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(
                f"Expected obstacle {name!r} to be Trimesh, got {type(mesh)}."
            )
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(
                f"Expected obstacle {name!r} pose shape (4, 4), got {pose.shape}."
            )
        manager.add_object(name, mesh, transform=pose)
    return manager


def _collision_violation(
    hand_model: HandModel,
    q: np.ndarray,
    collision_manager: trimesh.collision.CollisionManager,
    robot_base_pose: np.ndarray,
    collision_threshold: float,
) -> float:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape != (len(hand_model.joint_names),):
        raise ValueError(
            f"Expected q shape {(len(hand_model.joint_names),)}, got {q.shape}."
        )
    robot_base_pose = np.asarray(robot_base_pose, dtype=np.float64)
    if robot_base_pose.shape != (4, 4):
        raise ValueError(
            f"Expected robot_base_pose shape (4, 4), got {robot_base_pose.shape}."
        )
    link_fk = hand_model.urdf.link_fk(cfg=_joint_dict(hand_model.joint_names, q))
    worst_violation = 0.0
    for link, pose in link_fk.items():
        mesh = link.collision_mesh
        if mesh is None:
            continue
        dist = float(
            collision_manager.min_distance_single(
                mesh,
                transform=robot_base_pose @ np.asarray(pose, dtype=np.float64),
                return_name=False,
            )
        )
        worst_violation = max(worst_violation, float(collision_threshold) - dist)
    return max(0.0, worst_violation)


def _joint_radius(name: str, cfg: RefinementConfig) -> float:
    if name.endswith(
        ("forearm_tx_link_joint", "forearm_ty_link_joint", "forearm_tz_link_joint")
    ):
        return float(cfg.root_translation_radius)
    if name.endswith(
        (
            "forearm_roll_link_joint",
            "forearm_pitch_link_joint",
            "forearm_yaw_link_joint",
        )
    ):
        return float(cfg.root_rotation_radius)
    return float(cfg.joint_radius)


def _candidate_values(
    q_reference: np.ndarray,
    joint_idx: int,
    hand_model: HandModel,
    cfg: RefinementConfig,
) -> np.ndarray:
    radius = _joint_radius(hand_model.joint_names[joint_idx], cfg)
    if radius <= 0.0:
        return np.asarray([q_reference[joint_idx]], dtype=np.float64)
    lower = max(
        float(hand_model.lower_limits[joint_idx]),
        float(q_reference[joint_idx]) - radius,
    )
    upper = min(
        float(hand_model.upper_limits[joint_idx]),
        float(q_reference[joint_idx]) + radius,
    )
    if lower > upper:
        raise ValueError(
            f"Line-search interval is empty for joint {hand_model.joint_names[joint_idx]!r}: "
            f"[{lower}, {upper}]."
        )
    values = np.concatenate(
        [
            np.linspace(lower, upper, int(cfg.line_search_samples)),
            np.asarray([q_reference[joint_idx]], dtype=np.float64),
        ]
    )
    return np.unique(values)


def _refine_frame(
    hand_model: HandModel,
    q_reference: np.ndarray,
    collision_manager: trimesh.collision.CollisionManager,
    robot_base_pose: np.ndarray,
    cfg: RefinementConfig,
) -> tuple[np.ndarray, float, float]:
    q_reference = np.asarray(q_reference, dtype=np.float64).reshape(-1)
    if q_reference.shape != (len(hand_model.joint_names),):
        raise ValueError(
            f"Expected {hand_model.side} q_reference shape "
            f"{(len(hand_model.joint_names),)}, got {q_reference.shape}."
        )

    q_current = np.clip(q_reference, hand_model.lower_limits, hand_model.upper_limits)
    initial_violation = _collision_violation(
        hand_model,
        q_current,
        collision_manager,
        robot_base_pose,
        cfg.collision_threshold,
    )
    if initial_violation <= 1e-12:
        return q_current, initial_violation, 0.0

    for _ in range(int(cfg.line_search_passes)):
        for joint_idx in hand_model.movable_indices.tolist():
            best_q = None
            best_key = None
            for joint_value in _candidate_values(
                q_reference, joint_idx, hand_model, cfg
            ):
                q_candidate = q_current.copy()
                q_candidate[joint_idx] = joint_value
                violation = _collision_violation(
                    hand_model,
                    q_candidate,
                    collision_manager,
                    robot_base_pose,
                    cfg.collision_threshold,
                )
                l1_delta = float(np.sum(np.abs(q_candidate - q_reference)))
                key = (violation > 1e-12, violation, l1_delta)
                if best_key is None or key < best_key:
                    best_key = key
                    best_q = q_candidate
            if best_q is None:
                raise RuntimeError(
                    f"No line-search candidates evaluated for joint "
                    f"{hand_model.joint_names[joint_idx]!r}."
                )
            q_current = best_q

    final_violation = _collision_violation(
        hand_model,
        q_current,
        collision_manager,
        robot_base_pose,
        cfg.collision_threshold,
    )
    return (
        q_current,
        initial_violation,
        final_violation if final_violation > 1e-12 else 0.0,
    )


def _payload_num_frames(payload: dict) -> int:
    if "retargeted_obj_pos" in payload:
        return int(np.asarray(payload["retargeted_obj_pos"]).shape[0])
    if "scene_object_pos" in payload:
        return int(np.asarray(payload["scene_object_pos"]).shape[0])
    raise ValueError(
        "Expected retargeted_obj_pos or scene_object_pos to determine frames."
    )


def _robot_base_pose(payload: dict) -> np.ndarray:
    if "traj_robot_base_pose" not in payload:
        return np.eye(4, dtype=np.float64)
    robot_base_pose = np.asarray(payload["traj_robot_base_pose"], dtype=np.float64)
    if robot_base_pose.shape != (4, 4):
        raise ValueError(
            f"Expected traj_robot_base_pose shape (4, 4), got {robot_base_pose.shape}."
        )
    return robot_base_pose


def _validate_cfg(cfg: RefinementConfig) -> None:
    if cfg.line_search_samples < 2:
        raise ValueError(
            f"Expected line_search_samples >= 2, got {cfg.line_search_samples}."
        )
    if cfg.line_search_passes < 1:
        raise ValueError(
            f"Expected line_search_passes >= 1, got {cfg.line_search_passes}."
        )
    for name, value in (
        ("joint_radius", cfg.joint_radius),
        ("root_translation_radius", cfg.root_translation_radius),
        ("root_rotation_radius", cfg.root_rotation_radius),
    ):
        if value < 0.0:
            raise ValueError(f"Expected {name} >= 0, got {value}.")
    if not cfg.include_target_object and not cfg.include_scene_objects:
        raise ValueError(
            "At least one of include_target_object or include_scene_objects must be true."
        )


def refine_payload(
    payload: dict, cfg: RefinementConfig
) -> tuple[dict, dict[str, RefinementStats]]:
    _validate_cfg(cfg)
    n_frames = _payload_num_frames(payload)
    robot_model = _normalize_robot_model(payload.get("robot_model", "xhand"))
    robot_base_pose = _robot_base_pose(payload)
    selected_hands = _selected_hands(payload, cfg.hand)
    target_urdf = _target_object_urdf(payload) if cfg.include_target_object else None
    scene_urdfs: list[URDF] = []
    if cfg.include_scene_objects:
        if "scene_object_urdf_paths" not in payload:
            raise ValueError(
                "Payload has no scene_object_urdf_paths; disable --include-scene-objects "
                "or provide a multi-object exported payload."
            )
        scene_urdfs = [
            _load_urdf(path, "scene_object_urdf_paths")
            for path in np.asarray(
                payload["scene_object_urdf_paths"], dtype=object
            ).tolist()
        ]

    refined_payload = dict(payload)
    stats: dict[str, RefinementStats] = {}
    for side in selected_hands:
        q_values = _joint_values_from_payload(payload, side)
        joint_names = _joint_names_from_payload(payload, side)
        if q_values.shape[0] != n_frames:
            raise ValueError(
                f"Expected {side}_jvals frame count {q_values.shape[0]} to match "
                f"payload frame count {n_frames}."
            )
        hand_model = _build_hand_model(
            side=side,
            robot_model=robot_model,
            payload_joint_names=joint_names,
            refine_root_joints=cfg.refine_root_joints,
        )

        refined_values = q_values.copy()
        changed = 0
        initially_colliding = 0
        unresolved = 0
        l1_deltas: list[float] = []
        iterator = tqdm(range(n_frames), desc=f"Refining {side} hand frames")
        for frame_idx in iterator:
            obstacles: list[tuple[str, trimesh.Trimesh, np.ndarray]] = []
            if target_urdf is not None:
                obstacles.extend(
                    _frame_target_obstacles(payload, frame_idx, target_urdf)
                )
            if scene_urdfs:
                obstacles.extend(
                    _frame_scene_obstacles(
                        payload,
                        frame_idx,
                        scene_urdfs,
                        include_inactive_scene_objects=cfg.include_inactive_scene_objects,
                    )
                )
            collision_manager = _build_collision_manager(obstacles)
            q_ref = q_values[frame_idx]
            q_new, initial_violation, final_violation = _refine_frame(
                hand_model,
                q_ref,
                collision_manager,
                robot_base_pose,
                cfg,
            )
            if initial_violation > 1e-12:
                initially_colliding += 1
            if final_violation > 1e-12:
                unresolved += 1
                if cfg.fail_on_unresolved:
                    raise RuntimeError(
                        f"Could not find collision-free {side} hand frame {frame_idx}; "
                        f"initial_violation={initial_violation:.6e}, "
                        f"final_violation={final_violation:.6e}."
                    )
            l1_delta = float(np.sum(np.abs(q_new - q_ref)))
            if l1_delta > 1e-12:
                changed += 1
            l1_deltas.append(l1_delta)
            refined_values[frame_idx] = q_new
        refined_payload[f"{side}_jvals"] = refined_values.astype(np.float32)
        stats[side] = RefinementStats(
            frames=n_frames,
            changed=changed,
            initially_colliding=initially_colliding,
            unresolved=unresolved,
            mean_l1_delta=float(np.mean(l1_deltas)) if l1_deltas else 0.0,
            max_l1_delta=float(np.max(l1_deltas)) if l1_deltas else 0.0,
        )

    refined_payload["collision_refinement_input_path"] = str(cfg.input_path)
    refined_payload["collision_refinement_hand"] = cfg.hand
    refined_payload["collision_refinement_line_search_samples"] = np.array(
        cfg.line_search_samples, dtype=np.int64
    )
    refined_payload["collision_refinement_line_search_passes"] = np.array(
        cfg.line_search_passes, dtype=np.int64
    )
    refined_payload["collision_refinement_joint_radius"] = np.array(
        cfg.joint_radius, dtype=np.float32
    )
    refined_payload["collision_refinement_collision_threshold"] = np.array(
        cfg.collision_threshold, dtype=np.float32
    )
    return refined_payload, stats


def parse_args() -> RefinementConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Refine an exported OakInk sequence payload with deterministic per-frame "
            "coordinate line search against trimesh collision objects."
        )
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input motion.npy payload."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output refined motion.npy payload."
    )
    parser.add_argument(
        "--hand",
        type=str,
        default="auto",
        choices=["auto", "left", "right", "both"],
        help="Hand trajectory to refine. auto uses payload active-hand flags.",
    )
    parser.add_argument("--line-search-samples", type=int, default=21)
    parser.add_argument("--line-search-passes", type=int, default=2)
    parser.add_argument("--joint-radius", type=float, default=0.5)
    parser.add_argument("--root-translation-radius", type=float, default=0.05)
    parser.add_argument("--root-rotation-radius", type=float, default=0.5)
    parser.add_argument("--collision-threshold", type=float, default=0.002)
    parser.add_argument(
        "--include-target-object",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check hand collision against the canonical retargeted target object.",
    )
    parser.add_argument(
        "--include-scene-objects",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Check hand collision against scene_object_* payload objects.",
    )
    parser.add_argument(
        "--include-inactive-scene-objects",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When scene objects are enabled, include objects inactive in a frame.",
    )
    parser.add_argument(
        "--refine-root-joints",
        action="store_true",
        help="Allow the 6-DoF forearm/root joints to move during line search.",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Save best-effort frames even if some remain colliding.",
    )
    args = parser.parse_args()
    return RefinementConfig(
        input_path=args.input.resolve(),
        output_path=args.output.resolve(),
        hand=str(args.hand),
        line_search_samples=int(args.line_search_samples),
        line_search_passes=int(args.line_search_passes),
        joint_radius=float(args.joint_radius),
        root_translation_radius=float(args.root_translation_radius),
        root_rotation_radius=float(args.root_rotation_radius),
        collision_threshold=float(args.collision_threshold),
        include_target_object=bool(args.include_target_object),
        include_scene_objects=bool(args.include_scene_objects),
        include_inactive_scene_objects=bool(args.include_inactive_scene_objects),
        refine_root_joints=bool(args.refine_root_joints),
        fail_on_unresolved=not bool(args.allow_unresolved),
    )


def main() -> None:
    cfg = parse_args()
    payload = _load_payload(cfg.input_path)
    refined_payload, stats = refine_payload(payload, cfg)
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.output_path, refined_payload, allow_pickle=True)
    print(f"Saved refined OakInk payload to {cfg.output_path}")
    for side, side_stats in stats.items():
        print(
            f"{side}: frames={side_stats.frames}, changed={side_stats.changed}, "
            f"initially_colliding={side_stats.initially_colliding}, "
            f"unresolved={side_stats.unresolved}, "
            f"mean_l1_delta={side_stats.mean_l1_delta:.6e}, "
            f"max_l1_delta={side_stats.max_l1_delta:.6e}"
        )


if __name__ == "__main__":
    main()
