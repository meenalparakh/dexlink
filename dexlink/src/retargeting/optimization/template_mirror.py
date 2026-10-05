from __future__ import annotations

from dataclasses import replace

import numpy as np

from dexlink.src.retargeting.optimization.sequence_primitives import (
    TemplateState,
    _active_wrist_pose,
    _apply_rigid_transform_to_active_hand,
    _object_base_pose,
)


def mirror_joint_name(joint_name: str) -> str:
    replacements = (
        ("R_", "L_"),
        ("L_", "R_"),
        ("right_", "left_"),
        ("left_", "right_"),
    )
    for source_prefix, target_prefix in replacements:
        if joint_name.startswith(source_prefix):
            return target_prefix + joint_name[len(source_prefix) :]
    return joint_name


def set_template_active_side(
    state: TemplateState,
    *,
    active_side: str,
    template_id: str | None = None,
) -> TemplateState:
    if active_side not in ("left", "right"):
        raise ValueError(
            f"Expected active_side 'left' or 'right', got {active_side!r}."
        )
    if active_side not in state.active_sides:
        raise ValueError(
            f"Cannot set template {state.template_id!r} active_side to {active_side!r}; "
            f"available active_sides are {state.active_sides}."
        )
    return replace(
        state,
        template_id=state.template_id if template_id is None else template_id,
        active_side=active_side,
    )


def _reflection_matrix(axis: str) -> np.ndarray:
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_index:
        raise ValueError(f"Expected mirror axis 'x', 'y', or 'z', got {axis!r}.")
    reflection = np.eye(3, dtype=np.float64)
    reflection[axis_to_index[axis], axis_to_index[axis]] = -1.0
    return reflection


def _mirrored_joint_values(
    *,
    source_names: list[str],
    source_vals: np.ndarray,
    target_names: list[str],
) -> np.ndarray:
    source_vals = np.asarray(source_vals, dtype=np.float32).reshape(-1)
    if source_vals.shape[0] != len(source_names):
        raise ValueError(
            f"Expected {len(source_names)} source joint values, got {source_vals.shape[0]}."
        )
    target_index_by_name = {name: idx for idx, name in enumerate(target_names)}
    target_vals = np.empty(len(target_names), dtype=np.float32)
    assigned: set[str] = set()
    for source_name, value in zip(source_names, source_vals):
        target_name = mirror_joint_name(source_name)
        if target_name not in target_index_by_name:
            raise ValueError(
                f"Mirrored joint {target_name!r} from {source_name!r} is missing "
                f"from target joints {target_names}."
            )
        if target_name in assigned:
            raise ValueError(f"Mirrored joint name {target_name!r} was assigned twice.")
        target_vals[target_index_by_name[target_name]] = float(value)
        assigned.add(target_name)
    missing = sorted(set(target_names) - assigned)
    if missing:
        raise ValueError(f"Source joints did not mirror onto target joints {missing}.")
    return target_vals


def _mirror_points_in_frame(
    points: np.ndarray,
    *,
    frame: np.ndarray,
    reflection: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected keypoint points shape (K, 3), got {points.shape}.")
    frame = np.asarray(frame, dtype=np.float64)
    if frame.shape != (4, 4):
        raise ValueError(f"Expected mirror frame shape (4, 4), got {frame.shape}.")
    local = (points - frame[:3, 3]) @ frame[:3, :3]
    mirrored = local @ reflection.T
    return (mirrored @ frame[:3, :3].T + frame[:3, 3]).astype(np.float32)


def _mirror_pose_in_frame(
    pose: np.ndarray,
    *,
    frame: np.ndarray,
    reflection: np.ndarray,
) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected pose shape (4, 4), got {pose.shape}.")
    frame = np.asarray(frame, dtype=np.float64)
    if frame.shape != (4, 4):
        raise ValueError(f"Expected mirror frame shape (4, 4), got {frame.shape}.")

    local_pos = frame[:3, :3].T @ (pose[:3, 3] - frame[:3, 3])
    local_rot = frame[:3, :3].T @ pose[:3, :3]
    mirrored_pose = np.eye(4, dtype=np.float64)
    mirrored_pose[:3, 3] = frame[:3, 3] + frame[:3, :3] @ (reflection @ local_pos)
    mirrored_pose[:3, :3] = frame[:3, :3] @ reflection @ local_rot @ reflection
    return mirrored_pose


def mirror_template_state(
    state: TemplateState,
    *,
    template_id: str,
    source_side: str,
    target_side: str,
    mirror_axis: str = "x",
    mirror_frame: str = "object",
) -> TemplateState:
    if source_side not in ("left", "right") or target_side not in ("left", "right"):
        raise ValueError(
            "Expected source_side and target_side to be 'left' or 'right', "
            f"got source_side={source_side!r}, target_side={target_side!r}."
        )
    if source_side == target_side:
        raise ValueError(
            f"source_side and target_side must differ, got {source_side!r}."
        )
    if source_side not in state.active_sides:
        raise ValueError(
            f"Cannot mirror source side {source_side!r} from template "
            f"{state.template_id!r}; active_sides are {state.active_sides}."
        )
    if mirror_frame not in ("object", "world"):
        raise ValueError(
            f"Expected mirror_frame 'object' or 'world', got {mirror_frame!r}."
        )

    reflection = _reflection_matrix(mirror_axis)
    frame = _object_base_pose(state) if mirror_frame == "object" else np.eye(4)
    source_state = replace(state, active_side=source_side)
    source_names = getattr(source_state, f"{source_side}_joint_names")
    source_vals = getattr(source_state, f"{source_side}_joint_vals")
    target_names = getattr(source_state, f"{target_side}_joint_names")
    target_vals = _mirrored_joint_values(
        source_names=source_names,
        source_vals=source_vals,
        target_names=target_names,
    )

    metadata = dict(state.metadata)
    source_by_side = dict(metadata.get("hand_config_source_by_side", {}))
    source_by_side[target_side] = target_side
    metadata["hand_config_source_by_side"] = source_by_side
    metadata["active_sides"] = (target_side,)
    metadata["mirrored_from_template_id"] = state.template_id
    metadata["mirrored_source_side"] = source_side
    metadata["mirror_axis"] = mirror_axis
    metadata["mirror_frame"] = mirror_frame

    keypoint_updates = {}
    for kind in ("skeleton", "bps"):
        source_attr = f"{source_side}_human_{kind}_keypoint_pos"
        target_attr = f"{target_side}_human_{kind}_keypoint_pos"
        source_points = getattr(source_state, source_attr)
        if source_points is None:
            keypoint_updates[target_attr] = None
            keypoint_updates[f"{target_side}_human_{kind}_keypoint_valid"] = False
        else:
            keypoint_updates[target_attr] = _mirror_points_in_frame(
                source_points,
                frame=frame,
                reflection=reflection,
            )
            keypoint_updates[f"{target_side}_human_{kind}_keypoint_valid"] = getattr(
                source_state,
                f"{source_side}_human_{kind}_keypoint_valid",
            )

    target_state = replace(
        source_state,
        template_id=template_id,
        active_side=target_side,
        metadata=metadata,
        **{f"{target_side}_joint_vals": target_vals},
        **keypoint_updates,
    )
    source_wrist_pose = _active_wrist_pose(source_state, source_vals)
    target_wrist_pose = _mirror_pose_in_frame(
        source_wrist_pose,
        frame=frame,
        reflection=reflection,
    )
    initial_target_wrist_pose = _active_wrist_pose(target_state, target_vals)
    target_delta = target_wrist_pose @ np.linalg.inv(initial_target_wrist_pose)
    solved_target_vals = _apply_rigid_transform_to_active_hand(
        target_state, target_delta
    )
    return replace(
        target_state,
        **{f"{target_side}_joint_vals": solved_target_vals.astype(np.float32)},
    )
