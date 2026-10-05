"""Trajectory-resolved deviations between an RL rollout and its reference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferencePolicyDeviation:
    rollout_timestep: np.ndarray
    command_timestep: np.ndarray
    phase_name: np.ndarray
    corresponding_object: np.ndarray
    object_position_delta_cm: np.ndarray
    object_rotation_delta_deg: np.ndarray
    wrist_object_position_delta_cm: np.ndarray
    wrist_object_rotation_delta_deg: np.ndarray
    finger_q_rms_delta_deg: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.rollout_timestep.shape[0])


def quaternion_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert WXYZ quaternions to rotation matrices without changing shape."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.ndim < 1 or quaternion.shape[-1] != 4:
        raise ValueError(
            f"Expected quaternion shape (..., 4), got {quaternion.shape}."
        )
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise ValueError("Expected nonzero quaternions.")
    w, x, y, z = np.moveaxis(quaternion / norm, -1, 0)
    matrix = np.empty((*quaternion.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def rotation_geodesic_deg(
    reference_rotation: np.ndarray,
    rollout_rotation: np.ndarray,
) -> np.ndarray:
    """Return the SO(3) geodesic angle in degrees."""
    reference_rotation = np.asarray(reference_rotation, dtype=np.float64)
    rollout_rotation = np.asarray(rollout_rotation, dtype=np.float64)
    if reference_rotation.shape != rollout_rotation.shape:
        raise ValueError(
            "Expected matching rotation shapes, got "
            f"{reference_rotation.shape} and {rollout_rotation.shape}."
        )
    if reference_rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation shape (..., 3, 3), got {reference_rotation.shape}."
        )
    delta = np.matmul(np.swapaxes(reference_rotation, -1, -2), rollout_rotation)
    trace = np.trace(delta, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def load_rollout_episode(
    rollout_path: Path,
    *,
    hand_side: str,
    env_index: int,
) -> dict[str, np.ndarray]:
    """Load one episode from a final-evaluation rollout chunk."""
    with np.load(rollout_path, allow_pickle=False) as archive:
        required = {
            "done",
            "command_timestep",
            "object_pos",
            "object_quat",
            f"{hand_side}_wrist_pos",
            f"{hand_side}_wrist_quat",
            f"{hand_side}_jvals",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(
                f"Rollout {rollout_path} is missing required arrays: {missing}."
            )
        done = np.asarray(archive["done"], dtype=bool)
        if done.ndim != 2:
            raise ValueError(f"Expected done shape (T, E), got {done.shape}.")
        if not 0 <= env_index < done.shape[1]:
            raise IndexError(
                f"Expected env_index in [0, {done.shape[1]}), got {env_index}."
            )
        done_indices = np.flatnonzero(done[:, env_index])
        length = int(done_indices[0] + 1) if done_indices.size else int(done.shape[0])
        arrays = {
            key: np.asarray(archive[key])[:length, env_index].copy()
            for key in required
            if key != "done"
        }
    return arrays


def load_reference_trajectory(reference_path: Path) -> dict[str, np.ndarray]:
    """Load the strict, pickle-free reference artifact written by evaluation."""
    with np.load(reference_path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def compute_reference_policy_deviation(
    rollout: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    hand_side: str,
) -> ReferencePolicyDeviation:
    """Compute per-step scene, object-relative wrist, and finger-q errors."""
    required_reference = {
        "object_ids",
        "object_static",
        "object_active_mask",
        "object_pos",
        "object_quat",
        "phase_names",
        "phase_lengths",
        f"{hand_side}_finger_joint_indices",
        f"{hand_side}_jvals",
        f"{hand_side}_wrist_pos",
        f"{hand_side}_wrist_quat",
    }
    missing = sorted(required_reference.difference(reference))
    if missing:
        raise KeyError(f"Reference trajectory is missing required arrays: {missing}.")

    command_timestep = np.asarray(rollout["command_timestep"], dtype=np.int64)
    if command_timestep.ndim != 1:
        raise ValueError(
            f"Expected command_timestep shape (T,), got {command_timestep.shape}."
        )
    num_steps = command_timestep.shape[0]
    reference_object_pos = np.asarray(reference["object_pos"], dtype=np.float64)
    reference_object_quat = np.asarray(reference["object_quat"], dtype=np.float64)
    if reference_object_pos.ndim != 3 or reference_object_pos.shape[-1] != 3:
        raise ValueError(
            "Expected reference object_pos shape (R, O, 3), got "
            f"{reference_object_pos.shape}."
        )
    expected_quat_shape = (*reference_object_pos.shape[:2], 4)
    if reference_object_quat.shape != expected_quat_shape:
        raise ValueError(
            f"Expected reference object_quat shape {expected_quat_shape}, got "
            f"{reference_object_quat.shape}."
        )
    if np.any(command_timestep < 0) or np.any(
        command_timestep >= reference_object_pos.shape[0]
    ):
        raise IndexError(
            "Command timesteps must lie inside the reference trajectory; got range "
            f"[{command_timestep.min()}, {command_timestep.max()}] for length "
            f"{reference_object_pos.shape[0]}."
        )

    object_ids = np.asarray(reference["object_ids"], dtype=str)
    object_static = np.asarray(reference["object_static"], dtype=bool)
    object_active_mask = np.asarray(reference["object_active_mask"], dtype=bool)
    num_objects = reference_object_pos.shape[1]
    if object_ids.shape != (num_objects,):
        raise ValueError(
            f"Expected object_ids shape {(num_objects,)}, got {object_ids.shape}."
        )
    if object_static.shape != (num_objects,):
        raise ValueError(
            f"Expected object_static shape {(num_objects,)}, got {object_static.shape}."
        )
    if object_active_mask.shape != reference_object_pos.shape[:2]:
        raise ValueError(
            "Expected object_active_mask shape "
            f"{reference_object_pos.shape[:2]}, got {object_active_mask.shape}."
        )
    rollout_object_pos = np.asarray(rollout["object_pos"], dtype=np.float64)
    rollout_object_quat = np.asarray(rollout["object_quat"], dtype=np.float64)
    if rollout_object_pos.shape != (num_steps, num_objects, 3):
        raise ValueError(
            "Expected rollout object_pos shape "
            f"{(num_steps, num_objects, 3)}, got {rollout_object_pos.shape}."
        )
    if rollout_object_quat.shape != (num_steps, num_objects, 4):
        raise ValueError(
            "Expected rollout object_quat shape "
            f"{(num_steps, num_objects, 4)}, got {rollout_object_quat.shape}."
        )

    moving_indices = np.flatnonzero(~object_static)
    if moving_indices.size == 0:
        raise ValueError("Expected at least one non-static object.")
    aligned_reference_pos = reference_object_pos[command_timestep]
    aligned_reference_quat = reference_object_quat[command_timestep]
    per_object_position = np.linalg.norm(
        rollout_object_pos[:, moving_indices]
        - aligned_reference_pos[:, moving_indices],
        axis=-1,
    )
    per_object_rotation = rotation_geodesic_deg(
        quaternion_to_matrix_wxyz(aligned_reference_quat[:, moving_indices]),
        quaternion_to_matrix_wxyz(rollout_object_quat[:, moving_indices]),
    )
    object_position_delta_cm = (
        100.0 * np.sqrt(np.mean(np.square(per_object_position), axis=1))
    )
    object_rotation_delta_deg = np.sqrt(
        np.mean(np.square(per_object_rotation), axis=1)
    )

    active_mask = object_active_mask[command_timestep]
    active_count = active_mask.sum(axis=1)
    if np.any(active_count > 1):
        bad_timestep = int(np.flatnonzero(active_count > 1)[0])
        raise ValueError(
            "Expected at most one corresponding object per step, got "
            f"{int(active_count[bad_timestep])} at rollout step {bad_timestep}."
        )
    has_correspondence = active_count == 1
    active_index = np.full(num_steps, -1, dtype=np.int64)
    active_index[has_correspondence] = np.argmax(
        active_mask[has_correspondence], axis=1
    )
    corresponding_object = np.full(num_steps, "none", dtype="<U128")
    corresponding_object[has_correspondence] = object_ids[
        active_index[has_correspondence]
    ]

    wrist_position_delta_cm = np.full(num_steps, np.nan, dtype=np.float64)
    wrist_rotation_delta_deg = np.full(num_steps, np.nan, dtype=np.float64)
    if np.any(has_correspondence):
        step_index = np.flatnonzero(has_correspondence)
        object_index = active_index[has_correspondence]
        rollout_object_rotation = quaternion_to_matrix_wxyz(
            rollout_object_quat[step_index, object_index]
        )
        reference_object_rotation = quaternion_to_matrix_wxyz(
            aligned_reference_quat[step_index, object_index]
        )
        rollout_wrist_pos = np.asarray(
            rollout[f"{hand_side}_wrist_pos"], dtype=np.float64
        )[has_correspondence]
        reference_wrist_pos = np.asarray(
            reference[f"{hand_side}_wrist_pos"], dtype=np.float64
        )[command_timestep[has_correspondence]]
        rollout_relative_position = np.einsum(
            "tji,tj->ti",
            rollout_object_rotation,
            rollout_wrist_pos - rollout_object_pos[step_index, object_index],
        )
        reference_relative_position = np.einsum(
            "tji,tj->ti",
            reference_object_rotation,
            reference_wrist_pos - aligned_reference_pos[step_index, object_index],
        )
        wrist_position_delta_cm[has_correspondence] = 100.0 * np.linalg.norm(
            rollout_relative_position - reference_relative_position,
            axis=-1,
        )

        rollout_wrist_rotation = quaternion_to_matrix_wxyz(
            np.asarray(rollout[f"{hand_side}_wrist_quat"], dtype=np.float64)[
                has_correspondence
            ]
        )
        reference_wrist_rotation = quaternion_to_matrix_wxyz(
            np.asarray(reference[f"{hand_side}_wrist_quat"], dtype=np.float64)[
                command_timestep[has_correspondence]
            ]
        )
        rollout_relative_rotation = np.matmul(
            np.swapaxes(rollout_object_rotation, -1, -2),
            rollout_wrist_rotation,
        )
        reference_relative_rotation = np.matmul(
            np.swapaxes(reference_object_rotation, -1, -2),
            reference_wrist_rotation,
        )
        wrist_rotation_delta_deg[has_correspondence] = rotation_geodesic_deg(
            reference_relative_rotation,
            rollout_relative_rotation,
        )

    finger_indices = np.asarray(
        reference[f"{hand_side}_finger_joint_indices"], dtype=np.int64
    )
    rollout_jvals = np.asarray(rollout[f"{hand_side}_jvals"], dtype=np.float64)
    reference_jvals = np.asarray(reference[f"{hand_side}_jvals"], dtype=np.float64)
    if rollout_jvals.ndim != 2 or rollout_jvals.shape[0] != num_steps:
        raise ValueError(
            f"Expected rollout hand q shape ({num_steps}, J), got {rollout_jvals.shape}."
        )
    if reference_jvals.ndim != 2 or reference_jvals.shape[1] != rollout_jvals.shape[1]:
        raise ValueError(
            "Expected reference hand q shape (R, J) matching rollout J, got "
            f"{reference_jvals.shape} and {rollout_jvals.shape}."
        )
    if finger_indices.ndim != 1 or finger_indices.size == 0:
        raise ValueError(
            f"Expected non-empty finger joint index vector, got {finger_indices.shape}."
        )
    if np.any(finger_indices < 0) or np.any(finger_indices >= rollout_jvals.shape[1]):
        raise IndexError(
            f"Finger joint indices exceed hand q dimension {rollout_jvals.shape[1]}."
        )
    q_delta = (
        rollout_jvals[:, finger_indices]
        - reference_jvals[command_timestep][:, finger_indices]
    )
    q_delta = np.arctan2(np.sin(q_delta), np.cos(q_delta))
    finger_q_rms_delta_deg = np.rad2deg(
        np.sqrt(np.mean(np.square(q_delta), axis=1))
    )

    phase_names = np.asarray(reference["phase_names"], dtype=str)
    phase_lengths = np.asarray(reference["phase_lengths"], dtype=np.int64)
    if phase_names.ndim != 1 or phase_lengths.shape != phase_names.shape:
        raise ValueError(
            "Expected matching one-dimensional phase names and lengths, got "
            f"{phase_names.shape} and {phase_lengths.shape}."
        )
    if phase_names.size == 0 or np.any(phase_lengths <= 0):
        raise ValueError("Expected non-empty phases with positive lengths.")
    if int(phase_lengths.sum()) != reference_object_pos.shape[0]:
        raise ValueError(
            f"Expected phase lengths to sum to {reference_object_pos.shape[0]}, got "
            f"{int(phase_lengths.sum())}."
        )
    phase_index = np.searchsorted(
        np.cumsum(phase_lengths), command_timestep, side="right"
    )

    return ReferencePolicyDeviation(
        rollout_timestep=np.arange(num_steps, dtype=np.int64),
        command_timestep=command_timestep,
        phase_name=phase_names[phase_index],
        corresponding_object=corresponding_object,
        object_position_delta_cm=object_position_delta_cm,
        object_rotation_delta_deg=object_rotation_delta_deg,
        wrist_object_position_delta_cm=wrist_position_delta_cm,
        wrist_object_rotation_delta_deg=wrist_rotation_delta_deg,
        finger_q_rms_delta_deg=finger_q_rms_delta_deg,
    )
