"""Vectorized wrist-frame observations and task-local reference clocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry_policy import GeometryPolicyInputs
from .policy import OFFSETS
from .task_data import TaskBank


def rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    if quaternion.shape[-1] != 4 or not torch.allclose(
        quaternion.norm(dim=-1), torch.ones_like(quaternion[..., 0]), atol=1e-4
    ):
        raise ValueError("Expected unit quaternions ordered w,x,y,z.")
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def rotate(rotation: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return (rotation @ vector[..., None]).squeeze(-1)


def pose_features(pose: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
    inverse = rotation_matrix(wrist[..., 3:]).transpose(-1, -2)
    position = rotate(inverse, pose[..., :3] - wrist[..., :3]) / 0.1
    orientation = inverse @ rotation_matrix(pose[..., 3:])
    return torch.cat(
        (position, orientation[..., :, :2].transpose(-1, -2).flatten(-2)), -1
    )


@dataclass(frozen=True)
class PhysicalState:
    wrist: torch.Tensor
    fingers: torch.Tensor
    finger_velocity: torch.Tensor
    cup: torch.Tensor
    cup_velocity: torch.Tensor
    wrist_velocity: torch.Tensor

    def validate(self, batch: int) -> None:
        for name, width in (
            ("wrist", 7),
            ("fingers", 12),
            ("finger_velocity", 12),
            ("cup", 7),
            ("cup_velocity", 6),
            ("wrist_velocity", 6),
        ):
            value = getattr(self, name)
            if tuple(value.shape) != (batch, width) or not torch.isfinite(value).all():
                raise ValueError(
                    f"Expected finite physical {name} ({batch},{width}), received {tuple(value.shape)}."
                )


def make_inputs(
    bank: TaskBank,
    task_ids: torch.Tensor,
    clocks: torch.Tensor,
    physical: PhysicalState,
) -> GeometryPolicyInputs:
    batch = task_ids.numel()
    if (
        task_ids.shape != (batch,)
        or clocks.shape != (batch,)
        or task_ids.dtype != torch.long
        or clocks.dtype != torch.long
    ):
        raise ValueError("Expected one-dimensional int64 task IDs and clocks.")
    if (
        (task_ids < 0).any()
        or (task_ids >= len(bank.lengths)).any()
        or (clocks < 0).any()
    ):
        raise ValueError("Task ID or task-local clock out of bounds.")
    physical.validate(batch)
    lengths = bank.lengths[task_ids]
    time = torch.minimum(clocks, lengths - 1)
    ends = bank.phase_ends[task_ids]
    phase = (time[:, None] >= ends).sum(-1)
    phase_lengths = bank.phase_lengths[task_ids, phase]
    phase_end = ends.gather(1, phase[:, None]).squeeze(1)
    phase_start = phase_end - phase_lengths
    endpoint = bank.cup[task_ids, phase_end - 1]
    final = bank.cup[task_ids, lengths - 1]
    inverse = rotation_matrix(physical.wrist[:, 3:]).transpose(-1, -2)
    displacement = physical.cup[:, :3] - physical.wrist[:, :3]
    relative_linear = (
        physical.cup_velocity[:, :3]
        - physical.wrist_velocity[:, :3]
        - torch.cross(physical.wrist_velocity[:, 3:], displacement, dim=-1)
    )
    relative_angular = physical.cup_velocity[:, 3:] - physical.wrist_velocity[:, 3:]
    gravity = physical.wrist.new_tensor((0, 0, -1)).expand(batch, -1)
    fingers = (
        2
        * (physical.fingers - bank.finger_lower)
        / (bank.finger_upper - bank.finger_lower)
        - 1
    )
    state = torch.cat(
        (
            fingers,
            physical.finger_velocity / 10,
            pose_features(physical.cup, physical.wrist),
            rotate(inverse, relative_linear),
            rotate(inverse, relative_angular) / 10,
            rotate(inverse, physical.wrist_velocity[:, :3]),
            rotate(inverse, physical.wrist_velocity[:, 3:]) / 10,
            rotate(inverse, gravity),
            physical.wrist[:, 2:3] / 0.1,
            pose_features(endpoint, physical.wrist),
            pose_features(final, physical.wrist),
        ),
        -1,
    )
    indices = time[:, None] + time.new_tensor(OFFSETS)
    valid = (indices >= 0) & (indices < lengths[:, None])
    indices = torch.minimum(indices.clamp_min(0), lengths[:, None] - 1)
    window_wrist = physical.wrist[:, None, :].expand(-1, 10, -1)
    targets = bank.fingers[task_ids[:, None], indices]
    targets = (
        2 * (targets - bank.finger_lower) / (bank.finger_upper - bank.finger_lower) - 1
    )
    window = torch.cat(
        (
            pose_features(bank.wrist[task_ids[:, None], indices], window_wrist),
            targets,
            pose_features(bank.cup[task_ids[:, None], indices], window_wrist),
        ),
        -1,
    )
    progress = torch.stack(
        (
            (time - phase_start) / (phase_lengths - 1).clamp_min(1),
            (phase_end - 1 - time).clamp_min(0) / 150,
            (lengths - 1 - time) / 150,
            (clocks >= lengths).to(state.dtype),
        ),
        -1,
    )
    return GeometryPolicyInputs(
        state,
        window,
        valid,
        bank.geometry_points,
        bank.geometry_metadata,
        bank.geometry_index[task_ids],
        bank.program_tokens,
        bank.program_mask,
        task_ids,
        phase,
        progress,
    )


def reference_state(
    bank: TaskBank, task_ids: torch.Tensor, clocks: torch.Tensor
) -> PhysicalState:
    """FK reference fixture for conditioning tests; never a policy evaluation."""
    time = torch.minimum(clocks, bank.lengths[task_ids] - 1)
    zeros6 = bank.wrist.new_zeros((len(task_ids), 6))
    return PhysicalState(
        bank.wrist[task_ids, time],
        bank.fingers[task_ids, time],
        zeros6.repeat(1, 2),
        bank.cup[task_ids, time],
        zeros6,
        zeros6,
    )


def absolute_targets(
    bank: TaskBank,
    task_ids: torch.Tensor,
    clocks: torch.Tensor,
    residual: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Residuals modify the current native target, independently of normalized policy features."""
    from dexlink.utils.retargeting.math_utils import quat_from_rotvec, quat_mul

    if residual.shape != (len(task_ids), 18) or scales.shape != (18,):
        raise ValueError("Expected residuals (B,18) and 18 physical action scales.")
    time = torch.minimum(clocks, bank.lengths[task_ids] - 1)
    delta = residual * scales
    pose = bank.wrist[task_ids, time]
    quaternion = quat_mul(quat_from_rotvec(delta[:, 3:6]), pose[:, 3:])
    fingers = (bank.fingers[task_ids, time] + delta[:, 6:]).clamp(
        bank.finger_lower, bank.finger_upper
    )
    return torch.cat((pose[:, :3] + delta[:, :3], quaternion, fingers), -1)
