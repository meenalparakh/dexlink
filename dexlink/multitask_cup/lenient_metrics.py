"""Prospective relaxed cup diagnostics; the frozen source-success predicates remain intact."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from dexlink.multitask_cup.conditioning import rotation_matrix
from dexlink.multitask_cup.source_metrics import CupTrace


@dataclass(frozen=True)
class RelaxedCriteria:
    position_tolerance_m: float = 0.05
    orientation_tolerance_deg: float = 30.0
    hold_frames: int = 3
    lift_fraction: float = 0.5
    lift_clear_frames: int = 2
    placement_linear_speed_m_s: float = 0.10
    placement_angular_speed_rad_s: float = 0.75
    placement_supported_fraction: float = 0.8
    placement_upward_weight_fraction: float = 0.25
    tracking_fraction_threshold: float = 0.8


def pose_close(
    actual: torch.Tensor, target: torch.Tensor, criteria: RelaxedCriteria
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(actual[..., :3] - target[..., :3], dim=-1)
    dot = (actual[..., 3:] * target[..., 3:]).sum(-1).abs().clamp(0, 1)
    return (distance <= criteria.position_tolerance_m) & (
        dot >= math.cos(math.radians(criteria.orientation_tolerance_deg) / 2)
    )


def consecutive(mask: torch.Tensor, frames: int) -> torch.Tensor:
    if mask.ndim != 2 or mask.dtype != torch.bool or not 1 <= frames <= len(mask):
        raise ValueError(
            "Expected a boolean (frames,episodes) mask and a valid positive hold length."
        )
    return mask.unfold(0, frames, 1).all(-1).any(0)


def score_relaxed(
    variant: dict,
    trace: CupTrace,
    reference: torch.Tensor,
    mass: float,
    criteria: RelaxedCriteria,
) -> dict[str, torch.Tensor]:
    trace.validate()
    length = variant["frames"]
    end = length + 10
    required = end + 15 if variant["axis"] == "P" else end
    if reference.shape != (length, 7) or len(trace.pose) < required + 1 or mass <= 0:
        raise ValueError(
            "Expected the complete initial-state task trace, matching reference and positive cup mass."
        )
    reference_lift = reference[:, 2].max() - reference[0, 2]
    if reference_lift <= 0:
        raise ValueError(
            "The approved cup-task diagnostic expects an authored lift phase."
        )
    lift_height = trace.pose[: end + 1, :, 2] - trace.pose[0, :, 2]
    clear_lift = (
        lift_height >= criteria.lift_fraction * reference_lift
    ) & ~trace.table_contact[: end + 1]
    lifted = consecutive(clear_lift, criteria.lift_clear_frames)
    goal = pose_close(trace.pose[: end + 1], reference[-1], criteria)
    completion = goal[end - criteria.hold_frames + 1 : end + 1].all(0) & lifted
    brief_goal = consecutive(goal, criteria.hold_frames) & lifted
    if variant["axis"] == "S":
        rotations = rotation_matrix(trace.pose[..., 3:])
        angle = torch.atan2(rotations[..., 0, 2], rotations[..., 0, 0])
        delta = torch.remainder(angle[1:] - angle[:-1] + math.pi, 2 * math.pi) - math.pi
        unwrapped = torch.cat((angle[:1], angle[:1] + delta.cumsum(0)), 0) - angle[:1]
        start = 0
        for name, duration in zip(variant["phase_names"], variant["phase_frames"]):
            stop = start + duration
            if name.startswith("hold_visit_"):
                visit = int(name.removeprefix("hold_visit_")) - 1
                target = math.radians(variant["sequence_degrees"][visit])
                held = min(criteria.hold_frames, duration)
                interval = slice(stop - held, stop)
                completion &= pose_close(
                    trace.pose[interval], reference[stop - 1], criteria
                ).all(0)
                completion &= (
                    (unwrapped[interval] - target).abs()
                    <= math.radians(criteria.orientation_tolerance_deg)
                ).all(0)
            start = stop
    elif variant["axis"] == "P":
        passive = slice(end + 1, end + 16)
        tail = slice(end + 11, end + 16)
        supported = trace.table_contact[passive] & (
            trace.table_upward_force[passive]
            >= criteria.placement_upward_weight_fraction * mass * 9.81
        )
        completion &= supported.float().mean(0) >= criteria.placement_supported_fraction
        completion &= ~trace.hand_contact[tail].any(0)
        completion &= (
            trace.velocity[tail, :, :3].norm(dim=-1)
            <= criteria.placement_linear_speed_m_s
        ).all(0)
        completion &= (
            trace.velocity[tail, :, 3:].norm(dim=-1)
            <= criteria.placement_angular_speed_rad_s
        ).all(0)
        completion &= pose_close(
            trace.pose[end + 13 : end + 16], reference[-1], criteria
        ).all(0)
    elif variant["axis"] not in ("G", "Q", "T"):
        raise ValueError(f"Unknown approved task axis: {variant['axis']}.")
    tracking = (
        pose_close(trace.pose[1:length], reference[1:, None], criteria).float().mean(0)
    )
    position = (trace.pose[end, :, :3] - reference[-1, :3]).norm(dim=-1)
    dot = (trace.pose[end, :, 3:] * reference[-1, 3:]).sum(-1).abs().clamp(0, 1)
    return dict(
        relaxed_success=completion,
        lift_achieved=lifted,
        brief_goal_attainment=brief_goal,
        tracking_80_percent=(tracking >= criteria.tracking_fraction_threshold) & lifted,
        relaxed_tracking_fraction=tracking,
        max_lift_m=lift_height.max(0).values,
        final_position_error_m=position,
        final_orientation_error_deg=2 * torch.acos(dot) * 180 / math.pi,
    )
