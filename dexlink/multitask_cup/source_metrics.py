"""Complete source-task predicates, including ordered signed turns and released placement."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .conditioning import rotation_matrix


@dataclass(frozen=True)
class CupTrace:
    pose: torch.Tensor
    velocity: torch.Tensor
    hand_contact: torch.Tensor
    table_contact: torch.Tensor
    table_upward_force: torch.Tensor

    def validate(self) -> None:
        if self.pose.ndim != 3 or self.pose.shape[-1] != 7:
            raise ValueError(
                "Expected evaluation poses (frames,episodes,7), including initial frame0."
            )
        shape = self.pose.shape[:2]
        if self.velocity.shape != (*shape, 6) or any(
            value.shape != shape
            for value in (
                self.hand_contact,
                self.table_contact,
                self.table_upward_force,
            )
        ):
            raise ValueError("Expected matching velocity/contact trace dimensions.")
        if (
            self.hand_contact.dtype != torch.bool
            or self.table_contact.dtype != torch.bool
        ):
            raise ValueError("Contact traces must be boolean.")
        if not all(
            torch.isfinite(value).all()
            for value in (self.pose, self.velocity, self.table_upward_force)
        ):
            raise ValueError(
                "Evaluation traces must be finite; failed physics cannot be scored as success."
            )


def pose_within(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distance = torch.linalg.vector_norm(actual[..., :3] - target[..., :3], dim=-1)
    dot = (actual[..., 3:] * target[..., 3:]).sum(-1).abs().clamp(0, 1)
    return (distance <= 0.03) & (dot >= math.cos(math.pi / 24))


def task_success(
    variant: dict, trace: CupTrace, reference: torch.Tensor, mass: float
) -> torch.Tensor:
    trace.validate()
    T = variant["frames"]
    end = T + 10
    required = end + 15 if variant["axis"] == "P" else end
    if trace.pose.shape[0] < required + 1 or reference.shape != (T, 7) or mass <= 0:
        raise ValueError(
            "Expected the complete positive-checkpoint task horizon and rigid-cup reference."
        )
    success = pose_within(trace.pose[end - 4 : end + 1], reference[-1]).all(0)
    if variant["axis"] == "S":
        rotation = rotation_matrix(trace.pose[..., 3:])
        angle = torch.atan2(rotation[..., 0, 2], rotation[..., 0, 0])
        difference = (
            torch.remainder(angle[1:] - angle[:-1] + math.pi, 2 * math.pi) - math.pi
        )
        unwrapped = (
            torch.cat((angle[:1], angle[:1] + difference.cumsum(0)), dim=0) - angle[:1]
        )
        start = 0
        for name, length in zip(variant["phase_names"], variant["phase_frames"]):
            stop = start + length
            if name.startswith("hold_visit_"):
                visit = int(name.removeprefix("hold_visit_")) - 1
                target_angle = math.radians(variant["sequence_degrees"][visit])
                success &= pose_within(trace.pose[start:stop], reference[stop - 1]).all(
                    0
                )
                success &= (
                    (unwrapped[start:stop] - target_angle).abs() <= math.pi / 12
                ).all(0)
            start = stop
    elif variant["axis"] == "P":
        # The final five active frames and full one-second passive interval must remain stable and released.
        interval = slice(end - 4, end + 16)
        success &= pose_within(trace.pose[interval], reference[-1]).all(0)
        success &= (trace.velocity[interval, :, :3].norm(dim=-1) <= 0.03).all(0)
        success &= (trace.velocity[interval, :, 3:].norm(dim=-1) <= 0.3).all(0)
        success &= ~trace.hand_contact[interval].any(0)
        success &= trace.table_contact[interval].all(0)
        success &= (trace.table_upward_force[interval] >= 0.5 * mass * 9.81).all(0)
    elif variant["axis"] not in ("G", "Q", "T"):
        raise ValueError(f"Unsupported source task axis {variant['axis']}.")
    return success
