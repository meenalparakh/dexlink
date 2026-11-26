"""A freely graspable nut with finite capture into a native helical constraint."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from dexlink.src.rl.config.passive_mechanisms import NUT_JOINTS, NutThreadConfig

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity


class NutThreadMechanism:
    def __init__(
        self, entity: RigidEntity, num_envs: int, config: NutThreadConfig
    ) -> None:
        from dexlink.src.rl.world.screw_constraint_backend import (
            NUT_EQUALITIES,
            set_thread_engagement,
            set_thread_phase_origin,
        )

        config.validate()
        if tuple(equality.name for equality in entity.equalities) != NUT_EQUALITIES:
            raise ValueError("Expected the five registered nut-thread equality rows.")
        self.entity, self.config, self.num_envs = entity, config, num_envs
        self.dofs = [entity.get_joint(name).dofs_idx_local[0] for name in NUT_JOINTS]
        self.global_dofs = [entity.dof_start + index for index in self.dofs]
        device = entity.get_dofs_position(self.dofs).device
        self.engaged = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.angle_origin = torch.zeros(num_envs, device=device)
        self.equality_indices = torch.tensor(
            [equality.idx for equality in entity.equalities],
            device=device,
            dtype=torch.int32,
        )
        self.set_engagement, self.set_phase_origin = (
            set_thread_engagement,
            set_thread_phase_origin,
        )
        self.free_lower = (
            torch.tensor(
                [-1, -1, -1, -4 * math.pi, -4 * math.pi, -32 * math.pi], device=device
            )
            .expand(num_envs, -1)
            .clone()
        )
        self.free_upper = -self.free_lower
        zeros = torch.zeros((num_envs, 6), device=device)
        entity.set_dofs_kp(zeros, self.dofs)
        entity.set_dofs_kv(zeros, self.dofs)
        # Explicit inertia regularization stabilizes the free Euler-coordinate
        # chain. The rotational addition is 0.36% of the nut's smallest inertia.
        entity.set_dofs_armature(
            torch.full_like(zeros, config.coordinate_armature), self.dofs
        )
        params = torch.tensor(
            [
                max(0.004, 2 * entity.solver.substep_dt),
                1.0,
                0.9999,
                0.9999,
                0.001,
                0.5,
                2.0,
            ],
            device=device,
        )
        entity.solver.set_sol_params(
            params.expand(num_envs, 5, -1).contiguous(), eqs_idx=self.equality_indices
        )
        self.prepare_reset(entity.get_dofs_position(self.dofs))

    def _apply_engagement(self, indices: torch.Tensor) -> None:
        lower, upper = (
            self.free_lower[indices].clone(),
            self.free_upper[indices].clone(),
        )
        engaged = self.engaged[indices]
        lower[engaged, 2] = -self.config.travel
        upper[engaged, 2] = 0
        lower[engaged, 5] = (
            self.angle_origin[indices][engaged]
            - 2 * math.pi * self.config.engaged_turns
        )
        upper[engaged, 5] = self.angle_origin[indices][engaged]
        self.entity.solver.set_dofs_limit(
            lower, upper, self.global_dofs, envs_idx=indices
        )
        self.set_phase_origin(
            self.entity.solver.equalities_info.eq_data,
            int(self.equality_indices[4]),
            indices.to(torch.int32),
            self.angle_origin[indices].float(),
        )
        self.set_engagement(
            self.entity.solver.equalities_info.eq_type,
            self.equality_indices,
            indices.to(torch.int32),
            engaged.to(torch.int32),
        )
        self.entity.set_dofs_frictionloss(
            (engaged[:, None] * self.config.thread_friction_torque).float(),
            [self.dofs[5]],
            envs_idx=indices,
        )
        self.entity.set_dofs_damping(
            (engaged[:, None] * self.config.thread_damping).float(),
            [self.dofs[5]],
            envs_idx=indices,
        )

    def prepare_reset(
        self, coordinates: torch.Tensor, envs_idx: torch.Tensor | None = None
    ) -> None:
        indices = (
            torch.arange(self.num_envs, device=self.engaged.device)
            if envs_idx is None
            else envs_idx
        )
        if coordinates.shape != (len(indices), 6) or not bool(
            torch.isfinite(coordinates).all()
        ):
            raise ValueError(
                f"Expected finite nut coordinates ({len(indices)}, 6), received {tuple(coordinates.shape)}."
            )
        origin = (
            2
            * math.pi
            * torch.round(
                (
                    coordinates[:, 5]
                    - 2 * math.pi / self.config.pitch * coordinates[:, 2]
                )
                / (2 * math.pi)
            )
        )
        residual = (
            coordinates[:, 5]
            - origin
            - 2 * math.pi / self.config.pitch * coordinates[:, 2]
        )
        aligned = (coordinates[:, :2].norm(dim=-1) <= self.config.capture_radius) & (
            coordinates[:, 3:5].abs().amax(-1) <= self.config.capture_tilt
        )
        threaded = (
            aligned
            & (coordinates[:, 2] < -self.config.capture_height)
            & (coordinates[:, 2] >= -self.config.travel - 0.0002)
        )
        if bool((threaded & (residual.abs() > self.config.capture_phase)).any()):
            raise ValueError("Threaded nut reset violates the physical helical pitch.")
        self.engaged[indices] = threaded
        self.angle_origin[indices] = origin
        self._apply_engagement(indices)

    def before_step(self) -> None:
        coordinates = self.entity.get_dofs_position(self.dofs)
        velocity = self.entity.get_dofs_velocity(self.dofs)
        phase_origin = 2 * math.pi * torch.round(coordinates[:, 5] / (2 * math.pi))
        aligned = (coordinates[:, :2].norm(dim=-1) <= self.config.capture_radius) & (
            coordinates[:, 3:5].abs().amax(-1) <= self.config.capture_tilt
        )
        capture = (
            ~self.engaged
            & aligned
            & (coordinates[:, 2].abs() <= self.config.capture_height)
            & ((coordinates[:, 5] - phase_origin).abs() <= self.config.capture_phase)
            & (velocity[:, 2] <= 0)
        )
        if bool(capture.any()):
            self.engaged[capture] = True
            self.angle_origin[capture] = phase_origin[capture]
            self._apply_engagement(torch.where(capture)[0])
