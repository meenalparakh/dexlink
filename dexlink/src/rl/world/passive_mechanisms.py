"""Passive mechanisms using Genesis joint constraints and spring forces."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from dexlink.src.rl.config.passive_mechanisms import (
    CLIP_JOINT,
    SCREW_JOINTS,
    ScrewCapConfig,
    SpringClipConfig,
)

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity


class ScrewCapMechanism:
    """Five native constraints leave only helical motion while engaged.

    Cap X/Y/roll/pitch are constrained to zero and cap Z equals pitch*yaw/(2pi).
    At runout all five constraints deactivate, leaving a free rigid cap.
    """

    def __init__(
        self,
        entity: RigidEntity,
        num_envs: int,
        config: ScrewCapConfig = ScrewCapConfig(),
    ) -> None:
        from dexlink.src.rl.world.screw_constraint_backend import (
            set_thread_engagement,
            THREAD_EQUALITIES,
        )

        config.validate()
        if entity.solver.n_envs != num_envs or num_envs < 1:
            raise ValueError(
                "ScrewCapMechanism requires scene.build(n_envs=N) with N >= 1."
            )
        self.entity, self.config, self.num_envs = entity, config, num_envs
        self.dofs = [entity.get_joint(name).dofs_idx_local[0] for name in SCREW_JOINTS]
        self.global_dofs = [entity.dof_start + i for i in self.dofs]
        expected_equalities = THREAD_EQUALITIES
        names = tuple(equality.name for equality in entity.equalities)
        if names != expected_equalities:
            raise ValueError(
                f"Expected screw equalities {expected_equalities}, got {names}."
            )
        coefficient = entity.equalities[4].eq_data[1]
        if not math.isclose(coefficient, 2 * math.pi / config.pitch, rel_tol=1e-6):
            raise ValueError(
                "Screw config pitch differs from the registered constraint."
            )
        device = entity.get_dofs_position(self.dofs).device
        self.released = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.equality_indices = torch.tensor(
            [e.idx for e in entity.equalities], dtype=torch.int32, device=device
        )
        self.set_engagement = set_thread_engagement
        self.engaged_lower = (
            torch.tensor(
                [-1.0, -1.0, 0.0, -4 * math.pi, -4 * math.pi, 0.0], device=device
            )
            .expand(num_envs, -1)
            .clone()
        )
        self.engaged_upper = (
            torch.tensor(
                [
                    1.0,
                    1.0,
                    config.thread_travel,
                    4 * math.pi,
                    4 * math.pi,
                    config.thread_end_angle,
                ],
                device=device,
            )
            .expand(num_envs, -1)
            .clone()
        )
        self.free_lower = self.engaged_lower.clone()
        self.free_lower[:, 2] = -1.0
        self.free_lower[:, 5] = -4 * math.pi
        self.free_upper = self.engaged_upper.clone()
        self.free_upper[:, 2] = 1.0
        self.free_upper[:, 5] = 4 * math.pi
        zero = torch.zeros((num_envs, 6), device=device)
        entity.set_dofs_kp(zero, self.dofs)
        entity.set_dofs_kv(zero, self.dofs)
        entity.set_dofs_armature(
            torch.full_like(zero, config.coordinate_armature), self.dofs
        )
        self.params = torch.tensor(
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
            self.params.expand(num_envs, 5, -1).contiguous(),
            eqs_idx=self.equality_indices,
        )
        joints = [entity.get_joint(name).idx for name in SCREW_JOINTS]
        entity.solver.set_sol_params(
            self.params.expand(6, -1).contiguous(), joints_idx=joints
        )
        self.reset()

    def _apply_engagement(self, envs_idx: torch.Tensor | None = None) -> None:
        indices = (
            torch.arange(self.num_envs, device=self.released.device)
            if envs_idx is None
            else envs_idx
        )
        lower = torch.where(
            self.released[:, None], self.free_lower, self.engaged_lower
        )[indices]
        upper = torch.where(
            self.released[:, None], self.free_upper, self.engaged_upper
        )[indices]
        self.entity.solver.set_dofs_limit(
            lower, upper, self.global_dofs, envs_idx=indices
        )
        engaged = (~self.released[indices]).to(torch.int32)
        self.set_engagement(
            self.entity.solver.equalities_info.eq_type,
            self.equality_indices,
            indices.to(torch.int32),
            engaged,
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
        expected = (self.num_envs if envs_idx is None else len(envs_idx), 6)
        if coordinates.shape != expected or not bool(torch.isfinite(coordinates).all()):
            raise ValueError(
                f"Expected finite screw reset coordinates of shape {expected}, got {tuple(coordinates.shape)}."
            )
        released = (
            coordinates[:, 5]
            >= self.config.thread_end_angle - self.config.release_angle_tolerance
        )
        error = coordinates[:, 2] - coordinates[:, 5] * (
            self.config.pitch / (2 * math.pi)
        )
        locked = coordinates[:, [0, 1, 3, 4]]
        if bool(
            (error[~released].abs() > self.config.release_position_tolerance).any()
        ) or bool(
            (locked[~released].abs() > self.config.release_position_tolerance).any()
        ):
            raise ValueError(
                "Engaged screw reset coordinates violate the helical constraint."
            )
        if envs_idx is None:
            self.released[:] = released
        else:
            self.released[envs_idx] = released
        self._apply_engagement(envs_idx)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        self.prepare_reset(
            self.entity.get_dofs_position(self.dofs, envs_idx=envs_idx), envs_idx
        )

    def before_step(self) -> None:
        coordinates = self.entity.get_dofs_position(self.dofs)
        error = coordinates[:, 2] - coordinates[:, 5] * (
            self.config.pitch / (2 * math.pi)
        )
        release = (
            ~self.released
            & (
                coordinates[:, 5]
                >= self.config.thread_end_angle - self.config.release_angle_tolerance
            )
            & (error.abs() < self.config.release_position_tolerance)
        )
        if bool(release.any()):
            self.released |= release
            self._apply_engagement(torch.where(release)[0])


class SpringClipMechanism:
    """Normally closed torsional spring: torque = -k*q - damping*qdot."""

    def __init__(
        self,
        entity: RigidEntity,
        num_envs: int,
        config: SpringClipConfig = SpringClipConfig(),
    ) -> None:
        config.validate()
        self.entity = entity
        self.config = config
        self.dof = entity.get_joint(CLIP_JOINT).dofs_idx_local[0]
        device = entity.get_dofs_position([self.dof]).device

        def value(number: float) -> torch.Tensor:
            return torch.full((num_envs, 1), number, device=device)

        entity.set_dofs_kp(value(0), [self.dof])
        entity.set_dofs_kv(value(0), [self.dof])
        entity.set_dofs_armature(value(0), [self.dof])
        entity.set_dofs_stiffness(value(config.stiffness), [self.dof])
        entity.set_dofs_damping(value(config.damping), [self.dof])
        entity.set_dofs_frictionloss(value(config.friction_torque), [self.dof])
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
            params[None], joints_idx=[entity.get_joint(CLIP_JOINT).idx]
        )
