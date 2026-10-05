from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from dexlink.src.rl.world.art_object import ArtObject

if TYPE_CHECKING:
    from dexlink.src.rl.world.physics_sim import PhysicsSim


CARDBOARD_ASSET_MODELS = {
    (
        "procedural_mailer_folding",
        "carton_folding",
        "carton.urdf",
    ): "finite_thickness_tuck_carton",
    (
        "procedural_mailer_folding",
        "mailer_box",
        "mailer.urdf",
    ): "procedural_hinged_mailer",
}


@dataclass(frozen=True)
class CardboardJointConfig:
    """Constitutive parameters for the two procedural cardboard assets.

    The PD target represents the current crease rest angle. Joint damping and
    friction dissipate motion, while the target is plastically updated after an
    applied joint load disappears.
    """

    stiffness: float = 0.12
    actuator_damping: float = 0.15
    joint_damping: float = 0.08
    joint_friction: float = 0.003
    force_limit: float = 0.35
    load_torque_threshold: float = 0.015
    stationary_velocity_threshold: float = 0.02
    retained_deformation: float = 0.70
    release_confirmation_steps: int = 12

    def validate(self) -> None:
        positive_values = {
            "stiffness": self.stiffness,
            "actuator_damping": self.actuator_damping,
            "joint_damping": self.joint_damping,
            "force_limit": self.force_limit,
            "load_torque_threshold": self.load_torque_threshold,
            "stationary_velocity_threshold": self.stationary_velocity_threshold,
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f"Expected cardboard {name} > 0, got {value}.")
        if self.joint_friction < 0.0:
            raise ValueError(
                f"Expected cardboard joint_friction >= 0, got {self.joint_friction}."
            )
        if not 0.0 <= self.retained_deformation <= 1.0:
            raise ValueError(
                "Expected cardboard retained_deformation in [0, 1], got "
                f"{self.retained_deformation}."
            )
        if self.release_confirmation_steps < 1:
            raise ValueError(
                "Expected cardboard release_confirmation_steps >= 1, got "
                f"{self.release_confirmation_steps}."
            )


def cardboard_model_for_path(urdf_path: Path | str) -> str | None:
    """Return the expected robot name only for an exact supported asset path."""

    path_parts = Path(urdf_path).parts
    for asset_suffix, robot_name in CARDBOARD_ASSET_MODELS.items():
        if tuple(path_parts[-len(asset_suffix) :]) == asset_suffix:
            return robot_name
    return None


def update_cardboard_plastic_state(
    *,
    rest_angles: torch.Tensor,
    loaded_angles: torch.Tensor,
    was_loaded: torch.Tensor,
    release_counts: torch.Tensor,
    joint_angles: torch.Tensor,
    joint_velocities: torch.Tensor,
    external_torques: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    config: CardboardJointConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the per-environment cardboard load/release state by one step."""

    expected_shape = rest_angles.shape
    tensors_with_joint_shape = {
        "loaded_angles": loaded_angles,
        "was_loaded": was_loaded,
        "release_counts": release_counts,
        "joint_angles": joint_angles,
        "joint_velocities": joint_velocities,
        "external_torques": external_torques,
        "lower_limits": lower_limits,
        "upper_limits": upper_limits,
    }
    if rest_angles.ndim != 2:
        raise ValueError(
            "Expected cardboard state tensors with shape (num_envs, num_joints), "
            f"got rest_angles shape {tuple(rest_angles.shape)}."
        )
    for name, value in tensors_with_joint_shape.items():
        if value.shape != expected_shape:
            raise ValueError(
                f"Expected {name} shape {tuple(expected_shape)}, got "
                f"{tuple(value.shape)}."
            )
    if was_loaded.dtype != torch.bool:
        raise ValueError(
            f"Expected was_loaded dtype torch.bool, got {was_loaded.dtype}."
        )
    if release_counts.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"Expected integral release_counts, got dtype {release_counts.dtype}."
        )
    config.validate()

    deformation = joint_angles - rest_angles
    torque_increases_deformation = external_torques * deformation > 0.0
    moving_away_or_held = (joint_velocities * deformation >= 0.0) | (
        joint_velocities.abs() <= config.stationary_velocity_threshold
    )
    starts_new_load = (
        (external_torques.abs() >= config.load_torque_threshold)
        & torque_increases_deformation
        & moving_away_or_held
    )
    continues_existing_load = was_loaded & (
        external_torques.abs() >= config.load_torque_threshold
    )
    load_is_applied = starts_new_load | continues_existing_load
    next_loaded_angles = torch.where(load_is_applied, joint_angles, loaded_angles)
    next_was_loaded = was_loaded | load_is_applied
    next_release_counts = torch.where(
        load_is_applied,
        torch.zeros_like(release_counts),
        torch.where(
            next_was_loaded,
            release_counts + 1,
            torch.zeros_like(release_counts),
        ),
    )

    released = next_was_loaded & (
        next_release_counts >= config.release_confirmation_steps
    )
    retained_angles = rest_angles + config.retained_deformation * (
        next_loaded_angles - rest_angles
    )
    retained_angles = torch.maximum(
        torch.minimum(retained_angles, upper_limits), lower_limits
    )
    next_rest_angles = torch.where(released, retained_angles, rest_angles)
    next_was_loaded = next_was_loaded & ~released
    next_release_counts = torch.where(
        released, torch.zeros_like(next_release_counts), next_release_counts
    )
    return (
        next_rest_angles,
        next_loaded_angles,
        next_was_loaded,
        next_release_counts,
    )


class CardboardObject(ArtObject):
    """Articulated cardboard with damped resistance and plastic crease memory."""

    def __init__(
        self,
        physics: PhysicsSim,
        obj_cfg: dict,
        num_envs: int,
        joint_config: CardboardJointConfig | None = None,
    ) -> None:
        expected_robot_name = obj_cfg.get("cardboard_model")
        if expected_robot_name not in CARDBOARD_ASSET_MODELS.values():
            raise ValueError(
                "CardboardObject requires obj_cfg['cardboard_model'] to be one of "
                f"{sorted(CARDBOARD_ASSET_MODELS.values())}, got "
                f"{expected_robot_name!r}."
            )
        if bool(obj_cfg["fixed"]):
            raise ValueError(
                "Cardboard constitutive behavior requires a dynamic object, but "
                f"{obj_cfg['name']!r} was configured as fixed."
            )

        self.joint_config = joint_config or CardboardJointConfig()
        self.joint_config.validate()
        self.rest_angles: torch.Tensor | None = None
        self.loaded_angles: torch.Tensor | None = None
        self.was_loaded: torch.Tensor | None = None
        self.release_counts: torch.Tensor | None = None
        self.lower_limits: torch.Tensor | None = None
        self.upper_limits: torch.Tensor | None = None
        super().__init__(physics=physics, obj_cfg=obj_cfg, num_envs=num_envs)

        if self.urdf_inst.name != expected_robot_name:
            raise ValueError(
                f"Expected cardboard URDF robot name {expected_robot_name!r}, got "
                f"{self.urdf_inst.name!r} from {obj_cfg['urdf_path']!r}."
            )
        if not self.actuated:
            raise ValueError(
                f"Cardboard asset {expected_robot_name!r} has no movable joints."
            )
        invalid_joint_types = {
            joint.name: joint.joint_type
            for joint in self.joints
            if joint.joint_type != "revolute"
        }
        if invalid_joint_types:
            raise ValueError(
                "Expected every cardboard joint to be revolute, got "
                f"{invalid_joint_types}."
            )
        if self.rest_joint_dofs:
            raise ValueError(
                "CardboardObject manages its own moving rest angles and cannot use "
                f"generic rest_joint_targets {self.rest_joint_names}."
            )

    def configure_joint_dynamics(self) -> None:
        """Configure Genesis hinge dynamics after the scene has been built."""

        n_joints = len(self.dof_map)
        shape = (self.num_envs, n_joints)
        config = self.joint_config
        dofs = self.dof_map
        self.physics._set_entity_dofs_kp(
            self.object,
            kp=np.full(shape, config.stiffness, dtype=np.float32),
            dofs_idx_local=dofs,
        )
        self.physics._set_entity_dofs_kv(
            self.object,
            kv=np.full(shape, config.actuator_damping, dtype=np.float32),
            dofs_idx_local=dofs,
        )
        self.physics._set_entity_dofs_damping(
            self.object,
            damping=np.full(shape, config.joint_damping, dtype=np.float32),
            dofs_idx_local=dofs,
        )
        self.physics._set_entity_dofs_frictionloss(
            self.object,
            frictionloss=np.full(shape, config.joint_friction, dtype=np.float32),
            dofs_idx_local=dofs,
        )
        self.physics._set_entity_dofs_force_range(
            self.object,
            lower=np.full(shape, -config.force_limit, dtype=np.float32),
            upper=np.full(shape, config.force_limit, dtype=np.float32),
            dofs_idx_local=dofs,
        )

        joint_angles = self.physics.get_entity_dofs_position(self.object, dofs)
        lower_limits, upper_limits = self.physics.get_entity_dofs_limit(
            self.object, dofs_idx_local=dofs
        )
        if joint_angles.shape != shape:
            raise ValueError(
                f"Expected cardboard joint state shape {shape}, got "
                f"{tuple(joint_angles.shape)}."
            )
        if lower_limits.shape != shape or upper_limits.shape != shape:
            raise ValueError(
                f"Expected cardboard joint limit shapes {shape}, got "
                f"{tuple(lower_limits.shape)} and {tuple(upper_limits.shape)}."
            )

        self.rest_angles = torch.zeros_like(joint_angles)
        self.loaded_angles = joint_angles.clone()
        self.was_loaded = torch.zeros_like(joint_angles, dtype=torch.bool)
        self.release_counts = torch.zeros_like(joint_angles, dtype=torch.int64)
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits

    def _require_plastic_state(self) -> tuple[torch.Tensor, ...]:
        state = (
            self.rest_angles,
            self.loaded_angles,
            self.was_loaded,
            self.release_counts,
            self.lower_limits,
            self.upper_limits,
        )
        if any(value is None for value in state):
            raise RuntimeError(
                "Cardboard joint state is unavailable. Call "
                "configure_joint_dynamics() after building the Genesis scene."
            )
        return cast(tuple[torch.Tensor, ...], state)

    def set_state(
        self,
        pos=None,
        quat=None,
        lin_vel=None,
        ang_vel=None,
        jpos: torch.Tensor | None = None,
        jvel: torch.Tensor | None = None,
        env_idx=None,
    ) -> None:
        super().set_state(
            pos=pos,
            quat=quat,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            jpos=jpos,
            jvel=jvel,
            env_idx=env_idx,
        )
        if self.rest_angles is None:
            return

        indices = slice(None) if env_idx is None else env_idx
        joint_angles = self.physics.get_entity_dofs_position(self.object, self.dof_map)
        rest_angles, loaded_angles, was_loaded, release_counts, _, _ = (
            self._require_plastic_state()
        )
        # Every episode starts with the URDF's flat zero-angle net as its rest state.
        rest_angles[indices] = 0.0
        loaded_angles[indices] = joint_angles[indices]
        was_loaded[indices] = False
        release_counts[indices] = 0

    def apply_rest_joint_targets(self) -> None:
        (
            rest_angles,
            loaded_angles,
            was_loaded,
            release_counts,
            lower_limits,
            upper_limits,
        ) = self._require_plastic_state()
        joint_angles = self.physics.get_entity_dofs_position(self.object, self.dof_map)
        joint_velocities = self.physics.get_entity_dofs_velocity(
            self.object, self.dof_map
        )
        total_torque = self.physics.get_entity_dofs_force(self.object, self.dof_map)
        controller_torque = self.physics.get_entity_dofs_control_force(
            self.object, self.dof_map
        )
        external_torque = total_torque - controller_torque

        (
            self.rest_angles,
            self.loaded_angles,
            self.was_loaded,
            self.release_counts,
        ) = update_cardboard_plastic_state(
            rest_angles=rest_angles,
            loaded_angles=loaded_angles,
            was_loaded=was_loaded,
            release_counts=release_counts,
            joint_angles=joint_angles,
            joint_velocities=joint_velocities,
            external_torques=external_torque,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            config=self.joint_config,
        )
        self.physics.control_entity_dofs_position(
            self.object,
            self.rest_angles,
            self.dof_map,
        )
