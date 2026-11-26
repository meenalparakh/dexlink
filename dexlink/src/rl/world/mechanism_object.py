"""ArtObject integration for passive screw and spring-clip assemblies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dexlink.src.rl.config.passive_mechanisms import (
    CLIP_JOINT,
    CLIP_ROBOT,
    SCREW_JOINTS,
    SCREW_ROBOT,
    NUT_ROBOT,
    NUT_JOINTS,
)
from dexlink.src.rl.world.art_object import ArtObject
from dexlink.src.rl.world.passive_mechanisms import (
    ScrewCapMechanism,
    SpringClipMechanism,
)
from dexlink.src.rl.world.nut_thread_mechanism import NutThreadMechanism

if TYPE_CHECKING:
    from dexlink.src.rl.world.physics_sim import PhysicsSim


class MechanismObject(ArtObject):
    def __init__(self, physics: PhysicsSim, obj_cfg: dict, num_envs: int) -> None:
        self.mechanism: (
            ScrewCapMechanism | SpringClipMechanism | NutThreadMechanism | None
        ) = None
        super().__init__(physics, obj_cfg, num_envs)
        expected = {
            SCREW_ROBOT: SCREW_JOINTS,
            CLIP_ROBOT: (CLIP_JOINT,),
            NUT_ROBOT: NUT_JOINTS,
        }
        if self.urdf_inst.name not in expected:
            raise ValueError(
                f"Unsupported passive mechanism robot {self.urdf_inst.name!r}."
            )
        names = tuple(joint.name for joint in self.joints)
        if names != expected[self.urdf_inst.name]:
            raise ValueError(
                f"Expected mechanism joints {expected[self.urdf_inst.name]}, got {names}."
            )
        if self.urdf_inst.name == SCREW_ROBOT:
            if not self.physics.sim_cfg.enable_self_collision:
                raise ValueError(
                    "Screw assemblies require sim_cfg.enable_self_collision=True for cap/bottle contact."
                )
            from dexlink.src.rl.world.screw_constraint_backend import (
                add_thread_constraints,
            )

            add_thread_constraints(self.object, self.physics.sim_cfg.screw_cap)
        elif self.urdf_inst.name == NUT_ROBOT:
            if not self.physics.sim_cfg.enable_self_collision:
                raise ValueError(
                    "Nut threads require self collision for nut/stud contact."
                )
            from dexlink.src.rl.world.screw_constraint_backend import (
                add_nut_constraints,
            )

            add_nut_constraints(self.object, self.physics.sim_cfg.nut_thread)
        if self.rest_joint_dofs:
            raise ValueError(
                "Passive mechanisms must not have generic position targets."
            )

    def configure_joint_dynamics(self) -> None:
        if (
            not self.physics.sim_cfg.batch_dofs_info
            or not self.physics.sim_cfg.enable_joint_limit
        ):
            raise ValueError(
                "Passive mechanisms require batch_dofs_info=True and enable_joint_limit=True."
            )
        if self.urdf_inst.name == SCREW_ROBOT:
            self.mechanism = ScrewCapMechanism(
                self.object, self.num_envs, self.physics.sim_cfg.screw_cap
            )
        elif self.urdf_inst.name == NUT_ROBOT:
            self.mechanism = NutThreadMechanism(
                self.object, self.num_envs, self.physics.sim_cfg.nut_thread
            )
        else:
            self.mechanism = SpringClipMechanism(
                self.object, self.num_envs, self.physics.sim_cfg.spring_clip
            )

    def apply_rest_joint_targets(self) -> None:
        if self.mechanism is None:
            raise RuntimeError(
                "Configure passive mechanism dynamics after scene.build()."
            )
        if isinstance(self.mechanism, (ScrewCapMechanism, NutThreadMechanism)):
            self.mechanism.before_step()

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
        if jpos is not None and not isinstance(jpos, torch.Tensor):
            raise ValueError(
                f"Expected mechanism reset jpos as a torch.Tensor, got {type(jpos).__name__}."
            )
        # Restore the appropriate locks before ArtObject clips reset coordinates
        # to joint limits. This also handles resetting a previously removed cap.
        if (
            isinstance(self.mechanism, (ScrewCapMechanism, NutThreadMechanism))
            and jpos is not None
        ):
            envs = (
                None
                if env_idx is None
                else torch.as_tensor(
                    env_idx,
                    dtype=torch.long,
                    device=self.object.get_dofs_position(self.dof_map).device,
                )
            )
            self.mechanism.prepare_reset(
                jpos.to(self.object.get_dofs_position(self.dof_map).device), envs
            )
        super().set_state(
            pos=pos,
            quat=quat,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            jpos=jpos,
            jvel=jvel,
            env_idx=env_idx,
        )
