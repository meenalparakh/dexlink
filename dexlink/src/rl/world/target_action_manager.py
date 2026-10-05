from __future__ import annotations

import torch

from dexlink.src.rl.world.action_manager import HandActionSpec
from dexlink.src.rl.world.action_manager_v2 import ActionManager


class AbsoluteTargetActionManager(ActionManager):
    """Execute absolute wrist-pose and finger-joint targets from the policy."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.term_dims = [
            7 + len(spec.hand_target_indices) for spec in self.hand_specs.values()
        ]

    def summary(self) -> str:
        return (
            super()
            .summary()
            .replace(
                "single IK solve + joint-space arm interpolation",
                "absolute target wrist pose + finger joints",
            )
        )

    def _compute_position_targets(
        self, *, spec: HandActionSpec, action: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        expected_shape = (self.env.num_envs, 7 + len(spec.hand_target_indices))
        if tuple(action.shape) != expected_shape:
            raise ValueError(
                f"{spec.side!r} absolute target action shape mismatch: expected "
                f"{expected_shape}, got {tuple(action.shape)}."
            )
        named_obs = self._require_current_policy_named_obs()
        current_qpos = self._require_obs_term(
            named_obs,
            key=f"{spec.side}_jpos",
            expected_last_dim=len(spec.hand.dof_map),
        ).clone()
        current_arm_jpos = self.env.current_wrist_control_jpos_by_side.get(spec.side)
        if current_arm_jpos is None:
            raise ValueError(
                f"Absolute target action requires current wrist-control jpos for hand {spec.side!r}."
            )
        current_qpos[:, spec.arm_joint_indices] = current_arm_jpos
        current_wrist_pos = self.env.current_wrist_pos_by_side.get(spec.side)
        current_wrist_quat = self.env.current_wrist_quat_by_side.get(spec.side)
        if current_wrist_pos is None or current_wrist_quat is None:
            raise ValueError(
                f"Absolute target action requires current wrist pose for hand {spec.side!r}."
            )
        target_wrist_pos = action[:, :3]
        target_wrist_quat = action[:, 3:7]
        target_hand_qpos = action[:, 7:]
        if self.env.cfg.backend == "real":
            current_qpos = self._to_real_control_tensor(current_qpos)
            current_wrist_pos = self._to_real_control_tensor(current_wrist_pos)
            current_wrist_quat = self._to_real_control_tensor(current_wrist_quat)
            target_wrist_pos = self._to_real_control_tensor(target_wrist_pos)
            target_wrist_quat = self._to_real_control_tensor(target_wrist_quat)
            target_hand_qpos = self._to_real_control_tensor(target_hand_qpos)
        target_wrist_pos = self._object_align_real_eval_wrist_target(
            spec=spec,
            target_wrist_pos=target_wrist_pos,
        )
        ik_qpos = self.physics.solve_entity_ik(
            spec.hand.robot,
            link_name=spec.hand.wrist_link_name,
            target_pos=target_wrist_pos,
            target_quat=target_wrist_quat,
            dofs_idx_local=spec.arm_ik_dof_indices.tolist(),
            init_qpos=current_qpos,
            init_qpos_dofs_idx_local=spec.hand.dof_map,
        )
        expected_ik_shape = (self.env.num_envs, len(spec.hand.dof_map))
        if tuple(ik_qpos.shape) != expected_ik_shape:
            raise ValueError(
                f"IK qpos shape mismatch: expected {expected_ik_shape}, got {tuple(ik_qpos.shape)}."
            )
        position_targets = current_qpos.clone()
        ik_qpos = ik_qpos.to(device=position_targets.device)
        position_targets[:, spec.arm_joint_indices] = ik_qpos[:, spec.arm_joint_indices]
        position_targets[:, spec.hand_target_indices] = target_hand_qpos
        lower_limit, upper_limit = self.physics.get_hand_dof_limits(
            spec.hand, spec.hand.dof_map
        )
        if self.env.cfg.backend == "real":
            lower_limit = self._to_real_control_tensor(lower_limit)
            upper_limit = self._to_real_control_tensor(upper_limit)
        position_targets = torch.clip(position_targets, lower_limit, upper_limit)
        target_hand_qpos = position_targets[:, spec.hand_target_indices]
        return (
            position_targets,
            current_qpos,
            current_wrist_pos,
            current_wrist_quat,
            target_wrist_pos,
            target_wrist_quat,
            target_hand_qpos,
        )
