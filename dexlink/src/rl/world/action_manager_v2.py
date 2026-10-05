from __future__ import annotations

import numpy as np
import torch

from dexlink.src.rl.world.action_manager import ActionManagerBase


class ActionManager(ActionManagerBase):
    """Single-IK action manager.

    This variant computes IK once for the final wrist target, then interpolates the
    Franka arm in joint space during decimation instead of solving IK at every
    decimation step.
    """

    def summary(self) -> str:
        spacing = 20
        lines = [
            (
                "[ActionManager] single IK solve + joint-space arm interpolation, "
                f"action_mode: {self.action_mode}, "
                f"residual_beta: {self.residual_beta}, "
                f"env_decimation: {self.decimation}, "
                f"arm_translation_delta_scale: {self.cfg.arm_translation_delta_scale}, "
                f"arm_rotation_delta_scale: {self.cfg.arm_rotation_delta_scale}, "
                f"hand_joint_delta_scale: {self.cfg.hand_joint_delta_scale}, "
                f"position_target_blend_alpha_start: {self.position_target_blend_alpha_start}, "
                f"position_target_blend_alpha_end: {self.position_target_blend_alpha_end}"
            ),
            f"{'Action Term':<{spacing}} {'Dimension':<10}",
            "-" * (spacing + 10),
        ]
        for term, dim in zip(self.action_terms, self.term_dims):
            lines.append(f"{term:<{spacing}} {dim:<10}")
        lines.append("-" * (spacing + 10))
        lines.append(f"{'Total':<{spacing}} {self.action_dim:<10}")
        return "\n".join(lines)

    def _interpolate_joint_targets(
        self,
        *,
        spec,
        phase_name: str,
        current_qpos: torch.Tensor,
        start_position_targets: torch.Tensor,
        target_position_targets: torch.Tensor,
        final_hold_steps: int = 0,
        check_eval_safety: bool = False,
    ) -> list[float] | None:
        expected_shape = (self.env.num_envs, len(spec.hand.dof_map))
        if tuple(current_qpos.shape) != expected_shape:
            raise ValueError(
                f"Expected current_qpos shape {expected_shape}, got {tuple(current_qpos.shape)}."
            )
        if tuple(start_position_targets.shape) != expected_shape:
            raise ValueError(
                "Start position target shape mismatch: expected "
                f"{expected_shape}, got {tuple(start_position_targets.shape)}."
            )
        if tuple(target_position_targets.shape) != expected_shape:
            raise ValueError(
                "Target position target shape mismatch: expected "
                f"{expected_shape}, got {tuple(target_position_targets.shape)}."
            )
        if final_hold_steps < 0:
            raise ValueError(f"Expected final_hold_steps >= 0, got {final_hold_steps}.")

        interpolation_steps = self._interpolation_num_steps(
            current_qpos=current_qpos,
            target_qpos=target_position_targets,
        )
        self._print_real_control_stage(
            f"{phase_name}: computed target joint positions "
            f"(interpolation_steps={interpolation_steps})"
        )

        start_arm_targets = start_position_targets[:, spec.arm_joint_indices]
        target_arm_targets = target_position_targets[:, spec.arm_joint_indices]
        start_hand_targets = start_position_targets[:, spec.hand_target_indices]
        target_hand_targets = target_position_targets[:, spec.hand_target_indices]

        alpha_schedule: list[float] = []
        for step_idx in range(interpolation_steps):
            self._print_real_control_stage(
                f"{phase_name}: interpolating targets "
                f"(step {step_idx + 1}/{interpolation_steps})"
            )
            if check_eval_safety and self._should_block_eval_action_near_base(
                refresh_from_physics=step_idx > 0
            ):
                return None
            alpha = self._blend_alpha_for_step(step_idx, interpolation_steps)
            alpha_schedule.append(alpha)
            blended_position_targets = start_position_targets.clone()
            blended_position_targets[:, spec.arm_joint_indices] = (
                start_arm_targets + alpha * (target_arm_targets - start_arm_targets)
            )
            blended_position_targets[:, spec.hand_target_indices] = (
                start_hand_targets + alpha * (target_hand_targets - start_hand_targets)
            )
            self.physics.control_entity_dofs_position(
                spec.hand.robot,
                blended_position_targets,
                spec.hand.dof_map,
            )
            self.env.physics_manager.step()

        for hold_step_idx in range(final_hold_steps):
            self._print_real_control_stage(
                f"{phase_name}: holding final target "
                f"(step {hold_step_idx + 1}/{final_hold_steps})"
            )
            self.physics.control_entity_dofs_position(
                spec.hand.robot,
                target_position_targets,
                spec.hand.dof_map,
            )
            self.env.physics_manager.step()

        return alpha_schedule

    def apply_actions(self, actions: torch.Tensor):
        expected_shape = (self.env.num_envs, self.action_dim)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"Action shape mismatch: expected {expected_shape}, got {tuple(actions.shape)}."
            )

        named_actions = self.named_view(actions)
        self.env.current_action_probe = None
        self.env.current_action_probe_by_side = {}
        targets_by_side = {}
        for side, spec in self.hand_specs.items():
            action = named_actions.get(side)
            if action is None:
                raise ValueError(f"Expected action term {side!r} to be present.")
            target_tuple = self._compute_position_targets(spec=spec, action=action)
            previous_position_targets = self.env.previous_position_targets_by_side.get(
                side
            )
            if previous_position_targets is None:
                raise ValueError(
                    f"Previous position targets are not initialized for hand {side!r}. "
                    "Reset must populate them before stepping."
                )
            if self.env.cfg.backend == "real":
                previous_position_targets = self._to_real_control_tensor(
                    previous_position_targets
                )
            expected_prev_shape = (self.env.num_envs, len(spec.hand.dof_map))
            if tuple(previous_position_targets.shape) != expected_prev_shape:
                raise ValueError(
                    f"Previous position target shape mismatch for hand {side!r}: "
                    f"expected {expected_prev_shape}, got {tuple(previous_position_targets.shape)}."
                )
            targets_by_side[side] = (
                spec,
                action,
                previous_position_targets,
                *target_tuple,
            )

        if self.env.cfg.backend == "real":
            if len(targets_by_side) != 1:
                raise ValueError("The real backend supports only one controlled hand.")
            side, (
                spec,
                action,
                previous_position_targets,
                position_targets,
                current_qpos,
                current_wrist_pos,
                current_wrist_quat,
                target_wrist_pos,
                target_wrist_quat,
                hand_target_qpos,
            ) = next(iter(targets_by_side.items()))
            self._print_real_control_stage(
                "policy phase: computed joint targets from current observations"
            )
            if self._should_block_eval_action_near_base(refresh_from_physics=False):
                self._print_real_control_stage(
                    "policy phase: blocked action near robot base safety threshold"
                )
                return
            self._print_real_control_stage(
                "policy phase: updating desired joint targets for continuous control"
            )
            self.physics.control_entity_dofs_position(
                spec.hand.robot, position_targets, spec.hand.dof_map
            )
            self._print_real_control_stage(
                "policy phase: refreshing real state snapshot after target update"
            )
            self.env.physics_manager.step()
            alpha_schedule = [1.0]
            self._record_action_probe(
                spec=spec,
                action=action,
                start_qpos=current_qpos,
                target_qpos=position_targets,
                start_wrist_pos=current_wrist_pos,
                start_wrist_quat=current_wrist_quat,
                target_wrist_pos=target_wrist_pos,
                target_wrist_quat=target_wrist_quat,
                hand_target_qpos=hand_target_qpos,
                alpha_schedule=alpha_schedule,
            )
            self.env.previous_position_targets_by_side[side] = (
                position_targets.detach().clone()
            )
            return

        alpha_schedule = [
            self._blend_alpha_for_step(step_idx, self.decimation)
            for step_idx in range(self.decimation)
        ]
        for alpha in alpha_schedule:
            for (
                spec,
                _action,
                previous_position_targets,
                position_targets,
                *_rest,
            ) in targets_by_side.values():
                blended_position_targets = previous_position_targets + alpha * (
                    position_targets - previous_position_targets
                )
                self.physics.control_entity_dofs_position(
                    spec.hand.robot, blended_position_targets, spec.hand.dof_map
                )
            self.env.physics_manager.step()

        for side, (
            spec,
            action,
            _previous_position_targets,
            position_targets,
            current_qpos,
            current_wrist_pos,
            current_wrist_quat,
            target_wrist_pos,
            target_wrist_quat,
            hand_target_qpos,
        ) in targets_by_side.items():
            self._record_action_probe(
                spec=spec,
                action=action,
                start_qpos=current_qpos,
                target_qpos=position_targets,
                start_wrist_pos=current_wrist_pos,
                start_wrist_quat=current_wrist_quat,
                target_wrist_pos=target_wrist_pos,
                target_wrist_quat=target_wrist_quat,
                hand_target_qpos=hand_target_qpos,
                alpha_schedule=alpha_schedule,
            )
            self.env.previous_position_targets_by_side[side] = (
                position_targets.detach().clone()
            )

    def move_to_reset_target(
        self,
        *,
        side: str = "right",
        env_ids: np.ndarray,
        target_wrist_pos: torch.Tensor,
        target_wrist_quat: torch.Tensor,
        target_hand_qpos: torch.Tensor,
    ) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        if env_ids.ndim != 1:
            raise ValueError(
                f"Expected env_ids to be rank-1, got shape {env_ids.shape}."
            )
        if len(env_ids) == 0:
            return

        expected_wrist_pos_shape = (len(env_ids), 3)
        if tuple(target_wrist_pos.shape) != expected_wrist_pos_shape:
            raise ValueError(
                "Reset wrist position shape mismatch: expected "
                f"{expected_wrist_pos_shape}, got {tuple(target_wrist_pos.shape)}."
            )
        expected_wrist_quat_shape = (len(env_ids), 4)
        if tuple(target_wrist_quat.shape) != expected_wrist_quat_shape:
            raise ValueError(
                "Reset wrist quaternion shape mismatch: expected "
                f"{expected_wrist_quat_shape}, got {tuple(target_wrist_quat.shape)}."
            )
        spec = self.hand_specs.get(side)
        if spec is None:
            raise ValueError(f"Unknown reset target hand side {side!r}.")
        expected_hand_shape = (len(env_ids), len(spec.hand_target_indices))
        if tuple(target_hand_qpos.shape) != expected_hand_shape:
            raise ValueError(
                "Reset hand target shape mismatch: expected "
                f"{expected_hand_shape}, got {tuple(target_hand_qpos.shape)}."
            )

        current_qpos = self.physics.get_hand_dof_positions(
            spec.hand, spec.hand.dof_map
        ).detach()
        current_wrist_pos = self.physics.get_entity_link_pos(
            spec.hand.robot,
            link_name=spec.hand.wrist_link_name,
            ref="link_origin",
        ).squeeze(1)
        current_wrist_quat = self.physics.get_entity_link_quat(
            spec.hand.robot,
            link_name=spec.hand.wrist_link_name,
        ).squeeze(1)

        if self.env.cfg.backend == "real":
            current_qpos = self._to_real_control_tensor(current_qpos)
            current_wrist_pos = self._to_real_control_tensor(current_wrist_pos)
            current_wrist_quat = self._to_real_control_tensor(current_wrist_quat)
            target_wrist_pos = self._to_real_control_tensor(target_wrist_pos)
            target_wrist_quat = self._to_real_control_tensor(target_wrist_quat)
            target_hand_qpos = self._to_real_control_tensor(target_hand_qpos)

        target_wrist_pos_full = current_wrist_pos.clone()
        target_wrist_pos_full[env_ids] = target_wrist_pos
        target_wrist_quat_full = current_wrist_quat.clone()
        target_wrist_quat_full[env_ids] = target_wrist_quat
        target_hand_qpos_full = current_qpos[:, spec.hand_target_indices].clone()
        target_hand_qpos_full[env_ids] = target_hand_qpos

        position_targets = current_qpos.clone()
        ik_qpos = self.physics.solve_entity_ik(
            spec.hand.robot,
            link_name=spec.hand.wrist_link_name,
            target_pos=target_wrist_pos_full,
            target_quat=target_wrist_quat_full,
            dofs_idx_local=spec.arm_ik_dof_indices.tolist(),
            init_qpos=current_qpos,
            init_qpos_dofs_idx_local=spec.hand.dof_map,
        )
        ik_expected_shape = (self.env.num_envs, len(spec.hand.dof_map))
        if tuple(ik_qpos.shape) != ik_expected_shape:
            raise ValueError(
                "Reset IK qpos shape mismatch: expected "
                f"{ik_expected_shape}, got {tuple(ik_qpos.shape)}."
            )
        ik_qpos = ik_qpos.to(device=position_targets.device)
        position_targets[:, spec.arm_joint_indices] = ik_qpos[:, spec.arm_joint_indices]
        position_targets[:, spec.hand_target_indices] = target_hand_qpos_full

        self._interpolate_joint_targets(
            spec=spec,
            phase_name=f"reset phase ({side})",
            current_qpos=current_qpos,
            start_position_targets=current_qpos,
            target_position_targets=position_targets,
            final_hold_steps=10,
        )
