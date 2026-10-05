from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from dexlink.src.rl.sim.robot import Hand

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import ActionCfg
    from dexlink.src.rl.sim.art_object import ArtObject
    from dexlink.src.rl.sim.env import Env


class ActionManager:
    def __init__(self, cfg: ActionCfg, env: Env):
        self.cfg = cfg
        self.env = env

        self.action_mode = cfg.mode
        self.scale = cfg.scale
        self.trans_scale = cfg.trans_scale
        self.rot_scale = cfg.rot_scale
        self.sim_dt = max(float(env.cfg.scene_cfg.dt), 1e-6)

        self.action_terms = []
        self.term_dims = []
        self.hand_joint_scales: dict[str, torch.Tensor] = {}
        if cfg.control_hands and (not env.cfg.scene_cfg.object_only_scene):
            for side in self.env.scene_manager.active_hand_sides:
                self.action_terms.append(side)
                hand = self.env.scene_manager.scene_entities[side]
                hand_scales = torch.full(
                    (len(hand.joints),),
                    float(self.cfg.rot_scale),
                    dtype=torch.float32,
                )
                if len(hand.root_joints_idx) > 0:
                    hand_scales[hand.root_joints_idx[:3]] = float(self.cfg.trans_scale)
                self.hand_joint_scales[hand.name] = hand_scales
                self.term_dims.append(hand.act_dim(cfg))

        if cfg.control_object:
            # also set object gains here
            object_entity: ArtObject = self.env.scene_manager.scene_entities["object"]
            if object_entity.actuated:
                self.action_terms.append("object")
                object_entity.set_gains()
                self.term_dims.append(len(object_entity.dof_map))  # pos

    def sample_random_action(self, object_pos, env_ids):
        # sample actions for hands given object pos
        lower, upper = 0.2, 0.3
        # Create coordinated signs so left/right move oppositely in x/y
        reference_x = object_pos[:, 0]
        reference_y = object_pos[:, 1]
        signs = torch.stack(
            [
                torch.randint(0, 2, reference_x.shape, device=reference_x.device) * 2
                - 1,
                torch.randint(0, 2, reference_y.shape, device=reference_y.device) * 2
                - 1,
                torch.ones_like(reference_x),  # z kept as-is
            ],
            dim=1,
        ).float()
        random_actions = {}
        for side in self.env.scene_manager.active_hand_sides:
            hand: Hand = self.env.scene_manager.scene_entities[side]
            side_signs = signs if side == "left" else -signs
            random_actions[side] = hand.sample_actions_near_object(
                obj_pos=object_pos,
                env_ids=env_ids,
                lower=lower,
                upper=upper,
                opposite_signs=side_signs,
            )
        return random_actions

    def initialize(self):
        pass

    def summary(self) -> str:
        spacing = 20
        lines = [
            (
                f"[ActionManager] mode: {self.action_mode}, "
                f"trans_scale: {self.trans_scale}, rot_scale: {self.rot_scale}, "
                f"object_scale: {self.scale}"
            ),
            f"{'Action Term':<{spacing}} {'Dimension':<10}",
        ]
        lines.append("-" * (spacing + 10))
        for term, dim in zip(self.action_terms, self.term_dims):
            lines.append(f"{term:<{spacing}} {dim:<10}")
        lines.append("-" * (spacing + 10))
        lines.append(f"{'Total':<{spacing}} {self.action_dim:<10}")
        return "\n".join(lines)

    def named_view(self, actions: torch.Tensor) -> dict:
        """Return a dictionary mapping action terms to their respective tensors."""
        splits = torch.split(actions, self.term_dims, dim=-1)
        named_splits = dict(zip(self.action_terms, splits))
        return named_splits

    # # the following functionality is captured by hand.control_robot class
    # def process_hand_actions(self, hand: Hand, action):
    #     if self.action_mode == "diff":
    #         # take the current jpos and add the action diff
    #         # the first dof dims are jpos and the rest are jvel
    #         jpos = hand.get_obs()[:, : len(hand.dof_map)]  # (n, dof)
    #         action = self.scale * action + jpos
    #     return action

    def process_object_actions(self, object_entity: ArtObject, action):
        if len(object_entity.dof_map) == 0:
            return action

        if self.action_mode == "diff":
            # take the current jpos and add the action diff
            object_obs = object_entity.get_obs()  # (n, 3+rot+[obj_dofs])
            n_joints = len(object_entity.dof_map)
            jpos = object_obs[:, -n_joints:]  # joints are at the end
            action = self.scale * action + jpos
        return action

    def _retargeted_joint_values(
        self, device: torch.device
    ) -> dict[str, torch.Tensor | None]:
        retargeted_motion = self.env.scene_manager.original_demo.retargeted_motion
        use_retarget_base = self.cfg.mode == "residual" or self.cfg.obj_dex_actions
        if use_retarget_base:
            assert (
                retargeted_motion is not None
            ), "Retargeted motion must be available for residual/objdex control."
            target_frame_indices = self.expected_base_action_timesteps()
            jvals_at_command = {}
            for side in self.env.scene_manager.active_hand_sides:
                frame_jvals = getattr(self.env.reset_manager, f"frame_{side}_jvals")
                jvals_at_command[side] = frame_jvals.to(device)[
                    target_frame_indices, ...
                ]
            return jvals_at_command
        return {side: None for side in self.env.scene_manager.active_hand_sides}

    def _require_retargeted_joint_values(
        self,
        cmd_jval_from_retgt: torch.Tensor | None,
        *,
        context: str,
    ) -> torch.Tensor:
        if cmd_jval_from_retgt is None:
            raise ValueError(
                f"{context} requires retargeted joint values for the current command timestep."
            )
        return cmd_jval_from_retgt

    def _get_hand_dof_positions(
        self,
        hand: Hand,
        dof_indices: list[int] | np.ndarray,
    ) -> torch.Tensor:
        return hand.robot.get_dofs_position(dof_indices).detach()

    def _get_hand_dof_velocities(
        self,
        hand: Hand,
        dof_indices: list[int] | np.ndarray,
    ) -> torch.Tensor:
        return hand.robot.get_dofs_velocity(dof_indices).detach()

    def _clip_hand_joint_targets(
        self,
        hand: Hand,
        targets: torch.Tensor,
        dof_indices: list[int] | np.ndarray,
    ) -> torch.Tensor:
        lower_limit, upper_limit = hand.robot.get_dofs_limit(dofs_idx_local=dof_indices)
        return torch.clip(targets, lower_limit, upper_limit)

    def _resolve_position_targets(
        self,
        hand: Hand,
        action: torch.Tensor,
        *,
        dof_indices: list[int] | np.ndarray,
        joint_indices: list[int] | np.ndarray | None = None,
        cmd_jval_from_retgt: torch.Tensor | None = None,
        residual_context: str,
    ) -> torch.Tensor:
        if self.cfg.mode == "abs":
            return action

        scales = self.hand_joint_scales[hand.name].to(
            device=action.device, dtype=action.dtype
        )
        if joint_indices is not None:
            joint_indices = np.asarray(joint_indices, dtype=np.int64)
            scales = scales[
                torch.as_tensor(joint_indices, device=action.device, dtype=torch.long)
            ]
        scaled_action = action * scales.unsqueeze(0)
        if self.cfg.mode == "diff":
            return scaled_action + self._get_hand_dof_positions(hand, dof_indices)
        if self.cfg.mode == "residual":
            base_targets = self._require_retargeted_joint_values(
                cmd_jval_from_retgt,
                context=residual_context,
            )
            return scaled_action + base_targets

        raise ValueError(f"Invalid action mode: {self.cfg.mode}")

    def _obj_dex_position_targets(
        self,
        hand: Hand,
        action: torch.Tensor,
        cmd_jval_from_retgt: torch.Tensor | None,
    ) -> torch.Tensor:
        base_targets = self._require_retargeted_joint_values(
            cmd_jval_from_retgt,
            context="obj_dex_actions",
        )
        target = action.clone()
        current_jpos = self._get_hand_dof_positions(hand, hand.dof_map)

        root_idx = torch.as_tensor(
            hand.root_joints_idx, dtype=torch.long, device=target.device
        )
        non_root_mask = torch.ones(
            len(hand.joints), dtype=torch.bool, device=target.device
        )
        non_root_mask[root_idx] = False

        trans_root_idx = torch.as_tensor(
            hand.root_joints_idx[:3], dtype=torch.long, device=target.device
        )
        rot_root_idx = torch.as_tensor(
            hand.root_joints_idx[3:], dtype=torch.long, device=target.device
        )

        if trans_root_idx.numel() > 0:
            target[:, trans_root_idx] = (
                base_targets[:, trans_root_idx]
                + float(self.cfg.obj_dex_trans_scale) * action[:, trans_root_idx]
            )
        if rot_root_idx.numel() > 0:
            target[:, rot_root_idx] = (
                base_targets[:, rot_root_idx]
                + float(self.cfg.obj_dex_rot_scale) * action[:, rot_root_idx]
            )
        target[:, non_root_mask] = (
            current_jpos[:, non_root_mask] + action[:, non_root_mask]
        )

        return self._clip_hand_joint_targets(hand, target, hand.dof_map)

    def _residual_velocity_targets(
        self,
        hand: Hand,
        velocity_targets: torch.Tensor,
        *,
        dof_indices: list[int] | np.ndarray,
        cmd_jval_from_retgt: torch.Tensor | None,
        residual_context: str,
        velocity_mask: np.ndarray | None = None,
    ) -> torch.Tensor:
        if self.cfg.mode != "residual" or velocity_targets.shape[1] == 0:
            return velocity_targets

        base_targets = self._require_retargeted_joint_values(
            cmd_jval_from_retgt,
            context=residual_context,
        )
        current_jpos = self._get_hand_dof_positions(hand, dof_indices)
        target_jpos = base_targets
        if velocity_mask is not None:
            target_jpos = target_jpos[:, velocity_mask]
        expected_motion = target_jpos - current_jpos
        predicted_motion = velocity_targets * self.sim_dt
        corrected_motion = expected_motion + predicted_motion
        return corrected_motion / self.sim_dt

    def process_hand_actions(
        self,
        hand: Hand,
        action: torch.Tensor,
        cmd_jval_from_retgt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | np.ndarray | str]:
        expected_shape = (hand.num_envs, len(hand.joints))
        if action.shape != expected_shape:
            raise ValueError(
                f"Hand action shape mismatch for {hand.name}: expected {expected_shape}, got {tuple(action.shape)}."
            )

        dofs_idx = np.asarray(hand.dof_map)

        if self.cfg.obj_dex_actions:
            return {
                "controller": "position",
                "position_targets": self._obj_dex_position_targets(
                    hand, action, cmd_jval_from_retgt
                ),
                "position_dofs": dofs_idx,
            }

        controller = self.cfg.controller
        if controller == "position":
            position_targets = self._resolve_position_targets(
                hand,
                action,
                dof_indices=hand.dof_map,
                cmd_jval_from_retgt=cmd_jval_from_retgt,
                residual_context="Residual position control",
            )
            position_targets = self._clip_hand_joint_targets(
                hand, position_targets, hand.dof_map
            )
            return {
                "controller": "position",
                "position_targets": position_targets,
                "position_dofs": dofs_idx,
            }

        if controller == "velocity":
            velocity_scales = self.hand_joint_scales[hand.name].to(
                device=action.device, dtype=action.dtype
            )
            velocity_targets = action * velocity_scales.unsqueeze(0)
            velocity_targets = self._residual_velocity_targets(
                hand,
                velocity_targets,
                dof_indices=hand.dof_map,
                cmd_jval_from_retgt=cmd_jval_from_retgt,
                residual_context="Residual velocity control",
            )
            return {
                "controller": "velocity",
                "velocity_targets": velocity_targets,
                "velocity_dofs": dofs_idx,
            }

        if controller == "hybrid":
            pos_joint_indices = np.flatnonzero(hand.position_control_mask)
            pos_dof_indices = dofs_idx[hand.position_control_mask]
            pos_vals = action[:, hand.position_control_mask]
            pos_cmd_jvals = None
            if cmd_jval_from_retgt is not None:
                pos_cmd_jvals = cmd_jval_from_retgt[:, hand.position_control_mask]
            pos_vals = self._resolve_position_targets(
                hand,
                pos_vals,
                dof_indices=pos_dof_indices,
                joint_indices=pos_joint_indices,
                cmd_jval_from_retgt=pos_cmd_jvals,
                residual_context="Residual hybrid control",
            )
            pos_vals = self._clip_hand_joint_targets(hand, pos_vals, pos_dof_indices)

            vel_joint_indices = np.flatnonzero(hand.velocity_control_mask)
            vel_dof_indices = dofs_idx[hand.velocity_control_mask]
            velocity_scales = self.hand_joint_scales[hand.name].to(
                device=action.device, dtype=action.dtype
            )
            velocity_scales = velocity_scales[
                torch.as_tensor(
                    vel_joint_indices, device=action.device, dtype=torch.long
                )
            ]
            vel_vals = action[
                :, hand.velocity_control_mask
            ] * velocity_scales.unsqueeze(0)

            vel_vals = self._residual_velocity_targets(
                hand,
                vel_vals,
                dof_indices=vel_dof_indices,
                cmd_jval_from_retgt=cmd_jval_from_retgt,
                residual_context="Residual hybrid control",
                velocity_mask=hand.velocity_control_mask,
            )

            return {
                "controller": "hybrid",
                "position_targets": pos_vals,
                "position_dofs": pos_dof_indices,
                "velocity_targets": vel_vals,
                "velocity_dofs": vel_dof_indices,
            }

        if controller == "impedance":
            pos_target = action[:, : len(hand.joints)]
            next_idx = len(hand.joints)

            stiffness = None
            if self.cfg.compute_stiffness:
                stiffness = torch.exp(action[:, next_idx : next_idx + len(hand.joints)])
                next_idx += len(hand.joints)

            damping = None
            if self.cfg.compute_damping_ratio:
                damping_ratio = torch.exp(
                    action[:, next_idx : next_idx + len(hand.joints)]
                )
                kp = (
                    stiffness
                    if stiffness is not None
                    else hand.robot.get_dofs_kp(dofs_idx_local=hand.dof_map)
                )
                damping = 2 * torch.sqrt(kp) * damping_ratio

            pos_target = self._resolve_position_targets(
                hand,
                pos_target,
                dof_indices=hand.dof_map,
                cmd_jval_from_retgt=cmd_jval_from_retgt,
                residual_context="Residual impedance control",
            )
            pos_target = self._clip_hand_joint_targets(hand, pos_target, hand.dof_map)

            kp = stiffness
            if kp is None:
                kp = hand.robot.get_dofs_kp(dofs_idx_local=hand.dof_map)
            kv = damping
            if kv is None:
                kv = hand.robot.get_dofs_kv(dofs_idx_local=hand.dof_map)

            dof_pos_error = pos_target - self._get_hand_dof_positions(
                hand, hand.dof_map
            )
            dof_vel_error = -self._get_hand_dof_velocities(hand, hand.dof_map)
            desired_force = kp * dof_pos_error + kv * dof_vel_error
            force_lower_limit, force_upper_limit = hand.robot.get_dofs_force_range(
                dofs_idx_local=hand.dof_map
            )
            desired_force = torch.clip(
                desired_force, force_lower_limit, force_upper_limit
            )

            processed_action: dict[str, torch.Tensor | np.ndarray | str] = {
                "controller": "impedance",
                "force_targets": desired_force,
                "force_dofs": dofs_idx,
            }
            if stiffness is not None:
                processed_action["stiffness"] = stiffness
            if damping is not None:
                processed_action["damping"] = damping
            return processed_action

        raise ValueError(f"Invalid controller type: {controller}.")

    def expected_base_action_timesteps(self):
        # For each env, map stage-local elapsed steps to an absolute reference timestep.
        # Stage start is inferred from current command horizon bookkeeping:
        #   stage_start = command_time_step - n_step_future_cmd
        # Then progress from stage_start toward command_time_step over the stage horizon.
        elapsed = self.env.timesteps.to(dtype=torch.long)
        horizon = self.env.n_step_future_cmd.to(dtype=torch.long)
        command_ts = self.env.command_time_step.to(dtype=torch.long)

        stage_start = command_ts - horizon
        elapsed_clamped = torch.clamp(
            elapsed, min=torch.zeros_like(elapsed), max=horizon
        )
        expected_ts = stage_start + elapsed_clamped

        # # Safety clamp to valid trajectory range.
        # expected_ts = torch.clamp(
        #     expected_ts, min=0, max=self.env.scene_manager.total_steps - 1
        # )
        return expected_ts

    def apply_actions(self, actions):
        named_actions = self.named_view(actions)

        device = actions.device
        jvals_at_command = self._retargeted_joint_values(device)

        for action_name, act in named_actions.items():
            if action_name in ["left", "right"]:
                hand: Hand = self.env.scene_manager.scene_entities[action_name]
                processed_action = self.process_hand_actions(
                    hand,
                    act,
                    cmd_jval_from_retgt=jvals_at_command[action_name],
                )
                hand.apply_processed_action(processed_action)
            elif action_name == "object":
                object_entity: ArtObject = self.env.scene_manager.scene_entities[
                    "object"
                ]
                _action = self.process_object_actions(object_entity, act)
                object_entity.control_object(_action)
            else:
                raise ValueError(f"Unknown action name: {action_name}")

        self.env.scene_manager.step()

    @property
    def action_dim(self) -> int:
        return sum(self.term_dims)
