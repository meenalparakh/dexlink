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

        self.action_terms = []
        self.term_dims = []
        if cfg.control_hands and (not env.cfg.scene_cfg.object_only_scene):
            for side in self.env.scene_manager.active_hand_sides:
                self.action_terms.append(side)
                hand = self.env.scene_manager.scene_entities[side]
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

    def _joint_action_scales(
        self,
        hand: Hand,
        joint_indices: list[int] | np.ndarray | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if joint_indices is None:
            joint_indices = np.arange(len(hand.joints))
        joint_indices = np.asarray(joint_indices, dtype=np.int64)
        scales = torch.full(
            (joint_indices.shape[0],),
            float(self.cfg.rot_scale),
            device=device,
            dtype=dtype,
        )
        if joint_indices.size == 0:
            return scales

        trans_joint_indices = np.asarray(hand.root_joints_idx[:3], dtype=np.int64)
        trans_mask = np.isin(joint_indices, trans_joint_indices)
        if np.any(trans_mask):
            scales[torch.as_tensor(trans_mask, device=device, dtype=torch.bool)] = (
                float(self.cfg.trans_scale)
            )
        return scales

    def _scale_joint_actions(
        self,
        hand: Hand,
        jvals: torch.Tensor,
        joint_indices: list[int] | np.ndarray | None = None,
    ) -> torch.Tensor:
        scales = self._joint_action_scales(
            hand,
            joint_indices,
            device=jvals.device,
            dtype=jvals.dtype,
        )
        return jvals * scales.unsqueeze(0)

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
            if cmd_jval_from_retgt is None:
                raise ValueError(
                    "obj_dex_actions requires retargeted joint values for the current command timestep."
                )
            target = action.clone()
            current_jpos = hand.robot.get_dofs_position(hand.dof_map).detach()
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
                    cmd_jval_from_retgt[:, trans_root_idx]
                    + float(self.cfg.obj_dex_trans_scale) * action[:, trans_root_idx]
                )
            if rot_root_idx.numel() > 0:
                target[:, rot_root_idx] = (
                    cmd_jval_from_retgt[:, rot_root_idx]
                    + float(self.cfg.obj_dex_rot_scale) * action[:, rot_root_idx]
                )
            target[:, non_root_mask] = (
                current_jpos[:, non_root_mask] + action[:, non_root_mask]
            )

            lower_limit, upper_limit = hand.robot.get_dofs_limit(
                dofs_idx_local=hand.dof_map
            )
            target = torch.clip(target, lower_limit, upper_limit)
            return {
                "controller": "position",
                "position_targets": target,
                "position_dofs": dofs_idx,
            }

        controller = self.cfg.controller
        if controller == "position":
            if self.cfg.mode == "residual":
                if cmd_jval_from_retgt is None:
                    raise ValueError(
                        "Residual position control requires retargeted joint values."
                    )
                position_targets = (
                    self._scale_joint_actions(hand, action) + cmd_jval_from_retgt
                )
            elif self.cfg.mode == "diff":
                current_jpos = hand.robot.get_dofs_position(hand.dof_map).detach()
                position_targets = (
                    self._scale_joint_actions(hand, action) + current_jpos
                )
            elif self.cfg.mode == "abs":
                position_targets = action
            else:
                raise ValueError(f"Invalid action mode: {self.cfg.mode}")

            lower_limit, upper_limit = hand.robot.get_dofs_limit(
                dofs_idx_local=hand.dof_map
            )
            position_targets = torch.clip(position_targets, lower_limit, upper_limit)
            return {
                "controller": "position",
                "position_targets": position_targets,
                "position_dofs": dofs_idx,
            }

        if controller == "velocity":
            velocity_targets = self._scale_joint_actions(hand, action)
            if self.cfg.mode == "residual":
                if cmd_jval_from_retgt is None:
                    raise ValueError(
                        "Residual velocity control requires retargeted joint values."
                    )
                cmd_jvel_from_retgt = (
                    hand.robot.get_dofs_position(hand.dof_map).detach()
                    - cmd_jval_from_retgt
                )
                velocity_targets = velocity_targets + cmd_jvel_from_retgt
            return {
                "controller": "velocity",
                "velocity_targets": velocity_targets,
                "velocity_dofs": dofs_idx,
            }

        if controller == "hybrid":
            pos_joint_indices = np.flatnonzero(hand.position_control_mask)
            pos_dof_indices = dofs_idx[hand.position_control_mask]
            pos_vals = action[:, hand.position_control_mask]

            if self.cfg.mode == "diff":
                current_pos = hand.robot.get_dofs_position(pos_dof_indices).detach()
                pos_vals = (
                    self._scale_joint_actions(hand, pos_vals, pos_joint_indices)
                    + current_pos
                )
            elif self.cfg.mode == "residual":
                if cmd_jval_from_retgt is None:
                    raise ValueError(
                        "Residual hybrid control requires retargeted joint values."
                    )
                pos_vals = (
                    self._scale_joint_actions(hand, pos_vals, pos_joint_indices)
                    + cmd_jval_from_retgt[:, hand.position_control_mask]
                )
            elif self.cfg.mode != "abs":
                raise ValueError(f"Invalid action mode: {self.cfg.mode}")

            lower_limit, upper_limit = hand.robot.get_dofs_limit(
                dofs_idx_local=pos_dof_indices
            )
            pos_vals = torch.clip(pos_vals, lower_limit, upper_limit)

            vel_joint_indices = np.flatnonzero(hand.velocity_control_mask)
            vel_dof_indices = dofs_idx[hand.velocity_control_mask]
            vel_vals = self._scale_joint_actions(
                hand,
                action[:, hand.velocity_control_mask],
                vel_joint_indices,
            )

            if (
                self.cfg.mode == "residual"
                and cmd_jval_from_retgt is not None
                and vel_vals.shape[1] > 0
            ):
                dt = getattr(hand.scene, "dt", None)
                if dt is None:
                    dt = 1.0 / 100.0
                dt = max(float(dt), 1e-6)

                current_jpos = hand.robot.get_dofs_position(hand.dof_map).detach()[
                    :, hand.velocity_control_mask
                ]
                target_jpos = cmd_jval_from_retgt[:, hand.velocity_control_mask]
                expected_motion = target_jpos - current_jpos

                predicted_motion = vel_vals * dt
                corrected_motion = expected_motion + predicted_motion
                vel_vals = corrected_motion / dt

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

            if self.cfg.mode == "diff":
                pos_target = (
                    self._scale_joint_actions(hand, pos_target)
                    + hand.robot.get_dofs_position(hand.dof_map).detach()
                )
            elif self.cfg.mode == "residual":
                if cmd_jval_from_retgt is None:
                    raise ValueError(
                        "Residual impedance control requires retargeted joint values."
                    )
                pos_target = self._scale_joint_actions(hand, pos_target) + (
                    cmd_jval_from_retgt
                )
            elif self.cfg.mode != "abs":
                raise ValueError(f"Invalid action mode: {self.cfg.mode}")

            lower_limit, upper_limit = hand.robot.get_dofs_limit(
                dofs_idx_local=hand.dof_map
            )
            pos_target = torch.clip(pos_target, lower_limit, upper_limit)

            kp = stiffness
            if kp is None:
                kp = hand.robot.get_dofs_kp(dofs_idx_local=hand.dof_map)
            kv = damping
            if kv is None:
                kv = hand.robot.get_dofs_kv(dofs_idx_local=hand.dof_map)

            dof_pos_error = (
                pos_target - hand.robot.get_dofs_position(hand.dof_map).detach()
            )
            dof_vel_error = -hand.robot.get_dofs_velocity(hand.dof_map).detach()
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
