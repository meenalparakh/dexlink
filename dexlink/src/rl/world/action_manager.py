from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from dexlink.utils.retargeting.math_utils import (
    quat_from_matrix,
    quat_from_rotvec,
    quat_mul,
)
from dexlink.src.rl.world.robot import Hand

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import ActionCfg
    from dexlink.src.rl.world.env import Env


@dataclass(frozen=True)
class HandActionSpec:
    side: str
    hand: Hand
    arm_joint_indices: np.ndarray
    arm_ik_dof_indices: np.ndarray
    hand_dof_indices: np.ndarray
    hand_target_indices: np.ndarray
    hand_delta_scale_factors: torch.Tensor

    @property
    def action_dim(self) -> int:
        return 6 + len(self.hand_dof_indices)


class ActionManagerBase:
    EVAL_BASE_HEIGHT_SAFETY_THRESHOLD_M = 0.01
    REAL_CONTROL_LOG_BLUE = "\033[94m"
    REAL_CONTROL_LOG_RESET = "\033[0m"

    def __init__(self, cfg: ActionCfg, env: Env):
        self.cfg = cfg
        self.env = env
        self.physics = self.env.physics_manager
        self.action_mode = str(self.cfg.action_mode)
        self.residual_beta = float(self.cfg.residual_beta)
        self.decimation = int(self.env.cfg.decimation)
        if self.action_mode not in {"diff", "residual", "absolute_target"}:
            raise ValueError(
                "Expected action_cfg.action_mode to be one of {'diff', 'residual', 'absolute_target'}, "
                f"got {self.action_mode!r}."
            )
        if not 0.0 <= self.residual_beta <= 1.0:
            raise ValueError(
                "Expected action_cfg.residual_beta to be in [0, 1], "
                f"got {self.residual_beta}."
            )
        if self.decimation <= 0:
            raise ValueError(
                f"Expected env.decimation to be positive, got {self.decimation}."
            )
        self.position_target_blend_alpha_start = float(
            self.cfg.position_target_blend_alpha_start
        )
        self.position_target_blend_alpha_end = float(
            self.cfg.position_target_blend_alpha_end
        )
        self.real_wris_pos_alpha = float(self.env.real_cfg.wris_pos_alpha)
        self.real_wrist_rot_alpha = float(self.env.real_cfg.wrist_rot_alpha)
        self.real_fingers_pos_alpha = float(self.env.real_cfg.fingers_pos_alpha)
        if not 0.0 <= self.position_target_blend_alpha_start <= 1.0:
            raise ValueError(
                "Expected action_cfg.position_target_blend_alpha_start to be in [0, 1], "
                f"got {self.position_target_blend_alpha_start}."
            )
        if not 0.0 <= self.position_target_blend_alpha_end <= 1.0:
            raise ValueError(
                "Expected action_cfg.position_target_blend_alpha_end to be in [0, 1], "
                f"got {self.position_target_blend_alpha_end}."
            )
        if (
            self.position_target_blend_alpha_end
            < self.position_target_blend_alpha_start
        ):
            raise ValueError(
                "Expected action_cfg.position_target_blend_alpha_end to be greater than or equal to "
                "action_cfg.position_target_blend_alpha_start, got "
                f"{self.position_target_blend_alpha_end} < {self.position_target_blend_alpha_start}."
            )
        if (
            self.position_target_blend_alpha_start == 0.0
            and self.position_target_blend_alpha_end != 0.0
        ):
            raise ValueError(
                "Exponential position target blending requires "
                "action_cfg.position_target_blend_alpha_start > 0 when "
                "action_cfg.position_target_blend_alpha_end is non-zero."
            )
        if not 0.0 <= self.real_wris_pos_alpha <= 1.0:
            raise ValueError(
                "Expected real_cfg.wris_pos_alpha to be in [0, 1], "
                f"got {self.real_wris_pos_alpha}."
            )
        if not 0.0 <= self.real_wrist_rot_alpha <= 1.0:
            raise ValueError(
                "Expected real_cfg.wrist_rot_alpha to be in [0, 1], "
                f"got {self.real_wrist_rot_alpha}."
            )
        if not 0.0 <= self.real_fingers_pos_alpha <= 1.0:
            raise ValueError(
                "Expected real_cfg.fingers_pos_alpha to be in [0, 1], "
                f"got {self.real_fingers_pos_alpha}."
            )

        if self.env.cfg.task_cfg.object_only_scene:
            raise ValueError(
                "ActionManager requires a robot scene, but task_cfg.object_only_scene is enabled."
            )
        active_hand_sides = tuple(self.env.task_manager.active_hand_sides)
        if len(active_hand_sides) == 0:
            raise ValueError(
                "ActionManager requires at least one active hand with wrist control joints and wrist IK."
            )
        if self.env.cfg.backend == "real" and active_hand_sides != ("right",):
            raise ValueError(
                "The real backend supports only a single active right hand, got "
                f"{active_hand_sides}."
            )
        self.hand_specs: dict[str, HandActionSpec] = {}
        for side in active_hand_sides:
            hand: Hand = self.env.physics_manager.scene_entities[side]
            if len(hand.wrist_control_joint_idx) == 0:
                raise ValueError(
                    f"Expected hand {side!r} ('{hand.name}') to include wrist control joints, but none were found."
                )
            if len(hand.hand_joint_idx) == 0:
                raise ValueError(
                    f"Expected hand {side!r} ('{hand.name}') to include finger joints outside the wrist control set, but none were found."
                )
            hand_dof_indices = np.asarray(hand.hand_joint_idx, dtype=np.int64)
            self.hand_specs[side] = HandActionSpec(
                side=side,
                hand=hand,
                arm_joint_indices=np.asarray(
                    hand.wrist_control_joint_idx, dtype=np.int64
                ),
                arm_ik_dof_indices=np.asarray(
                    [hand.dof_map[idx] for idx in hand.wrist_control_joint_idx],
                    dtype=np.int64,
                ),
                hand_dof_indices=hand_dof_indices,
                hand_target_indices=np.asarray(
                    [hand.joint_names.index(name) for name in hand.finger_joint_names],
                    dtype=np.int64,
                ),
                hand_delta_scale_factors=self._hand_delta_scale_factors(
                    hand=hand,
                    hand_dof_indices=hand_dof_indices,
                ),
            )
        self.action_terms = list(self.hand_specs)
        self.term_dims = [spec.action_dim for spec in self.hand_specs.values()]

    def initialize(self):
        pass

    def _print_real_control_stage(self, message: str) -> None:
        if self.env.cfg.backend != "real":
            return
        if not self.env.real_cfg.print_real_control_stages:
            return
        print(
            f"{self.REAL_CONTROL_LOG_BLUE}[RealControl] {message}{self.REAL_CONTROL_LOG_RESET}"
        )

    def _single_hand_spec(self) -> HandActionSpec:
        if tuple(self.hand_specs) != ("right",):
            raise ValueError(
                "Single-hand convenience access is only valid for a right-hand-only setup; "
                f"got active hands {tuple(self.hand_specs)}."
            )
        return self.hand_specs["right"]

    @property
    def hand(self) -> Hand:
        return self._single_hand_spec().hand

    @property
    def arm_joint_indices(self) -> np.ndarray:
        return self._single_hand_spec().arm_joint_indices

    @property
    def arm_ik_dof_indices(self) -> np.ndarray:
        return self._single_hand_spec().arm_ik_dof_indices

    @property
    def hand_dof_indices(self) -> np.ndarray:
        return self._single_hand_spec().hand_dof_indices

    @property
    def hand_target_indices(self) -> np.ndarray:
        return self._single_hand_spec().hand_target_indices

    def _hand_delta_scale_factors(
        self, *, hand: Hand, hand_dof_indices: np.ndarray
    ) -> torch.Tensor:
        prismatic_factor = float(self.cfg.hand_prismatic_joint_delta_scale_factor)
        if prismatic_factor <= 0.0:
            raise ValueError(
                "Expected action_cfg.hand_prismatic_joint_delta_scale_factor > 0, "
                f"got {prismatic_factor}."
            )
        factors = []
        for joint_index in hand_dof_indices.tolist():
            joint = hand.joints[joint_index]
            if joint.joint_type == "prismatic":
                factors.append(prismatic_factor)
            elif joint.joint_type in {"revolute", "continuous"}:
                factors.append(1.0)
            else:
                raise ValueError(
                    f"Unsupported hand action joint type {joint.joint_type!r} for "
                    f"joint {joint.name!r}; expected prismatic, revolute, or continuous."
                )
        return torch.as_tensor(factors, dtype=torch.float32, device=self.env.device)

    def summary(self) -> str:
        spacing = 20
        lines = [
            (
                "[ActionManagerBase] delta wrist pose + delta hand joints, "
                f"action_mode: {self.action_mode}, "
                f"residual_beta: {self.residual_beta}, "
                f"env_decimation: {self.decimation}, "
                f"arm_translation_delta_scale: {self.cfg.arm_translation_delta_scale}, "
                f"arm_rotation_delta_scale: {self.cfg.arm_rotation_delta_scale}, "
                f"hand_joint_delta_scale: {self.cfg.hand_joint_delta_scale}, "
                f"hand_prismatic_joint_delta_scale_factor: "
                f"{self.cfg.hand_prismatic_joint_delta_scale_factor}, "
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

    def named_view(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        splits = torch.split(actions, self.term_dims, dim=-1)
        return dict(zip(self.action_terms, splits))

    def _require_current_policy_named_obs(self) -> dict[str, torch.Tensor]:
        named_obs = self.env.current_named_policy_obs
        if named_obs is None:
            raise ValueError(
                "env.current_named_policy_obs is not populated. Observations must be computed before actions are applied."
            )
        return named_obs

    def _require_obs_term(
        self,
        named_obs: dict[str, torch.Tensor],
        key: str,
        expected_last_dim: int,
    ) -> torch.Tensor:
        value = named_obs.get(key)
        if value is None:
            raise ValueError(f"Required policy observation term '{key}' is missing.")
        expected_shape = (self.env.num_envs, expected_last_dim)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Observation term '{key}' shape mismatch: expected {expected_shape}, got {tuple(value.shape)}."
            )
        return value

    def _blend_alpha_for_decimation_step(self, step_idx: int) -> float:
        if step_idx < 0 or step_idx >= self.decimation:
            raise ValueError(
                f"Decimation step index out of range: expected [0, {self.decimation}), got {step_idx}."
            )
        if self.decimation == 1:
            return self.position_target_blend_alpha_end
        progress = step_idx / float(self.decimation - 1)
        start = self.position_target_blend_alpha_start
        end = self.position_target_blend_alpha_end
        if start == end:
            return start
        return start * ((end / start) ** progress)

    def _blend_alpha_for_step(self, step_idx: int, num_steps: int) -> float:
        if num_steps <= 0:
            raise ValueError(f"Expected num_steps > 0, got {num_steps}.")
        if step_idx < 0 or step_idx >= num_steps:
            raise ValueError(
                f"Interpolation step index out of range: expected [0, {num_steps}), got {step_idx}."
            )
        if num_steps == 1:
            return self.position_target_blend_alpha_end
        progress = step_idx / float(num_steps - 1)
        start = self.position_target_blend_alpha_start
        end = self.position_target_blend_alpha_end
        if start == end:
            return start
        return start * ((end / start) ** progress)

    def _interpolation_num_steps(
        self,
        *,
        current_qpos: torch.Tensor,
        target_qpos: torch.Tensor,
    ) -> int:
        if self.env.cfg.backend != "real":
            return self.decimation
        expected_shape = (self.env.num_envs, len(self.hand.dof_map))
        if tuple(current_qpos.shape) != expected_shape:
            raise ValueError(
                f"Expected current_qpos shape {expected_shape}, got {tuple(current_qpos.shape)}."
            )
        if tuple(target_qpos.shape) != expected_shape:
            raise ValueError(
                f"Expected target_qpos shape {expected_shape}, got {tuple(target_qpos.shape)}."
            )
        max_step_deg = float(self.env.real_cfg.adaptive_decimation_joint_step_deg)
        if max_step_deg <= 0.0:
            raise ValueError(
                "Expected real_cfg.adaptive_decimation_joint_step_deg > 0, got "
                f"{max_step_deg}."
            )
        max_delta_rad = torch.max(torch.abs(target_qpos - current_qpos)).item()
        step_rad = math.radians(max_step_deg)
        return max(1, int(math.ceil(max_delta_rad / step_rad)))

    def _reference_joint_targets(self, *, spec: HandActionSpec) -> torch.Tensor:
        frame_jvals = getattr(self.env.reset_manager, f"frame_{spec.side}_jvals", None)
        if frame_jvals is None:
            raise ValueError(
                "Residual action mode requires reset_manager."
                f"frame_{spec.side}_jvals to be populated."
            )
        expected_shape = (self.env.task_manager.total_steps, len(spec.hand.dof_map))
        if tuple(frame_jvals.shape) != expected_shape:
            raise ValueError(
                "Reference hand joint target shape mismatch: expected "
                f"{expected_shape}, got {tuple(frame_jvals.shape)}."
            )
        frame_indices = self.env.command_time_step.to(
            device=frame_jvals.device,
            dtype=torch.long,
        )
        if tuple(frame_indices.shape) != (self.env.num_envs,):
            raise ValueError(
                "Command timestep shape mismatch: expected "
                f"{(self.env.num_envs,)}, got {tuple(frame_indices.shape)}."
            )
        return frame_jvals.index_select(0, frame_indices)

    def _mix_policy_and_reference_targets(
        self, policy_targets: torch.Tensor, *, spec: HandActionSpec
    ) -> torch.Tensor:
        expected_shape = (self.env.num_envs, len(spec.hand.dof_map))
        if tuple(policy_targets.shape) != expected_shape:
            raise ValueError(
                "Policy joint target shape mismatch: expected "
                f"{expected_shape}, got {tuple(policy_targets.shape)}."
            )
        if self.action_mode == "diff":
            return policy_targets
        reference_targets = self._reference_joint_targets(spec=spec).to(
            device=policy_targets.device,
            dtype=policy_targets.dtype,
        )
        return (
            self.residual_beta * policy_targets
            + (1.0 - self.residual_beta) * reference_targets
        )

    def _lock_wrist_targets_to_reference(
        self, position_targets: torch.Tensor, *, spec: HandActionSpec
    ) -> torch.Tensor:
        if not self.cfg.lock_wrist_targets_to_reference:
            return position_targets
        reference_targets = self._reference_joint_targets(spec=spec).to(
            device=position_targets.device,
            dtype=position_targets.dtype,
        )
        locked_targets = position_targets.clone()
        locked_targets[:, spec.arm_joint_indices] = reference_targets[
            :, spec.arm_joint_indices
        ]
        return locked_targets

    def _get_policy_named_obs_for_eval_safety(
        self, *, refresh_from_physics: bool
    ) -> dict[str, torch.Tensor]:
        if refresh_from_physics:
            self.env.observation_manager.reset_step_cache()
            obs = self.env.observation_manager.compute_observations()
            return self.env.observation_manager.named_view_policy(obs["policy"])
        return self._require_current_policy_named_obs()

    def _should_block_eval_action_near_base(
        self, *, refresh_from_physics: bool
    ) -> bool:
        if self.env.mode != "eval":
            return False
        if self.env.cfg.backend != "real":
            return False

        named_policy_obs = self._get_policy_named_obs_for_eval_safety(
            refresh_from_physics=refresh_from_physics
        )
        hand_pcd_rel = self.env.observation_manager.get_hand_pcd_from_named_obs(
            named_policy_obs,
            expected_feat_dim=self.env.observation_manager.policy_hand_pcd_dim,
        )
        current_wrist_pos = self.env.current_wrist_pos_by_side.get("right")
        if current_wrist_pos is None:
            raise ValueError(
                "Evaluation safety check requires env.current_wrist_pos to be populated."
            )
        hand_pcd_world = hand_pcd_rel + current_wrist_pos[:, None, :]
        base_height = float(self.hand.base_pose[2, 3])
        min_height_delta = torch.abs(hand_pcd_world[..., 2] - base_height).amin(dim=1)
        threshold = float(self.EVAL_BASE_HEIGHT_SAFETY_THRESHOLD_M)
        blocked_env_ids = torch.nonzero(min_height_delta <= threshold, as_tuple=False)
        if blocked_env_ids.numel() == 0:
            return False

        blocked_ids = blocked_env_ids.squeeze(-1).detach().cpu().tolist()
        min_delta_cm = (min_height_delta[blocked_ids] * 100.0).detach().cpu().tolist()
        print(
            "Action can't be executed as the hand has come within 5cm of robot base height. "
            f"blocked_env_ids={blocked_ids}, min_height_delta_cm={min_delta_cm}"
        )
        return True

    def _to_real_control_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device="cpu")

    def _object_align_real_eval_wrist_target(
        self, *, spec: HandActionSpec, target_wrist_pos: torch.Tensor
    ) -> torch.Tensor:
        if self.env.cfg.backend != "real" or self.env.mode != "eval":
            return target_wrist_pos
        align_fn = getattr(
            self.env.reset_manager,
            "real_eval_align_wrist_position_to_object_anchor",
            None,
        )
        if align_fn is None:
            raise AttributeError(
                "Real eval wrist target object alignment requires "
                "reset_manager.real_eval_align_wrist_position_to_object_anchor()."
            )
        return align_fn(
            side=spec.side,
            wrist_pos=target_wrist_pos,
            frame_idx=self.env.command_time_step,
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
        expected_shape = (self.env.num_envs, spec.action_dim)
        if tuple(action.shape) != expected_shape:
            raise ValueError(
                f"{spec.side!r} hand action shape mismatch: expected {expected_shape}, got {tuple(action.shape)}."
            )

        named_obs = self._require_current_policy_named_obs()
        current_qpos = self._require_obs_term(
            named_obs,
            key=f"{spec.side}_jpos",
            expected_last_dim=len(spec.hand.dof_map),
        )
        current_arm_jpos = self.env.current_wrist_control_jpos_by_side.get(spec.side)
        if current_arm_jpos is None:
            raise ValueError(
                f"Action computation requires current wrist-control jpos for hand {spec.side!r}."
            )
        current_qpos = current_qpos.clone()
        current_qpos[:, spec.arm_joint_indices] = current_arm_jpos
        current_wrist_pos = self.env.current_wrist_pos_by_side.get(spec.side)
        if current_wrist_pos is None:
            raise ValueError(
                f"Action computation requires current wrist position for hand {spec.side!r}."
            )
        current_wrist_pose = self._require_obs_term(
            named_obs,
            key=f"{spec.side}_wrist_pose",
            expected_last_dim=9,
        )

        wrist_delta = action[:, :6]
        hand_delta = action[:, 6:]

        if self.env.cfg.backend == "real":
            current_qpos = self._to_real_control_tensor(current_qpos)
            current_wrist_pos = self._to_real_control_tensor(current_wrist_pos)
            current_wrist_pose = self._to_real_control_tensor(current_wrist_pose)
            wrist_delta = self._to_real_control_tensor(wrist_delta)
            hand_delta = self._to_real_control_tensor(hand_delta)

        wrist_pos_delta_scale = float(self.cfg.arm_translation_delta_scale)
        wrist_rot_delta_scale = float(self.cfg.arm_rotation_delta_scale)
        hand_delta_scale = float(self.cfg.hand_joint_delta_scale)
        hand_delta_scale_factors = spec.hand_delta_scale_factors
        if self.env.cfg.backend == "real":
            wrist_pos_delta_scale *= self.real_wris_pos_alpha
            wrist_rot_delta_scale *= self.real_wrist_rot_alpha
            hand_delta_scale *= self.real_fingers_pos_alpha
            hand_delta_scale_factors = self._to_real_control_tensor(
                hand_delta_scale_factors
            )

        target_wrist_pos = current_wrist_pos + (
            wrist_delta[:, :3] * wrist_pos_delta_scale
        )
        target_wrist_pos = self._object_align_real_eval_wrist_target(
            spec=spec,
            target_wrist_pos=target_wrist_pos,
        )
        wrist_x_axis = current_wrist_pose[:, 3:6]
        wrist_y_axis = current_wrist_pose[:, 6:9]
        wrist_z_axis = torch.cross(wrist_x_axis, wrist_y_axis, dim=-1)
        current_wrist_quat = quat_from_matrix(
            torch.stack([wrist_x_axis, wrist_y_axis, wrist_z_axis], dim=-1)
        )
        delta_rotvec = wrist_delta[:, 3:] * wrist_rot_delta_scale
        delta_quat = quat_from_rotvec(delta_rotvec)
        target_wrist_quat = quat_mul(current_wrist_quat, delta_quat)

        ik_qpos = self.physics.solve_entity_ik(
            spec.hand.robot,
            link_name=spec.hand.wrist_link_name,
            target_pos=target_wrist_pos,
            target_quat=target_wrist_quat,
            dofs_idx_local=spec.arm_ik_dof_indices.tolist(),
            init_qpos=current_qpos,
            init_qpos_dofs_idx_local=spec.hand.dof_map,
        )
        ik_expected_shape = (self.env.num_envs, len(spec.hand.dof_map))
        if tuple(ik_qpos.shape) != ik_expected_shape:
            raise ValueError(
                f"IK qpos shape mismatch: expected {ik_expected_shape}, got {tuple(ik_qpos.shape)}."
            )

        position_targets = current_qpos.clone()
        position_targets[:, spec.arm_joint_indices] = ik_qpos[:, spec.arm_joint_indices]
        position_targets[:, spec.hand_dof_indices] = current_qpos[
            :, spec.hand_dof_indices
        ] + hand_delta * hand_delta_scale * hand_delta_scale_factors.unsqueeze(0)
        position_targets = self._mix_policy_and_reference_targets(
            position_targets, spec=spec
        )
        position_targets = self._lock_wrist_targets_to_reference(
            position_targets, spec=spec
        )

        lower_limit, upper_limit = self.physics.get_hand_dof_limits(
            spec.hand, spec.hand.dof_map
        )
        if self.env.cfg.backend == "real":
            lower_limit = self._to_real_control_tensor(lower_limit)
            upper_limit = self._to_real_control_tensor(upper_limit)
        position_targets = torch.clip(position_targets, lower_limit, upper_limit)
        hand_target_qpos = position_targets[:, spec.hand_target_indices]
        return (
            position_targets,
            current_qpos,
            current_wrist_pos,
            current_wrist_quat,
            target_wrist_pos,
            target_wrist_quat,
            hand_target_qpos,
        )

    def _record_action_probe(
        self,
        *,
        spec: HandActionSpec,
        action: torch.Tensor,
        start_qpos: torch.Tensor,
        target_qpos: torch.Tensor,
        start_wrist_pos: torch.Tensor,
        start_wrist_quat: torch.Tensor,
        target_wrist_pos: torch.Tensor,
        target_wrist_quat: torch.Tensor,
        hand_target_qpos: torch.Tensor,
        alpha_schedule: list[float],
    ) -> None:
        achieved_qpos = self.physics.get_hand_dof_positions(
            spec.hand, spec.hand.dof_map
        ).detach()
        achieved_wrist_pos = self.physics.get_entity_link_pos(
            spec.hand.robot,
            link_name=spec.hand.wrist_link_name,
            ref="link_origin",
        ).squeeze(1)
        achieved_wrist_quat = self.physics.get_entity_link_quat(
            spec.hand.robot,
            link_name=spec.hand.wrist_link_name,
        ).squeeze(1)
        probe = {
            "action": action.detach().clone(),
            "start_qpos": start_qpos.detach().clone(),
            "target_qpos": target_qpos.detach().clone(),
            "achieved_qpos": achieved_qpos.clone(),
            "start_wrist_pos": start_wrist_pos.detach().clone(),
            "start_wrist_quat": start_wrist_quat.detach().clone(),
            "target_wrist_pos": target_wrist_pos.detach().clone(),
            "target_wrist_quat": target_wrist_quat.detach().clone(),
            "achieved_wrist_pos": achieved_wrist_pos.detach().clone(),
            "achieved_wrist_quat": achieved_wrist_quat.detach().clone(),
            "hand_target_qpos": hand_target_qpos.detach().clone(),
            "side": spec.side,
            "achieved_hand_qpos": achieved_qpos[:, spec.hand_target_indices].clone(),
            "joint_names": list(spec.hand.joint_names),
            "arm_joint_indices": spec.arm_joint_indices.tolist(),
            "hand_joint_indices": spec.hand_dof_indices.tolist(),
            "hand_target_indices": spec.hand_target_indices.tolist(),
            "alpha_schedule": [float(alpha) for alpha in alpha_schedule],
        }
        self.env.current_action_probe_by_side[spec.side] = probe
        if len(self.hand_specs) == 1:
            self.env.current_action_probe = probe
        else:
            self.env.current_action_probe = self.env.current_action_probe_by_side
        self.env.extras[f"log/action_l2_{spec.side}"] = torch.linalg.vector_norm(
            action, dim=-1
        )
        self.env.extras[f"log/action_abs_mean_{spec.side}"] = torch.mean(
            torch.abs(action), dim=-1
        )
        self.env.extras[f"log/action_wrist_l2_{spec.side}"] = torch.linalg.vector_norm(
            action[:, :6], dim=-1
        )
        self.env.extras[f"log/action_finger_l2_{spec.side}"] = torch.linalg.vector_norm(
            action[:, 6:], dim=-1
        )

    def apply_actions(self, actions: torch.Tensor):
        raise NotImplementedError(
            "ActionManagerBase.apply_actions must be implemented by subclasses."
        )

    @property
    def action_dim(self) -> int:
        return sum(self.term_dims)
