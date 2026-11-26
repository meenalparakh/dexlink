"""Worker-local scene/reference adapter; a process constructs exactly one Env."""

from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
import torch

from .config import PilotConfig
from .policy import CONDITION_DIM, PolicySchema, local_window


def build_env(config: PilotConfig, variant: dict, *, sequence_scene: bool = False):
    from dexlink.src.rl.config.base import (
        apply_overrides_from_dict,
        load_experiment_config,
    )
    from .scene import CupEnv, CupSequenceEnv

    cfg = load_experiment_config(
        "seq_oakink_residual_target_trajectory_weighted_policy_v3"
    )
    cfg = apply_overrides_from_dict(
        cfg,
        {
            "env.seed": config.seed,
            "env.device": config.env.device,
            "env.task_cfg.oakink_exported_path": variant["reference"],
            "env.sim_cfg.num_envs": config.env.num_envs_per_variant,
            "env.sim_cfg.show_viewer": False,
            "env.sim_cfg.video_save_freq": 0,
            "env.sim_cfg.n_rendered_envs": 0,
            "env.task_cfg.save_reference_video": False,
            "env.reset_cfg.start_range": (0, 0),
            "env.reset_cfg.eval_start_timestep": 0,
            "env.reset_cfg.hand_noise_reach_phase": 0.0,
            "env.reset_cfg.hand_noise_lift_phase": 0.0,
            "env.reset_cfg.reset_timestep_sampling_mode": "uniform",
            "env.curriculum_cfg.use_curriculum": False,
            "env.rewards_cfg.last_reference_hold_steps": 10,
            "env.task_cfg.metric_cfg.metric_id": "object_pose",
            "env.task_cfg.metric_cfg.object_pose_object_id": "cup",
            "env.task_cfg.metric_cfg.object_pose_position_tolerance_m": 0.03,
            "env.task_cfg.metric_cfg.object_pose_orientation_tolerance_rad": math.pi
            / 12,
            "env.task_cfg.lift_support_height": 0.0,
            "env.task_cfg.lift_support_margin": 0.002,
            "env.obs_cfg.rot_format": "quat",
        },
    )
    env_type = CupSequenceEnv if sequence_scene else CupEnv
    return env_type(cfg.env), asdict(cfg.env)


class CupAdapter:
    def __init__(self, env, variant: dict) -> None:
        self.env = env
        self.variant = variant
        if tuple(env.task_manager.active_hand_sides) != ("right",):
            raise ValueError(
                "multi-task pilot requires exactly one active right XHand."
            )
        if tuple(env.task_manager.scene_object_ids) != ("cup", "table"):
            raise ValueError("Expected scene actors in the order cup, table.")
        spec = env.action_manager.hand_specs["right"]
        manager = env.reference_trajectory_manager
        self.reference = torch.cat(
            (
                manager.hand_wrist_pos("right"),
                manager.hand_wrist_quat("right"),
                manager.hand_joint_positions("right")[:, spec.hand_target_indices],
                manager.object_pos(),
                manager.object_quat(),
            ),
            dim=-1,
        )
        if self.reference.shape[0] != variant["frames"]:
            raise ValueError(
                "Loaded reference length disagrees with the frozen variant manifest."
            )
        fingers = len(spec.hand_target_indices)
        self.scales = torch.cat(
            (
                torch.full(
                    (3,),
                    env.cfg.action_cfg.arm_translation_delta_scale,
                    device=env.device,
                ),
                torch.full(
                    (3,), env.cfg.action_cfg.arm_rotation_delta_scale, device=env.device
                ),
                env.cfg.action_cfg.hand_joint_delta_scale
                * spec.hand_delta_scale_factors,
            )
        )
        self.schema = PolicySchema(
            env.observation_dim, self.reference.shape[1], 6 + fingers
        )
        self.durations = torch.tensor(
            variant["phase_frames"], device=env.device, dtype=torch.long
        )
        if (
            self.durations.shape != (6,)
            or self.durations.sum() != self.reference.shape[0]
        ):
            raise ValueError("Expected six positive phase lengths summing to T.")
        self.ends = self.durations.cumsum(0)
        self.starts = self.ends - self.durations
        static = [
            *variant["geometry_extent_xyz"],
            *variant["initial_pose"],
            *variant["goal_pose"],
            *[frames / 100 for frames in variant["phase_frames"]],
            math.radians(variant["tilt_degrees"]),
            0.0,
        ]
        self.static_condition = self.reference.new_tensor(static)
        if self.static_condition.shape != (25,):
            raise ValueError("Expected 25 declared static conditioning values.")

    def inputs(
        self, state: torch.Tensor | None = None, timesteps: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if state is None:
            state = self.env.current_policy_obs
        if timesteps is None:
            timesteps = self.env.command_time_step
        window, mask = local_window(self.reference, timesteps)
        phase = torch.searchsorted(self.ends, timesteps, right=True)
        fraction = (timesteps - self.starts[phase]) / (
            self.durations[phase] - 1
        ).clamp_min(1)
        condition = torch.cat(
            (
                self.static_condition.expand(state.shape[0], -1),
                torch.nn.functional.one_hot(phase, num_classes=6),
                fraction[:, None],
                (timesteps / max(1, self.reference.shape[0] - 1))[:, None],
            ),
            dim=-1,
        )
        if condition.shape[1] != CONDITION_DIM:
            raise ValueError("Conditioning schema dimension mismatch.")
        return {"state": state, "window": window, "mask": mask, "condition": condition}

    def terminal_inputs(
        self, before_timesteps: torch.Tensor, extras: dict, done: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Base Env auto-resets. Restore the actual next observation for bootstrapping.
        state = self.env.current_policy_obs.clone()
        if done.any():
            state[done] = extras["pre_reset_obs"]
        timesteps = (before_timesteps + 1).clamp_max(self.reference.shape[0] - 1)
        return self.inputs(state, timesteps)

    def pose_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Policy observations are hand-relative; goals and physics poses are in world coordinates.
        pose = self.env.observation_manager.object_entities["cup"].get_obs(
            rot_format="quat"
        )
        if tuple(pose.shape) != (self.env.num_envs, 7):
            raise ValueError(
                f"Expected rigid cup world pose ({self.env.num_envs}, 7), received {tuple(pose.shape)}."
            )
        # The loader may translate the whole reference to its configured support height.
        target = self.reference[-1, -7:]
        position = torch.linalg.vector_norm(pose[:, :3] - target[:3], dim=-1)
        dot = (pose[:, 3:7] * target[3:]).sum(-1).abs().clamp(0, 1)
        return position, 2 * torch.acos(dot)

    def reset_probe(self) -> dict:
        env = self.env
        reference = self.reference.clone()
        if env.num_envs > 1:
            # Use distinct clock states so a mistaken whole-batch reset is observable.
            # This is a reset-test fixture; the full reset below restores the scene.
            env.timesteps[1:] = 3
            env.update_command_time_step()
            before = env.timesteps[1:].clone()
            before_command = env.command_time_step[1:].clone()
            env.reset_env_idxs(np.array([0], dtype=np.int64))
            if not torch.equal(before, env.timesteps[1:]) or not torch.equal(
                before_command, env.command_time_step[1:]
            ):
                raise RuntimeError("Reset leaked into another vector environment.")
        if not torch.equal(reference, self.reference):
            raise RuntimeError("Reset changed the worker-local reference.")
        env.reset()
        if torch.any(env.command_time_step != 0):
            raise RuntimeError("Expected initial-frame resets in every environment.")
        return {
            "partial_reset_isolated": True,
            "reference_unchanged": True,
            "initial_frame": True,
        }
