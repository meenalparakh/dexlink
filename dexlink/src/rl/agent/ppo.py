from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from dexlink.src.rl.core_agent.discriminator import Discriminator
from dexlink.src.rl.core_agent.memory import Memory
from dexlink.src.rl.core_agent.wandb_logger import WandbLogger
from dexlink.src.rl.agent.agent_utils import (
    DiscObservationProcessor,
    PolicyModel,
    ValueModel,
)
from dexlink.src.rl.config.base import AgentConfig, EvalConfig
from dexlink.utils.rl.eval_utils import SHORT_NAMES, EvalMetrics, TrajInfoBatched
from dexlink.utils.rl.overheat import load_overheat_guard

if TYPE_CHECKING:
    from dexlink.src.rl.world.env import Env


def _resume_start_global_step(
    *,
    resume_training: bool,
    resume_start_timestep: int,
    total_timesteps: int,
    rollout_steps: int,
) -> int:
    if resume_start_timestep == -1:
        return 0
    if not resume_training:
        raise ValueError("agent.resume_start_timestep requires agent.resume=True.")
    if resume_start_timestep < 0:
        raise ValueError(
            "Expected agent.resume_start_timestep >= 0, got "
            f"{resume_start_timestep}."
        )
    if resume_start_timestep % rollout_steps != 0:
        raise ValueError(
            "Expected agent.resume_start_timestep to be divisible by "
            f"agent.n_rollout_steps={rollout_steps}, got {resume_start_timestep}."
        )
    if resume_start_timestep > total_timesteps:
        raise ValueError(
            "Expected agent.resume_start_timestep <= agent.timesteps, got "
            f"{resume_start_timestep} > {total_timesteps}."
        )
    return resume_start_timestep // rollout_steps


def _validate_latest_fixed_budget_resume(
    *,
    resume_training: bool,
    resume_start_timestep: int,
    resume_latest_fixed_budget: bool,
) -> None:
    if not resume_latest_fixed_budget:
        return
    if not resume_training:
        raise ValueError("agent.resume_latest_fixed_budget requires agent.resume=True.")
    if resume_start_timestep != -1:
        raise ValueError(
            "agent.resume_latest_fixed_budget requires "
            "agent.resume_start_timestep=-1 so the latest checkpoint is selected."
        )


def _latest_timestep_in_model_dir(
    model_dir: Path,
    required_checkpoint_files: tuple[str, ...] = ("policy.pt",),
) -> int | None:
    if not model_dir.exists():
        return None
    timesteps: list[int] = []
    for policy_path in model_dir.glob("*/policy.pt"):
        checkpoint_name = policy_path.parent.name
        if checkpoint_name == "best":
            continue
        if not checkpoint_name.isdecimal():
            raise ValueError(
                "Expected checkpoint directories containing policy.pt to be "
                f"numeric periodic timesteps or 'best', got {policy_path.parent}."
            )
        checkpoint_dir = policy_path.parent
        if not all(
            (checkpoint_dir / filename).is_file()
            for filename in required_checkpoint_files
        ):
            continue
        timesteps.append(int(checkpoint_name))
    if len(timesteps) == 0:
        return None
    return max(timesteps)


class PPO:
    def __init__(
        self,
        env: Env,
        logger: WandbLogger,
        agent_cfg: AgentConfig,
        eval_cfg: EvalConfig | None = None,
    ):
        self.timesteps = agent_cfg.timesteps
        self.history_len = agent_cfg.history_len
        self.resume_training = agent_cfg.resume
        self.resume_from_ckpt = agent_cfg.resume_from_ckpt
        self.resume_start_timestep = agent_cfg.resume_start_timestep
        self.resume_latest_fixed_budget = agent_cfg.resume_latest_fixed_budget
        _validate_latest_fixed_budget_resume(
            resume_training=self.resume_training,
            resume_start_timestep=self.resume_start_timestep,
            resume_latest_fixed_budget=self.resume_latest_fixed_budget,
        )
        self.num_envs = env.num_envs
        self.cfg = agent_cfg
        self.eval_cfg = eval_cfg
        self.policy_lr = agent_cfg.policy_lr
        self.value_lr = agent_cfg.value_lr
        self.gamma = agent_cfg.gamma
        self.lam = agent_cfg.lam
        self.n_rollout_steps = agent_cfg.n_rollout_steps
        self.batch_size = agent_cfg.batch_size
        self.update_epochs = agent_cfg.update_epochs
        self.clip_ratio = agent_cfg.clip_ratio
        self.entropy_coef = agent_cfg.entropy_coef
        self.vf_coef = agent_cfg.vf_coef
        self.use_discriminator = agent_cfg.use_discriminator and env.use_discriminator
        self.normalize_advantages = agent_cfg.normalize_advantages
        # self.per_loop_env_count = env.num_envs * self.n_rollout_steps
        self.per_loop_env_count = (
            self.n_rollout_steps
        )  # ignore the num_envs factor for now

        self.return_queue = deque(maxlen=10)
        self.num_train_iters = self.timesteps // self.n_rollout_steps
        self.resume_start_global_step = _resume_start_global_step(
            resume_training=self.resume_training,
            resume_start_timestep=self.resume_start_timestep,
            total_timesteps=self.timesteps,
            rollout_steps=self.n_rollout_steps,
        )

        logger.configure_frequencies(self.num_train_iters)
        # Keep the legacy default: if eval config is provided, align legacy eval freq
        # with model-save cadence unless overridden by dedicated eval freqs.

        legacy_eval_freq = int(self.cfg.eval_freq)
        if legacy_eval_freq <= 0:
            legacy_eval_freq = int(logger.model_save_freq)
        self.eval_metrics_freq = self._resolve_eval_frequency(
            int(self.cfg.eval_metrics_freq),
            legacy_eval_freq,
        )
        self.eval_video_freq = self._resolve_eval_frequency(
            int(self.cfg.eval_video_freq),
            legacy_eval_freq,
        )

        # Initialize other PPO components here (networks, optimizers, etc.)
        self.device = env.cfg.device

        self.env = env
        self.logger = logger
        self.best_train_checkpoint_metric = self._load_best_train_checkpoint_metric()
        self.use_reference_motion_update = bool(
            self.env.reference_trajectory_manager.is_enabled()
        )
        self.reference_update_hand = "right"
        self.log_essential_metrics_only = bool(self.cfg.log_essential_metrics_only)
        self._print_training_variant_summary()

        self.use_hand_obj_bps_aux = agent_cfg.hand_obj_bps_aux_loss_coef > 0.0
        self.hand_obj_bps_aux_loss_coef = agent_cfg.hand_obj_bps_aux_loss_coef
        self.hand_obj_bps_dim = 0
        if self.use_hand_obj_bps_aux:
            if not self.env.reward_manager.common.has_hand_obj_rel:
                raise ValueError(
                    "hand_obj_bps_aux_loss_coef > 0 requires the hand-object-relative reward "
                    "path to be enabled in the reward manager."
                )
            self.hand_obj_bps_dim = self.env.reward_manager.get_hand_obj_rel_bps_dim()
            if self.hand_obj_bps_dim <= 0:
                raise ValueError(
                    "Expected positive hand-object-relative BPS dim for auxiliary loss."
                )
        self.act_dim = env.action_space.shape[1]  # remove the batch dimension
        self.policy_variant = str(env.cfg.action_cfg.policy_variant)
        if (
            self.policy_variant
            in {
                "trajectory_weighted_residual_target_v3",
                "trajectory_mean_residual_target_v3",
                "fixed_residual_target_v1",
            }
            and env.cfg.action_cfg.action_mode != "absolute_target"
        ):
            raise ValueError(
                f"{self.policy_variant} requires action_mode='absolute_target'."
            )
        if (
            env.cfg.action_cfg.action_mode == "absolute_target"
            and self.policy_variant
            not in {
                "trajectory_weighted_residual_target_v3",
                "trajectory_mean_residual_target_v3",
                "fixed_residual_target_v1",
            }
        ):
            raise ValueError(
                "action_mode='absolute_target' currently requires "
                "policy_variant='trajectory_weighted_residual_target_v3' or "
                "'trajectory_mean_residual_target_v3' or "
                "'fixed_residual_target_v1'."
            )
        self.policy_net = PolicyModel(
            env=env,
            selected_terms=agent_cfg.policy_inputs,
            obs_embed_dim=agent_cfg.obs_proc_embed_dim,
            obs_layer_norm=agent_cfg.obs_proc_layer_norm,
            pcd_hidden_dims=agent_cfg.obs_proc_pcd_hidden_dims,
            pcd_pool=agent_cfg.obs_proc_pcd_pool,
            action_dim=self.act_dim,
            policy_embed_dim=agent_cfg.policy_embed_dim,
            hand_synergy_dim=agent_cfg.hand_synergy_dim,
            history_len=self.history_len,
            variant=self.policy_variant,
            aux_output_dim=self.hand_obj_bps_dim,
            aux_hidden_dims=agent_cfg.hand_obj_bps_aux_hidden_dims,
        ).to(self.device)
        self.value_net = ValueModel(
            env=env,
            selected_terms=agent_cfg.critic_inputs,
            obs_embed_dim=agent_cfg.obs_proc_embed_dim,
            obs_layer_norm=agent_cfg.obs_proc_layer_norm,
            pcd_hidden_dims=agent_cfg.obs_proc_pcd_hidden_dims,
            pcd_pool=agent_cfg.obs_proc_pcd_pool,
            value_embed_dim=agent_cfg.value_embed_dim,
            history_len=self.history_len,
        ).to(self.device)
        # Keep checkpoint layout compatible with earlier runs while ownership now
        # lives inside the policy/value modules.
        self.policy_obs_processor = self.policy_net.obs_processor
        self.value_obs_processor = self.value_net.obs_processor
        self.hand_obj_bps_decoder = self.policy_net.aux_decoder
        self.policy_obs_dim = self.policy_net.obs_dim
        self.value_obs_dim = self.value_net.obs_dim

        self.optimizer = AdamW(
            [
                {
                    "params": list(self.policy_net.parameters()),
                    "lr": self.policy_lr,
                },
                {
                    "params": list(self.value_net.parameters()),
                    "lr": self.value_lr,
                },
            ]
        )

        if self.use_discriminator:
            # create a memory buffer for discriminator obs
            disc_cfg = self.env.cfg.disc_cfg
            self.discriminator_memory = Memory(
                n_rollout_steps=disc_cfg.memory_size,
                # 1 * agent_cfg.n_rollout_steps,
                n_envs=env.num_envs,
                seq_len=disc_cfg.seq_len,
                name="discriminator",
            )
            # discriminator memory is larger than policy memory
            # to store samples from previous timesteps.
            # seq len is 3, because we want to make transitions of length 1, 2, 3

            self.discriminator_memory.create_tensors(
                {
                    "disc_obs": (env.disc_observation_manager.observation_dim,),
                    "dones": (1,),
                }
            )

            # discriminator internally has its own optimizer
            self.disc_obs_processor = DiscObservationProcessor(
                env, mask_prob=disc_cfg.input_mask_prob
            ).to(self.device)
            self.discriminator = Discriminator(
                input_dim=env.disc_observation_manager.observation_dim,
                expert_sampler=env.disc_observation_manager.expert_sampler,
                memory_buffer=self.discriminator_memory,
                logger=logger,
                disc_cfg=disc_cfg,
                input_processor=self.disc_obs_processor,
                device=self.device,
            ).to(self.device)
            self.env.reward_manager.set_discriminator(self.discriminator)

        if self.history_len > 0:
            self.initialize_circular_buffer()

        self.initialize_memory()

        self.value_loss_fn = nn.MSELoss(reduction="mean")
        self.hand_obj_bps_aux_loss_fn = nn.MSELoss(reduction="mean")
        self.logger = logger
        self.seed = agent_cfg.seed
        self.overheat_guard = load_overheat_guard(agent_cfg.check_overheat)
        self.print_agent_summary()

    @staticmethod
    def _resolve_eval_frequency(freq: int, fallback_freq: int) -> int:
        # 0 disables a specific eval stream, negative values fall back to legacy eval_freq.
        if freq == 0:
            return 0
        if freq > 0:
            return freq
        return fallback_freq

    def print_agent_summary(self):
        # obs dim, action dim, n_policy_params, n_value_params, n_disc_params
        n_policy_params = sum(p.numel() for p in self.policy_net.net.parameters())
        n_policy_obs_params = sum(
            p.numel() for p in self.policy_net.obs_processor.parameters()
        )
        n_value_params = sum(p.numel() for p in self.value_net.net.parameters())
        n_value_obs_params = sum(
            p.numel() for p in self.value_net.obs_processor.parameters()
        )
        n_disc_params = (
            sum(p.numel() for p in self.discriminator.parameters())
            if self.use_discriminator
            else 0
        )
        n_aux_params = (
            sum(p.numel() for p in self.policy_net.aux_decoder.parameters())
            if self.policy_net.aux_decoder is not None
            else 0
        )

        # convert num params to millions
        n_policy_params = np.round(n_policy_params / 1e6, 3)
        n_policy_obs_params = np.round(n_policy_obs_params / 1e6, 3)
        n_value_params = np.round(n_value_params / 1e6, 3)
        n_value_obs_params = np.round(n_value_obs_params / 1e6, 3)
        n_disc_params = np.round(n_disc_params / 1e6, 3)
        n_aux_params = np.round(n_aux_params / 1e6, 3)

        info = {
            "policy_obs_dim": self.policy_obs_dim,
            "value_obs_dim": self.value_obs_dim,
            "action_dim": self.act_dim,
            "n_policy_obs_params": f"{n_policy_obs_params} M",
            "n_policy_params": f"{n_policy_params} M",
            "n_value_obs_params": f"{n_value_obs_params} M",
            "n_value_params": f"{n_value_params} M",
            "n_disc_params": f"{n_disc_params} M",
            "n_aux_params": f"{n_aux_params} M",
        }

        # print as a table
        lines = []
        lines.append("-" * 40)
        lines.append(f"{'Agent Summary':^40}")
        lines.append("-" * 40)
        for k, v in info.items():
            lines.append(f"{k:<20}: {v:>18}")
        lines.append("-" * 40)
        print("\n".join(lines))
        # self._print_lightning_summaries()

    def initialize_circular_buffer(self):
        # for storing history
        self.history = Memory(
            n_rollout_steps=self.history_len,
            n_envs=self.env.num_envs,
            seq_len=1,  # does not matter, we are not sampling from this buffer
            name="agent_history",
        )
        self.history.create_tensors(
            {
                "obs": (self.policy_net.obs_save_dim,),
                "actions": (self.act_dim,),
                "dones": (1,),  # to track episode boundaries
            }
        )
        self.reset_history_buffer()

    def reset_history_buffer(self):
        self.history._tensors["obs"].zero_()
        self.history._tensors["actions"].zero_()
        self.history._tensors["dones"].fill_(1.0)
        self.history.reset()

    def _history_dones_for_model(
        self, history_dones: torch.Tensor | None
    ) -> torch.Tensor | None:
        if history_dones is None:
            return None
        return history_dones.transpose(0, 1).to(self.device)

    def initialize_memory(self):
        tensor_info = {
            "obs": (self.policy_net.obs_save_dim,),
            "actions": (self.act_dim,),
            "log_probs": (1,),
            "rewards": (1,),
            "terminated": (1,),
            "truncated": (1,),
            "dones": (1,),
            "values": (1,),
            "advantages": (1,),
            "target_returns": (1,),
        }
        if self.use_hand_obj_bps_aux:
            tensor_info["hand_obj_rel_bps"] = (self.hand_obj_bps_dim,)
        if self.use_reference_motion_update:
            tensor_info["reference_timestep"] = (1,)
            tensor_info["object_pos_world"] = (3,)
            tensor_info["object_quat_world"] = (4,)
            tensor_info["wrist_pos_world"] = (3,)
            tensor_info["wrist_quat_world"] = (4,)
        self.memory = Memory(
            n_rollout_steps=self.n_rollout_steps,
            n_envs=self.env.num_envs,
            # history is past steps + 1 for current timestep
            seq_len=self.history_len + 1,
            name="ppo",
        )
        self.memory.create_tensors(tensor_info)

    def _get_hand_obj_rel_bps_target(self, info: dict) -> torch.Tensor:
        if not self.use_hand_obj_bps_aux:
            raise ValueError("Auxiliary hand-object-relative BPS loss is disabled.")
        if "hand_obj_rel_bps" not in info:
            raise ValueError(
                "Expected env info to contain 'hand_obj_rel_bps' when the auxiliary "
                "hand-object-relative BPS loss is enabled."
            )
        target = info["hand_obj_rel_bps"]
        if not isinstance(target, torch.Tensor):
            raise ValueError(
                f"Expected 'hand_obj_rel_bps' to be a torch.Tensor, got {type(target).__name__}."
            )
        expected_shape = (self.env.num_envs, self.hand_obj_bps_dim)
        if tuple(target.shape) != expected_shape:
            raise ValueError(
                f"Expected 'hand_obj_rel_bps' shape {expected_shape}, got {tuple(target.shape)}."
            )
        return target

    def _get_reference_update_memory_kwargs(self) -> dict[str, torch.Tensor]:
        # The saved policy observation intentionally zeroes wrist-control joints
        # and wrist translation, so PPO caches the world-frame wrist pose
        # explicitly if we want to reconstruct robot reference motion after the rollout.
        if not self.use_reference_motion_update:
            return {}
        if len(self.env.task_manager.active_hand_sides) != 1:
            raise NotImplementedError(
                "Reference motion updates are currently unsupported for bimanual tasks."
            )
        if self.reference_update_hand != "right":
            raise ValueError(
                f"Reference motion updates currently support only 'right', got {self.reference_update_hand!r}."
            )
        wrist_pos = self.env.current_wrist_pos_by_side.get(self.reference_update_hand)
        wrist_quat = self.env.current_wrist_quat_by_side.get(self.reference_update_hand)
        object_entity = self.env.physics_manager.scene_entities["object"]
        object_obs = object_entity.get_obs(rot_format="quat")
        if object_obs.ndim != 2 or object_obs.shape[1] < 7:
            raise ValueError(
                "Expected object observation with shape (num_envs, >= 7) containing "
                f"position and quaternion, got {tuple(object_obs.shape)}."
            )
        object_pos = object_obs[:, :3]
        object_quat = object_obs[:, 3:7]
        if wrist_pos is None or wrist_quat is None:
            raise ValueError(
                f"Reference motion update requires wrist pose for hand {self.reference_update_hand!r}."
            )
        return {
            "reference_timestep": self.env.command_time_step.unsqueeze(-1)
            .detach()
            .clone(),
            "object_pos_world": object_pos.detach().clone(),
            "object_quat_world": object_quat.detach().clone(),
            "wrist_pos_world": wrist_pos.detach().clone(),
            "wrist_quat_world": wrist_quat.detach().clone(),
        }

    def _extract_rollout_robot_reference_targets(
        self,
    ) -> dict[str, torch.Tensor] | None:
        # Each stored PPO state is treated as a weighted rollout sample whose
        # target return serves as the score for GP fitting. We export the GP
        # state directly: wrist pose in world frame plus finger joint angles.
        if not self.use_reference_motion_update:
            return None
        if len(self.env.task_manager.active_hand_sides) != 1:
            raise NotImplementedError(
                "Reference motion updates are currently unsupported for bimanual tasks."
            )
        if self.reference_update_hand != "right":
            raise ValueError(
                f"Reference motion updates currently support only 'right', got {self.reference_update_hand!r}."
            )

        required_keys = (
            "obs",
            "target_returns",
            "reference_timestep",
            "object_pos_world",
            "object_quat_world",
            "wrist_pos_world",
            "wrist_quat_world",
        )
        for key in required_keys:
            if key not in self.memory.tensors:
                raise ValueError(
                    f"Reference motion update expected memory tensor {key!r} to exist."
                )

        obs = self.memory.tensors["obs"]
        target_returns = self.memory.tensors["target_returns"].squeeze(-1)
        reference_timestep = self.memory.tensors["reference_timestep"].squeeze(-1)
        object_pos_world = self.memory.tensors["object_pos_world"]
        object_quat_world = self.memory.tensors["object_quat_world"]
        wrist_pos_world = self.memory.tensors["wrist_pos_world"]
        wrist_quat_world = self.memory.tensors["wrist_quat_world"]
        named_obs = self.env.observation_manager.named_view_policy(
            obs.reshape(-1, obs.shape[-1])
        )
        joint_key = f"{self.reference_update_hand}_jpos"
        if joint_key not in named_obs:
            raise ValueError(
                f"Expected saved policy observations to contain {joint_key!r}."
            )
        masked_hand_qpos = named_obs[joint_key].reshape(
            self.n_rollout_steps, self.env.num_envs, -1
        )

        action_manager = self.env.action_manager
        hand = action_manager.hand
        if self.reference_update_hand != getattr(
            hand, "name", self.reference_update_hand
        ):
            raise ValueError(
                "Reference motion update currently assumes the PPO-controlled hand "
                f"matches {self.reference_update_hand!r}."
            )

        expected_hand_qpos_shape = (
            self.n_rollout_steps,
            self.env.num_envs,
            len(hand.dof_map),
        )
        if tuple(masked_hand_qpos.shape) != expected_hand_qpos_shape:
            raise ValueError(
                "Decoded hand qpos shape mismatch: expected "
                f"{expected_hand_qpos_shape}, got {tuple(masked_hand_qpos.shape)}."
            )

        finger_joint_indices = [
            hand.joint_names.index(joint_name) for joint_name in hand.finger_joint_names
        ]
        finger_joint_positions = masked_hand_qpos[:, :, finger_joint_indices]

        return {
            "timesteps": reference_timestep.transpose(0, 1).contiguous(),
            "returns": target_returns.transpose(0, 1).reshape(-1),
            "object_pos": object_pos_world.transpose(0, 1).contiguous(),
            "object_quat": object_quat_world.transpose(0, 1).contiguous(),
            "hand_wrist_pos": {
                self.reference_update_hand: wrist_pos_world.transpose(0, 1).contiguous()
            },
            "hand_wrist_quat": {
                self.reference_update_hand: wrist_quat_world.transpose(
                    0, 1
                ).contiguous()
            },
            "hand_joint_positions": {
                self.reference_update_hand: finger_joint_positions.transpose(
                    0, 1
                ).contiguous()
            },
        }

    def _maybe_update_reference_motion(
        self, *, global_step: int
    ) -> dict[str, float] | None:
        if not self.use_reference_motion_update:
            return None
        update_freq = int(self.env.cfg.reference_cfg.update_freq_policy_iters)
        if update_freq <= 0:
            raise ValueError(
                "Expected env.reference_cfg.update_freq_policy_iters > 0, got "
                f"{self.env.cfg.reference_cfg.update_freq_policy_iters}."
            )
        if (global_step + 1) % update_freq != 0:
            return None
        rollout_batch = self._extract_rollout_robot_reference_targets()
        if rollout_batch is None:
            return None
        num_rollouts = rollout_batch["returns"].numel()
        if num_rollouts < int(self.env.cfg.reference_cfg.min_rollouts_for_update):
            return None

        rollout_timesteps = rollout_batch["timesteps"].reshape(num_rollouts, 1)
        object_pos = rollout_batch["object_pos"].reshape(num_rollouts, 1, -1)
        object_quat = rollout_batch["object_quat"].reshape(num_rollouts, 1, -1)
        hand_joint_positions = {
            side: joints.reshape(num_rollouts, 1, joints.shape[-1])
            for side, joints in rollout_batch["hand_joint_positions"].items()
        }
        hand_wrist_pos = {
            side: wrist_pos.reshape(num_rollouts, 1, wrist_pos.shape[-1])
            for side, wrist_pos in rollout_batch["hand_wrist_pos"].items()
        }
        hand_wrist_quat = {
            side: wrist_quat.reshape(num_rollouts, 1, wrist_quat.shape[-1])
            for side, wrist_quat in rollout_batch["hand_wrist_quat"].items()
        }
        stats = self.env.update_reference_trajectory(
            timesteps=rollout_timesteps,
            returns=rollout_batch["returns"],
            object_pos=object_pos,
            object_quat=object_quat,
            hand_wrist_pos=hand_wrist_pos,
            hand_wrist_quat=hand_wrist_quat,
            hand_joint_positions=hand_joint_positions,
            policy_iter=global_step + 1,
        )
        plot_path = self.env.reference_trajectory_manager.save_gp_plot(
            output_dir=self.logger.experiment_dir / "reference_gp_plots",
            update_step=global_step + 1,
            side=self.reference_update_hand,
        )
        if plot_path is not None:
            stats["reference_motion_plot_saved"] = 1.0
        stats["reference_motion_update_iteration"] = float(global_step + 1)
        return stats

    def compute_gae(self, last_values):
        # Return: target_returns, advantages
        # use the entire memory buffer to compute gae

        rewards = self.memory.tensors["rewards"].to(
            self.device
        )  # (n_rollout_steps, n_envs, 1)
        values = self.memory.tensors["values"].to(
            self.device
        )  # (n_rollout_steps, n_envs, 1)
        dones = self.memory.tensors["dones"].to(
            self.device
        )  # (n_rollout_steps, n_envs, 1)

        # last_values is of shape (n_envs, )
        # should be (1, n_envs, 1)
        last_values = last_values.unsqueeze(0).unsqueeze(-1)  # (1, n_envs, 1)

        all_values = torch.cat(
            [values, last_values], dim=0
        )  # (n_rollout_steps + 1, n_envs, 1)
        assert all_values.shape == (
            self.n_rollout_steps + 1,
            self.env.num_envs,
            1,
        ), f"all_values shape mismatch: expected {(self.n_rollout_steps + 1, self.env.num_envs, 1)}, got {all_values.shape}"

        # compute deltas
        deltas = rewards + self.gamma * all_values[1:] * (1 - dones) - all_values[:-1]
        advantages = torch.zeros_like(rewards).to(self.device)
        last_adv = 0.0
        for t_step in range(self.n_rollout_steps - 1, -1, -1):
            adv = (
                deltas[t_step] + self.gamma * self.lam * (1 - dones[t_step]) * last_adv
            )
            assert adv.shape == (
                self.env.num_envs,
                1,
            ), f"adv shape mismatch: expected {(self.env.num_envs, 1)}, got {adv.shape}"
            advantages[t_step] = adv
            last_adv = adv

        target_returns = advantages + values
        # normalize the advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            # just center the advantages
            advantages = advantages - advantages.mean()
        return target_returns, advantages

    def update_log_extras(self, log_extras, info):
        for k, v in info.items():
            if k.startswith("log/"):
                # try:
                _k = f"{k[4:]}"  # remove 'log/' prefix
                if not self._should_collect_train_log_extra(_k):
                    continue
                if _k not in log_extras:
                    log_extras[_k] = []
                if isinstance(v, (float, int, np.float32, np.float64)):
                    log_extras[_k].append(v)
                elif isinstance(v, torch.Tensor):
                    log_extras[_k].append(v.float().mean().item())
            # except Exception as e:
            #     print(f"Error logging info key {k}: {e}")
            #     import pdb; pdb.set_trace()

    def _load_best_train_checkpoint_metric(self) -> float:
        metric_name = self.env.task_manager.checkpoint_metric_name
        if metric_name is None:
            return float("-inf")
        metadata = self.logger.load_named_checkpoint_metadata("best")
        if metadata is None:
            return float("-inf")
        recorded_name = metadata.get("selection_metric_name")
        if recorded_name != metric_name:
            raise ValueError(
                "Existing best checkpoint uses a different selection metric: "
                f"expected {metric_name!r}, found {recorded_name!r}."
            )
        return float(metadata["selection_metric_value"])

    def _new_episode_task_metric_values(self) -> dict[str, list[float]]:
        return {name: [] for name in self.env.task_manager.task_metric_names}

    def _update_episode_task_metric_values(
        self,
        values_by_name: dict[str, list[float]],
        info: dict,
    ) -> None:
        episode_metrics = info.get("episode_task_metrics")
        if episode_metrics is None:
            return
        expected_names = tuple(values_by_name)
        if tuple(episode_metrics) != expected_names:
            raise ValueError(
                "Terminal task metric keys do not match the configured keys: "
                f"expected {expected_names}, got {tuple(episode_metrics)}."
            )
        episode_count = None
        for name, values in episode_metrics.items():
            if not isinstance(values, torch.Tensor) or values.ndim != 1:
                raise ValueError(
                    f"Expected terminal task metric {name!r} to be a rank-1 tensor, "
                    f"got {type(values).__name__} with shape "
                    f"{getattr(values, 'shape', None)}."
                )
            if episode_count is None:
                episode_count = int(values.numel())
            elif int(values.numel()) != episode_count:
                raise ValueError(
                    "Expected the same completed-episode count for all task metrics, "
                    f"got {episode_count} and {int(values.numel())} for {name!r}."
                )
            values_by_name[name].extend(values.detach().cpu().tolist())

    def _summarize_episode_task_metrics(
        self,
        values_by_name: dict[str, list[float]],
    ) -> tuple[dict[str, float], int]:
        if not values_by_name:
            return {}, 0
        counts = {name: len(values) for name, values in values_by_name.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"Expected equal terminal task metric sample counts, got {counts}."
            )
        episode_count = next(iter(counts.values()))
        if episode_count == 0:
            return {}, 0
        means = {}
        for name, values in values_by_name.items():
            array = np.asarray(values, dtype=np.float64)
            finite = array[np.isfinite(array)]
            means[name] = float(np.mean(finite)) if finite.size > 0 else float("nan")
        return (
            means,
            episode_count,
        )

    @staticmethod
    def _new_episode_progress_values() -> dict[str, list[float]]:
        return {
            "progress_fraction": [],
            "span_steps": [],
        }

    @staticmethod
    def _update_episode_progress_values(
        values_by_name: dict[str, list[float]],
        info: dict,
    ) -> None:
        info_keys = {
            "progress_fraction": "episode_progress_fraction",
            "span_steps": "episode_span_steps",
        }
        available = [info_key in info for info_key in info_keys.values()]
        if not any(available):
            return
        if not all(available):
            present = [key for key in info_keys.values() if key in info]
            raise KeyError(
                "Completed-episode progress logging requires both progress and "
                f"span values; present keys are {present}."
            )
        episode_count = None
        for name, info_key in info_keys.items():
            values = info[info_key]
            if not isinstance(values, torch.Tensor) or values.ndim != 1:
                raise ValueError(
                    f"Expected {info_key!r} to be a rank-1 tensor, got "
                    f"{type(values).__name__} with shape "
                    f"{getattr(values, 'shape', None)}."
                )
            if episode_count is None:
                episode_count = int(values.numel())
            elif int(values.numel()) != episode_count:
                raise ValueError(
                    "Expected matching completed-episode progress sample counts, "
                    f"got {episode_count} and {int(values.numel())}."
                )
            values_by_name[name].extend(values.detach().cpu().tolist())

    @staticmethod
    def _summarize_episode_progress_values(
        values_by_name: dict[str, list[float]],
    ) -> tuple[dict[str, float], int]:
        counts = {name: len(values) for name, values in values_by_name.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"Expected equal completed-episode progress sample counts, got {counts}."
            )
        episode_count = next(iter(counts.values()))
        if episode_count == 0:
            return {}, 0
        return (
            {
                name: float(np.mean(np.asarray(values, dtype=np.float64)))
                for name, values in values_by_name.items()
            },
            episode_count,
        )

    def _update_episode_checkpoint_metric_values(
        self,
        values: list[float],
        info: dict,
    ) -> None:
        metric_values = info.get("episode_checkpoint_metric_values")
        if metric_values is None:
            return
        expected_name = self.env.task_manager.checkpoint_metric_name
        recorded_name = info.get("episode_checkpoint_metric_name")
        if recorded_name != expected_name:
            raise ValueError(
                "Terminal checkpoint metric name does not match the configured "
                f"name: expected {expected_name!r}, got {recorded_name!r}."
            )
        if not isinstance(metric_values, torch.Tensor) or metric_values.ndim != 1:
            raise ValueError(
                "Expected terminal checkpoint metric values to be a rank-1 "
                f"tensor, got {type(metric_values).__name__} with shape "
                f"{getattr(metric_values, 'shape', None)}."
            )
        values.extend(metric_values.detach().cpu().tolist())

    def _checkpoint_models(self) -> dict[str, nn.Module | AdamW | None]:
        return {
            "policy_obs_processor": self.policy_obs_processor,
            "value_obs_processor": self.value_obs_processor,
            "policy": self.policy_net.net,
            "value": self.value_net.net,
            "hand_obj_bps_decoder": self.hand_obj_bps_decoder,
            "discriminator": self.discriminator if self.use_discriminator else None,
            "optimizer": self.optimizer,
            "discriminator_optimizer": (
                self.discriminator.optimizer if self.use_discriminator else None
            ),
        }

    def _maybe_save_best_checkpoint_metric(
        self,
        metric_value: float | None,
        episode_count: int,
        global_step: int,
    ) -> None:
        if self.logger.cfg.disable:
            return
        metric_name = self.env.task_manager.checkpoint_metric_name
        if metric_name is None or metric_value is None:
            return
        if metric_value <= self.best_train_checkpoint_metric:
            return
        policy_timestep = global_step * self.n_rollout_steps
        # The policy producing the first rollout has not received a PPO update.
        # Never register that initialization as an evaluable best checkpoint.
        if policy_timestep <= 0:
            return
        rollout_end_timestep = (global_step + 1) * self.n_rollout_steps
        self.logger.save_named_checkpoint(
            self._checkpoint_models(),
            "best",
            {
                "training_protocol_version": self.env.cfg.key_task_protocol_version,
                "selection_metric_name": metric_name,
                "selection_metric_value": metric_value,
                "completed_episode_count": episode_count,
                "policy_timestep": policy_timestep,
                "rollout_end_timestep": rollout_end_timestep,
                "policy_train_iteration": global_step,
                "rollout_iteration": global_step + 1,
            },
        )
        self.best_train_checkpoint_metric = metric_value

    def _filtered_train_log_extras(self, log_extras):
        reduced = {k: np.mean(v) for k, v in log_extras.items()}
        if not self.log_essential_metrics_only:
            return reduced
        return {
            k: value
            for k, value in reduced.items()
            if self._should_collect_train_log_extra(k)
        }

    def _should_collect_train_log_extra(self, key: str) -> bool:
        if not self.log_essential_metrics_only:
            return True
        allowed = {
            "actual_success",
            "all_objects_pose_achieved",
            "current_retargeted_joint_mse_enabled",
            "far_away",
            "final_scene_configuration_success",
            "multi_object_far_away",
            "object_tracking_product_rew",
            "paper_final_scene_configuration_success",
            "paper_final_scene_max_arti_error",
            "paper_final_scene_max_pos_error_m",
            "paper_final_scene_max_quat_error_rad",
            "paper_hand_obj_rel_direction_cost",
            "paper_hand_obj_rel_displacement_cost",
            "paper_hand_obj_rel_displacement_reward",
            "paper_hand_obj_rel_distance_cost",
            "paper_hand_obj_rel_normal_cost",
            "paper_hand_obj_rel_normal_reward",
            "paper_object_arti_error",
            "paper_object_pos_error_m",
            "paper_object_quat_error_rad",
            "pose_achieved",
            "reward_object_tracking",
            "reward_hand_cfg",
            "reward_hand_obj_rel",
            "reward_retargeted_joint_mse",
            "reward_task_completion",
        }
        return (
            key in allowed
            or (key.startswith("reward_") and key.endswith("_raw"))
            or (key.startswith("reward_") and key.endswith("_weighted"))
            or key.startswith("retargeted_finger_joint_mse_")
            or key.startswith("retargeted_wrist_pos_mse_")
            or key.startswith("retargeted_wrist_pose_error_")
            or key.startswith("retargeted_wrist_rot_error_")
        )

    def _print_training_variant_summary(self) -> None:
        action_cfg = self.env.cfg.action_cfg
        reset_cfg = self.env.cfg.reset_cfg
        rewards_cfg = self.env.cfg.rewards_cfg
        commands_cfg = self.env.cfg.commands_cfg
        task = self.env.task_manager
        lines = [
            "[PPO] Training variant summary",
            f"logger.group={self.logger.cfg.group}",
            f"logger.runname={self.logger.cfg.runname}",
            f"active_hands={tuple(task.active_hand_sides)}",
            f"action_mode={action_cfg.action_mode}",
            f"policy_variant={action_cfg.policy_variant}",
            f"residual_target_reference_mode={action_cfg.residual_target_reference_mode}",
            f"residual_beta={action_cfg.residual_beta}",
            f"lock_wrist_targets_to_reference={action_cfg.lock_wrist_targets_to_reference}",
            f"commands_use_retargeted_motion={commands_cfg.use_retargeted_motion}",
            f"reset_start_range={reset_cfg.start_range}",
            f"reset_timestep_sampling_mode={reset_cfg.reset_timestep_sampling_mode}",
            f"reset_start_phase_probs={reset_cfg.start_phase_probs}",
            f"reverse_curriculum={reset_cfg.reverse_curriculum}",
            f"retargeted_hand_reference_mode={reset_cfg.retargeted_hand_reference_mode}",
            f"mask_reference_hand_action_obs={self.env.cfg.obs_cfg.mask_reference_hand_action_obs}",
            f"reward_log_metrics={rewards_cfg.log_metrics}",
            f"reward_log_essential_metrics_only={rewards_cfg.log_essential_metrics_only}",
            f"agent_log_essential_metrics_only={self.log_essential_metrics_only}",
            f"reward_use_retargeted_motion={rewards_cfg.use_retargeted_motion}",
            f"reward_weights=object:{rewards_cfg.task_weight}, "
            f"hand_cfg:{rewards_cfg.hand_cfg_weight}, "
            f"hand_obj_bps:{rewards_cfg.hand_obj_rel_weight}, "
            f"retargeted_mse:{rewards_cfg.retargeted_joint_mse_reward_weight}, "
            f"smoothness:{rewards_cfg.joint_smoothness_reward_weight}, "
            f"task_success:{rewards_cfg.task_completion_weight}",
            f"retargeted_joint_mse_joint_mode={rewards_cfg.retargeted_joint_mse_joint_mode}",
            f"retargeted_joint_mse_reference_noise={rewards_cfg.retargeted_joint_mse_reference_noise}",
            f"retargeted_joint_mse_noise_std=pos:{rewards_cfg.retargeted_joint_mse_wrist_pos_noise_std}, "
            f"rot:{rewards_cfg.retargeted_joint_mse_wrist_rot_noise_std}, "
            f"finger:{rewards_cfg.retargeted_joint_mse_finger_joint_noise_std}",
            f"policy_inputs={self.cfg.policy_inputs}",
            f"critic_inputs={self.cfg.critic_inputs}",
        ]
        print("\n".join(lines))

    def save(self, timestep, global_step):
        # dump the model and optimizer states to logger

        self.logger.log(
            {"model": self._checkpoint_models()},
            timestep=timestep,
            force_model_save=True,
            train_iters=global_step,
        )

    def resume(self, timestep=None) -> int | None:
        ckpt_dir, ckpt_source = self._resolve_resume_source(timestep=timestep)
        if ckpt_dir is None:
            print("No saved models found to resume.")
            return None

        print(f"Resuming from checkpoint: {ckpt_dir / 'policy.pt'}")
        print(f"Resume source: {ckpt_source}")
        self._load_checkpoint_dir(ckpt_dir, load_optimizer=True)
        if timestep is not None:
            return int(timestep)
        try:
            return int(ckpt_dir.name)
        except ValueError:
            return None

    def _run_prefix_for_seed(self, run_name: str) -> str:
        """Return stable run prefix up to '-s<seed>', stripping optional timestamp."""
        # Support both '<name>-s42-03_03-13_41' and '<name>__s42-03_03-13_41' naming styles.
        match = re.match(r"^(.*(?:-s|__s)\d+)(?:-\d{2}_\d{2}-\d{2}_\d{2})?$", run_name)
        if match is not None:
            return match.group(1)
        return run_name

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> Path:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(self.logger.cfg.logdir) / ckpt_path
        if ckpt_path.name != "policy.pt":
            raise ValueError(
                "agent.resume_from_ckpt must point to a checkpoint file ending in "
                f"'policy.pt', but received: {ckpt_path}"
            )
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        return ckpt_path

    def _load_checkpoint_dir(self, ckpt_dir: Path, load_optimizer: bool):
        policy_path = ckpt_dir / "policy.pt"
        value_path = ckpt_dir / "value.pt"
        policy_obs_proc_path = ckpt_dir / "policy_obs_processor.pt"
        value_obs_proc_path = ckpt_dir / "value_obs_processor.pt"
        legacy_obs_proc_path = ckpt_dir / "obs_processor.pt"
        disc_path = ckpt_dir / "discriminator.pt"
        opt_path = ckpt_dir / "optimizer.pt"
        disc_opt_path = ckpt_dir / "discriminator_optimizer.pt"
        hand_obj_bps_decoder_path = ckpt_dir / "hand_obj_bps_decoder.pt"

        assert policy_path.exists(), f"Policy model file not found: {policy_path}"
        assert value_path.exists(), f"Value model file not found: {value_path}"

        if policy_obs_proc_path.exists():
            self.policy_obs_processor.load_state_dict(
                torch.load(policy_obs_proc_path, map_location=self.device)
            )
        elif legacy_obs_proc_path.exists():
            self.policy_obs_processor.load_state_dict(
                torch.load(legacy_obs_proc_path, map_location=self.device)
            )
        else:
            if self.policy_obs_processor.scene_cloud_contract is not None:
                raise FileNotFoundError(
                    f"scene_cloud_v1 requires trained policy observation processor state: {policy_obs_proc_path}"
                )
            print(
                "Warning: checkpoint is missing policy observation processor state. "
                "Using a freshly initialized policy_obs_processor."
            )
        if value_obs_proc_path.exists():
            self.value_obs_processor.load_state_dict(
                torch.load(value_obs_proc_path, map_location=self.device)
            )
        else:
            print(
                "Warning: checkpoint is missing value observation processor state. "
                "Using a freshly initialized value_obs_processor."
            )
        self.policy_net.net.load_state_dict(
            torch.load(policy_path, map_location=self.device)
        )
        self.value_net.net.load_state_dict(
            torch.load(value_path, map_location=self.device)
        )
        if self.hand_obj_bps_decoder is not None:
            if hand_obj_bps_decoder_path.exists():
                self.hand_obj_bps_decoder.load_state_dict(
                    torch.load(hand_obj_bps_decoder_path, map_location=self.device)
                )
            else:
                print(
                    "Warning: checkpoint is missing hand-object-relative BPS decoder state. "
                    "Using a freshly initialized auxiliary decoder."
                )

        if load_optimizer and opt_path.exists():
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location=self.device)
            )

        if self.use_discriminator:
            assert (
                disc_path.exists()
            ), f"Discriminator model file not found: {disc_path}"
            self.discriminator.load_state_dict(
                torch.load(disc_path, map_location=self.device)
            )
            if load_optimizer and disc_opt_path.exists():
                self.discriminator.optimizer.load_state_dict(
                    torch.load(disc_opt_path, map_location=self.device)
                )

    def _resolve_resume_source(self, timestep: int | None):
        if self.resume_from_ckpt is not None:
            policy_path = self._resolve_checkpoint_path(self.resume_from_ckpt)
            if not self.resume_latest_fixed_budget:
                ckpt_dir = policy_path.parent
                print(f"[resume] using explicit checkpoint: {policy_path}")
                return ckpt_dir, str(policy_path)
            explicit_model_dir = policy_path.parent.parent
            if explicit_model_dir.name != "models":
                raise ValueError(
                    "agent.resume_latest_fixed_budget requires an explicit initial "
                    "checkpoint with layout '<run_dir>/models/<timestep>/policy.pt', "
                    f"got {policy_path}."
                )
        else:
            explicit_model_dir = None

        current_run_dir = self.logger.experiment_dir
        group_dir = current_run_dir.parent
        run_prefix = self._run_prefix_for_seed(current_run_dir.name)

        print(f"[resume] current run dir: {current_run_dir}")
        print(f"[resume] group dir: {group_dir}")
        print(f"[resume] run prefix for seed: {run_prefix}")

        candidate_model_dirs = [current_run_dir / "models"]
        if explicit_model_dir is not None:
            candidate_model_dirs.append(explicit_model_dir)
        if group_dir.exists():
            for run_dir in group_dir.iterdir():
                if not run_dir.is_dir() or run_dir == current_run_dir:
                    continue
                if self._run_prefix_for_seed(run_dir.name) == run_prefix:
                    candidate_model_dirs.append(run_dir / "models")

        # Preserve order while avoiding repeated scans when the explicit source is
        # already one of the matching attempts.
        candidate_model_dirs = list(dict.fromkeys(candidate_model_dirs))

        print("[resume] candidate model dirs:")
        for model_dir in candidate_model_dirs:
            print(f"  - {model_dir}")

        if timestep is not None:
            print(f"[resume] requested timestep: {timestep}")
            for model_dir in candidate_model_dirs:
                if (model_dir / str(timestep) / "policy.pt").exists():
                    print(f"[resume] found requested timestep in: {model_dir}")
                    return model_dir / str(timestep), str(model_dir.parent)
                print(f"[resume] missing requested timestep in: {model_dir}")
            print("[resume] requested timestep was not found in any candidate dir.")
            return None, None

        best: tuple[int, float, Path] | None = None
        required_checkpoint_files = tuple(
            f"{name}.pt"
            for name, module in self._checkpoint_models().items()
            if module is not None
        )
        for model_dir in candidate_model_dirs:
            latest = _latest_timestep_in_model_dir(
                model_dir,
                required_checkpoint_files=required_checkpoint_files,
            )
            if latest is None:
                print(f"[resume] no checkpoints found in: {model_dir}")
                continue
            try:
                mtime = model_dir.parent.stat().st_mtime
            except OSError:
                mtime = 0.0
            print(
                f"[resume] found checkpoint in {model_dir}: latest_timestep={latest}, run_mtime={mtime}"
            )
            candidate = (latest, mtime, model_dir)
            if (
                best is None
                or candidate[0] > best[0]
                or (candidate[0] == best[0] and candidate[1] > best[1])
            ):
                best = candidate

        if best is None:
            print("[resume] no valid checkpoints discovered in candidate dirs.")
            return None, None
        print(
            f"[resume] selected source: {best[2]} (timestep={best[0]}, run_mtime={best[1]})"
        )
        return best[2] / str(best[0]), str(best[2].parent)

    def load(self, policy_ckpt):
        policy_ckpt = Path(policy_ckpt)
        self._load_checkpoint_dir(policy_ckpt.parent, load_optimizer=False)

    @staticmethod
    def _policy_obs(obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(obs, dict):
            if "policy" not in obs:
                raise ValueError(
                    f"Expected observation dict to contain 'policy', got keys {list(obs.keys())}."
                )
            return obs["policy"]
        return obs

    def init_env(self):

        # note: in env step, the next_obs may not correspond to
        # reward, terminated, truncated, if done is True (terminated | truncated)
        # in that case, next_obs is the first obs after reset, while
        # reward etc correspond to the last step before reset.
        # the last step obs is available in the info dict.
        # the action taken corresponds to the outcome, but not with the next_obs.
        # the values correspond to the previous obs as well.

        # the memory however stores, st, r1, at (that is, the obs on which action is taken)
        obs, info = self.env.reset(seed=self.seed)
        policy_obs = self._policy_obs(obs)

        # reset the history buffer
        if self.history_len > 0:
            self.reset_history_buffer()

        # in the first step there is no history?
        # if self.history_len > 0:
        #     # proc_obs = self.obs_processor(obs, update_stats=True)
        #     self.history.add_samples(
        #         obs=obs,
        #         actions=torch.ones(self.env.num_envs, self.act_dim).to(self.device)
        #         * torch.nan,
        #         dones=torch.ones(self.env.num_envs, 1).to(
        #             self.device
        #         ),  # the dones indicate the reset
        #     )

        return policy_obs, info

    def train(
        self,
        eval_cfg: EvalConfig | None = None,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> int:
        # trainer method
        # the logging convention here is
        # all logs are wrt the number of env steps taken so far (ignoring the parallel env count)
        # also save the data collection + update cycles (global_step)
        # the log_freq and model_save_freq though are in terms of global step.

        # resume from last checkpoint if any
        # by default all runs are suffixed with timestamp, so resuming happens only when
        # a full run name is provided and timestamp is excluded.
        start_global_step = 0
        if self.resume_training:
            requested_timestep = (
                None if self.resume_start_timestep == -1 else self.resume_start_timestep
            )
            loaded_timestep = self.resume(timestep=requested_timestep)
            if self.resume_latest_fixed_budget:
                if loaded_timestep is None:
                    print(
                        "[resume] no complete numeric checkpoint exists yet; "
                        f"starting the fixed budget from timestep 0 to {self.timesteps}."
                    )
                else:
                    start_global_step = _resume_start_global_step(
                        resume_training=True,
                        resume_start_timestep=loaded_timestep,
                        total_timesteps=self.timesteps,
                        rollout_steps=self.n_rollout_steps,
                    )
                    print(
                        "[resume] dynamically selected latest complete checkpoint at "
                        f"timestep {loaded_timestep}; continuing fixed budget to "
                        f"{self.timesteps} (iterations {start_global_step} to "
                        f"{self.num_train_iters})."
                    )
            elif requested_timestep is not None:
                if loaded_timestep != self.resume_start_timestep:
                    raise RuntimeError(
                        "Could not load the requested resume checkpoint at timestep "
                        f"{self.resume_start_timestep}; loaded {loaded_timestep}."
                    )
                start_global_step = self.resume_start_global_step
                print(
                    "[resume] continuing fixed budget from timestep "
                    f"{loaded_timestep} to {self.timesteps} "
                    f"(iterations {start_global_step} to {self.num_train_iters})."
                )
        obs, info = self.init_env()

        # convert to cuda tensors
        self.policy_net.to(self.device)
        self.value_net.to(self.device)
        self.policy_obs_processor.to(self.device)
        self.value_obs_processor.to(self.device)
        if self.hand_obj_bps_decoder is not None:
            self.hand_obj_bps_decoder.to(self.device)
        if self.use_discriminator:
            self.discriminator.to(self.device)

        train_loop = tqdm(
            range(start_global_step, self.num_train_iters),
            total=self.num_train_iters - start_global_step,
            desc="Train",
        )
        ran_eval_code = (
            False  # running policy eval resets the env state, so obs and info
        )
        # become obsolete.

        completed_timesteps = start_global_step * self.per_loop_env_count
        for global_step in train_loop:

            self.env.set_mode("train")

            if ran_eval_code:
                obs, info = self.init_env()
                ran_eval_code = False

            train_loop.set_description(
                f"Train | iteration {global_step} | Sim Env Counter {self.env.env_step_counter}"
            )

            # collect rollout data for n_rollout_steps to fill the buffer

            if self.overheat_guard is not None:
                if self.overheat_guard.pause_needed():
                    self.save(
                        timestep=global_step * self.n_rollout_steps,
                        global_step=global_step,
                    )
                    self.overheat_guard.pause()

            log_extras = {}
            episode_task_metric_values = self._new_episode_task_metric_values()
            episode_progress_values = self._new_episode_progress_values()
            episode_checkpoint_metric_values: list[float] = []

            with torch.no_grad():
                for rollout_step in range(self.n_rollout_steps):

                    # get action from policy, also provide history if needed.
                    # history of obs can be obtained from the memory buffer.
                    if self.history_len == 0:
                        policy_history_obs = None
                        value_history_obs = None
                        history_dones = None
                    else:
                        history_obs, _, raw_history_dones = (
                            self.history.get_previous_timesteps_data(
                                "obs", "actions", "dones", device=self.device
                            )
                        )
                        policy_history_obs = history_obs
                        value_history_obs = history_obs
                        history_dones = self._history_dones_for_model(raw_history_dones)

                    action, log_prob = self.policy_net.act(
                        obs,
                        history_obs=policy_history_obs,
                        history_dones=history_dones,
                        update_stats=True,
                    )
                    values = self.value_net(
                        obs,
                        history_obs=value_history_obs,
                        history_dones=history_dones,
                        update_stats=True,
                    )

                    # step the environment
                    next_obs, reward, terminated, truncated, info = self.env.step(
                        action
                    )
                    next_obs = self._policy_obs(next_obs)
                    total_step = global_step * self.n_rollout_steps + rollout_step
                    # print(f"\rGlobal Step {global_step}, Rollout Step {rollout_step}, Total Step {total_step}", end='', flush=True)
                    # video recording hook
                    self.env.physics_manager.video_recording_hook(total_step)
                    self.update_log_extras(log_extras, info)
                    self._update_episode_task_metric_values(
                        episode_task_metric_values,
                        info,
                    )
                    self._update_episode_progress_values(
                        episode_progress_values,
                        info,
                    )
                    self._update_episode_checkpoint_metric_values(
                        episode_checkpoint_metric_values,
                        info,
                    )

                    dones = torch.logical_or(terminated, truncated)
                    memory_kwargs = {}
                    if self.use_hand_obj_bps_aux:
                        memory_kwargs["hand_obj_rel_bps"] = (
                            self._get_hand_obj_rel_bps_target(info)
                        )
                    memory_kwargs.update(self._get_reference_update_memory_kwargs())
                    # store transition in buffer
                    self.memory.add_samples(
                        obs=obs,
                        actions=action,
                        log_probs=log_prob.unsqueeze(-1),
                        rewards=reward.unsqueeze(-1),
                        terminated=terminated.unsqueeze(-1),
                        truncated=truncated.unsqueeze(-1),
                        dones=dones.unsqueeze(-1),
                        values=values.unsqueeze(-1),
                        **memory_kwargs,
                    )
                    if self.use_discriminator:
                        self.discriminator_memory.add_samples(
                            disc_obs=info["disc_obs"], dones=dones.unsqueeze(-1)
                        )

                    # update history buffer
                    if self.history_len > 0:
                        self.history.add_samples(
                            obs=obs, actions=action, dones=dones.unsqueeze(-1)
                        )

                    obs = next_obs

                # the memory buffer is now fully populated
                # we compute the advantages and target returns here
                if self.history_len > 0:
                    history_obs, _, history_dones = (
                        self.history.get_previous_timesteps_data(
                            "obs", "actions", "dones", device=self.device
                        )
                    )
                    value_history_obs = history_obs
                    history_dones = self._history_dones_for_model(history_dones)
                    last_values = self.value_net(
                        obs,
                        history_obs=value_history_obs,
                        history_dones=history_dones,
                        update_stats=True,
                    )
                else:
                    last_values = self.value_net(obs, update_stats=True)
                target_returns, advantages = self.compute_gae(last_values)

                # store advantages and target returns in memory
                self.memory._tensors["advantages"].copy_(advantages)
                self.memory._tensors["target_returns"].copy_(target_returns)
                # the idx for these should still remain at 0.

            task_metric_summary, task_metric_episode_count = (
                self._summarize_episode_task_metrics(episode_task_metric_values)
            )
            episode_progress_summary, completed_episode_count = (
                self._summarize_episode_progress_values(episode_progress_values)
            )
            checkpoint_metric_value = None
            checkpoint_metric_episode_count = len(episode_checkpoint_metric_values)
            checkpoint_metric_name = self.env.task_manager.checkpoint_metric_name
            if checkpoint_metric_episode_count > 0:
                if checkpoint_metric_name is None:
                    raise ValueError(
                        "Collected checkpoint metric values without a configured "
                        "checkpoint metric name."
                    )
                checkpoint_metric_value = float(
                    np.mean(
                        np.asarray(
                            episode_checkpoint_metric_values,
                            dtype=np.float64,
                        )
                    )
                )
            self._maybe_save_best_checkpoint_metric(
                checkpoint_metric_value,
                checkpoint_metric_episode_count,
                global_step,
            )

            # optimize the policy and value networks

            if self.history_len > 0:
                batch_size = min(self.batch_size, self.memory.total_seqs)
            else:
                batch_size = self.batch_size

            avg_return = self.memory.tensors["target_returns"].mean().item()
            self.return_queue.append(avg_return)
            rollout_log_payload = {
                "returns": sum(self.return_queue) / len(self.return_queue),
                "logs": self._filtered_train_log_extras(log_extras),
            }
            if task_metric_summary:
                rollout_log_payload.update(
                    {
                        "train_task_metrics": task_metric_summary,
                        "train_task_metric_episode_count": task_metric_episode_count,
                    }
                )
            if episode_progress_summary:
                rollout_log_payload.update(
                    {
                        "train_episode": episode_progress_summary,
                        "train_episode_count": completed_episode_count,
                    }
                )
            if checkpoint_metric_value is not None:
                rollout_log_payload.update(
                    {
                        "train_checkpoint_metric": {
                            checkpoint_metric_name: checkpoint_metric_value
                        },
                        "train_checkpoint_metric_episode_count": (
                            checkpoint_metric_episode_count
                        ),
                    }
                )

            # cast the model to train mode and device
            self.policy_net.train()
            self.value_net.train()
            if self.use_discriminator:
                self.discriminator.train()
                # check if memory is filled
                if self.discriminator_memory.filled:
                    self.discriminator.update_net(
                        timestep=(global_step + 1) * self.per_loop_env_count,
                        train_iters=global_step + 1,
                    )

            policy_losses = []
            value_losses = []
            combined_losses = []
            aux_losses = []
            ratios = []
            entropies = []
            entropy_losses = []

            for epoch in range(self.update_epochs):
                if self.history_len > 0:
                    sampler = self.memory.sample_sequence_indices(batch_size=batch_size)
                else:
                    sampler = self.memory.sample_indices(batch_size)

                for mb_idx, mem_indices in enumerate(sampler):

                    if self.history_len > 0:
                        batch = self.memory.query_sequence_indices(
                            mem_indices, seq_first=True, device=self.device
                        )
                    else:
                        batch = self.memory.query_indices(
                            mem_indices, device=self.device
                        )

                    # update policy network
                    sampled_actions = batch["actions"]
                    sampled_obs = batch["obs"]
                    # need to formulate the above as current_obs, history
                    if self.history_len > 0:
                        sampled_old_log_prob = batch["log_probs"][-1].squeeze(-1)
                        sampled_advantages = batch["advantages"][-1].squeeze(-1)
                        sampled_last_target_returns = batch["target_returns"][
                            -1
                        ].squeeze(-1)
                        sampled_last_obs = sampled_obs[-1]
                        sampled_last_action = sampled_actions[-1]
                        sampled_policy_history_obs = sampled_obs[:-1]
                        sampled_value_history_obs = sampled_obs[:-1]
                        sampled_history_dones = self._history_dones_for_model(
                            batch["dones"][:-1]
                        )
                        new_log_prob, entropy = self.policy_net.get_log_prob(
                            sampled_last_obs,
                            sampled_last_action,
                            history_obs=sampled_policy_history_obs,
                            history_dones=sampled_history_dones,
                        )
                        new_values = self.value_net(
                            sampled_last_obs,
                            history_obs=sampled_value_history_obs,
                            history_dones=sampled_history_dones,
                        )
                    else:
                        sampled_old_log_prob = batch["log_probs"].squeeze(-1)
                        sampled_advantages = batch["advantages"].squeeze(-1)
                        sampled_last_target_returns = batch["target_returns"].squeeze(
                            -1
                        )
                        new_log_prob, entropy = self.policy_net.get_log_prob(
                            sampled_obs, sampled_actions
                        )
                        new_values = self.value_net(sampled_obs)

                    assert (
                        sampled_advantages.shape
                        == new_log_prob.shape
                        == new_values.shape
                        == sampled_last_target_returns.shape
                    ), f"Shape mismatch: advantages {sampled_advantages.shape}, log_prob {new_log_prob.shape}, values {new_values.shape}, target_returns {sampled_last_target_returns.shape}"
                    ratio = torch.exp(new_log_prob - sampled_old_log_prob)
                    surrogate = ratio * sampled_advantages
                    clipped_surrogate = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    )
                    policy_loss = -torch.min(surrogate, clipped_surrogate).sum()

                    # entropy bonus
                    entropy_bonus = (
                        -self.entropy_coef * entropy.mean()
                    )  # higher entropy, more exploration

                    total_policy_loss = policy_loss + entropy_bonus
                    value_loss = self.value_loss_fn(
                        new_values, sampled_last_target_returns
                    )
                    aux_loss = torch.zeros((), device=self.device)
                    if self.policy_net.aux_decoder is not None:
                        if self.history_len > 0:
                            sampled_hand_obj_rel_bps = batch["hand_obj_rel_bps"][-1]
                            aux_pred = self.policy_net.predict_aux(sampled_last_obs)
                        else:
                            sampled_hand_obj_rel_bps = batch["hand_obj_rel_bps"]
                            aux_pred = self.policy_net.predict_aux(sampled_obs)
                        aux_loss = self.hand_obj_bps_aux_loss_fn(
                            aux_pred, sampled_hand_obj_rel_bps
                        )

                    combined_loss = (
                        total_policy_loss
                        + self.cfg.vf_coef * value_loss
                        + self.hand_obj_bps_aux_loss_coef * aux_loss
                    )

                    self.optimizer.zero_grad()
                    combined_loss.backward()
                    self.optimizer.step()

                    # logging
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    combined_losses.append(combined_loss.item())
                    if self.hand_obj_bps_decoder is not None:
                        aux_losses.append(aux_loss.item())
                    ratios.append(ratio.mean().item())
                    entropies.append(entropy.mean().item())
                    entropy_losses.append(entropy_bonus.item())

            mean_aux_loss = (
                sum(aux_losses) / len(aux_losses) if len(aux_losses) > 0 else 0.0
            )
            optimize_log_payload = {
                "policy_loss": sum(policy_losses) / len(policy_losses),
                "value_loss": sum(value_losses) / len(value_losses),
                "weighted_value_loss": (
                    self.cfg.vf_coef * sum(value_losses) / len(value_losses)
                ),
                "combined_loss": sum(combined_losses) / len(combined_losses),
                "entropy": sum(entropies) / len(entropies),
                "entropy_loss": sum(entropy_losses) / len(entropy_losses),
                "policy_lr": float(self.optimizer.param_groups[0]["lr"]),
                "value_lr": float(self.optimizer.param_groups[1]["lr"]),
                "ratio": sum(ratios) / len(ratios),
                "action_std": torch.exp(self.policy_net.log_std).mean().item(),
                "iteration": global_step + 1,
                "model": self._checkpoint_models(),
            }
            if not self.log_essential_metrics_only:
                optimize_log_payload.update(
                    {
                        "std": optimize_log_payload["action_std"],
                    }
                )
            if self.hand_obj_bps_decoder is not None:
                optimize_log_payload["hand_obj_bps_aux_loss"] = mean_aux_loss
            optimize_log_payload.update(rollout_log_payload)
            # Stop only after a complete PPO update and save its actual timestep.
            completed_timesteps = (global_step + 1) * self.per_loop_env_count
            stop_training = stop_requested is not None and stop_requested()
            self.logger.log(
                optimize_log_payload,
                timestep=(global_step + 1) * self.per_loop_env_count,
                train_iters=global_step + 1,
                force_model_save=stop_training or (global_step == (self.num_train_iters - 1)),
                force_log=bool(
                    task_metric_summary
                    or episode_progress_summary
                    or checkpoint_metric_value is not None
                ),
            )

            reference_update_stats = self._maybe_update_reference_motion(
                global_step=global_step
            )
            if (
                reference_update_stats is not None
                and not self.log_essential_metrics_only
            ):
                self.logger.log(
                    {"logs": reference_update_stats},
                    timestep=(global_step + 1) * self.per_loop_env_count,
                    train_iters=global_step + 1,
                )

            if stop_training:
                break

            # periodic evaluation (metrics and video can have separate cadences)
            should_eval_metrics = (self.eval_metrics_freq > 0) and (
                (global_step + 1) % self.eval_metrics_freq == 0
            )
            should_log_video = (self.eval_video_freq > 0) and (
                (global_step + 1) % self.eval_video_freq == 0
            )
            if (should_eval_metrics or should_log_video) and (
                not self.logger.cfg.disable
            ):
                self._eval_current_policy(
                    train_iter=global_step + 1,
                    timestep=(global_step + 1) * self.per_loop_env_count,
                    eval_cfg=eval_cfg,
                    run_metrics=should_eval_metrics,
                    log_video=should_log_video,
                )
                ran_eval_code = True
                self.env.set_mode("train")

            self.policy_net.eval()
            self.value_net.eval()
            if self.hand_obj_bps_decoder is not None:
                self.hand_obj_bps_decoder.eval()

        return completed_timesteps

    def _eval_current_policy(
        self,
        train_iter,
        timestep,
        eval_cfg: EvalConfig | None = None,
        run_metrics: bool = True,
        log_video: bool = True,
    ):
        """Evaluate current policy and log requested eval artifacts."""
        self.env.set_mode("eval")
        n_eval_envs = self.cfg.num_eval_envs

        obs, info = self.env.reset(seed=self.seed, options={"gt_info": True})
        obs = self._policy_obs(obs)
        if self.history_len > 0:
            self.reset_history_buffer()

        if self.env.physics_manager.is_recording:
            save_fname = f"{self.env.physics_manager.logs_dir}/{timestep}_train.mp4"
            self.env.physics_manager.stop_recording(save_to_filename=save_fname, fps=30)
        if log_video:
            # start recording eval episodes
            self.env.physics_manager.start_recording()
            self.env.physics_manager.video_recording_hook(0)

        episode_returns = np.zeros(self.env.num_envs, dtype=np.float32)
        log_extras = {}
        traj_log = (
            TrajInfoBatched(f"eval_{timestep}", self.env.num_envs)
            if run_metrics
            else None
        )
        completion_mask = np.zeros(self.env.num_envs, dtype=bool)
        completion_mask[n_eval_envs:] = True  # only care about first n_eval_envs

        print("Starting evaluation...")
        n_eval_steps = 0
        self.policy_net.eval()
        with torch.no_grad():
            # for step_idx in range(max_timesteps):
            while True:
                print(f"\rEval Step {n_eval_steps}", end="", flush=True)
                history = None
                if self.history_len > 0:
                    history_obs, _, history_dones = (
                        self.history.get_previous_timesteps_data(
                            "obs", "actions", "dones", device=self.device
                        )
                    )
                    history = (
                        history_obs,
                        self._history_dones_for_model(history_dones),
                    )

                if history is None:
                    action, _ = self.policy_net.act_greedy(obs)
                else:
                    history_obs, history_dones = history
                    action, _ = self.policy_net.act_greedy(
                        obs,
                        history_obs=history_obs,
                        history_dones=history_dones,
                    )

                next_obs, reward, terminated, truncated, info = self.env.step(
                    action, compute_gt=True
                )
                next_obs = self._policy_obs(next_obs)

                n_eval_steps += 1
                if log_video:
                    self.env.physics_manager.video_recording_hook(0)
                dones = terminated | truncated
                episode_returns += reward.cpu().numpy() * (
                    1 - completion_mask.astype(float)
                )

                if run_metrics and (traj_log is not None):
                    traj_info = {"reward": reward, "done": dones}
                    if "gt_info" in info:
                        traj_info.update(info["gt_info"])
                    traj_log.append(traj_info)

                self.update_log_extras(log_extras, info)

                if self.history_len > 0:
                    self.history.add_samples(
                        obs=obs, actions=action, dones=dones.unsqueeze(-1)
                    )

                completion_mask = np.logical_or(completion_mask, dones.cpu().numpy())
                if completion_mask.all():
                    print(f"\nAll eval envs completed after {n_eval_steps} steps.")
                    break

                obs = next_obs

        # stop recording
        fname = Path(self.env.physics_manager.logs_dir) / f"eval_{timestep}.mp4"
        fname.parent.mkdir(parents=True, exist_ok=True)
        self.env.physics_manager.stop_recording(save_to_filename=fname, fps=30)

        eval_log_data = {
            "periodic_eval/returns": episode_returns.mean().item(),
            "periodic_eval/logs": {k: np.mean(v) for k, v in log_extras.items()},
        }
        eval_log_data["periodic_eval/logs"].update(
            self.env.eval_would_termination_summary(n_eval_envs)
        )
        if run_metrics and (traj_log is not None):
            save_dir = self.logger.experiment_dir / "eval_runs"
            paths = traj_log.save_per_env(save_dir, max_envs=n_eval_envs)

            if self.eval_cfg is None:
                self.eval_cfg = EvalConfig(logdir=str(self.logger.experiment_dir))
            eval_metrics = EvalMetrics(self.eval_cfg, self.env.cfg)
            prop_cols = eval_metrics.property_fields
            val_cols = eval_metrics.value_fields
            dfs = []
            for path in paths:
                eval_metrics.set_demo(path)
                df = eval_metrics.compute_metrics(
                    only_demo=True, demo_name=path.stem, save_pd=False
                )
                # replace all occurrences of demo_name with just "demo" (will make it easy to combine the dfs)
                df = df.replace(path.stem, "demo")
                dfs.append(df)

            combined_df = (
                pd.concat(
                    dfs,
                    ignore_index=True,
                )
                .groupby(prop_cols, as_index=False)
                .agg({k: list for k in val_cols})
            )

            # dump this combined df to csv
            combined_csv_path = save_dir / f"combined_metrics_{timestep}.csv"
            combined_df.to_csv(combined_csv_path, index=False)
            eval_log_data["Metrics"] = {
                f"{'/'.join(SHORT_NAMES.get(str(field), str(field)) for field in row[prop_cols])}/{SHORT_NAMES.get(v, v.lower())}": float(
                    np.mean(row[v])
                )
                for _, row in combined_df.iterrows()
                for v in val_cols
            }

        self.logger.log(
            eval_log_data,
            timestep=timestep,
            train_iters=train_iter,  # doesn't really matter,
            force_log=True,
        )
