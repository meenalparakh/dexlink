from __future__ import annotations

from typing import TYPE_CHECKING, List

import gymnasium as gym
import numpy as np
import torch

from dexlink.src.rl.world.action_manager_v2 import ActionManager
from dexlink.src.rl.world.commands_manager import CommandsManager
from dexlink.src.rl.world.curriculum_manager import CurriculumManager
from dexlink.src.rl.world.disc_manager import DiscObsManager
from dexlink.src.rl.world.obs_manager import ObservationManager
from dexlink.src.rl.world.reference_trajectory_manager import ReferenceTrajectoryManager
from dexlink.src.rl.world.reset_manager import ResetManager
from dexlink.src.rl.world.reward_manager import RewardManager
from dexlink.src.rl.world.scene_manager import SceneManager
from dexlink.src.rl.world.task_manager import TaskManager

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import EnvConfig


class Env(gym.Env):

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.obs_cfg = cfg.obs_cfg
        self.task_cfg = cfg.task_cfg
        self.sim_cfg = cfg.sim_cfg
        self.real_cfg = cfg.real_cfg
        self.action_cfg = cfg.action_cfg
        self.disc_cfg = cfg.disc_cfg
        self.rewards_cfg = cfg.rewards_cfg
        self.device = cfg.device
        self.num_envs = 1 if self.cfg.backend == "real" else self.cfg.sim_cfg.num_envs
        self.visualize_policy_pcd = bool(cfg.visualize_policy_pcd)
        self.visualize_policy_pcd_on = str(cfg.visualize_policy_pcd_on)
        self.visualize_policy_pcd_point_size = float(
            cfg.visualize_policy_pcd_point_size
        )

        self.scene_manager = SceneManager(self)
        self.task_manager = self._create_task_manager()
        self.physics_manager = self._create_physics_manager()

        self.task_manager.initialize()
        self.physics_manager.initialize(
            camera_lookat=self.task_manager.compute_camera_lookat()
        )
        self.scene_manager.initialize()
        self.reference_trajectory_manager = ReferenceTrajectoryManager(
            cfg.reference_cfg, self
        )
        self.reference_trajectory_manager.initialize()

        self.action_manager = self._create_action_manager()

        self.commands_manager = self._create_commands_manager()
        self.observation_manager = self._create_observation_manager()

        self.use_discriminator = self.cfg.rewards_cfg.discriminator_weight > 0.0
        if self.use_discriminator:
            self.disc_observation_manager = self._create_disc_observation_manager()

        self.reward_manager = self._create_reward_manager()
        self.reset_manager = self._create_reset_manager()
        self.curriculum_manager = self._create_curriculum_manager()

        self.action_manager.initialize()
        self.observation_manager.initialize()
        self.reward_manager.initialize()
        self.commands_manager.initialize()
        self.reset_manager.initialize()
        self.curriculum_manager.initialize()

        if self.use_discriminator:
            self.disc_observation_manager.initialize()

        self.initialize()
        self._print_manager_summary()

    def _create_physics_manager(self):
        if self.cfg.backend == "sim":
            from dexlink.src.rl.world.physics_sim import PhysicsSim

            return PhysicsSim(self.cfg.sim_cfg, self)
        if self.cfg.backend == "real":
            from dexlink.src.rl.world.physics_real import PhysicsReal

            return PhysicsReal(self.cfg.real_cfg, self)
        raise ValueError(
            f"Expected cfg.backend to be 'sim' or 'real', got {self.cfg.backend!r}."
        )

    def _create_task_manager(self):
        return TaskManager(self.cfg.task_cfg, self)

    def _create_action_manager(self):
        if self.cfg.action_cfg.action_mode == "absolute_target":
            from dexlink.src.rl.world.target_action_manager import (
                AbsoluteTargetActionManager,
            )

            return AbsoluteTargetActionManager(self.cfg.action_cfg, self)
        return ActionManager(self.cfg.action_cfg, self)

    def _create_commands_manager(self):
        return CommandsManager(self.cfg.commands_cfg, self)

    def _create_observation_manager(self):
        return ObservationManager(self.cfg.obs_cfg, self)

    def _create_disc_observation_manager(self):
        return DiscObsManager(self.cfg.disc_obs_cfg, self)

    def _create_reward_manager(self):
        return RewardManager(self.cfg.rewards_cfg, self)

    def _create_reset_manager(self):
        return ResetManager(self.cfg, self)

    def _create_curriculum_manager(self):
        return CurriculumManager(self.cfg.curriculum_cfg, self)

    def initialize(self):
        # RL env specific initialization

        # set action and observation spaces, used by agents
        self.action_dim = self.action_manager.action_dim
        self.observation_dim = self.observation_manager.observation_dim
        self.privileged_observation_dim = (
            self.observation_manager.privileged_observation_dim
        )
        self.mode = "train"  # 'train' or 'eval'

        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, self.action_dim),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_envs, self.observation_dim),
            dtype=np.float32,
        )

        # for each env, how many steps from reset have we taken.
        self.episode_length = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # initialize buffers and counters
        self.num_resets = 0
        self.env_step_counter = 0
        # self.n_terminated = 0
        # self.n_truncated = 0
        self.start_timestep = torch.zeros(self.num_envs, dtype=torch.long, device="cpu")
        self.command_time_step = torch.empty(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self.timesteps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )  # elapsed steps since reset
        self.eval_would_terminate_timestep = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.eval_would_terminate_progress = torch.full(
            (self.num_envs,), -1.0, dtype=torch.float32, device=self.device
        )
        self.command_timestep_substeps = 1

        self.current_obs = None
        self.current_policy_obs = torch.empty(self.num_envs, self.observation_dim)
        self.current_privileged_obs = torch.empty(
            self.num_envs, self.privileged_observation_dim
        )
        self.current_wrist_pos_by_side: dict[str, torch.Tensor] = {}
        self.current_wrist_quat_by_side: dict[str, torch.Tensor] = {}
        self.current_wrist_control_jpos_by_side: dict[str, torch.Tensor] = {}
        self.current_named_policy_obs = None
        self.current_named_privileged_obs = None
        self.previous_position_targets_by_side: dict[str, torch.Tensor] = {}
        self.current_action_probe: dict | None = None
        self.current_action_probe_by_side: dict[str, dict] = {}
        self.current_contact_probe: dict | None = None
        self.current_disc_obs = (
            torch.empty(self.num_envs, self.disc_observation_manager.observation_dim)
            if self.use_discriminator
            else None
        )
        self.extras = {}

    def refresh_reference_trajectory(self) -> None:
        self.reset_manager.sync_reference_buffers()
        self.commands_manager.refresh_reference_buffers()
        self.reward_manager.refresh_reference_buffers()

    def update_reference_trajectory(
        self,
        *,
        timesteps,
        returns,
        object_pos=None,
        object_quat=None,
        hand_wrist_pos=None,
        hand_wrist_quat=None,
        hand_joint_positions=None,
        anchors=None,
        policy_iter: int | None = None,
    ) -> dict:
        anchors = self.reset_manager.reference_update_anchors(anchors)
        stats = self.reference_trajectory_manager.update_from_rollout_batch(
            timesteps=timesteps,
            returns=returns,
            object_pos=object_pos,
            object_quat=object_quat,
            hand_wrist_pos=hand_wrist_pos,
            hand_wrist_quat=hand_wrist_quat,
            hand_joint_positions=hand_joint_positions,
            anchors=anchors,
        )
        if anchors is not None:
            stats["reference_anchor_count"] = float(len(anchors))
        self.refresh_reference_trajectory()
        if policy_iter is not None:
            stats.update(
                self.reset_manager.maybe_advance_reverse_curriculum(
                    policy_iter=policy_iter,
                    reference_update_stats=stats,
                )
            )
        return stats

    def get_command_timestep_bounds(self) -> tuple[int, int]:
        min_timestep = 0
        max_timestep = self.task_manager.total_steps - 1
        if min_timestep > max_timestep:
            raise ValueError(
                f"Invalid command timestep bounds: min_timestep={min_timestep}, "
                f"max_timestep={max_timestep}."
            )
        return min_timestep, max_timestep

    def clamp_command_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        min_timestep, max_timestep = self.get_command_timestep_bounds()
        return torch.clamp(timesteps, min=min_timestep, max=max_timestep)

    def update_command_time_step(
        self, env_ids: List[int] | np.ndarray | None = None
    ) -> None:
        env_ids = self.reset_manager.normalize_env_ids(env_ids)
        command_timesteps = self.get_reference_timesteps_unclamped(env_ids)
        self.command_time_step[env_ids] = self.clamp_command_timesteps(
            command_timesteps
        )

    def get_reference_timesteps_unclamped(
        self, env_ids: List[int] | np.ndarray | None = None
    ) -> torch.Tensor:
        env_ids = self.reset_manager.normalize_env_ids(env_ids)
        elapsed_timesteps = self.timesteps[env_ids]
        if self.cfg.backend == "real" and self.mode == "eval":
            substeps = int(self.command_timestep_substeps)
            if substeps <= 0:
                raise ValueError(
                    "Expected command_timestep_substeps > 0, got "
                    f"{self.command_timestep_substeps}."
                )
            elapsed_timesteps = torch.div(
                elapsed_timesteps,
                substeps,
                rounding_mode="floor",
            )
        return self.start_timestep[env_ids].to(self.device) + elapsed_timesteps

    def _print_manager_summary(self):
        lines = []
        lines.append(self.observation_manager.summary())
        lines.append(self.action_manager.summary())
        lines.append(self.commands_manager.summary())
        lines.append(self.reward_manager.summary())

        if self.use_discriminator:
            lines.append(self.disc_observation_manager.summary())

        line_divider = "\n" + "=" * 36 + "\n"
        divider = "=" * 60
        summary = line_divider.join(lines)
        print(f"{divider}\n{summary}\n{divider}")

    def reset_env_idxs(self, env_ids: List[int] | np.ndarray | None = None):

        # print(f"At step {self.env_step_counter}, resetting {len(env_ids) if env_ids is not None else self.num_envs} envs.")

        env_ids = self.reset_manager.normalize_env_ids(env_ids)
        self.reset_manager.resample_random_force(env_ids)
        start_frame_idx = self.reset_manager.sample_start_timesteps(env_ids)
        reset_frame_idx = self.reset_manager.clamp_to_reference_frame_timesteps(
            start_frame_idx
        )

        self.start_timestep[env_ids] = torch.from_numpy(start_frame_idx)
        # pos, _, _ = self.reset_manager.set_object_state(start_frame_idx, env_ids)
        if self.mode == "eval":
            self.reset_manager.set_object_state_for_eval(
                reset_frame_idx,
                env_ids,
            )
        else:
            self.reset_manager.set_object_state_from_retargeted_results(
                reset_frame_idx,
                env_ids,
            )
        self.reset_manager.set_robot_state_from_retargeted_results(
            reset_frame_idx, env_ids
        )
        used_action_manager_reset = False
        if not self.task_manager.object_only:
            # Grasp refinement starts with an unsupported dynamic object. Moving
            # the hand through simulation during eval reset lets gravity drop the
            # object before the rollout begins, unlike the synchronized direct
            # state reset used for training.
            if self.mode == "eval" and not self.task_cfg.grasp_refinement_mode:
                self.reset_manager.reset_robot_state_for_eval(reset_frame_idx, env_ids)
                used_action_manager_reset = True

            if not used_action_manager_reset:
                self.physics_manager.step()
            for side, spec in self.action_manager.hand_specs.items():
                full_current_qpos = self.physics_manager.get_hand_dof_positions(
                    spec.hand, spec.hand.dof_map
                ).detach()
                if self.cfg.backend == "real":
                    full_current_qpos = full_current_qpos.to("cpu")
                current_qpos = full_current_qpos[env_ids]
                if side not in self.previous_position_targets_by_side:
                    self.previous_position_targets_by_side[side] = (
                        full_current_qpos.clone()
                    )
                else:
                    self.previous_position_targets_by_side[side][env_ids] = current_qpos
            # random_actions = self.action_manager.sample_random_action(
            #     pos, env_ids=env_ids
            # )
            # self.reset_manager.set_robot_state(
            #     start_frame_idx, env_ids, random_actions=random_actions
            # )
        else:
            self.physics_manager.step()

        # Domain randomization is applied once across all parallel environments
        # during ResetManager.initialize(). Keep each environment's sampled
        # dynamics fixed across episode resets.

        self.timesteps[env_ids] = 0
        self.eval_would_terminate_timestep[env_ids] = -1
        self.eval_would_terminate_progress[env_ids] = -1.0
        self.update_command_time_step(env_ids)
        self.episode_length[env_ids] = 0
        self.task_manager.reset_task_metric_accumulators(env_ids)

    def set_mode(self, mode: str):  # mode is 'train' or 'eval'
        self.mode = mode

    def eval_episode_horizon(self) -> torch.Tensor:
        max_reference_timestep = self.task_manager.total_steps - 1
        hold_steps = int(self.cfg.rewards_cfg.last_reference_hold_steps)
        horizon = torch.full(
            (self.num_envs,),
            max_reference_timestep + hold_steps,
            dtype=torch.float32,
            device=self.start_timestep.device,
        ) - self.start_timestep.to(torch.float32)
        return horizon.to(self.timesteps.device).clamp_min(1.0)

    def record_eval_would_termination(self, would_terminate: torch.Tensor) -> None:
        if self.mode != "eval":
            return
        expected_shape = (self.num_envs,)
        if tuple(would_terminate.shape) != expected_shape:
            raise ValueError(
                "Expected eval would-termination mask shape "
                f"{expected_shape}, got {tuple(would_terminate.shape)}."
            )
        newly_terminated = torch.logical_and(
            would_terminate,
            self.eval_would_terminate_timestep < 0,
        )
        if not torch.any(newly_terminated):
            return
        self.eval_would_terminate_timestep[newly_terminated] = self.timesteps[
            newly_terminated
        ]
        progress = self.timesteps.to(torch.float32) / self.eval_episode_horizon()
        self.eval_would_terminate_progress[newly_terminated] = progress[
            newly_terminated
        ]

    def eval_would_termination_summary(self, n_envs: int) -> dict[str, float]:
        if n_envs <= 0 or n_envs > self.num_envs:
            raise ValueError(f"Expected n_envs in [1, {self.num_envs}], got {n_envs}.")
        recorded = self.eval_would_terminate_timestep[:n_envs] >= 0
        horizon = self.eval_episode_horizon()[:n_envs]
        timestep = torch.where(
            recorded,
            self.eval_would_terminate_timestep[:n_envs].to(torch.float32),
            horizon,
        )
        progress = torch.where(
            recorded,
            self.eval_would_terminate_progress[:n_envs],
            torch.ones_like(horizon),
        )
        return {
            "eval_would_terminate_rate": recorded.to(torch.float32).mean().item(),
            "eval_would_terminate_timestep": timestep.mean().item(),
            "eval_would_terminate_progress": progress.mean().item(),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.num_resets += 1
        self.current_action_probe = None
        self.current_action_probe_by_side = {}
        self.current_contact_probe = None
        if self.cfg.backend == "real":
            self.physics_manager.prepare_episode_scene()
        self.reset_env_idxs(env_ids=None)
        self.refresh_current_observations(visualize_event="reset")

        # discriminator obs is only needed for reward computation, not needed during reset
        # if self.use_discriminator:
        #     disc_obs = self.disc_observation_manager.compute_observations(
        #         _named_obs=named_obs
        #     )
        #     self.current_disc_obs = disc_obs
        #     self.extras["disc_obs"] = self.current_disc_obs

        if options and options.get("gt_info", False):
            gt_info = self.observation_manager.get_ground_truth_info()
            self.extras["gt_info"] = gt_info

        return self.current_obs, self.extras

    def _compute_observation_bundle(self):
        self.observation_manager.reset_step_cache()
        obs = self.observation_manager.compute_observations()
        policy_obs = obs["policy"]
        privileged_obs = obs["privileged"]
        named_policy_obs = self.observation_manager.named_view_policy(policy_obs)
        named_privileged_obs = self.observation_manager.named_view_privileged(
            privileged_obs
        )
        return obs, policy_obs, privileged_obs, named_policy_obs, named_privileged_obs

    def _store_current_observations(
        self,
        obs,
        policy_obs,
        privileged_obs,
        named_policy_obs,
        named_privileged_obs,
    ) -> None:
        self.current_obs = obs
        self.current_policy_obs = policy_obs
        self.current_privileged_obs = privileged_obs
        self.current_named_policy_obs = named_policy_obs
        self.current_named_privileged_obs = named_privileged_obs

    def refresh_current_observations(
        self, *, visualize_event: str | None = None
    ) -> dict[str, torch.Tensor]:
        (
            obs,
            policy_obs,
            privileged_obs,
            named_policy_obs,
            named_privileged_obs,
        ) = self._compute_observation_bundle()
        self._store_current_observations(
            obs,
            policy_obs,
            privileged_obs,
            named_policy_obs,
            named_privileged_obs,
        )
        if visualize_event is not None:
            self.scene_manager.maybe_visualize_policy_point_cloud(
                named_policy_obs, event=visualize_event
            )
        return obs

    def _apply_actions_and_advance(self, actions: torch.Tensor) -> None:

        assert actions.shape == (
            self.num_envs,
            self.action_dim,
        ), f"Action shape mismatch: expected {(self.num_envs, self.action_dim)}, got {actions.shape}"

        self.action_manager.apply_actions(actions)

        self.timesteps += 1
        self.update_command_time_step()
        self.env_step_counter += 1
        self.episode_length += 1

    def _finish_step_after_actions(
        self, actions: torch.Tensor, compute_gt: bool = False
    ):
        self.extras = {}  # empty extras
        self.curriculum_manager.update()

        prev_named_privileged_obs = self.current_named_privileged_obs
        obs, policy_obs, privileged_obs, named_policy_obs, named_privileged_obs = (
            self._compute_observation_bundle()
        )
        if (self.mode == "eval") and ("contact" in named_policy_obs):
            self.current_contact_probe = self.observation_manager.get_contact_probe(
                named_policy_obs
            )
        else:
            self.current_contact_probe = None

        disc_obs = (
            self.disc_observation_manager.compute_disc_obs(
                current_named_obs=named_privileged_obs,
                prev_named_obs=prev_named_privileged_obs,
            )
            if self.use_discriminator
            else None
        )

        if self.cfg.backend == "real" and self.mode == "eval":
            rewards = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            terminated = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            rewards, terminated, truncated = self.reward_manager.compute_rewards(
                named_privileged_obs,
                disc_obs,
                actions,
                prev_named_privileged_obs=prev_named_privileged_obs,
            )

        self.task_manager.update_task_metric_accumulators()

        policy_obs_dims = tuple(np.arange(1, len(policy_obs.shape)))
        privileged_obs_dims = tuple(np.arange(1, len(privileged_obs.shape)))
        is_nan_env = torch.isnan(policy_obs).any(dim=policy_obs_dims) | torch.isnan(
            privileged_obs
        ).any(dim=privileged_obs_dims)

        num_nan_envs = is_nan_env.sum().item()
        if num_nan_envs > 0:
            print(
                f"Warning: {num_nan_envs} envs have NaN observations. Resetting those envs."
            )

        truncated = truncated | is_nan_env

        reset_env_ids = torch.where(terminated | truncated)[0]

        if len(reset_env_ids) > 0:
            early_terminated = terminated[reset_env_ids]
            episode_span_steps = self.episode_length[reset_env_ids].to(torch.float32)
            episode_horizon_steps = self.eval_episode_horizon()[reset_env_ids]
            episode_progress_fraction = torch.clamp(
                episode_span_steps / episode_horizon_steps,
                min=0.0,
                max=1.0,
            )
            self.extras["episode_progress_fraction"] = episode_progress_fraction.clone()
            self.extras["episode_span_steps"] = episode_span_steps.clone()
            episode_task_metrics = self.task_manager.compute_episode_end_task_metrics(
                reset_env_ids,
                early_terminated,
            )
            if episode_task_metrics:
                self.extras["episode_task_metrics"] = episode_task_metrics
                self.extras["episode_task_metric_early_terminated"] = (
                    early_terminated.clone()
                )
            checkpoint_metric = None
            if self.mode == "train":
                checkpoint_metric = (
                    self.task_manager.compute_episode_end_checkpoint_metric(
                        reset_env_ids,
                        early_terminated,
                        episode_task_metrics,
                    )
                )
            if checkpoint_metric is not None:
                self.extras["episode_checkpoint_metric_name"] = (
                    self.task_manager.checkpoint_metric_name
                )
                self.extras["episode_checkpoint_metric_values"] = (
                    checkpoint_metric.clone()
                )

            # save the obs which will be overwritten after reset
            self.extras["pre_reset_obs"] = policy_obs[reset_env_ids].clone()
            gt_info = (
                self.observation_manager.get_ground_truth_info() if compute_gt else None
            )

            self.extras["pre_reset_gt_info"] = gt_info
            # if self.use_discriminator:
            #     self.extras["pre_reset_disc_obs"] = disc_obs[reset_env_ids].clone()

            self.extras["logs/avg_ep_len"] = self.episode_length[reset_env_ids].to(
                torch.float32
            )

            self.reset_env_idxs(reset_env_ids.cpu().numpy())
            obs, policy_obs, privileged_obs, named_policy_obs, named_privileged_obs = (
                self._compute_observation_bundle()
            )

            # disc_obs = (
            #     self.disc_observation_manager.compute_observations(named_obs)
            #     if self.use_discriminator
            #     else None
            # )

            # self.n_terminated += terminated.sum().item()
            # self.n_truncated += truncated.sum().item()
            # self.extras["log/truncated-total"] = self.n_truncated / (
            #     self.n_terminated + self.n_truncated
            # )

        # if torch.isnan(obs).any():
        #     raise ValueError("NaN detected in observations after step.")
        self._store_current_observations(
            obs,
            policy_obs,
            privileged_obs,
            named_policy_obs,
            named_privileged_obs,
        )
        self.scene_manager.maybe_visualize_policy_point_cloud(
            named_policy_obs, event="step"
        )

        if compute_gt:
            gt_info = self.observation_manager.get_ground_truth_info()
            self.extras["gt_info"] = gt_info

        if self.use_discriminator:
            self.current_disc_obs = disc_obs
            self.extras["disc_obs"] = disc_obs.clone()

        return obs, rewards, terminated, truncated, self.extras

    def step(self, actions: torch.Tensor, compute_gt: bool = False):
        self._apply_actions_and_advance(actions)
        return self._finish_step_after_actions(actions, compute_gt=compute_gt)

    def close(self):
        if hasattr(self.physics_manager, "close"):
            self.physics_manager.close()
        super().close()
