from __future__ import annotations

from typing import TYPE_CHECKING, List

import gymnasium as gym
import numpy as np
import torch

from dexlink.src.rl.sim.action_manager import ActionManager
from dexlink.src.rl.sim.commands_manager import CommandsManager
from dexlink.src.rl.sim.curriculum_manager import CurriculumManager
from dexlink.src.rl.sim.disc_manager import DiscObsManager
from dexlink.src.rl.sim.obs_manager import ObservationManager
from dexlink.src.rl.sim.reset_manager import ResetManager
from dexlink.src.rl.sim.reward_manager_v2 import RewardManager
from dexlink.src.rl.sim.scene_manager import SceneManager

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import EnvConfig


class Env(gym.Env):

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.obs_cfg = cfg.obs_cfg
        self.scene_cfg = cfg.scene_cfg
        self.action_cfg = cfg.action_cfg
        self.disc_cfg = cfg.disc_cfg
        self.rewards_cfg = cfg.rewards_cfg
        self.device = cfg.device
        self.num_envs = self.cfg.scene_cfg.num_envs

        self.scene_manager = SceneManager(cfg.scene_cfg, self)
        self.scene_manager.initialize()  # initializes the sim and the scene

        self.action_manager = ActionManager(cfg.action_cfg, self)

        self.commands_manager = CommandsManager(cfg.commands_cfg, self)
        self.observation_manager = ObservationManager(cfg.obs_cfg, self)

        self.use_discriminator = self.cfg.rewards_cfg.discriminator_weight > 0.0
        if self.use_discriminator:
            self.disc_observation_manager = DiscObsManager(cfg.disc_obs_cfg, self)

        self.reward_manager = RewardManager(cfg.rewards_cfg, self)
        self.reset_manager = ResetManager(cfg, self)
        self.curriculum_manager = CurriculumManager(cfg.curriculum_cfg, self)

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

    def initialize(self):
        # RL env specific initialization

        # set action and observation spaces, used by agents
        self.action_dim = self.action_manager.action_dim
        self.observation_dim = self.observation_manager.observation_dim
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
        )  # to test for truncation
        self.is_first_stage = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )  # all envs start in first stage where the hands first need to reach valid state wrt object

        self.current_obs = torch.empty(self.num_envs, self.observation_dim)
        self.current_named_obs = None
        self.current_disc_obs = (
            torch.empty(self.num_envs, self.disc_observation_manager.observation_dim)
            if self.use_discriminator
            else None
        )
        self.extras = {}

    def get_command_timestep_bounds(self) -> tuple[int, int]:
        min_timestep = 0
        max_timestep = self.scene_manager.total_steps - 1
        original_demo = self.scene_manager.original_demo
        if hasattr(original_demo, "n_reach_frames"):
            min_timestep = int(original_demo.n_reach_frames)
        if min_timestep > max_timestep:
            raise ValueError(
                f"Invalid command timestep bounds: min_timestep={min_timestep}, "
                f"max_timestep={max_timestep}."
            )
        return min_timestep, max_timestep

    def clamp_command_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        min_timestep, max_timestep = self.get_command_timestep_bounds()
        return torch.clamp(timesteps, min=min_timestep, max=max_timestep)

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

        if self.mode == "eval":
            # all envs start from the beginning in eval mode
            start_frame_idx = np.full(
                len(env_ids), self.cfg.reset_cfg.eval_start_timestep
            )
        else:
            start_idx, end_idx = self.cfg.reset_cfg.start_range
            if end_idx < 0:
                end_idx = self.scene_manager.total_steps + end_idx

            start_frame_idx = np.random.randint(
                start_idx, end_idx + 1, size=len(env_ids)
            )

        self.start_timestep[env_ids] = torch.from_numpy(start_frame_idx)
        command_steps = np.random.randint(
            self.cfg.obs_cfg.command_steps_range[0],
            self.cfg.obs_cfg.command_steps_range[1] + 1,
            size=len(env_ids),
        )
        command_steps = (
            self.start_timestep[env_ids] + torch.from_numpy(command_steps)
        ).to(self.device)
        self.command_time_step[env_ids] = self.clamp_command_timesteps(command_steps)

        # this is the length of the task given. ideally this should be the horizon for the rollout.
        self.n_step_future_cmd = self.command_time_step - self.start_timestep.to(
            self.device
        )

        if self.obs_cfg.task_command or (self.rewards_cfg.task_weight > 0.0):
            self.commands_manager.reset_obs_commands(env_ids)
        # if self.rewards_cfg.task_weight > 0.0:
        #     self.commands_manager.reset_reward_commands(env_ids)

        # pos, _, _ = self.reset_manager.set_object_state(start_frame_idx, env_ids)
        self.reset_manager.set_object_state_from_retargeted_results(
            start_frame_idx,
            env_ids,
        )

        if not self.scene_manager.object_only:
            self.reset_manager.set_robot_state_from_retargeted_results(
                start_frame_idx, env_ids
            )
            # random_actions = self.action_manager.sample_random_action(
            #     pos, env_ids=env_ids
            # )
            # self.reset_manager.set_robot_state(
            #     start_frame_idx, env_ids, random_actions=random_actions
            # )

        self.reset_manager.apply_domain_randomization(env_ids)

        self.timesteps[env_ids] = 0
        self.episode_length[env_ids] = 0
        self.is_first_stage[env_ids] = True

    def set_mode(self, mode: str):  # mode is 'train' or 'eval'
        # in eval mode, early termination is disabled, and commands are updated
        # assuming the previous stages were successfully completed.
        self.mode = mode

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.num_resets += 1
        self.reset_env_idxs(env_ids=None)
        obs = self.observation_manager.compute_observations()
        named_obs = self.observation_manager.named_view(obs)
        self.current_obs = obs
        self.current_named_obs = named_obs

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

        # compute obj pcd
        self._current_obj_pcd = self.observation_manager._get_object_pcd_from_named_obs(
            named_obs=named_obs,
            normals=True,
        )

        return self.current_obs, self.extras

    def step(self, actions: torch.Tensor, compute_gt: bool = False):

        assert actions.shape == (
            self.num_envs,
            self.action_dim,
        ), f"Action shape mismatch: expected {(self.num_envs, self.action_dim)}, got {actions.shape}"

        self.action_manager.apply_actions(actions)

        self.timesteps += 1
        self.env_step_counter += 1
        self.episode_length += 1

        self.extras = {}  # empty extras
        self.curriculum_manager.update()

        obs = self.observation_manager.compute_observations()
        named_obs = self.observation_manager.named_view(obs)

        self._prev_obj_pcd = self._current_obj_pcd
        # precompute object pcd if needed for discriminator and reward
        self._current_obj_pcd = self.observation_manager._get_object_pcd_from_named_obs(
            named_obs=named_obs,
            normals=True,
        )

        disc_obs = (
            self.disc_observation_manager.compute_disc_obs(
                current_named_obs=named_obs, prev_named_obs=self.current_named_obs
            )
            if self.use_discriminator
            else None
        )

        rewards, terminated, truncated = self.reward_manager.compute_rewards(
            named_obs, disc_obs, actions, prev_obs=self.current_obs
        )

        obs_dims = tuple(np.arange(1, len(obs.shape)))  # all dims except first
        is_nan_env = torch.isnan(obs).any(dim=obs_dims)

        num_nan_envs = is_nan_env.sum().item()
        if num_nan_envs > 0:
            print(
                f"Warning: {num_nan_envs} envs have NaN observations. Resetting those envs."
            )

        truncated = truncated | is_nan_env

        reset_env_ids = torch.where(terminated | truncated)[0]

        if len(reset_env_ids) > 0:
            # save the obs which will be overwritten after reset
            self.extras["pre_reset_obs"] = obs[reset_env_ids].clone()
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
            obs = self.observation_manager.compute_observations()
            named_obs = self.observation_manager.named_view(obs)

            # recompute obj pcd for reset envs
            self._current_obj_pcd[reset_env_ids] = (
                self.observation_manager.compute_object_pcd_obs(
                    pos=named_obs["obj_pos"][reset_env_ids],
                    quat=named_obs["obj_quat"][reset_env_ids],
                    arti=named_obs["obj_arti"][reset_env_ids],
                    normals=True,
                )
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

        self.current_obs = obs
        self.current_named_obs = named_obs

        if compute_gt:
            gt_info = self.observation_manager.get_ground_truth_info()
            self.extras["gt_info"] = gt_info

        if self.use_discriminator:
            self.current_disc_obs = disc_obs
            self.extras["disc_obs"] = disc_obs.clone()

        return obs, rewards, terminated, truncated, self.extras
