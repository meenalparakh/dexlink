from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from dexlink.src.rl.core_agent.agent_utils import (
    DiscObservationProcessor,
    ObservationProcessor,
    PolicyNetwork,
    ValueNetwork,
)
from dexlink.src.rl.core_agent.discriminator import Discriminator
from dexlink.src.rl.core_agent.memory import Memory
from dexlink.src.rl.core_agent.wandb_logger import WandbLogger
from dexlink.src.rl.config.base import AgentConfig, EvalConfig
from dexlink.utils.rl.overheat import load_overheat_guard
from dexlink.utils.retargeting.eval_utils import SHORT_NAMES, EvalMetrics, TrajInfoBatched

if TYPE_CHECKING:
    from dexlink.src.rl.sim.env import Env


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

        logger.configure_frequencies(self.num_train_iters)
        # Keep the legacy default: if eval config is provided, align legacy eval freq
        # with model-save cadence unless overridden by dedicated eval freqs.

        # save video more frequently than metrics, to get more eval snapshots, unless overridden by eval config
        self.eval_metrics_freq = logger.model_save_freq
        self.eval_video_freq = logger.model_save_freq
        # override
        if self.cfg.eval_metrics_freq > 0:
            self.eval_metrics_freq = self.cfg.eval_metrics_freq
        if self.cfg.eval_video_freq > 0:
            self.eval_video_freq = self.cfg.eval_video_freq

        # Initialize other PPO components here (networks, optimizers, etc.)
        self.device = env.cfg.device

        self.env = env

        self.obs_processor = ObservationProcessor(
            env,
            embed_dim=agent_cfg.obs_proc_embed_dim,
            layer_norm=agent_cfg.obs_proc_layer_norm,
        ).to(self.device)
        self.obs_dim = self.obs_processor.obs_dim
        self.act_dim = env.action_space.shape[1]  # remove the batch dimension
        self.policy_net = PolicyNetwork(
            self.obs_dim,
            self.act_dim,
            embed_dim=agent_cfg.policy_embed_dim,
            hand_synergy_dim=agent_cfg.hand_synergy_dim,
            history_len=self.history_len,
            device=self.device,
            # last_activation=None,
            last_activation=(
                nn.Tanh
                if (
                    env.cfg.action_cfg.mode in ["diff", "residual"]
                    or env.cfg.action_cfg.obj_dex_actions
                )
                else None
            ),
            action_residuals=(
                env.cfg.action_cfg.mode in ["residual", "diff"]
                or env.cfg.action_cfg.obj_dex_actions
            ),
        ).to(self.device)
        self.value_net = ValueNetwork(
            self.obs_dim,
            embed_dim=agent_cfg.value_embed_dim,
            history_len=self.history_len,
            device=self.device,
        ).to(self.device)

        shared_obs_params = list(self.obs_processor.parameters())
        self.optimizer = AdamW(
            [
                {
                    "params": list(self.policy_net.parameters()) + shared_obs_params,
                    "lr": self.policy_lr,
                },
                {
                    "params": self.value_net.parameters(),
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
        n_policy_params = sum(p.numel() for p in self.policy_net.parameters())
        n_obs_params = sum(p.numel() for p in self.obs_processor.parameters())
        n_value_params = sum(p.numel() for p in self.value_net.parameters())
        n_disc_params = (
            sum(p.numel() for p in self.discriminator.parameters())
            if self.use_discriminator
            else 0
        )

        # convert num params to millions
        n_policy_params = np.round(n_policy_params / 1e6, 3)
        n_obs_params = np.round(n_obs_params / 1e6, 3)
        n_value_params = np.round(n_value_params / 1e6, 3)
        n_disc_params = np.round(n_disc_params / 1e6, 3)

        info = {
            "obs_dim": self.obs_dim,
            "action_dim": self.act_dim,
            "n_obs_params": f"{n_obs_params} M",
            "n_policy_params": f"{n_policy_params} M",
            "n_value_params": f"{n_value_params} M",
            "n_disc_params": f"{n_disc_params} M",
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

    # def _print_lightning_summaries(self):
    #     def _summ(name, model, depth=2):
    #         try:
    #             summary = ModelSummary(model, max_depth=depth)
    #             print(f"\n{name} summary:\n{summary}")
    #         except Exception as e:
    #             print(f"\n{name} summary unavailable: {e}")

    #     _summ("Policy", self.policy_net)
    #     _summ("Value", self.value_net)
    #     if self.use_discriminator:
    #         _summ("Discriminator", self.discriminator)

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
                "obs": (self.obs_processor.obs_save_dim,),
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

    def _process_history_obs(
        self, history_obs: torch.Tensor | None
    ) -> torch.Tensor | None:
        if history_obs is None:
            return None
        if history_obs.shape[0] == 0:
            return torch.zeros(
                history_obs.shape[1],
                0,
                self.obs_dim,
                device=self.device,
            )

        seq_len, batch_size = history_obs.shape[:2]
        processed_history = self.obs_processor(
            history_obs.reshape(seq_len * batch_size, -1), update_stats=False
        )
        return processed_history.view(seq_len, batch_size, self.obs_dim).transpose(0, 1)

    def _get_history_inputs(
        self, history_obs: torch.Tensor | None, history_dones: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        processed_history_obs = self._process_history_obs(history_obs)
        processed_history_dones = None
        if history_dones is not None:
            processed_history_dones = history_dones.transpose(0, 1).to(self.device)
        return processed_history_obs, processed_history_dones

    def initialize_memory(self):
        self.memory = Memory(
            n_rollout_steps=self.n_rollout_steps,
            n_envs=self.env.num_envs,
            # history is past steps + 1 for current timestep
            seq_len=self.history_len + 1,
            name="ppo",
        )
        self.memory.create_tensors(
            {
                "obs": (self.obs_processor.obs_save_dim,),
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
        )

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
                if _k not in log_extras:
                    log_extras[_k] = []
                if isinstance(v, (float, int, np.float32, np.float64)):
                    log_extras[_k].append(v)
                elif isinstance(v, torch.Tensor):
                    log_extras[_k].append(v.float().mean().item())
            # except Exception as e:
            #     print(f"Error logging info key {k}: {e}")
            #     import pdb; pdb.set_trace()

    def save(self, timestep, global_step):
        # dump the model and optimizer states to logger

        models = {
            "policy": self.policy_net,
            "value": self.value_net,
            "optimizer": self.optimizer,
        }
        if self.use_discriminator:
            models["discriminator"] = self.discriminator
            models["discriminator_optimizer"] = self.discriminator.optimizer

        self.logger.log(
            {"model": models},
            timestep=timestep,
            force_model_save=True,
            train_iters=global_step,
        )

    def resume(self, timestep=None):
        # Search current run first, then other timestamped runs for same seed/run prefix.
        model_dir, timestep = self._resolve_resume_source(timestep=timestep)
        if model_dir is None or timestep is None:
            print("No saved models found to resume.")
            return

        print(f"Resuming from timestep: {timestep}")
        print(f"Resume source: {model_dir.parent}")
        policy_path = model_dir / str(timestep) / "policy.pt"
        value_path = model_dir / str(timestep) / "value.pt"
        disc_path = model_dir / str(timestep) / "discriminator.pt"
        opt_path = model_dir / str(timestep) / "optimizer.pt"
        disc_opt_path = model_dir / str(timestep) / "discriminator_optimizer.pt"

        assert policy_path.exists(), f"Policy model file not found: {policy_path}"
        assert value_path.exists(), f"Value model file not found: {value_path}"

        self.policy_net.load_state_dict(
            torch.load(policy_path, map_location=self.device)
        )
        self.value_net.load_state_dict(torch.load(value_path, map_location=self.device))

        if opt_path.exists():
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
            if disc_opt_path.exists():
                self.discriminator.optimizer.load_state_dict(
                    torch.load(disc_opt_path, map_location=self.device)
                )

    def _run_prefix_for_seed(self, run_name: str) -> str:
        """Return stable run prefix up to '-s<seed>', stripping optional timestamp."""
        # Support both '<name>-s42-03_03-13_41' and '<name>__s42-03_03-13_41' naming styles.
        match = re.match(r"^(.*(?:-s|__s)\d+)(?:-\d{2}_\d{2}-\d{2}_\d{2})?$", run_name)
        if match is not None:
            return match.group(1)
        return run_name

    def _latest_timestep_in_model_dir(self, model_dir: Path) -> int | None:
        if not model_dir.exists():
            return None
        policy_paths = list(model_dir.glob("*/policy.pt"))
        if len(policy_paths) == 0:
            return None
        return max(int(path.parent.name) for path in policy_paths)

    def _resolve_resume_source(self, timestep: int | None):
        current_run_dir = self.logger.experiment_dir
        group_dir = current_run_dir.parent
        run_prefix = self._run_prefix_for_seed(current_run_dir.name)

        print(f"[resume] current run dir: {current_run_dir}")
        print(f"[resume] group dir: {group_dir}")
        print(f"[resume] run prefix for seed: {run_prefix}")

        candidate_model_dirs = [current_run_dir / "models"]
        if group_dir.exists():
            for run_dir in group_dir.iterdir():
                if not run_dir.is_dir() or run_dir == current_run_dir:
                    continue
                if self._run_prefix_for_seed(run_dir.name) == run_prefix:
                    candidate_model_dirs.append(run_dir / "models")

        print("[resume] candidate model dirs:")
        for model_dir in candidate_model_dirs:
            print(f"  - {model_dir}")

        if timestep is not None:
            print(f"[resume] requested timestep: {timestep}")
            for model_dir in candidate_model_dirs:
                if (model_dir / str(timestep) / "policy.pt").exists():
                    print(f"[resume] found requested timestep in: {model_dir}")
                    return model_dir, timestep
                print(f"[resume] missing requested timestep in: {model_dir}")
            print("[resume] requested timestep was not found in any candidate dir.")
            return None, None

        best: tuple[int, float, Path] | None = None
        for model_dir in candidate_model_dirs:
            latest = self._latest_timestep_in_model_dir(model_dir)
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
        return best[2], best[0]

    def load(self, policy_ckpt):
        self.policy_net.load_state_dict(
            torch.load(policy_ckpt, map_location=self.device)
        )

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

        return obs, info

    def train(self, eval_cfg: EvalConfig | None = None):
        # trainer method
        # the logging convention here is
        # all logs are wrt the number of env steps taken so far (ignoring the parallel env count)
        # also save the data collection + update cycles (global_step)
        # the log_freq and model_save_freq though are in terms of global step.

        # resume from last checkpoint if any
        # by default all runs are suffixed with timestamp, so resuming happens only when
        # a full run name is provided and timestamp is excluded.
        self.resume()
        obs, info = self.init_env()

        # convert to cuda tensors
        self.policy_net.to(self.device)
        self.value_net.to(self.device)
        self.obs_processor.to(self.device)
        if self.use_discriminator:
            self.discriminator.to(self.device)

        train_loop = tqdm(
            range(self.num_train_iters), total=self.num_train_iters, desc="Train"
        )
        ran_eval_code = (
            False  # running policy eval resets the env state, so obs and info
        )
        # become obsolete.

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

            with torch.no_grad():
                for rollout_step in range(self.n_rollout_steps):

                    # get action from policy, also provide history if needed.
                    # history of obs can be obtained from the memory buffer.
                    if self.history_len == 0:
                        history_obs = None
                        history_dones = None
                    else:
                        history_obs, _, history_dones = (
                            self.history.get_previous_timesteps_data(
                                "obs", "actions", "dones", device=self.device
                            )
                        )
                        history_obs, history_dones = self._get_history_inputs(
                            history_obs=history_obs, history_dones=history_dones
                        )

                    processed_obs = self.obs_processor(obs, update_stats=True)
                    action, log_prob = self.policy_net.act(
                        processed_obs,
                        history_obs=history_obs,
                        history_dones=history_dones,
                    )
                    values = self.value_net(
                        processed_obs,
                        history_obs=history_obs,
                        history_dones=history_dones,
                    )

                    # step the environment
                    next_obs, reward, terminated, truncated, info = self.env.step(
                        action
                    )
                    total_step = global_step * self.n_rollout_steps + rollout_step
                    # print(f"\rGlobal Step {global_step}, Rollout Step {rollout_step}, Total Step {total_step}", end='', flush=True)
                    # video recording hook
                    self.env.scene_manager.video_recording_hook(total_step)
                    self.update_log_extras(log_extras, info)

                    dones = torch.logical_or(terminated, truncated)
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
                processed_obs = self.obs_processor(obs, update_stats=True)
                if self.history_len > 0:
                    history_obs, _, history_dones = (
                        self.history.get_previous_timesteps_data(
                            "obs", "actions", "dones", device=self.device
                        )
                    )
                    history_obs, history_dones = self._get_history_inputs(
                        history_obs=history_obs, history_dones=history_dones
                    )
                    last_values = self.value_net(
                        processed_obs,
                        history_obs=history_obs,
                        history_dones=history_dones,
                    )
                else:
                    last_values = self.value_net(processed_obs)
                target_returns, advantages = self.compute_gae(last_values)

                # store advantages and target returns in memory
                self.memory._tensors["advantages"].copy_(advantages)
                self.memory._tensors["target_returns"].copy_(target_returns)
                # the idx for these should still remain at 0.

            # optimize the policy and value networks

            if self.history_len > 0:
                batch_size = min(self.batch_size, self.memory.total_seqs)
            else:
                batch_size = self.batch_size

            avg_return = self.memory.tensors["target_returns"].mean().item()
            self.return_queue.append(avg_return)

            self.logger.log(
                {
                    "returns": sum(self.return_queue) / len(self.return_queue),
                    "logs": {k: np.mean(v) for k, v in log_extras.items()},
                },
                timestep=(global_step + 1) * self.per_loop_env_count,
                train_iters=global_step + 1,
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
            ratios = []
            entropies = []

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
                        sampled_history_obs, sampled_history_dones = (
                            self._get_history_inputs(
                                history_obs=sampled_obs[:-1],
                                history_dones=batch["dones"][:-1],
                            )
                        )
                        sampled_proc_last_obs = self.obs_processor(
                            sampled_last_obs, update_stats=False
                        )
                        new_log_prob, entropy = self.policy_net.get_log_prob(
                            sampled_proc_last_obs,
                            sampled_last_action,
                            history_obs=sampled_history_obs,
                            history_dones=sampled_history_dones,
                        )
                        new_values = self.value_net(
                            sampled_proc_last_obs,
                            history_obs=sampled_history_obs,
                            history_dones=sampled_history_dones,
                        )
                    else:
                        sampled_old_log_prob = batch["log_probs"].squeeze(-1)
                        sampled_advantages = batch["advantages"].squeeze(-1)
                        sampled_last_target_returns = batch["target_returns"].squeeze(
                            -1
                        )
                        sampled_proc_obs = self.obs_processor(
                            sampled_obs, update_stats=False
                        )
                        new_log_prob, entropy = self.policy_net.get_log_prob(
                            sampled_proc_obs, sampled_actions
                        )
                        new_values = self.value_net(sampled_proc_obs)

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

                    combined_loss = total_policy_loss + self.cfg.vf_coef * value_loss

                    self.optimizer.zero_grad()
                    combined_loss.backward()
                    self.optimizer.step()

                    # logging
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item() * self.cfg.vf_coef)
                    ratios.append(ratio.mean().item())
                    entropies.append(entropy_bonus.item())

            self.logger.log(
                {
                    "policy_loss": sum(policy_losses) / len(policy_losses),
                    "value_loss": sum(value_losses) / len(value_losses),
                    "ratio": sum(ratios) / len(ratios),
                    "entropy": sum(entropies) / len(entropies),
                    "std": torch.exp(self.policy_net.log_std).mean().item(),
                    "iteration": global_step + 1,
                    "model": {
                        "policy": self.policy_net,
                        "value": self.value_net,
                        "discriminator": (
                            self.discriminator if self.use_discriminator else None
                        ),
                        "optimizer": self.optimizer,
                        "discriminator_optimizer": (
                            self.discriminator.optimizer
                            if self.use_discriminator
                            else None
                        ),
                    },
                },
                timestep=(global_step + 1) * self.per_loop_env_count,
                train_iters=global_step + 1,
                force_model_save=(global_step == (self.num_train_iters - 1)),
            )

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
        if self.history_len > 0:
            self.reset_history_buffer()

        if self.env.scene_manager.is_recording:
            save_fname = f"{self.env.scene_manager.logs_dir}/{timestep}_train.mp4"
            self.env.scene_manager.cam.stop_recording(
                save_to_filename=save_fname, fps=30
            )
        if log_video:
            # start recording eval episodes
            self.env.scene_manager.is_recording = True
            self.env.scene_manager.cam.start_recording()
            self.env.scene_manager.video_recording_hook(0)

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
                    history = self._get_history_inputs(
                        history_obs=history_obs, history_dones=history_dones
                    )

                proc_obs = self.obs_processor(obs, update_stats=False)
                if history is None:
                    action, _ = self.policy_net.act(proc_obs)
                else:
                    history_obs, history_dones = history
                    action, _ = self.policy_net.act(
                        proc_obs,
                        history_obs=history_obs,
                        history_dones=history_dones,
                    )

                next_obs, reward, terminated, truncated, info = self.env.step(
                    action, compute_gt=True
                )

                n_eval_steps += 1
                if log_video:
                    self.env.scene_manager.video_recording_hook(0)
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
        fname = self.env.scene_manager.logs_dir / f"eval_{timestep}.mp4"
        fname.parent.mkdir(parents=True, exist_ok=True)
        self.env.scene_manager.cam.stop_recording(save_to_filename=fname, fps=30)
        self.env.scene_manager.is_recording = False

        eval_log_data = {
            "periodic_eval/returns": episode_returns.mean().item(),
            "periodic_eval/logs": {k: np.mean(v) for k, v in log_extras.items()},
        }
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
