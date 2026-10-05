from __future__ import annotations

import numpy as np
import torch

from dexlink.src.rl.multi_object.commands_manager import MultiObjectCommandsManager
from dexlink.src.rl.multi_object.obs_manager import MultiObjectObservationManager
from dexlink.src.rl.multi_object.reset_manager import MultiObjectResetManager
from dexlink.src.rl.multi_object.reward_manager import MultiObjectRewardManager
from dexlink.src.rl.multi_object.task_manager import MultiObjectTaskManager
from dexlink.src.rl.world.env import Env


class MultiObjectEnv(Env):
    """Env variant that keeps single-object RL code intact via derived managers."""

    def initialize(self):
        super().initialize()
        ema_alpha = float(
            self.cfg.reset_cfg.multi_object_reset_timestep_density_ema_alpha
        )
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError(
                "Expected reset_cfg.multi_object_reset_timestep_density_ema_alpha "
                f"to be in (0, 1], got {ema_alpha}."
            )
        self.reset_timestep_density_ema_alpha = ema_alpha
        self.reset_timestep_density = torch.zeros(
            self.task_manager.total_steps,
            dtype=torch.float32,
            device=self.device,
        )

    def update_reset_timestep_density(
        self, env_ids: list[int] | np.ndarray | None = None
    ) -> None:
        env_ids_array = self.reset_manager.normalize_env_ids(env_ids)
        encountered_timesteps = self.command_time_step[env_ids_array]
        if encountered_timesteps.ndim != 1:
            raise ValueError(
                "Expected command_time_step entries to be a 1D tensor, got "
                f"shape {tuple(encountered_timesteps.shape)}."
            )
        if torch.any(encountered_timesteps < 0) or torch.any(
            encountered_timesteps >= self.task_manager.total_steps
        ):
            raise ValueError(
                "Expected command timesteps in "
                f"[0, {self.task_manager.total_steps - 1}], got "
                f"{encountered_timesteps}."
            )
        counts = torch.bincount(
            encountered_timesteps,
            minlength=self.task_manager.total_steps,
        ).to(dtype=self.reset_timestep_density.dtype)
        self.reset_timestep_density.mul_(1.0 - self.reset_timestep_density_ema_alpha)
        self.reset_timestep_density.add_(
            counts,
            alpha=self.reset_timestep_density_ema_alpha,
        )

    def reset_env_idxs(self, env_ids=None):
        super().reset_env_idxs(env_ids)
        if self.mode == "train":
            self.update_reset_timestep_density(env_ids)

    def _apply_actions_and_advance(self, actions: torch.Tensor) -> None:
        super()._apply_actions_and_advance(actions)
        if self.mode == "train":
            self.update_reset_timestep_density()

    def _create_task_manager(self):
        return MultiObjectTaskManager(self.cfg.task_cfg, self)

    def _create_physics_manager(self):
        if self.cfg.backend == "sim":
            from dexlink.src.rl.multi_object.physics_sim import MultiObjectPhysicsSim

            return MultiObjectPhysicsSim(self.cfg.sim_cfg, self)
        raise ValueError(
            "MultiObjectEnv currently supports only cfg.backend='sim'. "
            f"Got {self.cfg.backend!r}."
        )

    def _create_commands_manager(self):
        return MultiObjectCommandsManager(self.cfg.commands_cfg, self)

    def _create_observation_manager(self):
        return MultiObjectObservationManager(self.cfg.obs_cfg, self)

    def _create_reward_manager(self):
        return MultiObjectRewardManager(self.cfg.rewards_cfg, self)

    def _create_reset_manager(self):
        return MultiObjectResetManager(self.cfg, self)
