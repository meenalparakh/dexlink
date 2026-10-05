from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import CurriculumCfg
    from dexlink.src.rl.sim.env import Env


class CurriculumManager:
    def __init__(self, cfg: CurriculumCfg, env: Env):
        self.cfg = cfg
        self.env = env
        self.start_scale = float(cfg.task_dense_product_scale[0])
        self.end_scale = float(cfg.task_dense_product_scale[1])
        self.last_updated_step = -1
        self.current_scale = self.start_scale
        self.enabled = (
            cfg.use_curriculum and cfg.update_freq > 0 and cfg.end_timestep > 0
        )

    def initialize(self):
        # Always set an explicit start value to avoid relying on reward defaults.
        self._set_scale(self.start_scale)

    def _set_scale(self, value: float):
        value = float(value)
        self.current_scale = value
        # self.env.rewards_cfg.task_dense_product_scale = value
        self.env.reward_manager.cfg.task_dense_product_scale = value

    def _compute_scale(self, step: int) -> float:
        progress = min(max(step / self.cfg.end_timestep, 0.0), 1.0)
        return self.start_scale + (self.end_scale - self.start_scale) * progress

    def update(self):
        if not self.enabled:
            return

        step = int(self.env.env_step_counter)
        should_update = (step % self.cfg.update_freq == 0) or (
            step >= self.cfg.end_timestep
        )
        if not should_update or step == self.last_updated_step:
            return

        new_scale = self._compute_scale(step)
        self._set_scale(new_scale)
        self.last_updated_step = step
        self.env.extras["log/curriculum_task_dense_product_scale"] = float(new_scale)
