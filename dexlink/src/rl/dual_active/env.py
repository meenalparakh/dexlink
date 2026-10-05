"""Environment for tasks where both hands support an object of their own."""

from __future__ import annotations

from dexlink.src.rl.dual_active.reward_manager import DualActiveRewardManager
from dexlink.src.rl.dual_active.task_manager import DualActiveTaskManager
from dexlink.src.rl.multi_object.env import MultiObjectEnv


class DualActiveEnv(MultiObjectEnv):
    """Multi-object env with the task and reward managers that allow two actives.

    Every other manager -- physics, commands, observations, reset -- is the
    multi-object one, so this changes only the two behaviours that assume a
    single active object.
    """

    def _create_task_manager(self):
        return DualActiveTaskManager(self.cfg.task_cfg, self)

    def _create_reward_manager(self):
        return DualActiveRewardManager(self.cfg.rewards_cfg, self)
