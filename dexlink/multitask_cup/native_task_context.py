"""Native full-method managers for task programs sharing a physical geometry scene."""

from __future__ import annotations

from contextlib import contextmanager
import random
from typing import Iterator

import numpy as np
import torch

from dexlink.src.rl.multi_object.reward_manager import MultiObjectRewardManager

from .scene import CupPhysicsSim, CupSequenceEnv


class TaskRandomStream:
    """Scope native reset draws to one task without changing their distributions."""

    def __init__(self, seed: int) -> None:
        self.python = random.Random(seed).getstate()
        self.numpy = np.random.RandomState(seed).get_state()
        self.cpu = torch.Generator(device="cpu").manual_seed(seed).get_state()
        self.cuda = torch.Generator(device="cuda:0").manual_seed(seed).get_state()

    @contextmanager
    def activate(self) -> Iterator[None]:
        previous = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state(),
        )
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.cpu)
        torch.cuda.set_rng_state(self.cuda)
        try:
            yield
        finally:
            self.python, self.numpy = random.getstate(), np.random.get_state()
            self.cpu, self.cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
            random.setstate(previous[0])
            np.random.set_state(previous[1])
            torch.set_rng_state(previous[2])
            torch.cuda.set_rng_state(previous[3])

    def state_dict(self) -> dict:
        name, values, position, has_gaussian, cached = self.numpy
        return dict(
            python=self.python,
            numpy=(name, values.tolist(), position, has_gaussian, cached),
            cpu=self.cpu.clone(),
            cuda=self.cuda.clone(),
        )

    def load_state_dict(self, state: dict) -> None:
        name, values, position, has_gaussian, cached = state["numpy"]
        self.python = state["python"]
        self.numpy = (
            name,
            np.array(values, dtype=np.uint32),
            position,
            has_gaussian,
            cached,
        )
        self.cpu, self.cuda = state["cpu"].clone(), state["cuda"].clone()


class SharedCupPhysics(CupPhysicsSim):
    """Attach native managers to an existing scene; never create or step another simulator."""

    def __init__(self, sim_cfg, env, owner: CupPhysicsSim) -> None:
        super().__init__(sim_cfg, env)
        self.owner = owner

    def initialize(self, camera_lookat: tuple[float, float, float]) -> None:
        if self.env.num_envs != self.owner.env.num_envs:
            raise ValueError(
                "Task contexts sharing physics require identical physical vector-slot counts."
            )
        self.camera_lookat = camera_lookat
        self.scene = self.owner.scene
        self.scene_entities = self.owner.scene_entities
        self.hand_sensors = self.owner.hand_sensors
        self.hand_cfgs = self.owner.hand_cfgs
        self.object_cfg = self.owner.object_cfg
        self.hammer_peg_force_sensor = self.owner.hammer_peg_force_sensor
        self._scene_table_top_height = self.owner._scene_table_top_height
        self.env.task_manager.process_retargeted_motion()


class TaskContextRewards(MultiObjectRewardManager):
    """Use native rewards unchanged; only this context's slots may end their episodes."""

    def compute_rewards(
        self, named_obs, disc_obs, actions, *, prev_named_privileged_obs=None
    ):
        reward, terminated, truncated = super().compute_rewards(
            named_obs,
            disc_obs,
            actions,
            prev_named_privileged_obs=prev_named_privileged_obs,
        )
        return (
            reward,
            terminated & self.env.task_slot_mask,
            truncated & self.env.task_slot_mask,
        )


class NativeTaskContext(CupSequenceEnv):
    """One native reference/reward/reset context over explicitly assigned physical slots."""

    def __init__(
        self,
        cfg,
        slots: torch.Tensor,
        task_index: int,
        physical_owner: CupPhysicsSim | None = None,
    ) -> None:
        if cfg.reset_cfg.dr:
            raise ValueError(
                "This validated full-method preset must have dr=False; shared-scene DR requires an explicit per-slot implementation."
            )
        self.physical_owner = physical_owner
        self.task_slots = slots
        self.task_slot_mask = torch.zeros(
            cfg.sim_cfg.num_envs, dtype=torch.bool, device=cfg.device
        )
        self.task_slot_mask[slots] = True
        self.random_seed = cfg.seed + 7919 * task_index
        self.random_stream = TaskRandomStream(self.random_seed)
        self.defer_resets = False
        self.pending_reset_ids = np.empty(0, dtype=np.int64)
        super().__init__(cfg)
        self.update_command_time_step()

    def _create_physics_manager(self):
        if self.physical_owner is None:
            return CupPhysicsSim(self.cfg.sim_cfg, self)
        return SharedCupPhysics(self.cfg.sim_cfg, self, self.physical_owner)

    def _create_reward_manager(self):
        return TaskContextRewards(self.cfg.rewards_cfg, self)

    def reset_env_idxs(self, env_ids=None) -> None:
        selected = (
            self.task_slots.cpu().numpy()
            if env_ids is None
            else self.reset_manager.normalize_env_ids(env_ids)
        )
        if not self.task_slot_mask[torch.as_tensor(selected, device=self.device)].all():
            raise ValueError(
                "A native task context may reset only its declared physical slots."
            )
        if self.defer_resets:
            self.pending_reset_ids = selected.copy()
        else:
            self.reset_slots(selected)

    def reset_slots(self, slots: np.ndarray, frames: np.ndarray | None = None) -> None:
        """Retain native sampling/noise; direct state writes keep other tasks' clocks and physics still."""
        if not self.task_slot_mask[torch.as_tensor(slots, device=self.device)].all():
            raise ValueError("Reset slots belong to another task context.")
        with self.random_stream.activate():
            manager = self.reset_manager
            manager.resample_random_force(slots)
            sampled = (
                manager.sample_start_timesteps(slots) if frames is None else frames
            )
            if sampled.shape != slots.shape or sampled.dtype != np.int64:
                raise ValueError(
                    "Expected matching int64 reset-frame and physical-slot arrays."
                )
            reference = manager.clamp_to_reference_frame_timesteps(sampled)
            self.start_timestep[slots] = torch.from_numpy(sampled)
            if self.mode == "eval":
                manager.set_object_state_for_eval(reference, slots)
            else:
                manager.set_object_state_from_retargeted_results(reference, slots)
            manager.set_robot_state_from_retargeted_results(reference, slots)
        # A whole-scene settle step on a partial reset would advance unrelated programs.
        for side, spec in self.action_manager.hand_specs.items():
            q = self.physics_manager.get_hand_dof_positions(
                spec.hand, spec.hand.dof_map
            ).detach()
            if side not in self.previous_position_targets_by_side:
                self.previous_position_targets_by_side[side] = q.clone()
            else:
                self.previous_position_targets_by_side[side][slots] = q[slots]
        self.timesteps[slots] = 0
        self.episode_length[slots] = 0
        self.eval_would_terminate_timestep[slots] = -1
        self.eval_would_terminate_progress[slots] = -1.0
        self.update_command_time_step(slots)
        self.task_manager.reset_task_metric_accumulators(slots)
        if self.mode == "train":
            self.update_reset_timestep_density(slots)
        self.pending_reset_ids = np.empty(0, dtype=np.int64)

    def finish_physical_step(self, actions: torch.Tensor):
        """Run the native observation/reward/termination path without resetting before bootstrapping."""
        self.timesteps += 1
        self.episode_length += 1
        self.env_step_counter += 1
        self.update_command_time_step()
        if self.mode == "train":
            self.update_reset_timestep_density(self.task_slots.cpu().numpy())
        self.defer_resets = True
        self.pending_reset_ids = np.empty(0, dtype=np.int64)
        try:
            result = self._finish_step_after_actions(actions)
            if len(self.pending_reset_ids):
                span = self.episode_length[self.pending_reset_ids].float()
                horizon = self.eval_episode_horizon()[self.pending_reset_ids]
                self.extras["episode_span_steps"] = span.clone()
                self.extras["episode_progress_fraction"] = (span / horizon).clamp(0, 1)
            return result
        finally:
            self.defer_resets = False

    def state_dict(self) -> dict:
        return dict(
            slots=self.task_slots.cpu(),
            start_timestep=self.start_timestep.clone(),
            timesteps=self.timesteps.clone(),
            episode_length=self.episode_length.clone(),
            env_step_counter=self.env_step_counter,
            reset_timestep_density=self.reset_timestep_density.clone(),
            curriculum_scale=self.curriculum_manager.current_scale,
            curriculum_last_update=self.curriculum_manager.last_updated_step,
            random_stream=self.random_stream.state_dict(),
        )

    def load_state_dict(self, state: dict) -> None:
        if not torch.equal(state["slots"], self.task_slots.cpu()):
            raise ValueError("Native task-context checkpoint slot assignment changed.")
        for name in (
            "start_timestep",
            "timesteps",
            "episode_length",
            "reset_timestep_density",
        ):
            target = getattr(self, name)
            value = state[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(
                    f"Native context checkpoint {name} shape/dtype changed."
                )
            target.copy_(value.to(target.device))
        self.env_step_counter = state["env_step_counter"]
        self.curriculum_manager._set_scale(state["curriculum_scale"])
        self.curriculum_manager.last_updated_step = state["curriculum_last_update"]
        self.random_stream.load_state_dict(state["random_stream"])
        self.update_command_time_step()
