"""One physical geometry scene with native full-method task-program contexts."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from .conditioning import absolute_targets, make_inputs
from .full_method_config import FullMethodRunConfig, task_configuration
from .full_method_policy import FullMethodInputs
from .grouped_environment import GeometryTaskEnv
from .native_task_context import NativeTaskContext, TaskRandomStream
from .policy import OFFSETS
from .task_data import TaskBank


class FullMethodGeometryEnv(GeometryTaskEnv):
    """Native per-program managers share a simulator and retain all full-method components."""

    def __init__(
        self,
        bank: TaskBank,
        variants: list[dict],
        tasks: tuple[int, ...],
        config: FullMethodRunConfig,
        *,
        evaluation: bool = False,
    ) -> None:
        if not tasks or bank.geometry_index[list(tasks)].unique().numel() != 1:
            raise ValueError(
                "Expected at least one task and exactly one physical geometry per shared scene."
            )
        count = (
            config.evaluation_episodes if evaluation else config.environments_per_task
        )
        device = torch.device("cuda:0")
        self.task_ids = torch.tensor(
            tasks, device=device, dtype=torch.long
        ).repeat_interleave(count)
        self.clocks = torch.zeros_like(self.task_ids)
        self.episode_ids = torch.zeros_like(self.task_ids)
        self.bank = bank.to(device)
        self.variants = variants
        self.contexts: dict[int, NativeTaskContext] = {}
        self.last_episode_metrics: list[dict] = []
        owner = None
        for task in tasks:
            resolved = task_configuration(
                config, variants[task], len(self.task_ids), evaluation=evaluation
            )
            slots = (self.task_ids == task).nonzero().flatten()
            context = NativeTaskContext(resolved.env, slots, task, owner)
            context.set_mode("eval" if evaluation else "train")
            if owner is None:
                self.env = context
                self.resolved_config = asdict(resolved.env)
                owner = context.physics_manager
            else:
                if context.physics_manager.scene is not owner.scene:
                    raise RuntimeError(
                        "Task contexts with matching geometry created different physical scenes."
                    )
                context.previous_position_targets_by_side = (
                    self.env.previous_position_targets_by_side
                )
            self.contexts[task] = context
            manager = context.reference_trajectory_manager
            length = variants[task]["frames"]
            for name, actual, expected in (
                (
                    "joints",
                    manager.hand_joint_positions("right"),
                    self.bank.joints[task, :length],
                ),
                (
                    "wrist",
                    torch.cat(
                        (
                            manager.hand_wrist_pos("right"),
                            manager.hand_wrist_quat("right"),
                        ),
                        -1,
                    ),
                    self.bank.wrist[task, :length],
                ),
                (
                    "cup",
                    torch.cat((manager.object_pos(), manager.object_quat()), -1),
                    self.bank.cup[task, :length],
                ),
            ):
                if actual.shape != expected.shape:
                    raise ValueError(
                        f"Native full-method{variants[task]['id']}/{name} reference shape differs from the approved bank."
                    )
                if name == "joints":
                    matches = torch.allclose(actual, expected, atol=2e-5, rtol=0)
                else:
                    matches = torch.allclose(
                        actual[:, :3], expected[:, :3], atol=2e-5, rtol=0
                    ) and torch.allclose(
                        (actual[:, 3:] * expected[:, 3:]).sum(-1).abs(),
                        torch.ones(length, device=device),
                        atol=2e-5,
                        rtol=0,
                    )
                if not matches:
                    raise ValueError(
                        f"Native full-method{variants[task]['id']}/{name} reference differs from the approved bank."
                    )
        self.spec = self.env.action_manager.hand_specs["right"]
        self.hand = self.spec.hand
        self.cup = self.env.observation_manager.object_entities["cup"]
        action = self.env.cfg.action_cfg
        if (
            action.action_mode != "absolute_target"
            or action.residual_target_reference_mode != "retargeted"
        ):
            raise ValueError(
                "Expected the full-method retargeted absolute-target residual controller."
            )
        self.scales = torch.cat(
            (
                torch.full((3,), action.arm_translation_delta_scale, device=device),
                torch.full((3,), action.arm_rotation_delta_scale, device=device),
                action.hand_joint_delta_scale * self.spec.hand_delta_scale_factors,
            )
        )
        self.residual_beta = action.residual_beta
        self.reset(
            torch.arange(len(self.task_ids), device=device), None, new_episode=False
        )
        terms = (
            tuple(self.env.observation_manager.obs_terms),
            tuple(self.env.observation_manager.term_dims),
        )
        if any(
            (
                tuple(context.observation_manager.obs_terms),
                tuple(context.observation_manager.term_dims),
            )
            != terms
            for context in self.contexts.values()
        ):
            raise ValueError(
                "Native task contexts must have the same policy observation contract."
            )

    def refresh_observations(self) -> None:
        for context in self.contexts.values():
            context.refresh_current_observations()

    def native_observations(self) -> torch.Tensor:
        result = self.env.current_policy_obs.new_empty(
            self.env.current_policy_obs.shape
        )
        for context in self.contexts.values():
            result[context.task_slots] = context.current_policy_obs[context.task_slots]
        if not torch.isfinite(result).all():
            raise ValueError(
                "Full-method task-conditioned policy observations must be finite."
            )
        return result

    def inputs(self) -> FullMethodInputs:
        reference = make_inputs(
            self.bank, self.task_ids, self.clocks, self.physical_state()
        )
        lengths = self.bank.lengths[self.task_ids]
        time = torch.minimum(self.clocks, lengths - 1)
        mask = reference.window_mask.clone()
        # Preserve native active-object gating as well as local-window boundary masks.
        for task, context in self.contexts.items():
            slots = context.task_slots
            active = torch.as_tensor(
                context.task_manager.scene_active_object_index, device=self.env.device
            )
            queries = time[slots, None] + time.new_tensor(OFFSETS)
            queries = queries.clamp(0, self.variants[task]["frames"] - 1)
            mask[slots] &= active[queries] == active[time[slots, None]]
        return FullMethodInputs(
            native_obs=self.native_observations(),
            normalized_reference_time=time / (lengths - 1),
            window=reference.window,
            window_mask=mask,
            **{
                name: getattr(reference, name)
                for name in (
                    "geometry_points",
                    "geometry_metadata",
                    "geometry_index",
                    "program_tokens",
                    "program_mask",
                    "program_index",
                    "current_phase",
                    "progress",
                )
            },
        )

    def reset_probe(self) -> dict:
        from .training_state import cpu_tree

        saved = cpu_tree(self.state_dict())
        result = super().reset_probe()
        self.load_state_dict(saved)
        return result

    def reset(
        self,
        slots: torch.Tensor,
        clocks: torch.Tensor | None,
        *,
        new_episode: bool = True,
    ) -> None:
        if (
            slots.ndim != 1
            or slots.dtype != torch.long
            or (
                clocks is not None
                and (clocks.shape != slots.shape or clocks.dtype != torch.long)
            )
        ):
            raise ValueError(
                "Expected int64 physical slots and optional matching explicit reference frames."
            )
        labels = self.task_ids[slots]
        for task, context in self.contexts.items():
            selection = (labels == task).nonzero().flatten()
            if not len(selection):
                continue
            selected = slots[selection]
            frames = None if clocks is None else clocks[selection].cpu().numpy()
            if frames is not None and (
                (frames < 0).any() or (frames >= self.variants[task]["frames"]).any()
            ):
                raise ValueError(
                    "Explicit reset frames must be inside this task reference."
                )
            context.reset_slots(selected.cpu().numpy(), frames)
            self.clocks[selected] = context.command_time_step[selected]
        if new_episode:
            self.episode_ids[slots] += 1
        self.refresh_observations()

    def prepare_evaluation(self) -> None:
        """Batch the native eval hand-reset motion across all task programs before frame0."""
        for context in self.contexts.values():
            context.random_stream = TaskRandomStream(context.random_seed)
        slots = torch.arange(len(self.task_ids), device=self.env.device)
        self.reset(slots, torch.zeros_like(slots), new_episode=False)
        pos = torch.empty((len(slots), 3), device=self.env.device)
        quat = torch.empty((len(slots), 4), device=self.env.device)
        fingers = torch.empty(
            (len(slots), len(self.spec.hand_target_indices)), device=self.env.device
        )
        finger_indices = torch.tensor(
            self.spec.hand_target_indices, device=self.env.device
        )
        for context in self.contexts.values():
            selected = context.task_slots
            manager = context.reset_manager
            frames = np.zeros(len(selected), dtype=np.int64)
            with context.random_stream.activate():
                q = manager._apply_hand_reset_noise(
                    side="right",
                    jpos=manager.frame_right_reset_jvals[frames].to(self.env.device),
                    noise=manager.hand_reset_noise(frames),
                )
            pos[selected] = manager.frame_right_wrist_pos[frames].to(self.env.device)
            quat[selected] = manager.frame_right_wrist_quat[frames].to(self.env.device)
            fingers[selected] = q.index_select(1, finger_indices)
        self.env.action_manager.move_to_reset_target(
            side="right",
            env_ids=slots.cpu().numpy(),
            target_wrist_pos=pos,
            target_wrist_quat=quat,
            target_hand_qpos=fingers,
        )
        self.refresh_observations()

    def step(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action = absolute_targets(
            self.bank,
            self.task_ids,
            self.clocks,
            self.residual_beta * residual,
            self.scales,
        )
        self.env.action_manager.apply_actions(action)
        self.clocks += 1
        reward = torch.zeros(len(self.task_ids), device=self.env.device)
        terminated = torch.zeros_like(self.task_ids, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        self.last_episode_metrics = []
        for task, context in self.contexts.items():
            _, native_reward, native_terminated, native_truncated, info = (
                context.finish_physical_step(action)
            )
            selected = context.task_slots
            reward[selected] = native_reward[selected]
            terminated[selected] = native_terminated[selected]
            truncated[selected] = native_truncated[selected]
            if len(context.pending_reset_ids):
                self.last_episode_metrics.append(
                    dict(
                        task_index=task,
                        episodes=len(context.pending_reset_ids),
                        progress_sum=info["episode_progress_fraction"].sum().item(),
                        span_sum=info["episode_span_steps"].sum().item(),
                    )
                )
        return reward, terminated, truncated

    def state_dict(self) -> dict:
        physics = self.env.physics_manager
        return dict(
            task_ids=self.task_ids.clone(),
            clocks=self.clocks.clone(),
            episode_ids=self.episode_ids.clone(),
            hand_q=physics.get_entity_dofs_position(
                self.hand.robot, self.hand.dof_map
            ).clone(),
            hand_qd=physics.get_entity_dofs_velocity(
                self.hand.robot, self.hand.dof_map
            ).clone(),
            cup_pose=self.cup.get_obs(rot_format="quat").clone(),
            cup_dof_velocity=self.cup.object.get_dofs_velocity().clone(),
            position_targets=self.env.previous_position_targets_by_side[
                "right"
            ].clone(),
            contexts={
                task: context.state_dict() for task, context in self.contexts.items()
            },
        )

    def load_state_dict(self, state: dict) -> None:
        if not torch.equal(
            state["task_ids"].to(self.task_ids.device), self.task_ids
        ) or set(state["contexts"]) != set(self.contexts):
            raise ValueError(
                "Full-method scene checkpoint task/context assignment changed."
            )
        physics = self.env.physics_manager
        device = self.env.device
        physics.set_entity_dofs_position(
            self.hand.robot, state["hand_q"].to(device), self.hand.dof_map
        )
        physics.set_entity_dofs_velocity(
            self.hand.robot, state["hand_qd"].to(device), self.hand.dof_map
        )
        pose = state["cup_pose"].to(device)
        self.cup.set_state(pos=pose[:, :3], quat=pose[:, 3:])
        self.cup.object.set_dofs_velocity(state["cup_dof_velocity"].to(device))
        self.env.previous_position_targets_by_side["right"] = (
            state["position_targets"].to(device).clone()
        )
        physics.control_entity_dofs_position(
            self.hand.robot,
            self.env.previous_position_targets_by_side["right"],
            self.hand.dof_map,
        )
        self.clocks.copy_(state["clocks"].to(device))
        self.episode_ids.copy_(state["episode_ids"].to(device))
        for task, context in self.contexts.items():
            context.previous_position_targets_by_side = (
                self.env.previous_position_targets_by_side
            )
            context.load_state_dict(state["contexts"][task])
        self.env.passive_mask = None
        self.env.passive_joint_targets = None
        self.refresh_observations()
