"""A single physical cup geometry with independent reference programs in vector slots."""

from __future__ import annotations

import torch

from .conditioning import PhysicalState, absolute_targets, make_inputs
from .config import PilotConfig
from .environment import build_env
from .task_buffers import TaskResetStatistics
from .task_data import TaskBank


class GeometryTaskEnv:
    def __init__(
        self,
        bank: TaskBank,
        variants: list[dict],
        tasks: tuple[int, ...],
        environments_per_task: int,
    ) -> None:
        if not tasks or environments_per_task < 2:
            raise ValueError(
                "Expected at least one task and two vector slots per task for reset isolation checks."
            )
        if bank.geometry_index[list(tasks)].unique().numel() != 1:
            raise ValueError(
                "One GeometryTaskEnv must contain exactly one physical geometry."
            )
        config = PilotConfig()
        config.env.num_envs_per_variant = len(tasks) * environments_per_task
        self.env, self.resolved_config = build_env(
            config, variants[tasks[0]], sequence_scene=True
        )
        self.bank = bank.to(self.env.device)
        self.task_ids = torch.tensor(
            tasks, device=self.env.device, dtype=torch.long
        ).repeat_interleave(environments_per_task)
        self.clocks = torch.zeros_like(self.task_ids)
        self.episode_ids = torch.zeros_like(self.task_ids)
        self.spec = self.env.action_manager.hand_specs["right"]
        self.hand = self.spec.hand
        self.cup = self.env.observation_manager.object_entities["cup"]
        self.statistics = TaskResetStatistics(bank.lengths.cpu())
        self.scales = torch.cat(
            (
                torch.full(
                    (3,),
                    self.env.cfg.action_cfg.arm_translation_delta_scale,
                    device=self.env.device,
                ),
                torch.full(
                    (3,),
                    self.env.cfg.action_cfg.arm_rotation_delta_scale,
                    device=self.env.device,
                ),
                self.env.cfg.action_cfg.hand_joint_delta_scale
                * self.spec.hand_delta_scale_factors,
            )
        )
        if (
            self.env.action_manager.action_mode != "absolute_target"
            or self.scales.shape != (18,)
        ):
            raise ValueError(
                "Expected the declared 18-residual / 19 absolute-target controller."
            )
        # Certify that the native loader and offline FK use exactly the same world frame and joint ordering.
        manager = self.env.reference_trajectory_manager
        task = tasks[0]
        T = int(bank.lengths[task])
        for name, actual, expected in (
            (
                "wrist",
                torch.cat(
                    (manager.hand_wrist_pos("right"), manager.hand_wrist_quat("right")),
                    -1,
                ),
                self.bank.wrist[task, :T],
            ),
            (
                "cup",
                torch.cat((manager.object_pos(), manager.object_quat()), -1),
                self.bank.cup[task, :T],
            ),
            (
                "joints",
                manager.hand_joint_positions("right"),
                self.bank.joints[task, :T],
            ),
        ):
            if name in ("wrist", "cup"):
                matches = torch.allclose(
                    actual[:, :3], expected[:, :3], atol=2e-5, rtol=0
                ) and torch.allclose(
                    (actual[:, 3:] * expected[:, 3:]).sum(-1).abs(),
                    torch.ones(T, device=self.env.device),
                    atol=2e-5,
                    rtol=0,
                )
            else:
                matches = torch.allclose(actual, expected, atol=2e-5, rtol=0)
            if not matches:
                raise ValueError(
                    f"Native loader and task bank differ for {name}: {(actual - expected).abs().max().item()}."
                )
        self.env.previous_position_targets_by_side["right"] = self.bank.joints[
            self.task_ids, 0
        ].clone()
        self.env.update_command_time_step()
        self.reset(
            torch.arange(len(self.task_ids), device=self.env.device),
            self.clocks.clone(),
            new_episode=False,
        )

    def physical_state(self) -> PhysicalState:
        physics = self.env.physics_manager
        robot = self.hand.robot
        wrist_index = physics.get_link_idx_local(robot, self.hand.wrist_link_name)
        wrist = torch.cat(
            (
                physics.get_entity_link_pos(
                    robot, link_name=self.hand.wrist_link_name, ref="link_origin"
                ).squeeze(1),
                physics.get_entity_link_quat(
                    robot, link_name=self.hand.wrist_link_name
                ).squeeze(1),
            ),
            -1,
        )
        q = physics.get_entity_dofs_position(robot, self.hand.dof_map)
        qd = physics.get_entity_dofs_velocity(robot, self.hand.dof_map)
        velocity = torch.cat(
            (
                physics.get_entity_links_vel(
                    robot, [wrist_index], ref="link_origin"
                ).squeeze(1),
                robot.get_links_ang(links_idx_local=[wrist_index]).squeeze(1),
            ),
            -1,
        )
        return PhysicalState(
            wrist,
            q[:, self.spec.hand_target_indices],
            qd[:, self.spec.hand_target_indices],
            self.cup.get_obs(rot_format="quat"),
            self.cup.get_vel_obs(),
            velocity,
        )

    def inputs(self):
        return make_inputs(self.bank, self.task_ids, self.clocks, self.physical_state())

    def state_dict(self) -> dict:
        physics = self.env.physics_manager
        return {
            "task_ids": self.task_ids.clone(),
            "clocks": self.clocks.clone(),
            "episode_ids": self.episode_ids.clone(),
            "hand_q": physics.get_entity_dofs_position(
                self.hand.robot, self.hand.dof_map
            ).clone(),
            "hand_qd": physics.get_entity_dofs_velocity(
                self.hand.robot, self.hand.dof_map
            ).clone(),
            "cup_pose": self.cup.get_obs(rot_format="quat").clone(),
            "cup_dof_velocity": self.cup.object.get_dofs_velocity().clone(),
            "position_targets": self.env.previous_position_targets_by_side[
                "right"
            ].clone(),
            "env_step_counter": self.env.env_step_counter,
            "reset_statistics": self.statistics.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        device = self.env.device
        if not torch.equal(state["task_ids"].to(device), self.task_ids):
            raise ValueError(
                "Checkpoint vector-slot task assignment differs from the current geometry scene."
            )
        slots = torch.arange(len(self.task_ids), device=device)
        clocks = state["clocks"].to(device)
        self.reset(
            slots,
            torch.minimum(clocks, self.bank.lengths[self.task_ids] - 1),
            new_episode=False,
        )
        physics = self.env.physics_manager
        physics.set_entity_dofs_position(
            self.hand.robot, state["hand_q"].to(device), self.hand.dof_map
        )
        physics.set_entity_dofs_velocity(
            self.hand.robot, state["hand_qd"].to(device), self.hand.dof_map
        )
        pose = state["cup_pose"].to(device)
        self.cup.set_state(pos=pose[:, :3], quat=pose[:, 3:])
        self.cup.object.set_dofs_velocity(state["cup_dof_velocity"].to(device))
        targets = state["position_targets"].to(device)
        self.env.previous_position_targets_by_side["right"] = targets.clone()
        physics.control_entity_dofs_position(
            self.hand.robot, targets, self.hand.dof_map
        )
        self.clocks.copy_(clocks)
        self.episode_ids.copy_(state["episode_ids"].to(device))
        self.env.env_step_counter = state["env_step_counter"]
        self.statistics.load_state_dict(state["reset_statistics"])
        self.env.passive_mask = None
        self.env.passive_joint_targets = None
        self.env.refresh_current_observations()

    def placement_contacts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        physics = self.env.physics_manager
        table = self.env.observation_manager.object_entities["table"]
        hand = physics.get_filtered_contact_info(
            self.cup.object, self.hand.robot, return_link_pos=True
        )
        support = physics.get_filtered_contact_info(
            self.cup.object, table.object, return_link_pos=True, return_link_force=True
        )
        hand_contact = hand["contact_pos_link_a_valid"].flatten(1).any(-1)
        table_contact = support["contact_pos_link_a_valid"].flatten(1).any(-1)
        upward_force = support["contact_force_link_a"].sum(dim=(1, 2))[:, 2]
        return hand_contact, table_contact, upward_force

    def reset(
        self, slots: torch.Tensor, clocks: torch.Tensor, *, new_episode: bool = True
    ) -> None:
        if (
            slots.ndim != 1
            or slots.dtype != torch.long
            or clocks.shape != slots.shape
            or clocks.dtype != torch.long
        ):
            raise ValueError(
                "Expected matching int64 reset-slot and reference-frame vectors."
            )
        tasks = self.task_ids[slots]
        if (clocks < 0).any() or (clocks >= self.bank.lengths[tasks]).any():
            raise ValueError(
                "Reset clock must be inside each slot's own approved trajectory."
            )
        q = self.bank.joints[tasks, clocks]
        pose = self.bank.cup[tasks, clocks]
        physics = self.env.physics_manager
        physics.set_entity_dofs_position(
            self.hand.robot, q, self.hand.dof_map, envs_idx=slots
        )
        physics.set_entity_dofs_velocity(
            self.hand.robot, torch.zeros_like(q), self.hand.dof_map, envs_idx=slots
        )
        self.cup.set_state(pos=pose[:, :3], quat=pose[:, 3:], env_idx=slots)
        self.cup.object.set_dofs_velocity(
            torch.zeros((len(slots), 6), device=self.env.device), envs_idx=slots
        )
        self.env.previous_position_targets_by_side["right"][slots] = q
        self.clocks[slots] = clocks
        if new_episode:
            self.episode_ids[slots] += 1
        # No scene.step here: resetting one slot must not advance any other physical state.
        self.env.refresh_current_observations()

    def step(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action = absolute_targets(
            self.bank, self.task_ids, self.clocks, residual, self.scales
        )
        self.env.action_manager.apply_actions(action)
        self.clocks += 1
        self.env.env_step_counter += 1
        self.env.refresh_current_observations()
        state = self.physical_state()
        time = torch.minimum(self.clocks, self.bank.lengths[self.task_ids] - 1)
        target = self.bank.cup[self.task_ids, time]
        position = (state.cup[:, :3] - target[:, :3]).square().sum(-1)
        dot = (state.cup[:, 3:] * target[:, 3:]).sum(-1).abs().clamp(0, 1)
        angle = 2 * torch.acos(dot)
        finger_error = (
            (state.fingers - self.bank.fingers[self.task_ids, time]).square().mean(-1)
        )
        # Explicit preliminary tracking reward; freeze the same formula for warm/scratch arms.
        reward = (
            torch.exp(-position / 0.03**2)
            + torch.exp(-angle.square() / 0.3**2)
            + 0.25 * torch.exp(-finger_error / 0.2**2)
        )
        terminated = state.cup[:, 2] < -0.15
        truncated = self.clocks >= self.bank.lengths[self.task_ids] + 10
        valid_time = torch.minimum(self.clocks, self.bank.lengths[self.task_ids] - 1)
        self.statistics.record(self.task_ids.cpu(), valid_time.cpu())
        return reward, terminated, truncated

    def reset_probe(self) -> dict:
        before = self.physical_state()
        clock_before = self.clocks.clone()
        slot = torch.tensor([0], device=self.env.device)
        self.reset(slot, torch.tensor([3], device=self.env.device))
        after = self.physical_state()
        for name in (
            "wrist",
            "fingers",
            "finger_velocity",
            "cup",
            "cup_velocity",
            "wrist_velocity",
        ):
            if not torch.equal(getattr(before, name)[1:], getattr(after, name)[1:]):
                raise RuntimeError(
                    f"Partial reset changed another slot's physical {name}."
                )
        if not torch.equal(clock_before[1:], self.clocks[1:]):
            raise RuntimeError("Partial reset changed another slot's task clock.")
        self.reset(
            slot,
            torch.zeros(1, device=self.env.device, dtype=torch.long),
            new_episode=False,
        )
        return {
            "partial_physical_reset_isolated": True,
            "partial_clock_reset_isolated": True,
        }
