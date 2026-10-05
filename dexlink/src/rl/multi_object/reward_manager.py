from __future__ import annotations

import torch

from dexlink.utils.retargeting.error_functions import angle_error
from dexlink.src.rl.world.reward_manager import (
    RewardCommonComputation,
    RewardComputation,
    RewardManager,
    RewardTerminationComputation,
)


class MultiObjectRewardCommonComputation(RewardCommonComputation):
    """Reward computation that tracks all scene objects with motion masks."""

    POSITION_MOTION_EPS = 0.0001
    ROTATION_MOTION_EPS = 0.01
    PRISMATIC_ARTICULATION_MOTION_EPS = 0.0001
    REVOLUTE_ARTICULATION_MOTION_EPS = 0.01

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.has_maniptrans = False

    def initialize(self) -> None:
        super().initialize()
        self._build_object_tracking_reference()
        self._build_reach_phase_reference()
        self._build_retargeted_joint_mse_phase_reference()

    def refresh_object_tracking_reference(self) -> None:
        self._build_object_tracking_reference()

    def _build_reach_phase_reference(self) -> None:
        task = self.env.task_manager
        reach_phase_indices = [
            phase_index
            for phase_index, phase_name in enumerate(task.phase_names)
            if "reach" in str(phase_name).lower()
        ]
        phase_index_by_timestep = task.phase_index_by_timestep
        self.reach_phase_by_timestep = torch.zeros(
            task.total_steps,
            dtype=torch.bool,
            device=self.env.device,
        )
        for phase_index in reach_phase_indices:
            self.reach_phase_by_timestep |= phase_index_by_timestep == phase_index

    def current_reach_phase_mask(self) -> torch.Tensor:
        reach_phase = self.reach_phase_by_timestep[self.current_reference_timesteps()]
        if self.log_metrics:
            self.env.extras["log/current_reach_phase"] = reach_phase.to(torch.float32)
        return reach_phase

    def _build_retargeted_joint_mse_phase_reference(self) -> None:
        task = self.env.task_manager
        disabled_phase_indices = [
            phase_index
            for phase_index, phase_name in enumerate(task.phase_names)
            if self._retargeted_joint_mse_disabled_phase_name(str(phase_name))
        ]
        phase_index_by_timestep = task.phase_index_by_timestep
        disabled_by_timestep = torch.zeros(
            task.total_steps,
            dtype=torch.bool,
            device=self.env.device,
        )
        for phase_index in disabled_phase_indices:
            disabled_by_timestep |= phase_index_by_timestep == phase_index
        self.retargeted_joint_mse_enabled_by_timestep = ~disabled_by_timestep

    def _retargeted_joint_mse_disabled_phase_name(self, phase_name: str) -> bool:
        phase_name_lower = phase_name.lower()
        if "hold" in phase_name_lower:
            return True
        if phase_name_lower.startswith("change_to"):
            return False
        return "grasp" in phase_name_lower

    def current_retargeted_joint_mse_enabled_mask(self) -> torch.Tensor:
        enabled = self.retargeted_joint_mse_enabled_by_timestep[
            self.current_reference_timesteps()
        ]
        if self.should_log_essential_metrics():
            self.env.extras["log/current_retargeted_joint_mse_enabled"] = enabled.to(
                torch.float32
            )
        return enabled

    def _build_object_tracking_reference(self) -> None:
        task = self.env.task_manager
        device = self.env.device
        self.object_ids = list(task.scene_object_ids)
        self.object_ref_pos = torch.as_tensor(
            task.scene_object_pos,
            dtype=torch.float32,
            device=device,
        )
        self.object_ref_quat = torch.as_tensor(
            task.scene_object_quat,
            dtype=torch.float32,
            device=device,
        )
        self.object_ref_arti = torch.as_tensor(
            task.scene_object_arti_vals,
            dtype=torch.float32,
            device=device,
        )
        self.object_ref_arti_valid_mask = torch.as_tensor(
            task.scene_object_arti_valid_mask,
            dtype=torch.bool,
            device=device,
        )
        self.object_ref_active_mask = torch.as_tensor(
            task.scene_object_active_mask,
            dtype=torch.bool,
            device=device,
        )
        (
            self.object_ref_arti_motion_eps,
            self.object_sim_arti_motion_eps_by_id,
        ) = self._object_articulation_motion_eps()
        self.object_ref_motion_mask = self._compute_reference_motion_mask()
        self._last_object_tracking_enabled_mask = None
        self._last_object_far_away_mask = None

    def _joint_motion_eps(
        self,
        joint_type: str,
        object_id: str,
        joint_name: str,
    ) -> float:
        if joint_type == "prismatic":
            return self.PRISMATIC_ARTICULATION_MOTION_EPS
        if joint_type in ("revolute", "continuous"):
            return self.REVOLUTE_ARTICULATION_MOTION_EPS
        raise ValueError(
            f"Unsupported articulation joint type {joint_type!r} for "
            f"object {object_id!r} joint {joint_name!r}; expected prismatic, "
            "revolute, or continuous."
        )

    def _object_articulation_motion_eps(
        self,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        task = self.env.task_manager
        ref_eps = torch.full_like(
            self.object_ref_arti_valid_mask,
            float("inf"),
            dtype=torch.float32,
        )
        sim_eps_by_id: dict[str, torch.Tensor] = {}

        for object_index, object_id in enumerate(self.object_ids):
            entity = self.env.physics_manager.object_entities[object_id]
            joint_type_by_name = {
                joint.name: joint.joint_type for joint in entity.joints
            }
            exported_joint_names = list(task.scene_object_arti_names[object_index])
            valid_indices = torch.nonzero(
                self.object_ref_arti_valid_mask[object_index],
                as_tuple=False,
            ).reshape(-1)
            if len(exported_joint_names) != int(valid_indices.numel()):
                raise ValueError(
                    "Expected scene_object_arti_names length to match valid "
                    f"articulation count for object {object_id!r}, got "
                    f"{len(exported_joint_names)} names and "
                    f"{int(valid_indices.numel())} valid values."
                )

            for valid_idx, joint_name in zip(
                valid_indices.tolist(),
                exported_joint_names,
            ):
                if joint_name not in joint_type_by_name:
                    raise ValueError(
                        f"Reference articulation for object {object_id!r} contains "
                        f"joint {joint_name!r}, but the simulated object joints are "
                        f"{list(joint_type_by_name)}."
                    )
                ref_eps[object_index, valid_idx] = self._joint_motion_eps(
                    joint_type_by_name[joint_name],
                    object_id,
                    joint_name,
                )

            sim_eps = [
                self._joint_motion_eps(joint.joint_type, object_id, joint.name)
                for joint in entity.joints
            ]
            sim_eps_by_id[object_id] = torch.as_tensor(
                sim_eps,
                dtype=torch.float32,
                device=self.env.device,
            )

        return ref_eps.to(self.env.device), sim_eps_by_id

    def _compute_reference_motion_mask(self) -> torch.Tensor:
        n_frames = self.object_ref_pos.shape[0]
        n_objects = self.object_ref_pos.shape[1]
        motion_mask = torch.zeros(
            (n_frames, n_objects),
            dtype=torch.bool,
            device=self.env.device,
        )
        if n_frames <= 1:
            return motion_mask

        pos_delta = torch.linalg.norm(
            self.object_ref_pos[1:] - self.object_ref_pos[:-1],
            dim=-1,
        )
        quat_delta = angle_error(
            self.object_ref_quat[1:].reshape(-1, 4),
            self.object_ref_quat[:-1].reshape(-1, 4),
            current_format="quat",
            target_format="quat",
        ).reshape(n_frames - 1, n_objects)
        arti_delta_raw = torch.abs(self.object_ref_arti[1:] - self.object_ref_arti[:-1])
        valid = self.object_ref_arti_valid_mask.unsqueeze(0)
        arti_motion = torch.logical_and(
            valid,
            arti_delta_raw > self.object_ref_arti_motion_eps.unsqueeze(0),
        ).any(dim=-1)
        delta_mask = (
            (pos_delta > self.POSITION_MOTION_EPS)
            | (quat_delta > self.ROTATION_MOTION_EPS)
            | arti_motion
        )
        motion_mask[:-1] |= delta_mask
        motion_mask[1:] |= delta_mask
        return motion_mask

    def _runtime_motion_mask(
        self,
        named_obs: dict[str, torch.Tensor],
        prev_named_obs: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        n_envs = self.env.num_envs
        n_objects = len(self.object_ids)
        if prev_named_obs is None:
            return torch.zeros(
                (n_envs, n_objects),
                dtype=torch.bool,
                device=self.env.device,
            )

        runtime_mask = torch.zeros(
            (n_envs, n_objects),
            dtype=torch.bool,
            device=self.env.device,
        )
        for object_index, object_id in enumerate(self.object_ids):
            current = self.env.observation_manager.get_named_object_obs(
                named_obs,
                object_id,
            )
            previous = self.env.observation_manager.get_named_object_obs(
                prev_named_obs,
                object_id,
            )
            pos_delta = torch.linalg.norm(current["pos"] - previous["pos"], dim=-1)
            quat_delta = angle_error(
                current["quat"],
                previous["quat"],
                current_format=self.env.cfg.obs_cfg.rot_format,
                target_format=self.env.cfg.obs_cfg.rot_format,
            )
            if current["arti"].shape[1] == 0:
                arti_motion = torch.zeros_like(pos_delta, dtype=torch.bool)
            else:
                eps = self.object_sim_arti_motion_eps_by_id[object_id]
                if current["arti"].shape[1] != eps.numel():
                    raise ValueError(
                        f"Expected object {object_id!r} runtime articulation dim "
                        f"{eps.numel()}, got {current['arti'].shape[1]}."
                    )
                arti_motion = (
                    torch.abs(current["arti"] - previous["arti"]) > eps.unsqueeze(0)
                ).any(dim=-1)
            runtime_mask[:, object_index] = (
                (pos_delta > self.POSITION_MOTION_EPS)
                | (quat_delta > self.ROTATION_MOTION_EPS)
                | arti_motion
            )
        return runtime_mask

    def _reference_motion_enabled_mask(
        self,
        reference_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        frame_idx = torch.clamp(
            reference_timesteps,
            min=0,
            max=self.object_ref_motion_mask.shape[0] - 1,
        )
        return torch.logical_and(
            self.object_ref_motion_mask[frame_idx],
            self.object_ref_active_mask[frame_idx],
        )

    def _reference_active_mask(
        self,
        reference_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        frame_idx = torch.clamp(
            reference_timesteps,
            min=0,
            max=self.object_ref_active_mask.shape[0] - 1,
        )
        return self.object_ref_active_mask[frame_idx]

    def _object_reference_arti_for_sim_joints(
        self,
        *,
        object_index: int,
        object_id: str,
        reference_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        entity = self.env.physics_manager.object_entities[object_id]
        if len(entity.joints) == 0:
            return torch.zeros(
                (self.env.num_envs, 0),
                dtype=torch.float32,
                device=self.env.device,
            )
        task = self.env.task_manager
        valid_indices = torch.nonzero(
            self.object_ref_arti_valid_mask[object_index],
            as_tuple=False,
        ).reshape(-1)
        exported_joint_names = list(task.scene_object_arti_names[object_index])
        if len(exported_joint_names) != int(valid_indices.numel()):
            raise ValueError(
                "Expected scene_object_arti_names length to match valid articulation "
                f"count for object {object_id!r}, got {len(exported_joint_names)} "
                f"names and {int(valid_indices.numel())} valid values."
            )
        missing = [
            joint.name
            for joint in entity.joints
            if joint.name not in exported_joint_names
        ]
        if missing:
            raise ValueError(
                f"Reference articulation for object {object_id!r} is missing "
                f"simulated joints {missing}."
            )
        joint_indices = torch.as_tensor(
            [exported_joint_names.index(joint.name) for joint in entity.joints],
            dtype=torch.long,
            device=self.env.device,
        )
        valid_values = self.object_ref_arti[
            reference_timesteps,
            object_index,
        ].index_select(1, valid_indices)
        return valid_values.index_select(1, joint_indices)

    def _final_scene_arti_thresholds(self, object_id: str) -> torch.Tensor:
        entity = self.env.physics_manager.object_entities[object_id]
        thresholds = []
        for joint in entity.joints:
            joint_type = str(joint.joint_type)
            if joint_type == "prismatic":
                thresholds.append(float(self.cfg.task_success_prismatic_arti_thd))
            elif joint_type in ("revolute", "continuous"):
                thresholds.append(float(self.cfg.task_success_revolute_arti_thd))
            else:
                raise ValueError(
                    f"Unsupported articulation joint type {joint_type!r} for "
                    f"object {object_id!r} joint {joint.name!r}; expected prismatic, "
                    "revolute, or continuous."
                )
        return torch.as_tensor(
            thresholds,
            dtype=torch.float32,
            device=self.env.device,
        )

    def object_tracking_progress(self, named_obs, **kwargs):
        if self._object_tracking_cache is not None:
            return self._object_tracking_cache

        reference_ts = self.current_reference_timesteps()
        active_mask = self._reference_active_mask(reference_ts)
        motion_tracking_mask = self._reference_motion_enabled_mask(reference_ts)
        rest_tracking_mask = torch.logical_and(
            active_mask,
            torch.logical_not(motion_tracking_mask),
        )
        enabled_mask = active_mask

        reward_sum = torch.zeros(
            self.env.num_envs,
            dtype=torch.float32,
            device=self.env.device,
        )
        enabled_count = enabled_mask.sum(dim=1).to(torch.float32).clamp_min(1.0)
        object_pose_achieved = torch.ones(
            self.env.num_envs,
            dtype=torch.bool,
            device=self.env.device,
        )
        early_terminate = torch.zeros(
            self.env.num_envs,
            dtype=torch.bool,
            device=self.env.device,
        )
        object_far_away_mask = torch.zeros_like(enabled_mask, dtype=torch.bool)

        for object_index, object_id in enumerate(self.object_ids):
            obs = self.env.observation_manager.get_named_object_obs(
                named_obs,
                object_id,
            )
            enabled = enabled_mask[:, object_index]
            motion_tracking = motion_tracking_mask[:, object_index]
            rest_tracking = rest_tracking_mask[:, object_index]
            ref_pos = self.object_ref_pos[reference_ts, object_index]
            ref_quat = self.object_ref_quat[reference_ts, object_index]
            ref_arti = self._object_reference_arti_for_sim_joints(
                object_index=object_index,
                object_id=object_id,
                reference_timesteps=reference_ts,
            )

            pos_diff = obs["pos"] - ref_pos
            pos_error = torch.linalg.norm(pos_diff, dim=-1)
            pos_reward = torch.exp(-float(self.cfg.task_pos_scale) * pos_error)
            rest_pos_reward = torch.exp(
                -float(self.cfg.rest_task_pos_scale) * pos_error
            )
            pos_completion = pos_error < self.cfg.completion_pos_thd
            pos_far_away = pos_error > self.cfg.obj_command_max_dist_pos

            quat_error = angle_error(
                obs["quat"],
                ref_quat,
                current_format=self.env.cfg.obs_cfg.rot_format,
                target_format="quat",
            )
            quat_reward = torch.exp(-float(self.cfg.task_rot_scale) * quat_error)
            rest_quat_reward = torch.exp(
                -float(self.cfg.rest_task_rot_scale) * quat_error
            )
            quat_completion = quat_error < self.cfg.completion_rot_thd
            quat_far_away = quat_error > self.cfg.obj_command_max_dist_rot

            object_reward = pos_reward * quat_reward
            object_rest_reward = rest_pos_reward * rest_quat_reward
            object_completion = torch.logical_and(pos_completion, quat_completion)
            object_far_away = torch.logical_or(pos_far_away, quat_far_away)

            if ref_arti.shape[1] > 0:
                arti_error = torch.abs(obs["arti"] - ref_arti).sum(dim=-1)
                arti_reward = torch.exp(-float(self.cfg.task_arti_scale) * arti_error)
                rest_arti_reward = torch.exp(
                    -float(self.cfg.rest_task_arti_scale) * arti_error
                )
                arti_completion = arti_error < self.cfg.completion_arti_thd
                arti_far_away = arti_error > self.cfg.obj_command_max_dist_arti
                object_reward = object_reward * arti_reward
                object_rest_reward = object_rest_reward * rest_arti_reward
                object_completion = torch.logical_and(
                    object_completion,
                    arti_completion,
                )
                object_far_away = torch.logical_or(object_far_away, arti_far_away)
                if self.log_metrics:
                    self.env.extras[f"log/{object_id}_arti_error"] = arti_error
                    self.env.extras[f"log/{object_id}_motion_tracking_arti_rew"] = (
                        arti_reward
                    )
                    self.env.extras[f"log/{object_id}_rest_tracking_arti_rew"] = (
                        rest_arti_reward
                    )
                    self.env.extras[f"log/{object_id}_arti_completion"] = (
                        arti_completion.to(torch.float32)
                    )
                    self.env.extras[f"log/{object_id}_arti_far_away"] = (
                        arti_far_away.to(torch.float32)
                    )

            object_tracking_term = torch.where(
                motion_tracking,
                object_reward,
                torch.where(
                    rest_tracking,
                    object_rest_reward,
                    torch.zeros_like(object_reward),
                ),
            )
            reward_sum += object_tracking_term
            object_pose_achieved = torch.logical_and(
                object_pose_achieved,
                torch.where(enabled, object_completion, torch.ones_like(enabled)),
            )
            early_terminate = torch.logical_or(
                early_terminate,
                torch.logical_and(enabled, object_far_away),
            )
            object_far_away_mask[:, object_index] = torch.logical_and(
                enabled,
                object_far_away,
            )

            if self.log_metrics:
                self.env.extras[f"log/{object_id}_tracking_enabled"] = enabled.to(
                    torch.float32
                )
                self.env.extras[f"log/{object_id}_motion_tracking_enabled"] = (
                    motion_tracking.to(torch.float32)
                )
                self.env.extras[f"log/{object_id}_rest_tracking_enabled"] = (
                    rest_tracking.to(torch.float32)
                )
                self.env.extras[f"log/{object_id}_pos_error"] = pos_error
                self.env.extras[f"log/{object_id}_quat_error"] = quat_error
                self.env.extras[f"log/{object_id}_motion_tracking_rew"] = object_reward
                self.env.extras[f"log/{object_id}_rest_tracking_rew"] = (
                    object_rest_reward
                )

        object_tracking_reward = reward_sum / enabled_count
        self._last_object_tracking_enabled_mask = enabled_mask
        self._last_object_far_away_mask = object_far_away_mask
        if self.should_log_essential_metrics():
            self.env.extras["log/object_tracking_enabled_count"] = enabled_count
            self.env.extras["log/object_tracking_product_rew"] = object_tracking_reward
            self.env.extras["log/all_objects_pose_achieved"] = object_pose_achieved.to(
                torch.float32
            )
        if self.should_log_full_metrics():
            self.env.extras["log/object_tracking_enabled_mask"] = enabled_mask.to(
                torch.float32
            )
            self.env.extras["log/object_motion_tracking_enabled_mask"] = (
                motion_tracking_mask.to(torch.float32)
            )
            self.env.extras["log/object_rest_tracking_enabled_mask"] = (
                rest_tracking_mask.to(torch.float32)
            )
            self.env.extras["log/object_far_away_mask"] = object_far_away_mask.to(
                torch.float32
            )
            self.env.extras["log/object_far_away"] = early_terminate.to(torch.float32)

        self._object_tracking_cache = (
            object_tracking_reward,
            object_pose_achieved,
            early_terminate,
        )
        return self._object_tracking_cache

    def log_paper_metrics(self, named_obs: dict[str, torch.Tensor]) -> None:
        if not self.should_log_essential_metrics():
            return

        reference_ts = self.current_reference_timesteps()
        active_mask = self._reference_active_mask(reference_ts)
        pos_errors = []
        quat_errors = []
        arti_errors = []
        for object_index, object_id in enumerate(self.object_ids):
            obs = self.env.observation_manager.get_named_object_obs(
                named_obs,
                object_id,
            )
            ref_pos = self.object_ref_pos[reference_ts, object_index]
            ref_quat = self.object_ref_quat[reference_ts, object_index]
            ref_arti = self._object_reference_arti_for_sim_joints(
                object_index=object_index,
                object_id=object_id,
                reference_timesteps=reference_ts,
            )
            pos_errors.append(torch.linalg.norm(obs["pos"] - ref_pos, dim=-1))
            quat_errors.append(
                angle_error(
                    obs["quat"],
                    ref_quat,
                    current_format=self.env.cfg.obs_cfg.rot_format,
                    target_format="quat",
                )
            )
            if ref_arti.shape[1] > 0:
                arti_errors.append(torch.abs(obs["arti"] - ref_arti).sum(dim=-1))

        pos_error = torch.stack(pos_errors, dim=1)
        quat_error = torch.stack(quat_errors, dim=1)
        inactive = torch.zeros_like(pos_error)
        self.env.extras["log/paper_object_pos_error_m"] = (
            torch.where(
                active_mask,
                pos_error,
                inactive,
            )
            .max(dim=1)
            .values
        )
        self.env.extras["log/paper_object_quat_error_rad"] = (
            torch.where(
                active_mask,
                quat_error,
                inactive,
            )
            .max(dim=1)
            .values
        )
        if arti_errors:
            arti_error = torch.stack(arti_errors, dim=1)
            self.env.extras["log/paper_object_arti_error"] = arti_error.max(
                dim=1
            ).values

        final_reference_timestep = self.env.task_manager.total_steps - 1
        final_reference_timesteps = torch.full(
            (self.env.num_envs,),
            final_reference_timestep,
            dtype=torch.long,
            device=self.env.device,
        )
        final_pos_errors = []
        final_quat_errors = []
        final_arti_errors = []
        final_success = torch.ones(
            self.env.num_envs,
            dtype=torch.bool,
            device=self.env.device,
        )
        for object_index, object_id in enumerate(self.object_ids):
            obs = self.env.observation_manager.get_named_object_obs(
                named_obs,
                object_id,
            )
            ref_pos = self.object_ref_pos[final_reference_timestep, object_index]
            ref_quat = self.object_ref_quat[final_reference_timestep, object_index]
            ref_arti = self._object_reference_arti_for_sim_joints(
                object_index=object_index,
                object_id=object_id,
                reference_timesteps=final_reference_timesteps,
            )
            final_pos_error = torch.linalg.norm(
                obs["pos"] - ref_pos.unsqueeze(0),
                dim=-1,
            )
            final_quat_error = angle_error(
                obs["quat"],
                ref_quat.unsqueeze(0).expand(self.env.num_envs, -1),
                current_format=self.env.cfg.obs_cfg.rot_format,
                target_format="quat",
            )
            object_success = torch.logical_and(
                final_pos_error <= float(self.cfg.task_success_pos_thd),
                final_quat_error <= float(self.cfg.task_success_rot_thd),
            )
            if ref_arti.shape[1] > 0:
                final_arti_abs_error = torch.abs(obs["arti"] - ref_arti)
                final_arti_error = final_arti_abs_error.sum(dim=-1)
                final_arti_errors.append(final_arti_error)
                thresholds = self._final_scene_arti_thresholds(object_id)
                if thresholds.numel() != ref_arti.shape[1]:
                    raise ValueError(
                        f"Expected articulation threshold count {ref_arti.shape[1]} "
                        f"for object {object_id!r}, got {thresholds.numel()}."
                    )
                object_success = torch.logical_and(
                    object_success,
                    (final_arti_abs_error <= thresholds.unsqueeze(0)).all(dim=-1),
                )
            final_pos_errors.append(final_pos_error)
            final_quat_errors.append(final_quat_error)
            final_success = torch.logical_and(final_success, object_success)

        self.env.extras["log/paper_final_scene_max_pos_error_m"] = (
            torch.stack(
                final_pos_errors,
                dim=1,
            )
            .max(dim=1)
            .values
        )
        self.env.extras["log/paper_final_scene_max_quat_error_rad"] = (
            torch.stack(
                final_quat_errors,
                dim=1,
            )
            .max(dim=1)
            .values
        )
        if final_arti_errors:
            self.env.extras["log/paper_final_scene_max_arti_error"] = (
                torch.stack(
                    final_arti_errors,
                    dim=1,
                )
                .max(dim=1)
                .values
            )
        self.env.extras["log/paper_final_scene_configuration_success"] = (
            final_success.to(torch.float32)
        )

        self._log_paper_hand_obj_rel_metrics(named_obs)

    def object_far_away_mask(self, named_obs, **kwargs) -> torch.Tensor:
        self.object_tracking_progress(named_obs=named_obs, **kwargs)
        if self._last_object_far_away_mask is None:
            raise ValueError("Object tracking did not produce a far-away mask.")
        return self._last_object_far_away_mask.any(dim=1)


class MultiObjectRewardTerminationComputation(RewardTerminationComputation):
    def _final_scene_arti_thresholds(self, object_id: str) -> torch.Tensor:
        entity = self.env.physics_manager.object_entities[object_id]
        thresholds = []
        for joint in entity.joints:
            if joint.joint_type == "prismatic":
                thresholds.append(float(self.cfg.task_success_prismatic_arti_thd))
            elif joint.joint_type in ("revolute", "continuous"):
                thresholds.append(float(self.cfg.task_success_revolute_arti_thd))
            else:
                raise ValueError(
                    f"Unsupported articulation joint type {joint.joint_type!r} for "
                    f"object {object_id!r} joint {joint.name!r}; expected prismatic, "
                    "revolute, or continuous."
                )
        return torch.as_tensor(
            thresholds,
            dtype=torch.float32,
            device=self.env.device,
        )

    def final_scene_configuration_success_mask(
        self,
        named_obs,
        **kwargs,
    ) -> torch.Tensor:
        final_reference_timestep = self.env.task_manager.total_steps - 1
        final_reference_timesteps = torch.full(
            (self.env.num_envs,),
            final_reference_timestep,
            dtype=torch.long,
            device=self.env.device,
        )
        success = torch.ones(
            self.env.num_envs,
            dtype=torch.bool,
            device=self.env.device,
        )

        for object_index, object_id in enumerate(self.common.object_ids):
            obs = self.env.observation_manager.get_named_object_obs(
                named_obs,
                object_id,
            )
            ref_pos = self.common.object_ref_pos[final_reference_timestep, object_index]
            ref_quat = self.common.object_ref_quat[
                final_reference_timestep,
                object_index,
            ]
            ref_arti = self.common._object_reference_arti_for_sim_joints(
                object_index=object_index,
                object_id=object_id,
                reference_timesteps=final_reference_timesteps,
            )

            pos_error = torch.linalg.norm(obs["pos"] - ref_pos.unsqueeze(0), dim=-1)
            quat_error = angle_error(
                obs["quat"],
                ref_quat.unsqueeze(0).expand(self.env.num_envs, -1),
                current_format=self.env.cfg.obs_cfg.rot_format,
                target_format="quat",
            )
            object_success = torch.logical_and(
                pos_error <= float(self.cfg.task_success_pos_thd),
                quat_error <= float(self.cfg.task_success_rot_thd),
            )

            if ref_arti.shape[1] > 0:
                thresholds = self._final_scene_arti_thresholds(object_id)
                if thresholds.numel() != ref_arti.shape[1]:
                    raise ValueError(
                        f"Expected articulation threshold count {ref_arti.shape[1]} "
                        f"for object {object_id!r}, got {thresholds.numel()}."
                    )
                arti_success = (
                    torch.abs(obs["arti"] - ref_arti) <= thresholds.unsqueeze(0)
                ).all(dim=-1)
                object_success = torch.logical_and(object_success, arti_success)

            success = torch.logical_and(success, object_success)
            if self.common.log_metrics:
                self.env.extras[f"log/task_success_{object_id}_pos_error"] = pos_error
                self.env.extras[f"log/task_success_{object_id}_quat_error"] = quat_error
                self.env.extras[f"log/task_success_{object_id}"] = object_success.to(
                    torch.float32
                )

        if self.common.should_log_essential_metrics():
            self.env.extras["log/final_scene_configuration_success"] = success.to(
                torch.float32
            )
        return success

    def actual_success_mask(
        self,
        pose_achieved: torch.Tensor,
        named_obs=None,
        **kwargs,
    ) -> torch.Tensor:
        if named_obs is None:
            raise ValueError(
                "MultiObjectRewardTerminationComputation.actual_success_mask "
                "requires named_obs to compute final scene configuration success."
            )
        unclamped_reference_timestep = self.env.get_reference_timesteps_unclamped()
        max_reference_timestep = self.env.task_manager.total_steps - 1
        hold_steps = int(self.cfg.last_reference_hold_steps)
        truncation = unclamped_reference_timestep > (
            max_reference_timestep + hold_steps
        )
        final_scene_success = self.final_scene_configuration_success_mask(
            named_obs=named_obs,
            **kwargs,
        )
        actual_success = torch.logical_and(final_scene_success, truncation)
        if self.common.should_log_essential_metrics():
            self.env.extras["log/actual_success"] = actual_success.to(torch.float32)
        return actual_success

    def far_away_mask(
        self, named_obs, disc_obs, action, prev_named_obs
    ) -> torch.Tensor:
        hand_far_away = torch.zeros_like(self.env.timesteps, dtype=torch.bool).to(
            self.env.device
        )
        if self.common.has_hand_obj_rel:
            _, hand_far_away, _ = self.common.hand_obj_relative_reward(
                named_obs=named_obs,
                disc_obs=disc_obs,
                action=action,
                prev_named_obs=prev_named_obs,
            )

        object_far_away = self.common.object_far_away_mask(
            named_obs=named_obs,
            disc_obs=disc_obs,
            action=action,
            prev_named_obs=prev_named_obs,
        )
        far_away = torch.logical_or(hand_far_away, object_far_away)
        if self.common.should_log_essential_metrics():
            self.env.extras["log/multi_object_far_away"] = object_far_away.to(
                torch.float32
            )
            self.env.extras["log/far_away"] = far_away.to(torch.float32)
        return far_away


class MultiObjectRewardManager(RewardManager):
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.common = MultiObjectRewardCommonComputation(cfg, env)
        self.reward_computation = MultiObjectRewardComputation(self.common)
        self.termination_computation = MultiObjectRewardTerminationComputation(
            self.common
        )

    def refresh_reference_buffers(self) -> None:
        super().refresh_reference_buffers()
        self.common.refresh_object_tracking_reference()


class MultiObjectRewardComputation(RewardComputation):
    def reward_term_weight(self, name: str, weight: float) -> float | torch.Tensor:
        if name != "retargeted_joint_mse":
            return weight
        enabled_mask = self.common.current_retargeted_joint_mse_enabled_mask()
        base_weight = torch.full_like(
            self.env.timesteps,
            float(weight),
            dtype=torch.float32,
            device=self.env.device,
        )
        object_tracking = self.reward_terms.get("object_tracking")
        if object_tracking is None:
            return torch.where(enabled_mask, base_weight, torch.zeros_like(base_weight))
        object_tracking_weight, _ = object_tracking
        reach_phase = self.common.current_reach_phase_mask()
        scheduled_weight = torch.where(
            reach_phase,
            base_weight + float(object_tracking_weight),
            base_weight,
        )
        return torch.where(
            enabled_mask,
            scheduled_weight,
            torch.zeros_like(scheduled_weight),
        )
