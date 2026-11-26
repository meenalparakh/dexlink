from __future__ import annotations

import numpy as np
import torch

from dexlink.src.rl.world.art_object import ArtObject
from dexlink.src.rl.world.cardboard_object import CardboardObject
from dexlink.src.rl.world.reset_manager import ResetManager
from dexlink.src.rl.world.robot import Hand


class MultiObjectResetManager(ResetManager):
    """Reset manager that writes every scene object from scene_object_* buffers."""

    def sync_reference_buffers(self) -> None:
        super().sync_reference_buffers()
        task = self.env.task_manager
        motion = self._retargeted_motion()
        rest_values = motion.get("cardboard_rest_arti_vals")
        rest_names = motion.get("cardboard_rest_arti_names")
        if (rest_values is None) != (rest_names is None):
            raise ValueError(
                "Retargeted motion must provide cardboard rest articulation values "
                "and names together."
            )
        self.frame_cardboard_rest_arti = None
        self.cardboard_rest_arti_names: list[str] = []
        if rest_values is not None:
            rest_array = np.asarray(rest_values, dtype=np.float32)
            if rest_array.shape != tuple(self.frame_arti.shape):
                raise ValueError(
                    "Expected cardboard rest articulation shape to match canonical "
                    f"reference articulation {tuple(self.frame_arti.shape)}, got "
                    f"{rest_array.shape}."
                )
            self.frame_cardboard_rest_arti = torch.as_tensor(
                rest_array, dtype=torch.float32, device=self.env.device
            )
            self.cardboard_rest_arti_names = [str(name) for name in rest_names]
        self.frame_scene_object_pos = torch.as_tensor(
            task.scene_object_pos, dtype=torch.float32, device=self.env.device
        )
        self.frame_scene_object_quat = torch.as_tensor(
            task.scene_object_quat, dtype=torch.float32, device=self.env.device
        )
        self.frame_scene_object_arti_vals = torch.as_tensor(
            task.scene_object_arti_vals, dtype=torch.float32, device=self.env.device
        )
        self.scene_object_arti_valid_mask = np.asarray(
            task.scene_object_arti_valid_mask, dtype=bool
        )
        self.scene_object_arti_names = list(task.scene_object_arti_names)
        self.scene_active_object_index = torch.as_tensor(
            task.scene_active_object_index, dtype=torch.long, device=self.env.device
        )

    def _sequence_phase_intervals(self) -> list[tuple[str, int, int]]:
        phase_names = list(getattr(self.task.original_demo, "phase_names", []))
        phase_lengths = np.asarray(
            getattr(self.task.original_demo, "phase_lengths", np.asarray([])),
            dtype=np.int64,
        )
        if len(phase_names) == 0 or phase_lengths.shape != (len(phase_names),):
            raise ValueError(
                "Multi-object reset sampling requires phase_names and phase_lengths "
                f"with matching lengths, got {len(phase_names)} names and "
                f"phase_lengths shape {phase_lengths.shape}."
            )
        if np.any(phase_lengths <= 0):
            raise ValueError(
                f"Expected all phase_lengths to be positive, got {phase_lengths}."
            )
        total_phase_frames = int(phase_lengths.sum())
        if total_phase_frames != self.task.total_steps:
            raise ValueError(
                "Expected phase_lengths to sum to task.total_steps, got "
                f"{total_phase_frames} and {self.task.total_steps}."
            )

        intervals: list[tuple[str, int, int]] = []
        cursor = 0
        for phase_name, phase_length in zip(phase_names, phase_lengths.tolist()):
            phase_start = cursor
            phase_end = cursor + int(phase_length) - 1
            cursor = phase_end + 1
            intervals.append((phase_name, phase_start, phase_end))
        return intervals

    def _phase_intervals_in_start_range(self) -> list[tuple[str, int, int]]:
        start_idx, end_idx = self._resolve_start_range()
        intervals: list[tuple[str, int, int]] = []
        for phase_name, phase_start, phase_end in self._sequence_phase_intervals():
            lo = max(phase_start, start_idx)
            hi = min(phase_end, end_idx)
            if lo <= hi:
                intervals.append((phase_name, lo, hi))
        if len(intervals) == 0:
            raise ValueError(
                "No exported phase interval overlaps reset_cfg.start_range "
                f"{(start_idx, end_idx)}."
            )
        return intervals

    def _eligible_density_sample_timesteps(self) -> np.ndarray:
        intervals = self._phase_intervals_in_start_range()
        candidates = np.concatenate(
            [np.arange(lo, hi + 1, dtype=np.int64) for _, lo, hi in intervals]
        )
        invalid_mask = self.drake_collision_refinement_invalid_frame_mask
        if invalid_mask is None:
            invalid_mask = self.drake_collision_refinement_unresolved_frame_mask
        if invalid_mask is not None:
            candidates = candidates[~invalid_mask[candidates]]
        if candidates.shape[0] == 0:
            raise ValueError(
                "No valid multi-object reset timesteps are available after applying "
                "the active start range and Drake refinement mask."
            )
        return candidates

    def _sample_density_weighted_start_timesteps(self, n_samples: int) -> np.ndarray:
        candidates = self._eligible_density_sample_timesteps()
        timestep_density = (
            self.env.reset_timestep_density[candidates]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        max_density = float(timestep_density.max())
        normalized_density = (
            timestep_density / max_density if max_density > 0.0 else timestep_density
        )
        weights = np.exp(-normalized_density)
        probabilities = weights / weights.sum()
        return np.random.choice(candidates, size=n_samples, p=probabilities)

    def sample_start_timesteps(self, env_ids: np.ndarray) -> np.ndarray:
        if self.env.mode == "eval":
            sampled = np.full(len(env_ids), self.cfg.reset_cfg.eval_start_timestep)
            return self._resolve_refinement_reset_timesteps(sampled)
        mode = str(self.cfg.reset_cfg.reset_timestep_sampling_mode)
        if mode in {"default", "density_weighted"}:
            sampled = self._sample_density_weighted_start_timesteps(len(env_ids))
        elif mode == "uniform":
            sampled = np.random.choice(
                self._eligible_density_sample_timesteps(),
                size=len(env_ids),
            )
        else:
            raise ValueError(
                "Expected reset_cfg.reset_timestep_sampling_mode to be one of "
                "{'default', 'density_weighted', 'uniform'}, got "
                f"{mode!r}."
            )
        return self._resolve_refinement_reset_timesteps(sampled)

    def hand_reset_noise(self, frame_idx: np.ndarray) -> torch.Tensor:
        if hasattr(self.task.original_demo, "n_reach_frames"):
            return super().hand_reset_noise(frame_idx)

        frame_idx_array = np.asarray(frame_idx, dtype=np.int64)
        if frame_idx_array.ndim != 1:
            raise ValueError(
                "Expected frame_idx to be a 1D array of frame indices, "
                f"got shape {frame_idx_array.shape}."
            )

        reach_noise = float(self.cfg.reset_cfg.hand_noise_reach_phase)
        lift_noise = float(self.cfg.reset_cfg.hand_noise_lift_phase)
        frame_idx_t = torch.as_tensor(
            frame_idx_array,
            dtype=torch.float32,
            device=self.env.device,
        )
        noise = torch.full(
            (len(frame_idx_array),),
            lift_noise,
            dtype=torch.float32,
            device=self.env.device,
        )

        for phase_name, phase_start, phase_end in self._sequence_phase_intervals():
            phase_mask = (frame_idx_t >= float(phase_start)) & (
                frame_idx_t <= float(phase_end)
            )
            phase_name_lower = str(phase_name).lower()
            if "reach" in phase_name_lower:
                noise = torch.where(
                    phase_mask,
                    torch.full_like(noise, reach_noise),
                    noise,
                )
            elif "transition" in phase_name_lower:
                denom = max(float(phase_end - phase_start), 1.0)
                progress = (frame_idx_t - float(phase_start)) / denom
                transition_noise = reach_noise + progress * (lift_noise - reach_noise)
                noise = torch.where(phase_mask, transition_noise, noise)

        return noise

    def set_object_state_from_retargeted_results(
        self, frame_idx: np.ndarray, env_ids: np.ndarray
    ):
        frame_idx_array = np.asarray(frame_idx, dtype=np.int64)
        env_ids_array = np.asarray(env_ids, dtype=np.int64)
        if frame_idx_array.ndim != 1:
            raise ValueError(
                f"Expected frame_idx shape (N,), got {frame_idx_array.shape}."
            )
        if env_ids_array.ndim != 1:
            raise ValueError(f"Expected env_ids shape (N,), got {env_ids_array.shape}.")
        if frame_idx_array.shape[0] != env_ids_array.shape[0]:
            raise ValueError(
                "Expected frame_idx and env_ids to have matching lengths, got "
                f"{frame_idx_array.shape[0]} and {env_ids_array.shape[0]}."
            )

        noise = float(self.cfg.reset_cfg.obj_noise)
        task = self.env.task_manager
        object_entities = self.physics.object_entities
        canonical_return = None

        for object_index, object_id in enumerate(task.scene_object_ids):
            asset: ArtObject = object_entities[object_id]
            if object_index == task.canonical_object_index:
                pos = self.frame_pos[frame_idx_array].to(self.env.device)
                quat = self.frame_quat[frame_idx_array].to(self.env.device)
                jpos_tensor = (
                    None
                    if self.frame_arti is None
                    else self.frame_arti[frame_idx_array].to(self.env.device)
                )
            else:
                pos = self.frame_scene_object_pos[frame_idx_array, object_index].to(
                    self.env.device
                )
                quat = self.frame_scene_object_quat[frame_idx_array, object_index].to(
                    self.env.device
                )
                jpos_tensor = self._scene_object_jpos(
                    object_index=object_index,
                    asset=asset,
                    frame_idx=frame_idx_array,
                )

            if noise > 0.0:
                pos = pos + torch.randn_like(pos) * noise
                quat = quat + torch.randn_like(quat) * noise
                quat = quat / torch.norm(quat, dim=-1, keepdim=True)
                if jpos_tensor is not None:
                    jpos_tensor = jpos_tensor + torch.randn_like(jpos_tensor) * noise

            state_kwargs = {
                "pos": pos,
                "quat": quat,
                "jpos": jpos_tensor,
                "env_idx": env_ids_array,
            }
            if isinstance(asset, CardboardObject):
                if object_index != task.canonical_object_index:
                    raise ValueError(
                        "Cardboard constitutive resets are only supported for the "
                        "canonical scene object."
                    )
                if self.frame_cardboard_rest_arti is None:
                    raise ValueError(
                        "CardboardObject reset requires recorded "
                        "cardboard_rest_arti_vals in the reference motion."
                    )
                asset_joint_names = [joint.name for joint in asset.joints]
                missing_rest_joints = [
                    name
                    for name in asset_joint_names
                    if name not in self.cardboard_rest_arti_names
                ]
                if missing_rest_joints:
                    raise ValueError(
                        "Recorded cardboard rest state is missing simulated joints "
                        f"{missing_rest_joints}."
                    )
                rest_indices = [
                    self.cardboard_rest_arti_names.index(name)
                    for name in asset_joint_names
                ]
                state_kwargs["rest_jpos"] = self.frame_cardboard_rest_arti[
                    frame_idx_array
                ][:, rest_indices]
            asset.set_state(**state_kwargs)
            if object_index == task.canonical_object_index:
                canonical_return = (pos, quat, jpos_tensor)

        if canonical_return is None:
            raise ValueError("Canonical scene object was not reset.")
        return canonical_return

    def _scene_object_jpos(
        self,
        *,
        object_index: int,
        asset: ArtObject,
        frame_idx: np.ndarray,
    ) -> torch.Tensor | None:
        if len(asset.joints) == 0:
            return None

        valid_indices = np.flatnonzero(self.scene_object_arti_valid_mask[object_index])
        exported_joint_names = list(self.scene_object_arti_names[object_index])
        if len(exported_joint_names) != len(valid_indices):
            raise ValueError(
                "Expected scene_object_arti_names length to match valid articulation "
                f"count for object index {object_index}, got "
                f"{len(exported_joint_names)} names and {len(valid_indices)} "
                "valid values."
            )
        missing = [
            joint.name
            for joint in asset.joints
            if joint.name not in exported_joint_names
        ]
        if missing:
            raise ValueError(
                "Scene object articulation is missing joints required by the "
                f"simulated object {asset.obj_cfg['name']!r}: {missing}."
            )
        joint_indices = [
            exported_joint_names.index(joint.name) for joint in asset.joints
        ]
        valid_values = self.frame_scene_object_arti_vals[frame_idx, object_index][
            :, valid_indices
        ]
        return valid_values[:, joint_indices].to(self.env.device)

    def _first_moving_object_index(self) -> int:
        active_indices = self.scene_active_object_index.detach().cpu().numpy()
        active_indices = active_indices[active_indices >= 0]
        if active_indices.shape[0] > 0:
            return int(active_indices[0])

        scene_pos = np.asarray(self.task.scene_object_pos)
        pos_delta = np.linalg.norm(
            scene_pos[1:] - scene_pos[:-1],
            axis=-1,
        )
        moving = np.argwhere(pos_delta > 1.0e-5)
        if moving.shape[0] == 0:
            raise ValueError(
                "Could not identify the first moving object from "
                "scene_active_object_index or scene_object_pos deltas."
            )
        return int(moving[0, 1])

    def _eval_target_object_id(self) -> str:
        object_index = self._first_moving_object_index()
        if object_index < 0 or object_index >= len(self.task.scene_object_ids):
            raise ValueError(
                f"First moving object index {object_index} is outside scene objects "
                f"{self.task.scene_object_ids}."
            )
        return self.task.scene_object_ids[object_index]

    def _get_demo_object_pcd(
        self, frame_idx: np.ndarray, *, num_pts_sample: int
    ) -> torch.Tensor:
        frame_idx_array = np.asarray(frame_idx, dtype=np.int64)
        object_id = self._eval_target_object_id()
        object_index = self.task.scene_object_ids.index(object_id)
        entity: ArtObject = self.physics.object_entities[object_id]
        pos = self.frame_scene_object_pos[frame_idx_array, object_index].to(
            self.env.device
        )
        quat = self.frame_scene_object_quat[frame_idx_array, object_index].to(
            self.env.device
        )
        arti = self._scene_object_jpos(
            object_index=object_index,
            asset=entity,
            frame_idx=frame_idx_array,
        )
        if arti is None:
            arti = torch.empty(
                (len(frame_idx_array), 0),
                dtype=torch.float32,
                device=self.env.device,
            )

        return self.env.observation_manager.compute_object_pcd_obs_for_entity(
            entity,
            pos=pos,
            quat=quat,
            arti=arti,
            normals=False,
            num_pts_sample=num_pts_sample,
        )

    def _get_demo_object_pcd_center(self, frame_idx: np.ndarray) -> torch.Tensor:
        demo_object_pcd = self._get_demo_object_pcd(
            frame_idx,
            num_pts_sample=self.env.observation_manager.policy_object_pcd_n_pts,
        )
        return demo_object_pcd.mean(dim=1)

    def _get_demo_object_pcd_min_z(self, frame_idx: np.ndarray) -> torch.Tensor:
        demo_object_pcd = self._get_demo_object_pcd(
            frame_idx,
            num_pts_sample=-1,
        )
        return demo_object_pcd[..., 2].amin(dim=1)

    def _object_alignment_anchor(self, object_pcd_world: torch.Tensor) -> torch.Tensor:
        if object_pcd_world.ndim != 3 or object_pcd_world.shape[2] != 3:
            raise ValueError(
                "Expected object_pcd_world shape (num_envs, num_points, 3), "
                f"got {tuple(object_pcd_world.shape)}."
            )
        if object_pcd_world.shape[0] != self.env.num_envs:
            raise ValueError(
                "Expected object_pcd_world first dim to match num_envs "
                f"{self.env.num_envs}, got {object_pcd_world.shape[0]}."
            )
        anchor = torch.empty(
            (self.env.num_envs, 3),
            dtype=torch.float32,
            device=object_pcd_world.device,
        )
        anchor[:, :2] = object_pcd_world[..., :2].mean(dim=1)
        anchor[:, 2] = object_pcd_world[..., 2].amin(dim=1)
        return anchor

    def _get_demo_object_alignment_anchor(self, frame_idx: np.ndarray) -> torch.Tensor:
        demo_object_pcd = self._get_demo_object_pcd(
            frame_idx,
            num_pts_sample=-1,
        )
        return self._object_alignment_anchor(demo_object_pcd)

    def real_eval_align_wrist_position_to_object_anchor(
        self,
        *,
        side: str,
        wrist_pos: torch.Tensor,
        frame_idx: torch.Tensor | np.ndarray,
        real_object_pcd_world: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.env.observation_manager.scene_cloud_contract is not None:
            # Scene policies use the calibrated workspace; no single-object anchor exists.
            return wrist_pos
        if side != "right":
            raise ValueError(
                "Real eval object alignment currently expects side='right', got "
                f"{side!r}."
            )
        expected_wrist_shape = (self.env.num_envs, 3)
        if tuple(wrist_pos.shape) != expected_wrist_shape:
            raise ValueError(
                "Expected wrist_pos shape "
                f"{expected_wrist_shape}, got {tuple(wrist_pos.shape)}."
            )
        frame_idx_tensor = torch.as_tensor(
            frame_idx, dtype=torch.long, device=self.env.device
        ).reshape(-1)
        if tuple(frame_idx_tensor.shape) != (self.env.num_envs,):
            raise ValueError(
                "Expected frame_idx shape "
                f"{(self.env.num_envs,)}, got {tuple(frame_idx_tensor.shape)}."
            )
        if real_object_pcd_world is None:
            if not hasattr(self.physics, "get_latest_isolated_object_pcd"):
                raise AttributeError(
                    "Real eval object alignment requires "
                    "physics.get_latest_isolated_object_pcd()."
                )
            real_object_pcd_world = self.physics.get_latest_isolated_object_pcd()
        real_object_pcd_world = torch.as_tensor(
            real_object_pcd_world, dtype=torch.float32, device=self.env.device
        )
        demo_anchor = self._get_demo_object_alignment_anchor(
            frame_idx_tensor.detach().cpu().numpy()
        )
        real_anchor = self._object_alignment_anchor(real_object_pcd_world)
        object_anchor_offset = real_anchor - demo_anchor
        aligned_wrist_pos = wrist_pos.to(
            device=self.env.device, dtype=torch.float32
        ).clone()
        aligned_wrist_pos = aligned_wrist_pos + object_anchor_offset
        return aligned_wrist_pos.to(device=wrist_pos.device, dtype=wrist_pos.dtype)

    def reset_robot_state_for_eval(
        self,
        frame_idx: np.ndarray,
        env_ids: np.ndarray,
    ) -> None:
        print("Moving robot to eval reset wrist pose")
        if self.env.cfg.backend == "real" and not self.task.has_hand("right"):
            raise ValueError(
                "Real-backend evaluation reset requires an active right hand for action-manager wrist control."
            )

        if (
            self.env.cfg.backend == "sim"
            and self.cfg.reset_cfg.retargeted_hand_reference_mode
            == "default_configuration"
        ):
            self.set_robot_state_from_retargeted_results(frame_idx, env_ids)
            print("Robot moved to default-configuration eval reset pose.")
            return

        if self.env.cfg.backend == "sim":
            noise = self.hand_reset_noise(frame_idx)
            for side, spec in self.env.action_manager.hand_specs.items():
                frame_jvals = getattr(self, f"frame_{side}_reset_jvals", None)
                frame_wrist_pos = getattr(self, f"frame_{side}_wrist_pos", None)
                frame_wrist_quat = getattr(self, f"frame_{side}_wrist_quat", None)
                if (
                    frame_jvals is None
                    or frame_wrist_pos is None
                    or frame_wrist_quat is None
                ):
                    raise ValueError(
                        f"Evaluation reset requires retargeted {side!r}-hand joints and wrist pose."
                    )
                side_jpos = frame_jvals[frame_idx].to(self.env.device)
                side_jpos = self._apply_hand_reset_noise(
                    side=side,
                    jpos=side_jpos,
                    noise=noise,
                )
                finger_target_indices = torch.as_tensor(
                    spec.hand_target_indices,
                    dtype=torch.long,
                    device=self.env.device,
                )
                self.env.action_manager.move_to_reset_target(
                    side=side,
                    env_ids=env_ids,
                    target_wrist_pos=frame_wrist_pos[frame_idx].to(self.env.device),
                    target_wrist_quat=frame_wrist_quat[frame_idx].to(self.env.device),
                    target_hand_qpos=side_jpos.index_select(1, finger_target_indices),
                )
            print("Robot moved to eval reset wrist pose.")
            return

        right_robot: Hand = self.physics.scene_entities["right"]
        frame_right_jvals = getattr(self, "frame_right_reset_jvals", None)
        frame_right_wrist_pos = getattr(self, "frame_right_wrist_pos", None)
        frame_right_wrist_quat = getattr(self, "frame_right_wrist_quat", None)
        if (
            frame_right_jvals is None
            or frame_right_wrist_pos is None
            or frame_right_wrist_quat is None
        ):
            raise ValueError(
                "Evaluation reset requires retargeted right-hand joints and wrist pose."
            )

        noise = self.hand_reset_noise(frame_idx)
        right_jpos = frame_right_jvals[frame_idx].to(self.env.device)
        right_jpos = self._apply_hand_reset_noise(
            side="right",
            jpos=right_jpos,
            noise=noise,
        )

        finger_target_indices = torch.as_tensor(
            self.env.action_manager.hand_target_indices,
            dtype=torch.long,
            device=self.env.device,
        )
        target_hand_qpos = right_jpos.index_select(1, finger_target_indices)
        target_wrist_quat = frame_right_wrist_quat[frame_idx].to(self.env.device)

        if (
            self.env.cfg.backend == "real"
            and self.env.observation_manager.scene_cloud_contract is not None
        ):
            target_wrist_pos = frame_right_wrist_pos[frame_idx].to(self.env.device)
        elif self.env.cfg.backend == "real":
            self.env.observation_manager.reset_step_cache()
            reset_obs = self.env.observation_manager.compute_observations()
            current_wrist_pos = self.env.current_wrist_pos_by_side.get("right")
            if current_wrist_pos is None:
                raise ValueError(
                    "Evaluation reset requires env.current_wrist_pos after computing "
                    "observations."
                )

            current_qpos = self.physics.get_hand_dof_positions(
                right_robot,
                right_robot.dof_map,
            ).detach()
            hand_pcd_world = right_robot.get_robot_pcd(
                current_qpos,
                links_to_include=right_robot.hand_tree_link_names,
            )
            hand_pcd_rel = hand_pcd_world - current_wrist_pos[:, None, :]
            object_pcd_world = self.physics.get_current_isolated_object_pcd()
            object_pcd_rel = object_pcd_world - current_wrist_pos[:, None, :]
            demo_object_pcd_center = self._get_demo_object_pcd_center(frame_idx)
            demo_wrist_offset = (
                frame_right_wrist_pos[frame_idx].to(self.env.device)
                - demo_object_pcd_center
            )
            target_wrist_pos = torch.empty(
                (len(env_ids), 3),
                dtype=torch.float32,
                device=self.env.device,
            )

            exclusion_radius = float(self.EVAL_RESET_HAND_EXCLUSION_RADIUS_M)
            for row, env_id in enumerate(env_ids.tolist()):
                point_to_hand_dist = torch.cdist(
                    object_pcd_rel[env_id],
                    hand_pcd_rel[env_id],
                )
                keep_mask = point_to_hand_dist.min(dim=1).values > exclusion_radius
                filtered_object_pcd = object_pcd_world[env_id][keep_mask]
                if filtered_object_pcd.shape[0] == 0:
                    raise ValueError(
                        "Evaluation reset removed all target-object PCD points after "
                        f"filtering points within {exclusion_radius:.3f} m of the hand "
                        f"for env {env_id}."
                    )
                object_pcd_center = filtered_object_pcd.mean(dim=0)
                target_wrist_pos[row] = object_pcd_center + demo_wrist_offset[row]
        else:
            raise ValueError(
                f"Evaluation reset expected backend 'sim' or 'real', got "
                f"{self.env.cfg.backend!r}."
            )

        self.env.action_manager.move_to_reset_target(
            env_ids=env_ids,
            target_wrist_pos=target_wrist_pos,
            target_wrist_quat=target_wrist_quat,
            target_hand_qpos=target_hand_qpos,
        )
        print("Robot moved to eval reset wrist pose.")
