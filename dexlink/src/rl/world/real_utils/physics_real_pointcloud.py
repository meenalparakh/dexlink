"""Camera, calibration, and point-cloud helpers for the real backend."""

from __future__ import annotations

import time
from math import radians
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from dexlink.utils.retargeting.path_utils import get_src
from dexlink.src.rl.world.real_utils import (
    detect_marker_poses,
    filter_world_points_x_min,
    full_point_cloud_with_rgb,
    get_aruco_dictionary,
    horizontal_plane_from_z,
    keep_largest_cluster,
    remove_points_near_reference_pcd,
    transform_points_homogeneous,
)
from dexlink.src.rl.world.real_utils.physics_real_mixin_base import PhysicsRealMixinBase
from dexlink.src.rl.world.real_utils.vision import StaticSceneCalibration
from dexlink.src.rl.world.real_utils.world_calibration_solver import (
    WorldCalibrationObservation,
    WorldCalibrationSolver,
)
from dexlink.src.rl.world.robot import Hand


class PhysicsRealPointCloudMixin(PhysicsRealMixinBase):
    def _sample_synthetic_table_points_around_object(
        self, *, object_pcd_world: np.ndarray
    ) -> np.ndarray:
        table_sample_n = int(self.real_cfg.table_plane_sample_points)
        if table_sample_n < 0:
            raise ValueError(
                "Expected real_cfg.table_plane_sample_points >= 0, got "
                f"{table_sample_n}."
            )
        if table_sample_n == 0:
            return np.empty((0, 3), dtype=np.float32)
        object_pcd_world = np.asarray(object_pcd_world, dtype=np.float32)
        if object_pcd_world.ndim != 2 or object_pcd_world.shape[1] != 3:
            raise ValueError(
                "Expected object_pcd_world shape (num_points, 3), got "
                f"{object_pcd_world.shape}."
            )
        if object_pcd_world.shape[0] == 0:
            raise ValueError(
                "Cannot synthesize policy table points because the isolated real "
                "object PCD is empty."
            )
        half_extent_m = float(self.real_cfg.table_plane_synthetic_half_extent_m)
        if half_extent_m <= 0.0:
            raise ValueError(
                "Expected real_cfg.table_plane_synthetic_half_extent_m > 0, got "
                f"{half_extent_m}."
            )
        depth_m = float(self.real_cfg.table_plane_synthetic_depth_m)
        if depth_m < 0.0:
            raise ValueError(
                "Expected real_cfg.table_plane_synthetic_depth_m >= 0, got "
                f"{depth_m}."
            )
        object_xy_center = object_pcd_world[:, :2].mean(axis=0)
        object_min_z = float(object_pcd_world[:, 2].min())
        rng = np.random.default_rng()
        xy_low = object_xy_center - half_extent_m
        xy_high = object_xy_center + half_extent_m
        xy = rng.uniform(
            low=xy_low,
            high=xy_high,
            size=(table_sample_n, 2),
        )
        z_low = object_min_z - depth_m
        z_high = object_min_z
        if depth_m == 0.0:
            z = np.full((table_sample_n, 1), object_min_z, dtype=np.float32)
        else:
            z = rng.uniform(
                low=z_low,
                high=z_high,
                size=(table_sample_n, 1),
            )
        return np.concatenate([xy, z], axis=1).astype(np.float32)

    def _current_wrist_pose_world(self) -> np.ndarray:
        hand = self._active_real_hand()
        wrist_pose_world = self._entity_link_pose_by_name_from_snapshot(hand.robot).get(
            hand.wrist_link_name
        )
        if wrist_pose_world is None:
            raise KeyError(
                f"Missing buffered FK pose for wrist link {hand.wrist_link_name!r}."
            )
        wrist_pose_world = np.asarray(wrist_pose_world, dtype=np.float32)
        if wrist_pose_world.shape != (4, 4):
            raise ValueError(
                f"Expected wrist_pose_world shape (4, 4), got {wrist_pose_world.shape}."
            )
        return wrist_pose_world.astype(np.float32)

    def _current_pre_lift_phase_mask(self) -> torch.Tensor:
        pre_lift_phase = self.env.reward_manager.current_pre_lift_phase_mask()
        expected_shape = (self.env.num_envs,)
        if tuple(pre_lift_phase.shape) != expected_shape:
            raise ValueError(
                "Expected pre-lift phase mask shape "
                f"{expected_shape}, got {tuple(pre_lift_phase.shape)}."
            )
        return pre_lift_phase

    def _current_before_cached_object_pcd_motion_phase_mask(self) -> torch.Tensor:
        phase_name = str(self.real_cfg.phase_cached_object_pcd_motion_start_phase)
        if len(phase_name) == 0:
            raise ValueError(
                "Expected real_cfg.phase_cached_object_pcd_motion_start_phase "
                "to be non-empty."
            )
        task_manager = self.env.task_manager
        phase_names = list(getattr(task_manager, "phase_names", []))
        phase_index_by_timestep = getattr(task_manager, "phase_index_by_timestep", None)
        if len(phase_names) == 0 or phase_index_by_timestep is None:
            if phase_name.lower() == "lift":
                return self._current_pre_lift_phase_mask()
            raise ValueError(
                "Phase-cached real object PCD requires task_manager.phase_names "
                "and phase_index_by_timestep when start phase is not 'lift'."
            )
        matching_phase_indices = [
            phase_idx
            for phase_idx, candidate_name in enumerate(phase_names)
            if phase_name.lower() in str(candidate_name).lower()
        ]
        if len(matching_phase_indices) == 0:
            raise ValueError(
                "Could not find real object PCD motion start phase "
                f"{phase_name!r} in phase_names={phase_names}."
            )
        phase_index_by_timestep = torch.as_tensor(
            phase_index_by_timestep, dtype=torch.long, device=self.env.device
        )
        matching_timestep_mask = torch.zeros_like(
            phase_index_by_timestep, dtype=torch.bool
        )
        for phase_idx in matching_phase_indices:
            matching_timestep_mask |= phase_index_by_timestep == int(phase_idx)
        matching_timesteps = torch.nonzero(
            matching_timestep_mask, as_tuple=False
        ).reshape(-1)
        if matching_timesteps.numel() == 0:
            raise ValueError(
                "Matched real object PCD motion start phase "
                f"{phase_name!r}, but it has no timesteps."
            )
        motion_start_timestep = int(torch.min(matching_timesteps).item())
        current_timesteps = self.env.command_time_step.to(
            dtype=torch.long, device=self.env.device
        )
        expected_shape = (self.env.num_envs,)
        if tuple(current_timesteps.shape) != expected_shape:
            raise ValueError(
                "Expected command_time_step shape "
                f"{expected_shape}, got {tuple(current_timesteps.shape)}."
            )
        return current_timesteps < motion_start_timestep

    def _sample_object_pcd(
        self,
        *,
        object_pcd_world: np.ndarray,
        normals: bool,
    ) -> torch.Tensor:
        object_pcd_world = np.asarray(object_pcd_world, dtype=np.float32)
        if object_pcd_world.ndim != 2 or object_pcd_world.shape[1] != 3:
            raise ValueError(
                "Expected object_pcd_world shape (num_points, 3), got "
                f"{object_pcd_world.shape}."
            )
        target_n_pts = int(self.env.obs_cfg.policy_object_pcd_n_pts)
        feat_dim = 6 if normals else 3
        if object_pcd_world.shape[0] == 0:
            return torch.zeros(
                (1, target_n_pts, feat_dim),
                dtype=torch.float32,
                device=self.env.device,
            )
        if object_pcd_world.shape[0] >= target_n_pts:
            sample_idx = np.linspace(
                0, object_pcd_world.shape[0] - 1, target_n_pts, dtype=np.int64
            )
            sampled_xyz = object_pcd_world[sample_idx]
        else:
            pad_count = target_n_pts - object_pcd_world.shape[0]
            pad_xyz = np.repeat(object_pcd_world[-1:, :], pad_count, axis=0)
            sampled_xyz = np.concatenate([object_pcd_world, pad_xyz], axis=0)

        if normals:
            sampled = np.concatenate(
                [sampled_xyz, np.zeros_like(sampled_xyz, dtype=np.float32)],
                axis=1,
            )
        else:
            sampled = sampled_xyz
        return torch.tensor(
            sampled[None, ...],
            dtype=torch.float32,
            device=self.env.device,
        )

    def _get_scene_and_filtered_object_pcd_world(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._world_calibration is None:
            raise RuntimeError(
                "World calibration is not available. Enable cached or live world-frame calibration in real_cfg."
            )

        frame = self._latest_camera_frame(force_capture=False)
        calibration = self._camera_calibration()
        object_pcd_cam, _ = full_point_cloud_with_rgb(
            depth_image=np.asarray(frame["depth"], dtype=np.float32),
            color_image_rgb=np.asarray(frame["color"]),
            min_depth_m=float(self.real_cfg.min_depth_m),
            max_depth_m=float(self.real_cfg.max_depth_m),
            intrinsics=np.asarray(
                calibration["aligned_depth_intrinsics"], dtype=np.float32
            ),
            max_points=int(self.real_cfg.object_pcd_max_points),
        )
        raw_pcd_world = transform_points_homogeneous(
            object_pcd_cam,
            self._world_calibration.world_from_cam,
        )
        raw_pcd_world = filter_world_points_x_min(
            raw_pcd_world,
            x_min=float(self.real_cfg.world_x_min),
        )
        raw_pcd_world = self._filter_world_points_z_min(
            raw_pcd_world,
            z_min=float(self.real_cfg.world_z_min),
        )
        table_plane_world = np.asarray(
            self._world_calibration.table_plane_world, dtype=np.float32
        )
        if table_plane_world.shape != (4,):
            raise ValueError(
                "Expected table_plane_world shape (4,), got "
                f"{table_plane_world.shape}."
            )
        table_signed_distance = (
            raw_pcd_world @ table_plane_world[:3] + table_plane_world[3]
        )
        scene_pcd_world = raw_pcd_world[
            table_signed_distance >= float(self.real_cfg.table_plane_height_margin_m)
        ].astype(np.float32)
        object_pcd_world = scene_pcd_world.copy()

        if self.real_cfg.initialize_robot:
            robot_pcd_world = self._current_robot_surface_pcd_world()
            scene_pcd_world = remove_points_near_reference_pcd(
                points=scene_pcd_world,
                reference_points=robot_pcd_world,
                exclusion_radius_m=float(self.real_cfg.robot_pcd_exclusion_radius_m),
            )
            object_pcd_world = remove_points_near_reference_pcd(
                points=object_pcd_world,
                reference_points=robot_pcd_world,
                exclusion_radius_m=float(self.real_cfg.robot_pcd_exclusion_radius_m),
            )
        object_pcd_world = keep_largest_cluster(
            points=object_pcd_world,
            eps_m=float(self.real_cfg.largest_cluster_eps_m),
            min_points=int(self.real_cfg.largest_cluster_min_points),
        )
        sampled_table_pcd_world = self._sample_synthetic_table_points_around_object(
            object_pcd_world=object_pcd_world,
        )
        if sampled_table_pcd_world.shape[0] > 0:
            scene_pcd_world = np.concatenate(
                [scene_pcd_world, sampled_table_pcd_world],
                axis=0,
            ).astype(np.float32)
        return (
            scene_pcd_world.astype(np.float32),
            object_pcd_world.astype(np.float32),
        )

    def _current_robot_surface_pcd_world(self) -> np.ndarray:
        hand = self._active_real_hand()
        current_jpos = self.get_hand_joint_positions(hand)
        robot_link_names = list(hand.hand_tree_link_names)
        if len(robot_link_names) == 0:
            raise ValueError(
                "Failed to find robot surface links for robot-point-cloud filtering."
            )
        robot_pcd = hand.get_robot_pcd(
            current_jpos,
            links_to_include=robot_link_names,
            use_dense_points=True,
        )
        return robot_pcd[0].detach().cpu().numpy().astype(np.float32)

    def _filter_world_points_z_min(
        self, points_world: np.ndarray, *, z_min: float
    ) -> np.ndarray:
        points_world = np.asarray(points_world, dtype=np.float32)
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError(
                "Expected points_world shape (num_points, 3), got "
                f"{points_world.shape}."
            )
        if points_world.shape[0] == 0:
            return points_world
        return points_world[points_world[:, 2] >= float(z_min)].astype(np.float32)

    def _base_pose_cam_errors(
        self,
        *,
        reference_pose_cam: np.ndarray,
        query_pose_cam: np.ndarray,
    ) -> tuple[float, float]:
        reference_pose_cam = np.asarray(reference_pose_cam, dtype=np.float32)
        query_pose_cam = np.asarray(query_pose_cam, dtype=np.float32)
        if reference_pose_cam.shape != (4, 4):
            raise ValueError(
                "Expected reference_pose_cam shape (4, 4), got "
                f"{reference_pose_cam.shape}."
            )
        if query_pose_cam.shape != (4, 4):
            raise ValueError(
                f"Expected query_pose_cam shape (4, 4), got {query_pose_cam.shape}."
            )
        pos_error_m = float(
            np.linalg.norm(reference_pose_cam[:3, 3] - query_pose_cam[:3, 3])
        )
        rel_rot = R.from_matrix(reference_pose_cam[:3, :3]).inv() * R.from_matrix(
            query_pose_cam[:3, :3]
        )
        rot_error_deg = float(np.rad2deg(rel_rot.magnitude()))
        return pos_error_m, rot_error_deg

    def _confirm_calibration_sample(self, *, frame_idx: int, sample_idx: int) -> bool:
        while True:
            response = (
                input(
                    "[WorldCalibration] "
                    f"Keep calibration sample {sample_idx:03d} from frame_idx={frame_idx}? [y/n]: "
                )
                .strip()
                .lower()
            )
            if response in {"y", "yes"}:
                return True
            if response in {"n", "no"}:
                return False
            print("Please answer 'y' or 'n'.")

    def _should_keep_calibration_sample(
        self,
        *,
        frame_idx: int,
        sample_idx: int,
        base_pose_cam: np.ndarray,
        accepted_base_pose_cam: list[np.ndarray],
    ) -> bool:
        manual_review_count = int(self.real_cfg.world_calibration_num_manual_reviews)
        if manual_review_count < 0:
            raise ValueError(
                "Expected world_calibration_num_manual_reviews >= 0, got "
                f"{manual_review_count}."
            )
        if len(accepted_base_pose_cam) < manual_review_count:
            return self._confirm_calibration_sample(
                frame_idx=frame_idx, sample_idx=sample_idx
            )
        pos_margin_m = float(self.real_cfg.world_calibration_auto_accept_pos_margin_m)
        rot_margin_deg = float(
            self.real_cfg.world_calibration_auto_accept_rot_margin_deg
        )
        if pos_margin_m <= 0.0:
            raise ValueError(
                "Expected world_calibration_auto_accept_pos_margin_m > 0, got "
                f"{pos_margin_m}."
            )
        if rot_margin_deg <= 0.0:
            raise ValueError(
                "Expected world_calibration_auto_accept_rot_margin_deg > 0, got "
                f"{rot_margin_deg}."
            )
        pos_errors_m = []
        rot_errors_deg = []
        for accepted_pose_cam in accepted_base_pose_cam:
            pos_error_m, rot_error_deg = self._base_pose_cam_errors(
                reference_pose_cam=accepted_pose_cam,
                query_pose_cam=base_pose_cam,
            )
            pos_errors_m.append(pos_error_m)
            rot_errors_deg.append(rot_error_deg)
        min_pos_error_m = min(pos_errors_m)
        min_rot_error_deg = min(rot_errors_deg)
        keep = min_pos_error_m <= pos_margin_m and min_rot_error_deg <= rot_margin_deg
        decision = "Keeping" if keep else "Discarding"
        print(
            "[WorldCalibration] "
            f"{decision} calibration sample {sample_idx:03d} from frame_idx={frame_idx} "
            f"using auto-filter: min_pos_error_m={min_pos_error_m:.6f}, "
            f"min_rot_error_deg={min_rot_error_deg:.6f}, "
            f"thresholds=({pos_margin_m:.6f} m, {rot_margin_deg:.6f} deg)."
        )
        return keep

    def _camera_calibration(self) -> dict[str, np.ndarray]:
        robot_entity = self._active_real_robot_entity()
        if robot_entity.camera_calibration is None:
            raise RuntimeError("Real camera calibration is not initialized.")
        return robot_entity.camera_calibration

    def _latest_camera_frame(self, *, force_capture: bool) -> dict[str, np.ndarray]:
        robot_entity = self._active_real_robot_entity()
        return robot_entity.latest_camera_frame(force_capture=force_capture)

    def _world_calibration_cache_path(self) -> Path:
        cache_path = Path(self.real_cfg.world_calibration_cache_path)
        if cache_path.is_absolute():
            return cache_path
        return get_src() / "camera_calib" / cache_path

    def _world_calibration_artifacts_dir(self) -> Path:
        return self._world_calibration_cache_path().parent

    def _marker_sizes_m(self) -> dict[int, float]:
        return {
            int(self.real_cfg.base_marker_id): float(self.real_cfg.base_marker_size_m),
            int(self.real_cfg.hand_outer_palm_marker_id): float(
                self.real_cfg.hand_outer_palm_marker_size_m
            ),
            int(self.real_cfg.hand_palm_side_marker_id): float(
                self.real_cfg.hand_palm_side_marker_size_m
            ),
            int(self.real_cfg.thumb_marker_id): float(
                self.real_cfg.thumb_marker_size_m
            ),
        }

    def _wrist_marker_pose_wrist_frame(self) -> np.ndarray:
        pos = np.asarray(self.real_cfg.wrist_to_marker_pos, dtype=np.float32)
        rotmat = np.asarray(self.real_cfg.wrist_to_marker_rotmat, dtype=np.float32)
        if pos.shape != (3,):
            raise ValueError(
                f"Expected wrist_to_marker_pos shape (3,), got {pos.shape}."
            )
        if rotmat.shape != (3, 3):
            raise ValueError(
                f"Expected wrist_to_marker_rotmat shape (3, 3), got {rotmat.shape}."
            )
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rotmat
        pose[:3, 3] = pos
        return pose

    def _current_wrist_marker_pose_wrist(self) -> np.ndarray:
        if (
            self._world_calibration is not None
            and self._world_calibration.wrist_marker_pose_wrist is not None
        ):
            return np.asarray(
                self._world_calibration.wrist_marker_pose_wrist, dtype=np.float32
            )
        return self._wrist_marker_pose_wrist_frame()

    def _world_calibration_robot_frame_indices(self) -> np.ndarray:
        hand = self._active_real_hand()
        motion = getattr(self.env.task_manager.original_demo, "retargeted_motion", None)
        if not isinstance(motion, dict):
            raise ValueError(
                "World calibration requires original_demo.retargeted_motion to be available."
            )
        jvals_key = f"{hand.name}_jvals"
        if jvals_key not in motion:
            raise KeyError(
                f"World calibration expected retargeted motion key {jvals_key!r}."
            )
        total_frames = int(np.asarray(motion[jvals_key]).shape[0])
        if total_frames <= 0:
            raise ValueError("Retargeted motion contains no frames for calibration.")
        start_idx = int(self.env.cfg.reset_cfg.eval_start_timestep)
        stride = int(self.real_cfg.world_calibration_frame_stride)
        count = int(self.real_cfg.world_calibration_num_robot_configs)
        if stride <= 0:
            raise ValueError(
                f"Expected world_calibration_frame_stride > 0, got {stride}."
            )
        if count <= 0:
            raise ValueError(
                f"Expected world_calibration_num_robot_configs > 0, got {count}."
            )
        frame_indices = start_idx + stride * np.arange(count, dtype=np.int64)
        frame_indices = frame_indices[frame_indices < total_frames]
        if frame_indices.shape[0] == 0:
            raise ValueError(
                "World calibration frame selection produced no valid frames. "
                f"start_idx={start_idx}, stride={stride}, total_frames={total_frames}."
            )
        return frame_indices

    def _mapped_hand_joint_positions_from_motion(
        self, hand: Hand, frame_idx: int
    ) -> torch.Tensor:
        motion = getattr(self.env.task_manager.original_demo, "retargeted_motion", None)
        if not isinstance(motion, dict):
            raise ValueError(
                "World calibration requires original_demo.retargeted_motion to be available."
            )
        jvals_key = f"{hand.name}_jvals"
        jnames_key = f"{hand.name}_jnames"
        if jvals_key not in motion or jnames_key not in motion:
            raise KeyError(
                f"World calibration expected retargeted motion keys {jvals_key!r} and {jnames_key!r}."
            )
        source_joint_names = [str(name) for name in motion[jnames_key]]
        joint_name_to_col = {
            joint_name: joint_col
            for joint_col, joint_name in enumerate(source_joint_names)
        }
        source_jvals = np.asarray(motion[jvals_key], dtype=np.float32)
        if frame_idx < 0 or frame_idx >= source_jvals.shape[0]:
            raise IndexError(
                f"Frame index {frame_idx} is out of range for motion with {source_jvals.shape[0]} frames."
            )
        mapped = np.zeros((1, len(hand.joints)), dtype=np.float32)
        for joint_col, joint_name in enumerate(hand.joint_names):
            source_col = joint_name_to_col.get(joint_name)
            if source_col is None:
                raise KeyError(
                    f"Joint {joint_name!r} is missing from retargeted motion. "
                    f"Available joints: {source_joint_names}."
                )
            mapped[0, joint_col] = source_jvals[frame_idx, source_col]
        return torch.from_numpy(mapped).to(self.env.device)

    def _local_wrist_pose_base_from_joint_target(
        self, hand: Hand, target_jpos: torch.Tensor
    ) -> np.ndarray:
        target_jpos = torch.as_tensor(
            target_jpos, dtype=torch.float32, device=self.env.device
        )
        expected_shape = (1, len(hand.joints))
        if tuple(target_jpos.shape) != expected_shape:
            raise ValueError(
                f"Expected target_jpos shape {expected_shape}, got {tuple(target_jpos.shape)}."
            )
        joint_cfg = {
            joint.name: float(target_jpos[0, joint_idx].item())
            for joint_idx, joint in enumerate(hand.joints)
        }
        link_fk = hand.urdf_inst.link_fk(cfg=joint_cfg)
        wrist_link = hand.urdf_inst.link_map.get(hand.wrist_link_name)
        if wrist_link is None:
            raise KeyError(
                f"Wrist link {hand.wrist_link_name!r} is missing from the URDF."
            )
        wrist_pose_base = link_fk.get(wrist_link)
        if wrist_pose_base is None:
            raise KeyError(
                f"FK did not return a pose for wrist link {hand.wrist_link_name!r}."
            )
        return np.asarray(wrist_pose_base, dtype=np.float32)

    def _project_points_to_image(
        self,
        points_cam: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points_cam = np.asarray(points_cam, dtype=np.float32)
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if points_cam.ndim != 2 or points_cam.shape[1] != 3:
            raise ValueError(
                f"Expected points_cam shape (N, 3), got {points_cam.shape}."
            )
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Expected intrinsics shape (3, 3), got {intrinsics.shape}."
            )
        positive_depth = points_cam[:, 2] > 1e-6
        pixels = np.zeros((points_cam.shape[0], 2), dtype=np.float32)
        pixels[positive_depth, 0] = (
            intrinsics[0, 0]
            * points_cam[positive_depth, 0]
            / points_cam[positive_depth, 2]
            + intrinsics[0, 2]
        )
        pixels[positive_depth, 1] = (
            intrinsics[1, 1]
            * points_cam[positive_depth, 1]
            / points_cam[positive_depth, 2]
            + intrinsics[1, 2]
        )
        return pixels, positive_depth

    def _score_candidate_marker_visibility(
        self,
        *,
        wrist_pose_base: np.ndarray,
        base_pose_cam_guess: np.ndarray,
        wrist_marker_pose_wrist: np.ndarray,
        color_intrinsics: np.ndarray,
        image_shape: tuple[int, int, int],
        marker_size_m: float,
    ) -> float:
        marker_pose_cam = (
            np.asarray(base_pose_cam_guess, dtype=np.float32)
            @ np.asarray(wrist_pose_base, dtype=np.float32)
            @ np.asarray(wrist_marker_pose_wrist, dtype=np.float32)
        )
        half_size = float(marker_size_m) / 2.0
        marker_corners = np.array(
            [
                [-half_size, half_size, 0.0, 1.0],
                [half_size, half_size, 0.0, 1.0],
                [half_size, -half_size, 0.0, 1.0],
                [-half_size, -half_size, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        corners_cam = (marker_pose_cam @ marker_corners.T).T[:, :3]
        pixels, positive_depth = self._project_points_to_image(
            corners_cam, color_intrinsics
        )
        if not np.all(positive_depth):
            return -np.inf
        image_h, image_w = image_shape[:2]
        margin = float(self.real_cfg.world_calibration_visibility_margin_px)
        if np.any(pixels[:, 0] < margin) or np.any(pixels[:, 0] > image_w - margin):
            return -np.inf
        if np.any(pixels[:, 1] < margin) or np.any(pixels[:, 1] > image_h - margin):
            return -np.inf
        bbox_min = np.min(pixels, axis=0)
        bbox_max = np.max(pixels, axis=0)
        area = float(np.prod(np.maximum(bbox_max - bbox_min, 0.0)))
        center = np.mean(pixels, axis=0)
        image_center = np.array([image_w / 2.0, image_h / 2.0], dtype=np.float32)
        center_penalty = float(np.linalg.norm(center - image_center))
        return area - center_penalty

    def _overlay_robot_points_on_image(
        self,
        *,
        color_image_rgb: np.ndarray,
        robot_points_cam: np.ndarray,
        intrinsics: np.ndarray,
        output_path: Path,
        overlay_rgb: tuple[int, int, int] = (255, 100, 0),
    ) -> None:
        image_rgb = np.asarray(color_image_rgb, dtype=np.uint8).copy()
        pixels, positive_depth = self._project_points_to_image(
            robot_points_cam, intrinsics
        )
        image_h, image_w = image_rgb.shape[:2]
        radius = int(self.real_cfg.world_calibration_overlay_point_size_px)
        for pixel, is_positive in zip(pixels, positive_depth.tolist()):
            if not is_positive:
                continue
            col = int(round(float(pixel[0])))
            row = int(round(float(pixel[1])))
            if row < 0 or row >= image_h or col < 0 or col >= image_w:
                continue
            cv2.circle(image_rgb, (col, row), radius, overlay_rgb, -1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    def _overlay_robot_mesh_on_image_pyrender(
        self,
        *,
        hand: Hand,
        measured_jpos: torch.Tensor,
        color_image_rgb: np.ndarray,
        color_intrinsics: np.ndarray,
        camera_from_world: np.ndarray,
        output_path: Path,
    ) -> None:
        try:
            import pyrender
        except ImportError as exc:
            raise ImportError(
                "pyrender is required for the calibration mesh overlay."
            ) from exc
        measured_jpos = torch.as_tensor(
            measured_jpos, dtype=torch.float32, device=self.env.device
        )
        expected_shape = (1, len(hand.joints))
        if tuple(measured_jpos.shape) != expected_shape:
            raise ValueError(
                f"Expected measured_jpos shape {expected_shape}, got {tuple(measured_jpos.shape)}."
            )
        color_image_rgb = np.asarray(color_image_rgb, dtype=np.uint8)
        color_intrinsics = np.asarray(color_intrinsics, dtype=np.float32)
        camera_from_world = np.asarray(camera_from_world, dtype=np.float32)
        if color_image_rgb.ndim != 3 or color_image_rgb.shape[2] != 3:
            raise ValueError(
                f"Expected color_image_rgb shape (H, W, 3), got {color_image_rgb.shape}."
            )
        if color_intrinsics.shape != (3, 3):
            raise ValueError(
                f"Expected color_intrinsics shape (3, 3), got {color_intrinsics.shape}."
            )
        if camera_from_world.shape != (4, 4):
            raise ValueError(
                f"Expected camera_from_world shape (4, 4), got {camera_from_world.shape}."
            )
        joint_cfg = {
            joint.name: float(measured_jpos[0, joint_idx].item())
            for joint_idx, joint in enumerate(hand.joints)
        }
        link_fk = hand.urdf_inst.link_fk(cfg=joint_cfg)
        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ambient_light=np.array([0.25, 0.25, 0.25], dtype=np.float32),
        )
        height, width = color_image_rgb.shape[:2]
        camera = pyrender.IntrinsicsCamera(
            fx=float(color_intrinsics[0, 0]),
            fy=float(color_intrinsics[1, 1]),
            cx=float(color_intrinsics[0, 2]),
            cy=float(color_intrinsics[1, 2]),
        )
        scene.add(camera, pose=np.eye(4, dtype=np.float32))
        scene.add(
            pyrender.DirectionalLight(
                color=np.ones(3, dtype=np.float32), intensity=2.5
            ),
            pose=np.eye(4, dtype=np.float32),
        )
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.0, 1.0, 1.0, 0.85),
            metallicFactor=0.0,
            roughnessFactor=0.8,
            alphaMode="BLEND",
        )
        for link in hand.urdf_inst.links:
            link_pose = link_fk.get(link)
            if link_pose is None:
                raise KeyError(f"FK did not return a pose for link {link.name!r}.")
            world_link_pose = np.asarray(hand.base_pose, dtype=np.float32) @ np.asarray(
                link_pose, dtype=np.float32
            )
            camera_link_pose = camera_from_world @ world_link_pose
            for visual in getattr(link, "visuals", None) or []:
                geom = getattr(visual, "geometry", None)
                if geom is None:
                    continue
                mesh_field = getattr(geom, "mesh", None)
                if mesh_field is None:
                    continue
                visual_pose = (
                    np.asarray(visual.origin, dtype=np.float32)
                    if visual.origin is not None
                    else np.eye(4, dtype=np.float32)
                )
                scale = getattr(mesh_field, "scale", None)
                for mesh in getattr(geom, "meshes", None) or []:
                    mesh_copy = mesh.copy()
                    if scale is not None:
                        scale_transform = np.eye(4, dtype=np.float32)
                        scale_transform[:3, :3] = np.diag(
                            np.asarray(scale, dtype=np.float32)
                        )
                        mesh_copy.apply_transform(scale_transform)
                    mesh_copy.apply_transform(visual_pose)
                    scene.add(
                        pyrender.Mesh.from_trimesh(
                            mesh_copy, material=material, smooth=False
                        ),
                        pose=camera_link_pose,
                    )
        renderer = pyrender.OffscreenRenderer(
            viewport_width=width, viewport_height=height
        )
        try:
            color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            renderer.delete()
        color_rgba = np.asarray(color_rgba, dtype=np.uint8)
        alpha_mask = color_rgba[..., 3] > 0
        overlay_rgb = color_image_rgb.copy()
        overlay_rgb[alpha_mask] = color_rgba[..., :3][alpha_mask]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    def _save_world_calibration_overlay(
        self,
        *,
        hand: Hand,
        measured_jpos: torch.Tensor,
        color_image_rgb: np.ndarray,
        color_intrinsics: np.ndarray,
        world_from_cam: np.ndarray,
        frame_idx: int,
        sample_idx: int,
    ) -> None:
        robot_pcd_world = (
            hand.get_robot_pcd(measured_jpos, use_dense_points=True)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        camera_from_world = np.linalg.inv(np.asarray(world_from_cam, dtype=np.float32))
        robot_pcd_cam = transform_points_homogeneous(robot_pcd_world, camera_from_world)
        overlay_dir = (
            self._world_calibration_artifacts_dir() / "world_calibration_overlays"
        )
        self._overlay_robot_points_on_image(
            color_image_rgb=color_image_rgb,
            robot_points_cam=robot_pcd_cam,
            intrinsics=color_intrinsics,
            output_path=overlay_dir
            / f"sample_{sample_idx:03d}_frame_{frame_idx:05d}.png",
        )
        self._overlay_robot_mesh_on_image_pyrender(
            hand=hand,
            measured_jpos=measured_jpos,
            color_image_rgb=color_image_rgb,
            color_intrinsics=color_intrinsics,
            camera_from_world=camera_from_world,
            output_path=overlay_dir
            / f"sample_{sample_idx:03d}_frame_{frame_idx:05d}_mesh.png",
        )

    def _candidate_satisfies_robot_min_z(
        self,
        *,
        hand: Hand,
        candidate_jpos: torch.Tensor,
    ) -> bool:
        links_to_include = [
            link_name
            for link_name in hand.hand_surface_link_names
            if link_name not in {"panda_link0", "panda_link1"}
        ]
        if len(links_to_include) == 0:
            raise ValueError(
                "Calibration z-constraint expected at least one robot link after excluding "
                "'panda_link0' and 'panda_link1'."
            )
        robot_pcd_world = (
            hand.get_robot_pcd(
                candidate_jpos,
                links_to_include=links_to_include,
                use_dense_points=True,
            )[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if robot_pcd_world.ndim != 2 or robot_pcd_world.shape[1] != 3:
            raise ValueError(
                f"Expected dense robot point cloud shape (N, 3), got {robot_pcd_world.shape}."
            )
        min_robot_z = float(np.min(robot_pcd_world[:, 2]))
        min_allowed_z = float(hand.base_pose[2, 3]) + float(
            self.real_cfg.world_calibration_robot_min_z_offset_m
        )
        return min_robot_z >= min_allowed_z

    def _measured_hand_joint_positions(self, hand: Hand) -> torch.Tensor:
        measured_jpos = self.get_entity_dofs_position(hand.robot, hand.dof_map)
        expected_shape = (1, len(hand.dof_map))
        if tuple(measured_jpos.shape) != expected_shape:
            raise ValueError(
                f"Expected measured_jpos shape {expected_shape}, got {tuple(measured_jpos.shape)}."
            )
        return measured_jpos

    def _candidate_is_far_enough_from_existing(
        self,
        *,
        hand: Hand,
        candidate_jpos: torch.Tensor,
        existing_candidates: list[torch.Tensor],
        panda_joint_indices: np.ndarray,
    ) -> bool:
        candidate_jpos = torch.as_tensor(
            candidate_jpos, dtype=torch.float32, device=self.env.device
        )
        expected_shape = (1, len(hand.joints))
        if tuple(candidate_jpos.shape) != expected_shape:
            raise ValueError(
                f"Expected candidate_jpos shape {expected_shape}, got {tuple(candidate_jpos.shape)}."
            )
        panda_joint_indices = np.asarray(panda_joint_indices, dtype=np.int64)
        if panda_joint_indices.ndim != 1 or panda_joint_indices.shape[0] == 0:
            raise ValueError(
                "Expected panda_joint_indices to be a non-empty rank-1 array, got "
                f"{panda_joint_indices.shape}."
            )
        min_joint_l1_deg = float(
            self.real_cfg.world_calibration_candidate_min_joint_l1_deg
        )
        if min_joint_l1_deg < 0.0:
            raise ValueError(
                "Expected world_calibration_candidate_min_joint_l1_deg >= 0, got "
                f"{min_joint_l1_deg}."
            )
        min_joint_l1_rad = radians(min_joint_l1_deg)
        candidate_panda_q = candidate_jpos[:, panda_joint_indices]
        for existing_candidate in existing_candidates:
            existing_candidate = torch.as_tensor(
                existing_candidate, dtype=torch.float32, device=self.env.device
            )
            if tuple(existing_candidate.shape) != tuple(candidate_jpos.shape):
                raise ValueError(
                    "Expected existing candidate shape to match candidate_jpos, got "
                    f"{tuple(existing_candidate.shape)} vs {tuple(candidate_jpos.shape)}."
                )
            existing_panda_q = existing_candidate[:, panda_joint_indices]
            joint_l1_rad = torch.sum(
                torch.abs(candidate_panda_q - existing_panda_q)
            ).item()
            if joint_l1_rad < min_joint_l1_rad:
                return False
        return True

    def _world_calibration_joint_candidates(
        self,
        hand: Hand,
        target_jpos: torch.Tensor,
        *,
        frame_idx: int,
        nearby_seed_jpos: torch.Tensor | None = None,
        base_pose_cam_guess: np.ndarray | None = None,
        wrist_marker_pose_wrist: np.ndarray | None = None,
        color_intrinsics: np.ndarray | None = None,
        image_shape: tuple[int, int, int] | None = None,
        marker_size_m: float | None = None,
    ) -> list[torch.Tensor]:
        target_jpos = torch.as_tensor(
            target_jpos, dtype=torch.float32, device=self.env.device
        )
        expected_shape = (1, len(hand.joints))
        if tuple(target_jpos.shape) != expected_shape:
            raise ValueError(
                f"Expected target_jpos shape {expected_shape}, got {tuple(target_jpos.shape)}."
            )
        fallback_tries = int(self.real_cfg.world_calibration_panda_fallback_tries)
        fallback_std_rad = float(self.real_cfg.world_calibration_panda_fallback_std_rad)
        if fallback_tries < 0:
            raise ValueError(
                "Expected world_calibration_panda_fallback_tries >= 0, got "
                f"{fallback_tries}."
            )
        if fallback_std_rad < 0.0:
            raise ValueError(
                "Expected world_calibration_panda_fallback_std_rad >= 0, got "
                f"{fallback_std_rad}."
            )
        candidate_center = target_jpos.clone()
        if nearby_seed_jpos is not None:
            nearby_seed_jpos = torch.as_tensor(
                nearby_seed_jpos, dtype=torch.float32, device=self.env.device
            )
            if tuple(nearby_seed_jpos.shape) != expected_shape:
                raise ValueError(
                    "Expected nearby_seed_jpos shape "
                    f"{expected_shape}, got {tuple(nearby_seed_jpos.shape)}."
                )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.env.cfg.seed) + int(frame_idx))
        lower_limit, upper_limit = self.get_hand_dof_limits(hand, hand.dof_map)
        lower_limit = lower_limit.to(self.env.device)
        upper_limit = upper_limit.to(self.env.device)
        panda_joint_indices = np.asarray(hand.panda_joint_idx, dtype=np.int64)
        if panda_joint_indices.shape[0] == 0:
            raise ValueError(
                "Expected Hand to have Panda joints for calibration fallback."
            )
        if nearby_seed_jpos is not None:
            candidate_center[:, panda_joint_indices] = nearby_seed_jpos[
                :, panda_joint_indices
            ]
        desired_candidate_count = fallback_tries + 1
        candidates: list[torch.Tensor] = []
        if self._candidate_satisfies_robot_min_z(
            hand=hand,
            candidate_jpos=candidate_center,
        ):
            candidates.append(candidate_center.clone())
        sampling_attempts = 0
        max_sampling_attempts = max(
            desired_candidate_count * 20, desired_candidate_count
        )
        while (
            len(candidates) < desired_candidate_count
            and sampling_attempts < max_sampling_attempts
        ):
            sampling_attempts += 1
            candidate = candidate_center.clone()
            noise = torch.randn(
                (1, panda_joint_indices.shape[0]),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            ).to(self.env.device)
            candidate[:, panda_joint_indices] = (
                candidate[:, panda_joint_indices] + fallback_std_rad * noise
            )
            candidate = torch.clip(candidate, lower_limit, upper_limit)
            if self._candidate_satisfies_robot_min_z(
                hand=hand,
                candidate_jpos=candidate,
            ) and self._candidate_is_far_enough_from_existing(
                hand=hand,
                candidate_jpos=candidate,
                existing_candidates=candidates,
                panda_joint_indices=panda_joint_indices,
            ):
                candidates.append(candidate)
        if len(candidates) == 0:
            raise RuntimeError(
                "All world calibration candidates violate the minimum-z robot safety constraint."
            )
        if (
            base_pose_cam_guess is not None
            and wrist_marker_pose_wrist is not None
            and color_intrinsics is not None
            and image_shape is not None
            and marker_size_m is not None
        ):
            scored_candidates: list[tuple[float, torch.Tensor]] = []
            for candidate in candidates:
                wrist_pose_base = self._local_wrist_pose_base_from_joint_target(
                    hand, candidate
                )
                visibility_score = self._score_candidate_marker_visibility(
                    wrist_pose_base=wrist_pose_base,
                    base_pose_cam_guess=base_pose_cam_guess,
                    wrist_marker_pose_wrist=wrist_marker_pose_wrist,
                    color_intrinsics=color_intrinsics,
                    image_shape=image_shape,
                    marker_size_m=marker_size_m,
                )
                scored_candidates.append((visibility_score, candidate))
            scored_candidates.sort(key=lambda item: item[0], reverse=True)
            num_visible_candidates = int(
                self.real_cfg.world_calibration_num_visible_candidates
            )
            if num_visible_candidates <= 0:
                raise ValueError(
                    "Expected world_calibration_num_visible_candidates > 0, got "
                    f"{num_visible_candidates}."
                )
            visible_candidates = [
                candidate
                for score, candidate in scored_candidates
                if np.isfinite(score)
            ]
            if len(visible_candidates) > 0:
                return visible_candidates[:num_visible_candidates]
        return candidates

    def _capture_manual_world_calibration_sample(
        self,
        hand: Hand,
        *,
        sample_idx: int,
        frame_idx: int,
        color_intrinsics: np.ndarray,
        aligned_depth_intrinsics: np.ndarray,
        aruco_dict,
        wrist_marker_id: int,
        wrist_marker_size: float,
        wrist_marker_pose_wrist: np.ndarray,
        world_base_pose: np.ndarray,
    ) -> (
        tuple[
            WorldCalibrationObservation,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            torch.Tensor,
        ]
        | None
    ):
        input(
            "[WorldCalibration] "
            f"Move the robot to manual calibration pose {sample_idx + 1:03d}, "
            "make sure the wrist marker is visible in the camera, then press Enter to capture."
        )
        time.sleep(float(self.real_cfg.world_calibration_settle_time_s))
        self._reload_entity_state_interface(hand.robot, force_update=True)
        frame = self._latest_camera_frame(force_capture=True)
        poses_cam = detect_marker_poses(
            color_image_rgb=frame["color"],
            intrinsics=color_intrinsics,
            aruco_dict=aruco_dict,
            marker_sizes_m={wrist_marker_id: wrist_marker_size},
        )
        marker_pose_cam = poses_cam.get(wrist_marker_id)
        if marker_pose_cam is None:
            print(
                "[WorldCalibration] "
                f"Marker {wrist_marker_id} was not detected for manual sample {sample_idx:03d}. "
                "Discarding this capture and continuing."
            )
            return None
        local_link_poses = hand.robot.latest_local_link_pose_by_name(force_update=True)
        wrist_pose_base = local_link_poses.get(hand.wrist_link_name)
        if wrist_pose_base is None:
            raise KeyError(
                f"Missing local FK pose for wrist link {hand.wrist_link_name!r}."
            )
        base_pose_cam = (
            marker_pose_cam
            @ np.linalg.inv(wrist_marker_pose_wrist)
            @ np.linalg.inv(wrist_pose_base)
        ).astype(np.float32)
        wrist_pose_cam = (
            base_pose_cam @ np.asarray(wrist_pose_base, dtype=np.float32)
        ).astype(np.float32)
        world_from_cam = (world_base_pose @ np.linalg.inv(base_pose_cam)).astype(
            np.float32
        )
        scene_pcd_cam, scene_colors_cam = full_point_cloud_with_rgb(
            depth_image=np.asarray(frame["depth"], dtype=np.float32),
            color_image_rgb=np.asarray(frame["color"], dtype=np.uint8),
            min_depth_m=float(self.real_cfg.min_depth_m),
            max_depth_m=float(self.real_cfg.max_depth_m),
            intrinsics=aligned_depth_intrinsics,
            max_points=int(self.real_cfg.object_pcd_max_points),
        )
        measured_jpos = self._measured_hand_joint_positions(hand)
        self._save_world_calibration_overlay(
            hand=hand,
            measured_jpos=measured_jpos,
            color_image_rgb=np.asarray(frame["color"]),
            color_intrinsics=color_intrinsics,
            world_from_cam=world_from_cam,
            frame_idx=frame_idx,
            sample_idx=sample_idx,
        )
        return (
            WorldCalibrationObservation(
                frame_idx=frame_idx,
                wrist_pose_base=np.asarray(wrist_pose_base, dtype=np.float32),
                marker_pose_cam=np.asarray(marker_pose_cam, dtype=np.float32),
            ),
            scene_pcd_cam,
            scene_colors_cam,
            base_pose_cam,
            marker_pose_cam,
            wrist_pose_cam,
            measured_jpos,
        )

    def _save_world_calibration(self, calibration: StaticSceneCalibration) -> None:
        cache_path = self._world_calibration_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "base_pose_cam": calibration.base_pose_cam,
            "world_from_cam": calibration.world_from_cam,
            "table_plane_world": calibration.table_plane_world,
            "num_frames": np.asarray([calibration.num_frames], dtype=np.int64),
        }
        if calibration.wrist_marker_pose_wrist is not None:
            save_kwargs["wrist_marker_pose_wrist"] = calibration.wrist_marker_pose_wrist
        if calibration.wrist_marker_delta_pose_wrist is not None:
            save_kwargs["wrist_marker_delta_pose_wrist"] = (
                calibration.wrist_marker_delta_pose_wrist
            )
        np.savez_compressed(cache_path, **save_kwargs)

    def _load_cached_world_calibration(self) -> StaticSceneCalibration | None:
        cache_path = self._world_calibration_cache_path()
        if not cache_path.exists():
            return None
        loaded = np.load(cache_path)
        return StaticSceneCalibration(
            base_pose_cam=np.asarray(loaded["base_pose_cam"], dtype=np.float32),
            world_from_cam=np.asarray(loaded["world_from_cam"], dtype=np.float32),
            table_plane_world=np.asarray(loaded["table_plane_world"], dtype=np.float32),
            num_frames=int(np.asarray(loaded["num_frames"]).reshape(-1)[0]),
            wrist_marker_pose_wrist=(
                np.asarray(loaded["wrist_marker_pose_wrist"], dtype=np.float32)
                if "wrist_marker_pose_wrist" in loaded
                else None
            ),
            wrist_marker_delta_pose_wrist=(
                np.asarray(loaded["wrist_marker_delta_pose_wrist"], dtype=np.float32)
                if "wrist_marker_delta_pose_wrist" in loaded
                else None
            ),
        )

    def _run_world_frame_calibration(self) -> StaticSceneCalibration:
        hand = self._active_real_hand()
        if hand.base_pose is None:
            raise ValueError(
                "World calibration requires Hand.base_pose from the task demo."
            )
        if not isinstance(hand.robot, type(self._active_real_robot_entity())):
            raise TypeError(
                f"Expected hand.robot to be RealRobotEntity, got {type(hand.robot).__name__}."
            )

        input(
            "World calibration will capture several manually positioned robot configurations. "
            "Clear the table, ensure the wrist marker can be seen by the camera, "
            "and be ready to move the robot by hand. Press Enter to continue..."
        )
        aruco_dict = get_aruco_dictionary(self.real_cfg.marker_dict_name)
        calibration = self._camera_calibration()
        color_intrinsics = calibration["color_intrinsics"]
        aligned_depth_intrinsics = calibration["aligned_depth_intrinsics"]
        wrist_marker_id = int(self.real_cfg.hand_outer_palm_marker_id)
        wrist_marker_size = float(self.real_cfg.hand_outer_palm_marker_size_m)
        wrist_marker_pose_wrist = self._wrist_marker_pose_wrist_frame()
        world_base_pose = np.asarray(hand.base_pose, dtype=np.float32)
        if world_base_pose.shape != (4, 4):
            raise ValueError(
                f"Expected hand.base_pose shape (4, 4), got {world_base_pose.shape}."
            )

        calibration_observations: list[WorldCalibrationObservation] = []
        base_pose_cam_closed_form: list[np.ndarray] = []
        num_robot_configs = int(self.real_cfg.world_calibration_num_robot_configs)
        if num_robot_configs <= 0:
            raise ValueError(
                "Expected world_calibration_num_robot_configs > 0, got "
                f"{num_robot_configs}."
            )
        sample_idx = 0
        for frame_idx in range(num_robot_configs):
            calibration_sample = self._capture_manual_world_calibration_sample(
                hand,
                sample_idx=sample_idx,
                frame_idx=frame_idx,
                color_intrinsics=color_intrinsics,
                aligned_depth_intrinsics=aligned_depth_intrinsics,
                aruco_dict=aruco_dict,
                wrist_marker_id=wrist_marker_id,
                wrist_marker_size=wrist_marker_size,
                wrist_marker_pose_wrist=wrist_marker_pose_wrist,
                world_base_pose=world_base_pose,
            )
            if calibration_sample is None:
                continue
            (
                observation,
                scene_pcd_cam,
                scene_colors_cam,
                base_pose_cam,
                marker_pose_cam,
                wrist_pose_cam,
                _measured_jpos,
            ) = calibration_sample
            self._visualize_world_calibration_sample(
                scene_pcd_cam=scene_pcd_cam,
                scene_colors_cam=scene_colors_cam,
                base_pose_cam=base_pose_cam,
                marker_pose_cam=marker_pose_cam,
                wrist_pose_cam=wrist_pose_cam,
                frame_idx=frame_idx,
            )
            if not self._should_keep_calibration_sample(
                frame_idx=frame_idx,
                sample_idx=sample_idx,
                base_pose_cam=base_pose_cam,
                accepted_base_pose_cam=base_pose_cam_closed_form,
            ):
                continue
            calibration_observations.append(observation)
            base_pose_cam_closed_form.append(base_pose_cam)
            sample_idx += 1

        if len(calibration_observations) == 0:
            raise RuntimeError(
                "World calibration collected zero accepted samples after visual filtering."
            )

        initial_base_pose_cam = np.asarray(
            base_pose_cam_closed_form[0], dtype=np.float32
        )
        solver = WorldCalibrationSolver(
            initial_wrist_marker_pose_wrist=wrist_marker_pose_wrist,
            marker_delta_translation_sigma_m=float(
                self.real_cfg.world_calibration_marker_delta_translation_sigma_m
            ),
            marker_delta_rotation_sigma_deg=float(
                self.real_cfg.world_calibration_marker_delta_rotation_sigma_deg
            ),
            marker_delta_regularization_weight=float(
                self.real_cfg.world_calibration_marker_delta_regularization_weight
            ),
        )
        solution = solver.solve(
            calibration_observations,
            initial_base_pose_cam=initial_base_pose_cam,
        )
        world_from_cam = (
            world_base_pose @ np.linalg.inv(solution.base_pose_cam)
        ).astype(np.float32)
        table_plane_world = horizontal_plane_from_z(
            z_height_world=float(world_base_pose[2, 3])
        )
        estimated = StaticSceneCalibration(
            base_pose_cam=solution.base_pose_cam,
            world_from_cam=world_from_cam,
            table_plane_world=table_plane_world,
            num_frames=len(calibration_observations),
            wrist_marker_pose_wrist=solution.wrist_marker_pose_wrist,
            wrist_marker_delta_pose_wrist=solution.wrist_marker_delta_pose_wrist,
        )
        print(
            "[WorldCalibration] "
            f"num_configs={estimated.num_frames}, "
            f"mean_pos_error_m={float(np.mean(solution.per_observation_position_errors_m)):.6f}, "
            f"max_pos_error_m={float(np.max(solution.per_observation_position_errors_m)):.6f}, "
            f"mean_rot_error_deg={float(np.mean(solution.per_observation_rotation_errors_deg)):.6f}, "
            f"max_rot_error_deg={float(np.max(solution.per_observation_rotation_errors_deg)):.6f}"
        )
        print(
            "[WorldCalibration] optimized_wrist_marker_pose_wrist=\n"
            f"{np.array2string(solution.wrist_marker_pose_wrist, precision=6, suppress_small=False)}"
        )
        print(
            "[WorldCalibration] wrist_marker_delta_pose_wrist=\n"
            f"{np.array2string(solution.wrist_marker_delta_pose_wrist, precision=6, suppress_small=False)}"
        )
        self._save_world_calibration(estimated)
        return estimated

    def _initialize_world_calibration(self) -> None:
        calibration = None
        if self.real_cfg.reuse_cached_world_calibration:
            calibration = self._load_cached_world_calibration()
        if calibration is None and self.real_cfg.run_world_frame_calibration:
            calibration = self._run_world_frame_calibration()
        self._world_calibration = calibration

    def _get_object_pcd_impl(self, normals: bool) -> torch.Tensor:
        if self.env.num_envs != 1:
            raise ValueError(
                f"PhysicsReal expects a single real environment, got {self.env.num_envs}."
            )
        scene_pcd_world, fresh_object_pcd_world = (
            self._get_scene_and_filtered_object_pcd_world()
        )
        if not self.real_cfg.use_phase_cached_object_pcd:
            self._episode_initial_object_pcd_world = None
            self._episode_initial_table_pcd_world = None
            self._lift_start_object_pcd_world = None
            self._lift_start_wrist_pose_world = None
            self._visualize_world_eval_state(
                scene_pcd_world=scene_pcd_world,
                object_pcd_world=fresh_object_pcd_world,
            )
            return self._sample_object_pcd(
                object_pcd_world=fresh_object_pcd_world,
                normals=normals,
            )
        if self._episode_initial_object_pcd_world is None:
            self._episode_initial_object_pcd_world = fresh_object_pcd_world.copy()

        current_object_pcd_world = np.asarray(
            self._episode_initial_object_pcd_world, dtype=np.float32
        )
        before_motion_phase_mask = (
            self._current_before_cached_object_pcd_motion_phase_mask()
        )
        before_object_motion_phase = bool(before_motion_phase_mask[0].item())

        if before_object_motion_phase:
            self._lift_start_object_pcd_world = None
            self._lift_start_wrist_pose_world = None
        else:
            if self._lift_start_wrist_pose_world is None:
                self._lift_start_wrist_pose_world = self._current_wrist_pose_world()
                self._lift_start_object_pcd_world = current_object_pcd_world.copy()
            else:
                if self._lift_start_object_pcd_world is None:
                    raise ValueError(
                        "lift-start object PCD cache is missing while a lift-start wrist pose is already set."
                    )
                current_wrist_pose_world = self._current_wrist_pose_world()
                wrist_delta_world = (
                    current_wrist_pose_world
                    @ np.linalg.inv(self._lift_start_wrist_pose_world)
                ).astype(np.float32)
                current_object_pcd_world = transform_points_homogeneous(
                    self._lift_start_object_pcd_world,
                    wrist_delta_world,
                ).astype(np.float32)

        self._visualize_world_eval_state(
            scene_pcd_world=scene_pcd_world,
            object_pcd_world=fresh_object_pcd_world,
            heuristic_object_pcd_world=current_object_pcd_world,
        )
        return self._sample_object_pcd(
            object_pcd_world=current_object_pcd_world,
            normals=normals,
        )
