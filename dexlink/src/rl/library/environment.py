"""Physical task-library environment with explicit insertion outcome metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dexlink.utils.retargeting.math_utils import matrix_from_quat
from dexlink.src.retargeting.data.process_arctic import RandomTrajProcessor
from dexlink.src.rl.multi_object.env import MultiObjectEnv
from dexlink.src.rl.multi_object.task_manager import MultiObjectTaskManager
from dexlink.src.rl.multi_object.reset_manager import MultiObjectResetManager

INSERTION_PHASES = {
    "long_peg_insertion": "sustained_insertion",
    "peg_extraction": "peg_lift",
    "insert_twist_lock": "twist_lock",
    "ring_on_post": "fit_piece_lower",
    "fit_recess_piece": "fit_piece_lower",
}


class LibraryProcessor(RandomTrajProcessor):
    """Read native robot motion directly, without inventing MANO observations."""

    def load_trajectory(self):
        motion = super().load_trajectory()
        with np.load(self.robot_obj_traj_path, allow_pickle=True) as data:
            for key in data.files:
                if key.startswith("scene_object_") or key in {
                    "scene_table_height",
                    "scene_table_size",
                    "scene_canonical_object_id",
                    "scene_active_object_index",
                    "traj_robot_base_pose",
                }:
                    motion[key] = data[key].copy()
            self.phase_names = data["phase_names"].tolist()
            self.phase_lengths = data["phase_lengths"].copy()
            motion["phase_names"] = self.phase_names
            motion["phase_lengths"] = self.phase_lengths
            motion["robot_model"] = data["traj_robot_model"].item()
        self.requires_human_reference_joint_pos_for_eval = False
        return motion


def relative_pose_metrics(
    moving_pos: torch.Tensor,
    moving_quat: torch.Tensor,
    fixture_pos: torch.Tensor,
    fixture_quat: torch.Tensor,
    goal_position: torch.Tensor,
    goal_rotation: torch.Tensor,
    yaw_symmetry: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure the physical fit in the receiver frame, respecting shape symmetry."""
    count = moving_pos.shape[0]
    if moving_pos.shape != (count, 3) or fixture_pos.shape != (count, 3):
        raise ValueError("Expected moving/fixture positions shaped (N,3).")
    if moving_quat.shape != (count, 4) or fixture_quat.shape != (count, 4):
        raise ValueError("Expected moving/fixture WXYZ quaternions shaped (N,4).")
    if goal_position.shape != (3,) or goal_rotation.shape != (3, 3) or yaw_symmetry < 0:
        raise ValueError(
            "Expected goal position (3,), rotation (3,3), nonnegative yaw symmetry."
        )
    receiver_rotation = matrix_from_quat(fixture_quat)
    position = torch.einsum("nji,nj->ni", receiver_rotation, moving_pos - fixture_pos)
    rotation = receiver_rotation.transpose(-1, -2) @ matrix_from_quat(moving_quat)
    delta = position - goal_position
    radial_error = torch.linalg.vector_norm(delta[:, :2], dim=-1)
    axial_error = delta[:, 2].abs()
    if yaw_symmetry == 0:
        # Round shafts and rings do not have a preferred rotation about their axis.
        cosine = (rotation[:, :, 2] * goal_rotation[:, 2]).sum(-1)
        angle = torch.acos(cosine.clamp(-1, 1))
    else:
        yaw = torch.arange(yaw_symmetry, device=rotation.device) * (
            2 * torch.pi / yaw_symmetry
        )
        symmetry = torch.zeros((yaw_symmetry, 3, 3), device=rotation.device)
        symmetry[:, 0, 0] = symmetry[:, 1, 1] = torch.cos(yaw)
        symmetry[:, 0, 1] = -torch.sin(yaw)
        symmetry[:, 1, 0] = torch.sin(yaw)
        symmetry[:, 2, 2] = 1
        targets = goal_rotation[None] @ symmetry
        difference = targets[None].transpose(-1, -2) @ rotation[:, None]
        cosine = (difference.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        angle = torch.acos(cosine.clamp(-1, 1)).amin(-1)
    return radial_error, axial_error, angle


class LibraryTaskManager(MultiObjectTaskManager):
    def initialize(self) -> None:
        cfg = self.task_cfg
        if cfg.start_timestep != 0 or cfg.end_timestep != -1 or cfg.skip != 1:
            raise ValueError(
                "Library references require the full phase timeline: start=0, end=-1, skip=1."
            )
        self.original_demo = LibraryProcessor(
            retargeted_results=cfg.oakink_exported_path
        )
        self.hand_side_available = self._resolve_hand_side_availability()
        self.total_steps = self.original_demo.n_frames
        self._initialize_phase_metadata()
        self._initialize_scene_object_metadata()
        self._initialize_task_metric()
        self.goal = self.env.library_goal
        self.goal_reached = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        if self.goal is not None:
            self.goal_position = torch.tensor(
                self.goal["relative_position"], device=self.env.device
            )
            self.goal_rotation = torch.tensor(
                self.goal["relative_rotation"], device=self.env.device
            )

    @property
    def task_metric_names(self) -> tuple[str, ...]:
        if self.env.library_goal is None:
            return ()
        return (
            "task_success",
            "insertion_goal_reached",
            "insertion_radial_error_m",
            "insertion_axial_error_m",
            "insertion_orientation_error_rad",
        )

    @property
    def primary_task_metric_name(self) -> str | None:
        return "task_success" if self.env.library_goal is not None else None

    def _goal_metrics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        entities = self.env.physics_manager.object_entities
        moving = entities[self.goal["moving_object"]].get_ground_truth_info()
        fixture = entities[self.goal["fixture_object"]].get_ground_truth_info()
        radial, axial, orientation = relative_pose_metrics(
            moving["pos"],
            moving["quat"],
            fixture["pos"],
            fixture["quat"],
            self.goal_position,
            self.goal_rotation,
            self.goal["yaw_symmetry"],
        )
        reached = (
            (radial <= self.goal["radial_tolerance_m"])
            & (axial <= self.goal["axial_tolerance_m"])
            & (orientation <= self.goal["orientation_tolerance_rad"])
        )
        return reached, radial, axial, orientation

    def compute_task_metrics(self) -> dict[str, torch.Tensor]:
        if self.env.library_goal is None:
            return {}
        reached, radial, axial, orientation = self._goal_metrics()
        return {
            "task_success": (self.goal_reached | reached).float(),
            "insertion_goal_reached": reached.float(),
            "insertion_radial_error_m": radial,
            "insertion_axial_error_m": axial,
            "insertion_orientation_error_rad": orientation,
        }

    def update_task_metric_accumulators(self) -> None:
        if self.env.library_goal is not None:
            self.goal_reached |= self._goal_metrics()[0]

    def reset_task_metric_accumulators(self, env_ids: np.ndarray) -> None:
        self.goal_reached[env_ids] = False


class LibraryResetManager(MultiObjectResetManager):
    def reset_robot_state_for_eval(
        self, frame_idx: np.ndarray, env_ids: np.ndarray
    ) -> None:
        # Resets already place the hands and all actors at the recorded waypoint.
        # An extra approach animation would advance gravity before the policy acts.
        return None


class LibraryEnv(MultiObjectEnv):
    def __init__(self, cfg) -> None:
        directory = Path(cfg.task_cfg.oakink_exported_path).parent
        self.library_task = json.loads((directory / "task.json").read_text())
        self.library_goal = json.loads((directory / "goal.json").read_text())
        if cfg.task_cfg.metric_cfg.metric_id is not None:
            raise ValueError(
                "LibraryEnv supplies insertion metrics; expected metric_id=None."
            )
        super().__init__(cfg)

    def _create_task_manager(self):
        return LibraryTaskManager(self.cfg.task_cfg, self)

    def _create_physics_manager(self):
        from dexlink.src.rl.library.physics import (
            NovelPhysicsSim,
        )

        return NovelPhysicsSim(self.cfg.sim_cfg, self)

    def _create_reset_manager(self):
        return LibraryResetManager(self.cfg, self)
