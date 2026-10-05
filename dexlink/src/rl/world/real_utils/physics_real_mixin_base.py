"""Shared base declarations for PhysicsReal mixins.

This class exists to document the attributes and cross-mixin methods that the
individual real-backend mixins rely on. It is not intended to be instantiated
directly; PhysicsReal provides the concrete implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from dexlink.utils.retargeting.vis_server import VizServer
from dexlink.src.rl.world.real_utils.robot_entity import RealRobotEntity
from dexlink.src.rl.world.real_utils.vision import StaticSceneCalibration
from dexlink.src.rl.world.robot import Hand


class PhysicsRealMixinBase:
    real_cfg: Any
    env: Any
    logs_dir: Path
    is_recording: bool
    scene_entities: dict[str, object]
    setup_viz: VizServer | None
    runtime_viz: VizServer | None
    _runtime_viz_is_shared: bool
    _world_calibration: StaticSceneCalibration | None
    _recorded_rgb_frames: list[np.ndarray]
    _recording_last_frame_time: float | None
    _recording_frame_interval_s: float
    _ik_scene: Any
    _ik_entity: Any
    _ik_entity_n_qs: int | None

    def _active_real_hand(self) -> Hand:
        raise NotImplementedError

    def _active_real_robot_entity(self) -> RealRobotEntity:
        raise NotImplementedError

    def _capture_tactile_reference(self) -> None:
        raise NotImplementedError

    def _entity_control_q_from_snapshot(
        self, entity: RealRobotEntity
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _entity_link_pose_by_name_from_snapshot(
        self, entity: RealRobotEntity
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _entity_state_snapshot(self, entity: RealRobotEntity) -> dict[str, object]:
        raise NotImplementedError

    def _hand_for_entity(self, entity: RealRobotEntity) -> Hand:
        raise NotImplementedError

    def _latest_camera_frame(self, *, force_capture: bool) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _pose_from_wxyz(
        self,
        pos: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
    ) -> np.ndarray:
        raise NotImplementedError

    def _wrist_marker_pose_wrist_frame(self) -> np.ndarray:
        raise NotImplementedError

    def _current_wrist_marker_pose_wrist(self) -> np.ndarray:
        raise NotImplementedError

    def _eef_pose_from_target(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_entity_link_pos(self, entity, *, link_name: str, ref: str):
        raise NotImplementedError

    def get_entity_link_quat(self, entity, *, link_name: str):
        raise NotImplementedError

    def get_hand_dof_limits(
        self, hand: Hand, dof_indices: list[int] | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_hand_joint_positions(self, hand: Hand) -> torch.Tensor:
        raise NotImplementedError

    def _initialize_ik_backend(self, hand: Hand) -> None:
        raise NotImplementedError

    def _solve_entity_ik_impl(
        self,
        entity,
        *,
        link_name: str,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        dofs_idx_local: list[int] | np.ndarray,
        init_qpos: torch.Tensor,
        init_qpos_dofs_idx_local: list[int] | np.ndarray,
    ) -> torch.Tensor:
        raise NotImplementedError
