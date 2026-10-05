"""Live robot specialization for the real backend.

RealRobotEntity extends RealEntity with hardware-facing runtime state:
- Franka/XHand controller handles
- RealSense camera handle
- optional background robot state refresh and FK updates
- optional background control target streaming

PhysicsReal reads from this class only through snapshots so env-side computations see
a stable state within one env step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING

import cv2
import numpy as np

from dexlink.src.rl.world.real_utils.camera import create_realsense_camera_bundle
from dexlink.src.rl.world.real_utils.control import create_franka_xhand_robot_base
from dexlink.src.rl.world.real_utils.entities import RealEntity

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import RealCfg


FRANKA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


@dataclass
class RealRobotEntity(RealEntity):
    """URDF-backed real robot entity with live I/O, state buffering, and control."""

    side: str = "right"
    robot_controller: object | None = None
    camera: object | None = None
    camera_calibration: dict[str, np.ndarray] | None = None
    robot_control_config_path: str | None = None
    state_poll_period_s: float = 0.01
    control_period_s: float = 0.01
    use_threaded_state_query: bool = True
    use_threaded_control: bool = True
    robot_init_wait_s: float = 2.0
    robot_init_state_timeout_s: float = 10.0
    robot_init_state_poll_s: float = 0.05
    franka_control_mode: str = "joint_impedance"
    bypass_low_level_control: bool = False
    control_freq_hz: float = 10.0
    control_target_progress_per_update: float = 0.3
    control_target_max_joint_step_rad: float = 0.05
    reset_motion_franka_max_step_rad: float = 0.05
    reset_motion_xhand_steps: int = 50
    camera_poll_period_s: float = 0.05
    camera_capture_timeout_s: float = 2.0
    _state_lock: Lock | None = None
    _state_stop_event: Event | None = None
    _state_thread: Thread | None = None
    _latest_robot_state: dict | None = None
    _latest_link_pose_by_name: dict[str, np.ndarray] | None = None
    _latest_state_timestamp: float | None = None
    _target_lock: Lock | None = None
    _control_stop_event: Event | None = None
    _control_thread: Thread | None = None
    _target_franka_q: np.ndarray | None = None
    _target_eef_pose: np.ndarray | None = None
    _target_xhand_q: np.ndarray | None = None
    _commanded_franka_q: np.ndarray | None = None
    _commanded_eef_pose: np.ndarray | None = None
    _commanded_xhand_q: np.ndarray | None = None
    _control_timestamp: float | None = None
    _camera_lock: Lock | None = None
    _camera_stop_event: Event | None = None
    _camera_thread: Thread | None = None
    _latest_camera_frame: dict[str, np.ndarray] | None = None
    _latest_camera_timestamp: float | None = None
    _camera_preview_dir: Path | None = None
    _camera_preview_write_period_s: float = 0.5
    _last_camera_preview_write_time: float | None = None
    _runtime_log_dir: Path | None = None

    @classmethod
    def from_entity(cls, entity: RealEntity, *, side: str) -> "RealRobotEntity":
        return cls(
            name=entity.name,
            urdf_path=entity.urdf_path,
            urdf_inst=entity.urdf_inst,
            fixed=entity.fixed,
            collision=entity.collision,
            convexify=entity.convexify,
            recompute_inertia=entity.recompute_inertia,
            pos=entity.pos,
            quat=entity.quat,
            world_base_pose=entity.world_base_pose.copy(),
            links_to_keep=list(entity.links_to_keep),
            rho=entity.rho,
            friction=entity.friction,
            gravity_compensation=entity.gravity_compensation,
            base_link_name=entity.base_link_name,
            base_link_idx=entity.base_link_idx,
            n_qs=entity.n_qs,
            joint_dofs_idx_local=dict(entity.joint_dofs_idx_local),
            joint_parent_link_name=dict(entity.joint_parent_link_name),
            joint_child_link_name=dict(entity.joint_child_link_name),
            link_idx_local=dict(entity.link_idx_local),
            link_idx=dict(entity.link_idx),
            dof_limit_lower=list(entity.dof_limit_lower),
            dof_limit_upper=list(entity.dof_limit_upper),
            side=side,
        )

    def initialize_real_interfaces(self, real_cfg: "RealCfg") -> None:
        self.use_threaded_state_query = bool(real_cfg.use_threaded_state_query)
        self.use_threaded_control = bool(real_cfg.use_threaded_control)
        if self.use_threaded_control and self.use_threaded_state_query:
            self.use_threaded_state_query = False
        self.state_poll_period_s = float(real_cfg.state_poll_period_s)
        self.control_period_s = float(real_cfg.control_period_s)
        self.robot_init_wait_s = float(real_cfg.robot_init_wait_s)
        self.robot_init_state_timeout_s = float(real_cfg.robot_init_state_timeout_s)
        self.robot_init_state_poll_s = float(real_cfg.robot_init_state_poll_s)
        self.franka_control_mode = str(real_cfg.franka_control_mode)
        self.bypass_low_level_control = bool(real_cfg.bypass_low_level_control)
        self.control_freq_hz = float(real_cfg.control_freq_hz)
        self.control_target_progress_per_update = float(
            real_cfg.control_target_progress_per_update
        )
        self.control_target_max_joint_step_rad = float(
            real_cfg.control_target_max_joint_step_rad
        )
        self.reset_motion_franka_max_step_rad = float(
            real_cfg.reset_motion_franka_max_step_rad
        )
        self.reset_motion_xhand_steps = int(real_cfg.reset_motion_xhand_steps)
        if self.reset_motion_franka_max_step_rad <= 0.0:
            raise ValueError(
                "Expected real_cfg.reset_motion_franka_max_step_rad > 0, got "
                f"{self.reset_motion_franka_max_step_rad}."
            )
        if self.reset_motion_xhand_steps <= 0:
            raise ValueError(
                "Expected real_cfg.reset_motion_xhand_steps > 0, got "
                f"{self.reset_motion_xhand_steps}."
            )
        if real_cfg.initialize_robot:
            output_log_path = None
            if self._runtime_log_dir is not None:
                output_log_path = (
                    self._runtime_log_dir / real_cfg.robot_control_log_filename
                )
            self.robot_controller, self.robot_control_config_path = (
                create_franka_xhand_robot_base(
                    real_cfg.robot_control_config_path,
                    real_cfg=real_cfg,
                    output_log_path=output_log_path,
                )
            )
            self._initialize_state_buffer()
            if real_cfg.read_robot_state_on_initialize:
                self._wait_for_initial_robot_state()
                self._refresh_state_buffer(force_update=True)
                self._initialize_position_targets()
                if self.use_threaded_state_query:
                    self._start_state_polling()
                if self.use_threaded_control and not self.bypass_low_level_control:
                    self._start_position_control()
        if real_cfg.initialize_camera:
            camera_bundle = create_realsense_camera_bundle(
                real_cfg,
                log_prefix=f"RealRobotEntity[{self.name or self.side}] camera",
            )
            self.camera = camera_bundle.camera
            self.camera_calibration = camera_bundle.calibration
            self.camera_capture_timeout_s = float(real_cfg.camera_capture_timeout_s)
            if real_cfg.start_camera_polling_on_initialize:
                self._start_camera_polling()

    def set_camera_preview_dir(self, preview_dir: str | Path) -> None:
        self._camera_preview_dir = Path(preview_dir)
        self._camera_preview_dir.mkdir(parents=True, exist_ok=True)

    def set_runtime_log_dir(self, log_dir: str | Path) -> None:
        self._runtime_log_dir = Path(log_dir)
        self._runtime_log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def xhand_joint_names(self) -> list[str]:
        if self.side not in {"left", "right"}:
            raise ValueError(
                f"Expected hand side to be 'left' or 'right', got {self.side!r}."
            )
        return [
            f"{self.side}_hand_thumb_bend_joint",
            f"{self.side}_hand_thumb_rota_joint1",
            f"{self.side}_hand_thumb_rota_joint2",
            f"{self.side}_hand_index_bend_joint",
            f"{self.side}_hand_index_joint1",
            f"{self.side}_hand_index_joint2",
            f"{self.side}_hand_mid_joint1",
            f"{self.side}_hand_mid_joint2",
            f"{self.side}_hand_ring_joint1",
            f"{self.side}_hand_ring_joint2",
            f"{self.side}_hand_pinky_joint1",
            f"{self.side}_hand_pinky_joint2",
        ]

    @property
    def helper_root_joint_names(self) -> list[str]:
        prefix = "L" if self.side == "left" else "R"
        return [
            f"{prefix}_forearm_tx_link_joint",
            f"{prefix}_forearm_ty_link_joint",
            f"{prefix}_forearm_tz_link_joint",
            f"{prefix}_forearm_roll_link_joint",
            f"{prefix}_forearm_pitch_link_joint",
            f"{prefix}_forearm_yaw_link_joint",
        ]

    def dof_idx_to_joint_name(self) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for joint_name, dofs_idx_local in self.joint_dofs_idx_local.items():
            if len(dofs_idx_local) != 1:
                raise ValueError(
                    f"Expected joint {joint_name!r} to map to exactly one DOF, got {dofs_idx_local}."
                )
            dof_idx = dofs_idx_local[0]
            if dof_idx in mapping:
                raise ValueError(
                    f"Duplicate local DOF index {dof_idx} for joints {mapping[dof_idx]!r} and {joint_name!r}."
                )
            mapping[dof_idx] = joint_name
        return mapping

    def current_control_q(self) -> tuple[np.ndarray, np.ndarray]:
        robot_state = self.latest_robot_state()
        franka_q = np.asarray(robot_state["franka"]["q"], dtype=np.float32).reshape(-1)
        xhand_q = np.asarray(robot_state["xhand"]["q"], dtype=np.float32).reshape(-1)
        if franka_q.shape != (len(FRANKA_JOINT_NAMES),):
            raise ValueError(
                f"Expected Franka state shape {(len(FRANKA_JOINT_NAMES),)}, got {franka_q.shape}."
            )
        if xhand_q.shape != (len(self.xhand_joint_names),):
            raise ValueError(
                f"Expected XHand state shape {(len(self.xhand_joint_names),)}, got {xhand_q.shape}."
            )
        return franka_q, xhand_q

    def _wait_for_initial_robot_state(self) -> None:
        if self.robot_controller is None:
            raise RuntimeError("robot_controller is not initialized.")
        if self.robot_init_wait_s < 0.0:
            raise ValueError(
                f"Expected robot_init_wait_s >= 0, got {self.robot_init_wait_s}."
            )
        if self.robot_init_state_timeout_s <= 0.0:
            raise ValueError(
                "Expected robot_init_state_timeout_s > 0, got "
                f"{self.robot_init_state_timeout_s}."
            )
        if self.robot_init_state_poll_s <= 0.0:
            raise ValueError(
                f"Expected robot_init_state_poll_s > 0, got {self.robot_init_state_poll_s}."
            )
        if self.robot_init_wait_s > 0.0:
            time.sleep(self.robot_init_wait_s)

        start_time = time.time()
        last_error: Exception | None = None
        while True:
            try:
                self.robot_controller._refresh_state(force_update=True)
                robot_state = self.robot_controller.robot_state
                franka_state = robot_state.get("franka")
                xhand_state = robot_state.get("xhand")
                if not isinstance(franka_state, dict):
                    raise TypeError(
                        f"Expected franka state dict during init, got {type(franka_state).__name__}."
                    )
                if not isinstance(xhand_state, dict):
                    raise TypeError(
                        f"Expected xhand state dict during init, got {type(xhand_state).__name__}."
                    )
                franka_q = np.asarray(franka_state["q"], dtype=np.float32).reshape(-1)
                xhand_q = np.asarray(xhand_state["q"], dtype=np.float32).reshape(-1)
                if franka_q.shape != (len(FRANKA_JOINT_NAMES),):
                    raise ValueError(
                        f"Expected initial Franka q shape {(len(FRANKA_JOINT_NAMES),)}, got {franka_q.shape}."
                    )
                if xhand_q.shape != (len(self.xhand_joint_names),):
                    raise ValueError(
                        f"Expected initial XHand q shape {(len(self.xhand_joint_names),)}, got {xhand_q.shape}."
                    )
                return
            except Exception as exc:
                last_error = exc
            if time.time() - start_time > self.robot_init_state_timeout_s:
                raise RuntimeError(
                    "Timed out waiting for the initial Franka/XHand state after "
                    f"{self.robot_init_state_timeout_s} seconds."
                ) from last_error
            time.sleep(self.robot_init_state_poll_s)

    def _initialize_state_buffer(self) -> None:
        if self.robot_controller is None:
            raise RuntimeError(
                "Cannot initialize state buffer without a robot_controller."
            )
        if self._state_lock is not None:
            return
        self._state_lock = Lock()

    def _start_state_polling(self) -> None:
        if self.robot_controller is None:
            raise RuntimeError("Cannot start state polling without a robot_controller.")
        if self._state_thread is not None and self._state_thread.is_alive():
            return
        if self._state_lock is None:
            raise RuntimeError("State lock is not initialized.")
        self._state_stop_event = Event()
        self._state_thread = Thread(
            target=self._state_poll_loop,
            name=f"real-state-poll-{self.name or self.side}",
            daemon=True,
        )
        self._state_thread.start()

    def _state_poll_loop(self) -> None:
        assert self._state_stop_event is not None
        while not self._state_stop_event.is_set():
            loop_start = time.time()
            try:
                self._refresh_state_buffer(force_update=False)
            except Exception:
                pass
            elapsed = time.time() - loop_start
            time.sleep(max(self.state_poll_period_s - elapsed, 0.0))

    def _refresh_state_buffer(self, *, force_update: bool) -> None:
        if self.robot_controller is None:
            raise RuntimeError(
                "Cannot refresh state buffer without a robot_controller."
            )
        self.robot_controller._refresh_state(force_update=force_update)
        self._store_state_buffer(self.robot_controller.robot_state)

    def _store_state_buffer(self, robot_state: dict) -> None:
        link_pose_by_name = self._compute_link_pose_by_name(robot_state)
        if self._state_lock is None:
            raise RuntimeError("State lock is not initialized.")
        with self._state_lock:
            self._latest_robot_state = robot_state
            self._latest_link_pose_by_name = link_pose_by_name
            self._latest_state_timestamp = time.time()

    def latest_robot_state(self) -> dict:
        if self.robot_controller is None:
            raise RuntimeError("Real robot controller is not initialized.")
        if self._state_lock is None:
            self._initialize_state_buffer()
        if self._latest_robot_state is None:
            self._refresh_state_buffer(force_update=True)
        if self._state_lock is None:
            raise RuntimeError("State lock is not initialized.")
        with self._state_lock:
            if self._latest_robot_state is None:
                raise RuntimeError("Latest real robot state is not available.")
            latest_state = self._latest_robot_state
        return latest_state

    def latest_link_pose_by_name(self) -> dict[str, np.ndarray]:
        if self._state_lock is None:
            self._initialize_state_buffer()
        if self._latest_link_pose_by_name is None:
            self._refresh_state_buffer(force_update=True)
        if self._state_lock is None:
            raise RuntimeError("State lock is not initialized.")
        with self._state_lock:
            if self._latest_link_pose_by_name is None:
                raise RuntimeError("Latest link poses are not available.")
            pose_by_name = {
                name: pose.copy()
                for name, pose in self._latest_link_pose_by_name.items()
            }
        return pose_by_name

    def latest_state_snapshot(self, *, force_update: bool = False) -> dict[str, object]:
        if self.robot_controller is None:
            raise RuntimeError("Real robot controller is not initialized.")
        if self._state_lock is None:
            self._initialize_state_buffer()
        if (
            force_update
            or self._latest_robot_state is None
            or self._latest_link_pose_by_name is None
        ):
            self._refresh_state_buffer(force_update=force_update)
        if self._state_lock is None:
            raise RuntimeError("State lock is not initialized.")
        with self._state_lock:
            if self._latest_robot_state is None:
                raise RuntimeError("Latest real robot state is not available.")
            if self._latest_link_pose_by_name is None:
                raise RuntimeError("Latest link poses are not available.")
            return {
                "robot_state": self._latest_robot_state,
                "link_pose_by_name": {
                    name: pose.copy()
                    for name, pose in self._latest_link_pose_by_name.items()
                },
                "timestamp": self._latest_state_timestamp,
            }

    def _compute_link_pose_by_name(self, robot_state: dict) -> dict[str, np.ndarray]:
        local_pose_by_name = self._compute_local_link_pose_by_name(robot_state)
        return {
            link_name: self.world_base_pose @ local_pose
            for link_name, local_pose in local_pose_by_name.items()
        }

    def _compute_local_link_pose_by_name(
        self, robot_state: dict
    ) -> dict[str, np.ndarray]:
        joint_cfg = self._joint_cfg_from_robot_state(robot_state)
        link_fk = self.urdf_inst.link_fk(cfg=joint_cfg)
        pose_by_name: dict[str, np.ndarray] = {}
        for link in self.urdf_inst.links:
            pose = link_fk.get(link)
            if pose is None:
                raise KeyError(f"Missing FK pose for link {link.name!r}.")
            pose_by_name[link.name] = np.asarray(pose, dtype=np.float32)
        return pose_by_name

    def latest_local_link_pose_by_name(
        self, *, force_update: bool = False
    ) -> dict[str, np.ndarray]:
        snapshot = self.latest_state_snapshot(force_update=force_update)
        robot_state = snapshot["robot_state"]
        if not isinstance(robot_state, dict):
            raise TypeError(
                f"Expected robot_state snapshot to be a dict, got {type(robot_state).__name__}."
            )
        return self._compute_local_link_pose_by_name(robot_state)

    def _joint_cfg_from_robot_state(self, robot_state: dict) -> dict[str, float]:
        franka_q = np.asarray(robot_state["franka"]["q"], dtype=np.float32).reshape(-1)
        xhand_q = np.asarray(robot_state["xhand"]["q"], dtype=np.float32).reshape(-1)
        if franka_q.shape != (len(FRANKA_JOINT_NAMES),):
            raise ValueError(
                f"Expected Franka state shape {(len(FRANKA_JOINT_NAMES),)}, got {franka_q.shape}."
            )
        if xhand_q.shape != (len(self.xhand_joint_names),):
            raise ValueError(
                f"Expected XHand state shape {(len(self.xhand_joint_names),)}, got {xhand_q.shape}."
            )
        franka_joint_to_idx = {
            joint_name: idx for idx, joint_name in enumerate(FRANKA_JOINT_NAMES)
        }
        xhand_joint_to_idx = {
            joint_name: idx for idx, joint_name in enumerate(self.xhand_joint_names)
        }
        helper_root_joint_names = set(self.helper_root_joint_names)
        joint_cfg: dict[str, float] = {}
        for joint_name in self.joint_names:
            if joint_name in franka_joint_to_idx:
                joint_cfg[joint_name] = float(franka_q[franka_joint_to_idx[joint_name]])
                continue
            if joint_name in xhand_joint_to_idx:
                joint_cfg[joint_name] = float(xhand_q[xhand_joint_to_idx[joint_name]])
                continue
            if joint_name in helper_root_joint_names:
                joint_cfg[joint_name] = 0.0
                continue
            raise NotImplementedError(
                f"Joint {joint_name!r} is not mapped to the buffered real robot state."
            )
        return joint_cfg

    def _initialize_position_targets(self) -> None:
        if self.robot_controller is None:
            raise RuntimeError(
                "Cannot initialize position targets without a robot_controller."
            )
        if self._target_lock is not None:
            return
        self._target_lock = Lock()
        robot_state = self.latest_robot_state()
        init_franka_q = np.asarray(
            robot_state["franka"]["q"], dtype=np.float32
        ).reshape(-1)
        init_eef_pose = np.asarray(
            robot_state["franka"]["eef_pose"], dtype=np.float32
        ).reshape(4, 4)
        init_xhand_q = np.asarray(robot_state["xhand"]["q"], dtype=np.float32).reshape(
            -1
        )
        with self._target_lock:
            self._target_franka_q = init_franka_q.copy()
            self._target_eef_pose = init_eef_pose.copy()
            self._target_xhand_q = init_xhand_q.copy()
            self._commanded_franka_q = init_franka_q.copy()
            self._commanded_eef_pose = init_eef_pose.copy()
            self._commanded_xhand_q = init_xhand_q.copy()
            self._control_timestamp = time.time()

    def reset_position_targets_from_current_state(self) -> None:
        if self.robot_controller is None:
            raise RuntimeError(
                "Cannot reset position targets without a robot_controller."
            )
        if self._target_lock is None:
            self._initialize_position_targets()
            return
        robot_state = self.latest_robot_state()
        franka_q = np.asarray(robot_state["franka"]["q"], dtype=np.float32).reshape(-1)
        eef_pose = np.asarray(
            robot_state["franka"]["eef_pose"], dtype=np.float32
        ).reshape(4, 4)
        xhand_q = np.asarray(robot_state["xhand"]["q"], dtype=np.float32).reshape(-1)
        with self._target_lock:
            self._target_franka_q = franka_q.copy()
            self._target_eef_pose = eef_pose.copy()
            self._target_xhand_q = xhand_q.copy()
            self._commanded_franka_q = franka_q.copy()
            self._commanded_eef_pose = eef_pose.copy()
            self._commanded_xhand_q = xhand_q.copy()
            self._control_timestamp = time.time()

    def _start_position_control(self) -> None:
        if self.robot_controller is None:
            raise RuntimeError(
                "Cannot start position control without a robot_controller."
            )
        if self._control_thread is not None and self._control_thread.is_alive():
            return
        if self._target_lock is None:
            self._initialize_position_targets()
        self._control_stop_event = Event()
        self._control_thread = Thread(
            target=self._position_control_loop,
            name=f"real-position-control-{self.name or self.side}",
            daemon=True,
        )
        self._control_thread.start()

    def _start_camera_polling(self) -> None:
        if self.camera is None:
            raise RuntimeError(
                "Cannot start camera polling without an initialized camera."
            )
        if self._camera_thread is not None and self._camera_thread.is_alive():
            return
        self._camera_lock = Lock()
        self._camera_stop_event = Event()
        self._refresh_camera_buffer(force_capture=True)
        self._camera_thread = Thread(
            target=self._camera_poll_loop,
            name=f"real-camera-poll-{self.name or self.side}",
            daemon=True,
        )
        self._camera_thread.start()

    def _camera_poll_loop(self) -> None:
        assert self._camera_stop_event is not None
        while not self._camera_stop_event.is_set():
            loop_start = time.time()
            try:
                self._refresh_camera_buffer(force_capture=False)
            except Exception:
                pass
            elapsed = time.time() - loop_start
            time.sleep(max(self.camera_poll_period_s - elapsed, 0.0))

    def _refresh_camera_buffer(self, *, force_capture: bool) -> None:
        if self.camera is None:
            raise RuntimeError(
                "Cannot refresh camera buffer without an initialized camera."
            )
        if self._camera_lock is None:
            raise RuntimeError("Camera lock is not initialized.")
        if not hasattr(self.camera, "capture"):
            raise AttributeError(
                f"Expected camera object to expose capture(), got {type(self.camera).__name__}."
            )
        obs = self.camera.capture(timeout=self.camera_capture_timeout_s)
        if obs is None:
            if force_capture:
                raise RuntimeError(
                    f"Camera capture timed out after {self.camera_capture_timeout_s} seconds."
                )
            return
        color = obs.get("color")
        depth = obs.get("depth")
        if color is None or depth is None:
            raise RuntimeError("Camera observation is missing color or depth.")
        with self._camera_lock:
            self._latest_camera_frame = {
                "color": np.asarray(color).copy(),
                "depth": np.asarray(depth).copy(),
            }
            self._latest_camera_timestamp = time.time()
        self._write_camera_preview()

    def _write_camera_preview(self) -> None:
        if self._camera_preview_dir is None:
            return
        if self._camera_lock is None:
            raise RuntimeError("Camera lock is not initialized.")
        with self._camera_lock:
            if self._latest_camera_frame is None:
                return
            frame_timestamp = self._latest_camera_timestamp
            if frame_timestamp is None:
                return
            if (
                self._last_camera_preview_write_time is not None
                and frame_timestamp - self._last_camera_preview_write_time
                < self._camera_preview_write_period_s
            ):
                return
            color = self._latest_camera_frame["color"].copy()
            depth = self._latest_camera_frame["depth"].copy()
            self._last_camera_preview_write_time = frame_timestamp
        color_path = self._camera_preview_dir / "latest_camera_color.png"
        depth_path = self._camera_preview_dir / "latest_camera_depth_vis.png"
        cv2.imwrite(str(color_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        finite_depth = np.isfinite(depth) & (depth > 0.0)
        depth_vis = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(finite_depth):
            depth_valid = depth[finite_depth]
            depth_min = float(np.min(depth_valid))
            depth_max = float(np.max(depth_valid))
            if depth_max > depth_min:
                scaled = (depth - depth_min) / (depth_max - depth_min)
                depth_vis[finite_depth] = np.clip(
                    scaled[finite_depth] * 255.0, 0.0, 255.0
                ).astype(np.uint8)
            else:
                depth_vis[finite_depth] = 255
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        depth_vis[~finite_depth] = 0
        cv2.imwrite(str(depth_path), depth_vis)

    def latest_camera_frame(
        self, *, force_capture: bool = False
    ) -> dict[str, np.ndarray]:
        if self.camera is None:
            raise RuntimeError("Real camera is not initialized.")
        if self._camera_lock is None:
            self._camera_lock = Lock()
        if force_capture or self._latest_camera_frame is None:
            self._refresh_camera_buffer(force_capture=True)
        if self._camera_lock is None:
            raise RuntimeError("Camera lock is not initialized.")
        with self._camera_lock:
            if self._latest_camera_frame is None:
                raise RuntimeError("Latest camera frame is not available.")
            return {
                "color": self._latest_camera_frame["color"].copy(),
                "depth": self._latest_camera_frame["depth"].copy(),
                "timestamp": self._latest_camera_timestamp,
            }

    def close(self) -> None:
        if self._state_stop_event is not None:
            self._state_stop_event.set()
        if self._control_stop_event is not None:
            self._control_stop_event.set()
        if self._camera_stop_event is not None:
            self._camera_stop_event.set()
        if self._state_thread is not None and self._state_thread.is_alive():
            self._state_thread.join(timeout=1.0)
        if self._control_thread is not None and self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0)
        if self._camera_thread is not None and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=1.0)
        if self.camera is not None and hasattr(self.camera, "close"):
            self.camera.close()
        if self.robot_controller is not None:
            if hasattr(self.robot_controller, "close"):
                self.robot_controller.close()
            elif hasattr(self.robot_controller, "__del__"):
                self.robot_controller.__del__()

    def latest_position_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._target_lock is None:
            raise RuntimeError("Target lock is not initialized.")
        with self._target_lock:
            if (
                self._target_franka_q is None
                or self._target_eef_pose is None
                or self._target_xhand_q is None
            ):
                raise RuntimeError("Position control targets are not initialized.")
            franka_q = self._target_franka_q.copy()
            eef_pose = self._target_eef_pose.copy()
            xhand_q = self._target_xhand_q.copy()
        return franka_q, eef_pose, xhand_q

    def update_position_targets(
        self,
        *,
        franka_q: np.ndarray | None = None,
        eef_pose: np.ndarray | None = None,
        xhand_q: np.ndarray | None = None,
    ) -> None:
        if franka_q is None and eef_pose is None and xhand_q is None:
            return
        if self._target_lock is None:
            raise RuntimeError("Target lock is not initialized.")
        with self._target_lock:
            if (
                self._target_franka_q is None
                or self._target_eef_pose is None
                or self._target_xhand_q is None
            ):
                robot_state = self.latest_robot_state()
                self._target_franka_q = np.asarray(
                    robot_state["franka"]["q"], dtype=np.float32
                ).reshape(-1)
                self._target_eef_pose = np.asarray(
                    robot_state["franka"]["eef_pose"], dtype=np.float32
                ).reshape(4, 4)
                init_xhand_q = np.asarray(
                    robot_state["xhand"]["q"], dtype=np.float32
                ).reshape(-1)
                self._target_xhand_q = init_xhand_q.copy()
            if franka_q is not None:
                franka_q = np.asarray(franka_q, dtype=np.float32).reshape(-1)
                if franka_q.shape != self._target_franka_q.shape:
                    raise ValueError(
                        f"Expected franka_q shape {self._target_franka_q.shape}, got {franka_q.shape}."
                    )
                self._target_franka_q = franka_q.copy()
            if eef_pose is not None:
                eef_pose = np.asarray(eef_pose, dtype=np.float32)
                if eef_pose.shape != (4, 4):
                    raise ValueError(
                        f"Expected eef_pose shape (4, 4), got {eef_pose.shape}."
                    )
                self._target_eef_pose = eef_pose.copy()
            if xhand_q is not None:
                xhand_q = np.asarray(xhand_q, dtype=np.float32).reshape(-1)
                if xhand_q.shape != self._target_xhand_q.shape:
                    raise ValueError(
                        f"Expected xhand_q shape {self._target_xhand_q.shape}, got {xhand_q.shape}."
                    )
                self._target_xhand_q = xhand_q.astype(np.float32)

    def send_latest_position_targets(self) -> None:
        desired_franka_q, desired_eef_pose, desired_xhand_q = (
            self.latest_position_targets()
        )
        if self.robot_controller is None:
            raise RuntimeError("robot_controller is not initialized.")
        if self._target_lock is None:
            raise RuntimeError("Target lock is not initialized.")
        now = time.time()
        with self._target_lock:
            if (
                self._commanded_franka_q is None
                or self._commanded_eef_pose is None
                or self._commanded_xhand_q is None
            ):
                self._commanded_franka_q = desired_franka_q.copy()
                self._commanded_eef_pose = desired_eef_pose.copy()
                self._commanded_xhand_q = desired_xhand_q.copy()
            elapsed_s = (
                1.0 / self.control_freq_hz
                if self._control_timestamp is None
                else max(now - self._control_timestamp, 0.0)
            )
            alpha = self._command_smoothing_alpha(elapsed_s)
            blended_franka_q = self._blend_joint_targets(
                self._commanded_franka_q,
                desired_franka_q,
                alpha,
            )
            self._commanded_franka_q = self._limit_joint_step(
                self._commanded_franka_q,
                blended_franka_q,
                self.control_target_max_joint_step_rad,
            )
            self._commanded_xhand_q = self._blend_joint_targets(
                self._commanded_xhand_q,
                desired_xhand_q,
                alpha,
            )
            franka_q = self._commanded_franka_q.copy()
            eef_pose = self._commanded_eef_pose.copy()
            xhand_q = self._commanded_xhand_q.copy()
        if self.franka_control_mode == "joint_impedance":
            self._send_joint_impedance_targets(
                franka_q=franka_q,
                xhand_q=xhand_q,
                blocking_reset=False,
            )
        elif self.franka_control_mode == "osc_absolute_pose":
            self.robot_controller.teleop_step(
                {
                    "hand_joints": xhand_q.copy(),
                    "hand_pose": eef_pose.copy(),
                }
            )
        else:
            raise ValueError(
                "Expected franka_control_mode to be 'joint_impedance' or "
                f"'osc_absolute_pose', got {self.franka_control_mode!r}."
            )
        self._store_state_buffer(self.robot_controller.robot_state)
        self._control_timestamp = time.time()

    def send_blocking_joint_targets(
        self,
        *,
        franka_q: np.ndarray,
        xhand_q: np.ndarray,
    ) -> None:
        if self.robot_controller is None:
            raise RuntimeError("robot_controller is not initialized.")
        if self.franka_control_mode != "joint_impedance":
            raise ValueError(
                "Blocking joint targets require franka_control_mode='joint_impedance', "
                f"got {self.franka_control_mode!r}."
            )
        self._send_joint_impedance_targets(
            franka_q=np.asarray(franka_q, dtype=np.float32).reshape(-1),
            xhand_q=np.asarray(xhand_q, dtype=np.float32).reshape(-1),
            blocking_reset=True,
        )
        self._refresh_state_buffer(force_update=True)
        self.reset_position_targets_from_current_state()

    def reset_default_position(self) -> None:
        if self.robot_controller is None:
            raise RuntimeError("robot_controller is not initialized.")
        if not hasattr(self.robot_controller, "reset_default_position"):
            raise AttributeError(
                "Real robot controller does not expose reset_default_position()."
            )
        self.robot_controller.reset_default_position()
        self._refresh_state_buffer(force_update=True)
        self.reset_position_targets_from_current_state()

    def _send_joint_impedance_targets(
        self,
        *,
        franka_q: np.ndarray,
        xhand_q: np.ndarray,
        blocking_reset: bool,
    ) -> None:
        franka_q = np.asarray(franka_q, dtype=np.float32).reshape(-1)
        xhand_q = np.asarray(xhand_q, dtype=np.float32).reshape(-1)
        if franka_q.shape != (len(FRANKA_JOINT_NAMES),):
            raise ValueError(
                f"Expected franka_q shape {(len(FRANKA_JOINT_NAMES),)}, got {franka_q.shape}."
            )
        controller = self.robot_controller
        if controller is None:
            raise RuntimeError("robot_controller is not initialized.")
        current_xhand_q = np.asarray(
            self.latest_robot_state()["xhand"]["q"], dtype=np.float32
        ).reshape(-1)
        if xhand_q.shape != current_xhand_q.shape:
            raise ValueError(
                f"Expected xhand_q shape {current_xhand_q.shape}, got {xhand_q.shape}."
            )
        if blocking_reset:
            if not hasattr(controller, "reset_franka_position"):
                raise AttributeError(
                    "Real robot controller does not expose reset_franka_position()."
                )
            if not hasattr(controller, "reset_xhand_position"):
                raise AttributeError(
                    "Real robot controller does not expose reset_xhand_position()."
                )
            current_franka_q = np.asarray(
                self.latest_robot_state()["franka"]["q"], dtype=np.float32
            ).reshape(-1)
            for franka_waypoint_q, xhand_waypoint_q in self._paired_joint_waypoints(
                current_franka_q=current_franka_q,
                target_franka_q=franka_q,
                current_xhand_q=current_xhand_q,
                target_xhand_q=xhand_q,
                max_step_rad=self.reset_motion_franka_max_step_rad,
            ):
                controller.reset_xhand_position(xhand_waypoint_q.copy(), num_steps=1)
                controller.franka_follow_traj(
                    [franka_waypoint_q.copy()], max_steps_repeat=1
                )
            controller.reset_xhand_position(
                xhand_q.copy(), num_steps=self.reset_motion_xhand_steps
            )
            return
        if not hasattr(controller, "franka_follow_traj"):
            raise AttributeError(
                "Real robot controller does not expose franka_follow_traj()."
            )
        if not hasattr(controller, "reset_xhand_position"):
            raise AttributeError(
                "Real robot controller does not expose reset_xhand_position()."
            )
        controller.reset_xhand_position(xhand_q.copy(), num_steps=1)
        controller.franka_follow_traj([franka_q.copy()], max_steps_repeat=1)
        controller.reset_xhand_position(xhand_q.copy(), num_steps=1)

    def _paired_joint_waypoints(
        self,
        *,
        current_franka_q: np.ndarray,
        target_franka_q: np.ndarray,
        current_xhand_q: np.ndarray,
        target_xhand_q: np.ndarray,
        max_step_rad: float,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        current_franka_q = np.asarray(current_franka_q, dtype=np.float32).reshape(-1)
        target_franka_q = np.asarray(target_franka_q, dtype=np.float32).reshape(-1)
        current_xhand_q = np.asarray(current_xhand_q, dtype=np.float32).reshape(-1)
        target_xhand_q = np.asarray(target_xhand_q, dtype=np.float32).reshape(-1)
        if current_franka_q.shape != target_franka_q.shape:
            raise ValueError(
                "Expected current_franka_q and target_franka_q to match, got "
                f"{current_franka_q.shape} and {target_franka_q.shape}."
            )
        if current_xhand_q.shape != target_xhand_q.shape:
            raise ValueError(
                "Expected current_xhand_q and target_xhand_q to match, got "
                f"{current_xhand_q.shape} and {target_xhand_q.shape}."
            )
        if max_step_rad <= 0.0:
            raise ValueError(f"Expected max_step_rad > 0, got {max_step_rad}.")
        franka_delta = target_franka_q - current_franka_q
        xhand_delta = target_xhand_q - current_xhand_q
        max_delta = max(
            float(np.max(np.abs(franka_delta))),
            float(np.max(np.abs(xhand_delta))),
        )
        num_steps = max(1, int(np.ceil(max_delta / max_step_rad)))
        return [
            (
                current_franka_q + franka_delta * np.float32(step_idx / num_steps),
                current_xhand_q + xhand_delta * np.float32(step_idx / num_steps),
            )
            for step_idx in range(1, num_steps + 1)
        ]

    def _position_control_loop(self) -> None:
        assert self._control_stop_event is not None
        while not self._control_stop_event.is_set():
            loop_start = time.time()
            try:
                self.send_latest_position_targets()
            except Exception:
                pass
            elapsed = time.time() - loop_start
            time.sleep(max(self.control_period_s - elapsed, 0.0))

    def _command_smoothing_alpha(self, elapsed_s: float) -> float:
        if self.control_freq_hz <= 0.0:
            raise ValueError(
                f"Expected control_freq_hz > 0, got {self.control_freq_hz}."
            )
        progress = float(self.control_target_progress_per_update)
        if not 0.0 < progress <= 1.0:
            raise ValueError(
                "Expected control_target_progress_per_update in (0, 1], got "
                f"{progress}."
            )
        if elapsed_s <= 0.0:
            return 0.0
        policy_period_s = 1.0 / self.control_freq_hz
        elapsed_s = min(elapsed_s, policy_period_s)
        if progress == 1.0:
            return 1.0
        remaining_fraction = (1.0 - progress) ** (elapsed_s / policy_period_s)
        return float(np.clip(1.0 - remaining_fraction, 0.0, 1.0))

    def _blend_joint_targets(
        self,
        current_q: np.ndarray,
        target_q: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        current_q = np.asarray(current_q, dtype=np.float32).reshape(-1)
        target_q = np.asarray(target_q, dtype=np.float32).reshape(-1)
        if current_q.shape != target_q.shape:
            raise ValueError(
                f"Expected current and target joint arrays to match, got {current_q.shape} and {target_q.shape}."
            )
        return current_q + np.float32(alpha) * (target_q - current_q)

    def _limit_joint_step(
        self,
        current_q: np.ndarray,
        target_q: np.ndarray,
        max_step_rad: float,
    ) -> np.ndarray:
        current_q = np.asarray(current_q, dtype=np.float32).reshape(-1)
        target_q = np.asarray(target_q, dtype=np.float32).reshape(-1)
        if current_q.shape != target_q.shape:
            raise ValueError(
                f"Expected current and target joint arrays to match, got {current_q.shape} and {target_q.shape}."
            )
        if max_step_rad <= 0.0:
            raise ValueError(f"Expected max_step_rad > 0, got {max_step_rad}.")
        delta_q = target_q - current_q
        max_delta = float(np.max(np.abs(delta_q)))
        if max_delta <= max_step_rad:
            return target_q.copy()
        return current_q + delta_q * np.float32(max_step_rad / max_delta)
