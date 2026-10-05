"""Real backend layering.

PhysicsReal is the public env-facing backend, analogous to PhysicsSim.

- RealEntity / RealRobotEntity provide the underlying URDF-backed entity objects.
- RealRobotEntity owns the live hardware interfaces, polling threads, and control threads.
- PhysicsReal owns the per-step state interface snapshot used by env computations so
  multiple state reads within one env step observe a consistent robot state.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from dexlink.src.retargeting.data.demo_loader import is_oakink_dataset
from dexlink.utils.retargeting.path_utils import get_logs_dir
from dexlink.utils.retargeting.vis_server import VizServer
from dexlink.src.rl.world.art_object import ArtObject, get_object_cfg
from dexlink.src.rl.world.real_utils import AssetEntity, RealEntity, RealRobotEntity
from dexlink.src.rl.world.real_utils.physics_real_ik import PhysicsRealIKMixin
from dexlink.src.rl.world.real_utils.physics_real_pointcloud import (
    PhysicsRealPointCloudMixin,
)
from dexlink.src.rl.world.real_utils.physics_real_recording import (
    PhysicsRealRecordingMixin,
)
from dexlink.src.rl.world.real_utils.physics_real_visualization import (
    PhysicsRealVisualizationMixin,
)
from dexlink.src.rl.world.real_utils.robot_entity import FRANKA_JOINT_NAMES
from dexlink.src.rl.world.real_utils.vision import StaticSceneCalibration
from dexlink.src.rl.world.robot import Hand, get_hand_specific_cfg

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import RealCfg
    from dexlink.src.rl.world.env import Env


class PhysicsReal(
    PhysicsRealIKMixin,
    PhysicsRealVisualizationMixin,
    PhysicsRealPointCloudMixin,
    PhysicsRealRecordingMixin,
):
    """Public real-world physics backend used by the env and manager classes."""

    def __init__(self, real_cfg: RealCfg, env: Env):
        self.real_cfg = real_cfg
        self.env = env
        video_save_dir = Path(self.real_cfg.video_save_dir)
        if video_save_dir.is_absolute():
            self.logs_dir = video_save_dir
        else:
            self.logs_dir = get_logs_dir() / video_save_dir
        self.record = False
        self.is_recording = False
        self.scene = None
        self.cam = None
        self.camera_lookat = (0.0, 0.0, 0.0)
        self.scene_entities: dict[str, object] = {}
        self.hand_sensors = None
        self.object_cfg = None
        self.hand_cfgs = None
        self.setup_viz: VizServer | None = None
        self.runtime_viz: VizServer | None = None
        self._runtime_viz_is_shared = False
        self._state_interface: dict[int, dict[str, object]] = {}
        self._world_calibration: StaticSceneCalibration | None = None
        self._tactile_reference_by_sensor: dict[str, np.ndarray] | None = None
        self._episode_initial_object_pcd_world: np.ndarray | None = None
        self._episode_initial_table_pcd_world: np.ndarray | None = None
        self._lift_start_object_pcd_world: np.ndarray | None = None
        self._lift_start_wrist_pose_world: np.ndarray | None = None
        self._recorded_rgb_frames: list[np.ndarray] = []
        self._recording_last_frame_time: float | None = None
        self._recording_frame_interval_s = float(
            self.real_cfg.recording_frame_interval_s
        )
        self._ik_scene = None
        self._ik_entity = None
        self._ik_entity_n_qs: int | None = None
        self._calibration_motion_viz_context: dict[str, object] | None = None

    def run_checks(self):
        # check the task's hand is franka_xhand
        return None

    def run_imports(self):
        return None

    def build_scene(self) -> None:
        assert self.scene is not None
        task_manager = self.env.task_manager

        self.scene_entities.clear()
        self.hand_sensors = None
        self.hand_cfgs = None
        self.object_cfg = None
        self.scene_entities["plane"] = self._add_plane(pos=(0, 0, -0.05))

        if self.env.task_cfg.demo_dataset == "oakink_lift":
            support_height = float(self.env.task_cfg.lift_support_height)
            support_surface = {
                "type": "rough",
                "roughness": 0.8,
                "color": (0.58, 0.50, 0.42, 1.0),
            }
            self.scene_entities["platform"] = self._add_box(
                pos=(0.0, 0.0, support_height / 2.0),
                size=(0.8, 1.2, support_height),
                fixed=True,
                collision=True,
                surface=support_surface,
            )
            self.scene_entities["block"] = self._add_box(
                pos=(0.0, 0.0, support_height + 0.05),
                size=(0.1, 0.1, 0.1),
                fixed=True,
                collision=True,
                surface={
                    "type": "rough",
                    "roughness": 0.9,
                    "color": (0.8, 0.8, 0.8, 1.0),
                },
            )

        if not is_oakink_dataset(self.env.task_cfg.demo_dataset):
            cardbox_size = (0.1, 0.2, 0.1)
            if "notebook" in self.env.task_cfg.target_object:
                print("Adding a SMALLER cardboard box for notebook")
                cardbox_size = (0.1, 0.15, 0.1)

            cardbox_surface = {
                "type": "rough",
                "roughness": 0.6,
                "color": (167 / 255.0, 134 / 255.0, 103 / 255.0, 1.0),
            }
            self.scene_entities["platform"] = self._add_box(
                pos=(0, -0.08, 0.90),
                size=cardbox_size,
                fixed=True,
                collision=True,
                surface=cardbox_surface,
            )

        self.object_cfg = get_object_cfg(
            reference_demo=None,
            urdf_path=task_manager.original_demo.target_obj_urdf_path,
            obj_name=task_manager.original_demo.target_obj_name,
            object_fixed=self.env.task_cfg.object_fixed,
        )

        self.scene_entities["object"] = ArtObject(
            physics=self,
            obj_cfg=self.object_cfg,
            num_envs=self.env.num_envs,
        )

        if len(self.env.task_manager.active_hand_sides) != 1:
            raise ValueError(
                "PhysicsReal currently expects exactly one active hand side. "
                f"Got {self.env.task_manager.active_hand_sides}."
            )

        self.hand_cfgs = get_hand_specific_cfg(self.env.task_cfg.hand_name)
        robot_base_pose = getattr(task_manager.original_demo, "robot_base_pose", None)
        if self.env.task_manager.has_hand("left"):
            self.scene_entities["left"] = Hand(
                physics=self,
                robo_cfg=self.hand_cfgs["left"],
                num_envs=self.env.num_envs,
                device=self.env.device,
                obs_cfg=self.env.cfg.obs_cfg.hand_cfg,
                env=self.env,
                name="left",
                base_pose=robot_base_pose,
            )
            self._initialize_real_hand_entity("left")
        if self.env.task_manager.has_hand("right"):
            self.scene_entities["right"] = Hand(
                physics=self,
                robo_cfg=self.hand_cfgs["right"],
                num_envs=self.env.num_envs,
                device=self.env.device,
                obs_cfg=self.env.cfg.obs_cfg.hand_cfg,
                env=self.env,
                name="right",
                base_pose=robot_base_pose,
            )
            self._initialize_ik_backend(self.scene_entities["right"])
            self._initialize_real_hand_entity("right")

        if (
            not self.real_cfg.visualization_only
            and self.real_cfg.initialize_robot
            and self.real_cfg.read_robot_state_on_initialize
        ):
            self._reload_state_interface(force_update=True)
        if not self.real_cfg.visualization_only:
            task_manager.process_retargeted_motion()

    def _initialize_real_hand_entity(self, side: str) -> None:
        hand = self.scene_entities[side]
        if not isinstance(hand, Hand):
            raise TypeError(
                f"Expected scene_entities[{side!r}] to be a Hand, got {type(hand).__name__}."
            )
        if not isinstance(hand.robot, RealEntity):
            raise TypeError(
                f"Expected hand.robot to be a RealEntity, got {type(hand.robot).__name__}."
            )
        robot_entity = RealRobotEntity.from_entity(hand.robot, side=side)
        robot_entity.set_runtime_log_dir(self.logs_dir)
        robot_entity.initialize_real_interfaces(self.real_cfg)
        if (
            self.real_cfg.reset_to_default_on_initialize
            and robot_entity.robot_controller is not None
        ):
            robot_entity.reset_default_position()
            time.sleep(float(self.real_cfg.world_calibration_settle_time_s))
        robot_entity.set_camera_preview_dir(self.logs_dir)
        hand.robot = robot_entity
        if robot_entity.camera is not None:
            self.cam = robot_entity.camera
        if robot_entity.robot_controller is not None:
            self._reload_entity_state_interface(robot_entity, force_update=True)

    def _clear_state_interface(self) -> None:
        self._state_interface.clear()

    def _reload_entity_state_interface(
        self, entity: RealRobotEntity, *, force_update: bool
    ) -> None:
        self._state_interface[id(entity)] = entity.latest_state_snapshot(
            force_update=force_update
        )

    def _reload_state_interface(self, *, force_update: bool) -> None:
        self._clear_state_interface()
        for scene_entity in self.scene_entities.values():
            if isinstance(scene_entity, Hand) and isinstance(
                scene_entity.robot, RealRobotEntity
            ):
                self._reload_entity_state_interface(
                    scene_entity.robot, force_update=force_update
                )

    def _entity_state_snapshot(self, entity: RealRobotEntity) -> dict[str, object]:
        snapshot = self._state_interface.get(id(entity))
        if snapshot is None:
            self._reload_entity_state_interface(entity, force_update=False)
            snapshot = self._state_interface.get(id(entity))
        if snapshot is None:
            raise RuntimeError(
                f"No cached state snapshot is available for entity {entity.name!r}."
            )
        return snapshot

    def _entity_control_q_from_snapshot(
        self, entity: RealRobotEntity
    ) -> tuple[np.ndarray, np.ndarray]:
        snapshot = self._entity_state_snapshot(entity)
        robot_state = snapshot["robot_state"]
        franka_q = np.asarray(robot_state["franka"]["q"], dtype=np.float32).reshape(-1)
        xhand_q = np.asarray(robot_state["xhand"]["q"], dtype=np.float32).reshape(-1)
        expected_franka_shape = (len(FRANKA_JOINT_NAMES),)
        expected_xhand_shape = (len(entity.xhand_joint_names),)
        if franka_q.shape != expected_franka_shape:
            raise ValueError(
                f"Expected Franka state shape {expected_franka_shape}, got {franka_q.shape}."
            )
        if xhand_q.shape != expected_xhand_shape:
            raise ValueError(
                f"Expected XHand state shape {expected_xhand_shape}, got {xhand_q.shape}."
            )
        return franka_q, xhand_q

    def _default_entity_dof_targets_from_controller(
        self, entity: RealRobotEntity
    ) -> tuple[torch.Tensor, list[int]]:
        controller = entity.robot_controller
        if controller is None:
            raise RuntimeError(
                "Cannot fetch default joint targets because robot_controller is not initialized."
            )
        if not hasattr(controller, "default_franka_joint_state"):
            raise AttributeError(
                "Robot controller does not expose default_franka_joint_state."
            )
        if not hasattr(controller, "default_hand_joint_state"):
            raise AttributeError(
                "Robot controller does not expose default_hand_joint_state."
            )
        default_franka_state: Any = getattr(controller, "default_franka_joint_state")
        default_hand_state: Any = getattr(controller, "default_hand_joint_state")

        default_franka_q = np.asarray(default_franka_state, dtype=np.float32).reshape(
            -1
        )
        default_xhand_q = np.asarray(default_hand_state, dtype=np.float32).reshape(-1)

        expected_franka_shape = (len(FRANKA_JOINT_NAMES),)
        if default_franka_q.shape != expected_franka_shape:
            raise ValueError(
                "Franka default joint target shape mismatch: expected "
                f"{expected_franka_shape}, got {default_franka_q.shape}."
            )
        expected_xhand_shape = (len(entity.xhand_joint_names),)
        if default_xhand_q.shape != expected_xhand_shape:
            raise ValueError(
                "XHand default joint target shape mismatch: expected "
                f"{expected_xhand_shape}, got {default_xhand_q.shape}."
            )
        dof_indices: list[int] = []
        dof_values: list[float] = []
        joint_targets = {
            **{
                joint_name: float(default_franka_q[idx])
                for idx, joint_name in enumerate(FRANKA_JOINT_NAMES)
            },
            **{
                joint_name: float(default_xhand_q[idx])
                for idx, joint_name in enumerate(entity.xhand_joint_names)
            },
        }
        for joint_name, target_value in joint_targets.items():
            dofs_idx_local = entity.joint_dofs_idx_local.get(joint_name)
            if dofs_idx_local is None:
                raise KeyError(
                    f"Missing local DOF mapping for joint {joint_name!r} on entity {entity.name!r}."
                )
            if len(dofs_idx_local) != 1:
                raise ValueError(
                    f"Expected joint {joint_name!r} to map to exactly one local DOF, got {dofs_idx_local}."
                )
            dof_indices.append(int(dofs_idx_local[0]))
            dof_values.append(target_value)

        default_targets = torch.tensor(
            [dof_values], dtype=torch.float32, device=self.env.device
        )
        return default_targets, dof_indices

    def _hand_for_entity(self, entity: RealRobotEntity) -> Hand:
        for scene_entity in self.scene_entities.values():
            if isinstance(scene_entity, Hand) and scene_entity.robot is entity:
                return scene_entity
        raise KeyError(
            f"Failed to find Hand wrapper for entity {entity.name!r} in scene_entities."
        )

    def _target_world_pose_to_franka_base(
        self, entity: RealRobotEntity, target_world_pose: np.ndarray
    ) -> np.ndarray:
        hand = self._hand_for_entity(entity)
        base_pose_world = np.asarray(hand.base_pose, dtype=np.float32)
        if base_pose_world.shape != (4, 4):
            raise ValueError(
                f"Expected hand.base_pose shape (4, 4), got {base_pose_world.shape}."
            )
        if target_world_pose.shape != (4, 4):
            raise ValueError(
                f"Expected target_world_pose shape (4, 4), got {target_world_pose.shape}."
            )
        return np.linalg.inv(base_pose_world) @ target_world_pose

    def _active_real_hand(self) -> Hand:
        active_sides = self.env.task_manager.active_hand_sides
        if len(active_sides) != 1:
            raise ValueError(
                "PhysicsReal expects exactly one active hand side, "
                f"got {active_sides}."
            )
        scene_entity = self.scene_entities[active_sides[0]]
        if not isinstance(scene_entity, Hand):
            raise TypeError(
                f"Expected scene entity {active_sides[0]!r} to be Hand, got {type(scene_entity).__name__}."
            )
        return scene_entity

    def _active_real_robot_entity(self) -> RealRobotEntity:
        hand = self._active_real_hand()
        if not isinstance(hand.robot, RealRobotEntity):
            raise TypeError(
                f"Expected active hand robot to be RealRobotEntity, got {type(hand.robot).__name__}."
            )
        return hand.robot

    def _tactile_sensor_name_by_link(self, link_name: str) -> str:
        side_prefix = f"{self._active_real_hand().name}_"
        suffix = link_name
        if suffix.startswith(side_prefix):
            suffix = suffix[len(side_prefix) :]
        if "thumb" in suffix:
            return "thumb"
        if "index" in suffix:
            return "index"
        if "mid" in suffix:
            return "middle"
        if "ring" in suffix:
            return "ring"
        if "pinky" in suffix:
            return "pinky"
        raise ValueError(
            f"Failed to map tactile link {link_name!r} to a tactile sensor name."
        )

    def _capture_tactile_reference(self) -> None:
        hand = self._active_real_hand()
        if not isinstance(hand.robot, RealRobotEntity):
            raise TypeError(
                f"Expected hand.robot to be RealRobotEntity, got {type(hand.robot).__name__}."
            )
        self._reload_entity_state_interface(hand.robot, force_update=True)
        snapshot = self._entity_state_snapshot(hand.robot)
        robot_state = snapshot["robot_state"]
        if not isinstance(robot_state, dict):
            raise TypeError(
                f"Expected robot_state snapshot to be a dict, got {type(robot_state).__name__}."
            )
        xhand_state = robot_state.get("xhand")
        if not isinstance(xhand_state, dict):
            raise TypeError(
                f"Expected xhand state to be a dict, got {type(xhand_state).__name__}."
            )
        tactile_state = xhand_state.get("tactile")
        if not isinstance(tactile_state, dict):
            raise TypeError(
                f"Expected tactile state to be a dict, got {type(tactile_state).__name__}."
            )
        tactile_reference_by_sensor: dict[str, np.ndarray] = {}
        for link_name in hand.tactile_link_names:
            sensor_name = self._tactile_sensor_name_by_link(link_name)
            sensor_state = tactile_state.get(sensor_name)
            if sensor_state is None:
                raise KeyError(
                    f"Missing tactile sensor {sensor_name!r}. Available sensors: {sorted(tactile_state)}."
                )
            calc_force = np.asarray(
                sensor_state["calc_force"], dtype=np.float32
            ).reshape(-1)
            if calc_force.shape != (3,):
                raise ValueError(
                    f"Expected tactile calc_force shape (3,), got {calc_force.shape} for sensor {sensor_name!r}."
                )
            tactile_reference_by_sensor[sensor_name] = calc_force.copy()
        self._tactile_reference_by_sensor = tactile_reference_by_sensor

    def _add_plane(self, pos: tuple[float, float, float]) -> AssetEntity:
        return AssetEntity(
            name="plane",
            kind="plane",
            pos=pos,
            fixed=True,
            collision=False,
        )

    def _add_box(
        self,
        pos: tuple[float, float, float],
        size: tuple[float, float, float],
        fixed: bool,
        collision: bool,
        surface,
    ) -> AssetEntity:
        return AssetEntity(
            name=None,
            kind="box",
            pos=pos,
            size=size,
            fixed=fixed,
            collision=collision,
            metadata={"surface": surface},
        )

    def prepare_episode_scene(self) -> None:
        self._episode_initial_object_pcd_world = None
        self._episode_initial_table_pcd_world = None
        self._lift_start_object_pcd_world = None
        self._lift_start_wrist_pose_world = None
        if self.real_cfg.initialize_robot:
            input(
                "Clear the table of movable objects for this episode, then press Enter to capture the no-contact tactile reference..."
            )
            self._capture_tactile_reference()
        else:
            self._tactile_reference_by_sensor = None
            print(
                "Skipping tactile reference capture because env.real_cfg.initialize_robot=False."
            )
        input("Place the target object in the scene, then press Enter to continue...")

    def _not_implemented(self, method_name: str) -> None:
        raise NotImplementedError(
            f"PhysicsReal.{method_name}() is a template stub. Connect it to the real robot/control API."
        )

    def _resolve_real_joint_targets(
        self,
        entity: RealRobotEntity,
        values,
        dofs_idx_local,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        jvals = torch.as_tensor(values, dtype=torch.float32)
        if jvals.ndim != 2:
            raise ValueError(
                f"Expected joint position tensor with shape (1, num_dofs), got {tuple(jvals.shape)}."
            )
        if jvals.shape[0] != 1:
            raise ValueError(
                f"PhysicsReal only supports a single real environment, got batch size {jvals.shape[0]}."
            )

        dofs_idx_local = np.asarray(dofs_idx_local, dtype=np.int64).reshape(-1)
        if jvals.shape[1] != dofs_idx_local.shape[0]:
            raise ValueError(
                f"Mismatch between provided joint values shape {tuple(jvals.shape)} and dofs_idx_local length {dofs_idx_local.shape[0]}."
            )

        dof_idx_to_joint_name = entity.dof_idx_to_joint_name()
        franka_q, xhand_q = self._entity_control_q_from_snapshot(entity)
        franka_joint_to_idx = {
            joint_name: idx for idx, joint_name in enumerate(FRANKA_JOINT_NAMES)
        }
        xhand_joint_to_idx = {
            joint_name: idx for idx, joint_name in enumerate(entity.xhand_joint_names)
        }
        helper_root_joint_names = set(entity.helper_root_joint_names)

        update_franka = False
        update_xhand = False
        for local_col, dof_idx in enumerate(dofs_idx_local.tolist()):
            joint_name = dof_idx_to_joint_name.get(dof_idx)
            if joint_name is None:
                raise KeyError(
                    f"Unknown local DOF index {dof_idx}. Available DOF indices: {sorted(dof_idx_to_joint_name)}."
                )
            joint_value = float(jvals[0, local_col].item())
            if joint_name in franka_joint_to_idx:
                franka_q[franka_joint_to_idx[joint_name]] = joint_value
                update_franka = True
                continue
            if joint_name in xhand_joint_to_idx:
                xhand_q[xhand_joint_to_idx[joint_name]] = joint_value
                update_xhand = True
                continue
            if joint_name in helper_root_joint_names:
                continue
            raise NotImplementedError(
                "PhysicsReal encountered a non-physical URDF joint "
                f"{joint_name!r} that is not mapped to the Franka or XHand control API."
            )
        return (
            franka_q if update_franka else None,
            xhand_q if update_xhand else None,
        )

    def _eef_pose_from_target(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> np.ndarray:
        pos = torch.as_tensor(target_pos, dtype=torch.float32)
        quat = torch.as_tensor(target_quat, dtype=torch.float32)
        if tuple(pos.shape) != (1, 3):
            raise ValueError(
                f"Expected target_pos shape (1, 3), got {tuple(pos.shape)}."
            )
        if tuple(quat.shape) != (1, 4):
            raise ValueError(
                f"Expected target_quat shape (1, 4), got {tuple(quat.shape)}."
            )
        quat_wxyz = quat[0].detach().cpu().numpy().astype(np.float32)
        return self._pose_from_wxyz(
            tuple(pos[0].detach().cpu().numpy().tolist()),
            tuple(quat_wxyz.tolist()),
        )

    # Required by: world/env.py
    def initialize(self, camera_lookat: tuple[float, float, float]) -> None:
        if self.env.num_envs != 1:
            raise ValueError(
                f"PhysicsReal expects env.num_envs == 1, got {self.env.num_envs}."
            )
        self.camera_lookat = camera_lookat
        self.run_imports()
        self.run_checks()
        self.scene = {"backend": "real"}
        self.build_scene()
        if self.real_cfg.visualization_only:
            print(
                "PhysicsReal visualization-only mode is enabled. "
                "Skipping robot-state initialization, camera-world calibration, and rollout setup."
            )
            return
        self._initialize_world_calibration()

    # Required by: world/reset_manager.py
    def object_domain_randomization_summary(self) -> dict[str, list[str]]:
        pass
        # self._not_implemented("object_domain_randomization_summary")

    # Required by: world/reset_manager.py
    def hand_domain_randomization_summary(self, side: str) -> dict[str, list[str]]:
        pass
        # self._not_implemented("hand_domain_randomization_summary")

    # Required by: world/reset_manager.py
    def apply_domain_randomization(self, reset_cfg, env_ids: np.ndarray) -> None:
        pass
        # self._not_implemented("apply_domain_randomization")

    # Required by: world/robot.py, world/art_object.py
    def add_rigid_urdf_entity(
        self,
        urdf_path: str,
        *,
        fixed: bool,
        collision: bool,
        convexify: bool,
        recompute_inertia: bool = False,
        pos: tuple[float, float, float] = (0, 0, 0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        name: str | None = None,
        links_to_keep: list[str] | None = None,
        rho: float | None = None,
        friction: float = 1.2,
        gravity_compensation: float | None = None,
    ) -> RealEntity:
        return RealEntity.from_urdf(
            urdf_path=urdf_path,
            name=name,
            fixed=fixed,
            collision=collision,
            convexify=convexify,
            recompute_inertia=recompute_inertia,
            pos=pos,
            quat=quat,
            links_to_keep=links_to_keep,
            rho=rho,
            friction=friction,
            gravity_compensation=gravity_compensation,
        )

    # Required by: world/robot.py, world/art_object.py
    def get_joint_dofs_idx_local(
        self, entity: RealEntity, joint_name: str
    ) -> list[int]:
        return entity.get_joint_dofs_idx_local(joint_name)

    # Required by: world/robot.py, world/art_object.py
    def get_link_idx_local(self, entity: RealEntity, link_name: str) -> int:
        return entity.get_link_idx_local(link_name)

    # Required by: world/robot.py
    def get_link_idx(self, entity: RealEntity, link_name: str) -> int:
        return entity.get_link_idx(link_name)

    # Required by: world/art_object.py
    def set_entity_pos(self, entity, pos, envs_idx=None) -> None:
        entity_name = getattr(entity, "name", None)
        if entity_name is None:
            entity_name = getattr(entity, "urdf_path", None)
        if entity_name is None:
            entity_name = getattr(entity, "kind", type(entity).__name__)
        input(
            f"Ideally set entity {entity_name!r} position to {pos} now. Press Enter to continue..."
        )

    # Required by: world/art_object.py
    def set_entity_quat(self, entity, quat, envs_idx=None) -> None:
        entity_name = getattr(entity, "name", None)
        if entity_name is None:
            entity_name = getattr(entity, "urdf_path", None)
        if entity_name is None:
            entity_name = getattr(entity, "kind", type(entity).__name__)
        input(
            f"Ideally set entity {entity_name!r} quaternion to {quat} now. Press Enter to continue..."
        )

    # Required by: world/robot.py, world/art_object.py
    def set_entity_dofs_position(
        self, entity, values, dofs_idx_local, envs_idx=None
    ) -> None:
        if not isinstance(entity, RealRobotEntity):
            raise NotImplementedError(
                "PhysicsReal.set_entity_dofs_position() is only implemented for RealRobotEntity."
            )
        if entity.robot_controller is None:
            raise RuntimeError(
                "Cannot set real robot joint positions because robot_controller is not initialized."
            )

        if envs_idx is not None:
            envs_idx = np.asarray(envs_idx, dtype=np.int64)
            if envs_idx.shape != (1,) or int(envs_idx[0]) != 0:
                raise ValueError(
                    f"PhysicsReal only supports envs_idx=[0] or None, got {envs_idx.tolist()}."
                )

        franka_q, xhand_q = self._resolve_real_joint_targets(
            entity,
            values,
            dofs_idx_local,
        )
        if not self.real_cfg.bypass_low_level_control:
            self._smooth_blocking_move_to_joint_targets(
                entity=entity,
                franka_q=franka_q,
                xhand_q=xhand_q,
            )
        entity.update_position_targets(
            franka_q=franka_q,
            xhand_q=xhand_q,
        )
        self._reload_entity_state_interface(entity, force_update=True)

    def _real_move_interpolation_num_steps(
        self,
        *,
        start_franka_q: np.ndarray,
        target_franka_q: np.ndarray,
        start_xhand_q: np.ndarray,
        target_xhand_q: np.ndarray,
    ) -> int:
        max_step_deg = float(self.real_cfg.adaptive_decimation_joint_step_deg)
        if max_step_deg <= 0.0:
            raise ValueError(
                "Expected real_cfg.adaptive_decimation_joint_step_deg > 0, got "
                f"{max_step_deg}."
            )
        start_franka_q = np.asarray(start_franka_q, dtype=np.float32).reshape(-1)
        target_franka_q = np.asarray(target_franka_q, dtype=np.float32).reshape(-1)
        start_xhand_q = np.asarray(start_xhand_q, dtype=np.float32).reshape(-1)
        target_xhand_q = np.asarray(target_xhand_q, dtype=np.float32).reshape(-1)
        all_delta_rad = np.concatenate(
            [
                np.abs(target_franka_q - start_franka_q),
                np.abs(target_xhand_q - start_xhand_q),
            ]
        )
        max_delta_rad = float(np.max(all_delta_rad))
        step_rad = np.deg2rad(max_step_deg)
        num_steps = max(1, int(np.ceil(max_delta_rad / step_rad)))
        if self._calibration_motion_viz_context is not None:
            total_delta_rad = float(np.sum(all_delta_rad))
            num_steps = max(num_steps, int(np.ceil(total_delta_rad / step_rad)))
        return num_steps

    def _publish_real_joint_targets_once(
        self,
        *,
        entity: RealRobotEntity,
        franka_q: np.ndarray,
        xhand_q: np.ndarray,
        waypoint_idx: int | None = None,
        total_waypoints: int | None = None,
    ) -> None:
        if self._calibration_motion_viz_context is not None:
            if waypoint_idx is None or total_waypoints is None:
                raise ValueError(
                    "Calibration command visualization requires waypoint_idx and total_waypoints."
                )
            context = self._calibration_motion_viz_context
            self._visualize_world_calibration_command(
                hand=context["hand"],
                franka_q=franka_q,
                xhand_q=xhand_q,
                base_pose_cam_guess=context["base_pose_cam_guess"],
                frame_idx=int(context["frame_idx"]),
                candidate_idx=int(context["candidate_idx"]),
                waypoint_idx=waypoint_idx,
                total_waypoints=total_waypoints,
            )
            self.pause_if_visualization_active(
                "Showing calibration command preview before sending to the robot."
            )
        entity.update_position_targets(
            franka_q=franka_q,
            xhand_q=xhand_q,
        )
        if not entity.use_threaded_control:
            entity.send_latest_position_targets()

    def _smooth_blocking_move_to_joint_targets(
        self,
        *,
        entity: RealRobotEntity,
        franka_q: np.ndarray,
        xhand_q: np.ndarray,
    ) -> None:
        start_franka_q, start_xhand_q = self._entity_control_q_from_snapshot(entity)
        interpolation_steps = self._real_move_interpolation_num_steps(
            start_franka_q=start_franka_q,
            target_franka_q=franka_q,
            start_xhand_q=start_xhand_q,
            target_xhand_q=xhand_q,
        )
        if self._calibration_motion_viz_context is not None:
            context = self._calibration_motion_viz_context
            print(
                "[WorldCalibration] "
                f"streaming {interpolation_steps} waypoint(s) for "
                f"frame_idx={int(context['frame_idx'])} candidate_idx={int(context['candidate_idx'])}."
            )
        calibration_control_freq_hz = float(self.real_cfg.calibration_control_freq_hz)
        if calibration_control_freq_hz <= 0.0:
            raise ValueError(
                "Expected real_cfg.calibration_control_freq_hz > 0, got "
                f"{calibration_control_freq_hz}."
            )
        command_period_s = 1.0 / calibration_control_freq_hz
        total_waypoints = interpolation_steps
        for step_idx in range(max(0, interpolation_steps - 1)):
            alpha = float(step_idx + 1) / float(interpolation_steps)
            blended_franka_q = start_franka_q + alpha * (franka_q - start_franka_q)
            blended_xhand_q = start_xhand_q + alpha * (xhand_q - start_xhand_q)
            self._publish_real_joint_targets_once(
                entity=entity,
                franka_q=blended_franka_q,
                xhand_q=blended_xhand_q,
                waypoint_idx=step_idx + 1,
                total_waypoints=total_waypoints,
            )
            time.sleep(command_period_s)
        if self._calibration_motion_viz_context is not None:
            context = self._calibration_motion_viz_context
            self._visualize_world_calibration_command(
                hand=context["hand"],
                franka_q=franka_q,
                xhand_q=xhand_q,
                base_pose_cam_guess=context["base_pose_cam_guess"],
                frame_idx=int(context["frame_idx"]),
                candidate_idx=int(context["candidate_idx"]),
                waypoint_idx=total_waypoints,
                total_waypoints=total_waypoints,
            )
            self.pause_if_visualization_active(
                "Showing final calibration command preview before sending to the robot."
            )
        self._stream_real_joint_targets_until_reached(
            entity=entity,
            franka_q=franka_q,
            xhand_q=xhand_q,
        )

    def _stream_real_joint_targets_until_reached(
        self,
        *,
        entity: RealRobotEntity,
        franka_q: np.ndarray,
        xhand_q: np.ndarray,
    ) -> None:
        franka_target = np.asarray(franka_q, dtype=np.float32).reshape(-1)
        xhand_target = np.asarray(xhand_q, dtype=np.float32).reshape(-1)
        franka_tol = float(self.real_cfg.reset_motion_franka_tol_rad)
        xhand_tol = float(self.real_cfg.reset_motion_xhand_tol_rad)
        entity.send_blocking_joint_targets(
            franka_q=franka_target,
            xhand_q=xhand_target,
        )
        self._reload_entity_state_interface(entity, force_update=True)
        current_franka_q, current_xhand_q = self._entity_control_q_from_snapshot(entity)
        franka_error = float(np.max(np.abs(current_franka_q - franka_target)))
        xhand_error = float(np.max(np.abs(current_xhand_q - xhand_target)))
        if franka_error > franka_tol or xhand_error > xhand_tol:
            raise RuntimeError(
                "Failed to reach real joint targets after control-base reset: "
                f"franka_error={franka_error}, xhand_error={xhand_error}, "
                f"franka_tol={franka_tol}, xhand_tol={xhand_tol}."
            )

    # Required by: world/robot.py, world/art_object.py
    def set_entity_dofs_velocity(
        self, entity, values, dofs_idx_local, envs_idx=None
    ) -> None:
        return

    # Required by: world/art_object.py
    def get_entity_pos(self, entity):
        # this should not be called during real world execution. the path to this function should be blocked.
        # return dummy value
        return torch.zeros((1, 3), dtype=torch.float32, device=self.env.device)

        # self._not_implemented("get_entity_pos")

    # Required by: world/art_object.py
    def get_entity_quat(self, entity):
        return torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.env.device
        )
        # this should not be called during real world execution. the path to this function should be blocked.
        self._not_implemented("get_entity_quat")

    # Required by: world/art_object.py
    def get_entity_vel(self, entity):
        # this should not be called during real world execution. the path to this function should be blocked.
        # return dummy value
        return torch.zeros((1, 6), dtype=torch.float32, device=self.env.device)
        self._not_implemented("get_entity_vel")

    # Required by: world/art_object.py
    def get_entity_ang(self, entity):
        # this should not be called during real world execution. the path to this function should be blocked.
        self._not_implemented("get_entity_ang")

    # Required by: world/robot.py, world/art_object.py
    def get_entity_dofs_position(self, entity, dofs_idx_local):
        if not isinstance(entity, RealRobotEntity):
            raise NotImplementedError(
                "PhysicsReal.get_entity_dofs_position() is only implemented for RealRobotEntity."
            )

        requested_dofs = np.asarray(dofs_idx_local, dtype=np.int64).reshape(-1)
        if requested_dofs.ndim != 1:
            raise ValueError(
                f"Expected dofs_idx_local to flatten to shape (num_dofs,), got {requested_dofs.shape}."
            )

        dof_idx_to_joint_name = entity.dof_idx_to_joint_name()
        franka_q, xhand_q = self._entity_control_q_from_snapshot(entity)
        franka_joint_to_idx = {
            joint_name: idx for idx, joint_name in enumerate(FRANKA_JOINT_NAMES)
        }
        xhand_joint_to_idx = {
            joint_name: idx for idx, joint_name in enumerate(entity.xhand_joint_names)
        }
        helper_root_joint_names = set(entity.helper_root_joint_names)

        result = np.zeros((1, requested_dofs.shape[0]), dtype=np.float32)
        for result_col, dof_idx in enumerate(requested_dofs.tolist()):
            joint_name = dof_idx_to_joint_name.get(dof_idx)
            if joint_name is None:
                raise KeyError(
                    f"Unknown local DOF index {dof_idx}. Available DOF indices: {sorted(dof_idx_to_joint_name)}."
                )
            if joint_name in franka_joint_to_idx:
                result[0, result_col] = franka_q[franka_joint_to_idx[joint_name]]
                continue
            if joint_name in xhand_joint_to_idx:
                result[0, result_col] = xhand_q[xhand_joint_to_idx[joint_name]]
                continue
            if joint_name in helper_root_joint_names:
                result[0, result_col] = 0.0
                continue
            raise NotImplementedError(
                "PhysicsReal.get_entity_dofs_position() encountered a non-physical URDF joint "
                f"{joint_name!r} that is not mapped to the Franka or XHand state."
            )

        return torch.from_numpy(result).to(self.env.device)

    # Required by: world/robot.py, world/art_object.py
    def get_entity_dofs_velocity(self, entity, dofs_idx_local):
        # this should not be called during real world execution. the path to this function should be blocked.
        self._not_implemented("get_entity_dofs_velocity")

    # Required by: world/robot.py
    def get_entity_dofs_acceleration(self, entity, dofs_idx_local):
        # this should not be called during real world execution. the path to this function should be blocked.
        self._not_implemented("get_entity_dofs_acceleration")

    # Required by: world/robot.py, world/art_object.py
    def get_entity_dofs_limit(self, entity, dofs_idx_local):
        if not isinstance(entity, RealEntity):
            raise TypeError(
                f"Expected RealEntity for get_entity_dofs_limit(), got {type(entity).__name__}."
            )
        requested_dofs = np.asarray(dofs_idx_local, dtype=np.int64).reshape(-1)
        if requested_dofs.ndim != 1:
            raise ValueError(
                f"Expected dofs_idx_local to flatten to shape (num_dofs,), got {requested_dofs.shape}."
            )
        lower, upper = entity.get_dofs_limit(requested_dofs.tolist())
        return (
            torch.tensor([lower], dtype=torch.float32, device="cpu"),
            torch.tensor([upper], dtype=torch.float32, device="cpu"),
        )

    # Required by: world/action_manager.py, world/art_object.py (blocked)
    def control_entity_dofs_position(
        self,
        entity,
        values,
        dofs_idx_local,
    ) -> None:
        if not isinstance(entity, RealRobotEntity):
            raise NotImplementedError(
                "PhysicsReal.control_entity_dofs_position() is only implemented for RealRobotEntity."
            )
        if entity.robot_controller is None:
            raise RuntimeError(
                "Cannot control real robot joint positions because robot_controller is not initialized."
            )
        franka_q, resolved_xhand_q = self._resolve_real_joint_targets(
            entity,
            values,
            dofs_idx_local,
        )
        if self.real_cfg.print_real_control_stages:
            print(
                "\033[94m[RealControl] physics_real: resolved joint targets for real controller\033[0m"
            )
        target_qpos = torch.as_tensor(values, dtype=torch.float32)
        xhand_q = resolved_xhand_q
        if self.real_cfg.pause_on_visualization_update:
            if self.real_cfg.print_real_control_stages:
                print(
                    "\033[94m[RealControl] physics_real: showing action visualization before target update\033[0m"
                )
            self.visualize_action_targets(
                entity,
                target_qpos=target_qpos,
                target_franka_q=franka_q,
            )
        if self.real_cfg.print_real_control_stages:
            print(
                "\033[94m[RealControl] physics_real: updating desired joint targets\033[0m"
            )
        entity.update_position_targets(
            franka_q=franka_q,
            xhand_q=xhand_q,
        )
        if self.real_cfg.bypass_low_level_control:
            if self.real_cfg.print_real_control_stages:
                print(
                    "\033[94m[RealControl] physics_real: bypass_low_level_control=True, skipping controller send\033[0m"
                )
            return
        if self.real_cfg.print_real_control_stages:
            print(
                "\033[94m[RealControl] physics_real: moving to target with control-base reset APIs\033[0m"
            )
        self._smooth_blocking_move_to_joint_targets(
            entity=entity,
            franka_q=franka_q,
            xhand_q=xhand_q,
        )
        self._reload_entity_state_interface(entity, force_update=True)

    # Required by: world/robot.py
    def get_entity_links_pos(self, entity, links_idx_local, ref: str):
        if not isinstance(entity, RealRobotEntity):
            raise NotImplementedError(
                "PhysicsReal.get_entity_links_pos() is only implemented for RealRobotEntity."
            )
        if ref != "link_origin":
            raise NotImplementedError(
                f"PhysicsReal.get_entity_links_pos() only supports ref='link_origin', got {ref!r}."
            )
        requested_links = np.asarray(links_idx_local, dtype=np.int64).reshape(-1)
        if requested_links.ndim != 1:
            raise ValueError(
                f"Expected links_idx_local to flatten to shape (num_links,), got {requested_links.shape}."
            )
        pose_by_name = self._entity_link_pose_by_name_from_snapshot(entity)
        idx_to_name = {idx: name for name, idx in entity.link_idx_local.items()}
        result = np.zeros((1, requested_links.shape[0], 3), dtype=np.float32)
        for result_col, link_idx_local in enumerate(requested_links.tolist()):
            link_name = idx_to_name.get(link_idx_local)
            if link_name is None:
                raise KeyError(
                    f"Unknown local link index {link_idx_local}. Available link indices: {sorted(idx_to_name)}."
                )
            pose = pose_by_name.get(link_name)
            if pose is None:
                raise KeyError(f"Missing buffered FK pose for link {link_name!r}.")
            result[0, result_col] = pose[:3, 3]
        return torch.from_numpy(result).to(self.env.device)

    # Required by: world/robot.py
    def get_entity_links_vel(self, entity, links_idx_local, ref: str):
        # this should not be called during real world execution. the path to this function should be blocked.
        self._not_implemented("get_entity_links_vel")

    # Required by: world/robot.py
    def get_entity_link_pos(self, entity, *, link_name: str, ref: str):
        link_idx_local = self.get_link_idx_local(entity, link_name)
        return self.get_entity_links_pos(
            entity, links_idx_local=[link_idx_local], ref=ref
        )

    # Required by: world/robot.py
    def get_entity_link_quat(self, entity, *, link_name: str):
        if not isinstance(entity, RealRobotEntity):
            raise NotImplementedError(
                "PhysicsReal.get_entity_link_quat() is only implemented for RealRobotEntity."
            )
        pose_by_name = self._entity_link_pose_by_name_from_snapshot(entity)
        pose = pose_by_name.get(link_name)
        if pose is None:
            raise KeyError(
                f"Missing buffered FK pose for link {link_name!r}. Available links: {sorted(pose_by_name)}."
            )
        quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat().astype(np.float32)
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float32,
        )
        return torch.from_numpy(quat_wxyz[None, None, :]).to(self.env.device)

    # Required by: world/action_manager.py
    def solve_entity_ik(
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
        return self._solve_entity_ik_impl(
            entity,
            link_name=link_name,
            target_pos=target_pos,
            target_quat=target_quat,
            dofs_idx_local=dofs_idx_local,
            init_qpos=init_qpos,
            init_qpos_dofs_idx_local=init_qpos_dofs_idx_local,
        )

    # Required by: world/reward_manager.py
    def get_filtered_contact_info(
        self,
        entity_a,
        entity_b=None,
        *,
        filter_geoms_a=None,
        filter_geoms_b=None,
        filter_links_a=None,
        filter_links_b=None,
        return_geom_force: bool = False,
        return_link_force: bool = False,
        return_geom_pos: bool = False,
        return_link_pos: bool = False,
        device: str | None = None,
    ):
        # this should not be called during real world execution. the path to this function should be blocked.
        self._not_implemented("get_filtered_contact_info")

    # Required by: world/obs_manager.py
    def get_hand_joint_positions(self, hand: Hand) -> torch.Tensor:
        return self.get_entity_dofs_position(hand.robot, hand.dof_map)

    # Required by: world/env.py
    def get_hand_dof_positions(
        self, hand: Hand, dof_indices: list[int] | np.ndarray
    ) -> torch.Tensor:
        return self.get_entity_dofs_position(hand.robot, dof_indices)

    # Required by: world/robot.py
    def get_hand_dof_accelerations(
        self, hand: Hand, dof_indices: list[int] | np.ndarray
    ) -> torch.Tensor:
        # this should not be called during real world execution. the path to this function should be blocked.
        self._not_implemented("get_hand_dof_accelerations")

    # Required by: world/action_manager.py
    def get_hand_dof_limits(
        self, hand: Hand, dof_indices: list[int] | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hand.get_dof_limits(
            dof_indices,
            effective=True,
            device=self.env.device,
            dtype=torch.float32,
        )

    # Required by: world/scene_manager.py
    def resample_random_force(
        self, env_ids: np.ndarray | list[int] | None = None
    ) -> None:
        # this should not be called during real world execution. the path to this function should be blocked.
        self._not_implemented("resample_random_force")

    # TODO: need to connect this with the rgb frames from the real robot setup.
    # this should involve setting up the directory for saving the video frames
    # and maintaining a episode counter to name the video files accordingly.
    # Required by: world/scene_manager.py, world/task_manager.py
    def start_recording(self) -> None:
        self._start_recording_impl()

    # TODO:
    # Required by: world/scene_manager.py, world/task_manager.py
    def stop_recording(self, save_to_filename: str, fps: int = 30) -> None:
        self._stop_recording_impl(save_to_filename, fps=fps)

    # TODO: need to fill this with sth or pass
    # Required by: world/scene_manager.py, world/task_manager.py
    def video_recording_hook(self, timestep: int):
        self._video_recording_hook_impl(timestep)

    # Required by: world/env.py, world/scene_manager.py, world/task_manager.py
    def step(self) -> None:
        self._reload_state_interface(
            force_update=not self.real_cfg.use_threaded_state_query
        )
        self._visualize_current_robot_state()

    # Required by: world/obs_manager.py
    # TODO: need to connect this with the binarized tactile readings from the sensor.
    def get_hand_object_binary_contact_signal(
        self,
        hand: Hand,
        object_entity: ArtObject | None,
        *,
        threshold: float,
    ) -> torch.Tensor:
        del object_entity
        if not isinstance(hand.robot, RealRobotEntity):
            raise TypeError(
                f"Expected hand.robot to be RealRobotEntity, got {type(hand.robot).__name__}."
            )
        snapshot = self._entity_state_snapshot(hand.robot)
        robot_state = snapshot["robot_state"]
        if not isinstance(robot_state, dict):
            raise TypeError(
                f"Expected robot_state snapshot to be a dict, got {type(robot_state).__name__}."
            )
        xhand_state = robot_state.get("xhand")
        if not isinstance(xhand_state, dict):
            raise TypeError(
                f"Expected xhand state to be a dict, got {type(xhand_state).__name__}."
            )
        tactile_state = xhand_state.get("tactile")
        if not isinstance(tactile_state, dict):
            raise TypeError(
                f"Expected tactile state to be a dict, got {type(tactile_state).__name__}."
            )
        if self._tactile_reference_by_sensor is None:
            raise RuntimeError(
                "Tactile reference is not available. Call PhysicsReal.prepare_episode_scene() before requesting tactile contact signals."
            )
        contact_signal = np.zeros(
            (1, len(hand.tactile_link_names), 3), dtype=np.float32
        )
        for tactile_idx, link_name in enumerate(hand.tactile_link_names):
            sensor_name = self._tactile_sensor_name_by_link(link_name)
            sensor_state = tactile_state.get(sensor_name)
            if sensor_state is None:
                raise KeyError(
                    f"Missing tactile sensor {sensor_name!r}. Available sensors: {sorted(tactile_state)}."
                )
            calc_force = np.asarray(
                sensor_state["calc_force"], dtype=np.float32
            ).reshape(-1)
            if calc_force.shape != (3,):
                raise ValueError(
                    f"Expected tactile calc_force shape (3,), got {calc_force.shape} for sensor {sensor_name!r}."
                )
            reference_force = self._tactile_reference_by_sensor.get(sensor_name)
            if reference_force is None:
                raise KeyError(
                    "Missing tactile reference for sensor "
                    f"{sensor_name!r}. Available references: {sorted(self._tactile_reference_by_sensor)}."
                )
            if reference_force.shape != (3,):
                raise ValueError(
                    f"Expected tactile reference shape (3,), got {reference_force.shape} for sensor {sensor_name!r}."
                )
            delta_force = calc_force - reference_force
            contact_signal[0, tactile_idx] = (np.abs(delta_force) > threshold).astype(
                np.float32
            )
        return torch.from_numpy(contact_signal).to(self.env.device)

    # Required by: world/obs_manager.py
    # TODO: need to connect this with camera readings from the real robot setup.
    def get_object_pcd(self, normals: bool) -> torch.Tensor:
        return self._get_object_pcd_impl(normals)

    # TODO: need to do this properly from real world qr marker tracking and pose transformations
    # to obtain the world pose for links.
    def _entity_link_pose_by_name_from_snapshot(
        self, entity: RealRobotEntity
    ) -> dict[str, np.ndarray]:
        snapshot = self._entity_state_snapshot(entity)
        pose_by_name = snapshot["link_pose_by_name"]
        if not isinstance(pose_by_name, dict):
            raise TypeError(
                f"Expected link_pose_by_name snapshot to be a dict, got {type(pose_by_name).__name__}."
            )
        return pose_by_name

    def close(self) -> None:
        self.is_recording = False
        self._recorded_rgb_frames = []
        if self.runtime_viz is not None and not self._runtime_viz_is_shared:
            pass
            # self.runtime_viz.close()
        self.runtime_viz = None
        self._runtime_viz_is_shared = False
        for scene_entity in self.scene_entities.values():
            if isinstance(scene_entity, Hand) and isinstance(
                scene_entity.robot, RealRobotEntity
            ):
                scene_entity.robot.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
