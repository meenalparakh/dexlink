"""MeshCat visualization helpers for the real backend."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from dexlink.utils.retargeting.vis_server import VizServer
from dexlink.src.rl.world.real_utils import RealRobotEntity
from dexlink.src.rl.world.real_utils.physics_real_mixin_base import PhysicsRealMixinBase
from dexlink.src.rl.world.real_utils.robot_entity import FRANKA_JOINT_NAMES
from dexlink.src.rl.world.robot import Hand


class PhysicsRealVisualizationMixin(PhysicsRealMixinBase):
    def _joint_cfg_from_hand_qpos(
        self,
        hand: Hand,
        target_qpos: torch.Tensor | np.ndarray,
    ) -> dict[str, float]:
        target_qpos = torch.as_tensor(target_qpos, dtype=torch.float32)
        if tuple(target_qpos.shape) != (1, len(hand.joints)):
            raise ValueError(
                f"Expected target_qpos shape {(1, len(hand.joints))}, got {tuple(target_qpos.shape)}."
            )
        return {
            joint.name: float(target_qpos[0, joint_idx].item())
            for joint_idx, joint in enumerate(hand.joints)
        }

    def _visualize_world_calibration_candidate(
        self,
        *,
        hand: Hand,
        candidate_qpos: torch.Tensor | np.ndarray,
        base_pose_cam_guess: np.ndarray | None,
        frame_idx: int,
        candidate_idx: int,
    ) -> None:
        viz = self._ensure_runtime_viz()
        if viz is None or base_pose_cam_guess is None:
            return
        base_pose_cam_guess = np.asarray(base_pose_cam_guess, dtype=np.float32)
        if base_pose_cam_guess.shape != (4, 4):
            raise ValueError(
                f"Expected base_pose_cam_guess shape (4, 4), got {base_pose_cam_guess.shape}."
            )
        target_joint_cfg = self._joint_cfg_from_hand_qpos(hand, candidate_qpos)
        self._view_urdf_instance_with_joint_cfg(
            viz=viz,
            name="runtime/calibration_candidate_robot_camera",
            urdf_inst=hand.urdf_inst,
            base_pose=base_pose_cam_guess,
            joint_cfg=target_joint_cfg,
            color=0xD84A3A,
            opacity=0.55,
        )
        print(
            "[WorldCalibration] "
            f"updated MeshCat preview for frame_idx={frame_idx} candidate_idx={candidate_idx}."
        )

    def _visualize_world_calibration_command(
        self,
        *,
        hand: Hand,
        franka_q: np.ndarray,
        xhand_q: np.ndarray,
        base_pose_cam_guess: np.ndarray | None,
        frame_idx: int,
        candidate_idx: int,
        waypoint_idx: int,
        total_waypoints: int,
    ) -> None:
        viz = self._ensure_runtime_viz()
        if viz is None or base_pose_cam_guess is None:
            return
        if not isinstance(hand.robot, RealRobotEntity):
            raise TypeError(
                f"Expected calibration hand robot to be RealRobotEntity, got {type(hand.robot).__name__}."
            )
        base_pose_cam_guess = np.asarray(base_pose_cam_guess, dtype=np.float32)
        if base_pose_cam_guess.shape != (4, 4):
            raise ValueError(
                f"Expected base_pose_cam_guess shape (4, 4), got {base_pose_cam_guess.shape}."
            )
        joint_cfg = self._joint_cfg_from_control_q(hand.robot, franka_q, xhand_q)
        self._view_urdf_instance_with_joint_cfg(
            viz=viz,
            name="runtime/calibration_command_robot_camera",
            urdf_inst=hand.urdf_inst,
            base_pose=base_pose_cam_guess,
            joint_cfg=joint_cfg,
            color=0xD84A3A,
            opacity=0.55,
        )
        print(
            "[WorldCalibration] "
            f"updated MeshCat command preview for frame_idx={frame_idx} "
            f"candidate_idx={candidate_idx} waypoint={waypoint_idx}/{total_waypoints}."
        )

    def _visualize_world_calibration_sample(
        self,
        *,
        scene_pcd_cam: np.ndarray,
        scene_colors_cam: np.ndarray,
        base_pose_cam: np.ndarray,
        marker_pose_cam: np.ndarray,
        wrist_pose_cam: np.ndarray,
        frame_idx: int,
    ) -> None:
        viz = self._ensure_runtime_viz()
        if viz is None:
            return
        scene_pcd_cam = np.asarray(scene_pcd_cam, dtype=np.float32)
        scene_colors_cam = np.asarray(scene_colors_cam, dtype=np.float32)
        base_pose_cam = np.asarray(base_pose_cam, dtype=np.float32)
        marker_pose_cam = np.asarray(marker_pose_cam, dtype=np.float32)
        wrist_pose_cam = np.asarray(wrist_pose_cam, dtype=np.float32)
        if scene_colors_cam.shape != scene_pcd_cam.shape:
            raise ValueError(
                "Expected calibration point-cloud colors to match point shape: "
                f"{scene_colors_cam.shape} vs {scene_pcd_cam.shape}."
            )
        viz.view_pcd(
            scene_pcd_cam,
            colors=scene_colors_cam,
            name="runtime/calibration_scene_camera_pcd",
            size=float(self.real_cfg.visualize_runtime_point_size),
        )
        viz.view_frames(
            self._meshcat_pose(np.eye(4, dtype=np.float32)),
            name="runtime/camera_frame/camera",
            scale=1.0,
        )
        viz.view_frames(
            self._meshcat_pose(base_pose_cam),
            name="runtime/camera_frame/robot_base",
            scale=1.0,
        )
        viz.view_frames(
            self._meshcat_pose(marker_pose_cam),
            name="runtime/camera_frame/qr_marker",
            scale=1.0,
        )
        viz.view_frames(
            name="runtime/camera_frame/wrist_marker",
            pose=self._meshcat_pose(wrist_pose_cam),
            scale=1.0,
        )
        self._visualize_current_robot_state_camera_frame(base_pose_cam=base_pose_cam)
        self.pause_if_visualization_active(
            f"Showing accepted calibration sample at frame_idx={frame_idx}."
        )

    def _visualize_scene_setup(self) -> None:
        return None

    def _ensure_runtime_viz(self) -> VizServer | None:
        if not self.real_cfg.visualize_runtime_debug:
            return None
        if self.runtime_viz is None:
            shared_viz = getattr(self.env.scene_manager, "policy_pcd_viz", None)
            if shared_viz is not None:
                self.runtime_viz = shared_viz
                self._runtime_viz_is_shared = True
                print(
                    "PhysicsReal runtime debug visualization is reusing "
                    "SceneManager.policy_pcd_viz."
                )
            else:
                self.runtime_viz = VizServer(
                    port_vis=int(self.real_cfg.runtime_meshcat_port)
                )
                self.runtime_viz.clear()
                self._runtime_viz_is_shared = False
                print(
                    "Initialized PhysicsReal runtime debug VizServer "
                    f"on port {int(self.real_cfg.runtime_meshcat_port)}."
                )
        return self.runtime_viz

    def _visualization_active(self) -> bool:
        return self._ensure_runtime_viz() is not None

    def pause_if_visualization_active(self, reason: str) -> None:
        if not self.real_cfg.pause_on_visualization_update:
            return
        input(f"[MeshCat] {reason} Press Enter to continue...")

    def _visualize_world_scene_preview(self) -> None:
        return None

    def _joint_cfg_from_control_q(
        self,
        entity: RealRobotEntity,
        franka_q: np.ndarray,
        xhand_q: np.ndarray,
    ) -> dict[str, float]:
        franka_q = np.asarray(franka_q, dtype=np.float32).reshape(-1)
        xhand_q = np.asarray(xhand_q, dtype=np.float32).reshape(-1)
        if franka_q.shape != (len(FRANKA_JOINT_NAMES),):
            raise ValueError(
                f"Expected Franka q shape {(len(FRANKA_JOINT_NAMES),)}, got {franka_q.shape}."
            )
        if xhand_q.shape != (len(entity.xhand_joint_names),):
            raise ValueError(
                f"Expected XHand q shape {(len(entity.xhand_joint_names),)}, got {xhand_q.shape}."
            )
        franka_joint_to_idx = {
            joint_name: idx for idx, joint_name in enumerate(FRANKA_JOINT_NAMES)
        }
        xhand_joint_to_idx = {
            joint_name: idx for idx, joint_name in enumerate(entity.xhand_joint_names)
        }
        helper_root_joint_names = set(entity.helper_root_joint_names)
        joint_cfg: dict[str, float] = {}
        for joint_name in entity.joint_names:
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
                f"Joint {joint_name!r} is not mapped to the real control state."
            )
        return joint_cfg

    def _view_urdf_instance_with_joint_cfg(
        self,
        *,
        viz: VizServer,
        name: str,
        urdf_inst,
        base_pose: np.ndarray,
        joint_cfg: dict[str, float] | None,
        color: int,
        opacity: float = 0.9,
        links_to_include: set[str] | None = None,
    ) -> None:
        mesh_fk = viz.custom_visual_trimesh_fk(urdf_inst, joint_dict=joint_cfg)
        for mesh, (mesh_pose, mesh_name) in mesh_fk.items():
            link_name = mesh_name.rsplit("_", 1)[0]
            if links_to_include is not None and link_name not in links_to_include:
                continue
            viz.view_trimesh(
                f"{name}/{mesh_name}",
                mesh,
                base_pose @ mesh_pose,
                color=color,
                opacity=opacity,
            )

    def _runtime_robot_links_to_include(self, hand) -> set[str] | None:
        if bool(self.real_cfg.visualize_runtime_full_arm_mesh):
            return None
        return set(hand.hand_tree_link_names)

    def _meshcat_pose(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"Expected pose shape (4, 4), got {pose.shape}.")
        return pose

    def _current_wrist_marker_pose_camera(self) -> np.ndarray:
        if self._world_calibration is None:
            raise RuntimeError(
                "World calibration must be available for wrist marker camera-frame visualization."
            )
        hand = self._active_real_hand()
        if not isinstance(hand.robot, RealRobotEntity):
            raise TypeError(
                f"Expected active hand robot to be RealRobotEntity, got {type(hand.robot).__name__}."
            )
        wrist_pose_world = self._entity_link_pose_by_name_from_snapshot(hand.robot).get(
            hand.wrist_link_name
        )
        if wrist_pose_world is None:
            raise KeyError(
                f"Missing buffered FK pose for wrist link {hand.wrist_link_name!r}."
            )
        wrist_marker_pose_world = (
            wrist_pose_world @ self._current_wrist_marker_pose_wrist()
        )
        camera_from_world = np.linalg.inv(self._world_calibration.world_from_cam)
        return (camera_from_world @ wrist_marker_pose_world).astype(np.float32)

    def _visualize_current_robot_state(self) -> None:
        viz = self._ensure_runtime_viz()
        if viz is None:
            return
        if not self.real_cfg.initialize_robot:
            return
        hand = self._active_real_hand()
        if not isinstance(hand.robot, RealRobotEntity):
            raise TypeError(
                f"Expected active hand robot to be RealRobotEntity, got {type(hand.robot).__name__}."
            )
        franka_q, xhand_q = self._entity_control_q_from_snapshot(hand.robot)
        joint_cfg = self._joint_cfg_from_control_q(hand.robot, franka_q, xhand_q)
        self._view_urdf_instance_with_joint_cfg(
            viz=viz,
            name="runtime/current_robot",
            urdf_inst=hand.urdf_inst,
            base_pose=np.asarray(hand.base_pose, dtype=np.float32),
            joint_cfg=joint_cfg,
            color=0x4D7EA8,
            opacity=0.85,
            links_to_include=self._runtime_robot_links_to_include(hand),
        )

    def _visualize_current_robot_state_camera_frame(
        self,
        *,
        base_pose_cam: np.ndarray,
    ) -> None:
        viz = self._ensure_runtime_viz()
        if viz is None:
            return
        if not self.real_cfg.initialize_robot:
            return
        hand = self._active_real_hand()
        if not isinstance(hand.robot, RealRobotEntity):
            raise TypeError(
                f"Expected active hand robot to be RealRobotEntity, got {type(hand.robot).__name__}."
            )
        franka_q, xhand_q = self._entity_control_q_from_snapshot(hand.robot)
        joint_cfg = self._joint_cfg_from_control_q(hand.robot, franka_q, xhand_q)
        self._view_urdf_instance_with_joint_cfg(
            viz=viz,
            name="runtime/calibration_current_robot_camera",
            urdf_inst=hand.urdf_inst,
            base_pose=np.asarray(base_pose_cam, dtype=np.float32),
            joint_cfg=joint_cfg,
            color=0x4D7EA8,
            opacity=0.85,
            links_to_include=self._runtime_robot_links_to_include(hand),
        )

    def _visualize_world_eval_state(
        self,
        *,
        scene_pcd_world: np.ndarray,
        object_pcd_world: np.ndarray,
        heuristic_object_pcd_world: np.ndarray | None = None,
    ) -> None:
        viz = self._ensure_runtime_viz()
        if viz is None:
            return
        scene_pcd_world = np.asarray(scene_pcd_world, dtype=np.float32)
        object_pcd_world = np.asarray(object_pcd_world, dtype=np.float32)
        if heuristic_object_pcd_world is not None:
            heuristic_object_pcd_world = np.asarray(
                heuristic_object_pcd_world, dtype=np.float32
            )
        scene_colors = np.tile(
            np.asarray([[0.65, 0.65, 0.65]], dtype=np.float32),
            (scene_pcd_world.shape[0], 1),
        )
        viz.view_pcd(
            scene_pcd_world,
            colors=scene_colors,
            name="runtime/scene_pcd_world",
            size=float(self.real_cfg.visualize_runtime_point_size),
        )
        if object_pcd_world.shape[0] > 0:
            object_colors = np.tile(
                np.asarray([[0.10, 0.60, 1.00]], dtype=np.float32),
                (object_pcd_world.shape[0], 1),
            )
            viz.view_pcd(
                object_pcd_world,
                colors=object_colors,
                name="runtime/object_pcd_world",
                size=float(self.real_cfg.visualize_runtime_point_size),
            )
        self._visualize_current_robot_state()
        object_mean_source = object_pcd_world
        if (
            heuristic_object_pcd_world is not None
            and heuristic_object_pcd_world.shape[0] > 0
        ):
            heuristic_colors = np.tile(
                np.asarray([[1.00, 0.50, 0.15]], dtype=np.float32),
                (heuristic_object_pcd_world.shape[0], 1),
            )
            viz.view_pcd(
                heuristic_object_pcd_world,
                colors=heuristic_colors,
                name="runtime/heuristic_object_pcd_world",
                size=float(self.real_cfg.visualize_runtime_point_size),
            )
            object_mean_source = heuristic_object_pcd_world
        if object_mean_source.shape[0] == 0:
            viz.view_frames(
                name="runtime/object_mean_frame",
                pose=self._meshcat_pose(np.eye(4, dtype=np.float32)),
                scale=0.12,
            )
            return
        object_mean = np.mean(object_mean_source, axis=0).astype(np.float32)
        object_mean_pose = np.eye(4, dtype=np.float32)
        object_mean_pose[:3, 3] = object_mean
        viz.view_frames(
            name="runtime/object_mean_frame",
            pose=self._meshcat_pose(object_mean_pose),
            scale=0.12,
        )

    def _world_link_pose_from_qpos(
        self,
        hand: Hand,
        target_qpos: torch.Tensor,
        *,
        link_name: str,
    ) -> np.ndarray:
        if tuple(target_qpos.shape) != (1, len(hand.dof_map)):
            raise ValueError(
                f"Expected target_qpos shape {(1, len(hand.dof_map))}, got {tuple(target_qpos.shape)}."
            )
        full_jpos = torch.zeros(
            (1, len(hand.joints)),
            dtype=target_qpos.dtype,
            device=target_qpos.device,
        )
        for local_col, joint_col in enumerate(
            hand._joint_col_by_dof_idx[dof_idx] for dof_idx in hand.dof_map
        ):
            full_jpos[:, joint_col] = target_qpos[:, local_col]
        joint_dict = {
            joint.name: full_jpos[:, joint_idx].detach().cpu().numpy()
            for joint_idx, joint in enumerate(hand.joints)
        }
        link_fk = hand.urdf_inst.link_fk_batch(cfgs=joint_dict)
        wrist_link = hand.urdf_inst.link_map[link_name]
        local_pose = link_fk.get(wrist_link)
        if local_pose is None:
            raise ValueError(f"URDF FK did not return a pose for link {link_name!r}.")
        world_pose = np.asarray(hand.base_pose, dtype=np.float32) @ np.asarray(
            local_pose[0], dtype=np.float32
        )
        return world_pose.astype(np.float32)

    def visualize_action_targets(
        self,
        entity: RealRobotEntity,
        *,
        target_qpos: torch.Tensor,
        target_franka_q: np.ndarray | torch.Tensor,
    ) -> None:
        viz = self._ensure_runtime_viz()
        if viz is None:
            return
        hand = self._hand_for_entity(entity)
        target_qpos = torch.as_tensor(target_qpos, dtype=torch.float32)
        target_franka_q = torch.as_tensor(target_franka_q, dtype=torch.float32)
        if tuple(target_qpos.shape) != (1, len(hand.dof_map)):
            raise ValueError(
                f"Expected target_qpos shape {(1, len(hand.dof_map))}, got {tuple(target_qpos.shape)}."
            )
        if tuple(target_franka_q.shape) != (len(FRANKA_JOINT_NAMES),):
            raise ValueError(
                "Expected target_franka_q shape "
                f"{(len(FRANKA_JOINT_NAMES),)}, got {tuple(target_franka_q.shape)}."
            )
        target_joint_cfg = self._joint_cfg_from_control_q(
            entity,
            target_franka_q.detach().cpu().numpy().astype(np.float32),
            target_qpos[0, self.env.action_manager.hand_target_indices]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
        )
        self._visualize_current_robot_state()
        self._view_urdf_instance_with_joint_cfg(
            viz=viz,
            name="runtime/action_target_robot",
            urdf_inst=hand.urdf_inst,
            base_pose=np.asarray(hand.base_pose, dtype=np.float32),
            joint_cfg=target_joint_cfg,
            color=0xD84A3A,
            opacity=0.55,
            links_to_include=self._runtime_robot_links_to_include(hand),
        )
        self.pause_if_visualization_active(
            "Showing action target visualization before control/decimation."
        )

    def _confirm_real_world_setup(self) -> None:
        pass
        # response = (
        #     input(
        #         "Check the MeshCat scene setup and the real-world setup, excluding the asset and the robot."
        #         "Do they match? [y/N]: "
        #     )
        #     .strip()
        #     .lower()
        # )
        # if response not in {"y", "yes"}:
        #     raise RuntimeError(
        #         "Aborting PhysicsReal initialization because the real-world setup was not confirmed."
        #     )

    def _pose_from_wxyz(
        self,
        pos: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
    ) -> np.ndarray:
        pose = np.eye(4, dtype=np.float32)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float32,
        )
        pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
        pose[:3, 3] = np.asarray(pos, dtype=np.float32)
        return pose
