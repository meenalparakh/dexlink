from __future__ import annotations

import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from dexlink.src.retargeting.data.demo_loader import is_oakink_dataset
from dexlink.src.rl.world.art_object import ArtObject, get_object_cfg
from dexlink.src.rl.world.physics_real import PhysicsReal
from dexlink.src.rl.world.real_utils import (
    RealEntity,
    RealRobotEntity,
    transform_points_homogeneous,
)
from dexlink.src.rl.world.real_utils.robot_entity import FRANKA_JOINT_NAMES
from dexlink.src.rl.world.robot import Hand, get_hand_specific_cfg


class MultiObjectPhysicsReal(PhysicsReal):
    """Real backend for multi-object sequence policies.

    ``scene_entities[side]`` remains the policy hand from the training config
    (for example ``xhand``). Hardware I/O is owned by ``real_control_hands`` so
    a hidden ``franka_xhand`` can be used for real state/control without
    changing policy observation or action dimensions.
    """

    def __init__(self, real_cfg, env):
        super().__init__(real_cfg, env)
        self.object_cfgs: dict[str, dict] = {}
        self.real_control_hands: dict[str, Hand] = {}
        self._latest_policy_qpos_by_side: dict[str, torch.Tensor] = {}
        self._latest_real_ik_qpos_by_side: dict[str, torch.Tensor] = {}
        self._latest_real_control_qpos_by_side: dict[str, torch.Tensor] = {}
        self._latest_isolated_object_pcd_world: torch.Tensor | None = None

    @property
    def object_entities(self) -> dict[str, ArtObject]:
        objects = self.scene_entities.get("objects")
        if not isinstance(objects, dict):
            raise ValueError("MultiObjectPhysicsReal scene has not created objects.")
        return objects

    def build_scene(self) -> None:
        assert self.scene is not None
        task_manager = self.env.task_manager

        self.scene_entities.clear()
        self.real_control_hands.clear()
        self._latest_policy_qpos_by_side.clear()
        self._latest_real_ik_qpos_by_side.clear()
        self._latest_real_control_qpos_by_side.clear()
        self._latest_isolated_object_pcd_world = None
        self.hand_sensors = None
        self.hand_cfgs = None
        self.object_cfg = None
        self.object_cfgs = {}
        self.scene_entities["plane"] = self._add_plane(pos=(0, 0, 0))

        if not is_oakink_dataset(self.env.task_cfg.demo_dataset):
            raise ValueError(
                "MultiObjectPhysicsReal currently expects an OakInk exported "
                f"sequence dataset, got {self.env.task_cfg.demo_dataset!r}."
            )

        self._populate_scene_objects()
        if self.env.task_manager.object_only:
            return

        policy_hand_cfgs = get_hand_specific_cfg(self.env.task_cfg.hand_name)
        robot_base_pose = np.eye(4, dtype=np.float32)
        for side in self.env.task_manager.active_hand_sides:
            if side not in {"left", "right"}:
                raise ValueError(
                    f"Expected active hand side 'left' or 'right', got {side!r}."
                )
            self.scene_entities[side] = Hand(
                physics=self,
                robo_cfg=self._hand_cfg_for_role(
                    policy_hand_cfgs,
                    role_side=side,
                    config_name=self.env.task_cfg.hand_name,
                ),
                num_envs=self.env.num_envs,
                device=self.env.device,
                obs_cfg=self.env.cfg.obs_cfg.hand_cfg,
                env=self.env,
                name=side,
                base_pose=robot_base_pose,
            )
            self._initialize_real_control_hand(side=side, base_pose=robot_base_pose)

        if (
            not self.real_cfg.visualization_only
            and self.real_cfg.initialize_robot
            and self.real_cfg.read_robot_state_on_initialize
        ):
            self._reload_state_interface(force_update=True)
        if not self.real_cfg.visualization_only:
            task_manager.process_retargeted_motion()

    def _hand_cfg_for_role(
        self,
        hand_cfgs: dict[str, dict],
        *,
        role_side: str,
        config_name: str,
    ) -> dict:
        assert self.env.task_manager.original_demo is not None
        retargeted_motion = self.env.task_manager.original_demo.retargeted_motion
        source_by_side = retargeted_motion.get("hand_config_source_by_side", {})
        if source_by_side is None:
            source_by_side = {}
        if not isinstance(source_by_side, dict):
            raise ValueError(
                "Expected retargeted_motion['hand_config_source_by_side'] to be a "
                f"dict, got {type(source_by_side).__name__}."
            )
        config_side = str(source_by_side.get(role_side, role_side))
        if config_side not in hand_cfgs:
            raise ValueError(
                f"Hand config source side {config_side!r} for role {role_side!r} "
                f"is not available in {config_name!r} hand cfgs {sorted(hand_cfgs)}."
            )
        return hand_cfgs[config_side]

    def _initialize_real_control_hand(
        self, *, side: str, base_pose: np.ndarray | None
    ) -> None:
        real_control_hand_name = str(self.real_cfg.real_control_hand_name)
        if real_control_hand_name != "franka_xhand":
            raise ValueError(
                "MultiObjectPhysicsReal currently expects "
                "real_cfg.real_control_hand_name == 'franka_xhand', got "
                f"{real_control_hand_name!r}."
            )
        control_cfgs = get_hand_specific_cfg(real_control_hand_name)
        control_hand = Hand(
            physics=self,
            robo_cfg=self._hand_cfg_for_role(
                control_cfgs,
                role_side=side,
                config_name=real_control_hand_name,
            ),
            num_envs=self.env.num_envs,
            device=self.env.device,
            obs_cfg=self.env.cfg.obs_cfg.hand_cfg,
            env=self.env,
            name=side,
            base_pose=base_pose,
        )
        self._initialize_real_control_hand_entity(control_hand, side=side)
        self._initialize_ik_backend(control_hand)
        self.real_control_hands[side] = control_hand

    def _initialize_real_control_hand_entity(self, hand: Hand, *, side: str) -> None:
        if not isinstance(hand.robot, RealEntity):
            raise TypeError(
                f"Expected hidden control hand robot to be RealEntity, got {type(hand.robot).__name__}."
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

    def _reload_state_interface(self, *, force_update: bool) -> None:
        self._clear_state_interface()
        for hand in self.real_control_hands.values():
            if isinstance(hand.robot, RealRobotEntity):
                self._reload_entity_state_interface(
                    hand.robot, force_update=force_update
                )

    def _populate_scene_objects(self) -> None:
        task_manager = self.env.task_manager
        object_entities: dict[str, ArtObject] = {}
        object_cfgs: dict[str, dict] = {}

        for object_id, object_name, urdf_path, object_static in zip(
            task_manager.scene_object_ids,
            task_manager.scene_object_names,
            task_manager.scene_object_urdf_paths,
            task_manager.scene_object_static,
        ):
            obj_cfg = get_object_cfg(
                reference_demo=None,
                urdf_path=urdf_path,
                obj_name=object_name,
                object_fixed=bool(self.env.task_cfg.object_fixed)
                or bool(object_static),
            )
            entity = ArtObject(
                physics=self,
                obj_cfg=obj_cfg,
                num_envs=self.env.num_envs,
            )
            object_entities[object_id] = entity
            object_cfgs[object_id] = obj_cfg
            self.scene_entities[task_manager.scene_object_entity_key(object_id)] = (
                entity
            )

        canonical_id = task_manager.scene_canonical_object_id
        self.scene_entities["objects"] = object_entities
        self.scene_entities["object"] = object_entities[canonical_id]
        self.object_cfg = object_cfgs[canonical_id]
        self.object_cfgs = object_cfgs

    def _active_real_hand(self) -> Hand:
        active_sides = self.env.task_manager.active_hand_sides
        if len(active_sides) != 1:
            raise ValueError(
                "MultiObjectPhysicsReal expects exactly one active real hand side, "
                f"got {active_sides}."
            )
        side = active_sides[0]
        hand = self.real_control_hands.get(side)
        if hand is None:
            raise KeyError(
                f"Missing hidden real control hand for side {side!r}. "
                f"Available sides: {sorted(self.real_control_hands)}."
            )
        return hand

    def _policy_hand_for_entity(self, entity: RealEntity) -> tuple[str, Hand] | None:
        for side in self.env.task_manager.active_hand_sides:
            scene_entity = self.scene_entities.get(side)
            if isinstance(scene_entity, Hand) and scene_entity.robot is entity:
                return side, scene_entity
        return None

    def _hand_for_entity(self, entity: RealRobotEntity) -> Hand:
        for hand in self.real_control_hands.values():
            if hand.robot is entity:
                return hand
        return super()._hand_for_entity(entity)

    def _policy_hand_qpos_from_real_state(self, side: str, hand: Hand) -> torch.Tensor:
        control_hand = self.real_control_hands.get(side)
        if control_hand is None or not isinstance(control_hand.robot, RealRobotEntity):
            raise ValueError(f"Missing real control hand for policy side {side!r}.")
        if control_hand.robot.robot_controller is None:
            if not self.real_cfg.initialize_robot:
                cached_qpos = self._latest_policy_qpos_by_side.get(side)
                if cached_qpos is not None:
                    return cached_qpos.to(self.env.device)
                return self._policy_hand_default_qpos(hand)
            raise RuntimeError(
                "Cannot read policy hand qpos because the hidden real control "
                "hand robot_controller is not initialized."
            )
        _franka_q, xhand_q = self._entity_control_q_from_snapshot(control_hand.robot)
        xhand_joint_to_idx = {
            joint_name: idx
            for idx, joint_name in enumerate(control_hand.robot.xhand_joint_names)
        }
        values = np.zeros((1, len(hand.dof_map)), dtype=np.float32)
        for col, joint_name in enumerate(hand.joint_names):
            if joint_name in hand.root_joints:
                values[0, col] = 0.0
            elif joint_name in xhand_joint_to_idx:
                values[0, col] = xhand_q[xhand_joint_to_idx[joint_name]]
            else:
                raise NotImplementedError(
                    f"Policy hand joint {joint_name!r} is not a root or XHand joint."
                )
        return torch.from_numpy(values).to(self.env.device)

    def _policy_hand_default_qpos(self, hand: Hand) -> torch.Tensor:
        qpos_cfg = hand.robo_cfg.get("default_qpos", None)
        if qpos_cfg is None:
            qpos_cfg = hand.robo_cfg.get("open_qpos", None)
        if not isinstance(qpos_cfg, dict):
            raise ValueError(
                f"Policy hand {hand.name!r} requires robo_cfg['default_qpos'] "
                "or robo_cfg['open_qpos'] when real robot initialization is disabled."
            )
        values = np.zeros((1, len(hand.dof_map)), dtype=np.float32)
        for col, joint_name in enumerate(hand.joint_names):
            if joint_name in hand.root_joints:
                values[0, col] = 0.0
                continue
            if joint_name not in qpos_cfg:
                raise ValueError(
                    f"Policy hand {hand.name!r} qpos config is missing joint "
                    f"{joint_name!r} needed for observation-only real eval."
                )
            values[0, col] = float(qpos_cfg[joint_name])
        return torch.from_numpy(values).to(self.env.device)

    def _control_hand_qpos_from_real_state(self, side: str) -> torch.Tensor:
        control_hand = self.real_control_hands.get(side)
        if control_hand is None or not isinstance(control_hand.robot, RealRobotEntity):
            raise ValueError(f"Missing real control hand for side {side!r}.")
        if control_hand.robot.robot_controller is None:
            if not self.real_cfg.initialize_robot:
                return self._control_hand_default_qpos(control_hand)
            raise RuntimeError(
                "Cannot read hidden real control hand qpos because robot_controller "
                "is not initialized."
            )
        franka_q, xhand_q = self._entity_control_q_from_snapshot(control_hand.robot)
        franka_joint_to_idx = {
            joint_name: idx for idx, joint_name in enumerate(FRANKA_JOINT_NAMES)
        }
        xhand_joint_to_idx = {
            joint_name: idx
            for idx, joint_name in enumerate(control_hand.robot.xhand_joint_names)
        }
        helper_root_joint_names = set(control_hand.robot.helper_root_joint_names)
        values = np.zeros((1, len(control_hand.dof_map)), dtype=np.float32)
        for col, joint_name in enumerate(control_hand.joint_names):
            if joint_name in franka_joint_to_idx:
                values[0, col] = franka_q[franka_joint_to_idx[joint_name]]
            elif joint_name in xhand_joint_to_idx:
                values[0, col] = xhand_q[xhand_joint_to_idx[joint_name]]
            elif joint_name in helper_root_joint_names:
                values[0, col] = 0.0
            else:
                raise NotImplementedError(
                    f"Hidden control hand joint {joint_name!r} is not mapped to "
                    "Franka, XHand, or helper-root state."
                )
        return torch.from_numpy(values).to(self.env.device)

    def _control_hand_default_qpos(self, control_hand: Hand) -> torch.Tensor:
        qpos_cfg = control_hand.robo_cfg.get("default_qpos", None)
        if qpos_cfg is None:
            qpos_cfg = control_hand.robo_cfg.get("open_qpos", None)
        if not isinstance(qpos_cfg, dict):
            raise ValueError(
                f"Hidden control hand {control_hand.name!r} requires "
                "robo_cfg['default_qpos'] or robo_cfg['open_qpos'] when real robot "
                "initialization is disabled."
            )
        helper_root_joint_names = set(control_hand.robot.helper_root_joint_names)
        franka_joint_names = set(FRANKA_JOINT_NAMES)
        values = np.zeros((1, len(control_hand.dof_map)), dtype=np.float32)
        for col, joint_name in enumerate(control_hand.joint_names):
            if (
                joint_name in franka_joint_names
                or joint_name in helper_root_joint_names
            ):
                values[0, col] = 0.0
                continue
            if joint_name not in qpos_cfg:
                raise ValueError(
                    f"Hidden control hand {control_hand.name!r} qpos config is "
                    f"missing joint {joint_name!r} needed for no-robot probes."
                )
            values[0, col] = float(qpos_cfg[joint_name])
        return torch.from_numpy(values).to(self.env.device)

    def set_entity_dofs_position(
        self, entity, values, dofs_idx_local, envs_idx=None
    ) -> None:
        policy_match = self._policy_hand_for_entity(entity)
        if policy_match is None:
            return super().set_entity_dofs_position(
                entity,
                values,
                dofs_idx_local,
                envs_idx=envs_idx,
            )
        side, hand = policy_match
        if envs_idx is not None:
            envs_idx = np.asarray(envs_idx, dtype=np.int64).reshape(-1)
            if envs_idx.tolist() != [0]:
                raise ValueError(
                    "Real policy-hand state writes support only envs_idx=[0], got "
                    f"{envs_idx.tolist()}."
                )
        dofs_idx = np.asarray(dofs_idx_local, dtype=np.int64).reshape(-1)
        if dofs_idx.tolist() != list(hand.dof_map):
            raise ValueError(
                "Policy hand state writes must use hand.dof_map order. Expected "
                f"{hand.dof_map}, got {dofs_idx.tolist()}."
            )
        qpos = torch.as_tensor(values, dtype=torch.float32, device=self.env.device)
        expected_shape = (1, len(hand.dof_map))
        if tuple(qpos.shape) != expected_shape:
            raise ValueError(
                f"Expected policy qpos shape {expected_shape}, got {tuple(qpos.shape)}."
            )
        self._latest_policy_qpos_by_side[side] = qpos.detach().clone()

    def get_entity_dofs_position(self, entity, dofs_idx_local):
        policy_match = self._policy_hand_for_entity(entity)
        if policy_match is None:
            return super().get_entity_dofs_position(entity, dofs_idx_local)
        side, hand = policy_match
        full_qpos = self._policy_hand_qpos_from_real_state(side, hand)
        dofs_idx_local = np.asarray(dofs_idx_local, dtype=np.int64).reshape(-1)
        dof_to_col = {dof_idx: col for col, dof_idx in enumerate(hand.dof_map)}
        cols = []
        for dof_idx in dofs_idx_local.tolist():
            col = dof_to_col.get(dof_idx)
            if col is None:
                raise KeyError(
                    f"Unknown policy hand DOF index {dof_idx}; "
                    f"available DOFs: {sorted(dof_to_col)}."
                )
            cols.append(col)
        return full_qpos[:, cols]

    def _policy_hand_joint_cfg(self, side: str, hand: Hand) -> dict[str, float]:
        qpos = self._policy_hand_qpos_from_real_state(side, hand)[0]
        return {
            joint_name: float(qpos[col].detach().cpu().item())
            for col, joint_name in enumerate(hand.joint_names)
        }

    def _policy_hand_link_fk_world(
        self, side: str, hand: Hand
    ) -> dict[str, np.ndarray]:
        joint_cfg = self._policy_hand_joint_cfg(side, hand)
        link_fk = hand.urdf_inst.link_fk(cfg=joint_cfg)
        world_from_policy_base = np.asarray(hand.base_pose, dtype=np.float32)
        control_hand = self.real_control_hands.get(side)
        if control_hand is None or not isinstance(control_hand.robot, RealRobotEntity):
            raise ValueError(f"Missing hidden real control hand for side {side!r}.")
        if control_hand.robot.robot_controller is not None:
            pose_by_name = self._entity_link_pose_by_name_from_snapshot(
                control_hand.robot
            )
            real_wrist_pose = pose_by_name.get(control_hand.wrist_link_name)
            if real_wrist_pose is None:
                raise KeyError(
                    "Missing real FK pose for hidden wrist link "
                    f"{control_hand.wrist_link_name!r}."
                )
            policy_wrist_link = hand.urdf_inst.link_map.get(hand.wrist_link_name)
            if policy_wrist_link is None:
                raise KeyError(
                    f"Policy hand URDF is missing wrist link {hand.wrist_link_name!r}."
                )
            policy_wrist_pose = link_fk.get(policy_wrist_link)
            if policy_wrist_pose is None:
                raise KeyError(
                    "Policy hand FK did not return wrist link pose for "
                    f"{hand.wrist_link_name!r}."
                )
            world_from_policy_base = np.asarray(
                real_wrist_pose, dtype=np.float32
            ) @ np.linalg.inv(np.asarray(policy_wrist_pose, dtype=np.float32))

        pose_by_name: dict[str, np.ndarray] = {}
        for link, local_pose in link_fk.items():
            pose_by_name[link.name] = (
                world_from_policy_base @ np.asarray(local_pose, dtype=np.float32)
            ).astype(np.float32)
        return pose_by_name

    def get_entity_links_pos(self, entity, links_idx_local, ref: str):
        policy_match = self._policy_hand_for_entity(entity)
        if policy_match is None:
            return super().get_entity_links_pos(entity, links_idx_local, ref)
        if ref != "link_origin":
            raise NotImplementedError(
                "MultiObjectPhysicsReal policy hand FK only supports "
                f"ref='link_origin', got {ref!r}."
            )
        side, hand = policy_match
        pose_by_name = self._policy_hand_link_fk_world(side, hand)
        idx_to_name = {idx: name for name, idx in entity.link_idx_local.items()}
        requested_links = np.asarray(links_idx_local, dtype=np.int64).reshape(-1)
        result = np.zeros((1, requested_links.shape[0], 3), dtype=np.float32)
        for result_col, link_idx_local in enumerate(requested_links.tolist()):
            link_name = idx_to_name.get(link_idx_local)
            if link_name is None:
                raise KeyError(
                    f"Unknown local link index {link_idx_local}. "
                    f"Available link indices: {sorted(idx_to_name)}."
                )
            pose = pose_by_name.get(link_name)
            if pose is None:
                raise KeyError(f"Missing policy FK pose for link {link_name!r}.")
            result[0, result_col] = pose[:3, 3]
        return torch.from_numpy(result).to(self.env.device)

    def get_entity_link_quat(self, entity, *, link_name: str):
        policy_match = self._policy_hand_for_entity(entity)
        if policy_match is None:
            return super().get_entity_link_quat(entity, link_name=link_name)
        side, hand = policy_match
        pose_by_name = self._policy_hand_link_fk_world(side, hand)
        pose = pose_by_name.get(link_name)
        if pose is None:
            raise KeyError(f"Missing policy FK pose for link {link_name!r}.")
        quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat().astype(np.float32)
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float32,
        )
        return torch.from_numpy(quat_wxyz[None, None, :]).to(self.env.device)

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
        policy_match = self._policy_hand_for_entity(entity)
        if policy_match is None:
            return super().solve_entity_ik(
                entity,
                link_name=link_name,
                target_pos=target_pos,
                target_quat=target_quat,
                dofs_idx_local=dofs_idx_local,
                init_qpos=init_qpos,
                init_qpos_dofs_idx_local=init_qpos_dofs_idx_local,
            )

        side, policy_hand = policy_match
        control_hand = self.real_control_hands.get(side)
        if control_hand is None or not isinstance(control_hand.robot, RealRobotEntity):
            raise ValueError(f"Missing hidden real control hand for side {side!r}.")
        if (
            control_hand.robot.robot_controller is None
            and self.real_cfg.initialize_robot
        ):
            raise RuntimeError(
                "Cannot solve real policy-hand IK because robot_controller is "
                "not initialized."
            )
        if link_name != policy_hand.wrist_link_name:
            raise ValueError(
                "MultiObjectPhysicsReal policy-hand IK only supports the policy "
                f"wrist link {policy_hand.wrist_link_name!r}, got {link_name!r}."
            )
        if control_hand.wrist_link_name != link_name:
            raise ValueError(
                "Expected policy hand and hidden control hand to share the wrist "
                f"link name {link_name!r}, got hidden wrist link "
                f"{control_hand.wrist_link_name!r}."
            )

        control_init_qpos = self._control_hand_qpos_from_real_state(side).to("cpu")
        control_ik_dofs = [
            control_hand.dof_map[joint_idx]
            for joint_idx in control_hand.wrist_control_joint_idx
        ]
        control_ik_qpos = self._solve_entity_ik_impl(
            control_hand.robot,
            link_name=link_name,
            target_pos=target_pos,
            target_quat=target_quat,
            dofs_idx_local=control_ik_dofs,
            init_qpos=control_init_qpos,
            init_qpos_dofs_idx_local=control_hand.dof_map,
        )
        expected_control_shape = (self.env.num_envs, len(control_hand.dof_map))
        if tuple(control_ik_qpos.shape) != expected_control_shape:
            raise ValueError(
                "Hidden real IK qpos shape mismatch: expected "
                f"{expected_control_shape}, got {tuple(control_ik_qpos.shape)}."
            )
        self._latest_real_ik_qpos_by_side[side] = control_ik_qpos.detach().clone()

        policy_init = torch.as_tensor(init_qpos, dtype=torch.float32)
        expected_policy_shape = (self.env.num_envs, len(policy_hand.dof_map))
        if tuple(policy_init.shape) != expected_policy_shape:
            raise ValueError(
                "Policy IK init qpos shape mismatch: expected "
                f"{expected_policy_shape}, got {tuple(policy_init.shape)}."
            )
        init_dofs = np.asarray(init_qpos_dofs_idx_local, dtype=np.int64).reshape(-1)
        if init_dofs.tolist() != list(policy_hand.dof_map):
            raise ValueError(
                "Policy IK init_qpos_dofs_idx_local must match policy hand.dof_map. "
                f"Expected {policy_hand.dof_map}, got {init_dofs.tolist()}."
            )
        requested_dofs = np.asarray(dofs_idx_local, dtype=np.int64).reshape(-1)
        expected_requested = [
            policy_hand.dof_map[joint_idx]
            for joint_idx in policy_hand.wrist_control_joint_idx
        ]
        if requested_dofs.tolist() != expected_requested:
            raise ValueError(
                "Policy IK dofs_idx_local must target the policy wrist-control "
                f"DOFs. Expected {expected_requested}, got {requested_dofs.tolist()}."
            )
        return policy_init.to(self.env.device)

    def get_object_pcd(self, normals: bool) -> torch.Tensor:
        if normals:
            raise NotImplementedError(
                "MultiObjectPhysicsReal only supports policy object point clouds "
                "without normals. Disable privileged object PCDs for real eval."
            )
        scene_pcd_world, object_pcd_world = (
            self._get_scene_and_filtered_object_pcd_world()
        )
        if self.real_cfg.use_phase_cached_object_pcd:
            object_pcd_world = self._phase_cached_object_pcd_world(
                fresh_object_pcd_world=object_pcd_world
            )
            table_pcd_world = self._phase_cached_table_pcd_world(
                object_pcd_world=object_pcd_world
            )
            scene_pcd_world = np.concatenate(
                [object_pcd_world, table_pcd_world],
                axis=0,
            ).astype(np.float32)
        else:
            self._episode_initial_object_pcd_world = None
            self._episode_initial_table_pcd_world = None
            self._lift_start_object_pcd_world = None
            self._lift_start_wrist_pose_world = None
        self._latest_isolated_object_pcd_world = torch.tensor(
            object_pcd_world[None],
            dtype=torch.float32,
            device=self.env.device,
        )
        self._visualize_world_eval_state(
            scene_pcd_world=scene_pcd_world,
            object_pcd_world=object_pcd_world,
        )
        return self._sample_multi_object_policy_pcd(object_pcd_world=scene_pcd_world)

    def _phase_cached_object_pcd_world(
        self, *, fresh_object_pcd_world: np.ndarray
    ) -> np.ndarray:
        fresh_object_pcd_world = np.asarray(fresh_object_pcd_world, dtype=np.float32)
        if fresh_object_pcd_world.ndim != 2 or fresh_object_pcd_world.shape[1] != 3:
            raise ValueError(
                "Expected fresh_object_pcd_world shape (num_points, 3), got "
                f"{fresh_object_pcd_world.shape}."
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
            return current_object_pcd_world.astype(np.float32)

        if self._lift_start_wrist_pose_world is None:
            self._lift_start_wrist_pose_world = self._current_wrist_pose_world()
            self._lift_start_object_pcd_world = current_object_pcd_world.copy()
            return current_object_pcd_world.astype(np.float32)
        if self._lift_start_object_pcd_world is None:
            raise ValueError(
                "Cached object PCD is missing while a cached wrist pose is set."
            )
        current_wrist_pose_world = self._current_wrist_pose_world()
        wrist_delta_world = (
            current_wrist_pose_world @ np.linalg.inv(self._lift_start_wrist_pose_world)
        ).astype(np.float32)
        return transform_points_homogeneous(
            self._lift_start_object_pcd_world,
            wrist_delta_world,
        ).astype(np.float32)

    def _phase_cached_table_pcd_world(
        self, *, object_pcd_world: np.ndarray
    ) -> np.ndarray:
        if self._episode_initial_table_pcd_world is None:
            self._episode_initial_table_pcd_world = (
                self._sample_synthetic_table_points_around_object(
                    object_pcd_world=object_pcd_world,
                )
            )
        return np.asarray(self._episode_initial_table_pcd_world, dtype=np.float32)

    def get_current_isolated_object_pcd(self) -> torch.Tensor:
        scene_pcd_world, object_pcd_world = (
            self._get_scene_and_filtered_object_pcd_world()
        )
        self._latest_isolated_object_pcd_world = torch.tensor(
            object_pcd_world[None],
            dtype=torch.float32,
            device=self.env.device,
        )
        self._visualize_world_eval_state(
            scene_pcd_world=scene_pcd_world,
            object_pcd_world=object_pcd_world,
        )
        return self._latest_isolated_object_pcd_world.clone()

    def get_latest_isolated_object_pcd(self) -> torch.Tensor:
        if self._latest_isolated_object_pcd_world is None:
            raise RuntimeError(
                "No cached isolated real object PCD is available. Compute real "
                "policy observations before requesting z-aligned wrist targets."
            )
        return self._latest_isolated_object_pcd_world.clone()

    def _sample_multi_object_policy_pcd(
        self, *, object_pcd_world: np.ndarray
    ) -> torch.Tensor:
        object_pcd_world = np.asarray(object_pcd_world, dtype=np.float32)
        if object_pcd_world.ndim != 2 or object_pcd_world.shape[1] != 3:
            raise ValueError(
                "Expected object_pcd_world shape (num_points, 3), got "
                f"{object_pcd_world.shape}."
            )
        obs_mgr = self.env.observation_manager
        total_n_pts = len(obs_mgr.object_ids) * int(obs_mgr.policy_object_pcd_n_pts)
        if total_n_pts <= 0:
            raise ValueError(
                f"Expected positive total policy PCD points, got {total_n_pts}."
            )
        if object_pcd_world.shape[0] == 0:
            sampled = np.zeros((total_n_pts, 3), dtype=np.float32)
        elif object_pcd_world.shape[0] >= total_n_pts:
            sample_idx = np.linspace(
                0, object_pcd_world.shape[0] - 1, total_n_pts, dtype=np.int64
            )
            sampled = object_pcd_world[sample_idx]
        else:
            pad_count = total_n_pts - object_pcd_world.shape[0]
            sampled = np.concatenate(
                [
                    object_pcd_world,
                    np.repeat(object_pcd_world[-1:], pad_count, axis=0),
                ],
                axis=0,
            )
        return torch.tensor(
            sampled[None],
            dtype=torch.float32,
            device=self.env.device,
        )

    def get_policy_hand_pcd(
        self,
        hand: Hand,
        *,
        links_to_include: list[str] | None,
    ) -> torch.Tensor:
        side = hand.name
        pose_by_name = self._policy_hand_link_fk_world(side, hand)
        selected_link_indices = hand._resolve_surface_link_indices(links_to_include)
        sample_counts = hand._get_robot_pcd_sample_counts(selected_link_indices)

        transformed_points = []
        for link_idx, n_pts in zip(selected_link_indices, sample_counts):
            link = hand.hand_surface_links[link_idx]
            dense_local_kpts = hand.dense_hand_surface_kpts[link_idx].to(
                device=self.env.device,
                dtype=torch.float32,
            )
            sample_indices = torch.randperm(
                dense_local_kpts.shape[0],
                device=self.env.device,
            )[:n_pts]
            local_kpts = dense_local_kpts[sample_indices]
            link_pose_np = pose_by_name.get(link.name)
            if link_pose_np is None:
                raise KeyError(
                    f"Missing real FK pose for hand surface link {link.name!r}."
                )
            link_pose = torch.as_tensor(
                link_pose_np,
                dtype=torch.float32,
                device=self.env.device,
            )
            n_local_pts = local_kpts.shape[0]
            local_kpts_h = torch.cat(
                [
                    local_kpts,
                    torch.ones(
                        n_local_pts,
                        1,
                        dtype=torch.float32,
                        device=self.env.device,
                    ),
                ],
                dim=1,
            )
            world_kpts = (link_pose @ local_kpts_h.T).T[:, :3]
            transformed_points.append(world_kpts.unsqueeze(0))

        return torch.cat(transformed_points, dim=1)

    def get_hand_object_binary_contact_signal(
        self,
        hand: Hand,
        object_entity: ArtObject | None,
        *,
        threshold: float,
    ) -> torch.Tensor:
        del object_entity
        side = hand.name
        real_hand = self.real_control_hands.get(side)
        if real_hand is None:
            raise KeyError(
                f"Missing hidden real control hand for policy side {side!r}. "
                f"Available sides: {sorted(self.real_control_hands)}."
            )
        if not self.real_cfg.initialize_robot:
            return torch.zeros(
                (1, len(hand.tactile_link_names), 3),
                dtype=torch.float32,
                device=self.env.device,
            )
        return super().get_hand_object_binary_contact_signal(
            real_hand,
            None,
            threshold=threshold,
        )

    def control_entity_dofs_position(self, entity, values, dofs_idx_local) -> None:
        policy_match = self._policy_hand_for_entity(entity)
        if policy_match is None:
            return super().control_entity_dofs_position(entity, values, dofs_idx_local)
        side, policy_hand = policy_match
        control_hand = self.real_control_hands.get(side)
        if control_hand is None or not isinstance(control_hand.robot, RealRobotEntity):
            raise ValueError(f"Missing hidden real control hand for side {side!r}.")
        control_qpos = self._control_qpos_from_policy_target(
            side=side,
            policy_hand=policy_hand,
            control_hand=control_hand,
            values=values,
            dofs_idx_local=dofs_idx_local,
        )
        self._latest_real_control_qpos_by_side[side] = control_qpos.detach().clone()
        if control_hand.robot.robot_controller is None:
            if not self.real_cfg.initialize_robot:
                return
            raise RuntimeError(
                "Cannot send real policy hand targets because robot_controller "
                "is not initialized."
            )
        return super().control_entity_dofs_position(
            control_hand.robot,
            control_qpos,
            control_hand.dof_map,
        )

    def _control_qpos_from_policy_target(
        self,
        *,
        side: str,
        policy_hand: Hand,
        control_hand: Hand,
        values,
        dofs_idx_local,
    ) -> torch.Tensor:
        policy_qpos = torch.as_tensor(values, dtype=torch.float32, device="cpu")
        expected_policy_shape = (self.env.num_envs, len(policy_hand.dof_map))
        if tuple(policy_qpos.shape) != expected_policy_shape:
            raise ValueError(
                "Expected policy target qpos in policy hand.dof_map order with "
                f"shape {expected_policy_shape}, got {tuple(policy_qpos.shape)}."
            )
        dofs_idx = np.asarray(dofs_idx_local, dtype=np.int64).reshape(-1)
        if dofs_idx.tolist() != list(policy_hand.dof_map):
            raise ValueError(
                "Real multi-object action conversion expects full policy hand.dof_map "
                f"targets. Expected {policy_hand.dof_map}, got {dofs_idx.tolist()}."
            )
        cached_ik = self._latest_real_ik_qpos_by_side.get(side)
        if cached_ik is None:
            raise RuntimeError(
                "Missing cached hidden franka_xhand IK solution. "
                "ActionManager must call solve_entity_ik() before "
                "control_entity_dofs_position() for real policy hands."
            )
        cached_ik = torch.as_tensor(cached_ik, dtype=torch.float32, device="cpu")
        expected_control_shape = (self.env.num_envs, len(control_hand.dof_map))
        if tuple(cached_ik.shape) != expected_control_shape:
            raise ValueError(
                "Cached hidden IK qpos shape mismatch: expected "
                f"{expected_control_shape}, got {tuple(cached_ik.shape)}."
            )

        policy_joint_target_by_name = {
            joint_name: policy_qpos[:, col]
            for col, joint_name in enumerate(policy_hand.joint_names)
        }
        control_qpos = self._control_hand_qpos_from_real_state(side).to("cpu")
        xhand_joint_names = set(control_hand.robot.xhand_joint_names)
        helper_root_joint_names = set(control_hand.robot.helper_root_joint_names)
        franka_joint_names = set(FRANKA_JOINT_NAMES)
        for col, joint_name in enumerate(control_hand.joint_names):
            if joint_name in franka_joint_names:
                control_qpos[:, col] = cached_ik[:, col]
                continue
            if joint_name in xhand_joint_names:
                policy_value = policy_joint_target_by_name.get(joint_name)
                if policy_value is None:
                    raise KeyError(
                        f"Policy hand target is missing XHand joint {joint_name!r} "
                        f"required by hidden control hand {side!r}."
                    )
                control_qpos[:, col] = policy_value
                continue
            if joint_name in helper_root_joint_names:
                continue
            raise NotImplementedError(
                f"Hidden control hand joint {joint_name!r} cannot be populated "
                "from the cached Franka IK solution or policy XHand target."
            )
        return control_qpos

    def close(self) -> None:
        super().close()
        for hand in self.real_control_hands.values():
            if isinstance(hand.robot, RealRobotEntity):
                hand.robot.close()
