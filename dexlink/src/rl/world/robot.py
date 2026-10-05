from __future__ import annotations

import importlib
import re
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.utils.retargeting.math_utils import matrix_from_quat, rot6d_from_matrix
from dexlink.utils.retargeting.mesh_utils import (
    construct_pytorch_mesh,
    sample_kpts_with_normals,
)
from dexlink.utils.retargeting.path_utils import process_urdf

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import RobotObsCfg
    from dexlink.src.rl.world.env import Env
    from dexlink.src.rl.world.physics_sim import PhysicsSim


def get_hand_specific_cfg(name="inspire_hand"):
    hand_prefix = name.replace("_hand", "")  # xhand still stays xhand
    try:
        module_name = f"dexlink.src.retargeting.data.hand_cfgs.{hand_prefix}"
        hand_cfg_module = importlib.import_module(module_name)
        # Assume the hand cfgs all ends with *_CFGs inside the module
        cfg_attr = next(
            attr for attr in dir(hand_cfg_module) if attr.upper().endswith("_CFGS")
        )
        hand_cfgs = getattr(hand_cfg_module, cfg_attr)
        assert isinstance(hand_cfgs, dict), f"{cfg_attr} should be a dict"
        return hand_cfgs

    except (ImportError, StopIteration, AttributeError, KeyError) as e:
        raise ValueError(f"Invalid hand name or configuration: {name} - {e}")


# same class could be used for both left and right hand, depending upon the passed config
class Hand:
    def __init__(
        self,
        physics: PhysicsSim,
        robo_cfg,
        num_envs: int,
        device,
        obs_cfg: RobotObsCfg,
        env: Env,
        name: str,
        base_pose: np.ndarray | None = None,
    ):

        # process the URDF path for cluster use
        robo_cfg["urdf_path"] = str(
            process_urdf(robo_cfg["urdf_path"], append_jobname=True)
        )
        self.device = device
        self.physics = physics
        self.env = env

        self.name = name
        self.num_envs = num_envs
        self.robo_cfg = robo_cfg
        self.obs_cfg = obs_cfg
        self.base_pose = self._resolve_base_pose(base_pose)
        self.urdf_inst = URDF.load(robo_cfg["urdf_path"])
        prefix_6d = "L" if "left" in robo_cfg["urdf_path"] else "R"
        self.root_joints = [
            f"{prefix_6d}_forearm_tx_link_joint",
            f"{prefix_6d}_forearm_ty_link_joint",
            f"{prefix_6d}_forearm_tz_link_joint",
            f"{prefix_6d}_forearm_roll_link_joint",
            f"{prefix_6d}_forearm_pitch_link_joint",
            f"{prefix_6d}_forearm_yaw_link_joint",
        ]
        self.panda_joints = [f"panda_joint{i}" for i in range(1, 8)]

        self.joints = [j for j in self.urdf_inst.joints if j.joint_type != "fixed"]
        self.joint_names = [j.name for j in self.joints]
        for j in self.joints:
            if j.name in self.root_joints:
                continue
            print(
                f"Joint name: {j.name}, type: {j.joint_type}, limit: ({j.limit.lower}, {j.limit.upper}), links: {j.parent} -> {j.child}"
            )

        self.panda_joints = [j for j in self.panda_joints if j in self.joint_names]
        self.root_joints = [j for j in self.root_joints if j in self.joint_names]
        self.wrist_control_joints = (
            list(self.root_joints)
            if len(self.root_joints) > 0
            else list(self.panda_joints)
        )

        self.root_joints_idx = [self.joint_names.index(rj) for rj in self.root_joints]
        self.wrist_link_name = robo_cfg["wrist_link_name"]
        self.hand_tree_link_names = self._compute_subtree_link_names(
            self.wrist_link_name
        )

        self.tactile_link_names = robo_cfg["all_tactile_links"]

        # links important for human configuration matching
        self.human_robot_link_map = robo_cfg["human_robot_link_map"]
        self.human_joint_names = list(self.human_robot_link_map.keys())
        self.human_mapped_link_names = [
            self.human_robot_link_map[jname] for jname in self.human_joint_names
        ]
        self.finger_joint_names = [
            joint.name
            for joint in self.joints
            if (joint.name not in self.panda_joints)
            and (joint.name not in self.root_joints)
        ]
        self.finger_mapped_link_names = [
            joint.child
            for joint in self.joints
            if joint.name in self.finger_joint_names
        ]
        self.hand_mapped_link_names = self.human_mapped_link_names
        base_pos = tuple(self.base_pose[:3, 3].tolist())
        base_quat = R.from_matrix(self.base_pose[:3, :3]).as_quat()
        base_quat = (
            base_quat[3],
            base_quat[0],
            base_quat[1],
            base_quat[2],
        )  # xyzw to wxyz

        self.robot = physics.add_rigid_urdf_entity(
            urdf_path=robo_cfg["urdf_path"],
            name=name,
            fixed=True,
            recompute_inertia=False,
            collision=True,
            convexify=True,
            pos=base_pos,
            quat=base_quat,
            links_to_keep=self.hand_mapped_link_names,
            gravity_compensation=1.0,
            friction=1.2,
        )

        self.dof_map = self._dof_map()
        self._joint_col_by_dof_idx = {
            dof_idx: joint_col for joint_col, dof_idx in enumerate(self.dof_map)
        }
        self._initialize_dof_limits()
        self.panda_joint_idx = [
            self.joint_names.index(jname) for jname in self.panda_joints
        ]
        self.wrist_control_joint_idx = [
            self.joint_names.index(jname) for jname in self.wrist_control_joints
        ]
        self.hand_joint_idx = [
            idx
            for idx, jname in enumerate(self.joint_names)
            if (jname not in self.panda_joints) and (jname not in self.root_joints)
        ]
        (
            self.default_dof_kp,
            self.default_dof_kp_available,
            self.default_dof_kv,
            self.default_dof_kv_available,
        ) = self._default_dof_gains()
        (
            self.default_dof_friction,
            self.default_dof_friction_available,
        ) = self._default_dof_friction()
        self.env_dof_stiffness = np.repeat(
            self.default_dof_kp[None, :], self.num_envs, axis=0
        )
        self.env_dof_damping = np.repeat(
            self.default_dof_kv[None, :], self.num_envs, axis=0
        )
        self.env_dof_friction = np.repeat(
            self.default_dof_friction[None, :], self.num_envs, axis=0
        )

        velocity_controlled_joints = robo_cfg.get("velocity_controlled_joints", [])
        # remaining are all position controlled
        self.position_controlled_joints = [
            jname
            for jname in self.joint_names
            if jname not in velocity_controlled_joints
        ]
        self.position_control_mask = [
            True if jname in self.position_controlled_joints else False
            for jname in self.joint_names
        ]
        self.velocity_control_mask = [
            True if jname in velocity_controlled_joints else False
            for jname in self.joint_names
        ]

        # alias for backwards compatibility
        self.human_mapped_link_indices_local = [
            self.physics.get_link_idx_local(self.robot, name)
            for name in self.human_mapped_link_names
        ]
        self.finger_mapped_link_indices_local = [
            self.physics.get_link_idx_local(self.robot, name)
            for name in self.finger_mapped_link_names
        ]
        self.tactile_links_global_indices = [
            self.physics.get_link_idx(self.robot, name)
            for name in self.tactile_link_names
        ]
        obs_terms = []
        if self.obs_cfg.use_jpos:
            obs_terms.append(("jpos", len(self.dof_map), self._jpos_obs))
        if self.obs_cfg.use_jvel:
            obs_terms.append(("jvel", len(self.dof_map), self._jvel_obs))
        if self.obs_cfg.use_wrist_pose:
            obs_terms.append(("wrist_pose", 9, self._wrist_pose_obs))
        if self.obs_cfg.use_lpos:
            obs_terms.append(
                ("lpos", 3 * len(self.human_mapped_link_names), self._lpos_obs)
            )  # x,y,z for each link
        if self.obs_cfg.use_lvel:
            obs_terms.append(
                ("lvel", 3 * len(self.human_mapped_link_names), self._lvel_obs)
            )  # vx,vy,vz for each link

        self.obs_terms = [term[0] for term in obs_terms]
        self.obs_dims = [term[1] for term in obs_terms]
        self.obs_fns = [term[2] for term in obs_terms]
        self._obs_dim = sum(self.obs_dims)
        self.hand_surface_links = []
        self.hand_surface_link_names = []
        self.hand_surface_area_ratios = []
        self._hand_surface_link_idx_by_name: dict[str, int] = {}
        self._initialize_surface_samples()

        # if physics, the robot is controlled as per the actuator models

    def canonical_finger_joint_values(
        self, *, device=None, dtype=torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qpos_cfg = self.robo_cfg.get("default_qpos", None)
        if qpos_cfg is None:
            qpos_cfg = self.robo_cfg.get("open_qpos", None)
        if not isinstance(qpos_cfg, dict):
            raise ValueError(
                f"Hand {self.name!r} requires robo_cfg['default_qpos'] or "
                "robo_cfg['open_qpos'] to construct canonical finger references."
            )

        indices = []
        values = []
        for joint_name in self.finger_joint_names:
            if joint_name not in qpos_cfg:
                raise ValueError(
                    f"Canonical finger qpos for hand {self.name!r} is missing "
                    f"joint {joint_name!r}."
                )
            indices.append(self.joint_names.index(joint_name))
            values.append(float(qpos_cfg[joint_name]))

        return (
            torch.as_tensor(indices, dtype=torch.long, device=device or self.device),
            torch.as_tensor(values, dtype=dtype, device=device or self.device),
        )

    def _resolve_base_pose(self, base_pose: np.ndarray | None) -> np.ndarray:
        if base_pose is None:
            return np.eye(4, dtype=np.float32)
        base_pose = np.asarray(base_pose, dtype=np.float32)
        if base_pose.shape != (4, 4):
            raise ValueError(
                f"Expected base_pose to have shape (4, 4), got {base_pose.shape}."
            )
        return base_pose.copy()

    def _compute_subtree_link_names(self, root_link_name: str) -> list[str]:
        if root_link_name not in self.urdf_inst.link_map:
            raise ValueError(
                f"Expected subtree root link {root_link_name!r} to exist in the URDF."
            )
        child_links_by_parent = {}
        for joint in self.urdf_inst.joints:
            child_links_by_parent.setdefault(joint.parent, []).append(joint.child)

        subtree_link_names = []
        stack = [root_link_name]
        visited = set()
        while stack:
            link_name = stack.pop()
            if link_name in visited:
                continue
            visited.add(link_name)
            subtree_link_names.append(link_name)
            children = child_links_by_parent.get(link_name, [])
            for child_link_name in reversed(children):
                stack.append(child_link_name)
        return subtree_link_names

    def _initialize_surface_samples(self) -> None:
        task_cfg = self.env.cfg.task_cfg
        if task_cfg.max_hand_pts_per_link <= 0:
            raise ValueError(
                "Expected task_cfg.max_hand_pts_per_link to be a positive integer, "
                f"got {task_cfg.max_hand_pts_per_link}."
            )
        link_names = []
        link_objs = []
        link_meshes = []
        link_areas = []
        for link in self.urdf_inst.links:
            visual_meshes = []
            for visual in getattr(link, "visuals", None) or []:
                geom = getattr(visual, "geometry", None)
                if geom is None:
                    continue
                visual_pose = (
                    visual.origin.copy()
                    if visual.origin is not None
                    else np.eye(4, dtype=np.float64)
                )
                mesh_field = getattr(geom, "mesh", None)
                scale = (
                    getattr(mesh_field, "scale", None)
                    if mesh_field is not None
                    else None
                )
                if scale is not None:
                    scale_pose = np.eye(4, dtype=np.float64)
                    scale_pose[:3, :3] = np.diag(scale)
                    visual_pose = visual_pose @ scale_pose
                for mesh in getattr(geom, "meshes", None) or []:
                    mesh_local = mesh.copy()
                    mesh_local.apply_transform(visual_pose)
                    visual_meshes.append(mesh_local)
            if len(visual_meshes) == 0:
                continue
            combined_link_mesh = trimesh.util.concatenate(visual_meshes)
            link_names.append(link.name)
            link_objs.append(link)
            link_meshes.append(combined_link_mesh)
            link_areas.append(float(combined_link_mesh.area))
        if len(link_meshes) == 0:
            raise ValueError("Expected at least one hand surface visual mesh.")
        max_link_area = max(link_areas)
        if max_link_area <= 0.0:
            raise ValueError(
                "Expected at least one hand surface visual mesh with positive area."
            )

        dense_hand_surface_kpts = []
        dense_hand_surface_normals = []
        for link_mesh, link_area in zip(link_meshes, link_areas):
            pytorch_mesh = construct_pytorch_mesh(
                meshes=[link_mesh], device=self.device
            )
            dense_kpts, dense_normals = sample_kpts_with_normals(
                pytorch_mesh,
                max_pts=task_cfg.dense_hand_pts_per_link,
            )
            dense_hand_surface_kpts.append(dense_kpts[0])
            dense_hand_surface_normals.append(dense_normals[0])

        for link_name, kpts, normals in zip(
            link_names, dense_hand_surface_kpts, dense_hand_surface_normals
        ):
            if kpts.shape != normals.shape:
                raise ValueError(
                    "Expected hand surface keypoints and normals to have matching "
                    f"shapes for link {link_name!r}, got {kpts.shape} and {normals.shape}."
                )

        self.hand_surface_links = link_objs
        self.hand_surface_link_names = link_names
        self.hand_surface_area_ratios = [
            float(link_area / max_link_area) for link_area in link_areas
        ]
        self.dense_hand_surface_kpts = dense_hand_surface_kpts
        self.dense_hand_surface_normals = dense_hand_surface_normals
        self._hand_surface_link_idx_by_name = {
            link_name: idx for idx, link_name in enumerate(self.hand_surface_link_names)
        }

    def _resolve_surface_link_indices(
        self, links_to_include: list[str] | None
    ) -> list[int]:
        if links_to_include is None:
            selected_link_indices = list(range(len(self.hand_surface_links)))
        else:
            selected_link_indices = []
            missing_link_names = []
            for link_name in links_to_include:
                link_idx = self._hand_surface_link_idx_by_name.get(link_name)
                if link_idx is None:
                    missing_link_names.append(link_name)
                else:
                    selected_link_indices.append(link_idx)
            if missing_link_names:
                raise ValueError(
                    "Requested links_to_include are not available in sampled visual meshes: "
                    f"{missing_link_names}. Available links: {self.hand_surface_link_names}."
                )
        if len(selected_link_indices) == 0:
            raise ValueError("Expected links_to_include to select at least one link.")
        return selected_link_indices

    def _get_robot_pcd_sample_counts(
        self, selected_link_indices: list[int]
    ) -> list[int]:
        max_pts_per_link = self.env.cfg.task_cfg.max_hand_pts_per_link
        selected_ratios = [
            self.hand_surface_area_ratios[link_idx]
            for link_idx in selected_link_indices
        ]
        max_selected_ratio = max(selected_ratios)

        sample_counts = []
        for link_idx, ratio in zip(selected_link_indices, selected_ratios):
            n_pts = max(1, int(round((ratio / max_selected_ratio) * max_pts_per_link)))
            available_pts = self.dense_hand_surface_kpts[link_idx].shape[0]
            if n_pts > available_pts:
                raise ValueError(
                    "Expected dense hand surface samples to cover requested robot pcd points, "
                    f"but link {self.hand_surface_link_names[link_idx]!r} only has "
                    f"{available_pts} dense samples for requested {n_pts} points."
                )
            sample_counts.append(n_pts)
        return sample_counts

    def get_robot_pcd_n_pts(self, links_to_include: list[str] | None = None) -> int:
        selected_link_indices = self._resolve_surface_link_indices(links_to_include)
        return sum(self._get_robot_pcd_sample_counts(selected_link_indices))

    # def root_dof_map(self):
    #     res = [self.robot.get_joint(jname).dofs_idx_local for jname in self.root_joints]
    #     assert all([len(v) == 1 for v in res]), "Each joint should map to exactly one DOF"
    #     return [v[0] for v in res]

    def _dof_map(self):
        res = [
            self.physics.get_joint_dofs_idx_local(self.robot, j.name)
            for j in self.joints
        ]
        assert all(
            [len(v) == 1 for v in res]
        ), "Each joint should map to exactly one DOF"
        return [v[0] for v in res]

    def _default_dof_gains(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kp = np.zeros(len(self.joints), dtype=np.float32)
        kp_available = np.zeros(len(self.joints), dtype=bool)
        kv = np.zeros(len(self.joints), dtype=np.float32)
        kv_available = np.zeros(len(self.joints), dtype=bool)
        for act_info in self.robo_cfg["actuators"].values():
            joint_regex_names = act_info["joint_exprs"]
            for j_idx, joint in enumerate(self.joints):
                if any(re.match(jregex, joint.name) for jregex in joint_regex_names):
                    kp[j_idx] = float(act_info["kp"])
                    kp_available[j_idx] = True
                    kv[j_idx] = float(act_info["kv"])
                    kv_available[j_idx] = True
        return kp, kp_available, kv, kv_available

    def _default_dof_friction(self) -> tuple[np.ndarray, np.ndarray]:
        friction = np.zeros(len(self.joints), dtype=np.float32)
        available = np.zeros(len(self.joints), dtype=bool)
        for j_idx, joint in enumerate(self.joints):
            joint_dynamics = getattr(joint, "dynamics", None)
            if joint_dynamics is None or joint_dynamics.friction is None:
                continue
            friction[j_idx] = float(joint_dynamics.friction)
            available[j_idx] = True
        return friction, available

    def _initialize_dof_limits(self) -> None:
        joint_limit_soft_margin = float(self.env.cfg.real_cfg.joint_limit_soft_margin)
        if not (0.0 < joint_limit_soft_margin <= 1.0):
            raise ValueError(
                "Expected env.real_cfg.joint_limit_soft_margin in (0, 1], "
                f"got {joint_limit_soft_margin}."
            )
        raw_limits_by_dof_idx: dict[int, tuple[float, float]] = {}
        effective_limits_by_dof_idx: dict[int, tuple[float, float]] = {}
        for joint_col, (joint, dof_idx) in enumerate(zip(self.joints, self.dof_map)):
            if joint.limit is None:
                raise ValueError(
                    f"Joint {joint.name!r} in {self.robo_cfg['urdf_path']!r} has no URDF limit information."
                )
            if joint.limit.lower is None or joint.limit.upper is None:
                raise ValueError(
                    f"Joint {joint.name!r} in {self.robo_cfg['urdf_path']!r} is missing lower/upper limits."
                )
            lower = float(joint.limit.lower)
            upper = float(joint.limit.upper)
            center = 0.5 * (lower + upper)
            half_range = 0.5 * (upper - lower) * joint_limit_soft_margin
            raw_limits_by_dof_idx[dof_idx] = (lower, upper)
            effective_limits_by_dof_idx[dof_idx] = (
                center - half_range,
                center + half_range,
            )
        if len(raw_limits_by_dof_idx) != len(self.dof_map):
            raise ValueError(
                "Expected one DOF limit entry per hand DOF, got "
                f"{len(raw_limits_by_dof_idx)} entries for {len(self.dof_map)} DOFs."
            )
        self._raw_dof_limits_by_dof_idx = raw_limits_by_dof_idx
        self._effective_dof_limits_by_dof_idx = effective_limits_by_dof_idx

    def get_dof_limits(
        self,
        dof_indices: list[int] | np.ndarray,
        *,
        effective: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        limits_by_dof_idx = (
            self._effective_dof_limits_by_dof_idx
            if effective
            else self._raw_dof_limits_by_dof_idx
        )
        lower = []
        upper = []
        for dof_idx in dof_indices:
            dof_idx_int = int(dof_idx)
            limits = limits_by_dof_idx.get(dof_idx_int)
            if limits is None:
                raise ValueError(
                    f"Unknown DOF index {dof_idx_int} for hand {self.name!r}. "
                    f"Available indices: {sorted(limits_by_dof_idx)}."
                )
            lower_val, upper_val = limits
            lower.append(lower_val)
            upper.append(upper_val)
        lower_tensor = torch.tensor(lower, device=device, dtype=dtype).unsqueeze(0)
        upper_tensor = torch.tensor(upper, device=device, dtype=dtype).unsqueeze(0)
        return lower_tensor.expand(self.num_envs, -1), upper_tensor.expand(
            self.num_envs, -1
        )

    def act_dim(self, act_cfg) -> int:
        """Return action dimension: joint pos targets."""
        # if position controller or velocity controller
        if act_cfg.controller in ["position", "velocity", "hybrid"]:
            return len(self.joints)
        # impedance controller
        dim = len(self.joints)  # joint positions
        if act_cfg.compute_stiffness:
            dim += len(self.joints)  # stiffness
        if act_cfg.compute_damping_ratio:
            dim += len(self.joints)  # damping ratio
        return dim

    def _debug_print_joints_out_of_limits(self, jvals):
        # only for the first env for debugging
        jvals = jvals[0]
        dofs_idx = self.dof_map
        lower_limit, upper_limit = self.physics.get_entity_dofs_limit(
            self.robot, dofs_idx_local=dofs_idx
        )
        lower_limit = lower_limit[0]
        upper_limit = upper_limit[0]

        lb = jvals >= lower_limit
        ub = jvals <= upper_limit
        out_of_limits = ~(lb & ub)
        # convert to list
        out_of_limits_indices = torch.nonzero(out_of_limits).cpu().numpy().tolist()

        # print names for joints that are out of limits
        for idx in out_of_limits_indices:
            joint_name = self.joints[idx].name
            print(
                f"[Hand.set_robot_state] Joint '{joint_name}' (index {idx}) is out of limits: value={jvals[idx]:.3f}, lower={lower_limit[idx]:.3f}, upper={upper_limit[idx]:.3f}"
            )

    def set_robot_state(
        self,
        jpos: torch.Tensor | None = None,
        jvel: torch.Tensor | None = None,
        env_idx=None,
    ):

        if env_idx is None:
            envs_idx = None
            n_envs = self.num_envs
        elif isinstance(env_idx, (int, np.integer)):
            raise ValueError(
                f"env_idx should be None or a list/array of indices, not a single integer. Got: {env_idx}"
            )
            envs_idx = np.array([int(env_idx)], dtype=np.int64)
            n_envs = 1
        else:
            envs_idx = np.asarray(env_idx, dtype=np.int64)
            n_envs = len(envs_idx)

        if jpos is not None:

            dofs_idx = self.dof_map
            lower_limit, upper_limit = self.physics.get_entity_dofs_limit(
                self.robot, dofs_idx_local=dofs_idx
            )
            lower_limit = lower_limit[envs_idx, :]
            upper_limit = upper_limit[envs_idx, :]

            jvals = torch.as_tensor(jpos, dtype=torch.float32)
            if len(jvals.shape) == 1:
                raise ValueError(
                    f"jpos should have shape (num_envs, num_dofs) or (num_dofs,), but got shape {jvals.shape}"
                )
                jvals = jvals[None, :]
                jvals = jvals.repeat(n_envs, 1)
            else:
                if (
                    jvals.shape == (self.num_envs, len(self.joints))
                    and envs_idx is not None
                ):
                    jvals = jvals[envs_idx]
                assert jvals.shape == (
                    n_envs,
                    len(self.joints),
                ), f"jpos shape mismatch: expected {(n_envs, len(self.joints))}, got {jvals.shape}"

            jvals = torch.clip(jvals, lower_limit, upper_limit)
            self._debug_print_joints_out_of_limits(jvals)

            # print(
            #     "[Hand.set_robot_state] "
            #     f"name={self.name}, "
            #     f"envs_idx_shape={None if envs_idx is None else envs_idx.shape}, "
            #     f"envs_idx={envs_idx}, "
            #     f"jpos_input_shape={tuple(jvals.shape)}, "
            #     f"num_joints={len(self.joints)}"
            # )

            self.physics.set_entity_dofs_position(
                self.robot, jvals, dofs_idx, envs_idx=envs_idx
            )

        if jvel is not None:
            jvals = torch.as_tensor(jvel, dtype=torch.float32)
            if len(jvals.shape) == 1:
                raise ValueError(
                    f"jvel should have shape (num_envs, num_dofs) or (num_dofs,), but got shape {jvals.shape}"
                )
                jvals = jvals[None, :]
                jvals = jvals.repeat(n_envs, 1)
            else:
                if (
                    jvals.shape == (self.num_envs, len(self.joints))
                    and envs_idx is not None
                ):
                    jvals = jvals[envs_idx]
                assert jvals.shape == (
                    n_envs,
                    len(self.joints),
                ), f"jvel shape mismatch: expected {(n_envs, len(self.joints))}, got {jvals.shape}"
        else:
            jvals = torch.zeros((n_envs, len(self.joints)), dtype=torch.float32)
        dofs_idx = self.dof_map
        if len(jvals.shape) == 1:
            jvals = jvals[None, :]  # (1, num_dofs)
            jvals = jvals.repeat(n_envs, 1)  # (num_envs, num_dofs)
        # print(
        #     "[Hand.set_robot_state] "
        #     f"name={self.name}, "
        #     f"velocity_input_shape={tuple(jvals.shape)}, "
        #     f"envs_idx_shape={None if envs_idx is None else envs_idx.shape}, "
        #     f"envs_idx={envs_idx}"
        # )
        self.physics.set_entity_dofs_velocity(
            self.robot, jvals, dofs_idx, envs_idx=envs_idx
        )

    def random_action(self, env_ids=None) -> torch.Tensor:
        """Sample uniformly within joint limits; returns (n_envs_sel, num_dofs)."""
        n_envs = self.num_envs if env_ids is None else len(env_ids)
        dofs_idx = self.dof_map
        lower_limit, upper_limit = self.physics.get_entity_dofs_limit(
            self.robot, dofs_idx_local=dofs_idx
        )
        sampler = torch.distributions.Uniform(lower_limit, upper_limit)
        jvals = sampler.sample((n_envs,))

        return jvals

    def sample_actions_near_object(
        self,
        obj_pos: torch.Tensor,
        env_ids=None,
        lower: float = 0.5,
        upper: float = 1.0,
        opposite_signs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample a random action biased near the object's root position.

        - Starts from a uniform random action within joint limits.
        - Overrides forearm tx/ty/tz joints to be around the object's position with random offsets.
        - If `opposite_signs` is provided, uses those signs (+1/-1) for x/y/z to coordinate with another hand.

        obj_pos: Tensor of shape (n_envs_sel, 3) corresponding to selected envs.
        env_ids: Optional indices of selected envs; affects random_action batching.
        lower/upper: Bounds for offset magnitude.
        opposite_signs: Optional tensor of shape (n_envs_sel, 3) with values in {+1, -1}.
        """
        # number of selected envs inferred from obj_pos; not directly needed
        # n_envs = obj_pos.shape[0]

        # Base random actions within joint limits
        jvals = self.random_action(env_ids=env_ids)

        scale = upper - lower
        reference_x = obj_pos[:, 0]
        reference_y = obj_pos[:, 1]
        reference_z = obj_pos[:, 2]

        if opposite_signs is not None:
            # Use provided signs for x/y/z
            x_sign = opposite_signs[:, 0].float()
            y_sign = opposite_signs[:, 1].float()
            # z_sign reserved if enabling z offsets later
            # z_sign = opposite_signs[:, 2].float()
        else:
            # Random +/- 1 signs
            x_sign = (
                torch.randint(0, 2, reference_x.shape, device=reference_x.device) * 2
                - 1
            ).float()
            y_sign = (
                torch.randint(0, 2, reference_y.shape, device=reference_y.device) * 2
                - 1
            ).float()
            # z_sign reserved if enabling z offsets later
            # z_sign = (torch.randint(0, 2, reference_z.shape, device=reference_z.device) * 2 - 1).float()

        x_offset = (torch.rand_like(reference_x) * scale + lower) * x_sign
        y_offset = (torch.rand_like(reference_y) * scale + lower) * y_sign
        # Currently no z offset to preserve height; keep option available
        z_offset = (torch.rand_like(reference_z) * scale + lower) * (
            -0.1
        )  # below object
        # z_vals = torch.ones_like(reference_z) * 0.1  # keep at fixed height above ground
        # z_vals = reference_z  # keep at object's z position

        # Resolve forearm joint names for this hand
        prefix = (
            "L"
            if any(
                "left" in s
                for s in [
                    self.robo_cfg.get("urdf_path", ""),
                    self.robo_cfg.get("name", ""),
                ]
            )
            else "R"
        )
        j_tx = f"{prefix}_forearm_tx_link_joint"
        j_ty = f"{prefix}_forearm_ty_link_joint"
        j_tz = f"{prefix}_forearm_tz_link_joint"

        # override columns corresponding to forearm translation joints
        idx_tx = self.joint_names.index(j_tx)
        idx_ty = self.joint_names.index(j_ty)
        idx_tz = self.joint_names.index(j_tz)
        jvals[:, idx_tx] = reference_x + x_offset
        jvals[:, idx_ty] = reference_y + y_offset
        jvals[:, idx_tz] = reference_z + z_offset
        # jvals[:, idx_tz] = z_vals

        return jvals

    @property
    def obs_dim(self):
        """Return observation dimension: joint pos + joint vel."""
        return self._obs_dim

    def obs_named_view(self, robot_obs: torch.Tensor):
        splits = torch.split(robot_obs, self.obs_dims, dim=-1)
        return dict(zip(self.obs_terms, splits))

    def _jpos_obs(self):
        jpos = self.physics.get_entity_dofs_position(self.robot, self.dof_map).detach()

        return jpos

    def _jvel_obs(self):
        jvel = self.physics.get_entity_dofs_velocity(self.robot, self.dof_map).detach()
        return jvel

    def _jacc_obs(self):
        return self.physics.get_hand_dof_accelerations(self, self.dof_map).detach()

    def _lpos_obs(self):
        link_pos = self.physics.get_entity_links_pos(
            self.robot,
            links_idx_local=self.human_mapped_link_indices_local,
            ref="link_origin",
        )
        link_pos = link_pos.view(self.num_envs, 3 * len(self.human_mapped_link_names))

        return link_pos

    def _lvel_obs(self):
        # Genesis API naming can vary by version; try known getters.

        link_vel = self.physics.get_entity_links_vel(
            self.robot,
            links_idx_local=self.human_mapped_link_indices_local,
            ref="link_origin",
        )
        link_vel = link_vel.view(self.num_envs, 3 * len(self.human_mapped_link_names))
        return link_vel

    def _wrist_pose_obs(self):
        wrist_pos = self.physics.get_entity_link_pos(
            self.robot,
            link_name=self.wrist_link_name,
            ref="link_origin",
        ).squeeze(1)
        wrist_quat = self.physics.get_entity_link_quat(
            self.robot,
            link_name=self.wrist_link_name,
        ).squeeze(1)
        wrist_rotmat = matrix_from_quat(wrist_quat)
        wrist_rot6d = rot6d_from_matrix(wrist_rotmat)
        return torch.cat([wrist_pos, wrist_rot6d], dim=-1)

    def get_obs(self, rot_format: str = "euler"):
        """Return current joint pos/vel as tensors.
        jpos, jvel: (num_envs, num_dofs)
        """
        obs = []
        for obs_fn in self.obs_fns:
            obs.append(obs_fn())

        return torch.cat(obs, dim=-1)

    def get_ground_truth_info(self):
        # ideally return the object contact map profile
        # and finger contacts
        return {}

    def get_robot_pcd(
        self,
        jpos: torch.Tensor,
        links_to_include: list[str] | None = None,
        use_dense_points: bool = False,
    ) -> torch.Tensor:
        # given the jpos, get the full point cloud.
        if jpos.ndim != 2:
            raise ValueError(
                f"Expected jpos to have shape (num_envs, num_dofs), got {tuple(jpos.shape)}."
            )
        if jpos.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected jpos batch dimension {self.num_envs}, got {jpos.shape[0]}."
            )
        expected_num_dofs = len(self.dof_map)
        if jpos.shape[1] != expected_num_dofs:
            raise ValueError(
                "Expected jpos columns to match hand.dof_map order with shape "
                f"({self.num_envs}, {expected_num_dofs}), got {tuple(jpos.shape)}."
            )

        full_jpos = torch.zeros(
            (self.num_envs, len(self.joints)),
            device=jpos.device,
            dtype=jpos.dtype,
        )
        for local_col, joint_col in enumerate(
            self._joint_col_by_dof_idx[dof_idx] for dof_idx in self.dof_map
        ):
            full_jpos[:, joint_col] = jpos[:, local_col]

        joint_dict = {
            joint.name: full_jpos[:, joint_idx].detach().cpu().numpy()
            for joint_idx, joint in enumerate(self.joints)
        }
        link_fk = self.urdf_inst.link_fk_batch(cfgs=joint_dict)
        base_pose = torch.as_tensor(
            self.base_pose, device=jpos.device, dtype=jpos.dtype
        ).unsqueeze(0)
        base_pose = base_pose.expand(self.num_envs, -1, -1)

        selected_link_indices = self._resolve_surface_link_indices(links_to_include)
        sample_counts = self._get_robot_pcd_sample_counts(selected_link_indices)

        transformed_points = []
        for link_idx, n_pts in zip(selected_link_indices, sample_counts):
            link = self.hand_surface_links[link_idx]
            dense_local_kpts = self.dense_hand_surface_kpts[link_idx].to(
                device=jpos.device, dtype=jpos.dtype
            )
            sample_indices = torch.randperm(
                dense_local_kpts.shape[0], device=jpos.device
            )[:n_pts]
            local_kpts = dense_local_kpts[sample_indices]
            poses_np = link_fk.get(link)
            if poses_np is None:
                raise ValueError(
                    f"URDF FK did not return a pose for hand surface link {link.name!r}."
                )
            link_pose = torch.as_tensor(
                poses_np,
                device=jpos.device,
                dtype=jpos.dtype,
            )
            world_pose = torch.bmm(base_pose, link_pose)

            n_pts = local_kpts.shape[0]
            local_kpts_h = torch.cat(
                [
                    local_kpts,
                    torch.ones(n_pts, 1, device=jpos.device, dtype=jpos.dtype),
                ],
                dim=1,
            ).transpose(0, 1)
            world_kpts = world_pose @ local_kpts_h.unsqueeze(0)
            transformed_points.append(world_kpts.transpose(1, 2)[..., :3])

        return torch.cat(transformed_points, dim=1)
