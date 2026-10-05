from __future__ import annotations

import importlib
import re
from typing import TYPE_CHECKING

import genesis as gs
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.utils.retargeting.path_utils import process_urdf

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import RobotObsCfg
    from dexlink.src.rl.sim.env import Env


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
        scene,
        robo_cfg,
        num_envs: int,
        device,
        obs_cfg: RobotObsCfg,
        env: Env,
        name: str,
        base_pose: np.ndarray | None = None,
    ):

        # process the URDF path for cluster use
        robo_cfg["urdf_path"] = str(process_urdf(robo_cfg["urdf_path"]))
        self.device = device
        self.scene = scene
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

        self.root_joints_idx = [self.joint_names.index(rj) for rj in self.root_joints]

        self.tactile_link_names = robo_cfg["all_tactile_links"]

        # links important for human configuration matching
        self.human_robot_link_map = robo_cfg["human_robot_link_map"]
        self.human_joint_names = list(self.human_robot_link_map.keys())
        self.human_mapped_link_names = [
            self.human_robot_link_map[jname] for jname in self.human_joint_names
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

        self.robot = scene.add_entity(
            gs.morphs.URDF(
                file=robo_cfg["urdf_path"],
                fixed=True,
                recompute_inertia=False,
                collision=True,
                convexify=True,
                pos=base_pos,
                quat=base_quat,
                links_to_keep=self.hand_mapped_link_names,
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0, friction=1.2),
        )

        self.dof_map = self._dof_map()
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
            self.robot.get_link(name).idx_local for name in self.human_mapped_link_names
        ]
        self.tactile_links_global_indices = [
            self.robot.get_link(name).idx for name in self.tactile_link_names
        ]
        obs_terms = []
        if self.obs_cfg.use_jpos:
            obs_terms.append(("jpos", len(self.dof_map), self._jpos_obs))
        if self.obs_cfg.use_jvel:
            obs_terms.append(("jvel", len(self.dof_map), self._jvel_obs))
        if self.obs_cfg.use_lpos:
            obs_terms.append(
                ("lpos", 3 * len(self.human_mapped_link_names), self._lpos_obs)
            )  # x,y,z for each link
        if self.obs_cfg.use_lvel:
            obs_terms.append(
                ("lvel", 3 * len(self.human_mapped_link_names), self._lvel_obs)
            )  # vx,vy,vz for each link

        self.use_retgt_relative_obs = self.obs_cfg.use_retgt_relative_obs
        self.obs_terms = [term[0] for term in obs_terms]
        self.obs_dims = [term[1] for term in obs_terms]
        self.obs_fns = [term[2] for term in obs_terms]
        self._obs_dim = sum(self.obs_dims)

        # if physics, the robot is controlled as per the actuator models

    def _resolve_base_pose(self, base_pose: np.ndarray | None) -> np.ndarray:
        if base_pose is None:
            return np.eye(4, dtype=np.float32)
        base_pose = np.asarray(base_pose, dtype=np.float32)
        if base_pose.shape != (4, 4):
            raise ValueError(
                f"Expected base_pose to have shape (4, 4), got {base_pose.shape}."
            )
        return base_pose.copy()

    # def root_dof_map(self):
    #     res = [self.robot.get_joint(jname).dofs_idx_local for jname in self.root_joints]
    #     assert all([len(v) == 1 for v in res]), "Each joint should map to exactly one DOF"
    #     return [v[0] for v in res]

    def _dof_map(self):
        res = [self.robot.get_joint(j.name).dofs_idx_local for j in self.joints]
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

    def domain_randomization_summary(self) -> dict[str, list[str]]:
        return {
            "stiffness_available": [
                self.joint_names[idx]
                for idx, available in enumerate(self.default_dof_kp_available)
                if available
            ],
            "stiffness_skipped": [
                self.joint_names[idx]
                for idx, available in enumerate(self.default_dof_kp_available)
                if not available
            ],
            "damping_available": [
                self.joint_names[idx]
                for idx, available in enumerate(self.default_dof_kv_available)
                if available
            ],
            "damping_skipped": [
                self.joint_names[idx]
                for idx, available in enumerate(self.default_dof_kv_available)
                if not available
            ],
            "friction_available": [
                self.joint_names[idx]
                for idx, available in enumerate(self.default_dof_friction_available)
                if available
            ],
            "friction_skipped": [
                self.joint_names[idx]
                for idx, available in enumerate(self.default_dof_friction_available)
                if not available
            ],
        }

    def set_gains(self):
        actuators = self.robo_cfg["actuators"]
        for act, act_info in actuators.items():
            joint_regex_names = act_info["joint_exprs"]
            # search in joint names to find matching joints
            selected_joints = set()
            for j in self.joints:
                for jregex in joint_regex_names:
                    if re.match(jregex, j.name):
                        selected_joints.add(j.name)
            selected_joint_names = list(selected_joints)
            selected_dofs_idx = [
                self.robot.get_joint(name).dofs_idx_local
                for name in selected_joint_names
            ]
            assert all(
                [len(v) == 1 for v in selected_dofs_idx]
            ), "Each actuator should map to exactly one DOF"
            # flatten the dofs list
            selected_dofs_idx = [v[0] for v in selected_dofs_idx]

            kp = act_info["kp"]
            kv = act_info["kv"]
            force_range = act_info["force_range"]
            n_selected_dofs = len(selected_dofs_idx)

            self.robot.set_dofs_kp(
                kp=np.full((self.num_envs, n_selected_dofs), kp, dtype=np.float32),
                dofs_idx_local=selected_dofs_idx,
            )
            self.robot.set_dofs_kv(
                kv=np.full((self.num_envs, n_selected_dofs), kv, dtype=np.float32),
                dofs_idx_local=selected_dofs_idx,
            )
            self.robot.set_dofs_force_range(
                upper=np.full(
                    (self.num_envs, n_selected_dofs), force_range, dtype=np.float32
                ),
                lower=np.full(
                    (self.num_envs, n_selected_dofs), -force_range, dtype=np.float32
                ),
                dofs_idx_local=selected_dofs_idx,
            )

    def randomize_dynamics(self, reset_cfg, env_ids: np.ndarray):
        if len(env_ids) == 0 or len(self.dof_map) == 0:
            return

        n_envs = len(env_ids)
        if np.any(self.default_dof_kp_available):
            stiffness_scale = np.random.uniform(
                reset_cfg.dr_stiffness[0],
                reset_cfg.dr_stiffness[1],
                size=(n_envs, int(self.default_dof_kp_available.sum())),
            ).astype(np.float32)
            stiffness = (
                self.default_dof_kp[self.default_dof_kp_available][None, :]
                * stiffness_scale
            )
            self.env_dof_stiffness[
                np.ix_(np.asarray(env_ids), self.default_dof_kp_available)
            ] = stiffness
            self.robot.set_dofs_stiffness(
                stiffness=self.env_dof_stiffness[:, self.default_dof_kp_available],
                dofs_idx_local=np.asarray(self.dof_map)[self.default_dof_kp_available],
            )
        if np.any(self.default_dof_kv_available):
            damping_scale = np.random.uniform(
                reset_cfg.dr_damping[0],
                reset_cfg.dr_damping[1],
                size=(n_envs, int(self.default_dof_kv_available.sum())),
            ).astype(np.float32)
            damping = (
                self.default_dof_kv[self.default_dof_kv_available][None, :]
                * damping_scale
            )
            self.env_dof_damping[
                np.ix_(np.asarray(env_ids), self.default_dof_kv_available)
            ] = damping
            self.robot.set_dofs_damping(
                damping=self.env_dof_damping[:, self.default_dof_kv_available],
                dofs_idx_local=np.asarray(self.dof_map)[self.default_dof_kv_available],
            )
        if np.any(self.default_dof_friction_available):
            friction_scale = np.random.uniform(
                reset_cfg.dr_friction[0],
                reset_cfg.dr_friction[1],
                size=(n_envs, int(self.default_dof_friction_available.sum())),
            ).astype(np.float32)
            friction = (
                self.default_dof_friction[self.default_dof_friction_available][None, :]
                * friction_scale
            )
            self.env_dof_friction[
                np.ix_(np.asarray(env_ids), self.default_dof_friction_available)
            ] = friction
            self.robot.set_dofs_frictionloss(
                frictionloss=self.env_dof_friction[
                    :, self.default_dof_friction_available
                ],
                dofs_idx_local=np.asarray(self.dof_map)[
                    self.default_dof_friction_available
                ],
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

    def apply_processed_action(
        self, action: dict[str, str | torch.Tensor | np.ndarray]
    ) -> None:
        controller = action["controller"]
        if not isinstance(controller, str):
            raise ValueError(
                f"Expected processed hand action to include a string controller, got {type(controller)}."
            )

        if controller == "position":
            self.robot.control_dofs_position(
                action["position_targets"], action["position_dofs"]
            )
            return

        if controller == "velocity":
            self.robot.control_dofs_velocity(
                action["velocity_targets"], action["velocity_dofs"]
            )
            return

        if controller == "hybrid":
            self.robot.control_dofs_position(
                action["position_targets"], action["position_dofs"]
            )
            self.robot.control_dofs_velocity(
                action["velocity_targets"], action["velocity_dofs"]
            )
            return

        if controller == "impedance":
            stiffness = action.get("stiffness")
            if stiffness is not None:
                self.robot.set_dofs_kp(kp=stiffness, dofs_idx_local=self.dof_map)

            damping = action.get("damping")
            if damping is not None:
                self.robot.set_dofs_kv(kv=damping, dofs_idx_local=self.dof_map)

            self.robot.control_dofs_force(action["force_targets"], action["force_dofs"])
            return

        raise ValueError(f"Invalid processed controller type: {controller}.")

    def _debug_print_joints_out_of_limits(self, jvals):
        # only for the first env for debugging
        jvals = jvals[0]
        dofs_idx = self.dof_map
        lower_limit, upper_limit = self.robot.get_dofs_limit(dofs_idx_local=dofs_idx)
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
            lower_limit, upper_limit = self.robot.get_dofs_limit(
                dofs_idx_local=dofs_idx
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

            self.robot.set_dofs_position(
                jvals,
                dofs_idx,
                envs_idx=envs_idx,
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
        self.robot.set_dofs_velocity(
            jvals,
            dofs_idx,
            envs_idx=envs_idx,
        )

    def random_action(self, env_ids=None) -> torch.Tensor:
        """Sample uniformly within joint limits; returns (n_envs_sel, num_dofs)."""
        n_envs = self.num_envs if env_ids is None else len(env_ids)
        dofs_idx = self.dof_map
        lower_limit, upper_limit = self.robot.get_dofs_limit(dofs_idx_local=dofs_idx)
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
        jpos = self.robot.get_dofs_position(self.dof_map).detach()

        return jpos

    def _jvel_obs(self):
        jvel = self.robot.get_dofs_velocity(self.dof_map).detach()
        return jvel

    def _lpos_obs(self):
        link_pos = self.robot.get_links_pos(
            links_idx_local=self.human_mapped_link_indices_local, ref="link_origin"
        )
        link_pos = link_pos.view(self.num_envs, 3 * len(self.human_mapped_link_names))

        return link_pos

    def _lvel_obs(self):
        # Genesis API naming can vary by version; try known getters.

        link_vel = self.robot.get_links_vel(
            links_idx_local=self.human_mapped_link_indices_local, ref="link_origin"
        )
        link_vel = link_vel.view(self.num_envs, 3 * len(self.human_mapped_link_names))
        return link_vel

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
