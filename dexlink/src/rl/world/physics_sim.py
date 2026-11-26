from __future__ import annotations

import re
from typing import TYPE_CHECKING

import genesis as gs
import numpy as np
import torch

from dexlink.src.retargeting.data.demo_loader import is_oakink_dataset
from dexlink.utils.retargeting.path_utils import get_assets_dir
from dexlink.src.rl.world.art_object import ArtObject, get_object_cfg
from dexlink.src.rl.world.cardboard_object import CardboardObject
from dexlink.src.rl.world.mechanism_object import MechanismObject
from dexlink.src.rl.world.contact_info_sim import get_filtered_contacts
from dexlink.src.rl.world.robot import Hand, get_hand_specific_cfg

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import SimCfg
    from dexlink.src.rl.world.env import Env


class PhysicsSim:
    def __init__(self, sim_cfg: SimCfg, env: Env):
        self.sim_cfg = sim_cfg
        self.env = env
        self.logs_dir = self.sim_cfg.video_save_dir
        self.record = (self.sim_cfg.video_save_freq > 0) and (
            not self.sim_cfg.show_viewer
        )
        self.is_recording = False
        self.scene: gs.Scene | None = None
        self.cam = None
        self.camera_lookat = self.sim_cfg.camera_look_at
        self.scene_entities: dict[str, object] = {}
        self.hand_sensors = None
        self.object_cfg = None
        self.hand_cfgs = None
        self._env_random_force: np.ndarray | None = None

    def initialize(self, camera_lookat: tuple[float, float, float]) -> None:
        self.camera_lookat = camera_lookat
        gs.init(backend=gs.cpu if self.env.device == "cpu" else gs.cuda)
        self.scene = gs.Scene(**self._get_scene_cfg())
        self._initialize_cameras()
        self._populate_scene()
        self._build()
        self.env.task_manager.process_retargeted_motion()

    def _populate_scene(self) -> None:
        assert self.scene is not None
        task_manager = self.env.task_manager

        self.scene_entities.clear()
        self.hand_sensors = None
        self.hand_cfgs = None
        self.object_cfg = None
        self.scene_entities["plane"] = self._add_plane(pos=(0, 0, -0.05))

        if self.env.task_cfg.demo_dataset == "oakink_lift":
            support_height = float(self.env.task_cfg.lift_support_height)
            object_raised_by = float(task_manager.object_raised_by)
            support_surface = gs.surfaces.Rough(
                roughness=0.8, color=(0.58, 0.50, 0.42, 1.0)
            )
            self.scene_entities["platform"] = self._add_box(
                pos=(0.0, 0.0, support_height / 2.0),
                size=(0.8, 1.2, support_height),
                fixed=True,
                collision=True,
                surface=support_surface,
            )
            self.scene_entities["block"] = self._add_box(
                pos=(0.0, 0.0, support_height + object_raised_by / 2.0),
                size=(0.2, 0.2, object_raised_by),
                fixed=True,
                collision=True,
                surface=gs.surfaces.Rough(roughness=0.9, color=(0.8, 0.8, 0.8, 1.0)),
            )

        if not is_oakink_dataset(self.env.task_cfg.demo_dataset):
            cardbox_size = (0.1, 0.2, 0.1)
            cardbox_pos = (0.0, -0.08, 0.90)
            if "notebook" in self.env.task_cfg.target_object:
                print("Adding notebook support block")
                cardbox_size = (0.10, 0.05, 0.05)
                cardbox_pos = (0.0, 0.0, 0.025)

            cardbox_surface = gs.surfaces.Rough(
                roughness=0.6, color=(167 / 255.0, 134 / 255.0, 103 / 255.0, 1.0)
            )
            self.scene_entities["platform"] = self._add_box(
                pos=cardbox_pos,
                size=cardbox_size,
                fixed=True,
                collision=True,
                surface=cardbox_surface,
            )

        self.object_cfg = get_object_cfg(
            reference_demo=task_manager.original_demo,
            urdf_path=task_manager.original_demo.target_obj_urdf_path,
            obj_name=task_manager.original_demo.target_obj_name,
            object_fixed=self.env.task_cfg.object_fixed,
        )
        object_type = (
            MechanismObject
            if self.object_cfg["passive_mechanism"]
            else CardboardObject
            if self.object_cfg["cardboard_model"] is not None
            else ArtObject
        )
        self.scene_entities["object"] = object_type(
            physics=self, obj_cfg=self.object_cfg, num_envs=self.sim_cfg.num_envs
        )

        if self.env.task_manager.object_only:
            return

        self.hand_cfgs = get_hand_specific_cfg(self.env.task_cfg.hand_name)
        robot_base_pose = self._resolve_robot_base_pose(
            getattr(task_manager.original_demo, "robot_base_pose", None)
        )
        if self.env.task_manager.has_hand("left"):
            self.scene_entities["left"] = Hand(
                physics=self,
                robo_cfg=self.hand_cfgs["left"],
                num_envs=self.sim_cfg.num_envs,
                device=self.env.device,
                obs_cfg=self.env.cfg.obs_cfg.hand_cfg,
                env=self.env,
                name="left",
                base_pose=robot_base_pose,
            )
        if self.env.task_manager.has_hand("right"):
            self.scene_entities["right"] = Hand(
                physics=self,
                robo_cfg=self.hand_cfgs["right"],
                num_envs=self.sim_cfg.num_envs,
                device=self.env.device,
                obs_cfg=self.env.cfg.obs_cfg.hand_cfg,
                env=self.env,
                name="right",
                base_pose=robot_base_pose,
            )

    def _resolve_robot_base_pose(
        self, robot_base_pose: np.ndarray | None
    ) -> np.ndarray | None:
        if self.env.task_cfg.hand_name != "franka_xhand":
            return robot_base_pose

        if robot_base_pose is not None:
            resolved = np.asarray(robot_base_pose, dtype=np.float32)
            if resolved.shape != (4, 4):
                raise ValueError(
                    "Expected robot_base_pose to have shape (4, 4), "
                    f"got {resolved.shape}."
                )
            return resolved.copy()

        resolved = np.eye(4, dtype=np.float32)

        resolved[:3, 3] = np.asarray(
            [-0.5, 0.0, self._scene_table_height_for_robot_base()],
            dtype=np.float32,
        )
        return resolved

    def _scene_table_height_for_robot_base(self) -> float:
        task_manager = self.env.task_manager
        table_height = getattr(task_manager, "scene_table_height", None)
        if table_height is None:
            original_demo = task_manager.original_demo
            retargeted_motion = getattr(original_demo, "retargeted_motion", None)
            if isinstance(retargeted_motion, dict):
                table_height = retargeted_motion.get("scene_table_height", None)
        if table_height is None:
            table_height = getattr(self.env.task_cfg, "lift_support_height", None)
        if table_height is None:
            raise ValueError(
                "franka_xhand base placement requires scene_table_height or "
                "task_cfg.lift_support_height."
            )
        return float(table_height)

    def _build(self) -> None:
        assert self.scene is not None
        print(f"BUILDING SCENE WITH {self.sim_cfg.num_envs} ENVS...")
        self.scene.build(
            n_envs=self.sim_cfg.num_envs,
            env_spacing=(3.0, 3.0),
            n_envs_per_row=int(np.sqrt(self.sim_cfg.num_envs)) + 1,
        )

        self._configure_object_gains()
        if not self.env.task_manager.object_only:
            for side in self.env.task_manager.active_hand_sides:
                self._configure_hand_gains(side)

        self._configure_collision_resolution()

        if self.sim_cfg.apply_random_force:
            self.resample_random_force(env_ids=None)

    def _configure_collision_resolution(self) -> None:
        if not self.sim_cfg.rapid_collision_resolution:
            return
        assert self.scene is not None

        sol_params = torch.as_tensor(
            [
                self._rapid_collision_timeconst(),
                3.0,
                0.95,
                0.999,
                1e-3,
                0.5,
                2.0,
            ],
            dtype=torch.float32,
            device=gs.device,
        )
        n_geoms = int(self.scene.rigid_solver.n_geoms)
        sol_params = sol_params.unsqueeze(0).expand(n_geoms, -1).contiguous()
        geoms_idx = torch.arange(n_geoms, dtype=torch.long, device=gs.device)
        self.scene.rigid_solver.set_sol_params(sol_params, geoms_idx=geoms_idx)

    def _rapid_collision_timeconst(self) -> float:
        return max(2e-3, 2.0 * self.sim_cfg.dt / 4)

    def _configure_object_gains(self) -> None:
        object_entity: ArtObject = self.env.physics_manager.scene_entities["object"]
        if not object_entity.actuated:
            return
        if isinstance(object_entity, (CardboardObject, MechanismObject)):
            object_entity.configure_joint_dynamics()
            return

        dofs_idx = object_entity.dof_map
        kp = object_entity.obj_cfg["kp"]
        kv = object_entity.obj_cfg["kv"]
        force_range = object_entity.obj_cfg["force_range"]
        n_dofs = len(dofs_idx)

        self._set_entity_dofs_kp(
            object_entity.object,
            kp=np.full((self.sim_cfg.num_envs, n_dofs), kp, dtype=np.float32),
            dofs_idx_local=dofs_idx,
        )
        self._set_entity_dofs_kv(
            object_entity.object,
            kv=np.full((self.sim_cfg.num_envs, n_dofs), kv, dtype=np.float32),
            dofs_idx_local=dofs_idx,
        )
        self._set_entity_dofs_force_range(
            object_entity.object,
            lower=np.full(
                (self.sim_cfg.num_envs, n_dofs), -force_range, dtype=np.float32
            ),
            upper=np.full(
                (self.sim_cfg.num_envs, n_dofs), force_range, dtype=np.float32
            ),
            dofs_idx_local=dofs_idx,
        )

    def _configure_hand_gains(self, side: str) -> None:
        hand: Hand = self.env.physics_manager.scene_entities[side]
        actuators = hand.robo_cfg["actuators"]
        for act_info in actuators.values():
            joint_regex_names = act_info["joint_exprs"]
            selected_joints = set()
            for joint in hand.joints:
                for jregex in joint_regex_names:
                    if re.match(jregex, joint.name):
                        selected_joints.add(joint.name)
            selected_joint_names = list(selected_joints)
            selected_dofs_idx = [
                self.get_joint_dofs_idx_local(hand.robot, name)
                for name in selected_joint_names
            ]
            assert all(
                len(v) == 1 for v in selected_dofs_idx
            ), "Each actuator should map to exactly one DOF"
            selected_dofs_idx = [v[0] for v in selected_dofs_idx]

            kp = act_info["kp"]
            kv = act_info["kv"]
            force_range = act_info["force_range"]
            n_selected_dofs = len(selected_dofs_idx)

            self._set_entity_dofs_kp(
                hand.robot,
                kp=np.full(
                    (self.sim_cfg.num_envs, n_selected_dofs), kp, dtype=np.float32
                ),
                dofs_idx_local=selected_dofs_idx,
            )
            self._set_entity_dofs_kv(
                hand.robot,
                kv=np.full(
                    (self.sim_cfg.num_envs, n_selected_dofs), kv, dtype=np.float32
                ),
                dofs_idx_local=selected_dofs_idx,
            )
            self._set_entity_dofs_force_range(
                hand.robot,
                upper=np.full(
                    (self.sim_cfg.num_envs, n_selected_dofs),
                    force_range,
                    dtype=np.float32,
                ),
                lower=np.full(
                    (self.sim_cfg.num_envs, n_selected_dofs),
                    -force_range,
                    dtype=np.float32,
                ),
                dofs_idx_local=selected_dofs_idx,
            )

    def object_domain_randomization_summary(self) -> dict[str, list[str]]:
        object_entity: ArtObject = self.env.physics_manager.scene_entities["object"]
        return {
            "mass_available": [
                object_entity.link_names[idx]
                for idx, available in enumerate(
                    object_entity.default_link_mass_available
                )
                if available
            ],
            "mass_skipped": [
                object_entity.link_names[idx]
                for idx, available in enumerate(
                    object_entity.default_link_mass_available
                )
                if not available
            ],
        }

    def hand_domain_randomization_summary(self, side: str) -> dict[str, list[str]]:
        hand: Hand = self.env.physics_manager.scene_entities[side]
        return {
            "stiffness_available": [
                hand.joint_names[idx]
                for idx, available in enumerate(hand.default_dof_kp_available)
                if available
            ],
            "stiffness_skipped": [
                hand.joint_names[idx]
                for idx, available in enumerate(hand.default_dof_kp_available)
                if not available
            ],
            "damping_available": [
                hand.joint_names[idx]
                for idx, available in enumerate(hand.default_dof_kv_available)
                if available
            ],
            "damping_skipped": [
                hand.joint_names[idx]
                for idx, available in enumerate(hand.default_dof_kv_available)
                if not available
            ],
            "friction_available": [
                hand.joint_names[idx]
                for idx, available in enumerate(hand.default_dof_friction_available)
                if available
            ],
            "friction_skipped": [
                hand.joint_names[idx]
                for idx, available in enumerate(hand.default_dof_friction_available)
                if not available
            ],
        }

    def apply_domain_randomization(self, reset_cfg, env_ids: np.ndarray) -> None:
        self._randomize_object_dynamics(reset_cfg, env_ids)
        self._randomize_object_surface_friction(reset_cfg, env_ids)
        # Other domain randomization modes are intentionally disabled for the
        # final experiments. In particular, do not randomize hand stiffness,
        # hand damping, or joint friction.
        # for side in self.env.task_manager.active_hand_sides:
        #     self._randomize_hand_dynamics(side, reset_cfg, env_ids)

    def _validate_domain_randomization_range(
        self, name: str, value: tuple[float, float]
    ) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError(
                f"Expected reset_cfg.{name} to contain two values, got {value}."
            )
        lower, upper = float(value[0]), float(value[1])
        if lower <= 0.0 or upper <= 0.0 or lower > upper:
            raise ValueError(
                f"Expected reset_cfg.{name} to be positive and ordered, got {value}."
            )
        return lower, upper

    def _randomize_object_dynamics(self, reset_cfg, env_ids: np.ndarray) -> None:
        object_entity: ArtObject = self.env.physics_manager.scene_entities["object"]
        if len(env_ids) == 0 or len(object_entity.link_idx_local) == 0:
            return
        if not np.any(object_entity.default_link_mass_available):
            return

        default_masses = object_entity.default_link_mass[
            object_entity.default_link_mass_available
        ]
        default_total_mass = float(default_masses.sum())
        if default_total_mass <= 0.0:
            raise ValueError(
                "Cannot randomize object mass because available object link "
                f"masses sum to {default_total_mass}."
            )
        min_total_mass = float(reset_cfg.dr_object_mass_min_abs)
        if min_total_mass <= 0.0:
            raise ValueError(
                "Expected reset_cfg.dr_object_mass_min_abs > 0, got "
                f"{min_total_mass}."
            )
        mass_min, mass_max = self._validate_domain_randomization_range(
            "dr_object_mass", reset_cfg.dr_object_mass
        )
        sampled_total_mass = np.random.uniform(
            max(mass_min, min_total_mass),
            mass_max,
            size=(len(env_ids), 1),
        ).astype(np.float32)
        sampled_total_mass = np.maximum(sampled_total_mass, min_total_mass)
        masses = default_masses[None, :] * (sampled_total_mass / default_total_mass)
        object_entity.env_link_mass[
            np.ix_(np.asarray(env_ids), object_entity.default_link_mass_available)
        ] = masses

        links_idx_local = np.asarray(object_entity.link_idx_local)[
            object_entity.default_link_mass_available
        ]
        self._set_entity_links_inertial_mass(
            object_entity.object,
            inertial_mass=object_entity.env_link_mass[
                :, object_entity.default_link_mass_available
            ],
            links_idx_local=links_idx_local,
        )

    def _randomize_object_surface_friction(
        self, reset_cfg, env_ids: np.ndarray
    ) -> None:
        object_entity: ArtObject = self.env.physics_manager.scene_entities["object"]
        self._randomize_single_object_surface_friction(
            object_entity, reset_cfg, env_ids
        )

    def _randomize_single_object_surface_friction(
        self, object_entity: ArtObject, reset_cfg, env_ids: np.ndarray
    ) -> None:
        if len(env_ids) == 0 or len(object_entity.link_idx_local) == 0:
            return
        friction_min, friction_max = self._validate_domain_randomization_range(
            "dr_contact_friction", reset_cfg.dr_contact_friction
        )
        base_friction = float(object_entity.obj_cfg["friction"])
        if base_friction <= 0.0:
            raise ValueError(
                "Cannot randomize absolute object surface friction because the "
                f"object base friction is {base_friction}."
            )
        env_ids = np.asarray(env_ids)
        absolute_friction = np.random.uniform(
            friction_min,
            friction_max,
            size=(len(env_ids), len(object_entity.link_idx_local)),
        ).astype(np.float32)
        friction_ratio = absolute_friction / np.float32(base_friction)
        object_entity.env_link_friction_ratio[env_ids, :] = friction_ratio
        self._set_entity_friction_ratio(
            object_entity.object,
            friction_ratio=object_entity.env_link_friction_ratio,
            links_idx_local=object_entity.link_idx_local,
        )

    def _randomize_hand_dynamics(
        self, side: str, reset_cfg, env_ids: np.ndarray
    ) -> None:
        hand: Hand = self.env.physics_manager.scene_entities[side]
        if len(env_ids) == 0 or len(hand.dof_map) == 0:
            return

        n_envs = len(env_ids)
        if np.any(hand.default_dof_kp_available):
            stiffness_scale = np.random.uniform(
                reset_cfg.dr_stiffness[0],
                reset_cfg.dr_stiffness[1],
                size=(n_envs, int(hand.default_dof_kp_available.sum())),
            ).astype(np.float32)
            stiffness = (
                hand.default_dof_kp[hand.default_dof_kp_available][None, :]
                * stiffness_scale
            )
            hand.env_dof_stiffness[
                np.ix_(np.asarray(env_ids), hand.default_dof_kp_available)
            ] = stiffness
            self._set_entity_dofs_stiffness(
                hand.robot,
                stiffness=hand.env_dof_stiffness[:, hand.default_dof_kp_available],
                dofs_idx_local=np.asarray(hand.dof_map)[hand.default_dof_kp_available],
            )

        if np.any(hand.default_dof_kv_available):
            damping_scale = np.random.uniform(
                reset_cfg.dr_damping[0],
                reset_cfg.dr_damping[1],
                size=(n_envs, int(hand.default_dof_kv_available.sum())),
            ).astype(np.float32)
            damping = (
                hand.default_dof_kv[hand.default_dof_kv_available][None, :]
                * damping_scale
            )
            hand.env_dof_damping[
                np.ix_(np.asarray(env_ids), hand.default_dof_kv_available)
            ] = damping
            self._set_entity_dofs_damping(
                hand.robot,
                damping=hand.env_dof_damping[:, hand.default_dof_kv_available],
                dofs_idx_local=np.asarray(hand.dof_map)[hand.default_dof_kv_available],
            )

        if np.any(hand.default_dof_friction_available):
            friction_scale = np.random.uniform(
                reset_cfg.dr_friction[0],
                reset_cfg.dr_friction[1],
                size=(n_envs, int(hand.default_dof_friction_available.sum())),
            ).astype(np.float32)
            friction = (
                hand.default_dof_friction[hand.default_dof_friction_available][None, :]
                * friction_scale
            )
            hand.env_dof_friction[
                np.ix_(np.asarray(env_ids), hand.default_dof_friction_available)
            ] = friction
            self._set_entity_dofs_frictionloss(
                hand.robot,
                frictionloss=hand.env_dof_friction[
                    :, hand.default_dof_friction_available
                ],
                dofs_idx_local=np.asarray(hand.dof_map)[
                    hand.default_dof_friction_available
                ],
            )

    def _add_plane(self, pos: tuple[float, float, float]):
        assert self.scene is not None
        plane_path = (
            get_assets_dir()
            / "genesis_safe_checkered_plane"
            / "white_checkered_plane_safe.obj"
            # / "white_checkered_plane_obj"
            # / "white_checkered_plane.obj"
        )
        return self.scene.add_entity(
            gs.morphs.Mesh(
                file=str(plane_path),
                pos=pos,
                fixed=True,
                collision=False,
                decimate=False,
                convexify=False,
            )
        )

    def _add_box(
        self,
        pos: tuple[float, float, float],
        size: tuple[float, float, float],
        fixed: bool,
        collision: bool,
        surface,
    ):
        assert self.scene is not None
        return self.scene.add_entity(
            gs.morphs.Box(
                pos=pos,
                size=size,
                fixed=fixed,
                collision=collision,
            ),
            surface=surface,
        )

    def add_rigid_urdf_entity(
        self,
        urdf_path: str,
        *,
        fixed: bool,
        collision: bool,
        convexify: bool,
        recompute_inertia: bool = False,
        pos: tuple[float, float, float] = (0, 0, 0),
        quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),  # wxyz
        name: str | None = None,
        links_to_keep: list[str] | None = None,
        rho: float | None = None,
        friction: float = 1.2,
        gravity_compensation: float | None = None,
    ):
        assert self.scene is not None
        material_kwargs = {"friction": friction}
        if rho is not None:
            material_kwargs["rho"] = rho
        if gravity_compensation is not None:
            material_kwargs["gravity_compensation"] = gravity_compensation
        return self.scene.add_entity(
            material=gs.materials.Rigid(**material_kwargs),
            morph=gs.morphs.URDF(
                file=urdf_path,
                fixed=fixed,
                recompute_inertia=recompute_inertia,
                collision=collision,
                convexify=convexify,
                pos=pos,
                quat=quat,
                links_to_keep=[] if links_to_keep is None else links_to_keep,
            ),
        )

    def get_joint_dofs_idx_local(self, entity, joint_name: str) -> list[int]:
        return entity.get_joint(joint_name).dofs_idx_local

    def get_link_idx_local(self, entity, link_name: str) -> int:
        return entity.get_link(link_name).idx_local

    def get_link_idx(self, entity, link_name: str) -> int:
        return entity.get_link(link_name).idx

    def _get_base_link_idx(self, entity) -> int:
        return entity.base_link_idx

    def _set_entity_dofs_kp(
        self, entity, kp: np.ndarray, dofs_idx_local: list[int]
    ) -> None:
        entity.set_dofs_kp(kp=kp, dofs_idx_local=dofs_idx_local)

    def _set_entity_dofs_kv(
        self, entity, kv: np.ndarray, dofs_idx_local: list[int]
    ) -> None:
        entity.set_dofs_kv(kv=kv, dofs_idx_local=dofs_idx_local)

    def _set_entity_dofs_force_range(
        self,
        entity,
        lower: np.ndarray,
        upper: np.ndarray,
        dofs_idx_local: list[int],
    ) -> None:
        entity.set_dofs_force_range(
            lower=lower,
            upper=upper,
            dofs_idx_local=dofs_idx_local,
        )

    def _set_entity_dofs_stiffness(
        self, entity, stiffness: np.ndarray, dofs_idx_local
    ) -> None:
        entity.set_dofs_stiffness(stiffness=stiffness, dofs_idx_local=dofs_idx_local)

    def _set_entity_dofs_damping(
        self, entity, damping: np.ndarray, dofs_idx_local
    ) -> None:
        entity.set_dofs_damping(damping=damping, dofs_idx_local=dofs_idx_local)

    def _set_entity_dofs_frictionloss(
        self, entity, frictionloss: np.ndarray, dofs_idx_local
    ) -> None:
        entity.set_dofs_frictionloss(
            frictionloss=frictionloss,
            dofs_idx_local=dofs_idx_local,
        )

    def _set_entity_links_inertial_mass(
        self, entity, inertial_mass: np.ndarray, links_idx_local
    ) -> None:
        entity.set_links_inertial_mass(
            inertial_mass=inertial_mass,
            links_idx_local=links_idx_local,
        )

    def _set_entity_friction_ratio(
        self, entity, friction_ratio: np.ndarray, links_idx_local
    ) -> None:
        entity.set_friction_ratio(
            friction_ratio=friction_ratio,
            links_idx_local=links_idx_local,
        )

    def set_entity_pos(self, entity, pos, envs_idx=None) -> None:
        entity.set_pos(pos=pos, envs_idx=envs_idx)

    def set_entity_quat(self, entity, quat, envs_idx=None) -> None:
        entity.set_quat(quat=quat, envs_idx=envs_idx)

    def set_entity_dofs_position(
        self, entity, values, dofs_idx_local, envs_idx=None
    ) -> None:
        entity.set_dofs_position(values, dofs_idx_local, envs_idx=envs_idx)

    def set_entity_dofs_velocity(
        self, entity, values, dofs_idx_local, envs_idx=None
    ) -> None:
        entity.set_dofs_velocity(values, dofs_idx_local, envs_idx=envs_idx)

    def get_entity_pos(self, entity):
        return entity.get_pos()

    def get_entity_quat(self, entity):
        return entity.get_quat()

    def get_entity_vel(self, entity):
        return entity.get_vel()

    def get_entity_ang(self, entity):
        return entity.get_ang()

    def get_entity_dofs_position(self, entity, dofs_idx_local):
        return entity.get_dofs_position(dofs_idx_local)

    def get_entity_dofs_velocity(self, entity, dofs_idx_local):
        return entity.get_dofs_velocity(dofs_idx_local)

    def get_entity_dofs_acceleration(self, entity, dofs_idx_local):
        return entity.get_dofs_acceleration(dofs_idx_local)

    def get_entity_dofs_force(self, entity, dofs_idx_local):
        return entity.get_dofs_force(dofs_idx_local)

    def get_entity_dofs_control_force(self, entity, dofs_idx_local):
        return entity.get_dofs_control_force(dofs_idx_local)

    def get_entity_dofs_limit(self, entity, dofs_idx_local):
        return entity.get_dofs_limit(dofs_idx_local=dofs_idx_local)

    def _get_entity_dofs_kp(self, entity, dofs_idx_local):
        return entity.get_dofs_kp(dofs_idx_local=dofs_idx_local)

    def _get_entity_dofs_kv(self, entity, dofs_idx_local):
        return entity.get_dofs_kv(dofs_idx_local=dofs_idx_local)

    def _get_entity_dofs_force_range(self, entity, dofs_idx_local):
        return entity.get_dofs_force_range(dofs_idx_local=dofs_idx_local)

    def control_entity_dofs_position(
        self,
        entity,
        values,
        dofs_idx_local,
    ) -> None:
        entity.control_dofs_position(values, dofs_idx_local)

    def _control_entity_dofs_velocity(self, entity, values, dofs_idx_local) -> None:
        entity.control_dofs_velocity(values, dofs_idx_local)

    def _control_entity_dofs_force(self, entity, values, dofs_idx_local) -> None:
        entity.control_dofs_force(values, dofs_idx_local)

    def get_entity_links_pos(self, entity, links_idx_local, ref: str):
        return entity.get_links_pos(links_idx_local=links_idx_local, ref=ref)

    def _get_entity_links_quat(self, entity, links_idx_local):
        return entity.get_links_quat(links_idx_local=links_idx_local)

    def get_entity_links_vel(self, entity, links_idx_local, ref: str):
        return entity.get_links_vel(links_idx_local=links_idx_local, ref=ref)

    def get_entity_link_pos(self, entity, *, link_name: str, ref: str):
        link_idx_local = self.get_link_idx_local(entity, link_name)
        return self.get_entity_links_pos(
            entity, links_idx_local=[link_idx_local], ref=ref
        )

    def get_entity_link_quat(self, entity, *, link_name: str):
        link_idx_local = self.get_link_idx_local(entity, link_name)
        return self._get_entity_links_quat(entity, links_idx_local=[link_idx_local])

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
        expected_pos_shape = (self.sim_cfg.num_envs, 3)
        if tuple(target_pos.shape) != expected_pos_shape:
            raise ValueError(
                f"IK target_pos shape mismatch: expected {expected_pos_shape}, got {tuple(target_pos.shape)}."
            )
        expected_quat_shape = (self.sim_cfg.num_envs, 4)
        if tuple(target_quat.shape) != expected_quat_shape:
            raise ValueError(
                f"IK target_quat shape mismatch: expected {expected_quat_shape}, got {tuple(target_quat.shape)}."
            )
        if init_qpos.ndim != 2:
            raise ValueError(
                f"IK init_qpos must be rank-2 with shape (num_envs, num_q), got {tuple(init_qpos.shape)}."
            )

        init_qpos_dofs_idx_local = np.asarray(init_qpos_dofs_idx_local, dtype=np.int64)
        if init_qpos_dofs_idx_local.ndim != 1:
            raise ValueError(
                "IK init_qpos_dofs_idx_local must be rank-1 with one entity-local DOF index "
                f"per init_qpos column, got shape {tuple(init_qpos_dofs_idx_local.shape)}."
            )
        if init_qpos.shape[1] != len(init_qpos_dofs_idx_local):
            raise ValueError(
                "IK init_qpos shape mismatch: expected compact joint-order qpos with "
                f"{len(init_qpos_dofs_idx_local)} columns, got {init_qpos.shape[1]}."
            )
        if np.any(init_qpos_dofs_idx_local < 0) or np.any(
            init_qpos_dofs_idx_local >= entity.n_qs
        ):
            raise ValueError(
                "IK init_qpos_dofs_idx_local must contain entity-local DOF indices; "
                f"received indices {init_qpos_dofs_idx_local.tolist()} for entity.n_qs={entity.n_qs}."
            )
        if len(np.unique(init_qpos_dofs_idx_local)) != len(init_qpos_dofs_idx_local):
            raise ValueError(
                "IK init_qpos_dofs_idx_local must not contain duplicate DOF indices."
            )

        dofs_idx_local = np.asarray(dofs_idx_local, dtype=np.int64)
        if dofs_idx_local.ndim != 1:
            raise ValueError(
                f"IK dofs_idx_local must be rank-1, got shape {tuple(dofs_idx_local.shape)}."
            )
        if np.any(dofs_idx_local < 0) or np.any(dofs_idx_local >= entity.n_qs):
            raise ValueError(
                "IK dofs_idx_local must contain entity-local DOF indices; "
                f"received indices {dofs_idx_local.tolist()} for entity.n_qs={entity.n_qs}."
            )

        init_qpos_entity_order = torch.empty(
            (self.sim_cfg.num_envs, entity.n_qs),
            dtype=init_qpos.dtype,
            device=init_qpos.device,
        )
        init_qpos_entity_order[:, init_qpos_dofs_idx_local.tolist()] = init_qpos
        link = entity.get_link(link_name)
        ik_qpos = entity.inverse_kinematics(
            link=link,
            pos=target_pos,
            quat=target_quat,
            dofs_idx_local=dofs_idx_local.tolist(),
            max_samples=10,
            max_solver_iters=10,
            init_qpos=init_qpos_entity_order,
        )
        if tuple(ik_qpos.shape) != (self.sim_cfg.num_envs, entity.n_qs):
            raise ValueError(
                "IK solver returned unexpected qpos shape: "
                f"expected {(self.sim_cfg.num_envs, entity.n_qs)}, got {tuple(ik_qpos.shape)}."
            )
        return ik_qpos[:, init_qpos_dofs_idx_local.tolist()]

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
        return get_filtered_contacts(
            entity_a=entity_a,
            entity_b=entity_b,
            filter_geoms_a=[] if filter_geoms_a is None else filter_geoms_a,
            filter_geoms_b=[] if filter_geoms_b is None else filter_geoms_b,
            filter_links_a=[] if filter_links_a is None else filter_links_a,
            filter_links_b=[] if filter_links_b is None else filter_links_b,
            return_geom_force=return_geom_force,
            return_link_force=return_link_force,
            return_geom_pos=return_geom_pos,
            return_link_pos=return_link_pos,
            device=self.env.device if device is None else device,
        )

    def _get_hand_object_contact_info(
        self,
        hand: Hand,
        object_entity: ArtObject | None,
        *,
        return_link_force: bool,
        return_link_pos: bool,
    ):
        target_entity = None if object_entity is None else object_entity.object
        return self.get_filtered_contact_info(
            entity_a=hand.robot,
            entity_b=target_entity,
            filter_links_a=torch.tensor(
                hand.tactile_links_global_indices, dtype=torch.long
            ),
            return_link_force=return_link_force,
            return_link_pos=return_link_pos,
        )

    def get_hand_object_binary_contact_signal(
        self,
        hand: Hand,
        object_entity: ArtObject | None,
        *,
        threshold: float,
    ) -> torch.Tensor:
        contact_info = self._get_hand_object_contact_info(
            hand,
            object_entity,
            return_link_force=True,
            return_link_pos=False,
        )
        force = contact_info["contact_force_link_a"]
        force_net = force.sum(dim=2)
        return (force_net.abs() > threshold).to(force_net.dtype)

    def get_hand_object_contact_support_diagnostics(
        self,
        hand: Hand,
        object_entity: ArtObject,
        *,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-hand-link contact flags and net force acting on the object."""
        contact_info = self.get_filtered_contact_info(
            entity_a=object_entity.object,
            entity_b=hand.robot,
            filter_links_b=torch.tensor(
                hand.tactile_links_global_indices,
                dtype=torch.long,
            ),
            return_link_force=True,
        )
        force_on_object = contact_info["contact_force_link_a"]
        if force_on_object.ndim != 4 or force_on_object.shape[-1] != 3:
            raise ValueError(
                "Expected object-hand contact force shape (N, L_object, L_hand, 3), "
                f"got {tuple(force_on_object.shape)}."
            )
        force_on_object_by_hand_link = force_on_object.sum(dim=1)
        contact_signal = (force_on_object_by_hand_link.abs() > threshold).to(
            force_on_object.dtype
        )
        net_force_on_object = force_on_object.sum(dim=(1, 2))
        return contact_signal, net_force_on_object

    def get_hand_joint_positions(self, hand: Hand) -> torch.Tensor:
        return self.get_entity_dofs_position(hand.robot, hand.dof_map)

    def get_hand_dof_positions(
        self, hand: Hand, dof_indices: list[int] | np.ndarray
    ) -> torch.Tensor:
        return self.get_entity_dofs_position(hand.robot, dof_indices)

    def _get_hand_dof_velocities(
        self, hand: Hand, dof_indices: list[int] | np.ndarray
    ) -> torch.Tensor:
        return self.get_entity_dofs_velocity(hand.robot, dof_indices)

    def get_hand_dof_accelerations(
        self, hand: Hand, dof_indices: list[int] | np.ndarray
    ) -> torch.Tensor:
        return self.get_entity_dofs_acceleration(hand.robot, dof_indices)

    def get_hand_dof_limits(
        self, hand: Hand, dof_indices: list[int] | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hand.get_dof_limits(
            dof_indices,
            effective=True,
            device=self.env.device,
            dtype=torch.float32,
        )

    def _get_hand_dof_kp(
        self, hand: Hand, dof_indices: list[int] | np.ndarray
    ) -> torch.Tensor:
        return self._get_entity_dofs_kp(hand.robot, dof_indices)

    def _get_hand_dof_kv(
        self, hand: Hand, dof_indices: list[int] | np.ndarray
    ) -> torch.Tensor:
        return self._get_entity_dofs_kv(hand.robot, dof_indices)

    def _get_hand_dof_force_range(
        self, hand: Hand, dof_indices: list[int] | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._get_entity_dofs_force_range(hand.robot, dof_indices)

    def get_object_pcd(self, normals: bool) -> torch.Tensor:
        obs_mgr = self.env.observation_manager
        privileged_object_pcd = obs_mgr._cached_privileged_object_pcd
        if privileged_object_pcd is None:
            privileged_object_pcd = (
                obs_mgr._compute_privileged_object_pcd_from_object_state()
            )
            obs_mgr._cached_privileged_object_pcd = privileged_object_pcd
        if normals:
            return privileged_object_pcd
        return self._prepare_policy_object_pcd(privileged_object_pcd)

    def _prepare_policy_object_pcd(
        self, privileged_object_pcd: torch.Tensor
    ) -> torch.Tensor:
        obs_cfg = self.env.cfg.obs_cfg
        target_n_pts = obs_cfg.policy_object_pcd_n_pts
        if privileged_object_pcd.ndim != 3:
            raise ValueError(
                f"Expected privileged_object_pcd to have shape (B, N, C), got {privileged_object_pcd.shape}."
            )
        if target_n_pts <= 0:
            raise ValueError(
                f"Expected obs_cfg.policy_object_pcd_n_pts > 0, got {target_n_pts}."
            )

        batch_size, n_pts, feat_dim = privileged_object_pcd.shape
        if feat_dim != 6:
            raise ValueError(
                "Expected privileged object PCD features to contain xyz+normals "
                f"(dim 6), got {feat_dim}."
            )
        if target_n_pts > n_pts:
            raise ValueError(
                f"Cannot subsample {target_n_pts} points from privileged object PCD with only {n_pts} points."
            )

        policy_object_pcd = privileged_object_pcd[..., :3]
        drop_ratio_min, drop_ratio_max = (
            obs_cfg.policy_object_pcd_truncation_drop_ratio_range
        )
        if not (0.0 <= drop_ratio_min <= drop_ratio_max <= 1.0):
            raise ValueError(
                "Expected obs_cfg.policy_object_pcd_truncation_drop_ratio_range to satisfy "
                f"0 <= min <= max <= 1, got {(drop_ratio_min, drop_ratio_max)}."
            )
        jitter_std = obs_cfg.policy_object_pcd_jitter_std
        if jitter_std < 0.0:
            raise ValueError(
                f"Expected obs_cfg.policy_object_pcd_jitter_std >= 0, got {jitter_std}."
            )

        centered_pcd = policy_object_pcd - policy_object_pcd.mean(dim=1, keepdim=True)
        plane_normals = torch.randn(
            batch_size,
            3,
            device=privileged_object_pcd.device,
            dtype=policy_object_pcd.dtype,
        )
        plane_normals[:, 2] = plane_normals[:, 2].abs().clamp_min(1e-6)
        plane_normals = plane_normals / plane_normals.norm(dim=-1, keepdim=True)
        projections = torch.sum(centered_pcd * plane_normals[:, None, :], dim=-1)

        negative_counts = (projections < 0.0).sum(dim=-1)
        max_drop_from_plane = (negative_counts - 1).clamp_min(0)
        max_drop_from_target = torch.full_like(
            max_drop_from_plane, n_pts - target_n_pts
        )
        max_drop_from_cfg = torch.full_like(
            max_drop_from_plane,
            int(drop_ratio_max * n_pts),
        )
        min_drop_from_cfg = torch.full_like(
            max_drop_from_plane,
            int(drop_ratio_min * n_pts),
        )
        max_drop = torch.minimum(
            max_drop_from_plane,
            torch.minimum(max_drop_from_target, max_drop_from_cfg),
        )
        min_drop = torch.minimum(min_drop_from_cfg, max_drop)
        drop_span = max_drop - min_drop + 1
        random_drop = min_drop + torch.floor(
            torch.rand(batch_size, device=privileged_object_pcd.device) * drop_span
        ).to(torch.long)

        sorted_indices = projections.argsort(dim=-1)
        sorted_rank = torch.empty_like(sorted_indices)
        rank_values = torch.arange(
            n_pts, device=privileged_object_pcd.device, dtype=sorted_indices.dtype
        ).expand(batch_size, -1)
        sorted_rank.scatter_(1, sorted_indices, rank_values)
        keep_mask = sorted_rank >= random_drop[:, None]

        random_scores = torch.rand(
            batch_size, n_pts, device=privileged_object_pcd.device
        )
        # Uniformly subsample from the truncated set without replacement.
        random_scores.masked_fill_(~keep_mask, torch.inf)
        sample_indices = torch.topk(
            random_scores, k=target_n_pts, dim=-1, largest=False
        ).indices
        gather_idx = sample_indices.unsqueeze(-1).expand(-1, -1, 3)
        subsampled_pcd = torch.gather(policy_object_pcd, dim=1, index=gather_idx)
        if jitter_std == 0.0:
            return subsampled_pcd
        return subsampled_pcd + torch.randn_like(subsampled_pcd) * jitter_std

    def start_recording(self) -> None:
        if self.cam is None:
            raise ValueError("Camera has not been initialized.")
        self.cam.start_recording()
        self.is_recording = True

    def stop_recording(self, save_to_filename: str, fps: int = 30) -> None:
        if self.cam is None:
            raise ValueError("Camera has not been initialized.")
        self.cam.stop_recording(save_to_filename=save_to_filename, fps=fps)
        self.is_recording = False

    def _get_scene_cfg(self) -> dict:
        cfg = self.sim_cfg
        substeps = 4
        rigid_options = dict(
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_self_collision=cfg.enable_self_collision,
            enable_joint_limit=cfg.enable_joint_limit,
            max_collision_pairs=cfg.max_collision_pairs,
            batch_links_info=cfg.batch_links_info,
            batch_dofs_info=cfg.batch_dofs_info,
        )
        if cfg.rapid_collision_resolution:
            rigid_options.update(
                iterations=80,
                tolerance=1e-8,
                ls_iterations=50,
                constraint_timeconst=max(2e-3, 2.0 * cfg.dt / substeps),
                use_gjk_collision=True,
                enable_multi_contact=True,
            )
        return dict(
            sim_options=gs.options.SimOptions(
                dt=cfg.dt,
                substeps=substeps,
                gravity=(0, 0, cfg.gravity),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_lookat=self.camera_lookat,
                camera_fov=cfg.camera_fov,
                max_FPS=cfg.max_FPS,
            ),
            show_viewer=cfg.show_viewer,
            vis_options=gs.options.VisOptions(show_world_frame=False),
            rigid_options=gs.options.RigidOptions(**rigid_options),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        )

    def _initialize_cameras(self) -> None:
        assert self.scene is not None
        cfg = self.sim_cfg
        self.cam_theta = cfg.camera_theta
        self.cam = self.scene.add_camera(
            res=cfg.resolution,
            pos=(
                cfg.camera_radius * np.sin(self.cam_theta),
                cfg.camera_radius * np.cos(self.cam_theta),
                cfg.camera_ht,
            ),
            lookat=self.camera_lookat,
            fov=cfg.camera_fov,
            GUI=False,
        )

    def resample_random_force(
        self, env_ids: np.ndarray | list[int] | None = None
    ) -> None:
        if not self.sim_cfg.apply_random_force:
            return

        if self._env_random_force is None:
            self._env_random_force = np.zeros(
                (self.env.num_envs, 1, 3), dtype=np.float32
            )

        if env_ids is None:
            env_ids_np = np.arange(self.env.num_envs, dtype=np.int64)
        else:
            env_ids_np = np.asarray(env_ids, dtype=np.int64).reshape(-1)
            if env_ids_np.size == 0:
                return

        f_mag = self.sim_cfg.random_force_mag

        random_force = np.random.normal(size=(len(env_ids_np), 1, 3)).astype(np.float32)
        random_force_norm = np.linalg.norm(random_force, axis=2, keepdims=True)
        if np.any(random_force_norm == 0.0):
            raise ValueError("Sampled zero-norm random force direction.")
        random_force /= random_force_norm
        self._env_random_force[env_ids_np] = random_force * np.float32(f_mag)

    def video_recording_hook(self, timestep: int):
        save_dir = self.logs_dir
        save_freq = self.sim_cfg.video_save_freq
        save_length = self.sim_cfg.video_save_length

        if (save_freq <= 0) or (self.cam is None) or (not self.record):
            return
        if (timestep % save_freq == 0) and (not self.is_recording):
            self.cam.start_recording()
            self.is_recording = True
        if self.is_recording and (timestep % save_freq == save_length):
            _start = timestep - save_length
            fname = f"{save_dir}/{_start}_{timestep}.mp4"
            self.cam.stop_recording(save_to_filename=fname, fps=30)
            self.is_recording = False
        if self.is_recording:
            dtheta = (2 * np.pi) / (save_length if save_length > 0 else 1)
            if not self.sim_cfg.camera_mov:
                dtheta = 0.0
            self.cam_theta = (self.cam_theta + dtheta) % (2 * np.pi)
            self.cam.set_pose(
                pos=(
                    self.sim_cfg.camera_radius * np.sin(self.cam_theta),
                    self.sim_cfg.camera_radius * np.cos(self.cam_theta),
                    self.sim_cfg.camera_ht,
                ),
                lookat=self.camera_lookat,
            )
            img, _, _, _ = self.cam.render(depth=True, segmentation=True)
            return img

    def step(self) -> None:
        assert self.scene is not None
        if self.sim_cfg.apply_random_force:
            if self._env_random_force is None:
                self.resample_random_force(env_ids=None)
            link_idx = self._get_base_link_idx(
                self.env.physics_manager.scene_entities["object"].object
            )
            self.scene.sim.rigid_solver.apply_links_external_force(
                force=self._env_random_force, links_idx=link_idx, ref="root_com"
            )
        object_entity: ArtObject = self.scene_entities["object"]
        object_entity.apply_rest_joint_targets()
        self.scene.step()
