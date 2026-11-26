from __future__ import annotations

import genesis as gs
import numpy as np
import torch

from dexlink.src.retargeting.data.demo_loader import is_oakink_dataset
from dexlink.utils.retargeting.mesh_utils import compute_mesh_transforms
from dexlink.src.rl.world.art_object import ArtObject, get_object_cfg
from dexlink.src.rl.world.cardboard_object import CardboardObject
from dexlink.src.rl.world.mechanism_object import MechanismObject
from dexlink.src.rl.world.physics_sim import PhysicsSim
from dexlink.src.rl.world.robot import Hand, get_hand_specific_cfg


class MultiObjectPhysicsSim(PhysicsSim):
    """Simulation backend that instantiates all exported scene objects."""

    @property
    def object_entities(self) -> dict[str, ArtObject]:
        objects = self.scene_entities.get("objects")
        if not isinstance(objects, dict):
            raise ValueError("MultiObjectPhysicsSim scene has not created objects.")
        return objects

    def _populate_scene(self) -> None:
        assert self.scene is not None
        task_manager = self.env.task_manager

        self.scene_entities.clear()
        self.hand_sensors = None
        self.hand_cfgs = None
        self.object_cfg = None
        self.hammer_peg_force_sensor = None
        self._scene_table_top_height = None
        self.scene_entities["plane"] = self._add_plane(pos=(0, 0, 0))

        if not (
            is_oakink_dataset(self.env.task_cfg.demo_dataset)
            or self.env.task_cfg.demo_dataset == "random_traj"
        ):
            raise ValueError(
                "MultiObjectPhysicsSim expects an OakInk sequence or native robot trajectory "
                f"dataset, got {self.env.task_cfg.demo_dataset!r}."
            )

        self._populate_scene_objects()
        self._add_task_metric_sensors()

        if self.env.task_manager.object_only:
            return

        self.hand_cfgs = get_hand_specific_cfg(self.env.task_cfg.hand_name)
        robot_base_pose = self._resolve_robot_base_pose(
            getattr(task_manager.original_demo, "robot_base_pose", None)
        )
        if self.env.task_manager.has_hand("left"):
            self.scene_entities["left"] = Hand(
                physics=self,
                robo_cfg=self._hand_cfg_for_role("left"),
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
                robo_cfg=self._hand_cfg_for_role("right"),
                num_envs=self.sim_cfg.num_envs,
                device=self.env.device,
                obs_cfg=self.env.cfg.obs_cfg.hand_cfg,
                env=self.env,
                name="right",
                base_pose=robot_base_pose,
            )

    def _add_task_metric_sensors(self) -> None:
        metric_cfg = self.env.task_cfg.metric_cfg
        if metric_cfg.metric_id not in {
            "hammer_peg_impact",
            "hammer_single_swing",
            "hammer_two_swing",
        }:
            return
        assert self.scene is not None
        peg = self.object_entities[metric_cfg.hammer_peg_object_id]
        if len(peg.link_idx_local) != 1:
            raise ValueError(
                "Hammer peg force sensor expects a single-link peg, got local "
                f"link indices {peg.link_idx_local} for "
                f"{metric_cfg.hammer_peg_object_id!r}."
            )
        self.hammer_peg_force_sensor = self.scene.add_sensor(
            gs.sensors.ContactForce(
                entity_idx=int(peg.object.idx),
                link_idx_local=int(peg.link_idx_local[0]),
                pos_offset=(
                    0.0,
                    0.0,
                    float(metric_cfg.hammer_peg_sensor_top_offset_m),
                ),
                update_ground_truth_only=True,
            )
        )

    def get_hammer_peg_contact_impact(self) -> tuple[torch.Tensor, torch.Tensor]:
        metric_cfg = self.env.task_cfg.metric_cfg
        if metric_cfg.metric_id not in {
            "hammer_peg_impact",
            "hammer_single_swing",
            "hammer_two_swing",
        }:
            raise ValueError(
                "Hammer impact requires metric_id='hammer_peg_impact', "
                "'hammer_single_swing', or 'hammer_two_swing'; got "
                f"{metric_cfg.metric_id!r}."
            )
        if self.hammer_peg_force_sensor is None:
            raise ValueError("Hammer peg force sensor has not been initialized.")
        sensor_force = self.hammer_peg_force_sensor.read_ground_truth().to(
            device=self.env.device,
            dtype=torch.float32,
        )
        expected_shape = (self.sim_cfg.num_envs, 3)
        if tuple(sensor_force.shape) != expected_shape:
            raise ValueError(
                f"Expected hammer peg force sensor shape {expected_shape}, "
                f"got {tuple(sensor_force.shape)}."
            )

        peg = self.object_entities[metric_cfg.hammer_peg_object_id]
        hammer = self.object_entities[metric_cfg.hammer_object_id]
        contact_info = self.get_filtered_contact_info(
            entity_a=peg.object,
            entity_b=hammer.object,
            return_link_force=True,
        )
        contact_force = contact_info["contact_force_link_a"]
        if contact_force.ndim != 4 or contact_force.shape[0] != self.sim_cfg.num_envs:
            raise ValueError(
                "Expected hammer-peg link contact force shape (N, L_peg, "
                f"L_hammer, 3), got {tuple(contact_force.shape)}."
            )
        contact_mask = contact_force.norm(dim=-1).reshape(
            self.sim_cfg.num_envs, -1
        ).amax(dim=-1) > float(metric_cfg.hammer_contact_force_threshold)
        impact = sensor_force.norm(dim=-1) * contact_mask.to(sensor_force.dtype)
        return impact, contact_mask

    def _scene_table_height_for_robot_base(self) -> float:
        if self._scene_table_top_height is not None:
            return self._scene_table_top_height

        task_manager = self.env.task_manager
        table_object_id = self._scene_table_object_id()
        table_object_index = task_manager.scene_object_ids.index(table_object_id)
        table_entity = self.object_entities[table_object_id]
        table_joint_names = list(
            task_manager.scene_object_arti_names[table_object_index]
        )
        table_pos = torch.as_tensor(
            task_manager.scene_object_pos[0:1, table_object_index],
            dtype=torch.float32,
            device=self.env.device,
        )
        table_quat = torch.as_tensor(
            task_manager.scene_object_quat[0:1, table_object_index],
            dtype=torch.float32,
            device=self.env.device,
        )
        table_arti = torch.as_tensor(
            task_manager.scene_object_arti_vals[
                0:1, table_object_index, : len(table_joint_names)
            ],
            dtype=torch.float32,
            device=self.env.device,
        )
        mesh_transforms = compute_mesh_transforms(
            table_pos,
            table_quat,
            table_arti,
            urdf_inst=table_entity.urdf_inst,
            joint_names=table_joint_names,
        )

        table_top_z = None
        for mesh, transform in mesh_transforms.items():
            vertices = torch.as_tensor(
                mesh.vertices,
                dtype=torch.float32,
                device=transform.device,
            )
            ones = torch.ones(
                (vertices.shape[0], 1),
                dtype=vertices.dtype,
                device=vertices.device,
            )
            vertices_h = torch.cat([vertices, ones], dim=-1)
            world_vertices = torch.matmul(transform[0], vertices_h.T).T[:, :3]
            mesh_top_z = world_vertices[:, 2].max()
            table_top_z = (
                mesh_top_z
                if table_top_z is None
                else torch.maximum(table_top_z, mesh_top_z)
            )
        if table_top_z is None:
            raise ValueError(
                f"Table object {table_object_id!r} has no collision meshes in "
                f"{table_entity.obj_cfg['urdf_path']!r}."
            )
        self._scene_table_top_height = float(table_top_z.detach().cpu().item())
        return self._scene_table_top_height

    def get_scene_table_top_height(self) -> float:
        return self._scene_table_height_for_robot_base()

    def _scene_table_object_id(self) -> str:
        task_manager = self.env.task_manager
        matches = [
            object_id
            for object_id, object_name in zip(
                task_manager.scene_object_ids,
                task_manager.scene_object_names,
            )
            if ("table" in object_id.lower()) or ("table" in object_name.lower())
        ]
        if len(matches) != 1:
            raise ValueError(
                "franka_xhand multi-object base placement expects exactly one table "
                "scene object. "
                f"Matched {matches} from ids={task_manager.scene_object_ids}, "
                f"names={task_manager.scene_object_names}."
            )
        return matches[0]

    def _hand_cfg_for_role(self, role_side: str) -> dict:
        if self.hand_cfgs is None:
            raise ValueError("Hand cfgs have not been initialized.")
        if role_side not in ("left", "right"):
            raise ValueError(
                f"Expected hand role side 'left' or 'right', got {role_side!r}."
            )
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
        if config_side not in self.hand_cfgs:
            raise ValueError(
                f"Hand config source side {config_side!r} for role {role_side!r} "
                f"is not available in hand cfgs {sorted(self.hand_cfgs)}."
            )
        return self.hand_cfgs[config_side]

    def _populate_scene_objects(self) -> None:
        task_manager = self.env.task_manager
        object_entities: dict[str, ArtObject] = {}
        object_cfgs: dict[str, dict] = {}

        for object_id, object_name, urdf_path, object_static, object_friction in zip(
            task_manager.scene_object_ids,
            task_manager.scene_object_names,
            task_manager.scene_object_urdf_paths,
            task_manager.scene_object_static,
            task_manager.scene_object_friction,
        ):
            obj_cfg = get_object_cfg(
                reference_demo=None,
                urdf_path=urdf_path,
                obj_name=object_name,
                object_fixed=bool(self.env.task_cfg.object_fixed)
                or bool(object_static),
                friction=float(object_friction),
            )
            object_type = (
                MechanismObject
                if obj_cfg["passive_mechanism"]
                else (
                    CardboardObject
                    if obj_cfg["cardboard_model"] is not None
                    else ArtObject
                )
            )
            entity = object_type(
                physics=self,
                obj_cfg=obj_cfg,
                num_envs=self.sim_cfg.num_envs,
            )
            object_entities[object_id] = entity
            object_cfgs[object_id] = obj_cfg
            entity_key = task_manager.scene_object_entity_key(object_id)
            self.scene_entities[entity_key] = entity

        canonical_id = task_manager.scene_canonical_object_id
        self.scene_entities["objects"] = object_entities
        self.scene_entities["object"] = object_entities[canonical_id]
        self.object_cfg = object_cfgs[canonical_id]
        self.object_cfgs = object_cfgs

    def step(self) -> None:
        assert self.scene is not None
        if self.sim_cfg.apply_random_force:
            if self._env_random_force is None:
                self.resample_random_force(env_ids=None)
            for object_entity in self.object_entities.values():
                if bool(object_entity.obj_cfg["fixed"]):
                    continue
                link_idx = self._get_base_link_idx(object_entity.object)
                self.scene.sim.rigid_solver.apply_links_external_force(
                    force=self._env_random_force,
                    links_idx=link_idx,
                    ref="root_com",
                )
        for object_entity in self.object_entities.values():
            object_entity.apply_rest_joint_targets()
        self.scene.step()

    def _configure_object_gains(self) -> None:
        for object_entity in self.object_entities.values():
            if not object_entity.actuated:
                continue
            if isinstance(object_entity, (CardboardObject, MechanismObject)):
                object_entity.configure_joint_dynamics()
                continue

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

    def object_domain_randomization_summary(self) -> dict[str, list[str]]:
        available: list[str] = []
        skipped: list[str] = []
        for object_id, object_entity in self.object_entities.items():
            available.extend(
                f"{object_id}:{object_entity.link_names[idx]}"
                for idx, is_available in enumerate(
                    object_entity.default_link_mass_available
                )
                if is_available
            )
            skipped.extend(
                f"{object_id}:{object_entity.link_names[idx]}"
                for idx, is_available in enumerate(
                    object_entity.default_link_mass_available
                )
                if not is_available
            )
        return {"mass_available": available, "mass_skipped": skipped}

    def _randomize_object_dynamics(self, reset_cfg, env_ids: np.ndarray) -> None:
        # Object mass randomization is disabled for multi-object sequence tasks.
        # Keep surface-friction randomization active via apply_domain_randomization().
        return

    def _randomize_single_object_dynamics(
        self, object_entity: ArtObject, reset_cfg, env_ids: np.ndarray
    ) -> None:
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
        for object_entity in self.object_entities.values():
            self._randomize_single_object_surface_friction(
                object_entity, reset_cfg, env_ids
            )

    def get_object_pcd(self, normals: bool) -> "torch.Tensor":
        if normals:
            return self.env.observation_manager._compute_privileged_object_pcd()

        obs_mgr = self.env.observation_manager
        if obs_mgr.scene_cloud_contract is not None:
            from dexlink.utils.rl.scene_cloud import sample_scene_cloud

            # Use full geometry for actor sampling, independently of critic budget.
            chunks = []
            for object_id in obs_mgr.object_ids:
                entity = obs_mgr.object_entities[object_id]
                pos, quat, arti = obs_mgr._current_object_state_for_id(object_id)
                chunks.append(
                    obs_mgr.compute_object_pcd_obs_for_entity(
                        entity,
                        pos=pos,
                        quat=quat,
                        arti=arti,
                        normals=False,
                        num_pts_sample=-1,
                    )
                )
            return sample_scene_cloud(
                torch.cat(chunks, dim=1), obs_mgr.scene_cloud_contract
            )
        privileged_object_pcd = obs_mgr._cached_privileged_object_pcd
        if privileged_object_pcd is None:
            privileged_object_pcd = (
                obs_mgr._compute_privileged_object_pcd_from_object_state()
            )
            obs_mgr._cached_privileged_object_pcd = privileged_object_pcd

        chunks = []
        for object_id in obs_mgr.object_ids:
            offset = obs_mgr.object_pcd_privileged_offsets[object_id]
            chunks.append(
                self._prepare_policy_object_pcd(privileged_object_pcd[:, offset])
            )
        return torch.cat(chunks, dim=1)
