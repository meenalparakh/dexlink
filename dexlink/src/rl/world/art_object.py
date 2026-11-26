from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from urdfpy import URDF

from dexlink.utils.retargeting.math_utils import matrix_from_quat, rot6d_from_matrix
from dexlink.utils.retargeting.mesh_utils import (
    compute_mesh_transforms,
    construct_pytorch_mesh,
    sample_kpts_with_normals,
)
from dexlink.utils.retargeting.path_utils import process_urdf
from dexlink.src.rl.config.passive_mechanisms import CLIP_ROBOT, NUT_ROBOT, SCREW_ROBOT

if TYPE_CHECKING:
    from dexlink.src.rl.world.physics_sim import PhysicsSim

DISPENSER_FINAL_ROBOT_NAME = "dispenser_variation_prismatic_v5"
DISPENSER_REST_JOINT_TARGETS = {"neck_to_head_prismatic": 0.0}
DISPENSER_REST_JOINT_KP = 2500.0
DISPENSER_REST_JOINT_KV = 100.0
DISPENSER_REST_JOINT_FORCE_RANGE = 10.0


def dispenser_rest_joint_targets(urdf_inst: URDF) -> dict[str, float]:
    if urdf_inst.name != DISPENSER_FINAL_ROBOT_NAME:
        return {}

    movable_joints = [
        joint for joint in urdf_inst.joints if joint.joint_type != "fixed"
    ]
    expected_joint_names = tuple(DISPENSER_REST_JOINT_TARGETS)
    received_joint_names = tuple(joint.name for joint in movable_joints)
    if received_joint_names != expected_joint_names:
        raise ValueError(
            f"Dispenser asset {DISPENSER_FINAL_ROBOT_NAME!r} must contain exactly "
            f"the movable joints {expected_joint_names}, got {received_joint_names}."
        )

    pump_joint = movable_joints[0]
    if pump_joint.joint_type != "prismatic":
        raise ValueError(
            f"Expected dispenser pump joint {pump_joint.name!r} to be prismatic, "
            f"got {pump_joint.joint_type!r}."
        )
    if pump_joint.limit is None or not np.isclose(pump_joint.limit.lower, -0.01):
        received_lower = None if pump_joint.limit is None else pump_joint.limit.lower
        raise ValueError(
            f"Expected dispenser pump lower limit -0.01 m, got {received_lower!r}."
        )
    if pump_joint.limit is None or not np.isclose(pump_joint.limit.upper, 0.0):
        received_upper = None if pump_joint.limit is None else pump_joint.limit.upper
        raise ValueError(
            f"Expected dispenser pump upper/rest limit 0.0 m, got {received_upper!r}."
        )
    return DISPENSER_REST_JOINT_TARGETS.copy()


def recompute_inertia_from_mesh(mesh: trimesh.Trimesh, mass: float):
    # assume uniform density, compute the origin (xyz)
    # and inertia tensor (ixx, iyy, izz, ixy, ixz, iyz) from the mesh
    pass


def get_object_cfg(
    reference_demo,
    urdf_path: Path | str | None = None,
    obj_name: str | None = None,
    object_fixed=False,
    friction: float = 1.0,
):
    if urdf_path is None:
        urdf_path = reference_demo["configs_object"]["urdf_path"]
    if obj_name is None:
        obj_name = reference_demo["configs_object"]["obj_name"]

    # this assumes that the urdf exist in current assets dir

    # relative_asset_path = str(urdf_path).split('assets/')[-1]
    # urdf_path = get_assets_dir() / relative_asset_path
    assert urdf_path is not None
    # Cardboard behavior is enabled only for these exact procedural asset paths.
    from dexlink.src.rl.world.cardboard_object import cardboard_model_for_path

    cardboard_model = cardboard_model_for_path(urdf_path)
    # Cluster jobs may construct the same processed URDF concurrently. A
    # process-specific copy prevents one run from replacing another run's file.
    urdf_path = process_urdf(urdf_path, append_jobname=True)
    urdf_inst = URDF.load(str(urdf_path))
    rest_joint_targets = dispenser_rest_joint_targets(urdf_inst)
    is_final_dispenser = bool(rest_joint_targets)

    obj_cfg = {
        "name": obj_name,
        "num_sample_vertics": 300,
        "urdf_path": str(urdf_path),
        "convexify": True,
        "fixed": object_fixed,
        # "actuated": True, # whether to control the object joint!
        # "kp": 1000.0,
        # "kv": 100.0,
        # "force_range": 200.0,
        "kp": DISPENSER_REST_JOINT_KP if is_final_dispenser else 0.0,
        "kv": DISPENSER_REST_JOINT_KV if is_final_dispenser else 0.0,
        "force_range": (
            DISPENSER_REST_JOINT_FORCE_RANGE if is_final_dispenser else 100.0
        ),
        "rest_joint_targets": (rest_joint_targets if is_final_dispenser else {}),
        "cardboard_model": cardboard_model,
        "passive_mechanism": urdf_inst.name in (SCREW_ROBOT, CLIP_ROBOT, NUT_ROBOT),
        "collect_data": False,
        "offset_pos": [0.0, 0.0, 0.0],  # useful for visualization
        "color": None,
        "show_link_frame": False,
        "scale": 1.0,  # scale down the object by default
        "object_mass_range": getattr(reference_demo, "object_mass_range", None),
        "friction": float(friction),
    }
    return obj_cfg


class ArtObject:
    def __init__(self, physics: PhysicsSim, obj_cfg, num_envs: int):
        self.physics = physics
        self.obj_cfg = obj_cfg
        self.num_envs = num_envs

        # urdf path could be an obj file as well

        urdf_path = obj_cfg["urdf_path"]

        if str(urdf_path).endswith(".obj"):
            raise ValueError(f"Obtained {str(urdf_path)} to target object. Not a urdf.")
            # create a simple urdf wrapper
            from dexlink.utils.retargeting.urdf_utils import create_urdf_from_obj

            urdf_path = create_urdf_from_obj(
                obj_path=urdf_path,
                save_dir=Path(urdf_path).parent,
                scale=obj_cfg["scale"],
            )
            obj_cfg["urdf_path"] = urdf_path

        self.urdf_inst = URDF.load(obj_cfg["urdf_path"])
        self.link_names = [link.name for link in self.urdf_inst.links]
        joints = [j for j in self.urdf_inst.joints if j.joint_type != "fixed"]

        if len(joints) == 0:
            self.actuated = False
            self.joints = []
        else:
            self.actuated = True
            self.joints = joints

        self.object = physics.add_rigid_urdf_entity(
            urdf_path=obj_cfg["urdf_path"],
            name=obj_cfg["name"],
            fixed=obj_cfg["fixed"],
            collision=True,
            convexify=obj_cfg["convexify"],
            recompute_inertia=False,
            friction=float(obj_cfg["friction"]),
            gravity_compensation=0.0,
            pos=(0, 0, 0),
            quat=(1.0, 0.0, 0.0, 0.0),  # wxyz
            links_to_keep=self.link_names,
        )
        self.mesh = self._mesh()
        self.object_surface_kpts = None
        self.object_surface_normals = None
        self.dof_map = self._dof_map()
        self.rest_joint_names, self.rest_joint_dofs, self.rest_joint_values = (
            self._rest_joint_targets()
        )
        self.link_idx_local = self._link_idx_local()
        self.default_link_mass, self.default_link_mass_available = (
            self._default_link_mass()
        )
        self.object_mass_range = self._parse_object_mass_range(
            obj_cfg["object_mass_range"]
        )
        self.env_link_mass = np.repeat(
            self.default_link_mass[None, :], self.num_envs, axis=0
        )
        self.env_link_friction_ratio = np.ones(
            (self.num_envs, len(self.link_idx_local)), dtype=np.float32
        )
        self._initialize_surface_samples()

    def _entity_link_names_for_debug(self) -> list[str]:
        for attr_name in ("links", "_links"):
            links = getattr(self.object, attr_name, None)
            if links is not None:
                return [str(getattr(link, "name", link)) for link in links]
        get_links = getattr(self.object, "get_links", None)
        if callable(get_links):
            links = get_links()
            return [str(getattr(link, "name", link)) for link in links]
        return []

    def _link_idx_local(self) -> list[int]:
        link_idx_local: list[int] = []
        missing_link_names: list[str] = []
        first_error: Exception | None = None
        root_link_name = self.urdf_inst.base_link.name
        available_link_names = self._entity_link_names_for_debug()
        for link_name in self.link_names:
            try:
                link_idx_local.append(
                    self.physics.get_link_idx_local(self.object, link_name)
                )
            except Exception as exc:
                if link_name == root_link_name and "base" in available_link_names:
                    link_idx_local.append(
                        self.physics.get_link_idx_local(self.object, "base")
                    )
                    continue
                missing_link_names.append(link_name)
                if first_error is None:
                    first_error = exc
        if missing_link_names:
            raise ValueError(
                "Genesis did not instantiate all URDF links for object "
                f"{self.obj_cfg['name']!r} from {self.obj_cfg['urdf_path']!r}. "
                f"Missing links: {missing_link_names}. "
                f"URDF root link: {root_link_name!r}. "
                f"URDF links requested via links_to_keep: {self.link_names}. "
                f"Genesis available links: {available_link_names}."
            ) from first_error
        return link_idx_local

    def _parse_object_mass_range(
        self, object_mass_range: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if object_mass_range is None:
            return None
        if len(object_mass_range) != 2:
            raise ValueError(
                f"Expected object_mass_range to have two values, got {object_mass_range}."
            )
        mass_min, mass_max = object_mass_range
        mass_min = float(mass_min)
        mass_max = float(mass_max)
        if mass_min <= 0.0 or mass_max <= 0.0 or mass_min > mass_max:
            raise ValueError(
                f"Expected object_mass_range to be positive and ordered, got {object_mass_range}."
            )
        return (mass_min, mass_max)

    def _mesh(self):
        # use the urdf inst to get the mesh
        # for now just return a cube trimesh as a placeholder
        box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        return box

    def _initialize_surface_samples(self) -> None:
        task_cfg = self.physics.env.cfg.task_cfg
        asset_mesh = construct_pytorch_mesh(
            urdf_inst=self.urdf_inst, device=self.physics.env.device
        )
        object_surface_kpts, object_surface_normals = sample_kpts_with_normals(
            asset_mesh,
            max_pts=task_cfg.max_pts_per_link,
        )
        for link_idx in range(len(object_surface_kpts)):
            if (
                object_surface_kpts[link_idx].shape
                != object_surface_normals[link_idx].shape
            ):
                raise ValueError(
                    "Expected object surface keypoints and normals to have matching "
                    f"shapes for link {link_idx}, got "
                    f"{object_surface_kpts[link_idx].shape} and "
                    f"{object_surface_normals[link_idx].shape}."
                )
        self.object_surface_kpts = object_surface_kpts
        self.object_surface_normals = object_surface_normals

    def _dof_map(self):
        if self.actuated:
            result = [
                self.physics.get_joint_dofs_idx_local(self.object, j.name)
                for j in self.joints
            ]
            assert all(
                [len(v) == 1 for v in result]
            ), "Each joint should map to exactly one DOF"
            # flatten the dofs list
            return [v[0] for v in result]
        else:
            return []

    def _rest_joint_targets(self) -> tuple[list[str], list[int], torch.Tensor]:
        configured_targets = self.obj_cfg["rest_joint_targets"]
        if not isinstance(configured_targets, dict):
            raise ValueError(
                "Expected obj_cfg['rest_joint_targets'] to be a dict, got "
                f"{type(configured_targets).__name__}."
            )
        if not configured_targets:
            return [], [], torch.empty(0, dtype=torch.float32)
        if not self.actuated:
            raise ValueError(
                f"Object {self.obj_cfg['name']!r} has rest joint targets but no actuated joints."
            )

        joint_dofs_by_name = dict(
            zip((joint.name for joint in self.joints), self.dof_map)
        )
        unknown_joint_names = sorted(set(configured_targets) - set(joint_dofs_by_name))
        if unknown_joint_names:
            raise ValueError(
                f"Object {self.obj_cfg['name']!r} rest joint targets reference unknown "
                f"joints {unknown_joint_names}; available joints are "
                f"{sorted(joint_dofs_by_name)}."
            )

        joint_names = list(configured_targets)
        joint_dofs = [joint_dofs_by_name[joint_name] for joint_name in joint_names]
        joint_values = torch.tensor(
            [float(configured_targets[joint_name]) for joint_name in joint_names],
            dtype=torch.float32,
        )
        return joint_names, joint_dofs, joint_values

    def _default_link_mass(self) -> tuple[np.ndarray, np.ndarray]:
        masses = np.zeros(len(self.urdf_inst.links), dtype=np.float32)
        available = np.zeros(len(self.urdf_inst.links), dtype=bool)
        for idx, link in enumerate(self.urdf_inst.links):
            inertial = getattr(link, "inertial", None)
            if inertial is None or inertial.mass is None:
                continue
            masses[idx] = float(inertial.mass)
            available[idx] = True
        return masses, available

    def set_state(
        self,
        pos=None,
        quat=None,
        lin_vel=None,
        ang_vel=None,
        jpos: torch.Tensor | None = None,
        jvel: torch.Tensor | None = None,
        env_idx=None,
    ):

        assert (
            lin_vel is None and ang_vel is None
        ), "Setting object velocity not supported yet"
        assert jvel is None, "Setting object joint velocity not supported yet"

        state = {
            "pos": pos,
            "quat": quat,
            # 'lin_vel': lin_vel,
            # 'ang_vel': ang_vel,
        }

        n_envs = self.num_envs if env_idx is None else len(env_idx)

        for k, v in state.items():
            if v is not None:
                if v.shape == (v.shape[-1],):
                    v = v[None].repeat(n_envs, 1)
                assert (
                    v.shape[0] == n_envs
                ), f"{k} shape mismatch: expected first dim {n_envs}, got {v.shape[0]}"
                state[k] = v

        if state["pos"] is not None:
            self.physics.set_entity_pos(self.object, pos=state["pos"], envs_idx=env_idx)
        if state["quat"] is not None:
            self.physics.set_entity_quat(
                self.object, quat=state["quat"], envs_idx=env_idx
            )

        if self.actuated:
            if jpos is not None:
                dofs_idx = self.dof_map
                jvals = torch.as_tensor(jpos, dtype=torch.float32)
                if len(jvals.shape) == 1:
                    jvals = jvals[None, :]
                    jvals = jvals.repeat(n_envs, 1)
                assert jvals.shape == (
                    n_envs,
                    len(self.joints),
                ), f"jpos shape mismatch: expected {(n_envs, len(self.joints))}, got {jvals.shape}"
                lower_limit, upper_limit = self.physics.get_entity_dofs_limit(
                    self.object, dofs_idx_local=dofs_idx
                )
                if env_idx is not None:
                    lower_limit = lower_limit[env_idx]
                    upper_limit = upper_limit[env_idx]
                jvals = torch.clip(jvals, lower_limit, upper_limit)
                self.physics.set_entity_dofs_position(
                    self.object, jvals, dofs_idx, envs_idx=env_idx
                )

            # set jvel to zero for now
            jvels = torch.tensor([0.0] * len(self.joints), dtype=torch.float32)
            dofs_idx = self.dof_map
            jvels = jvels[None, :].repeat(n_envs, 1)  # (num_envs, num_dofs)
            self.physics.set_entity_dofs_velocity(
                self.object, jvels, dofs_idx, envs_idx=env_idx
            )

    def get_obs(self, rot_format: str = "quat") -> torch.Tensor:
        pos = self.physics.get_entity_pos(self.object)
        quat = self.physics.get_entity_quat(self.object)

        if rot_format == "6d":
            # convert to 6D as [col0, col1]
            mat = matrix_from_quat(quat)  # (n, 3, 3)
            rot_6d = rot6d_from_matrix(mat)
            quat = rot_6d

        # pos, quat, jpos
        if self.actuated:
            dofs_idx = self.dof_map
            jpos = self.physics.get_entity_dofs_position(self.object, dofs_idx).detach()
            obs = torch.cat([pos, quat, jpos], dim=1)
        else:
            obs = torch.cat([pos, quat], dim=1)

        return obs

    def get_vel_obs(self):
        # Return velocity observations for the object
        lin_vel = self.physics.get_entity_vel(self.object)
        ang_vel = self.physics.get_entity_ang(self.object)

        obs = [lin_vel, ang_vel]

        # if actuated, include the joint vel
        if self.actuated:
            dofs_idx = self.dof_map
            jvel = self.physics.get_entity_dofs_velocity(self.object, dofs_idx).detach()
            obs.append(jvel)

        return torch.cat(obs, dim=1)

    def get_ground_truth_info(self) -> dict:
        asset = self.object
        if self.actuated:
            arti = self.physics.get_entity_dofs_position(asset, self.dof_map)
        else:
            # Keep a consistent tensor type/shape for evaluation logging on rigid objects.
            # An empty articulation vector lets downstream code stack trajectories and
            # compute "arti" metrics (which become zero distances) without special cases.
            pos = self.physics.get_entity_pos(asset)
            arti = torch.zeros((pos.shape[0], 0), dtype=pos.dtype, device=pos.device)
        return {
            "pos": self.physics.get_entity_pos(asset),
            "quat": self.physics.get_entity_quat(asset),
            "arti": arti,
        }

    def control_object(self, action: torch.Tensor):
        if not self.actuated:
            return

        raise NotImplementedError("object control has been blocked for now.")
        # similar to control robot joints
        dofs_idx = self.dof_map
        lower_limit, upper_limit = self.physics.get_entity_dofs_limit(
            self.object, dofs_idx_local=dofs_idx
        )
        jvals = torch.clip(action, lower_limit, upper_limit)
        self.physics.control_entity_dofs_position(self.object, jvals, dofs_idx)

    def apply_rest_joint_targets(self) -> None:
        if not self.rest_joint_dofs:
            return
        target_values = self.rest_joint_values[None, :].repeat(self.num_envs, 1)
        self.physics.control_entity_dofs_position(
            self.object,
            target_values,
            self.rest_joint_dofs,
        )

    def compute_mesh(self, pos, quat, arti):
        if arti is None and not self.actuated:
            arti = torch.zeros(
                (pos.shape[0], 0), dtype=torch.float32, device=pos.device
            )
        return compute_mesh_transforms(
            pos, quat, arti, self.urdf_inst, [j.name for j in self.joints]
        )

    def construct_pytorch_meshes(self, device="cuda"):
        return construct_pytorch_mesh(None, device=device, urdf_inst=self.urdf_inst)
