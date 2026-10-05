from pathlib import Path

import genesis as gs
import numpy as np
import torch
import trimesh
from urdfpy import URDF

from dexlink.utils.retargeting.math_utils import matrix_from_quat, rot6d_from_matrix
from dexlink.utils.retargeting.mesh_utils import compute_mesh_transforms, construct_pytorch_mesh
from dexlink.utils.retargeting.path_utils import process_urdf


def recompute_inertia_from_mesh(mesh: trimesh.Trimesh, mass: float):
    # assume uniform density, compute the origin (xyz)
    # and inertia tensor (ixx, iyy, izz, ixy, ixz, iyz) from the mesh
    pass


def get_object_cfg(
    reference_demo,
    urdf_path: Path | str | None = None,
    obj_name: str | None = None,
    object_fixed=False,
):
    if urdf_path is None:
        urdf_path = reference_demo["configs_object"]["urdf_path"]
    if obj_name is None:
        obj_name = reference_demo["configs_object"]["obj_name"]

    # this assumes that the urdf exist in current assets dir

    # relative_asset_path = str(urdf_path).split('assets/')[-1]
    # urdf_path = get_assets_dir() / relative_asset_path
    assert urdf_path is not None
    urdf_path = process_urdf(urdf_path)

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
        "kp": 0.0,
        "kv": 0.0,
        "force_range": 100.0,
        "collect_data": False,
        "offset_pos": [0.0, 0.0, 0.0],  # useful for visualization
        "color": None,
        "show_link_frame": False,
        "scale": 1.0,  # scale down the object by default
        "object_mass_range": getattr(reference_demo, "object_mass_range", None),
    }
    return obj_cfg


class ArtObject:
    def __init__(self, scene, obj_cfg, num_envs: int):
        self.scene = scene
        self.obj_cfg = obj_cfg
        self.num_envs = num_envs

        # urdf path could be an obj file as well

        urdf_path = obj_cfg["urdf_path"]

        if str(urdf_path).endswith(".obj"):
            # create a simple urdf wrapper
            from dexlink.utils.retargeting.urdf_utils import create_urdf_from_obj

            urdf_path = create_urdf_from_obj(
                obj_path=urdf_path,
                save_dir=Path(urdf_path).parent,
                scale=obj_cfg["scale"],
            )
            obj_cfg["urdf_path"] = urdf_path

        self.urdf_inst = URDF.load(obj_cfg["urdf_path"])
        joints = [j for j in self.urdf_inst.joints if j.joint_type != "fixed"]

        if len(joints) == 0:
            self.actuated = False
            self.joints = []
        else:
            self.actuated = True
            self.joints = joints

        self.object = scene.add_entity(
            material=gs.materials.Rigid(
                rho=10.0,
                friction=1.2,
            ),
            morph=gs.morphs.URDF(
                file=obj_cfg["urdf_path"],
                fixed=obj_cfg["fixed"],
                recompute_inertia=False,
                collision=True,
                convexify=obj_cfg["convexify"],
                # pos   = obj_cfg['base_init_pos'],
                # quat  = obj_cfg['base_init_quat'],
            ),
        )
        self.mesh = self._mesh()
        self.dof_map = self._dof_map()
        self.link_names = [link.name for link in self.urdf_inst.links]
        self.link_idx_local = [
            self.object.get_link(link_name).idx_local for link_name in self.link_names
        ]
        self.default_link_mass, self.default_link_mass_available = (
            self._default_link_mass()
        )
        self.object_mass_range = self._parse_object_mass_range(
            obj_cfg["object_mass_range"]
        )
        self.env_link_mass = np.repeat(
            self.default_link_mass[None, :], self.num_envs, axis=0
        )

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

    def set_gains(self):
        if not self.actuated:
            return

        dofs_idx = [self.object.get_joint(j.name).dofs_idx_local for j in self.joints]
        assert all(
            [len(v) == 1 for v in dofs_idx]
        ), "Each joint should map to exactly one DOF"
        # flatten the dofs list
        dofs_idx = [v[0] for v in dofs_idx]

        kp = self.obj_cfg["kp"]
        kv = self.obj_cfg["kv"]
        force_range = self.obj_cfg["force_range"]
        n_dofs = len(dofs_idx)

        self.object.set_dofs_kp(
            kp=np.full((self.num_envs, n_dofs), kp, dtype=np.float32),
            dofs_idx_local=dofs_idx,
        )
        self.object.set_dofs_kv(
            kv=np.full((self.num_envs, n_dofs), kv, dtype=np.float32),
            dofs_idx_local=dofs_idx,
        )
        self.object.set_dofs_force_range(
            lower=np.full((self.num_envs, n_dofs), -force_range, dtype=np.float32),
            upper=np.full((self.num_envs, n_dofs), force_range, dtype=np.float32),
            dofs_idx_local=dofs_idx,
        )

    def _dof_map(self):
        if self.actuated:
            result = [self.object.get_joint(j.name).dofs_idx_local for j in self.joints]
            assert all(
                [len(v) == 1 for v in result]
            ), "Each joint should map to exactly one DOF"
            # flatten the dofs list
            return [v[0] for v in result]
        else:
            return []

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

    def domain_randomization_summary(self) -> dict[str, list[str]]:
        return {
            "mass_available": [
                self.link_names[idx]
                for idx, available in enumerate(self.default_link_mass_available)
                if available
            ],
            "mass_skipped": [
                self.link_names[idx]
                for idx, available in enumerate(self.default_link_mass_available)
                if not available
            ],
        }

    def randomize_dynamics(self, reset_cfg, env_ids: np.ndarray):
        if len(env_ids) == 0 or len(self.link_idx_local) == 0:
            return

        if not np.any(self.default_link_mass_available):
            return

        default_masses = self.default_link_mass[self.default_link_mass_available]
        if self.object_mass_range is None:
            mass_scale = np.random.uniform(
                reset_cfg.dr_object_mass[0],
                reset_cfg.dr_object_mass[1],
                size=(len(env_ids), int(self.default_link_mass_available.sum())),
            ).astype(np.float32)
            masses = default_masses[None, :] * mass_scale
        else:
            default_total_mass = float(default_masses.sum())
            if default_total_mass <= 0.0:
                raise ValueError(
                    "Cannot apply object_mass_range because available object link "
                    f"masses sum to {default_total_mass}."
                )
            sampled_total_mass = np.random.uniform(
                self.object_mass_range[0],
                self.object_mass_range[1],
                size=(len(env_ids), 1),
            ).astype(np.float32)
            masses = default_masses[None, :] * (sampled_total_mass / default_total_mass)
        self.env_link_mass[
            np.ix_(np.asarray(env_ids), self.default_link_mass_available)
        ] = masses

        links_idx_local = np.asarray(self.link_idx_local)[
            self.default_link_mass_available
        ]

        self.object.set_links_inertial_mass(
            inertial_mass=self.env_link_mass[:, self.default_link_mass_available],
            links_idx_local=links_idx_local,
        )

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
            self.object.set_pos(
                pos=state["pos"],
                envs_idx=env_idx,
            )
        if state["quat"] is not None:
            self.object.set_quat(
                quat=state["quat"],
                envs_idx=env_idx,
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
                lower_limit, upper_limit = self.object.get_dofs_limit(
                    dofs_idx_local=dofs_idx
                )
                jvals = torch.clip(jvals, lower_limit, upper_limit)
                self.object.set_dofs_position(
                    jvals,
                    dofs_idx,
                    envs_idx=env_idx,
                )

            # set jvel to zero for now
            jvels = torch.tensor([0.0] * len(self.joints), dtype=torch.float32)
            dofs_idx = self.dof_map
            jvels = jvels[None, :].repeat(n_envs, 1)  # (num_envs, num_dofs)
            self.object.set_dofs_velocity(
                jvels,
                dofs_idx,
                envs_idx=env_idx,
            )

    def get_obs(self, rot_format: str = "quat") -> torch.Tensor:
        pos, quat = self.object.get_pos(), self.object.get_quat()

        if rot_format == "6d":
            # convert to 6D as [col0, col1]
            mat = matrix_from_quat(quat)  # (n, 3, 3)
            rot_6d = rot6d_from_matrix(mat)
            quat = rot_6d

        # pos, quat, jpos
        if self.actuated:
            dofs_idx = self.dof_map
            jpos = self.object.get_dofs_position(dofs_idx).detach()
            obs = torch.cat([pos, quat, jpos], dim=1)
        else:
            obs = torch.cat([pos, quat], dim=1)

        return obs

    def get_vel_obs(self):
        # Return velocity observations for the object
        lin_vel = self.object.get_vel()
        ang_vel = self.object.get_ang()

        obs = [lin_vel, ang_vel]

        # if actuated, include the joint vel
        if self.actuated:
            dofs_idx = self.dof_map
            jvel = self.object.get_dofs_velocity(dofs_idx).detach()
            obs.append(jvel)

        return torch.cat(obs, dim=1)

    def get_ground_truth_info(self) -> dict:
        asset = self.object
        if self.actuated:
            arti = asset.get_dofs_position(self.dof_map)
        else:
            # Keep a consistent tensor type/shape for evaluation logging on rigid objects.
            # An empty articulation vector lets downstream code stack trajectories and
            # compute "arti" metrics (which become zero distances) without special cases.
            pos = asset.get_pos()
            arti = torch.zeros((pos.shape[0], 0), dtype=pos.dtype, device=pos.device)
        return {
            "pos": asset.get_pos(),
            "quat": asset.get_quat(),
            "arti": arti,
        }

    def control_object(self, action: torch.Tensor):
        if not self.actuated:
            return

        # similar to control robot joints
        dofs_idx = self.dof_map
        lower_limit, upper_limit = self.object.get_dofs_limit(dofs_idx_local=dofs_idx)
        jvals = torch.clip(action, lower_limit, upper_limit)
        self.object.control_dofs_position(jvals, dofs_idx)

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
