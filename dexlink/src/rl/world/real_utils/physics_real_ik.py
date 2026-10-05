"""Genesis IK helpers for the real backend."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from dexlink.src.rl.world.real_utils import RealRobotEntity
from dexlink.src.rl.world.real_utils.physics_real_mixin_base import PhysicsRealMixinBase
from dexlink.src.rl.world.robot import Hand


class PhysicsRealIKMixin(PhysicsRealMixinBase):
    def _initialize_ik_backend(self, hand: Hand) -> None:
        try:
            import genesis as gs
        except ImportError as exc:
            raise ImportError(
                "PhysicsReal IK backend requires genesis to be installed."
            ) from exc
        if self._ik_scene is not None:
            return
        gs.init(backend=gs.cpu)
        self._ik_scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=0.01,
                substeps=1,
                gravity=(0.0, 0.0, 0.0),
            ),
            show_viewer=False,
            vis_options=gs.options.VisOptions(show_world_frame=False),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=False,
                enable_self_collision=False,
                enable_joint_limit=True,
                max_collision_pairs=64,
                batch_links_info=True,
                batch_dofs_info=True,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        )
        base_pos = tuple(np.asarray(hand.base_pose[:3, 3], dtype=np.float32).tolist())
        base_quat_xyzw = R.from_matrix(hand.base_pose[:3, :3]).as_quat()
        base_quat_wxyz = (
            float(base_quat_xyzw[3]),
            float(base_quat_xyzw[0]),
            float(base_quat_xyzw[1]),
            float(base_quat_xyzw[2]),
        )
        self._ik_entity = self._ik_scene.add_entity(
            material=gs.materials.Rigid(friction=1.2, gravity_compensation=1.0),
            morph=gs.morphs.URDF(
                file=hand.robo_cfg["urdf_path"],
                fixed=True,
                recompute_inertia=False,
                collision=False,
                convexify=True,
                pos=base_pos,
                quat=base_quat_wxyz,
                links_to_keep=hand.hand_mapped_link_names,
            ),
        )
        self._ik_scene.build(n_envs=1, env_spacing=(3.0, 3.0), n_envs_per_row=1)
        self._ik_entity_n_qs = int(self._ik_entity.n_qs)

    def _solve_entity_ik_impl(
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
        if not isinstance(entity, RealRobotEntity):
            raise TypeError(
                f"Expected RealRobotEntity for solve_entity_ik(), got {type(entity).__name__}."
            )
        if self._ik_entity is None or self._ik_entity_n_qs is None:
            raise RuntimeError("Genesis IK backend is not initialized.")
        target_pos = torch.as_tensor(target_pos, dtype=torch.float32, device="cpu")
        target_quat = torch.as_tensor(target_quat, dtype=torch.float32, device="cpu")
        init_qpos = torch.as_tensor(init_qpos, dtype=torch.float32, device="cpu")
        if tuple(target_pos.shape) != (1, 3):
            raise ValueError(
                f"Expected target_pos shape (1, 3), got {tuple(target_pos.shape)}."
            )
        if tuple(target_quat.shape) != (1, 4):
            raise ValueError(
                f"Expected target_quat shape (1, 4), got {tuple(target_quat.shape)}."
            )
        if init_qpos.ndim != 2:
            raise ValueError(
                f"Expected init_qpos shape (1, num_dofs), got {tuple(init_qpos.shape)}."
            )
        if init_qpos.shape[0] != 1:
            raise ValueError(
                f"PhysicsReal only supports a single real environment, got batch size {init_qpos.shape[0]}."
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
            init_qpos_dofs_idx_local >= self._ik_entity_n_qs
        ):
            raise ValueError(
                "IK init_qpos_dofs_idx_local must contain entity-local DOF indices; "
                f"received indices {init_qpos_dofs_idx_local.tolist()} for entity.n_qs={self._ik_entity_n_qs}."
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
        if np.any(dofs_idx_local < 0) or np.any(dofs_idx_local >= self._ik_entity_n_qs):
            raise ValueError(
                "IK dofs_idx_local must contain entity-local DOF indices; "
                f"received indices {dofs_idx_local.tolist()} for entity.n_qs={self._ik_entity_n_qs}."
            )

        init_qpos_entity_order = torch.empty(
            (1, self._ik_entity_n_qs),
            dtype=init_qpos.dtype,
            device=init_qpos.device,
        )
        init_qpos_entity_order[:, init_qpos_dofs_idx_local.tolist()] = init_qpos
        ik_link = self._ik_entity.get_link(link_name)
        ik_qpos = self._ik_entity.inverse_kinematics(
            link=ik_link,
            pos=target_pos,
            quat=target_quat,
            dofs_idx_local=dofs_idx_local.tolist(),
            max_samples=10,
            max_solver_iters=10,
            init_qpos=init_qpos_entity_order,
        )
        if tuple(ik_qpos.shape) != (1, self._ik_entity_n_qs):
            raise ValueError(
                "IK solver returned unexpected qpos shape: "
                f"expected {(1, self._ik_entity_n_qs)}, got {tuple(ik_qpos.shape)}."
            )
        return ik_qpos[:, init_qpos_dofs_idx_local.tolist()]
