"""Static real-backend entity definitions.

RealEntity stores the URDF-derived structure and fixed placement used by the real
backend: joint/link mappings, raw URDF limits, and the precomputed world-frame base pose.
It intentionally does not own live robot I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF


@dataclass
class RealEntity:
    """URDF-backed static entity description for the real backend."""

    name: str | None
    urdf_path: str
    urdf_inst: object
    fixed: bool
    collision: bool
    convexify: bool
    recompute_inertia: bool
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    world_base_pose: np.ndarray
    links_to_keep: list[str]
    rho: float | None = None
    friction: float = 1.2
    gravity_compensation: float | None = None
    base_link_name: str | None = None
    base_link_idx: int = 0
    n_qs: int = 0
    joint_dofs_idx_local: dict[str, list[int]] = field(default_factory=dict)
    joint_parent_link_name: dict[str, str | None] = field(default_factory=dict)
    joint_child_link_name: dict[str, str | None] = field(default_factory=dict)
    link_idx_local: dict[str, int] = field(default_factory=dict)
    link_idx: dict[str, int] = field(default_factory=dict)
    dof_limit_lower: list[float] = field(default_factory=list)
    dof_limit_upper: list[float] = field(default_factory=list)

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        *,
        name: str | None,
        fixed: bool,
        collision: bool,
        convexify: bool,
        recompute_inertia: bool,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float],
        links_to_keep: list[str] | None,
        rho: float | None,
        friction: float,
        gravity_compensation: float | None,
    ) -> "RealEntity":
        urdf = URDF.load(str(Path(urdf_path)))
        link_idx_local = {
            link.name: idx_local for idx_local, link in enumerate(urdf.links)
        }
        link_idx = dict(link_idx_local)
        non_fixed_joints = [
            joint for joint in urdf.joints if joint.joint_type != "fixed"
        ]
        joint_dofs_idx_local = {
            joint.name: [dof_idx] for dof_idx, joint in enumerate(non_fixed_joints)
        }
        joint_parent_link_name = {
            joint.name: joint.parent for joint in non_fixed_joints
        }
        joint_child_link_name = {joint.name: joint.child for joint in non_fixed_joints}
        dof_limit_lower: list[float] = []
        dof_limit_upper: list[float] = []
        for joint in non_fixed_joints:
            if joint.limit is None:
                raise ValueError(
                    f"Joint {joint.name!r} in {urdf_path!r} has no URDF limit information."
                )
            if joint.limit.lower is None or joint.limit.upper is None:
                raise ValueError(
                    f"Joint {joint.name!r} in {urdf_path!r} is missing lower/upper limits."
                )
            lower = float(joint.limit.lower)
            upper = float(joint.limit.upper)
            dof_limit_lower.append(lower)
            dof_limit_upper.append(upper)
        base_link_name = None if urdf.base_link is None else urdf.base_link.name
        base_link_idx = 0 if base_link_name is None else link_idx_local[base_link_name]
        quat_wxyz = np.asarray(quat, dtype=np.float32)
        if quat_wxyz.shape != (4,):
            raise ValueError(
                f"Expected base quaternion shape (4,), got {quat_wxyz.shape}."
            )
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float32,
        )
        world_base_pose = np.eye(4, dtype=np.float32)
        world_base_pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
        world_base_pose[:3, 3] = np.asarray(pos, dtype=np.float32)
        entity = cls(
            name=name,
            urdf_path=str(Path(urdf_path)),
            urdf_inst=urdf,
            fixed=fixed,
            collision=collision,
            convexify=convexify,
            recompute_inertia=recompute_inertia,
            pos=pos,
            quat=quat,
            world_base_pose=world_base_pose,
            links_to_keep=[] if links_to_keep is None else list(links_to_keep),
            rho=rho,
            friction=friction,
            gravity_compensation=gravity_compensation,
            base_link_name=base_link_name,
            base_link_idx=base_link_idx,
            n_qs=len(joint_dofs_idx_local),
            joint_dofs_idx_local=joint_dofs_idx_local,
            joint_parent_link_name=joint_parent_link_name,
            joint_child_link_name=joint_child_link_name,
            link_idx_local=link_idx_local,
            link_idx=link_idx,
            dof_limit_lower=dof_limit_lower,
            dof_limit_upper=dof_limit_upper,
        )
        return entity

    def get_joint_dofs_idx_local(self, joint_name: str) -> list[int]:
        dofs_idx_local = self.joint_dofs_idx_local.get(joint_name)
        if dofs_idx_local is None:
            raise KeyError(
                f"Unknown joint {joint_name!r}. Available joints: {sorted(self.joint_dofs_idx_local)}."
            )
        return dofs_idx_local

    def get_link_idx_local(self, link_name: str) -> int:
        idx_local = self.link_idx_local.get(link_name)
        if idx_local is None:
            raise KeyError(
                f"Unknown link {link_name!r}. Available links: {sorted(self.link_idx_local)}."
            )
        return idx_local

    def get_link_idx(self, link_name: str) -> int:
        idx = self.link_idx.get(link_name)
        if idx is None:
            raise KeyError(
                f"Unknown link {link_name!r}. Available links: {sorted(self.link_idx)}."
            )
        return idx

    def get_dofs_limit(
        self, dofs_idx_local: list[int] | tuple[int, ...]
    ) -> tuple[list[float], list[float]]:
        lower = []
        upper = []
        for dof_idx in dofs_idx_local:
            if dof_idx < 0 or dof_idx >= self.n_qs:
                raise IndexError(
                    f"DOF index {dof_idx} is out of range for entity with n_qs={self.n_qs}."
                )
            lower.append(self.dof_limit_lower[dof_idx])
            upper.append(self.dof_limit_upper[dof_idx])
        return lower, upper

    @property
    def joint_names(self) -> list[str]:
        return list(self.joint_dofs_idx_local)

    @property
    def link_names(self) -> list[str]:
        return list(self.link_idx_local)


@dataclass
class AssetEntity:
    """Simple placeholder entity for non-robot scene assets in the real backend."""

    name: str | None
    kind: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float] | None = None
    fixed: bool = True
    collision: bool = False
    metadata: dict[str, object] = field(default_factory=dict)
