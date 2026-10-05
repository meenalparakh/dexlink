from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from urdfpy import URDF

from dexlink.utils.retargeting.math_utils import matrix_from_quat
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

PENETRATION_BPS_DIR_MAGNITUDE = 1e-4


def compute_mesh_transforms(
    pos, quat, arti, urdf_inst, joint_names, world2cam=np.eye(4)
):

    b = pos.shape[0]
    assert quat.shape == (b, 4), f"Expected quat shape {(b, 4)}, got {quat.shape}"
    assert arti.shape == (
        b,
        len(joint_names),
    ), f"Expected arti shape {(b, len(joint_names))}, got {arti.shape}"

    mat = matrix_from_quat(quat)

    # construct base pose: (b, 4, 4), with quat being (w, x, y, z)
    base_pose = torch.zeros((b, 4, 4), dtype=torch.float32).to(pos.device)
    base_pose[:, 3, 3] = 1.0  # set homogeneous coordinate
    base_pose[:, :3, :3] = mat
    base_pose[:, :3, 3] = pos

    # apply camera pose
    # c2w = np.linalg.inv(world2cam)
    c2w = world2cam
    c2w_torch = torch.from_numpy(c2w).to(pos.device).float()[None, ...]
    base_pose = torch.bmm(c2w_torch.expand(b, -1, -1), base_pose)

    joint_dict = {
        jname: arti[:, i].detach().cpu().numpy() for i, jname in enumerate(joint_names)
    }

    # urdfpy batch FK fails for rigid objects (no movable joints) because it cannot
    # infer the batch size from an empty cfg dict. Use static FK and tile instead.
    if len(joint_names) == 0:
        fk_static = urdf_inst.collision_trimesh_fk()
        fk = {
            mesh: np.tile(mesh_pose[None, ...], (b, 1, 1))
            for mesh, mesh_pose in fk_static.items()
        }
    else:
        fk = urdf_inst.collision_trimesh_fk_batch(cfgs=joint_dict)

    pose_results = OrderedDict()
    # update the account for pos and quat.
    for mesh, mesh_pose in fk.items():
        transform = torch.bmm(
            base_pose, torch.from_numpy(mesh_pose).to(base_pose.device).float()
        )
        pose_results[mesh] = transform

    return pose_results


def construct_pytorch_mesh(
    meshes: List[trimesh.Trimesh] | None = None,
    device="cuda",
    urdf_inst: URDF | None = None,
):
    if meshes is None:
        if urdf_inst is None:
            raise ValueError("Either meshes or urdf_inst must be provided")
        fk = urdf_inst.collision_trimesh_fk()
        meshes = []
        for mesh, _ in fk.items():
            meshes.append(mesh)

    verts_list = []
    faces_list = []
    pytorch_meshes = []
    for mesh in meshes:

        mesh = mesh.copy()
        mesh.update_faces(mesh.unique_faces())
        mask = mesh.nondegenerate_faces()
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()
        mesh.remove_infinite_values()

        _ = mesh.vertex_normals

        verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
        verts_list.append(verts)
        faces_list.append(faces)
        pytorch_meshes.append(
            Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0))
        )

    return pytorch_meshes


# def sample_kpts_with_normals(
#     meshes: List[Meshes], num_additional_pts: int = 100, max_pts: int = -1
# ):
#     pcds = []
#     normals = []
#     for mesh in meshes:
#         base_verts = mesh.verts_packed()
#         assert base_verts is not None, "Mesh has no vertices"
#         n = base_verts.shape[0]
#         base_normals = mesh.verts_normals_packed()
#         assert base_normals is not None, "Mesh has no vertex normals"

#         if 0 < max_pts < n:
#             # randomly sample max_pts from n
#             indices = torch.randperm(n)[:max_pts]
#             base_verts = base_verts[indices]
#             base_normals = base_normals[indices]
#         if max_pts > n:
#             num_additional_pts = max_pts - n
#         elif max_pts == -1:
#             num_additional_pts = num_additional_pts

#         if num_additional_pts > 0:
#             sampled_verts = sample_points_from_meshes(
#                 mesh, num_samples=num_additional_pts, return_normals=True
#             )
#             base_verts = torch.cat([base_verts, sampled_verts[0][0]], dim=0)
#             base_normals = torch.cat([base_normals, sampled_verts[1][0]], dim=0)

#         pcds.append(base_verts)
#         normals.append(base_normals)

#     return pcds, normals


# def sample_kpts(meshes: List[Meshes], num_additional_pts: int = 100, max_pts: int = -1):
#     # have all the vertices, as well as num_additional_pts sampled uniformly
#     pcds = []
#     for mesh in meshes:
#         base_verts = mesh.verts_packed()
#         assert base_verts is not None, "Mesh has no vertices"
#         n = base_verts.shape[0]

#         if 0 < max_pts < n:
#             # randomly sample max_pts from n
#             indices = torch.randperm(n)[:max_pts]
#             base_verts = base_verts[indices]
#         if max_pts > n:
#             num_additional_pts = max_pts - n
#         elif max_pts == -1:
#             num_additional_pts = num_additional_pts

#         if num_additional_pts > 0:
#             sampled_verts = sample_points_from_meshes(
#                 mesh, num_samples=num_additional_pts, return_normals=False
#             )
#             base_verts = torch.cat([base_verts, sampled_verts[0]], dim=0)

#         pcds.append(base_verts)

#     return pcds


# ////////////// do not use vertices for kpts. use face based sampling


def sample_kpts_with_normals(
    meshes: List[Meshes], num_additional_pts: int = 100, max_pts: int = -1
):
    pcds = []
    normals = []
    for mesh in meshes:
        if max_pts > 0:
            num_additional_pts = max_pts

        sampled_verts = sample_points_from_meshes(
            mesh, num_samples=num_additional_pts, return_normals=True
        )
        verts = sampled_verts[0][0]
        norms = sampled_verts[1][0]

        pcds.append(verts)
        normals.append(norms)

    return pcds, normals


def sample_kpts(meshes: List[Meshes], num_additional_pts: int = 100, max_pts: int = -1):
    # have all the vertices, as well as num_additional_pts sampled uniformly
    pcds = []
    for mesh in meshes:

        if max_pts > 0:
            num_additional_pts = max_pts

        sampled_verts = sample_points_from_meshes(
            mesh, num_samples=num_additional_pts, return_normals=False
        )
        verts = sampled_verts[0]

        pcds.append(verts)

    return pcds


def compute_kpts_traj(
    local_frame_kpts_list,
    pos,
    quat,
    arti,
    obj_inst: URDF,
    jnames,
    local_frame_normals_list=None,
    num_pts_sample: int = -1,
    world2cam=np.eye(4),
):
    """
    pos: (N, 3) tensor of object positions
    quat: (N, 4) tensor of object orientations
    arti: (N, J) tensor of object articulation parameters
    obj_inst: URDF instance of the object
    jnames: list of joint names
    """
    if num_pts_sample == 0 or num_pts_sample < -1:
        raise ValueError(
            f"Expected num_pts_sample to be -1 or a positive integer, got {num_pts_sample}."
        )

    kpts_list = local_frame_kpts_list
    normals_list = local_frame_normals_list
    total_pts = sum(kpts.shape[0] for kpts in kpts_list)
    if 0 < num_pts_sample < total_pts:
        flat_kpts = torch.cat(kpts_list, dim=0)
        link_ids = torch.cat(
            [
                torch.full((kpts.shape[0],), idx, device=kpts.device, dtype=torch.long)
                for idx, kpts in enumerate(kpts_list)
            ],
            dim=0,
        )
        flat_normals = None
        if normals_list is not None:
            flat_normals = torch.cat(normals_list, dim=0)

        sample_indices = torch.randperm(total_pts, device=flat_kpts.device)[
            :num_pts_sample
        ]
        sampled_kpts = flat_kpts[sample_indices]
        sampled_link_ids = link_ids[sample_indices]
        sampled_normals = None if flat_normals is None else flat_normals[sample_indices]

        kpts_list = []
        normals_list = [] if sampled_normals is not None else None
        n_links = len(local_frame_kpts_list)
        for link_idx in range(n_links):
            link_mask = sampled_link_ids == link_idx
            kpts_list.append(sampled_kpts[link_mask])
            if sampled_normals is not None:
                assert normals_list is not None
                normals_list.append(sampled_normals[link_mask])

    traj_len = pos.shape[0]
    pose_results = compute_mesh_transforms(
        pos,
        quat,
        arti,
        urdf_inst=obj_inst,
        joint_names=[j.name for j in obj_inst.joints if j.joint_type != "fixed"],
        world2cam=world2cam,
    )

    pcd_time_series = []
    normals_time_series = []

    for idx, (kpt, poses) in enumerate(zip(kpts_list, pose_results.values())):
        # apply poses to kpts
        pcd = kpt.clone()
        n_pts = pcd.shape[0]
        assert pcd.shape[1:] == (3,), f"Expected shape (3,), got {pcd.shape[1:]}"
        pcd_homo = torch.cat(
            [pcd, torch.ones(n_pts, 1, device=pcd.device)], dim=1
        ).transpose(0, 1)
        assert pcd_homo.shape == (4, n_pts)
        assert poses.shape == (traj_len, 4, 4)
        pcd_transformed = poses @ pcd_homo.unsqueeze(0)
        assert pcd_transformed.shape == (traj_len, 4, n_pts)
        pcd_transformed = pcd_transformed.transpose(1, 2)[..., :3]
        assert pcd_transformed.shape == (traj_len, n_pts, 3)
        pcd_time_series.append(pcd_transformed)

        if normals_list is not None:
            normals = normals_list[idx]
            assert normals.shape[1:] == (
                3,
            ), f"Expected shape (3,), got {normals.shape[1:]}"
            normals_t = normals.transpose(0, 1)
            normals_transformed = poses[:, :3, :3] @ normals_t.unsqueeze(0)
            normals_transformed = normals_transformed.transpose(1, 2)
            normals_time_series.append(normals_transformed)

    pcd_time_series_tensor = torch.cat(pcd_time_series, dim=1)
    if normals_list is not None:
        normals_time_series_tensor = torch.cat(normals_time_series, dim=1)
        pcd_time_series_tensor = torch.cat(
            [pcd_time_series_tensor, normals_time_series_tensor], dim=-1
        )

    return pcd_time_series_tensor


def compute_bps(pcd, bps):
    # pcd: (N, n_pts, 6) -> x, y, z, nx, ny, nz
    # bps: (N, n_bps, 3)
    # returns: (N, n_bps, 6) -> direction to closest point + normal at that point

    if pcd.ndim != 3:
        raise ValueError(f"`pcd` must have shape (N, n_pts, 6), got {pcd.shape}.")
    if bps.ndim != 3:
        raise ValueError(f"`bps` must have shape (N, n_bps, 3), got {bps.shape}.")
    if pcd.shape[-1] != 6:
        raise ValueError(f"`pcd` last dimension must be 6, got shape {pcd.shape}.")
    if bps.shape[-1] != 3:
        raise ValueError(f"`bps` last dimension must be 3, got shape {bps.shape}.")
    if pcd.shape[0] != bps.shape[0]:
        raise ValueError(
            f"`pcd` and `bps` must have matching batch size, got {pcd.shape[0]} "
            f"and {bps.shape[0]}."
        )

    pcd_pts = pcd[..., :3]
    pcd_normals = pcd[..., 3:]

    # pairwise vector from each basis point to every point on the surface
    diff = pcd_pts.unsqueeze(1) - bps.unsqueeze(2)  # (N, n_bps, n_pts, 3)
    dist_sq = (diff**2).sum(dim=-1)  # (N, n_bps, n_pts)

    # find closest point per basis point
    closest_idx = torch.argmin(dist_sq, dim=-1)  # (N, n_bps)

    # construct mask
    selection = F.one_hot(closest_idx, num_classes=diff.shape[2])
    selection = selection.to(diff.dtype).unsqueeze(-1)  # (N, n_bps, n_pts, 1)

    closest_vec = (diff * selection).sum(dim=2)  # (N, n_bps, 3)
    closest_normals = (pcd_normals.unsqueeze(1).expand_as(diff) * selection).sum(
        dim=2
    )  # (N, n_bps, 3)

    closest_normals_unit = closest_normals / closest_normals.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-9)
    penetration_mask = (closest_vec * closest_normals_unit).sum(dim=-1) > 0.0
    closest_vec = torch.where(
        penetration_mask.unsqueeze(-1),
        -PENETRATION_BPS_DIR_MAGNITUDE * closest_normals_unit,
        closest_vec,
    )

    return torch.cat([closest_vec, closest_normals], dim=-1)


def compute_bps_efficient(pcd, bps, chunk_size: int = 1024):
    """
    Memory-efficient BPS computation to avoid large (N, n_bps, n_pts) allocations.
    Processes basis points in chunks and accumulates results.
    """
    if pcd.ndim != 3:
        raise ValueError(f"`pcd` must have shape (N, n_pts, 6), got {pcd.shape}.")
    if bps.ndim != 3:
        raise ValueError(f"`bps` must have shape (N, n_bps, 3), got {bps.shape}.")
    if pcd.shape[-1] != 6:
        raise ValueError(f"`pcd` last dimension must be 6, got shape {pcd.shape}.")
    if bps.shape[-1] != 3:
        raise ValueError(f"`bps` last dimension must be 3, got shape {bps.shape}.")
    if pcd.shape[0] != bps.shape[0]:
        raise ValueError(
            f"`pcd` and `bps` must have matching batch size, got {pcd.shape[0]} "
            f"and {bps.shape[0]}."
        )

    pcd_pts = pcd[..., :3]
    pcd_normals = pcd[..., 3:]

    N = pcd_pts.shape[0]
    n_bps = bps.shape[1]

    out_vec = torch.empty((N, n_bps, 3), device=pcd.device, dtype=pcd.dtype)
    out_normals = torch.empty((N, n_bps, 3), device=pcd.device, dtype=pcd.dtype)

    for start in range(0, n_bps, chunk_size):
        end = min(start + chunk_size, n_bps)
        bps_chunk = bps[:, start:end]  # (N, chunk, 3)

        diff = pcd_pts.unsqueeze(1) - bps_chunk.unsqueeze(2)  # (N, chunk, n_pts, 3)
        dist_sq = (diff**2).sum(dim=-1)  # (N, chunk, n_pts)
        closest_idx = torch.argmin(dist_sq, dim=-1)  # (N, chunk)

        # gather closest vectors and normals
        idx_expanded = closest_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
        closest_vec = torch.gather(diff, 2, idx_expanded).squeeze(2)  # (N, chunk, 3)

        normals_expanded = pcd_normals.unsqueeze(1).expand(-1, end - start, -1, -1)
        closest_normals = torch.gather(normals_expanded, 2, idx_expanded).squeeze(
            2
        )  # (N, chunk, 3)

        closest_normals_unit = closest_normals / closest_normals.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
        penetration_mask = (closest_vec * closest_normals_unit).sum(dim=-1) > 0.0
        closest_vec = torch.where(
            penetration_mask.unsqueeze(-1),
            -PENETRATION_BPS_DIR_MAGNITUDE * closest_normals_unit,
            closest_vec,
        )

        out_vec[:, start:end] = closest_vec
        out_normals[:, start:end] = closest_normals

    return torch.cat([out_vec, out_normals], dim=-1)


def sample_unit_ball(n, device="cpu"):
    # direction
    x = torch.randn(n, 3, device=device)
    x = x / x.norm(dim=1, keepdim=True)

    # radius (volume-uniform)
    r = torch.rand(n, 1, device=device) ** (1 / 3)

    return r * x


def pcd_distance(pcd1, pcd2, metric="chamfer", bps_n_pts: int = 128):
    if metric == "chamfer":
        return chamfer_distance(pcd1, pcd2, batch_reduction=None)[0]

    if metric == "bps":
        # sample bps points in a shphere and compare representation
        # separate out the normalization

        xyz1 = pcd1[..., :3]
        xyz2 = pcd2[..., :3]

        mean1 = xyz1.mean(dim=1, keepdim=True)
        mean2 = xyz2.mean(dim=1, keepdim=True)

        xyz1_centered = xyz1 - mean1
        xyz2_centered = xyz2 - mean2

        xyz1_dist = torch.norm(xyz1_centered, dim=-1, keepdim=True)
        xyz2_dist = torch.norm(xyz2_centered, dim=-1, keepdim=True)

        # normalize pcd to unit sphere
        scale1 = torch.max(xyz1_dist, dim=1, keepdim=True)[0]
        scale2 = torch.max(xyz2_dist, dim=1, keepdim=True)[0]

        xyz1_normalized = xyz1_centered / scale1
        xyz2_normalized = xyz2_centered / scale2

        pcd1_normalized = torch.cat([xyz1_normalized, pcd1[..., 3:]], dim=-1)
        pcd2_normalized = torch.cat([xyz2_normalized, pcd2[..., 3:]], dim=-1)

        # sample bps on unit sphere
        bps = sample_unit_ball(bps_n_pts, device=pcd1.device).unsqueeze(
            0
        )  # (1, bps_n_pts, 3)
        bps = bps.expand(pcd1.shape[0], -1, -1)  # (N, bps_n_pts, 3)
        bps_repr1 = compute_bps_efficient(
            pcd1_normalized, bps, chunk_size=32
        )  # (N, bps_n_pts, 6)
        bps_repr2 = compute_bps_efficient(pcd2_normalized, bps, chunk_size=32)

        mean_loss = F.mse_loss(mean1, mean2, reduction="none").mean(dim=(1, 2))  # (N,)
        bps_loss = F.mse_loss(bps_repr1, bps_repr2, reduction="none").mean(
            dim=(1, 2)
        )  # (N,)
        total_loss = 0.5 * mean_loss + 0.5 * bps_loss
        return total_loss


# def normalized_chamfer_distance(pcd1, pcd2, *args, **kwargs):
#     # before calling pytorch3d chamfer distance, normalize the point clouds
#     # so that the pcds have similar scale, when centered at origin
#     # **bring pcd1 to same scale as pcd2**

#     mean1 = pcd1.mean(dim=1, keepdim=True)
#     mean2 = pcd2.mean(dim=1, keepdim=True)


#     # the scale: (s, s, s) (same scale for x, y, z)
