"""Exact OBB SAT narrow phase with AABB broad phase. No inter-link exclusions."""

import numpy as np
from kinematics import Model


def sat_overlap(
    Ta: np.ndarray, ha: np.ndarray, Tb: np.ndarray, hb: np.ndarray
) -> float:
    """Minimum interval overlap on separating axes; >0 means penetration."""
    A, B = Ta[:3, :3], Tb[:3, :3]
    axes = np.concatenate(
        (A.T, B.T, np.cross(A.T[:, None, :], B.T[None, :, :]).reshape(-1, 3))
    )
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-10] / lengths[lengths > 1e-10, None]
    r = np.abs(axes @ A) @ ha + np.abs(axes @ B) @ hb
    overlaps = r - np.abs(axes @ (Tb[:3, 3] - Ta[:3, 3]))
    return float(np.min(overlaps))


def collisions_at(model: Model, cfg: dict[str, float], tolerance: float = 1e-8):
    poses = model.box_poses(cfg, collision=True)
    centers = np.array([T[:3, 3] for _, T in poses])
    half = np.array([b.size / 2 for b, _ in poses])
    radii = np.array([np.abs(T[:3, :3]) @ h for (_, T), h in zip(poses, half)])
    broad = np.all(
        np.abs(centers[:, None, :] - centers[None, :, :])
        < radii[:, None, :] + radii[None, :, :] - tolerance,
        axis=2,
    )
    result = []
    for i, j in np.argwhere(np.triu(broad, 1)):
        bi, Ti = poses[i]
        bj, Tj = poses[j]
        if bi.link == bj.link:
            continue
        penetration = sat_overlap(Ti, half[i], Tj, half[j])
        if penetration > tolerance:
            result.append((bi.name, bj.name, penetration))
    return result
