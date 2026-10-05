from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

PENETRATION_BPS_DIR_MAGNITUDE = 1e-2


def sample_pts_from_obj(obj_path, obj_pose, n_pts, return_nocs=True):
    # use trimesh to sample n_pts from the obj
    # returns (N, 3) pts and (N, 3) normal vectors
    if int(n_pts) <= 0:
        raise ValueError(f"`n_pts` must be > 0, got {n_pts}.")

    mesh_or_scene = trimesh.load(obj_path, force="scene")
    if isinstance(mesh_or_scene, trimesh.Scene):
        if len(mesh_or_scene.geometry) == 0:
            raise ValueError(f"No geometry found in mesh file: {obj_path}")
        mesh = mesh_or_scene.dump(concatenate=True)
    else:
        mesh = mesh_or_scene

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh, got {type(mesh).__name__}.")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces and cannot be surface-sampled: {obj_path}")

    # Apply the pose transformation to the mesh
    if obj_pose is not None:
        if obj_pose.shape != (4, 4):
            raise ValueError(
                f"`obj_pose` must be a 4x4 matrix, got shape {obj_pose.shape}."
            )
        mesh.apply_transform(obj_pose)

    pts, face_idx = trimesh.sample.sample_surface(mesh, int(n_pts))
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[
        np.asarray(face_idx, dtype=np.int64)
    ]

    if return_nocs:
        assert obj_pose is not None, (
            "To compute NOCs, `obj_pose` must be provided."
            + "otherwise the normalization may be inconsistent."
        )
        nocs = compute_normalized_coordinates(pts)
        return np.concatenate([pts, normals, nocs], axis=1)  # (N, 9)
    else:
        return np.concatenate([pts, normals], axis=1)  # (N, 6)


def get_X(pos, quat):
    # quat is in (, 4) wxyz
    return_batched = len(pos.shape) > 1

    pos = pos.reshape((-1, 3))
    quat = quat.reshape((-1, 4))

    quat_xyzw = np.roll(quat, shift=1, axis=1)

    n_batches = pos.shape[0]

    X = np.zeros((n_batches, 4, 4))
    X[:, 3, 3] = 1.0

    X[:, :3, :3] = R.from_quat(quat_xyzw).as_matrix()
    X[:, :3, 3] = pos

    if return_batched:
        return X
    else:
        return X[0]


def compute_hand_cfg_vecs(jpos_dict):
    pairs_to_optimize = [
        ("wrist", "thumb_tip", "normalize"),
        ("wrist", "index_tip", "normalize"),
        ("wrist", "middle_tip", "normalize"),
        ("wrist", "ring_tip", "normalize"),
        ("wrist", "little_tip", "normalize"),
        ("thumb_base", "thumb_tip", "normalize"),
        ("index_base", "index_tip", "normalize"),
        ("middle_base", "middle_tip", "normalize"),
        ("ring_base", "ring_tip", "normalize"),
        ("little_base", "little_tip", "normalize"),
        ("thumb_tip", "index_tip", "unnormalize"),
        ("thumb_tip", "middle_tip", "unnormalize"),
        ("thumb_tip", "ring_tip", "unnormalize"),
        ("thumb_tip", "little_tip", "unnormalize"),
    ]
    # cross product direction should match
    triplets_to_optimize = [
        ("wrist", "thumb_base", "little_base", "normalize"),
    ]

    vectors = []

    for joint1, joint2, mode in pairs_to_optimize:
        lpos1 = jpos_dict[joint1]
        lpos2 = jpos_dict[joint2]

        lvec = lpos2 - lpos1

        if mode == "normalize":
            vec_norm = np.sqrt(np.square(lvec).sum()) + 1e-8
            lvec = lvec / vec_norm

        vectors.append(lvec)

    triplet_vecs = []
    for joint1, joint2, joint3, mode in triplets_to_optimize:
        lpos1 = jpos_dict[joint1]
        lpos2 = jpos_dict[joint2]
        lpos3 = jpos_dict[joint3]

        vec1 = lpos2 - lpos1
        vec2 = lpos3 - lpos1

        cross_vec = np.cross(vec1, vec2)

        if mode == "normalize":
            cross_vec = cross_vec / (np.linalg.norm(cross_vec) + 1e-8)

        triplet_vecs.append(cross_vec)

    # ref vecs is (num_pairs, num_frames, 3)

    ref_vectors = np.stack(vectors, axis=0)
    triplet_ref_vectors = np.stack(triplet_vecs, axis=0)

    return {
        "pairs": (pairs_to_optimize, ref_vectors),
        "triplets": (triplets_to_optimize, triplet_ref_vectors),
    }


def compute_object_bps_differentiable(
    pcd,
    base_points,
    beta=50.0,
    normalize=True,
    include_other_features=True,
    apply_penetration_correction=False,
):
    """
    Compute BPS representation for the object at a given pose.

    Args:
        pcd: (N, 6 + F) point cloud with normals and optional extra features.
            If batched version (say time series), (T, N, 6 + F)
        base_points: (M, 3) fixed base points. If batched version, (B, M, 3)
        beta: temperature parameter for soft-argmin
        normalize: whether to normalize the direction and normal vectors
        include_other_features: whether to append soft-nearest expected extra
            point features after the displacement and normal channels
        apply_penetration_correction: whether to replace positive
            direction-normal dot products with a small negative-normal vector
    Returns:
        bps: (M, 6 + F) BPS representation of the whole object
            (direction + normal + optional extra features). If batched
            version, (T, M, 6 + F)
    """

    # bps representation is computed on the pcd, which is already oriented properly
    pcd = np.asarray(pcd)
    base_points = np.asarray(base_points)

    if pcd.ndim == 2:
        pcd = pcd[None, :, :]  # (1, N, 6)
    if base_points.ndim == 2:
        base_points = base_points[None, :, :]  # (1, M, 3)
    if pcd.ndim != 3:
        raise ValueError(f"`pcd` must have shape (T, N, 6 + F), got {pcd.shape}.")
    if base_points.ndim != 3:
        raise ValueError(
            f"`base_points` must have shape (T, M, 3), got {base_points.shape}."
        )
    if pcd.shape[-1] < 6:
        raise ValueError(
            f"`pcd` must contain xyz and normal channels, got shape {pcd.shape}."
        )
    if base_points.shape[-1] != 3:
        raise ValueError(
            f"`base_points` last dimension must be 3, got shape {base_points.shape}."
        )
    if pcd.shape[0] != base_points.shape[0]:
        raise ValueError(
            "`pcd` and `base_points` must have matching batch dimensions, got "
            f"{pcd.shape[0]} and {base_points.shape[0]}."
        )

    bps_list = []
    eps = 1e-9
    for idx, frame_pcd in enumerate(pcd):
        pts = frame_pcd[:, :3]
        normals = frame_pcd[:, 3:6]
        other_features = frame_pcd[:, 6:]
        frame_base_points = base_points[idx]

        if normalize:
            # Center and scale to fit within unit sphere.
            center = np.mean(pts, axis=0, keepdims=True)
            pts_centered = pts - center
            scale = np.max(np.sqrt(np.sum(pts_centered * pts_centered, axis=1))) + eps
            pts = pts_centered / scale

        # Pairwise squared distances: (M, N)
        diff = frame_base_points[:, None, :] - pts[None, :, :]
        sq_dists = np.sum(diff**2, axis=-1)

        # Softmin weights (numerically stable)
        scaled = -beta * sq_dists
        scaled -= np.max(scaled, axis=1, keepdims=True)
        weights = np.exp(scaled)
        weights /= np.sum(weights, axis=1, keepdims=True) + eps

        # Expected closest point, normal, and any optional trailing features.
        closest_pts = weights @ pts
        closest_normals = weights @ normals
        closest_other_features = None
        if include_other_features and other_features.shape[1] > 0:
            closest_other_features = weights @ other_features

        # displacement from base point to closest point
        dir_vec = closest_pts - frame_base_points
        # dir_norm = np.sqrt(np.sum(dir_vec * dir_vec, axis=1, keepdims=True)) + eps
        # dir_unit = dir_vec / dir_norm

        # Normalize normals
        nrm = (
            np.sqrt(np.sum(closest_normals * closest_normals, axis=1, keepdims=True))
            + eps
        )
        normals_unit = closest_normals / nrm

        if apply_penetration_correction:
            penetration_mask = np.sum(dir_vec * normals_unit, axis=1) > 0.0
            dir_vec[penetration_mask] = (
                -PENETRATION_BPS_DIR_MAGNITUDE * normals_unit[penetration_mask]
            )

        bps_parts = [dir_vec, normals_unit]
        if closest_other_features is not None:
            bps_parts.append(closest_other_features)
        bps = np.concatenate(bps_parts, axis=1)
        bps_list.append(bps)

    return np.stack(bps_list, axis=0) if len(bps_list) > 1 else bps_list[0]


def compute_bps_distance_sigmoid_weights(
    bps_distances, distance_offset=0.01, denominator_scale=0.001
):
    """Compute adaptive weights from BPS distance magnitudes.

    Formula:
        w = 1 / (1 + exp(distance_offset - d) / denominator_scale)
    where d is a fingertip BPS distance magnitude.
    """
    d = np.asarray(bps_distances)
    return 1.0 / (1.0 + np.exp(distance_offset - d) / denominator_scale)


def compute_normalized_coordinates(pcd):
    # fit the pcd in a unit sphere (centered at origin, and max distance to any pt is 1)
    center = np.mean(pcd, axis=0, keepdims=True)
    pcd_centered = pcd - center
    scale = np.max(np.sqrt(np.sum(pcd_centered * pcd_centered, axis=1))) + 1e-8
    pcd_normalized = pcd_centered / scale
    return pcd_normalized


class CostTraceLogger:
    def __init__(self, *, enabled: bool, plot: bool, every: int):
        self.enabled = bool(enabled)
        self.plot = bool(plot)
        self.every = max(1, int(every))
        self.eval_count = 0
        self.trace = {
            "eval": [],
            "total_cost": [],
            "hand_cfg_cost": [],
            "hand_object_bps_cost": [],
            "bps_beta": [],
        }

    def increment(self) -> int:
        self.eval_count += 1
        return self.eval_count

    def clear(self):
        self.eval_count = 0
        for key in self.trace.keys():
            self.trace[key] = []

    def record(
        self,
        *,
        eval_idx: int,
        total_cost,
        hand_cfg_cost,
        hand_object_bps_cost,
        bps_beta,
        **extra_terms,
    ) -> None:
        if not self.enabled:
            return
        if eval_idx != 1 and eval_idx % self.every != 0:
            return
        values = {
            "total_cost": total_cost,
            "hand_cfg_cost": hand_cfg_cost,
            "hand_object_bps_cost": hand_object_bps_cost,
            "bps_beta": bps_beta,
            **extra_terms,
        }
        scalarized_values = {}
        for key, value in values.items():
            scalar = self._scalarize(value)
            if scalar is None:
                return
            scalarized_values[key] = scalar

        self.trace["eval"].append(int(eval_idx))
        for key, scalar in scalarized_values.items():
            if key not in self.trace:
                self.trace[key] = [np.nan] * (len(self.trace["eval"]) - 1)
            self.trace[key].append(scalar)

        for key, values_list in self.trace.items():
            if key == "eval" or key in scalarized_values:
                continue
            values_list.append(np.nan)

    @staticmethod
    def _scalarize(value) -> float | None:
        try:
            return float(value)
        except Exception:
            pass
        if hasattr(value, "value"):
            try:
                return float(value.value())
            except Exception:
                pass
        try:
            arr = np.asarray(value, dtype=object).reshape(-1)
            if arr.size > 0:
                v0 = arr[0]
                if hasattr(v0, "value"):
                    return float(v0.value())
                return float(v0)
        except Exception:
            pass
        return None

    def print_counts(self) -> None:
        print(f"[cost_eval] total_callback_count={self.eval_count}")

    def dump(self, run_dir: Path) -> tuple[Path | None, Path | None]:
        if not self.enabled:
            return None, None

        run_dir.mkdir(parents=True, exist_ok=True)
        npz_path = run_dir / "optimizer_final_cost_trace.npz"
        np.savez(
            npz_path,
            **{
                key: np.asarray(values, dtype=np.float64 if key != "eval" else np.int64)
                for key, values in self.trace.items()
            },
        )
        print(f"[cost_trace] saved: {npz_path}")

        plot_path = None
        if self.plot:
            try:
                import matplotlib

                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt

                plot_keys = [key for key in self.trace.keys() if key != "eval"]
                n_plots = len(plot_keys)
                n_cols = 2
                n_rows = max(1, int(np.ceil(n_plots / n_cols)))
                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(12, 4 * n_rows),
                    constrained_layout=True,
                )
                axes = np.atleast_1d(axes).ravel()
                xs = np.asarray(self.trace["eval"])
                for ax, key in zip(axes, plot_keys):
                    ys = np.asarray(self.trace[key])
                    ax.set_title(key.replace("_", " ").title())
                    ax.set_xlabel("Evaluation Call")
                    if xs.size > 0:
                        ax.plot(xs, ys, linewidth=1.0)
                        if key != "bps_beta":
                            valid = ys[np.isfinite(ys)]
                            if valid.size > 0 and np.all(valid > 0):
                                ax.set_yscale("log")
                            else:
                                ax.set_yscale("symlog", linthresh=1e-8)
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No samples",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                for ax in axes[n_plots:]:
                    ax.axis("off")
                plot_path = run_dir / "optimizer_final_cost_trace.png"
                fig.savefig(plot_path, dpi=160)
                plt.close(fig)
                print(f"[cost_trace] plot saved: {plot_path}")
            except Exception as exc:
                print(f"[cost_trace] plot skipped: {exc}")
        return npz_path, plot_path
