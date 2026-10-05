from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import trimesh
from scipy.optimize import linprog
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

from dexlink.src.retargeting.data.hand_cfgs.franka_xhand import XHAND_CFGs as FRANKA_XHAND_CFGs
from dexlink.src.retargeting.data.hand_cfgs.inspire import INSPIRE_CFGs
from dexlink.src.retargeting.data.hand_cfgs.xhand import XHAND_CFGs
from dexlink.utils.retargeting.mesh_utils import compute_bps_efficient
from dexlink.utils.retargeting.path_utils import process_urdf
from pytorch3d.structures import Meshes

ROBOT_SPECS = {
    "inspire": {
        "left": (INSPIRE_CFGs["left"]["urdf_path"], INSPIRE_CFGs["left"]),
        "right": (
            INSPIRE_CFGs["right"]["urdf_path"],
            INSPIRE_CFGs["right"],
        ),
    },
    "xhand": {
        "left": (XHAND_CFGs["left"]["urdf_path"], XHAND_CFGs["left"]),
        "right": (
            XHAND_CFGs["right"]["urdf_path"],
            XHAND_CFGs["right"],
        ),
    },
    "franka_xhand": {
        "left": (
            FRANKA_XHAND_CFGs["left"]["urdf_path"],
            FRANKA_XHAND_CFGs["left"],
        ),
        "right": (
            FRANKA_XHAND_CFGs["right"]["urdf_path"],
            FRANKA_XHAND_CFGs["right"],
        ),
    },
}


def _load_robot_urdfs(robot_model: str) -> dict[str, URDF]:
    """Load each hand from a process-specific URDF copy on shared filesystems."""
    if robot_model not in ROBOT_SPECS:
        raise ValueError(f"Unsupported resolved robot_model={robot_model!r}.")
    return {
        side: URDF.load(
            str(
                process_urdf(
                    ROBOT_SPECS[robot_model][side][0],
                    append_jobname=True,
                )
            )
        )
        for side in ("left", "right")
    }


HAND_SKELETON_ANGLE_TRIPLETS = (
    ("wrist", "thumb_base", "thumb_proximal"),
    ("thumb_base", "thumb_proximal", "thumb_tip"),
    ("wrist", "index_base", "index_proximal"),
    ("index_base", "index_proximal", "index_tip"),
    ("wrist", "middle_base", "middle_proximal"),
    ("middle_base", "middle_proximal", "middle_tip"),
    ("wrist", "ring_base", "ring_proximal"),
    ("ring_base", "ring_proximal", "ring_tip"),
    ("wrist", "little_base", "little_proximal"),
    ("little_base", "little_proximal", "little_tip"),
)
HAND_SKELETON_EDGES = tuple(
    dict.fromkeys(
        edge for a, b, c in HAND_SKELETON_ANGLE_TRIPLETS for edge in ((a, b), (b, c))
    )
)


@dataclass(frozen=True)
class GraspEvalConfig:
    mesh_type: str = "collision"  # "collision" | "visual"
    robot_model: str = "auto"  # "auto" | "inspire" | "xhand" | "franka_xhand"
    penetration_surface_samples: int = 1200
    intersection_method: str = "grid"  # "grid" | "mc"
    intersection_mc_samples: int = 8000
    intersection_grid_max_points: int = 4096
    intersection_grid_min_cells_per_axis: int = 6
    chamfer_max_points: int = 6000
    hand_object_bps_surface_samples: int = 2048
    hand_object_bps_chunk_size: int = 128
    force_closure_thresholds_m: tuple[float, ...] = (0.005, 0.01)
    force_closure_hand_surface_samples: int = 3000
    rng_seed: int = 0


def _joint_dict_for_frame(joint_names: Iterable[str], joint_vals_frame: Any) -> dict:
    joint_names = list(joint_names)
    if joint_vals_frame is None or len(joint_names) == 0:
        return {}
    vals = np.asarray(joint_vals_frame, dtype=float).reshape(-1)
    if vals.shape[0] != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} joint values, got {vals.shape[0]}."
        )
    return {name: float(val) for name, val in zip(joint_names, vals.tolist())}


def _normalize_robot_model(robot_model: str | None) -> str:
    if robot_model is None:
        return "auto"
    aliases = {
        "auto": "auto",
        "inspire": "inspire",
        "inspire_hand": "inspire",
        "xhand": "xhand",
        "franka": "franka_xhand",
        "franka_xhand": "franka_xhand",
        "franka-xhand": "franka_xhand",
        "panda_xhand": "franka_xhand",
    }
    value = str(robot_model).lower()
    if value not in aliases:
        raise ValueError(
            f"Unsupported robot_model {robot_model!r}; expected one of {sorted(aliases)}."
        )
    return aliases[value]


def _resolve_robot_model(processor: Any, motion: dict, cfg: GraspEvalConfig) -> str:
    requested = _normalize_robot_model(cfg.robot_model)
    if requested != "auto":
        return requested

    candidates = [
        motion.get("robot_model", None),
        motion.get("traj_robot_model", None),
        getattr(processor, "robot_model", None),
        getattr(processor, "traj_hand_name", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, np.ndarray) and candidate.shape == ():
            candidate = candidate.item()
        candidate_str = str(candidate)
        if candidate_str:
            return _normalize_robot_model(candidate_str)

    return "inspire"


def _resolve_robot_base_pose(processor: Any, motion: dict) -> np.ndarray:
    candidate = motion.get("robot_base_pose", None)
    if candidate is None:
        candidate = getattr(processor, "robot_base_pose", None)
    if candidate is None:
        return np.eye(4, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != (4, 4):
        raise ValueError(
            f"Expected robot_base_pose to have shape (4, 4), got {candidate.shape}."
        )
    return candidate


def _active_hands(processor: Any, motion: dict) -> dict[str, bool]:
    active = {"left": True, "right": True}
    hand_side_available = getattr(processor, "hand_side_available", None)
    if isinstance(hand_side_available, dict):
        for side in ("left", "right"):
            if side in hand_side_available:
                active[side] = bool(hand_side_available[side])
    for side in ("left", "right"):
        key = f"is_active_{side}"
        if key not in motion:
            continue
        value = motion[key]
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0]
        active[side] = bool(value)
    return active


def _pose_dict_for_object_motion_frame(motion: dict, frame_idx: int) -> dict:
    pos = np.asarray(motion["obj_pos"][frame_idx], dtype=float)

    if "obj_quat" in motion and motion["obj_quat"] is not None:
        quat_wxyz = np.asarray(motion["obj_quat"][frame_idx], dtype=float)
        quat_xyzw = np.roll(quat_wxyz, -1)
        rot = R.from_quat(quat_xyzw).as_matrix()
    else:
        rot = R.from_rotvec(
            np.asarray(motion["obj_rot"][frame_idx], dtype=float)
        ).as_matrix()

    joints = _joint_dict_for_frame(
        motion.get("joint_names", []),
        None if motion.get("obj_arti", None) is None else motion["obj_arti"][frame_idx],
    )
    return {"pos": pos, "rot": rot, "joints": joints}


def _pose_dict_for_robot_motion_frame(
    joint_names: Iterable[str],
    joint_vals_frame: Any,
    base_pose: np.ndarray,
) -> dict:
    base_pose = np.asarray(base_pose, dtype=float)
    if base_pose.shape != (4, 4):
        raise ValueError(
            f"Expected robot base pose to have shape (4, 4), got {base_pose.shape}."
        )
    return {
        "pos": base_pose[:3, 3],
        "rot": base_pose[:3, :3],
        "joints": _joint_dict_for_frame(joint_names, joint_vals_frame),
    }


def _pose_dict_to_matrix(pose_dict: dict) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(pose_dict["rot"], dtype=float)
    transform[:3, 3] = np.asarray(pose_dict["pos"], dtype=float).reshape(3)
    return transform


def _human_keypoint_dict_for_frame(
    processor: Any,
    side: str,
    keypoint_names: Iterable[str],
    frame_idx: int,
) -> dict[str, np.ndarray]:
    human_reference_joint_pos = getattr(
        processor,
        f"human_reference_{side}_joint_pos",
        None,
    )
    human_reference_joint_valid = getattr(
        processor,
        f"human_reference_{side}_joint_valid",
        None,
    )
    if human_reference_joint_pos is None and bool(
        getattr(processor, "requires_human_reference_joint_pos_for_eval", False)
    ):
        raise ValueError(
            f"Processor requires human_reference_{side}_joint_pos for eval, but it "
            "was not found. Regenerate the OakInk sequence export so it includes "
            f"{side}_human_reference_keypoint_pos."
        )
    if human_reference_joint_valid is not None:
        valid = np.asarray(human_reference_joint_valid, dtype=bool).reshape(-1)
        if frame_idx < 0 or frame_idx >= valid.shape[0]:
            raise ValueError(
                f"Frame index {frame_idx} is out of range for "
                f"human_reference_{side}_joint_valid with {valid.shape[0]} frames."
            )
        if not bool(valid[frame_idx]):
            raise ValueError(
                f"Frame {frame_idx} has no valid explicit human reference keypoints "
                f"for {side} hand. Use a repeated/static grasp frame for grasp eval."
            )
    joint_pos = (
        human_reference_joint_pos
        if isinstance(human_reference_joint_pos, dict)
        else getattr(processor, f"relevant_{side}_joint_pos", None)
    )
    if not isinstance(joint_pos, dict):
        raise ValueError(f"Processor missing relevant_{side}_joint_pos dictionary.")

    names = [str(name) for name in keypoint_names]
    out: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name in names:
        if name in joint_pos:
            point = np.asarray(joint_pos[name][frame_idx], dtype=np.float64)
            if point.shape != (3,):
                raise ValueError(
                    f"Expected human keypoint {side}/{name} to have shape (3,), "
                    f"got {point.shape}."
                )
            out[name] = point
        else:
            missing.append(name)

    if missing:
        if not hasattr(processor, "get_human_hand_kpt_positions"):
            raise ValueError(
                f"Processor is missing human keypoints {missing} and does not provide "
                "get_human_hand_kpt_positions()."
            )
        surface_points = processor.get_human_hand_kpt_positions(
            side,
            names=missing,
            frame_idx=frame_idx,
        )
        for name in missing:
            if name not in surface_points:
                raise ValueError(
                    f"Could not resolve human surface keypoint {side}/{name}."
                )
            point = np.asarray(surface_points[name], dtype=np.float64)
            if point.shape != (3,):
                raise ValueError(
                    f"Expected human surface keypoint {side}/{name} to have shape (3,), "
                    f"got {point.shape}."
                )
            out[name] = point

    return out


def _robot_keypoint_dict_for_frame(
    *,
    urdf_inst: URDF,
    robot_cfg: dict,
    pose_dict: dict,
    keypoint_names: Iterable[str],
) -> dict[str, np.ndarray]:
    frame_map = dict(robot_cfg["human_robot_link_map"])
    frame_map.update(robot_cfg.get("human_robot_kpt_map", {}))

    requested_names = [str(name) for name in keypoint_names]
    missing_mapping = [name for name in requested_names if name not in frame_map]
    if missing_mapping:
        raise ValueError(
            f"Robot config is missing frame mappings for keypoints: {missing_mapping}."
        )

    base_pose = _pose_dict_to_matrix(pose_dict)
    link_fk = urdf_inst.link_fk(cfg=pose_dict["joints"])
    link_fk = {link.name: transform for link, transform in link_fk.items()}

    out: dict[str, np.ndarray] = {}
    missing_links = []
    for name in requested_names:
        link_name = frame_map[name]
        if link_name not in link_fk:
            missing_links.append(link_name)
            continue
        link_pose = base_pose @ np.asarray(link_fk[link_name], dtype=float)
        out[name] = link_pose[:3, 3].astype(np.float64, copy=False)

    if missing_links:
        raise ValueError(
            "URDF FK missing robot keypoint links required for metric computation: "
            f"{missing_links}."
        )
    return out


def _collision_or_visual_fk(urdf_inst: URDF, joint_dict: dict, mesh_type: str) -> dict:
    if mesh_type == "collision":
        fk = urdf_inst.collision_trimesh_fk(cfg=joint_dict)
        if len(fk) == 0:
            fk = urdf_inst.visual_trimesh_fk(cfg=joint_dict)
        return fk
    if mesh_type == "visual":
        fk = urdf_inst.visual_trimesh_fk(cfg=joint_dict)
        if len(fk) == 0:
            fk = urdf_inst.collision_trimesh_fk(cfg=joint_dict)
        return fk
    raise ValueError(
        f"Unsupported mesh_type={mesh_type!r} (expected 'collision'|'visual')."
    )


def _angle_at_point(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = np.asarray(a, dtype=np.float64).reshape(3) - np.asarray(
        b, dtype=np.float64
    ).reshape(3)
    bc = np.asarray(c, dtype=np.float64).reshape(3) - np.asarray(
        b, dtype=np.float64
    ).reshape(3)
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm <= 1e-12 or bc_norm <= 1e-12:
        raise ValueError(
            "Cannot compute skeleton angle from a zero-length segment: "
            f"|ba|={ba_norm}, |bc|={bc_norm}."
        )
    cosine = float(np.dot(ba, bc) / (ba_norm * bc_norm))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _skeleton_angles(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    missing = sorted(
        {
            name
            for triplet in HAND_SKELETON_ANGLE_TRIPLETS
            for name in triplet
            if name not in keypoints
        }
    )
    if missing:
        raise ValueError(f"Missing skeleton keypoints for angle metric: {missing}.")

    return np.asarray(
        [
            _angle_at_point(keypoints[a], keypoints[b], keypoints[c])
            for a, b, c in HAND_SKELETON_ANGLE_TRIPLETS
        ],
        dtype=np.float64,
    )


def _skeleton_angle_names(side: str) -> list[str]:
    return [f"{side}:{a}-{b}-{c}" for a, b, c in HAND_SKELETON_ANGLE_TRIPLETS]


def _sample_mesh_pcd_with_normals(
    mesh: trimesh.Trimesh | None,
    n_surface_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if mesh is None:
        raise ValueError("Cannot compute BPS metric because object mesh is None.")
    if n_surface_samples <= 0:
        raise ValueError(
            f"Expected hand_object_bps_surface_samples > 0, got {n_surface_samples}."
        )
    seed = int(rng.integers(0, 2**31 - 1))
    points, face_idx = trimesh.sample.sample_surface(
        mesh, int(n_surface_samples), seed=seed
    )
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_idx]
    normal_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.clip(normal_norm, 1e-12, None)
    return np.concatenate(
        [np.asarray(points, dtype=np.float64), normals],
        axis=-1,
    )


def _compute_hand_object_bps(
    pcd: np.ndarray,
    keypoints: dict[str, np.ndarray],
    keypoint_names: Iterable[str],
    chunk_size: int,
) -> np.ndarray:
    ordered_points = np.stack(
        [
            np.asarray(keypoints[name], dtype=np.float64).reshape(3)
            for name in keypoint_names
        ],
        axis=0,
    )
    pcd_t = torch.as_tensor(pcd[None, ...], dtype=torch.float32)
    bps_t = torch.as_tensor(ordered_points[None, ...], dtype=torch.float32)
    bps = compute_bps_efficient(pcd_t, bps_t, chunk_size=int(chunk_size))
    return bps[0].detach().cpu().numpy().astype(np.float64, copy=False)


def _bps_comparison_metrics(
    robot_bps: np.ndarray,
    human_bps: np.ndarray,
) -> tuple[float, float, float, float]:
    if robot_bps.shape != human_bps.shape:
        raise ValueError(
            "Expected robot and human hand-object BPS to have matching shapes, got "
            f"{robot_bps.shape} and {human_bps.shape}."
        )
    if robot_bps.ndim != 2 or robot_bps.shape[1] != 6:
        raise ValueError(
            f"Expected BPS arrays with shape (N,6), got {robot_bps.shape}."
        )

    robot_vec = robot_bps[:, :3]
    human_vec = human_bps[:, :3]
    robot_normal = robot_bps[:, 3:6]
    human_normal = human_bps[:, 3:6]

    robot_dist_sq = np.sum(robot_vec**2, axis=-1) + 1e-8
    human_dist_sq = np.sum(human_vec**2, axis=-1) + 1e-8

    robot_dir = robot_vec / np.sqrt(robot_dist_sq)[:, None]
    human_dir = human_vec / np.sqrt(human_dist_sq)[:, None]

    robot_normal = robot_normal / np.clip(
        np.linalg.norm(robot_normal, axis=-1, keepdims=True), 1e-12, None
    )
    human_normal = human_normal / np.clip(
        np.linalg.norm(human_normal, axis=-1, keepdims=True), 1e-12, None
    )

    bps_mse = float(np.mean((robot_bps - human_bps) ** 2))
    distance_rel_abs = float(np.mean(np.abs(1.0 - robot_dist_sq / human_dist_sq)))
    direction_cosine_distance = float(
        np.mean(1.0 - np.sum(robot_dir * human_dir, axis=-1))
    )
    normal_cosine_distance = float(
        np.mean(1.0 - np.sum(robot_normal * human_normal, axis=-1))
    )
    return (
        bps_mse,
        distance_rel_abs,
        direction_cosine_distance,
        normal_cosine_distance,
    )


def _set_axes_equal_3d(ax, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(points), axis=1)
    if not np.any(finite):
        return
    points = points[finite]
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = 0.5 * float(np.max(points.max(axis=0) - points.min(axis=0)))
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_skeleton_overlay(
    *,
    human_keypoints: dict[str, np.ndarray],
    robot_keypoints: dict[str, np.ndarray],
    side: str,
    frame_idx: int,
    save_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    all_points = []
    for label, keypoints, color in (
        ("human", human_keypoints, "tab:blue"),
        ("robot", robot_keypoints, "tab:orange"),
    ):
        points = np.stack(
            [np.asarray(keypoints[name], dtype=np.float64) for name in keypoints],
            axis=0,
        )
        all_points.append(points)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=18,
            color=color,
            label=label,
        )
        for a, b in HAND_SKELETON_EDGES:
            if a not in keypoints or b not in keypoints:
                continue
            edge = np.stack([keypoints[a], keypoints[b]], axis=0)
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=color, linewidth=1.5)

    _set_axes_equal_3d(ax, np.concatenate(all_points, axis=0))
    ax.set_title(f"{side} hand skeletons, frame {frame_idx}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    ax.view_init(elev=20.0, azim=-65.0)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def export_hand_skeleton_plots_oakink(
    *,
    exported_oakink_processor: Any,
    out_dir: str | Path,
    cfg: GraspEvalConfig | None = None,
    max_frames: int = 20,
) -> list[Path]:
    if cfg is None:
        cfg = GraspEvalConfig()

    processor = exported_oakink_processor
    motion = processor.retargeted_motion
    if motion is None:
        raise ValueError("Processor has no retargeted_motion.")

    n_frames = int(processor.n_frames)
    if n_frames <= 0:
        raise ValueError(f"Expected processor.n_frames > 0, got {n_frames}.")
    if max_frames <= 0 or max_frames >= n_frames:
        frame_indices = np.arange(n_frames, dtype=np.int64)
    else:
        frame_indices = np.unique(
            np.linspace(0, n_frames - 1, num=int(max_frames), dtype=np.int64)
        )

    robot_model = _resolve_robot_model(processor, motion, cfg)
    if robot_model not in ROBOT_SPECS:
        raise ValueError(f"Unsupported resolved robot_model={robot_model!r}.")
    robot_base_pose = _resolve_robot_base_pose(processor, motion)
    active_hands = _active_hands(processor, motion)

    urdf_by_side = _load_robot_urdfs(robot_model)
    cfg_by_side = {
        side: ROBOT_SPECS[robot_model][side][1] for side in ("left", "right")
    }

    saved_paths: list[Path] = []
    out_dir = Path(out_dir)
    for frame_idx in frame_indices.tolist():
        for side in ("left", "right"):
            if not active_hands[side]:
                continue
            jnames = motion.get(f"{side}_jnames", [])
            jvals = motion.get(f"{side}_jvals", None)
            if jvals is None or len(jnames) == 0:
                continue

            robot_cfg = cfg_by_side[side]
            skeleton_names = tuple(robot_cfg["human_robot_link_map"].keys())
            human_keypoints = _human_keypoint_dict_for_frame(
                processor=processor,
                side=side,
                keypoint_names=skeleton_names,
                frame_idx=int(frame_idx),
            )
            robot_pose = _pose_dict_for_robot_motion_frame(
                jnames,
                jvals[int(frame_idx)],
                robot_base_pose,
            )
            robot_keypoints = _robot_keypoint_dict_for_frame(
                urdf_inst=urdf_by_side[side],
                robot_cfg=robot_cfg,
                pose_dict=robot_pose,
                keypoint_names=skeleton_names,
            )
            save_path = out_dir / f"{side}_frame_{int(frame_idx):04d}.png"
            _plot_skeleton_overlay(
                human_keypoints=human_keypoints,
                robot_keypoints=robot_keypoints,
                side=side,
                frame_idx=int(frame_idx),
                save_path=save_path,
            )
            saved_paths.append(save_path)

    return saved_paths


def merged_mesh_from_urdf_frame(
    urdf_inst: URDF,
    joint_dict: dict,
    pos: np.ndarray,
    rot: np.ndarray,
    mesh_type: str,
) -> trimesh.Trimesh | None:
    base_pose = np.eye(4, dtype=float)
    base_pose[:3, :3] = np.asarray(rot, dtype=float)
    base_pose[:3, 3] = np.asarray(pos, dtype=float).reshape(3)

    meshes = []
    fk = _collision_or_visual_fk(
        urdf_inst=urdf_inst, joint_dict=joint_dict, mesh_type=mesh_type
    )
    for mesh, link_pose in fk.items():
        mesh_t = mesh.copy()
        mesh_t.apply_transform(base_pose @ link_pose)
        meshes.append(mesh_t)

    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _points_inside_mesh(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    try:
        return mesh.contains(points)
    except Exception:
        signed_dist = trimesh.proximity.signed_distance(mesh, points)
        return signed_dist > 0.0


def _mesh_triangles_pytorch3d(
    mesh: trimesh.Trimesh, device: torch.device
) -> torch.Tensor:
    verts = torch.as_tensor(
        np.asarray(mesh.vertices), dtype=torch.float64, device=device
    )
    faces = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int64, device=device)
    if verts.numel() == 0 or faces.numel() == 0:
        return torch.empty((0, 3, 3), dtype=torch.float64, device=device)
    mesh_p3d = Meshes(verts=[verts], faces=[faces])
    verts_packed = mesh_p3d.verts_packed()
    faces_packed = mesh_p3d.faces_packed()
    return verts_packed[faces_packed]


def _points_inside_mesh_pytorch3d(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    point_chunk_size: int = 1024,
    face_chunk_size: int = 512,
) -> np.ndarray:
    """
    Robust point-in-mesh test via ray casting using torch/PyTorch3D mesh tensors.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tris = _mesh_triangles_pytorch3d(mesh, device=device)
    if tris.shape[0] == 0:
        return np.zeros((pts.shape[0],), dtype=bool)

    ray_dirs = torch.tensor(
        [[1.0, 0.371, 0.127], [-0.291, 1.0, 0.563], [0.193, -0.417, 1.0]],
        dtype=torch.float64,
        device=device,
    )
    ray_dirs = ray_dirs / torch.linalg.norm(ray_dirs, dim=1, keepdim=True).clamp_min(
        1e-12
    )
    eps = 1e-9

    pts_t = torch.as_tensor(pts, dtype=torch.float64, device=device)
    votes = torch.zeros((pts_t.shape[0],), dtype=torch.int64, device=device)

    for ray_dir in ray_dirs:
        hit_counts = torch.zeros((pts_t.shape[0],), dtype=torch.int64, device=device)
        for p0 in range(0, pts_t.shape[0], point_chunk_size):
            p1 = min(pts_t.shape[0], p0 + point_chunk_size)
            p_chunk = pts_t[p0:p1]
            local_counts = torch.zeros(
                (p_chunk.shape[0],), dtype=torch.int64, device=device
            )

            for f0 in range(0, tris.shape[0], face_chunk_size):
                f1 = min(tris.shape[0], f0 + face_chunk_size)
                tri = tris[f0:f1]
                v0 = tri[:, 0, :]
                v1 = tri[:, 1, :]
                v2 = tri[:, 2, :]

                e1 = v1 - v0
                e2 = v2 - v0
                h = torch.cross(ray_dir.unsqueeze(0).expand_as(e2), e2, dim=1)
                a = torch.sum(e1 * h, dim=1)

                valid = torch.abs(a) > eps
                if not torch.any(valid):
                    continue

                f = torch.zeros_like(a)
                f[valid] = 1.0 / a[valid]

                s = p_chunk[:, None, :] - v0[None, :, :]
                u = torch.sum(s * h[None, :, :], dim=2) * f[None, :]
                q = torch.cross(s, e1[None, :, :], dim=2)
                v = torch.sum(q * ray_dir[None, None, :], dim=2) * f[None, :]
                t = torch.sum(q * e2[None, :, :], dim=2) * f[None, :]

                hit = (
                    valid[None, :]
                    & (u >= -eps)
                    & (v >= -eps)
                    & ((u + v) <= (1.0 + eps))
                    & (t > eps)
                )
                local_counts += torch.sum(hit, dim=1).to(torch.int64)

            hit_counts[p0:p1] = local_counts

        votes += (hit_counts % 2 == 1).to(torch.int64)

    inside = votes >= 2
    return inside.cpu().numpy().astype(bool)


def _max_inside_depth(query_points: np.ndarray, target_mesh: trimesh.Trimesh) -> float:
    inside = _points_inside_mesh(target_mesh, query_points)
    if not np.any(inside):
        return 0.0

    inside_points = query_points[inside]
    _, dist, _ = trimesh.proximity.closest_point(target_mesh, inside_points)
    if dist.size == 0:
        return 0.0
    return float(np.max(dist))


def penetration_depth_m(
    mesh_a: trimesh.Trimesh | None,
    mesh_b: trimesh.Trimesh | None,
    n_surface_samples: int,
    rng: np.random.Generator,
) -> float:
    if mesh_a is None or mesh_b is None:
        return 0.0

    if n_surface_samples <= 0:
        return 0.0

    try:
        seed_a = int(rng.integers(0, 2**31 - 1))
        seed_b = int(rng.integers(0, 2**31 - 1))
        pts_a, _ = trimesh.sample.sample_surface(mesh_a, n_surface_samples, seed=seed_a)
        pts_b, _ = trimesh.sample.sample_surface(mesh_b, n_surface_samples, seed=seed_b)
    except Exception:
        return float("nan")

    depth_a_in_b = _max_inside_depth(pts_a, mesh_b)
    depth_b_in_a = _max_inside_depth(pts_b, mesh_a)
    return float(max(depth_a_in_b, depth_b_in_a))


def intersection_volume_m3_mc(
    mesh_a: trimesh.Trimesh | None,
    mesh_b: trimesh.Trimesh | None,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    if mesh_a is None or mesh_b is None:
        return 0.0
    if n_samples <= 0:
        return 0.0

    lo = np.maximum(mesh_a.bounds[0], mesh_b.bounds[0])
    hi = np.minimum(mesh_a.bounds[1], mesh_b.bounds[1])
    side = hi - lo

    if np.any(side <= 0.0):
        return 0.0

    overlap_vol = float(np.prod(side))
    pts = rng.random((n_samples, 3), dtype=np.float64) * side[None, :] + lo[None, :]

    inside_a = _points_inside_mesh_pytorch3d(mesh_a, pts)
    inside_b = _points_inside_mesh_pytorch3d(mesh_b, pts)
    inside_both = np.logical_and(inside_a, inside_b)

    return overlap_vol * float(np.mean(inside_both.astype(np.float64)))


def intersection_volume_m3_grid(
    mesh_a: trimesh.Trimesh | None,
    mesh_b: trimesh.Trimesh | None,
    max_points: int,
    min_cells_per_axis: int,
) -> float:
    if mesh_a is None or mesh_b is None:
        return 0.0
    if max_points <= 0:
        return 0.0

    lo = np.maximum(mesh_a.bounds[0], mesh_b.bounds[0])
    hi = np.minimum(mesh_a.bounds[1], mesh_b.bounds[1])
    side = hi - lo

    if np.any(side <= 0.0):
        return 0.0

    overlap_vol = float(np.prod(side))
    side_norm = side / float(np.max(side))
    side_norm = np.maximum(side_norm, 1e-6)

    k = (float(max_points) / float(np.prod(side_norm))) ** (1.0 / 3.0)
    grid_shape = np.maximum(
        int(min_cells_per_axis), np.round(k * side_norm).astype(np.int32)
    ).astype(np.int32)

    while int(np.prod(grid_shape)) > int(max_points):
        axis = int(np.argmax(grid_shape))
        if int(grid_shape[axis]) <= int(min_cells_per_axis):
            break
        grid_shape[axis] -= 1

    nx, ny, nz = [int(v) for v in grid_shape.tolist()]
    if nx <= 0 or ny <= 0 or nz <= 0:
        return 0.0

    xs = lo[0] + (np.arange(nx, dtype=np.float64) + 0.5) * (side[0] / nx)
    ys = lo[1] + (np.arange(ny, dtype=np.float64) + 0.5) * (side[1] / ny)
    zs = lo[2] + (np.arange(nz, dtype=np.float64) + 0.5) * (side[2] / nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack((gx.ravel(), gy.ravel(), gz.ravel()), axis=1)

    inside_a = _points_inside_mesh_pytorch3d(mesh_a, pts)
    inside_b = _points_inside_mesh_pytorch3d(mesh_b, pts)
    inside_both = np.logical_and(inside_a, inside_b)
    return overlap_vol * float(np.mean(inside_both.astype(np.float64)))


def chamfer_distance_m(
    points_a: np.ndarray,
    points_b: np.ndarray,
    max_points: int | None,
    rng: np.random.Generator,
) -> float:
    a = np.asarray(points_a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(points_b, dtype=np.float64).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")

    if max_points is not None and max_points > 0:
        if a.shape[0] > max_points:
            a = a[rng.choice(a.shape[0], size=max_points, replace=False)]
        if b.shape[0] > max_points:
            b = b[rng.choice(b.shape[0], size=max_points, replace=False)]

    tree_b = cKDTree(b)
    d_a, _ = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    d_b, _ = tree_a.query(b, k=1)
    return float(np.mean(d_a) + np.mean(d_b))


def _force_closure_feasible(wrenches: np.ndarray) -> bool:
    wrenches = np.asarray(wrenches, dtype=np.float64)
    if wrenches.ndim != 2 or wrenches.shape[1] != 6:
        raise ValueError(f"Expected wrenches shaped (K,6), got {wrenches.shape}.")
    k = int(wrenches.shape[0])
    if k < 4:
        return False

    # Check whether origin is in the convex hull of wrenches (feasible convex combination).
    # Solve: W^T a = 0, sum(a)=1, a>=0
    A_eq = np.concatenate([wrenches.T, np.ones((1, k), dtype=np.float64)], axis=0)
    b_eq = np.zeros(7, dtype=np.float64)
    b_eq[-1] = 1.0
    res = linprog(
        c=np.zeros(k, dtype=np.float64),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * k,
        method="highs",
    )
    return bool(res.success)


def force_closure_satisfied_from_meshes(
    hand_mesh: trimesh.Trimesh | None,
    obj_mesh: trimesh.Trimesh | None,
    threshold_m: float,
    hand_surface_samples: int,
    rng: np.random.Generator,
) -> tuple[bool, int]:
    if hand_mesh is None or obj_mesh is None:
        return False, 0
    if threshold_m <= 0 or hand_surface_samples <= 0:
        return False, 0

    try:
        seed = int(rng.integers(0, 2**31 - 1))
        hand_pts, _ = trimesh.sample.sample_surface(
            hand_mesh, int(hand_surface_samples), seed=seed
        )
    except Exception:
        return False, 0

    closest_pts, dist, tri_id = trimesh.proximity.closest_point(obj_mesh, hand_pts)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    tri_id = np.asarray(tri_id).reshape(-1)
    keep = dist <= float(threshold_m)
    if not np.any(keep):
        return False, 0

    contact_pts = np.asarray(closest_pts, dtype=np.float64)[keep]
    contact_tri = tri_id[keep]
    contact_tri = np.clip(contact_tri, 0, len(obj_mesh.face_normals) - 1)
    normals = np.asarray(obj_mesh.face_normals, dtype=np.float64)[contact_tri]

    # Force on object is opposite its outward normal.
    forces = -normals

    # Use mesh center of mass if available, else centroid.
    try:
        obj_center = np.asarray(obj_mesh.center_mass, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(obj_center)):
            raise ValueError("Non-finite center_mass.")
    except Exception:
        obj_center = np.asarray(obj_mesh.centroid, dtype=np.float64).reshape(3)

    lever = contact_pts - obj_center[None, :]
    torques = np.cross(lever, forces)
    wrenches = np.concatenate([forces, torques], axis=1)
    return _force_closure_feasible(wrenches), int(contact_pts.shape[0])


def compute_grasp_eval_metrics_oakink(
    *,
    exported_oakink_processor: Any,
    cfg: GraspEvalConfig | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute grasp evaluation metrics for an ExportedOakInkProcessor instance.

    Returns a dict of per-frame numpy arrays with keys:
      - penetration_depth_cm
      - intersection_volume_cm3
      - chamfer_human_robot_cm
      - hand_skeleton_angle_mse_rad2
      - hand_skeleton_angle_mae_deg
      - hand_object_bps_mse
      - hand_object_bps_distance_rel_abs
      - hand_object_bps_direction_cosine_distance
      - hand_object_bps_normal_cosine_distance
      - force_closure_satisfied_t{thr}
      - force_closure_contact_count_t{thr}
    """
    if cfg is None:
        cfg = GraspEvalConfig()

    if cfg.mesh_type not in ("collision", "visual"):
        raise ValueError("--mesh_type must be 'collision' or 'visual'.")
    if cfg.intersection_method not in ("grid", "mc"):
        raise ValueError("--intersection_method must be 'grid' or 'mc'.")

    processor = exported_oakink_processor
    motion = processor.retargeted_motion
    if motion is None:
        raise ValueError("Processor has no retargeted_motion.")

    n_frames = int(processor.n_frames)
    obj_motion = processor.object_reference_motion
    robot_model = _resolve_robot_model(processor, motion, cfg)
    if robot_model not in ROBOT_SPECS:
        raise ValueError(f"Unsupported resolved robot_model={robot_model!r}.")
    robot_base_pose = _resolve_robot_base_pose(processor, motion)
    active_hands = _active_hands(processor, motion)

    left_robot_cfg = ROBOT_SPECS[robot_model]["left"][1]
    right_robot_cfg = ROBOT_SPECS[robot_model]["right"][1]
    robot_urdfs = _load_robot_urdfs(robot_model)
    left_urdf = robot_urdfs["left"]
    right_urdf = robot_urdfs["right"]
    object_urdf = URDF.load(str(process_urdf(processor.target_obj_urdf_path)))
    reference_object_urdf = URDF.load(str(process_urdf(processor.obj_urdf_path)))

    left_jnames = motion.get("left_jnames", [])
    right_jnames = motion.get("right_jnames", [])
    left_jvals = motion.get("left_jvals", None)
    right_jvals = motion.get("right_jvals", None)

    rng = np.random.default_rng(int(cfg.rng_seed))

    penetration_depth_cm = np.zeros(n_frames, dtype=np.float64)
    intersection_volume_cm3 = np.zeros(n_frames, dtype=np.float64)
    chamfer_human_robot_cm = np.full(n_frames, np.nan, dtype=np.float64)
    hand_skeleton_angle_mse_rad2 = np.full(n_frames, np.nan, dtype=np.float64)
    hand_skeleton_angle_mae_deg = np.full(n_frames, np.nan, dtype=np.float64)
    hand_object_bps_mse = np.full(n_frames, np.nan, dtype=np.float64)
    hand_object_bps_distance_rel_abs = np.full(n_frames, np.nan, dtype=np.float64)
    hand_object_bps_direction_cosine_distance = np.full(
        n_frames, np.nan, dtype=np.float64
    )
    hand_object_bps_normal_cosine_distance = np.full(n_frames, np.nan, dtype=np.float64)

    thr_list = [float(t) for t in cfg.force_closure_thresholds_m]
    thr_keys = [f"t{t:g}".replace(".", "p") for t in thr_list]
    force_closure_satisfied = {k: np.zeros(n_frames, dtype=np.int8) for k in thr_keys}
    force_closure_contact_count = {
        k: np.zeros(n_frames, dtype=np.int32) for k in thr_keys
    }
    skeleton_angle_names = [
        angle_name
        for side in ("left", "right")
        if active_hands[side]
        for angle_name in _skeleton_angle_names(side)
    ]
    hand_skeleton_angle_abs_error_deg = np.full(
        (n_frames, len(skeleton_angle_names)),
        np.nan,
        dtype=np.float64,
    )
    hand_skeleton_angle_sq_error_rad2 = np.full(
        (n_frames, len(skeleton_angle_names)),
        np.nan,
        dtype=np.float64,
    )

    for frame_idx in range(n_frames):
        obj_pose = _pose_dict_for_object_motion_frame(obj_motion, frame_idx)
        retargeted_obj_pose = _pose_dict_for_object_motion_frame(motion, frame_idx)
        reference_obj_mesh = merged_mesh_from_urdf_frame(
            urdf_inst=reference_object_urdf,
            joint_dict=obj_pose["joints"],
            pos=obj_pose["pos"],
            rot=obj_pose["rot"],
            mesh_type=cfg.mesh_type,
        )
        obj_mesh = merged_mesh_from_urdf_frame(
            urdf_inst=object_urdf,
            joint_dict=retargeted_obj_pose["joints"],
            pos=retargeted_obj_pose["pos"],
            rot=retargeted_obj_pose["rot"],
            mesh_type=cfg.mesh_type,
        )

        frame_pen_m = 0.0
        frame_vol_m3 = 0.0
        frame_human_angles: list[np.ndarray] = []
        frame_robot_angles: list[np.ndarray] = []
        frame_angle_offsets: list[tuple[int, int]] = []
        frame_human_bps: list[np.ndarray] = []
        frame_robot_bps: list[np.ndarray] = []
        reference_obj_pcd = _sample_mesh_pcd_with_normals(
            reference_obj_mesh,
            n_surface_samples=int(cfg.hand_object_bps_surface_samples),
            rng=rng,
        )
        retargeted_obj_pcd = _sample_mesh_pcd_with_normals(
            obj_mesh,
            n_surface_samples=int(cfg.hand_object_bps_surface_samples),
            rng=rng,
        )

        # Evaluate robot/object using both hands if present, consistent with vis_robot_retgt_motion.
        angle_offset = 0
        for side, side_urdf, robot_cfg, jnames, jvals in (
            ("left", left_urdf, left_robot_cfg, left_jnames, left_jvals),
            ("right", right_urdf, right_robot_cfg, right_jnames, right_jvals),
        ):
            if not active_hands[side]:
                continue
            side_angle_count = len(HAND_SKELETON_ANGLE_TRIPLETS)
            if jvals is None or len(jnames) == 0:
                angle_offset += side_angle_count
                continue

            joint_dict = _joint_dict_for_frame(jnames, jvals[frame_idx])
            robot_pose = _pose_dict_for_robot_motion_frame(
                jnames,
                jvals[frame_idx],
                robot_base_pose,
            )
            hand_mesh = merged_mesh_from_urdf_frame(
                urdf_inst=side_urdf,
                joint_dict=joint_dict,
                pos=robot_pose["pos"],
                rot=robot_pose["rot"],
                mesh_type=cfg.mesh_type,
            )

            skeleton_names = tuple(robot_cfg["human_robot_link_map"].keys())
            human_skeleton_keypoints = _human_keypoint_dict_for_frame(
                processor=processor,
                side=side,
                keypoint_names=skeleton_names,
                frame_idx=frame_idx,
            )
            robot_skeleton_keypoints = _robot_keypoint_dict_for_frame(
                urdf_inst=side_urdf,
                robot_cfg=robot_cfg,
                pose_dict=robot_pose,
                keypoint_names=skeleton_names,
            )
            frame_human_angles.append(_skeleton_angles(human_skeleton_keypoints))
            frame_robot_angles.append(_skeleton_angles(robot_skeleton_keypoints))
            frame_angle_offsets.append((angle_offset, angle_offset + side_angle_count))
            angle_offset += side_angle_count

            bps_names = tuple(
                robot_cfg.get(
                    "human_robot_kpt_map",
                    robot_cfg["human_robot_link_map"],
                ).keys()
            )
            human_bps_keypoints = _human_keypoint_dict_for_frame(
                processor=processor,
                side=side,
                keypoint_names=bps_names,
                frame_idx=frame_idx,
            )
            robot_bps_keypoints = _robot_keypoint_dict_for_frame(
                urdf_inst=side_urdf,
                robot_cfg=robot_cfg,
                pose_dict=robot_pose,
                keypoint_names=bps_names,
            )
            frame_human_bps.append(
                _compute_hand_object_bps(
                    reference_obj_pcd,
                    human_bps_keypoints,
                    bps_names,
                    chunk_size=int(cfg.hand_object_bps_chunk_size),
                )
            )
            frame_robot_bps.append(
                _compute_hand_object_bps(
                    retargeted_obj_pcd,
                    robot_bps_keypoints,
                    bps_names,
                    chunk_size=int(cfg.hand_object_bps_chunk_size),
                )
            )

            pen_m = penetration_depth_m(
                mesh_a=hand_mesh,
                mesh_b=obj_mesh,
                n_surface_samples=int(cfg.penetration_surface_samples),
                rng=rng,
            )
            if not np.isnan(pen_m):
                frame_pen_m = max(frame_pen_m, float(pen_m))

            if cfg.intersection_method == "mc":
                vol_m3 = intersection_volume_m3_mc(
                    mesh_a=hand_mesh,
                    mesh_b=obj_mesh,
                    n_samples=int(cfg.intersection_mc_samples),
                    rng=rng,
                )
            else:
                vol_m3 = intersection_volume_m3_grid(
                    mesh_a=hand_mesh,
                    mesh_b=obj_mesh,
                    max_points=int(cfg.intersection_grid_max_points),
                    min_cells_per_axis=int(cfg.intersection_grid_min_cells_per_axis),
                )
            frame_vol_m3 += float(vol_m3)

            # Force closure only evaluated on the right hand by default (human mano verts are right-hand).
            if side == "right":
                for thr, key in zip(thr_list, thr_keys):
                    ok, cnt = force_closure_satisfied_from_meshes(
                        hand_mesh=hand_mesh,
                        obj_mesh=obj_mesh,
                        threshold_m=float(thr),
                        hand_surface_samples=int(
                            cfg.force_closure_hand_surface_samples
                        ),
                        rng=rng,
                    )
                    force_closure_satisfied[key][frame_idx] = 1 if ok else 0
                    force_closure_contact_count[key][frame_idx] = int(cnt)

                # Chamfer between human (MANO) vertices and robot (URDF mesh) vertices.
                human_verts = None
                if hasattr(processor, "right_hand_verts"):
                    human_verts = np.asarray(processor.right_hand_verts[frame_idx])
                if human_verts is not None and hand_mesh is not None:
                    chamfer_m = chamfer_distance_m(
                        points_a=human_verts,
                        points_b=np.asarray(hand_mesh.vertices),
                        max_points=(
                            int(cfg.chamfer_max_points)
                            if cfg.chamfer_max_points > 0
                            else None
                        ),
                        rng=rng,
                    )
                    chamfer_human_robot_cm[frame_idx] = float(chamfer_m) * 100.0

        penetration_depth_cm[frame_idx] = frame_pen_m * 100.0
        intersection_volume_cm3[frame_idx] = frame_vol_m3 * 1e6
        if frame_human_angles:
            angle_diff_chunks = [
                robot_angles - human_angles
                for human_angles, robot_angles in zip(
                    frame_human_angles,
                    frame_robot_angles,
                )
            ]
            for (start, end), angle_diff_chunk in zip(
                frame_angle_offsets,
                angle_diff_chunks,
            ):
                hand_skeleton_angle_abs_error_deg[frame_idx, start:end] = np.rad2deg(
                    np.abs(angle_diff_chunk)
                )
                hand_skeleton_angle_sq_error_rad2[frame_idx, start:end] = (
                    angle_diff_chunk**2
                )
            angle_diff = np.concatenate(angle_diff_chunks, axis=0)
            hand_skeleton_angle_mse_rad2[frame_idx] = float(np.mean(angle_diff**2))
            hand_skeleton_angle_mae_deg[frame_idx] = float(
                np.rad2deg(np.mean(np.abs(angle_diff)))
            )
        if frame_human_bps:
            robot_bps = np.concatenate(frame_robot_bps, axis=0)
            human_bps = np.concatenate(frame_human_bps, axis=0)
            (
                hand_object_bps_mse[frame_idx],
                hand_object_bps_distance_rel_abs[frame_idx],
                hand_object_bps_direction_cosine_distance[frame_idx],
                hand_object_bps_normal_cosine_distance[frame_idx],
            ) = _bps_comparison_metrics(robot_bps=robot_bps, human_bps=human_bps)

    out: dict[str, np.ndarray] = {
        "penetration_depth_cm": penetration_depth_cm,
        "intersection_volume_cm3": intersection_volume_cm3,
        "chamfer_human_robot_cm": chamfer_human_robot_cm,
        "hand_skeleton_angle_mse_rad2": hand_skeleton_angle_mse_rad2,
        "hand_skeleton_angle_mae_deg": hand_skeleton_angle_mae_deg,
        "hand_skeleton_angle_names": np.asarray(skeleton_angle_names, dtype=object),
        "hand_skeleton_angle_abs_error_deg": hand_skeleton_angle_abs_error_deg,
        "hand_skeleton_angle_sq_error_rad2": hand_skeleton_angle_sq_error_rad2,
        "hand_object_bps_mse": hand_object_bps_mse,
        "hand_object_bps_distance_rel_abs": hand_object_bps_distance_rel_abs,
        "hand_object_bps_direction_cosine_distance": (
            hand_object_bps_direction_cosine_distance
        ),
        "hand_object_bps_normal_cosine_distance": (
            hand_object_bps_normal_cosine_distance
        ),
        "robot_model": np.asarray(robot_model, dtype=object),
        "force_closure_thresholds_m": np.asarray(thr_list, dtype=np.float64),
    }
    for k in thr_keys:
        out[f"force_closure_satisfied_{k}"] = force_closure_satisfied[k]
        out[f"force_closure_contact_count_{k}"] = force_closure_contact_count[k]
    return out
