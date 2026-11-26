"""Shared mesh/FK helpers for standalone reference and policy rendering."""


from __future__ import annotations


from pathlib import Path


import numpy as np


import trimesh


from PIL import Image


from scipy.spatial.transform import Rotation as R


from urdfpy import URDF


from dexlink.visualizations.retargeting.render_oakink_last_frame_views import _copy_with_color, _pose_from_pos_rot


from dexlink.utils.retargeting.render_utils import TrimeshInstance


PROJECT_ROOT = Path(__file__).resolve().parents[3]


OBJECT_COLOR_RGBA = np.array([0x64, 0x95, 0xED, 255], dtype=np.uint8)


def _localize_path(path: str | Path) -> Path:
    from dexlink.utils.retargeting.path_utils import resolve_project_path
    return resolve_project_path(path)


def _unwrap_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _motion_joint_names(value, field_name: str) -> list[str]:
    arr = np.asarray(value, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"Expected {field_name} shape (J,), got {arr.shape}.")
    return [str(name) for name in arr.tolist()]


def _motion_scalar(value, field_name: str):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        raise ValueError(f"Expected scalar {field_name}, got shape {value.shape}.")
    return value


def _resolve_visualized_joint_motion(
    motion: dict,
    *,
    side: str,
    use_refined: bool = True,
) -> tuple[list[str], np.ndarray | None]:
    if side not in ("left", "right"):
        raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")

    base_jnames = _motion_joint_names(
        motion.get(f"{side}_jnames", []), f"{side}_jnames"
    )
    base_jvals = motion.get(f"{side}_jvals", None)
    if not use_refined or "drake_collision_refinement_jvals" not in motion:
        return base_jnames, (
            None if base_jvals is None else np.asarray(base_jvals, dtype=np.float32)
        )

    refined_hand = str(
        _motion_scalar(
            motion["drake_collision_refinement_hand"],
            "drake_collision_refinement_hand",
        )
    )
    if refined_hand != side:
        return base_jnames, (
            None if base_jvals is None else np.asarray(base_jvals, dtype=np.float32)
        )

    refined_jnames = _motion_joint_names(
        motion["drake_collision_refinement_jnames"],
        "drake_collision_refinement_jnames",
    )
    refined_key = f"drake_collision_refinement_{side}_jvals"
    if refined_key not in motion:
        refined_key = "drake_collision_refinement_jvals"
    refined_jvals = np.asarray(motion[refined_key], dtype=np.float32)
    if refined_jvals.ndim != 2:
        raise ValueError(
            f"Expected {refined_key} shape (T, J), got {refined_jvals.shape}."
        )
    if refined_jvals.shape[1] != len(refined_jnames):
        raise ValueError(
            f"Expected {refined_key} width to match refined joint names, got "
            f"{refined_jvals.shape[1]} and {len(refined_jnames)}."
        )
    return refined_jnames, refined_jvals


def _hand_config_source_by_side(motion: dict) -> dict[str, str]:
    raw = motion.get("hand_config_source_by_side", {})
    if isinstance(raw, np.ndarray):
        if raw.shape != ():
            raise ValueError(
                "Expected hand_config_source_by_side to be a scalar object array, "
                f"got shape {raw.shape}."
            )
        raw = raw.item()
    if raw in (None, "", "None"):
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            "Expected hand_config_source_by_side to be a dict, "
            f"got {type(raw).__name__}."
        )
    source_by_side: dict[str, str] = {}
    for role_side, source_side in raw.items():
        role_side = str(role_side)
        source_side = str(source_side)
        if role_side not in ("left", "right") or source_side not in ("left", "right"):
            raise ValueError(
                "Expected hand_config_source_by_side to map left/right role sides "
                f"to left/right config sides, got {raw!r}."
            )
        source_by_side[role_side] = source_side
    return source_by_side


def _scene_object_joint_dict_for_frame(
    motion: dict[str, np.ndarray], frame_idx: int, object_idx: int
) -> dict[str, float]:
    raw_names = motion["scene_object_arti_names"][object_idx]
    names = [str(name) for name in list(raw_names)]
    valid_mask = np.asarray(
        motion["scene_object_arti_valid_mask"][object_idx], dtype=bool
    )
    vals = np.asarray(
        motion["scene_object_arti_vals"][frame_idx, object_idx], dtype=np.float32
    )
    if int(valid_mask.sum()) != len(names):
        raise ValueError(
            f"scene object {object_idx}: valid articulation count "
            f"{int(valid_mask.sum())} does not match names length {len(names)}."
        )
    return {
        name: float(value)
        for name, value in zip(names, vals[valid_mask].reshape(-1).tolist())
    }


def _scene_object_pose_for_frame(
    motion: dict[str, np.ndarray], frame_idx: int, object_idx: int
) -> np.ndarray:
    pos = np.asarray(
        motion["scene_object_pos"][frame_idx, object_idx], dtype=np.float32
    )
    quat_wxyz = np.asarray(
        motion["scene_object_quat"][frame_idx, object_idx], dtype=np.float32
    )
    rot = R.from_quat(np.roll(quat_wxyz, -1)).as_matrix().astype(np.float32)
    return _pose_from_pos_rot(pos, rot)


def _fk_mesh_instances_with_color(
    *,
    urdf_inst: URDF,
    joint_dict: dict[str, float],
    base_pose: np.ndarray,
    mesh_type: str,
    color_rgba: np.ndarray,
) -> list[TrimeshInstance]:
    if mesh_type == "visual":
        fk = urdf_inst.visual_trimesh_fk(cfg=joint_dict)
    elif mesh_type == "collision":
        fk = urdf_inst.collision_trimesh_fk(cfg=joint_dict)
    else:
        raise ValueError(f"Unsupported mesh_type {mesh_type!r}.")
    return [
        TrimeshInstance(
            mesh=_copy_with_color(mesh, color_rgba),
            pose=base_pose @ np.asarray(local_pose, dtype=np.float32),
        )
        for mesh, local_pose in fk.items()
    ]


def _cividis_link_colors(n_colors: int) -> list[np.ndarray]:
    if n_colors <= 0:
        raise ValueError(f"Expected n_colors > 0, got {n_colors}.")
    import seaborn as sns

    colors = sns.color_palette("cividis", n_colors=n_colors)
    return [
        np.array(
            [round(channel * 255.0) for channel in color] + [255],
            dtype=np.uint8,
        )
        for color in colors
    ]


def _fk_robot_link_instances(
    *,
    urdf_inst: URDF,
    joint_dict: dict[str, float],
    base_pose: np.ndarray,
    mesh_type: str,
) -> list[TrimeshInstance]:
    if mesh_type == "visual":
        fk = urdf_inst.visual_trimesh_fk(cfg=joint_dict)
    elif mesh_type == "collision":
        fk = urdf_inst.collision_trimesh_fk(cfg=joint_dict)
    else:
        raise ValueError(f"Unsupported mesh_type {mesh_type!r}.")

    colors = _cividis_link_colors(len(fk))
    return [
        TrimeshInstance(
            mesh=_copy_with_color(mesh, color),
            pose=base_pose @ np.asarray(local_pose, dtype=np.float32),
        )
        for color, (mesh, local_pose) in zip(colors, fk.items())
    ]


def _transformed_meshes(instances: list[TrimeshInstance]) -> list[trimesh.Trimesh]:
    meshes = []
    for instance in instances:
        mesh = instance.mesh.copy()
        pose = np.asarray(instance.pose, dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"Expected pose shape (4, 4), got {pose.shape}.")
        mesh.apply_transform(pose)
        meshes.append(mesh)
    if len(meshes) == 0:
        raise ValueError("Expected at least one mesh.")
    return meshes


def _look_at(
    camera_position: np.ndarray, target: np.ndarray, up: np.ndarray
) -> np.ndarray:
    camera_position = np.asarray(camera_position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    z_axis = camera_position - target
    z_norm = np.linalg.norm(z_axis)
    if z_norm <= 1e-8:
        raise ValueError("Camera position must differ from target.")
    z_axis /= z_norm
    x_axis = np.cross(up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= 1e-8:
        raise ValueError("Failed to construct camera x axis.")
    x_axis /= x_norm
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = camera_position
    return pose


def _render_panel(
    meshes: list[trimesh.Trimesh],
    *,
    camera_pose: np.ndarray,
    xmag: float,
    ymag: float,
    width: int,
    height: int,
) -> Image.Image:
    import pyrender

    scene = pyrender.Scene(
        bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
        ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
    )
    for mesh in meshes:
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    scene.add(pyrender.OrthographicCamera(xmag=xmag, ymag=ymag), pose=camera_pose)
    for offset, intensity in (
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), 2.0),
        (np.array([0.25, -0.15, 0.35], dtype=np.float32), 1.1),
        (np.array([-0.25, -0.10, -0.20], dtype=np.float32), 0.7),
    ):
        light_pose = camera_pose.copy()
        light_pose[:3, 3] = camera_pose[:3, 3] + offset
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
            pose=light_pose,
        )

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _ = renderer.render(
            scene,
            flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES,
        )
    finally:
        renderer.delete()
    return Image.fromarray(color, mode="RGBA").convert("RGB")
