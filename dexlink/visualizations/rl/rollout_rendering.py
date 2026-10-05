"""Shared, portable helpers for rendering recorded DexLink policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
from urdfpy import URDF

from dexlink.utils.retargeting.path_utils import resolve_project_path
from dexlink.utils.retargeting.render_utils import TrimeshInstance
from dexlink.visualizations.retargeting.render_oakink_last_frame_views import (
    _copy_with_color,
    _pose_from_pos_rot,
)


OBJECT_COLOR_RGBA = np.array([0x64, 0x95, 0xED, 255], dtype=np.uint8)


@dataclass(frozen=True)
class CameraSpec:
    """Fixed camera direction and orthographic framing margin."""

    direction: tuple[float, float, float]
    margin: float


def load_episode_from_eval(
    eval_dir: Path, episode_index: int
) -> dict[str, np.ndarray]:
    """Load one recorded environment episode from chunked evaluation output."""
    if episode_index < 0:
        raise ValueError(f"Expected episode_index >= 0, got {episode_index}.")
    rollouts_dir = eval_dir / "rollouts"
    chunk_paths = sorted(rollouts_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No rollout chunks found in {rollouts_dir}.")

    episode_offset = 0
    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=True) as data:
            chunk = {key: data[key] for key in data.files}
        if "done" not in chunk:
            raise KeyError(f"Rollout chunk is missing required field 'done': {chunk_path}")
        done = np.asarray(chunk["done"])
        if done.ndim != 2:
            raise ValueError(
                f"Expected {chunk_path} field 'done' shape (T, N), got {done.shape}."
            )

        episode_count = int(done.shape[1])
        if episode_index < episode_offset + episode_count:
            environment_index = episode_index - episode_offset
            done_indices = np.flatnonzero(done[:, environment_index])
            episode_length = (
                int(done_indices[0] + 1)
                if done_indices.size > 0
                else int(done.shape[0])
            )
            episode = {
                key: np.asarray(value)[:episode_length, environment_index]
                for key, value in chunk.items()
                if key != "start_timestep"
                and np.asarray(value).ndim >= 2
                and np.asarray(value).shape[0] >= episode_length
                and np.asarray(value).shape[1] > environment_index
            }
            if "start_timestep" in chunk:
                start_timestep = np.asarray(chunk["start_timestep"])
                if start_timestep.ndim != 1 or start_timestep.shape[0] != episode_count:
                    raise ValueError(
                        f"Expected {chunk_path} field 'start_timestep' shape "
                        f"{(episode_count,)}, got {start_timestep.shape}."
                    )
                episode["start_timestep"] = start_timestep[environment_index]
            episode["length"] = np.asarray(episode_length, dtype=np.int64)
            return episode
        episode_offset += episode_count

    raise IndexError(
        f"Episode index {episode_index} is outside {episode_offset} episodes in "
        f"{eval_dir}."
    )


def rollout_motion_from_reference(
    reference_motion: dict[str, object], episode: dict[str, np.ndarray]
) -> dict[str, object]:
    """Combine immutable reference metadata with recorded simulator state."""
    required_episode_fields = ("object_pos", "object_quat", "object_arti")
    missing = [field for field in required_episode_fields if field not in episode]
    if missing:
        raise KeyError(f"Recorded episode is missing required fields {missing}.")

    motion = {
        key: value
        for key, value in reference_motion.items()
        if not key.startswith("drake_collision_refinement_")
    }
    motion["scene_object_pos"] = np.asarray(
        episode["object_pos"], dtype=np.float32
    )
    motion["scene_object_quat"] = np.asarray(
        episode["object_quat"], dtype=np.float32
    )
    motion["scene_object_arti_vals"] = np.asarray(
        episode["object_arti"], dtype=np.float32
    )
    for side in ("left", "right"):
        field = f"{side}_jvals"
        if field in episode:
            motion[field] = np.asarray(episode[field], dtype=np.float32)
    return motion


def unwrap_scalar(value: object) -> object:
    """Unwrap an explicitly scalar NumPy value."""
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ValueError(f"Expected a scalar array, got shape {value.shape}.")
        return value.item()
    return value


def load_exported_npy_payload(path: Path) -> dict[str, object]:
    """Load the dictionary payload produced by the reference exporter."""
    if path.suffix != ".npy":
        raise ValueError(f"Expected an exported .npy payload, got {path}.")
    raw = np.load(path, allow_pickle=True)
    if not isinstance(raw, np.ndarray) or raw.shape != ():
        raise ValueError(
            f"Expected {path} to contain one scalar dictionary, got "
            f"{type(raw).__name__} with shape {getattr(raw, 'shape', None)}."
        )
    payload = raw.item()
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected {path} to contain a dictionary, got {type(payload).__name__}."
        )
    return payload


def localize_path(path: str | Path) -> Path:
    """Resolve an absolute or configured DexLink-relative path strictly."""
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Expected an existing absolute or DexLink-relative path, got {path!r} "
            f"(resolved to {resolved})."
        )
    return resolved


def _motion_joint_names(value: object, field_name: str) -> tuple[str, ...]:
    names = np.asarray(value, dtype=object)
    if names.ndim != 1:
        raise ValueError(f"Expected {field_name} shape (J,), got {names.shape}.")
    return tuple(str(name) for name in names.tolist())


def resolve_visualized_joint_motion(
    motion: dict[str, object], *, side: str
) -> tuple[tuple[str, ...], np.ndarray | None]:
    """Return the active hand joint sequence from a rollout payload."""
    if side not in {"left", "right"}:
        raise ValueError(f"Expected side 'left' or 'right', got {side!r}.")
    names = _motion_joint_names(motion.get(f"{side}_jnames", []), f"{side}_jnames")
    values = motion.get(f"{side}_jvals")
    if values is None:
        return names, None
    joint_values = np.asarray(values, dtype=np.float32)
    if joint_values.ndim != 2 or joint_values.shape[1] != len(names):
        raise ValueError(
            f"Expected {side}_jvals shape (T, {len(names)}), got "
            f"{joint_values.shape}."
        )
    return names, joint_values


def hand_config_source_by_side(motion: dict[str, object]) -> dict[str, str]:
    """Validate role-hand to configuration-hand mappings."""
    raw = motion.get("hand_config_source_by_side", {})
    if isinstance(raw, np.ndarray):
        raw = unwrap_scalar(raw)
    if raw in (None, "", "None"):
        return {}
    if not isinstance(raw, dict):
        raise TypeError(
            "Expected hand_config_source_by_side to be a dictionary, got "
            f"{type(raw).__name__}."
        )
    mapping = {str(role): str(source) for role, source in raw.items()}
    invalid = {
        role: source
        for role, source in mapping.items()
        if role not in {"left", "right"} or source not in {"left", "right"}
    }
    if invalid:
        raise ValueError(
            "Expected hand_config_source_by_side to map left/right to left/right, "
            f"got invalid entries {invalid}."
        )
    return mapping


def scene_object_joint_dict_for_frame(
    motion: dict[str, object], frame_index: int, object_index: int
) -> dict[str, float]:
    """Extract one scene object's articulated configuration."""
    names_by_object = np.asarray(motion["scene_object_arti_names"], dtype=object)
    valid_masks = np.asarray(motion["scene_object_arti_valid_mask"], dtype=bool)
    values = np.asarray(motion["scene_object_arti_vals"], dtype=np.float32)
    names = tuple(str(name) for name in list(names_by_object[object_index]))
    valid_mask = valid_masks[object_index]
    object_values = values[frame_index, object_index]
    if valid_mask.shape != object_values.shape:
        raise ValueError(
            f"Expected articulation mask and value shapes to match for object "
            f"{object_index}, got {valid_mask.shape} and {object_values.shape}."
        )
    if int(valid_mask.sum()) != len(names):
        raise ValueError(
            f"Scene object {object_index} has {int(valid_mask.sum())} valid "
            f"articulation values but {len(names)} names."
        )
    return {
        name: float(value)
        for name, value in zip(names, object_values[valid_mask].tolist(), strict=True)
    }


def scene_object_pose_for_frame(
    motion: dict[str, object], frame_index: int, object_index: int
) -> np.ndarray:
    """Build the world transform for one scene object and frame."""
    positions = np.asarray(motion["scene_object_pos"], dtype=np.float32)
    quaternions = np.asarray(motion["scene_object_quat"], dtype=np.float32)
    position = positions[frame_index, object_index]
    quaternion_wxyz = quaternions[frame_index, object_index]
    if position.shape != (3,) or quaternion_wxyz.shape != (4,):
        raise ValueError(
            "Expected object position/quaternion shapes (3,) and (4,), got "
            f"{position.shape} and {quaternion_wxyz.shape}."
        )
    rotation = Rotation.from_quat(np.roll(quaternion_wxyz, -1)).as_matrix()
    return _pose_from_pos_rot(position, rotation)


def fk_mesh_instances_with_color(
    *,
    urdf_inst: URDF,
    joint_dict: dict[str, float],
    base_pose: np.ndarray,
    mesh_type: str,
    color_rgba: np.ndarray,
) -> list[TrimeshInstance]:
    """Create colored mesh instances from one URDF configuration."""
    if mesh_type == "visual":
        transforms = urdf_inst.visual_trimesh_fk(cfg=joint_dict)
    elif mesh_type == "collision":
        transforms = urdf_inst.collision_trimesh_fk(cfg=joint_dict)
    else:
        raise ValueError(f"Expected mesh_type 'visual' or 'collision', got {mesh_type!r}.")
    return [
        TrimeshInstance(
            mesh=_copy_with_color(mesh, color_rgba),
            pose=base_pose @ np.asarray(local_pose, dtype=np.float32),
        )
        for mesh, local_pose in transforms.items()
    ]


def fk_robot_link_instances(
    *,
    urdf_inst: URDF,
    joint_dict: dict[str, float],
    base_pose: np.ndarray,
    mesh_type: str,
) -> list[TrimeshInstance]:
    """Create robot-link instances colored with the paper's cividis palette."""
    if mesh_type == "visual":
        transforms = urdf_inst.visual_trimesh_fk(cfg=joint_dict)
    elif mesh_type == "collision":
        transforms = urdf_inst.collision_trimesh_fk(cfg=joint_dict)
    else:
        raise ValueError(f"Expected mesh_type 'visual' or 'collision', got {mesh_type!r}.")
    if not transforms:
        return []

    from matplotlib import colormaps

    color_values = np.linspace(0.0, 1.0, len(transforms), dtype=np.float32)
    colors = np.rint(colormaps["cividis"](color_values) * 255.0).astype(np.uint8)
    return [
        TrimeshInstance(
            mesh=_copy_with_color(mesh, color),
            pose=base_pose @ np.asarray(local_pose, dtype=np.float32),
        )
        for color, (mesh, local_pose) in zip(colors, transforms.items(), strict=True)
    ]


def transformed_meshes(instances: list[TrimeshInstance]) -> list[trimesh.Trimesh]:
    """Apply instance transforms and return independent render meshes."""
    if not instances:
        raise ValueError("Expected at least one mesh instance.")
    meshes: list[trimesh.Trimesh] = []
    for instance in instances:
        pose = np.asarray(instance.pose, dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"Expected instance pose shape (4, 4), got {pose.shape}.")
        mesh = instance.mesh.copy()
        mesh.apply_transform(pose)
        meshes.append(mesh)
    return meshes


def look_at(
    camera_position: np.ndarray, target: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """Construct a camera-to-world transform looking at a target."""
    camera_position = np.asarray(camera_position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    if camera_position.shape != (3,) or target.shape != (3,) or up.shape != (3,):
        raise ValueError("Expected camera_position, target, and up to have shape (3,).")
    camera_z = camera_position - target
    z_norm = float(np.linalg.norm(camera_z))
    if z_norm <= 1.0e-8:
        raise ValueError("Camera position must differ from target.")
    camera_z /= z_norm
    camera_x = np.cross(up, camera_z)
    x_norm = float(np.linalg.norm(camera_x))
    if x_norm <= 1.0e-8:
        raise ValueError("Camera up vector must not be parallel to the view direction.")
    camera_x /= x_norm
    camera_y = np.cross(camera_z, camera_x)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = camera_x
    pose[:3, 1] = camera_y
    pose[:3, 2] = camera_z
    pose[:3, 3] = camera_position
    return pose


def render_panel(
    meshes: list[trimesh.Trimesh],
    *,
    camera_pose: np.ndarray,
    xmag: float,
    ymag: float,
    width: int,
    height: int,
) -> Image.Image:
    """Render one orthographic RGB frame with neutral lighting."""
    if not meshes:
        raise ValueError("Expected at least one mesh to render.")
    if xmag <= 0.0 or ymag <= 0.0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Expected positive render dimensions, got {xmag=}, {ymag=}, "
            f"{width=}, {height=}."
        )
    try:
        import pyrender
    except ImportError as error:
        raise ImportError(
            "pyrender is required for rollout rendering. Install DexLink with "
            "the 'visualization' extra."
        ) from error

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
        light_pose[:3, 3] += offset
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
