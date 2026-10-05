from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import trimesh
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class TrimeshInstance:
    mesh: trimesh.Trimesh
    pose: np.ndarray


def _validate_pose(pose: np.ndarray) -> np.ndarray:
    pose_arr = np.asarray(pose, dtype=np.float32)
    if pose_arr.shape != (4, 4):
        raise ValueError(f"Expected pose shape (4, 4), got {pose_arr.shape}.")
    return pose_arr


def _center_mesh_instances(
    mesh_instances: Sequence[TrimeshInstance],
) -> list[trimesh.Trimesh]:
    if len(mesh_instances) == 0:
        raise ValueError("Expected at least one mesh instance to render.")

    transformed_meshes: list[trimesh.Trimesh] = []
    bounds_min = np.full(3, np.inf, dtype=np.float32)
    bounds_max = np.full(3, -np.inf, dtype=np.float32)

    for instance in mesh_instances:
        if not isinstance(instance.mesh, trimesh.Trimesh):
            raise TypeError(
                "Expected each mesh instance to contain a trimesh.Trimesh, got "
                f"{type(instance.mesh)!r}."
            )

        mesh = instance.mesh.copy()
        mesh.apply_transform(_validate_pose(instance.pose))
        transformed_meshes.append(mesh)

        mesh_bounds = np.asarray(mesh.bounds, dtype=np.float32)
        bounds_min = np.minimum(bounds_min, mesh_bounds[0])
        bounds_max = np.maximum(bounds_max, mesh_bounds[1])

    scene_center = 0.5 * (bounds_min + bounds_max)
    for mesh in transformed_meshes:
        mesh.vertices = mesh.vertices - scene_center

    return transformed_meshes


def _camera_pose_from_position(
    camera_position: np.ndarray,
    target_position: np.ndarray = np.zeros(3, dtype=np.float32),
) -> np.ndarray:
    camera_position = np.asarray(camera_position, dtype=np.float32)
    target_position = np.asarray(target_position, dtype=np.float32)

    forward = target_position - camera_position
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1e-8:
        raise ValueError("Camera position must differ from target position.")
    forward = forward / forward_norm

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    camera_z = -forward
    camera_x = np.cross(world_up, camera_z)
    camera_x_norm = np.linalg.norm(camera_x)
    if camera_x_norm <= 1e-8:
        raise ValueError("Failed to construct a valid camera frame.")
    camera_x = camera_x / camera_x_norm
    camera_y = np.cross(camera_z, camera_x)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = camera_x
    pose[:3, 1] = camera_y
    pose[:3, 2] = camera_z
    pose[:3, 3] = camera_position
    return pose


def _opaque_alpha_for_dtype(dtype: np.dtype) -> float | int:
    if np.issubdtype(dtype, np.integer):
        return int(np.iinfo(dtype).max)
    if np.issubdtype(dtype, np.floating):
        return 1.0
    raise TypeError(f"Expected integer or floating color dtype, got {dtype}.")


def _rgba_with_opaque_alpha(colors: np.ndarray, field_name: str) -> np.ndarray:
    color_arr = np.asarray(colors)
    if color_arr.ndim != 2 or color_arr.shape[1] != 4:
        raise ValueError(
            f"Expected {field_name} to have shape (N, 4), got {color_arr.shape}."
        )
    opaque_colors = color_arr.copy()
    opaque_colors[:, 3] = _opaque_alpha_for_dtype(opaque_colors.dtype)
    return opaque_colors


def _make_visuals_opaque(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    opaque_mesh = mesh.copy()
    visual = opaque_mesh.visual

    for field_name in ("face_colors", "vertex_colors"):
        if not hasattr(visual, field_name):
            continue
        colors = np.asarray(getattr(visual, field_name))
        if colors.size == 0:
            continue
        if colors.ndim != 2 or colors.shape[1] not in {3, 4}:
            raise ValueError(
                f"Expected {field_name} to have shape (N, 3) or (N, 4), "
                f"got {colors.shape}."
            )
        if colors.shape[1] == 4:
            setattr(visual, field_name, _rgba_with_opaque_alpha(colors, field_name))

    material = getattr(visual, "material", None)
    if material is not None:
        image = getattr(material, "image", None)
        if image is not None and image.mode == "RGBA":
            opaque_image = image.copy()
            opaque_image.putalpha(255)
            material.image = opaque_image

        for field_name in ("diffuse", "ambient", "specular"):
            if not hasattr(material, field_name):
                continue
            color = np.asarray(getattr(material, field_name))
            if color.shape == (4,):
                opaque_color = color.copy()
                opaque_color[3] = _opaque_alpha_for_dtype(opaque_color.dtype)
                setattr(material, field_name, opaque_color)

    return opaque_mesh


def _pyrender_mesh_from_trimesh(mesh: trimesh.Trimesh, pyrender):
    render_mesh = pyrender.Mesh.from_trimesh(_make_visuals_opaque(mesh), smooth=False)
    for primitive in render_mesh.primitives:
        material = primitive.material
        if material is None:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32),
                metallicFactor=0.0,
                roughnessFactor=0.8,
                alphaMode="OPAQUE",
                doubleSided=True,
            )
            primitive.material = material
            continue

        material.alphaMode = "OPAQUE"
        material.doubleSided = True
        base_color = np.asarray(material.baseColorFactor, dtype=np.float32)
        if base_color.shape != (4,):
            raise ValueError(
                "Expected pyrender material baseColorFactor to have shape (4,), "
                f"got {base_color.shape}."
            )
        base_color[3] = 1.0
        material.baseColorFactor = base_color
    return render_mesh


def _render_view(
    centered_meshes: Sequence[trimesh.Trimesh],
    camera_position: np.ndarray,
    image_width: int,
    image_height: int,
) -> Image.Image:
    try:
        import pyrender
    except ImportError as exc:
        raise ImportError(
            "pyrender is required for render_centered_four_views. "
            "Install it in the active environment before using this utility."
        ) from exc

    scene = pyrender.Scene(
        bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
        ambient_light=np.array([0.25, 0.25, 0.25], dtype=np.float32),
    )

    for mesh in centered_meshes:
        scene.add(_pyrender_mesh_from_trimesh(mesh, pyrender))

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    camera_pose = _camera_pose_from_position(camera_position)
    scene.add(camera, pose=camera_pose)

    light_positions = (
        np.array([1.0, 1.0, 1.4], dtype=np.float32),
        np.array([-1.0, 1.0, 1.2], dtype=np.float32),
        np.array([0.0, -1.2, 1.6], dtype=np.float32),
    )
    for light_position in light_positions:
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
        scene.add(light, pose=_camera_pose_from_position(light_position))

    renderer = pyrender.OffscreenRenderer(
        viewport_width=image_width,
        viewport_height=image_height,
    )
    try:
        color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    return Image.fromarray(color_rgba, mode="RGBA").convert("RGB")


def render_centered_four_views(
    mesh_instances: Sequence[TrimeshInstance],
    image_width: int = 640,
    image_height: int = 480,
    camera_distance_m: float = 1.0,
    camera_elevation_m: float = 0.5,
    grid_columns: int = 2,
) -> Image.Image:
    if image_width <= 0:
        raise ValueError(f"Expected image_width > 0, got {image_width}.")
    if image_height <= 0:
        raise ValueError(f"Expected image_height > 0, got {image_height}.")
    if camera_distance_m <= 0.0:
        raise ValueError(f"Expected camera_distance_m > 0, got {camera_distance_m}.")
    if grid_columns <= 0:
        raise ValueError(f"Expected grid_columns > 0, got {grid_columns}.")

    centered_meshes = _center_mesh_instances(mesh_instances)

    view_positions = (
        (
            "left",
            np.array([0.0, camera_distance_m, camera_elevation_m], dtype=np.float32),
        ),
        (
            "right",
            np.array([0.0, -camera_distance_m, camera_elevation_m], dtype=np.float32),
        ),
        (
            "front",
            np.array([camera_distance_m, 0.0, camera_elevation_m], dtype=np.float32),
        ),
        (
            "back",
            np.array([-camera_distance_m, 0.0, camera_elevation_m], dtype=np.float32),
        ),
    )

    rendered_views = [
        (name, _render_view(centered_meshes, position, image_width, image_height))
        for name, position in view_positions
    ]

    n_views = len(rendered_views)
    n_cols = min(grid_columns, n_views)
    n_rows = int(np.ceil(n_views / n_cols))
    composite = Image.new(
        "RGB",
        (n_cols * image_width, n_rows * image_height),
        color="white",
    )
    draw = ImageDraw.Draw(composite)
    label_margin_px = 12

    for idx, (name, image) in enumerate(rendered_views):
        col = idx % n_cols
        row = idx // n_cols
        offset = (col * image_width, row * image_height)
        composite.paste(image, offset)
        draw.text(
            (offset[0] + label_margin_px, offset[1] + label_margin_px),
            name,
            fill=(0, 0, 0),
        )

    return composite
