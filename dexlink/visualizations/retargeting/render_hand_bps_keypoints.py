from __future__ import annotations

import argparse
import colorsys
import os
from dataclasses import dataclass
from pathlib import Path

# Default to EGL for headless rendering. Override before launching if needed.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from urdfpy import URDF

from dexlink.src.retargeting.data.hand_cfgs.xhand import XHAND_LEFT_CFG, XHAND_RIGHT_CFG
from dexlink.src.retargeting.data.human_hand_kpts import (
    HUMAN_HAND_KPT_SPECS,
    MANO_FINGERTIP_VERTEX_IDS,
    MANO_ORIGINAL_JOINT_MAP,
    load_mano_faces,
    load_mano_model_data,
)
from dexlink.utils.retargeting.path_utils import get_assets_dir, get_tmp_dir, process_urdf
from dexlink.utils.retargeting.render_utils import TrimeshInstance, _center_mesh_instances

# NumPy 2.x compatibility for urdfpy.
if not hasattr(np, "product"):
    np.product = np.prod

XHAND_CFG_BY_SIDE = {
    "left": XHAND_LEFT_CFG,
    "right": XHAND_RIGHT_CFG,
}
XHAND_RENDER_URDF_BY_SIDE = {
    "left": get_assets_dir() / "robots" / "xhand" / "xhand_left_6dof_abs.urdf",
    "right": get_assets_dir() / "robots" / "xhand" / "xhand_right_6dof_abs.urdf",
}
XHAND_CIVIDIS_PALETTE = {
    "palm": np.array([77, 107, 145, 255], dtype=np.uint8),
    "bend": np.array([128, 135, 152, 255], dtype=np.uint8),
    "proximal": np.array([164, 163, 161, 255], dtype=np.uint8),
    "distal": np.array([203, 194, 154, 255], dtype=np.uint8),
    "tip": np.array([247, 230, 123, 255], dtype=np.uint8),
}


@dataclass(frozen=True)
class RenderHandBpsKeypointArgs:
    render_mode: str
    side: str
    output: Path
    width: int
    height: int
    camera_distance: float
    human_x_offset: float
    robot_x_offset: float
    human_scale: float
    robot_scale: float
    hand_alpha: int
    human_kpt_radius: float
    robot_kpt_radius: float
    trim_padding: int


def _parse_args() -> RenderHandBpsKeypointArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Render either canonical MANO/xhand BPS keypoint correspondences or "
            "a clean, palm-facing xhand open pose."
        )
    )
    parser.add_argument(
        "--render-mode",
        choices=("bps-keypoints", "xhand-open-palm"),
        default="bps-keypoints",
    )
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--output",
        type=Path,
        default=get_tmp_dir() / "hand_bps_keypoints.png",
        help="Output PNG path.",
    )
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--camera-distance", type=float, default=0.75)
    parser.add_argument("--human-x-offset", type=float, default=0.16)
    parser.add_argument("--robot-x-offset", type=float, default=-0.16)
    parser.add_argument("--human-scale", type=float, default=1.0)
    parser.add_argument("--robot-scale", type=float, default=1.0)
    parser.add_argument("--hand-alpha", type=int, default=145)
    parser.add_argument("--human-kpt-radius", type=float, default=0.005)
    parser.add_argument("--robot-kpt-radius", type=float, default=0.006)
    parser.add_argument(
        "--trim-padding",
        type=int,
        default=36,
        help="White-background crop padding in pixels. Use 0 for a tight crop.",
    )
    args = parser.parse_args()
    return RenderHandBpsKeypointArgs(**vars(args))


def _validate_render_args(args: RenderHandBpsKeypointArgs) -> None:
    if args.width <= 0:
        raise ValueError(f"Expected --width > 0, got {args.width}.")
    if args.height <= 0:
        raise ValueError(f"Expected --height > 0, got {args.height}.")
    if args.camera_distance <= 0.0:
        raise ValueError(f"Expected --camera-distance > 0, got {args.camera_distance}.")
    for name, value in (
        ("--human-scale", args.human_scale),
        ("--robot-scale", args.robot_scale),
        ("--human-kpt-radius", args.human_kpt_radius),
        ("--robot-kpt-radius", args.robot_kpt_radius),
    ):
        if value <= 0.0:
            raise ValueError(f"Expected {name} > 0, got {value}.")
    if args.hand_alpha < 0 or args.hand_alpha > 255:
        raise ValueError(f"Expected --hand-alpha in [0, 255], got {args.hand_alpha}.")
    if args.trim_padding < 0:
        raise ValueError(f"Expected --trim-padding >= 0, got {args.trim_padding}.")


def _mano_joint_positions(side: str, vertices: np.ndarray) -> dict[str, np.ndarray]:
    data = load_mano_model_data(side)
    j_regressor = data["J_regressor"]
    if hasattr(j_regressor, "toarray"):
        j_regressor = j_regressor.toarray()
    joints = np.asarray(j_regressor, dtype=np.float64) @ vertices
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError(
            f"Expected MANO canonical joints shape (J, 3), got {joints.shape}."
        )

    positions = {
        name: joints[joint_idx] for name, joint_idx in MANO_ORIGINAL_JOINT_MAP.items()
    }
    for name, vertex_idx in MANO_FINGERTIP_VERTEX_IDS.items():
        if vertex_idx >= vertices.shape[0]:
            raise ValueError(
                f"MANO fingertip vertex {vertex_idx} for {name!r} is out of range "
                f"for vertex count {vertices.shape[0]}."
            )
        positions[name] = vertices[vertex_idx]
    return positions


def _human_keypoint_positions(
    side: str,
    vertices: np.ndarray,
    keypoint_names: list[str],
) -> dict[str, np.ndarray]:
    positions = _mano_joint_positions(side, vertices)
    for name, start_name, end_name, fraction in HUMAN_HAND_KPT_SPECS:
        if start_name not in positions or end_name not in positions:
            raise ValueError(
                f"Cannot compute human keypoint {name!r}: expected start/end "
                f"positions {start_name!r}/{end_name!r} in MANO joint positions."
            )
        start = positions[start_name]
        end = positions[end_name]
        positions[name] = start + float(fraction) * (end - start)

    for name in keypoint_names:
        if name not in positions:
            raise ValueError(
                f"Human keypoint {name!r} is missing from MANO joints, fingertips, "
                "and semantic interpolation specs."
            )

    return {name: positions[name] for name in keypoint_names}


def _human_frame_transform(side: str, x_offset: float, scale: float) -> np.ndarray:
    if side == "right":
        rot = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    elif side == "left":
        rot = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = scale * rot
    pose[:3, 3] = np.array([x_offset, 0.0, 0.0], dtype=np.float32)
    return pose


def _robot_frame_transform(side: str, x_offset: float, scale: float) -> np.ndarray:
    if side == "right":
        rot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
    elif side == "left":
        rot = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Expected side to be 'left' or 'right', got {side!r}.")

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = scale * rot
    pose[:3, 3] = np.array([x_offset, 0.0, 0.0], dtype=np.float32)
    return pose


def _transform_points(
    points: dict[str, np.ndarray],
    pose: np.ndarray,
) -> dict[str, np.ndarray]:
    transformed = {}
    for name, point in points.items():
        point_arr = np.asarray(point, dtype=np.float32)
        if point_arr.shape != (3,):
            raise ValueError(
                f"Expected keypoint {name!r} shape (3,), got {point_arr.shape}."
            )
        point_h = np.ones(4, dtype=np.float32)
        point_h[:3] = point_arr
        transformed[name] = (pose @ point_h)[:3]
    return transformed


def _sphere_mesh(
    center: np.ndarray, radius: float, color_rgba: np.ndarray
) -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere.vertices += np.asarray(center, dtype=np.float32)
    color = np.asarray(color_rgba, dtype=np.uint8).reshape(4)
    sphere.visual.face_colors = np.tile(color[None, :], (len(sphere.faces), 1))
    return sphere


def _keypoint_colors(keypoint_names: list[str]) -> dict[str, np.ndarray]:
    colors = {}
    for idx, name in enumerate(keypoint_names):
        hue = idx / float(len(keypoint_names))
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 0.92)
        colors[name] = np.array(
            [round(255.0 * red), round(255.0 * green), round(255.0 * blue), 255],
            dtype=np.uint8,
        )
    return colors


def _make_colored_mesh(
    mesh: trimesh.Trimesh, color_rgba: np.ndarray
) -> trimesh.Trimesh:
    colored = mesh.copy()
    color = np.asarray(color_rgba, dtype=np.uint8).reshape(4)
    colored.visual.face_colors = np.tile(color[None, :], (len(colored.faces), 1))
    return colored


def _transformed_bounds(
    mesh_instances: list[TrimeshInstance],
) -> tuple[np.ndarray, np.ndarray]:
    if len(mesh_instances) == 0:
        raise ValueError("Expected at least one mesh instance for bounds.")

    bounds_min = np.full(3, np.inf, dtype=np.float32)
    bounds_max = np.full(3, -np.inf, dtype=np.float32)
    for instance in mesh_instances:
        mesh = instance.mesh.copy()
        mesh.apply_transform(instance.pose)
        mesh_bounds = np.asarray(mesh.bounds, dtype=np.float32)
        bounds_min = np.minimum(bounds_min, mesh_bounds[0])
        bounds_max = np.maximum(bounds_max, mesh_bounds[1])
    return bounds_min, bounds_max


def _camera_pose_from_position(camera_position: np.ndarray) -> np.ndarray:
    camera_position = np.asarray(camera_position, dtype=np.float32)
    target_position = np.zeros(3, dtype=np.float32)
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


def _mesh_base_color(mesh: trimesh.Trimesh) -> np.ndarray:
    face_colors = np.asarray(mesh.visual.face_colors)
    if face_colors.ndim != 2 or face_colors.shape[1] != 4 or face_colors.shape[0] == 0:
        raise ValueError(
            "Expected mesh.visual.face_colors to have non-empty shape (N, 4), "
            f"got {face_colors.shape}."
        )
    return face_colors[0].astype(np.float32) / 255.0


def _pyrender_mesh_preserve_alpha(mesh: trimesh.Trimesh, pyrender):
    base_color = _mesh_base_color(mesh)
    alpha_mode = "BLEND" if float(base_color[3]) < 1.0 else "OPAQUE"
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=base_color,
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode=alpha_mode,
        doubleSided=True,
    )
    return pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)


def _render_view_preserve_alpha(
    centered_meshes: list[trimesh.Trimesh],
    camera_position: np.ndarray,
    image_width: int,
    image_height: int,
) -> Image.Image:
    try:
        import pyrender
    except ImportError as exc:
        raise ImportError(
            "pyrender is required for render_hand_bps_keypoints.py."
        ) from exc

    scene = pyrender.Scene(
        bg_color=np.array([255, 255, 255, 255], dtype=np.uint8),
        ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
    )

    opaque_meshes = []
    transparent_meshes = []
    for mesh in centered_meshes:
        if float(_mesh_base_color(mesh)[3]) < 1.0:
            transparent_meshes.append(mesh)
        else:
            opaque_meshes.append(mesh)

    for mesh in opaque_meshes + transparent_meshes:
        scene.add(_pyrender_mesh_preserve_alpha(mesh, pyrender))

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    scene.add(camera, pose=_camera_pose_from_position(camera_position))

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


def _trim_white_background(image: Image.Image, padding_px: int) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    foreground = np.any(rgb < 250, axis=2)
    if not np.any(foreground):
        raise ValueError("Rendered image is blank; cannot trim white background.")

    ys, xs = np.nonzero(foreground)
    x0 = max(int(np.min(xs)) - padding_px, 0)
    x1 = min(int(np.max(xs)) + padding_px + 1, image.width)
    y0 = max(int(np.min(ys)) - padding_px, 0)
    y1 = min(int(np.max(ys)) + padding_px + 1, image.height)
    return image.crop((x0, y0, x1, y1))


def _human_mesh_instance(
    side: str,
    pose: np.ndarray,
    hand_alpha: int,
) -> TrimeshInstance:
    data = load_mano_model_data(side)
    vertices = np.asarray(data["v_template"], dtype=np.float32)
    faces = load_mano_faces(side)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    color = np.array([218, 204, 188, hand_alpha], dtype=np.uint8)
    return TrimeshInstance(mesh=_make_colored_mesh(mesh, color), pose=pose)


def _robot_mesh_instances(
    *,
    side: str,
    pose: np.ndarray,
    hand_alpha: int,
) -> list[TrimeshInstance]:
    cfg = XHAND_CFG_BY_SIDE[side]
    urdf = URDF.load(str(process_urdf(cfg["urdf_path"])))
    fk = urdf.visual_trimesh_fk(cfg=cfg["open_qpos"])
    color = np.array([185, 195, 205, hand_alpha], dtype=np.uint8)
    return [
        TrimeshInstance(
            mesh=_make_colored_mesh(mesh, color),
            pose=pose @ np.asarray(local_pose, dtype=np.float32),
        )
        for mesh, local_pose in fk.items()
    ]


def _xhand_cividis_color(link_name: str, hand_alpha: int) -> np.ndarray:
    if link_name.endswith("_hand_link"):
        palette_name = "palm"
    elif link_name.endswith("_bend_link"):
        palette_name = "bend"
    elif link_name.endswith("_link1"):
        palette_name = "proximal"
    elif link_name.endswith("_link2"):
        palette_name = "distal"
    elif link_name.endswith("_tip"):
        palette_name = "tip"
    else:
        raise ValueError(
            "Expected an xhand visual link ending in '_hand_link', '_bend_link', "
            f"'_link1', '_link2', or '_tip', got {link_name!r}."
        )

    color = XHAND_CIVIDIS_PALETTE[palette_name].copy()
    color[3] = hand_alpha
    return color


def _xhand_open_palm_mesh_instances(
    *,
    side: str,
    pose: np.ndarray,
    hand_alpha: int,
) -> list[TrimeshInstance]:
    cfg = XHAND_CFG_BY_SIDE[side]
    urdf_path = XHAND_RENDER_URDF_BY_SIDE[side]
    urdf = URDF.load(str(urdf_path))
    link_fk = urdf.link_fk(cfg=cfg["open_qpos"])

    mesh_instances = []
    for link in urdf.links:
        if not link.visuals:
            continue
        if len(link.visuals) != 1:
            raise ValueError(
                f"Expected exactly one visual on xhand link {link.name!r}, "
                f"got {len(link.visuals)} in {urdf_path}."
            )
        visual = link.visuals[0]
        meshes = visual.geometry.meshes
        if len(meshes) != 1:
            raise ValueError(
                f"Expected exactly one visual mesh on xhand link {link.name!r}, "
                f"got {len(meshes)} in {urdf_path}."
            )
        mesh_instances.append(
            TrimeshInstance(
                mesh=_make_colored_mesh(
                    meshes[0], _xhand_cividis_color(link.name, hand_alpha)
                ),
                pose=(
                    pose
                    @ np.asarray(link_fk[link], dtype=np.float32)
                    @ np.asarray(visual.origin, dtype=np.float32)
                ),
            )
        )

    if len(mesh_instances) == 0:
        raise ValueError(
            f"Expected visual meshes in xhand URDF, found none: {urdf_path}"
        )
    return mesh_instances


def _robot_keypoint_positions(side: str) -> dict[str, np.ndarray]:
    cfg = XHAND_CFG_BY_SIDE[side]
    urdf = URDF.load(str(process_urdf(cfg["urdf_path"])))
    link_fk = urdf.link_fk(cfg=cfg["open_qpos"])
    link_positions = {
        link.name: np.asarray(pose[:3, 3], dtype=np.float32)
        for link, pose in link_fk.items()
    }

    keypoint_positions = {}
    for human_name, robot_link_name in cfg["human_robot_kpt_map"].items():
        if robot_link_name not in link_positions:
            raise ValueError(
                f"Robot keypoint map entry {human_name!r} references missing link "
                f"{robot_link_name!r} in {cfg['urdf_path']}."
            )
        keypoint_positions[human_name] = link_positions[robot_link_name]
    return keypoint_positions


def _build_scene(args: RenderHandBpsKeypointArgs) -> list[TrimeshInstance]:
    human_pose = _human_frame_transform(
        args.side, args.human_x_offset, args.human_scale
    )
    robot_pose = _robot_frame_transform(
        args.side, args.robot_x_offset, args.robot_scale
    )
    cfg = XHAND_CFG_BY_SIDE[args.side]
    keypoint_names = list(cfg["human_robot_kpt_map"].keys())

    human_mesh_instance = _human_mesh_instance(args.side, human_pose, args.hand_alpha)
    robot_mesh_instances = _robot_mesh_instances(
        side=args.side,
        pose=robot_pose,
        hand_alpha=args.hand_alpha,
    )
    human_bounds_min, human_bounds_max = _transformed_bounds([human_mesh_instance])
    robot_bounds_min, robot_bounds_max = _transformed_bounds(robot_mesh_instances)
    human_center_z = 0.5 * (human_bounds_min[2] + human_bounds_max[2])
    robot_center_z = 0.5 * (robot_bounds_min[2] + robot_bounds_max[2])
    robot_pose = robot_pose.copy()
    robot_pose[2, 3] += float(human_center_z - robot_center_z)
    robot_mesh_instances = _robot_mesh_instances(
        side=args.side,
        pose=robot_pose,
        hand_alpha=args.hand_alpha,
    )

    data = load_mano_model_data(args.side)
    human_vertices = np.asarray(data["v_template"], dtype=np.float32)
    human_kpts = _transform_points(
        _human_keypoint_positions(args.side, human_vertices, keypoint_names),
        human_pose,
    )
    robot_kpts = _transform_points(_robot_keypoint_positions(args.side), robot_pose)
    colors = _keypoint_colors(keypoint_names)

    mesh_instances = [human_mesh_instance]
    mesh_instances.extend(robot_mesh_instances)

    identity = np.eye(4, dtype=np.float32)
    for name in keypoint_names:
        mesh_instances.append(
            TrimeshInstance(
                mesh=_sphere_mesh(
                    human_kpts[name], args.human_kpt_radius, colors[name]
                ),
                pose=identity,
            )
        )
        mesh_instances.append(
            TrimeshInstance(
                mesh=_sphere_mesh(
                    robot_kpts[name], args.robot_kpt_radius, colors[name]
                ),
                pose=identity,
            )
        )

    return mesh_instances


def _build_xhand_open_palm_scene(
    args: RenderHandBpsKeypointArgs,
) -> list[TrimeshInstance]:
    robot_pose = _robot_frame_transform(args.side, 0.0, args.robot_scale)
    return _xhand_open_palm_mesh_instances(
        side=args.side,
        pose=robot_pose,
        hand_alpha=args.hand_alpha,
    )


def main() -> None:
    args = _parse_args()
    _validate_render_args(args)

    if args.render_mode == "bps-keypoints":
        mesh_instances = _build_scene(args)
    elif args.render_mode == "xhand-open-palm":
        mesh_instances = _build_xhand_open_palm_scene(args)
    else:
        raise ValueError(
            "Expected --render-mode to be 'bps-keypoints' or 'xhand-open-palm', "
            f"got {args.render_mode!r}."
        )
    centered_meshes = _center_mesh_instances(mesh_instances)
    image = _render_view_preserve_alpha(
        centered_meshes,
        camera_position=np.array([0.0, args.camera_distance, 0.04], dtype=np.float32),
        image_width=args.width,
        image_height=args.height,
    )

    image = _trim_white_background(image, args.trim_padding)
    if args.render_mode == "bps-keypoints":
        draw = ImageDraw.Draw(image)
        draw.text((12, 10), f"MANO {args.side} hand", fill=(0, 0, 0))
        draw.text(
            (image.width // 2 + 12, 10),
            f"xhand {args.side} hand",
            fill=(0, 0, 0),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved {args.render_mode} render to {args.output.resolve()}")


if __name__ == "__main__":
    main()
