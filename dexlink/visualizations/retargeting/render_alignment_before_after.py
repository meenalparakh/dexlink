from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Default to EGL for headless rendering. Override before launching if needed.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from dexlink.scripts.retargeting.align_experiment_category_transforms import (
    ExperimentPair,
    _is_urdf_path,
    _iter_pairs,
    _load_arctic_source_mesh_and_pose,
    _load_saved_values,
    _load_target_mesh,
    _load_target_mesh_from_demo,
    _materialize_urdf_visualization_obj,
    _materialize_urdfpy_compatible_urdf,
    _relative_or_abs_path,
    _rotation_path,
    _scale_path,
    _transformed_target_mesh,
)
from dexlink.src.retargeting.optimization.oakink_processor import OakInkProcessor
from dexlink.utils.retargeting.path_utils import get_logs_dir, get_tmp_dir
from dexlink.utils.retargeting.render_utils import (
    TrimeshInstance,
    _center_mesh_instances,
    _render_view,
)
from dexlink.visualizations.rl.render_paper_retargeting_figures import (
    DEFAULT_EXPORTS_ROOT,
    RenderPaperRetargetingArgs,
    _build_target_instances,
    _internal_cache_root,
    _load_sequence_motion,
    _render_meshes_with_own_camera,
    _transformed_meshes,
    discover_jobs,
    render_retargeting_panels,
)

# NumPy 2.x compatibility for urdfpy.
if not hasattr(np, "product"):
    np.product = np.prod


@dataclass(frozen=True)
class RenderAlignmentArgs:
    category: str
    source_index: int
    target_index: int
    output: Path | None
    hand: str
    object_width: int
    object_height: int
    camera_distance: float
    camera_elevation: float
    trim_padding: int
    label_height: int
    template_pipeline: str
    template_task: str | None
    template_retargeting_id: str | None
    processor_cache_root: Path
    share_export_root: Path


@dataclass(frozen=True)
class AlignmentAssets:
    pair: ExperimentPair
    source_mesh: trimesh.Trimesh
    source_pose: np.ndarray
    target_mesh: trimesh.Trimesh
    target_pose: np.ndarray
    euler_xyz: np.ndarray
    scale: float


def _parse_args() -> RenderAlignmentArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Render source/target object frames before and after saved experiment "
            "category rotation alignment."
        )
    )
    parser.add_argument("--category", required=True, type=str)
    parser.add_argument("--source-index", required=True, type=int)
    parser.add_argument("--target-index", required=True, type=int)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--hand", default="right", choices=("left", "right"))
    parser.add_argument("--object-width", default=260, type=int)
    parser.add_argument("--object-height", default=220, type=int)
    parser.add_argument("--camera-distance", default=0.35, type=float)
    parser.add_argument("--camera-elevation", default=0.08, type=float)
    parser.add_argument("--trim-padding", default=8, type=int)
    parser.add_argument("--label-height", default=22, type=int)
    parser.add_argument(
        "--template-pipeline",
        default="v2_03",
        type=str,
        help="Pipeline containing --template-task for the optional hand-object panel.",
    )
    parser.add_argument(
        "--template-task",
        default=None,
        type=str,
        help=(
            "Optional task id whose retargeting is rendered as a third panel: "
            "human hand with source object and robot hand with target object."
        ),
    )
    parser.add_argument(
        "--template-retargeting-id",
        default=None,
        type=str,
        help=(
            "Retargeting id to use for --template-task. If omitted, the first "
            "retargeting job discovered for the task is used."
        ),
    )
    parser.add_argument(
        "--processor-cache-root",
        default=get_tmp_dir() / "alignment_before_after_cache",
        type=Path,
        help="Writable cache root for OakInk processed mesh exports.",
    )
    parser.add_argument(
        "--share-export-root",
        default=Path("supplementary_files/share_export"),
        type=Path,
        help=(
            "Optional share-export root used to load human/source template "
            "renders from 01_source_templates."
        ),
    )
    args = parser.parse_args()
    return RenderAlignmentArgs(**vars(args))


def _validate_args(args: RenderAlignmentArgs) -> None:
    for name, value in (
        ("--object-width", args.object_width),
        ("--object-height", args.object_height),
        ("--label-height", args.label_height),
    ):
        if value <= 0:
            raise ValueError(f"Expected {name} > 0, got {value}.")
    if args.camera_distance <= 0.0:
        raise ValueError(f"Expected --camera-distance > 0, got {args.camera_distance}.")
    if args.trim_padding < 0:
        raise ValueError(f"Expected --trim-padding >= 0, got {args.trim_padding}.")
    if args.template_retargeting_id is not None and args.template_task is None:
        raise ValueError("--template-retargeting-id requires --template-task.")


def _select_pair(args: RenderAlignmentArgs) -> ExperimentPair:
    pairs = _iter_pairs(
        category=args.category,
        source_index=args.source_index,
        target_index=args.target_index,
    )
    if len(pairs) != 1:
        raise ValueError(
            "Expected exactly one source-target pair from "
            f"category={args.category!r}, source_index={args.source_index}, "
            f"target_index={args.target_index}; got {len(pairs)}."
        )
    return pairs[0]


def _load_alignment_assets(
    pair: ExperimentPair, args: RenderAlignmentArgs
) -> AlignmentAssets:
    source_demo_path = _relative_or_abs_path(pair.source.demo_path)
    if not source_demo_path.exists():
        raise FileNotFoundError(f"Source demo not found: {source_demo_path}")

    target_path = (
        None
        if pair.target.mesh_path is None
        else _relative_or_abs_path(pair.target.mesh_path)
    )
    target_demo_path = (
        None
        if pair.target.target_demo_path is None
        else _relative_or_abs_path(pair.target.target_demo_path)
    )
    if target_path is None and target_demo_path is None:
        raise ValueError(
            f"Expected {pair.category}/{pair.pair_id} to define a mesh_path "
            "or target_demo_path."
        )
    if target_path is not None and not target_path.exists():
        raise FileNotFoundError(f"Target object not found: {target_path}")
    if target_demo_path is not None and not target_demo_path.exists():
        raise FileNotFoundError(f"Target demo not found: {target_demo_path}")

    is_urdf_target = _is_urdf_path(target_path)
    target_urdf_path = (
        None
        if not is_urdf_target
        else _materialize_urdfpy_compatible_urdf(pair, target_path)
    )

    processor = None
    if pair.source.dataset == "arctic":
        if target_demo_path is not None:
            raise ValueError(
                f"ARCTIC transform render for {pair.pair_id} does not support "
                "`target_demo_path`; define a target mesh_path instead."
            )
        source_mesh, source_pose = _load_arctic_source_mesh_and_pose(
            pair=pair,
            source_demo_path=source_demo_path,
            target_path=None if target_path is None or is_urdf_target else target_path,
        )
    elif pair.source.dataset == "oakink":
        processor = OakInkProcessor(
            original_demo_path=str(source_demo_path),
            target_obj_path=(
                None if target_path is None or is_urdf_target else str(target_path)
            ),
            target_urdf=None if target_urdf_path is None else str(target_urdf_path),
            target_demo_path=(
                None if target_demo_path is None else str(target_demo_path)
            ),
            oakink_hand_side=args.hand,
            run_dir=args.processor_cache_root / pair.category,
        )
        source_mesh = processor._load_mesh(processor.obj_mesh_path)
        source_pose = processor.get_object_base_pose(obj_type="original")
    else:
        raise ValueError(
            f"Unsupported source dataset {pair.source.dataset!r}; expected "
            "'oakink' or 'arctic'."
        )

    if target_path is None:
        if processor is None:
            raise ValueError(
                f"{pair.pair_id} uses an ARCTIC source with target_demo_path; "
                "ARCTIC transform render requires a target mesh_path."
            )
        target_mesh = _load_target_mesh_from_demo(processor)
    elif is_urdf_target:
        target_mesh = _materialize_urdf_visualization_obj(pair, target_path)
    else:
        target_mesh = _load_target_mesh(target_path)

    euler_xyz, scale = _load_saved_values(
        pair,
        source_mesh=source_mesh,
        target_mesh=target_mesh,
    )
    return AlignmentAssets(
        pair=pair,
        source_mesh=source_mesh,
        source_pose=source_pose,
        target_mesh=target_mesh,
        target_pose=source_pose.copy(),
        euler_xyz=euler_xyz,
        scale=scale,
    )


def _mesh_with_color(
    mesh: trimesh.Trimesh, rgba: tuple[int, int, int, int]
) -> trimesh.Trimesh:
    colored = mesh.copy()
    color = np.asarray(rgba, dtype=np.uint8)
    colored.visual.face_colors = np.tile(color[None, :], (len(colored.faces), 1))
    return colored


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


def _render_object(
    *,
    mesh: trimesh.Trimesh,
    pose: np.ndarray,
    rgba: tuple[int, int, int, int],
    width: int,
    height: int,
    camera_distance: float,
    camera_elevation: float,
    trim_padding: int,
) -> Image.Image:
    mesh_instances = [
        TrimeshInstance(
            mesh=_mesh_with_color(mesh, rgba),
            pose=np.asarray(pose, dtype=np.float32),
        )
    ]
    rendered = _render_view(
        _center_mesh_instances(mesh_instances),
        camera_position=np.array(
            [camera_distance, 0.0, camera_elevation],
            dtype=np.float32,
        ),
        image_width=width,
        image_height=height,
    )
    return _trim_white_background(rendered, trim_padding)


def _fit_on_canvas(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.width > width or image.height > height:
        scale = min(width / image.width, height / image.height)
        new_size = (
            max(int(round(image.width * scale)), 1),
            max(int(round(image.height * scale)), 1),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), color="white")
    offset = ((width - image.width) // 2, (height - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def _resize_to_limit(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.width <= width and image.height <= height:
        return image
    scale = min(width / image.width, height / image.height)
    new_size = (
        max(int(round(image.width * scale)), 1),
        max(int(round(image.height * scale)), 1),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _labeled_object_canvas(
    image: Image.Image,
    label: str,
    width: int,
    height: int,
    label_height: int,
) -> Image.Image:
    image = _resize_to_limit(image, width, height)
    label_pad_x = 8
    label_width = max(image.width, 120)
    canvas = Image.new("RGB", (label_width, image.height + label_height), color="white")
    draw = ImageDraw.Draw(canvas)
    draw.text((label_pad_x, 5), label, fill=(0, 0, 0))
    canvas.paste(image, ((canvas.width - image.width) // 2, label_height))
    return canvas


def _concat_horizontal(images: list[Image.Image], gutter: int) -> Image.Image:
    if len(images) == 0:
        raise ValueError("Expected at least one image to concatenate.")
    height = max(image.height for image in images)
    width = sum(image.width for image in images) + gutter * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), color="white")
    x = 0
    for image in images:
        canvas.paste(image, (x, (height - image.height) // 2))
        x += image.width + gutter
    return canvas


def _panel_with_title(
    title: str, source_image: Image.Image, target_image: Image.Image
) -> Image.Image:
    content = _concat_horizontal([source_image, target_image], gutter=8)
    title_height = 26
    border_padding = 8
    panel = Image.new(
        "RGB",
        (
            content.width + 2 * border_padding,
            content.height + title_height + 2 * border_padding,
        ),
        color="white",
    )
    draw = ImageDraw.Draw(panel)
    draw.rectangle(
        (0, 0, panel.width - 1, panel.height - 1),
        outline=(30, 30, 30),
        width=2,
    )
    draw.text((border_padding, 6), title, fill=(0, 0, 0))
    panel.paste(content, (border_padding, title_height + border_padding))
    return panel


def _retargeting_render_args(args: RenderAlignmentArgs) -> RenderPaperRetargetingArgs:
    if args.template_task is None:
        raise ValueError("Expected --template-task before building retargeting args.")
    return RenderPaperRetargetingArgs(
        exports_root=DEFAULT_EXPORTS_ROOT,
        output_root=(
            args.processor_cache_root
            / "alignment_before_after_figures"
            / "_retargeting_cache"
        ),
        pipelines=(args.template_pipeline,),
        tasks=(args.template_task,),
        all_tasks=False,
        width=args.object_width,
        height=args.object_height,
        camera_distance=args.camera_distance,
        camera_elevation=args.camera_elevation,
        mesh_type="visual",
        pyopengl_platform=None,
        mano_assets_root=None,
        oakink_smplx_model_path=None,
        allow_obj_source_fallback=False,
        save_panels=False,
        skip_existing=False,
        dry_run=False,
        geometry_robustness_grid=False,
        geometry_robustness_output=None,
        geometry_robustness_manifest=None,
        v2_03_retargeting_grid=False,
        v2_03_grid_output=None,
        v2_03_grid_manifest=None,
        v2_03_grid_cols=4,
        v2_03_motion_strips=False,
        v2_03_motion_strip_output_dir=None,
        v2_03_motion_strip_manifest=None,
        v2_03_motion_strip_frame_step=1,
        v2_03_template_pair_grid=False,
        v2_03_template_pair_output_dir=None,
        v2_03_template_pair_grid_output=None,
        v2_03_template_pair_manifest=None,
        v2_03_template_pair_gap=0,
        v2_03_template_pair_per_task=False,
        grid_gap=8,
        grid_trim=True,
        grid_trim_padding=args.trim_padding,
        grid_trim_threshold=248,
    )


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _load_share_stage_instances(stage_dir: Path) -> list[TrimeshInstance]:
    manifest_path = stage_dir / "stage_manifest.json"
    poses_path = stage_dir / "poses.npz"
    if not manifest_path.is_file() or not poses_path.is_file():
        raise FileNotFoundError(f"Expected share-export stage files in {stage_dir}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(poses_path, allow_pickle=True) as data:
        poses = np.asarray(data["poses"], dtype=np.float32)
    if poses.ndim != 4 or poses.shape[0] < 1 or poses.shape[2:] != (4, 4):
        raise ValueError(
            f"Expected poses shape (T, N, 4, 4) in {poses_path}, got {poses.shape}."
        )
    elements = manifest.get("elements")
    if not isinstance(elements, list) or poses.shape[1] != len(elements):
        raise ValueError(
            f"{manifest_path}: elements must match poses element count "
            f"{poses.shape[1]}."
        )
    instances: list[TrimeshInstance] = []
    for elem_idx, element in enumerate(elements):
        mesh_rel = element.get("mesh")
        if not isinstance(mesh_rel, str):
            raise ValueError(f"{manifest_path}: element {elem_idx} missing mesh path.")
        mesh_path = stage_dir / mesh_rel
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                raise ValueError(f"Share-export source mesh is empty: {mesh_path}")
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected Trimesh from {mesh_path}, got {type(mesh)!r}.")
        color = np.asarray(
            element.get("color_rgba", [180, 180, 180, 255]), dtype=np.uint8
        )
        if color.shape != (4,):
            raise ValueError(f"{manifest_path}: invalid color_rgba for {mesh_path}.")
        mesh.visual.face_colors = np.tile(color[None, :], (len(mesh.faces), 1))
        instances.append(
            TrimeshInstance(
                mesh=mesh,
                pose=np.asarray(poses[0, elem_idx], dtype=np.float32),
            )
        )
    return instances


def _render_share_stage_image(
    stage_dir: Path,
    args: RenderAlignmentArgs,
) -> Image.Image:
    instances = _load_share_stage_instances(stage_dir)
    rendered = _render_view(
        _center_mesh_instances(instances),
        camera_position=np.array(
            [args.camera_distance, 0.0, args.camera_elevation],
            dtype=np.float32,
        ),
        image_width=args.object_width,
        image_height=args.object_height,
    )
    return _trim_white_background(rendered, args.trim_padding)


def _share_source_stage_dir(
    args: RenderAlignmentArgs, retargeting_id: str
) -> Path | None:
    if args.template_task is None:
        return None
    candidates = [
        args.share_export_root,
        Path("supplementary_files/share_export_rl"),
    ]
    rel_stage = (
        Path("tasks")
        / args.template_task
        / "stages"
        / "01_source_templates"
        / _slug(retargeting_id)
    )
    for root in candidates:
        stage_dir = root / rel_stage
        if (stage_dir / "stage_manifest.json").is_file():
            return stage_dir
    return None


def _share_template_stage_dir(
    args: RenderAlignmentArgs,
    *,
    stage_name: str,
    retargeting_id: str,
) -> Path | None:
    if args.template_task is None:
        return None
    candidates = [
        args.share_export_root,
        Path("supplementary_files/share_export_rl"),
    ]
    rel_stage = (
        Path("tasks")
        / args.template_task
        / "stages"
        / stage_name
        / _slug(retargeting_id)
    )
    for root in candidates:
        stage_dir = root / rel_stage
        if (stage_dir / "stage_manifest.json").is_file():
            return stage_dir
    return None


def _share_retargeted_stage_dir(
    args: RenderAlignmentArgs,
    retargeting_id: str,
) -> Path | None:
    return _share_template_stage_dir(
        args,
        stage_name="02_retargeted_templates",
        retargeting_id=retargeting_id,
    )


def _target_object_index_from_share_stage(stage_dir: Path) -> int:
    manifest_path = stage_dir / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elements = manifest.get("elements")
    if not isinstance(elements, list):
        raise ValueError(f"{manifest_path}: expected elements list.")
    object_color = np.asarray([100, 149, 237, 255], dtype=np.uint8)
    matches = [
        idx
        for idx, element in enumerate(elements)
        if np.array_equal(
            np.asarray(element.get("color_rgba"), dtype=np.uint8),
            object_color,
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{manifest_path}: expected exactly one target object element with "
            f"color {object_color.tolist()}, got {matches}."
        )
    return matches[0]


def _target_object_pose_from_motion(
    motion: dict[str, np.ndarray], frame_idx: int
) -> np.ndarray:
    if "scene_object_pos" in motion:
        raise ValueError(
            "Aligning the third-panel target object to panel 2 currently supports "
            "single-object retargeting motions only. Multi-object scene motions "
            "need an explicit scene object id."
        )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = (
        R.from_quat(np.roll(np.asarray(motion["asset_quat"][frame_idx]), -1))
        .as_matrix()
        .astype(np.float32)
    )
    pose[:3, 3] = np.asarray(motion["asset_pos"][frame_idx], dtype=np.float32)
    return pose


def _render_aligned_share_retargeted_stage_image(
    *,
    stage_dir: Path,
    desired_target_pose: np.ndarray,
    args: RenderAlignmentArgs,
) -> Image.Image:
    instances = _load_share_stage_instances(stage_dir)
    object_idx = _target_object_index_from_share_stage(stage_dir)
    object_pose = instances[object_idx].pose
    rebase = np.asarray(desired_target_pose, dtype=np.float32) @ np.linalg.inv(
        object_pose
    ).astype(np.float32)
    rebased_instances = [
        TrimeshInstance(mesh=instance.mesh, pose=rebase @ instance.pose)
        for instance in instances
    ]
    rendered = _render_view(
        _center_mesh_instances(rebased_instances),
        camera_position=np.array(
            [args.camera_distance, 0.0, args.camera_elevation],
            dtype=np.float32,
        ),
        image_width=args.object_width,
        image_height=args.object_height,
    )
    return _trim_white_background(rendered, args.trim_padding)


def _render_aligned_retargeted_target_image(
    *,
    job,
    render_args: RenderPaperRetargetingArgs,
    desired_target_pose: np.ndarray,
) -> Image.Image:
    motion = _load_sequence_motion(job.motion_path)
    n_frames = int(np.asarray(motion["asset_pos"]).shape[0])
    if n_frames <= 0:
        raise ValueError(f"{job.motion_path}: expected at least one frame.")
    frame_idx = n_frames - 1
    retargeted_target_pose = _target_object_pose_from_motion(motion, frame_idx)
    rebase = np.asarray(desired_target_pose, dtype=np.float32) @ np.linalg.inv(
        retargeted_target_pose
    ).astype(np.float32)
    target_instances = _build_target_instances(
        motion=motion,
        frame_idx=frame_idx,
        mesh_type=render_args.mesh_type,
        robot_model_arg="auto",
        urdf_cache_root=_internal_cache_root(render_args) / "_urdf_cache",
    )
    rebased_instances = [
        TrimeshInstance(mesh=instance.mesh, pose=rebase @ instance.pose)
        for instance in target_instances
    ]
    return _render_meshes_with_own_camera(
        _transformed_meshes(rebased_instances),
        render_args,
    )


def _render_hand_object_panel(
    args: RenderAlignmentArgs,
    *,
    desired_target_pose: np.ndarray,
    target_alignment_euler_xyz: np.ndarray,
) -> Image.Image:
    render_args = _retargeting_render_args(args)
    jobs = [job for job in discover_jobs(render_args) if job.kind == "retargeting"]
    if args.template_retargeting_id is not None:
        jobs = [
            job for job in jobs if job.retargeting_id == args.template_retargeting_id
        ]
    if len(jobs) != 1:
        raise ValueError(
            "Expected exactly one retargeting job for the hand-object panel; got "
            f"{len(jobs)} for task={args.template_task!r}, "
            f"retargeting_id={args.template_retargeting_id!r}."
        )
    retargeting_id = jobs[0].retargeting_id
    share_stage_dir = _share_source_stage_dir(args, retargeting_id)
    aligned_target_pose = _target_alignment_pose(
        desired_target_pose,
        target_alignment_euler_xyz,
    )
    share_retargeted_stage_dir = _share_retargeted_stage_dir(args, retargeting_id)
    if share_retargeted_stage_dir is not None:
        target_image = _render_aligned_share_retargeted_stage_image(
            stage_dir=share_retargeted_stage_dir,
            desired_target_pose=aligned_target_pose,
            args=args,
        )
    else:
        target_image = _render_aligned_retargeted_target_image(
            job=jobs[0],
            render_args=render_args,
            desired_target_pose=aligned_target_pose,
        )
    if share_stage_dir is not None:
        source_image = _render_share_stage_image(share_stage_dir, args)
    else:
        source_image, _ = render_retargeting_panels(jobs[0], render_args)
    return _panel_with_title(
        "Source template and retargeted hand",
        _labeled_object_canvas(
            source_image,
            "Human + source",
            args.object_width,
            args.object_height,
            args.label_height,
        ),
        _labeled_object_canvas(
            target_image,
            "Robot + target",
            args.object_width,
            args.object_height,
            args.label_height,
        ),
    )


def _target_alignment_pose(
    base_pose: np.ndarray,
    euler_xyz: np.ndarray,
) -> np.ndarray:
    pose = np.asarray(base_pose, dtype=np.float32).copy()
    rotation = R.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float64)).as_matrix()
    pose[:3, :3] = pose[:3, :3] @ rotation.astype(np.float32)
    return pose


def _default_output_path(pair: ExperimentPair) -> Path:
    return (
        get_logs_dir()
        / "alignment_before_after_figures"
        / pair.category
        / f"{pair.pair_id}.png"
    )


def _render_figure(args: RenderAlignmentArgs) -> Path:
    pair = _select_pair(args)
    assets = _load_alignment_assets(pair, args)

    source_rgba = (46, 134, 193, 255)
    target_rgba = (211, 84, 0, 255)
    target_before = _transformed_target_mesh(
        assets.target_mesh,
        euler_xyz=np.zeros(3, dtype=np.float64),
        scale=assets.scale,
    )
    target_after = _transformed_target_mesh(
        assets.target_mesh,
        euler_xyz=assets.euler_xyz,
        scale=assets.scale,
    )

    render_kwargs = {
        "width": args.object_width,
        "height": args.object_height,
        "camera_distance": args.camera_distance,
        "camera_elevation": args.camera_elevation,
        "trim_padding": args.trim_padding,
    }
    source_before_img = _render_object(
        mesh=assets.source_mesh,
        pose=assets.source_pose,
        rgba=source_rgba,
        **render_kwargs,
    )
    target_before_img = _render_object(
        mesh=target_before,
        pose=assets.target_pose,
        rgba=target_rgba,
        **render_kwargs,
    )
    source_after_img = _render_object(
        mesh=assets.source_mesh,
        pose=assets.source_pose,
        rgba=source_rgba,
        **render_kwargs,
    )
    target_after_img = _render_object(
        mesh=target_after,
        pose=assets.target_pose,
        rgba=target_rgba,
        **render_kwargs,
    )

    before_panel = _panel_with_title(
        "Before rotation alignment",
        _labeled_object_canvas(
            source_before_img,
            "Source",
            args.object_width,
            args.object_height,
            args.label_height,
        ),
        _labeled_object_canvas(
            target_before_img,
            "Target, scaled",
            args.object_width,
            args.object_height,
            args.label_height,
        ),
    )
    after_panel = _panel_with_title(
        "After rotation alignment",
        _labeled_object_canvas(
            source_after_img,
            "Source",
            args.object_width,
            args.object_height,
            args.label_height,
        ),
        _labeled_object_canvas(
            target_after_img,
            "Target, scaled + rotated",
            args.object_width,
            args.object_height,
            args.label_height,
        ),
    )
    panels = [before_panel, after_panel]
    if args.template_task is not None:
        panels.append(
            _render_hand_object_panel(
                args,
                desired_target_pose=assets.target_pose,
                target_alignment_euler_xyz=assets.euler_xyz,
            )
        )
    figure = _concat_horizontal(panels, gutter=28)

    output = _default_output_path(pair) if args.output is None else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output)

    print(f"Pair: {pair.category}/{pair.pair_id}")
    print(f"Rotation input: {_rotation_path(pair)}")
    print(f"Scale input: {_scale_path(pair)}")
    if args.template_task is not None:
        print(
            "Template panel: "
            f"{args.template_pipeline}/{args.template_task}/"
            f"{args.template_retargeting_id or '<auto>'}"
        )
    print(f"Saved alignment before/after figure to {output.resolve()}")
    return output


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    _render_figure(args)


if __name__ == "__main__":
    main()
