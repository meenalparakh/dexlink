#!/usr/bin/env python

from __future__ import annotations

import argparse
import itertools
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from dexlink.src.retargeting.data.process_arctic import get_processor_cfg
from dexlink.src.retargeting.optimization.oakink_processor import OakInkProcessor
from dexlink.src.retargeting.optimization.target_transform_utils import (
    apply_target_mesh_transform,
    euler_xyz_from_rotation_matrix,
    load_target_transform,
    parse_transform_scalar,
    rotation_matrix_from_euler_xyz,
)
from dexlink.utils.retargeting.path_utils import get_src
from dexlink.utils.retargeting.urdf_utils import (
    load_urdf_combined_mesh,
    materialize_scaled_urdf,
)
from dexlink.utils.retargeting.vis_server import VizServer
from dexlink.src.reference.common.exp_info import SOURCE_TEMPLATES, TARGET_OBJECTS
from dexlink.src.reference.common.exp_schema import SourceTemplate, TargetObject

DEXLINK_ROOT = get_src()
TRANSFORM_ROOT = DEXLINK_ROOT / "dexlink/src/reference/common/categories/transforms"


@dataclass(frozen=True)
class ExperimentPair:
    category: str
    source_index: int
    source: SourceTemplate
    target_index: int
    target: TargetObject

    @property
    def source_object_key(self) -> str:
        return f"{self.source.category}_{self.source.source_object_id}"

    @property
    def pair_id(self) -> str:
        source_id = f"source_{self.source_object_key}"
        target_id = f"target_{self.target_index:02d}_{self.target.target_id}"
        return f"{source_id}__{target_id}"


def _relative_or_abs_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return DEXLINK_ROOT / path


def _rotation_path(pair: ExperimentPair) -> Path:
    return TRANSFORM_ROOT / pair.category / pair.pair_id / "rotation.json"


def _scale_path(pair: ExperimentPair) -> Path:
    return TRANSFORM_ROOT / pair.category / pair.pair_id / "scale.txt"


def _manifest_path(pair: ExperimentPair) -> Path:
    return TRANSFORM_ROOT / pair.category / pair.pair_id / "manifest.json"


def _scaled_urdf_path(pair: ExperimentPair) -> Path:
    return (
        TRANSFORM_ROOT
        / pair.category
        / pair.pair_id
        / f"{pair.target.target_id}_scaled.urdf"
    )


def _urdfpy_compatible_urdf_path(pair: ExperimentPair) -> Path:
    return (
        TRANSFORM_ROOT
        / pair.category
        / pair.pair_id
        / f"{pair.target.target_id}_urdfpy.urdf"
    )


def _urdf_visualization_obj_path(pair: ExperimentPair) -> Path:
    return (
        TRANSFORM_ROOT
        / pair.category
        / pair.pair_id
        / f"{pair.target.target_id}_default_visual.obj"
    )


def _axis_order(extent: np.ndarray) -> np.ndarray:
    extent = np.asarray(extent, dtype=np.float64)
    if extent.shape != (3,):
        raise ValueError(f"`extent` must have shape (3,), got {extent.shape}.")
    if not np.all(np.isfinite(extent)) or np.any(extent <= 0.0):
        raise ValueError(f"`extent` must be finite and positive, got {extent}.")
    if np.unique(np.round(extent, decimals=8)).shape[0] != 3:
        raise ValueError(
            "Bounding-box axis order is ambiguous because at least two extents "
            f"are equal or nearly equal: {extent}."
        )
    order = np.empty(3, dtype=np.int64)
    order[np.argsort(extent)] = np.arange(3, dtype=np.int64)
    return order


def _bbox_axis_order_rotation(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
) -> np.ndarray:
    source_extent = np.asarray(source_mesh.extents, dtype=np.float64)
    target_extent = np.asarray(target_mesh.extents, dtype=np.float64)
    source_order = _axis_order(source_extent)

    best_rotation = None
    best_score = None
    for permutation in itertools.permutations(range(3)):
        abs_rotation = np.zeros((3, 3), dtype=np.float64)
        for output_axis, input_axis in enumerate(permutation):
            abs_rotation[output_axis, input_axis] = 1.0
        transformed_extent = abs_rotation @ target_extent
        if not np.array_equal(_axis_order(transformed_extent), source_order):
            continue
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = abs_rotation * np.asarray(signs, dtype=np.float64)[:, None]
            if not np.isclose(np.linalg.det(rotation), 1.0):
                continue
            angle = np.linalg.norm(euler_xyz_from_rotation_matrix(rotation))
            score = (angle, permutation, signs)
            if best_score is None or score < best_score:
                best_score = score
                best_rotation = rotation

    if best_rotation is None:
        raise ValueError(
            "Could not find a proper axis permutation that matches source and "
            f"target bounding-box extent order. Source extent={source_extent}, "
            f"target extent={target_extent}."
        )
    return best_rotation


def _banana_ketchup_bbox_initial_values(
    pair: ExperimentPair,
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, float] | None:
    if pair.category != "banana_handover":
        return None
    if pair.source.source_object_id != "ketchup_use_01":
        return None
    if pair.target.target_id != "banana_handover":
        return None

    rotation = _bbox_axis_order_rotation(source_mesh, target_mesh)
    euler_xyz = euler_xyz_from_rotation_matrix(rotation)
    return euler_xyz, 1.0


def _read_scale_file(path: Path) -> float:
    try:
        scale = float(path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ValueError(f"Expected numeric scale in {path}.") from exc
    if scale <= 0.0:
        raise ValueError(f"Expected positive scale in {path}, got {scale}.")
    return scale


def _existing_target_scale(pair: ExperimentPair) -> float | None:
    existing: list[tuple[int, Path, float]] = []
    for source_index in sorted(SOURCE_TEMPLATES[pair.category]):
        source_pair = ExperimentPair(
            category=pair.category,
            source_index=source_index,
            source=SOURCE_TEMPLATES[pair.category][source_index],
            target_index=pair.target_index,
            target=pair.target,
        )
        scale_path = _scale_path(source_pair)
        if not scale_path.exists():
            continue
        existing.append((source_index, scale_path, _read_scale_file(scale_path)))
    if not existing:
        return None

    first_scale = existing[0][2]
    conflicting = [
        (source_index, scale_path, scale)
        for source_index, scale_path, scale in existing
        if abs(scale - first_scale) > 1e-9
    ]
    if conflicting:
        details = ", ".join(
            f"source {source_index}: {scale} ({scale_path})"
            for source_index, scale_path, scale in existing
        )
        raise ValueError(
            f"Existing scale.txt values disagree for {pair.category} target "
            f"{pair.target_index}. Run "
            "the category scale metadata first. "
            f"Existing values: {details}"
        )
    return first_scale


def _load_saved_values(
    pair: ExperimentPair,
    source_mesh: trimesh.Trimesh | None = None,
    target_mesh: trimesh.Trimesh | None = None,
) -> tuple[np.ndarray, float]:
    euler_xyz = np.zeros(3, dtype=np.float64)
    scale = float(pair.target.scale)
    rotation_path = _rotation_path(pair)
    scale_path = _scale_path(pair)
    if rotation_path.exists():
        euler_xyz = euler_xyz_from_rotation_matrix(load_target_transform(rotation_path))
    elif source_mesh is not None and target_mesh is not None:
        initial_values = _banana_ketchup_bbox_initial_values(
            pair=pair,
            source_mesh=source_mesh,
            target_mesh=target_mesh,
        )
        if initial_values is not None:
            euler_xyz, scale = initial_values
    if scale_path.exists():
        scale = _read_scale_file(scale_path)
    else:
        existing_scale = _existing_target_scale(pair)
        if existing_scale is not None:
            scale = existing_scale
    return euler_xyz, scale


def _parse_user_values(raw_value: str) -> tuple[np.ndarray, float]:
    tokens = [token for token in raw_value.replace(",", " ").split() if token]
    if len(tokens) != 4:
        raise ValueError(
            "Expected exactly four values: rx ry rz scale, all in radians except scale."
        )
    euler_xyz = np.asarray(
        [parse_transform_scalar(token, "input") for token in tokens[:3]],
        dtype=np.float64,
    )
    scale = parse_transform_scalar(tokens[3], "input")
    if scale <= 0.0:
        raise ValueError(f"`scale` must be positive, got {scale}.")
    return euler_xyz, float(scale)


def _load_target_mesh(target_path: Path) -> trimesh.Trimesh:
    suffix = target_path.suffix.lower()
    if suffix == ".urdf":
        return load_urdf_combined_mesh(target_path, mesh_type="visual")
    if suffix in (".obj", ".ply", ".stl", ".glb", ".gltf"):
        loaded = trimesh.load(target_path, process=False)
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [geometry for geometry in loaded.geometry.values()]
            )
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise TypeError(
                f"Expected trimesh to load {target_path} as Trimesh or Scene, got {type(loaded)}."
            )
        return mesh
    raise ValueError(
        f"Unsupported target path suffix {suffix!r} for {target_path}. "
        "Expected .obj, .ply, .stl, .glb, .gltf, or .urdf."
    )


def _materialize_urdfpy_compatible_urdf(
    pair: ExperimentPair,
    target_path: Path,
) -> Path:
    if target_path.suffix.lower() != ".urdf":
        raise ValueError(f"Expected a .urdf target path, got {target_path}.")
    if not target_path.exists():
        raise FileNotFoundError(f"URDF file not found: {target_path}")

    tree = ET.parse(target_path)
    root = tree.getroot()
    changed = False
    for joint in root.findall(".//joint"):
        if joint.attrib.get("type") == "fixed":
            limit = joint.find("limit")
            if limit is not None:
                joint.remove(limit)
                changed = True
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        if "effort" not in limit.attrib:
            limit.set("effort", "1.0")
            changed = True
        if "velocity" not in limit.attrib:
            limit.set("velocity", "1.0")
            changed = True
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename is None:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh.set("filename", str((target_path.parent / mesh_path).resolve()))
            changed = True

    if not changed:
        return target_path

    output_path = _urdfpy_compatible_urdf_path(pair)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _materialize_urdf_visualization_obj(
    pair: ExperimentPair,
    target_path: Path,
) -> trimesh.Trimesh:
    if target_path.suffix.lower() != ".urdf":
        raise ValueError(f"Expected a .urdf target path, got {target_path}.")
    visualization_obj_path = _urdf_visualization_obj_path(pair)
    visualization_obj_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_path = _materialize_urdfpy_compatible_urdf(pair, target_path)
    mesh = load_urdf_combined_mesh(urdf_path, mesh_type="visual")
    mesh.export(visualization_obj_path)
    return _load_target_mesh(visualization_obj_path)


def _load_target_mesh_from_demo(processor: OakInkProcessor) -> trimesh.Trimesh:
    target_mesh_path = processor._target_obj_mesh_path
    if target_mesh_path is not None:
        return processor._load_mesh(target_mesh_path)
    if processor.provided_target_urdf is not None:
        return load_urdf_combined_mesh(
            Path(processor.provided_target_urdf),
            mesh_type="visual",
        )
    raise ValueError(
        "Expected OakInkProcessor target demo to resolve `target_obj_mesh_path` "
        "or `provided_target_urdf`."
    )


def _arctic_local_timestep(source: SourceTemplate) -> int:
    if source.arctic_timestep is None:
        raise ValueError(f"ARCTIC source {source.template_id} has no arctic_timestep.")
    if source.arctic_start_timestep is None:
        raise ValueError(
            f"ARCTIC source {source.template_id} has no arctic_start_timestep."
        )
    if source.arctic_skip <= 0:
        raise ValueError(
            f"ARCTIC source {source.template_id} must have positive arctic_skip, "
            f"got {source.arctic_skip}."
        )
    offset = source.arctic_timestep - source.arctic_start_timestep
    if offset < 0 or offset % source.arctic_skip != 0:
        raise ValueError(
            f"ARCTIC timestep {source.arctic_timestep} is not aligned with "
            f"start={source.arctic_start_timestep}, skip={source.arctic_skip}."
        )
    return offset // source.arctic_skip


def _load_arctic_source_mesh_and_pose(
    pair: ExperimentPair,
    source_demo_path: Path,
    target_path: Path | None,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    if pair.source.arctic_target_object is None:
        raise ValueError(
            f"ARCTIC source {pair.source.template_id} has no target object."
        )
    if (
        pair.source.arctic_start_timestep is None
        or pair.source.arctic_end_timestep is None
    ):
        raise ValueError(
            f"ARCTIC source {pair.source.template_id} must define start/end timesteps."
        )

    processor_cfg = get_processor_cfg(pair.source.arctic_target_object)
    processor_cls = processor_cfg["processor_cls"]
    processor = processor_cls(
        original_demo_path=str(source_demo_path),
        target_object=pair.source.arctic_target_object,
        start_timestep=pair.source.arctic_start_timestep,
        end_timestep=pair.source.arctic_end_timestep,
        skip=pair.source.arctic_skip,
        reference_urdf=processor_cfg["reference_urdf"],
        target_urdf=processor_cfg["target_urdf_path"],
        target_obj_path=None if target_path is None else str(target_path),
        run_dir=DEXLINK_ROOT
        / "logs"
        / "experiment_category_transform_preview"
        / pair.category,
    )
    return processor.get_trimesh_at_timestep(
        timestep=_arctic_local_timestep(pair.source),
        obj_type="original",
        mesh_type="visual",
    )


def _mesh_with_face_color(mesh: trimesh.Trimesh, rgba: np.ndarray) -> trimesh.Trimesh:
    colored = mesh.copy()
    colored.visual.face_colors = np.tile(rgba[None, :], (len(colored.faces), 1))
    return colored


def _transformed_target_mesh(
    target_mesh: trimesh.Trimesh,
    euler_xyz: np.ndarray,
    scale: float,
) -> trimesh.Trimesh:
    return apply_target_mesh_transform(
        target_mesh,
        transform=rotation_matrix_from_euler_xyz(euler_xyz),
        scale=scale,
    )


def _print_overlay_stats(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    euler_xyz: np.ndarray,
    scale: float,
) -> None:
    source_extent = source_mesh.bounds[1] - source_mesh.bounds[0]
    target_extent = target_mesh.bounds[1] - target_mesh.bounds[0]
    print(
        "Current values: "
        f"rx={euler_xyz[0]:.6f} ry={euler_xyz[1]:.6f} rz={euler_xyz[2]:.6f} scale={scale:.6f}"
    )
    print(f"Source extent: {np.array2string(source_extent, precision=4)}")
    print(f"Target extent: {np.array2string(target_extent, precision=4)}")


def _render_overlay_meshcat(
    vis: VizServer,
    source_mesh: trimesh.Trimesh,
    source_pose: np.ndarray,
    target_mesh: trimesh.Trimesh,
    target_pose: np.ndarray,
    euler_xyz: np.ndarray,
    scale: float,
) -> None:
    transformed_target_mesh = _transformed_target_mesh(target_mesh, euler_xyz, scale)
    vis.view_trimesh(
        name="reference/object",
        mesh=source_mesh,
        pose=source_pose,
        color=0x2E86C1,
        opacity=0.55,
    )
    vis.view_trimesh(
        name="target/object",
        mesh=transformed_target_mesh,
        pose=target_pose,
        color=0xD35400,
        opacity=0.55,
    )
    vis.view_frames(source_pose, name="reference/frame", scale=0.15)
    vis.view_frames(target_pose, name="target/root_frame", scale=0.15)
    _print_overlay_stats(source_mesh, transformed_target_mesh, euler_xyz, scale)


def _render_overlay_trimesh(
    source_mesh: trimesh.Trimesh,
    source_pose: np.ndarray,
    target_mesh: trimesh.Trimesh,
    target_pose: np.ndarray,
    euler_xyz: np.ndarray,
    scale: float,
) -> None:
    transformed_target_mesh = _transformed_target_mesh(target_mesh, euler_xyz, scale)
    _print_overlay_stats(source_mesh, transformed_target_mesh, euler_xyz, scale)

    scene = trimesh.Scene()
    scene.add_geometry(
        _mesh_with_face_color(
            source_mesh,
            np.asarray([46, 134, 193, 140], dtype=np.uint8),
        ),
        node_name="reference/object",
        transform=source_pose,
    )
    scene.add_geometry(
        _mesh_with_face_color(
            transformed_target_mesh,
            np.asarray([211, 84, 0, 140], dtype=np.uint8),
        ),
        node_name="target/object",
        transform=target_pose,
    )
    scene.add_geometry(
        trimesh.creation.axis(origin_size=0.01, axis_length=0.15),
        node_name="reference/frame",
        transform=source_pose,
    )
    scene.add_geometry(
        trimesh.creation.axis(origin_size=0.01, axis_length=0.15),
        node_name="target/root_frame",
        transform=target_pose,
    )
    scene.show()


def _save_pair_transform(
    pair: ExperimentPair,
    euler_xyz: np.ndarray,
    scale: float,
) -> None:
    rotation_path = _rotation_path(pair)
    scale_path = _scale_path(pair)
    manifest_path = _manifest_path(pair)
    rotation_path.parent.mkdir(parents=True, exist_ok=True)
    rotation_payload = {"euler_xyz": [float(value) for value in euler_xyz.tolist()]}
    rotation_path.write_text(
        json.dumps(rotation_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    scale_path.write_text(f"{float(scale):.8f}\n", encoding="utf-8")

    scaled_urdf = None
    if pair.target.mesh_path is not None:
        target_path = _relative_or_abs_path(pair.target.mesh_path)
    else:
        target_path = None
    urdfpy_compatible_urdf = None
    if target_path is not None and target_path.suffix.lower() == ".urdf":
        urdfpy_compatible_urdf = _materialize_urdfpy_compatible_urdf(pair, target_path)
        scaled_urdf = materialize_scaled_urdf(
            input_path=urdfpy_compatible_urdf,
            output_dir=rotation_path.parent,
            scale=scale,
            output_name=_scaled_urdf_path(pair).name,
        )
    visualization_obj = (
        _urdf_visualization_obj_path(pair)
        if target_path is not None and target_path.suffix.lower() == ".urdf"
        else None
    )

    manifest = {
        "category": pair.category,
        "source_index": pair.source_index,
        "source_template_id": pair.source.template_id,
        "source_demo_path": pair.source.demo_path,
        "target_index": pair.target_index,
        "target_id": pair.target.target_id,
        "target_mesh_path": pair.target.mesh_path,
        "target_demo_path": pair.target.target_demo_path,
        "rotation_path": str(rotation_path.relative_to(DEXLINK_ROOT)),
        "scale_path": str(scale_path.relative_to(DEXLINK_ROOT)),
        "target_transform_info": str(rotation_path.relative_to(DEXLINK_ROOT)),
        "target_scale": float(scale),
        "rotation_semantics": (
            "For mesh targets, rotation is applied to target vertices before retargeting. "
            "For URDF targets, rotation is a root/base rotation to use at placement time; "
            "the saved scaled_urdf_path only materializes mesh/origin scale."
        ),
        "scaled_urdf_path": (
            None
            if scaled_urdf is None
            else str(scaled_urdf.relative_to(DEXLINK_ROOT))
        ),
        "urdfpy_compatible_urdf_path": (
            None
            if urdfpy_compatible_urdf is None
            else str(urdfpy_compatible_urdf.relative_to(DEXLINK_ROOT))
        ),
        "target_visualization_obj_path": (
            None
            if visualization_obj is None
            else str(visualization_obj.relative_to(DEXLINK_ROOT))
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved rotation preset to {rotation_path}")
    print(f"Saved scale preset to {scale_path}")
    print(f"Saved manifest to {manifest_path}")
    if scaled_urdf is not None:
        print(f"Saved scaled URDF to {scaled_urdf}")


def _is_urdf_path(path: Path | None) -> bool:
    return path is not None and path.suffix.lower() == ".urdf"


def _iter_pairs(
    category: str | None,
    source_index: int | None,
    target_index: int | None,
) -> tuple[ExperimentPair, ...]:
    categories = [category] if category is not None else sorted(SOURCE_TEMPLATES)
    pairs = []
    for category_name in categories:
        if category_name not in SOURCE_TEMPLATES:
            raise KeyError(f"Unknown category {category_name!r}.")
        if category_name not in TARGET_OBJECTS:
            raise KeyError(f"Unknown target category {category_name!r}.")
        sources = SOURCE_TEMPLATES[category_name]
        targets = TARGET_OBJECTS[category_name]
        if source_index is not None:
            if source_index not in sources:
                raise KeyError(
                    f"Expected source index {source_index} for category {category_name!r}."
                )
            sources = {source_index: sources[source_index]}
        if target_index is not None:
            if target_index not in targets:
                raise KeyError(
                    f"Expected target index {target_index} for category {category_name!r}."
                )
            targets = {target_index: targets[target_index]}
        targets = {
            tgt_idx: target
            for tgt_idx, target in targets.items()
            if not target.same_as_source
        }
        unique_sources = _unique_source_objects(sources)
        for src_idx, source in unique_sources.items():
            for tgt_idx, target in targets.items():
                pairs.append(
                    ExperimentPair(
                        category=category_name,
                        source_index=src_idx,
                        source=source,
                        target_index=tgt_idx,
                        target=target,
                    )
                )
    return tuple(pairs)


def _unique_source_objects(
    sources: dict[int, SourceTemplate],
) -> dict[int, SourceTemplate]:
    unique_sources: dict[int, SourceTemplate] = {}
    seen_object_keys: set[tuple[str, str]] = set()
    for source_index, source in sources.items():
        object_key = (source.category, source.source_object_id)
        if object_key in seen_object_keys:
            continue
        seen_object_keys.add(object_key)
        unique_sources[source_index] = source
    return unique_sources


def _preview_pair(
    pair: ExperimentPair,
    hand: str,
    viewer: str,
    port: int,
) -> None:
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
                f"ARCTIC transform preview for {pair.pair_id} does not support "
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
            oakink_hand_side=hand,
            run_dir=DEXLINK_ROOT
            / "logs"
            / "experiment_category_transform_preview"
            / pair.category,
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
                "ARCTIC transform preview requires a target mesh_path."
            )
        target_mesh = _load_target_mesh_from_demo(processor)
    elif is_urdf_target:
        target_mesh = _materialize_urdf_visualization_obj(pair, target_path)
    else:
        target_mesh = _load_target_mesh(target_path)
    target_pose = source_pose.copy()

    euler_xyz, scale = _load_saved_values(
        pair,
        source_mesh=source_mesh,
        target_mesh=target_mesh,
    )
    vis = None
    if viewer == "meshcat":
        vis = VizServer(port_vis=port)
        print(f"MeshCat ZMQ URL: {vis.zmq_url}")

    print()
    print(f"Pair: {pair.category}/{pair.pair_id}")
    print(f"Source demo: {source_demo_path}")
    if target_path is not None:
        print(f"Target object: {target_path}")
    if target_demo_path is not None:
        print(f"Target demo: {target_demo_path}")
    print(f"Rotation output: {_rotation_path(pair)}")
    print(f"Scale output: {_scale_path(pair)}")
    print(
        "Type four values `rx ry rz scale` to update the overlay, "
        "`s` to save, `n` to skip this pair, or `q` to quit."
    )
    if is_urdf_target:
        print(f"URDF visualization OBJ: {_urdf_visualization_obj_path(pair)}")
        print(
            "URDF target: scale will be materialized into a copied URDF on save; "
            "rotation is saved as root/base rotation metadata."
        )

    while True:
        if viewer == "meshcat":
            assert vis is not None
            _render_overlay_meshcat(
                vis=vis,
                source_mesh=source_mesh,
                source_pose=source_pose,
                target_mesh=target_mesh,
                target_pose=target_pose,
                euler_xyz=euler_xyz,
                scale=scale,
            )
        else:
            _render_overlay_trimesh(
                source_mesh=source_mesh,
                source_pose=source_pose,
                target_mesh=target_mesh,
                target_pose=target_pose,
                euler_xyz=euler_xyz,
                scale=scale,
            )
        user_input = input("> ").strip()
        if not user_input:
            continue
        if user_input.lower() == "q":
            raise KeyboardInterrupt
        if user_input.lower() == "n":
            return
        if user_input.lower() == "s":
            _save_pair_transform(pair, euler_xyz, scale)
            return
        try:
            euler_xyz, scale = _parse_user_values(user_input)
        except ValueError as exc:
            print(exc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively align experiment category source-target pairs and save "
            "rotation/scale presets under dexlink/src/reference/common/categories/transforms."
        )
    )
    parser.add_argument("--category", default=None, type=str)
    parser.add_argument("--source-index", default=None, type=int)
    parser.add_argument("--target-index", default=None, type=int)
    parser.add_argument("--hand", default="right", choices=("left", "right"))
    parser.add_argument(
        "--viewer",
        default="meshcat",
        choices=("meshcat", "trimesh"),
        help="Visualization backend. `trimesh` opens a local scene window for each update.",
    )
    parser.add_argument("--port", default=6000, type=int)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip pairs that already have both rotation.json and scale.txt.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected pairs and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pairs = _iter_pairs(
        category=args.category,
        source_index=args.source_index,
        target_index=args.target_index,
    )
    if args.skip_existing:
        pairs = tuple(
            pair
            for pair in pairs
            if not (_rotation_path(pair).exists() and _scale_path(pair).exists())
        )

    if args.list:
        for pair in pairs:
            print(
                f"{pair.category}\tsource_object={pair.source_object_key}"
                f"\trepresentative_source={pair.source_index}:{pair.source.template_id}"
                f"\ttarget={pair.target_index}:{pair.target.target_id}"
                f"\trotation={_rotation_path(pair).relative_to(DEXLINK_ROOT)}"
            )
        return 0

    if not pairs:
        print("No source-target pairs selected.")
        return 0

    for pair in pairs:
        try:
            _preview_pair(
                pair=pair,
                hand=args.hand,
                viewer=args.viewer,
                port=args.port,
            )
        except KeyboardInterrupt:
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
