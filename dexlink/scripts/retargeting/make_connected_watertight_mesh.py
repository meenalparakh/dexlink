import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes

from dexlink.utils.retargeting.path_utils import get_src

DEXLINK_ROOT = get_src()


@dataclass(frozen=True)
class MeshRepairConfig:
    pitch: float | None = None
    pitch_divisions: int | None = 600
    closing_iterations: int = 1
    dilation_iterations: int = 1
    padding: int = 4
    min_volume: float = 1e-14
    max_voxels: int = 25_000_000
    auto_coarsen: bool = False
    skip_valid: bool = False
    overwrite: bool = False


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _relative_or_abs_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return DEXLINK_ROOT / path


def _resolve_pitch(mesh: trimesh.Trimesh, cfg: MeshRepairConfig) -> float:
    if cfg.pitch is not None:
        if cfg.pitch <= 0.0:
            raise ValueError(f"Expected positive pitch, got {cfg.pitch}.")
        return float(cfg.pitch)
    if cfg.pitch_divisions is None or cfg.pitch_divisions <= 0:
        raise ValueError(
            f"Expected positive pitch_divisions, got {cfg.pitch_divisions}."
        )
    return float(mesh.scale) / float(cfg.pitch_divisions)


def _voxel_grid_shape(mesh: trimesh.Trimesh, pitch: float, padding: int) -> np.ndarray:
    extents = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float64)
    return np.ceil(extents / float(pitch)).astype(np.int64) + 1 + 2 * int(padding)


def _guard_or_coarsen_pitch(
    mesh: trimesh.Trimesh,
    pitch: float,
    padding: int,
    cfg: MeshRepairConfig,
) -> float:
    if cfg.max_voxels <= 0:
        raise ValueError(f"Expected positive max_voxels, got {cfg.max_voxels}.")
    shape = _voxel_grid_shape(mesh, pitch, padding)
    voxel_count = int(np.prod(shape, dtype=np.int64))
    if voxel_count <= cfg.max_voxels:
        return pitch
    if not cfg.auto_coarsen:
        raise ValueError(
            "Requested voxel grid is too large before repair. "
            f"shape={shape.tolist()}, voxels={voxel_count}, "
            f"max_voxels={cfg.max_voxels}, pitch={pitch}. Use a smaller "
            "--pitch-divisions value, larger --pitch, or pass --auto-coarsen."
        )

    scale = (float(voxel_count) / float(cfg.max_voxels)) ** (1.0 / 3.0)
    coarsened_pitch = float(pitch) * scale * 1.05
    coarsened_shape = _voxel_grid_shape(mesh, coarsened_pitch, padding)
    coarsened_count = int(np.prod(coarsened_shape, dtype=np.int64))
    print(
        "Coarsened pitch to respect max voxel budget: "
        f"{pitch} -> {coarsened_pitch}; "
        f"shape {shape.tolist()} -> {coarsened_shape.tolist()}; "
        f"voxels {voxel_count} -> {coarsened_count}"
    )
    return coarsened_pitch


def _load_mesh(input_path: str | Path) -> trimesh.Trimesh:
    input_path = Path(input_path)
    mesh = trimesh.load(input_path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Input scene has no geometry: {input_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh from {input_path}, got {type(mesh)}.")
    if mesh.is_empty:
        raise ValueError(f"Input mesh is empty: {input_path}")
    if len(mesh.faces) == 0:
        raise ValueError(f"Input mesh has no faces: {input_path}")
    return mesh


def mesh_check(mesh: trimesh.Trimesh, min_volume: float = 1e-14) -> dict[str, object]:
    signed_volume = float(mesh.volume) if mesh.is_watertight else 0.0
    abs_volume = abs(signed_volume)
    inertia = np.asarray(mesh.moment_inertia, dtype=np.float64)
    inertia_diag = np.diag(inertia)
    principal_moments = (
        np.linalg.eigvalsh(inertia)
        if inertia.shape == (3, 3) and np.all(np.isfinite(inertia))
        else np.full(3, np.nan, dtype=np.float64)
    )
    triangle_tolerance = max(
        1.0e-12,
        1.0e-9 * float(np.sum(np.abs(principal_moments))),
    )
    inertia_ok = bool(
        inertia.shape == (3, 3)
        and np.all(np.isfinite(inertia))
        and np.allclose(inertia, inertia.T)
        and np.all(principal_moments > 0.0)
        and 2.0 * float(np.max(principal_moments))
        <= float(np.sum(principal_moments)) + triangle_tolerance
    )
    return {
        "watertight": bool(mesh.is_watertight),
        "volume": float(signed_volume),
        "abs_volume": float(abs_volume),
        "volume_ok": bool(signed_volume >= min_volume),
        "inertia_diag": inertia_diag.tolist(),
        "inertia_principal_moments": principal_moments.tolist(),
        "inertia_ok": inertia_ok,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "valid_for_drake_inertia": bool(
            mesh.is_watertight and signed_volume >= min_volume and inertia_ok
        ),
    }


def check_mesh_path(
    mesh_path: str | Path,
    min_volume: float = 1e-14,
) -> dict[str, object]:
    return mesh_check(_load_mesh(mesh_path), min_volume=min_volume)


def _rewind_positive_volume_mesh(
    mesh: trimesh.Trimesh,
    min_volume: float,
) -> trimesh.Trimesh | None:
    if not mesh.is_watertight:
        return None

    signed_volume = float(mesh.volume)
    if abs(signed_volume) < min_volume:
        return None

    rewound = mesh.copy()
    rewound.fix_normals()
    if float(rewound.volume) < 0.0:
        rewound.invert()

    check = mesh_check(rewound, min_volume=min_volume)
    if not check["valid_for_drake_inertia"]:
        return None
    return rewound


def _voxel_matrix_to_world_mesh(
    matrix: np.ndarray,
    *,
    voxel_transform: np.ndarray,
    pitch: float,
    input_padding: int,
) -> trimesh.Trimesh:
    transform = np.asarray(voxel_transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(
            f"Expected voxel transform shape (4, 4), got {transform.shape}."
        )
    expected_linear = np.eye(3, dtype=np.float64) * float(pitch)
    if not np.allclose(transform[:3, :3], expected_linear):
        raise ValueError(
            "Expected an axis-aligned voxel transform with pitch on its diagonal, "
            f"got linear transform {transform[:3, :3].tolist()}."
        )

    index_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(matrix, pitch=pitch)
    index_vertices = index_mesh.vertices / float(pitch) - float(input_padding)
    world_vertices = trimesh.transform_points(index_vertices, transform)
    world_mesh = trimesh.Trimesh(
        vertices=world_vertices,
        faces=index_mesh.faces,
        process=True,
        validate=True,
    )
    world_mesh.fix_normals()
    return world_mesh


def make_connected_solid_mesh(
    input_path,
    output_path=None,
    pitch=None,
    pitch_divisions=600,
    closing_iterations=1,
    dilation_iterations=1,
    padding=4,
    min_volume=1e-14,
    max_voxels=25_000_000,
    auto_coarsen=False,
    export=True,
):
    input_path = Path(input_path)
    mesh = _load_mesh(input_path)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_solid_watertight.obj")
    else:
        output_path = Path(output_path)

    rewound_mesh = _rewind_positive_volume_mesh(mesh, min_volume=min_volume)
    if rewound_mesh is not None:
        print(
            "Input mesh is already solid; exporting positive-volume winding: "
            f"{output_path}"
        )
        if export:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            rewound_mesh.export(output_path)
        return rewound_mesh

    if pitch is None:
        pitch = _resolve_pitch(
            mesh,
            MeshRepairConfig(pitch=None, pitch_divisions=pitch_divisions),
        )
    pitch = _guard_or_coarsen_pitch(
        mesh,
        pitch,
        padding,
        MeshRepairConfig(
            max_voxels=max_voxels,
            auto_coarsen=auto_coarsen,
        ),
    )
    print(f"pitch: {pitch}, padding: {padding}")

    vox = mesh.voxelized(pitch)
    grid = vox.matrix.astype(bool)

    if not grid.any():
        raise ValueError("Voxelization produced an empty grid. Try increasing pitch.")

    # Give the volume a margin so outside is clearly defined
    grid = np.pad(grid, padding, mode="constant", constant_values=False)

    # Slightly thicken the surface so it encloses volume better
    if dilation_iterations > 0:
        grid = binary_dilation(grid, iterations=dilation_iterations)

    # Connect nearby components
    if closing_iterations > 0:
        grid = binary_closing(grid, iterations=closing_iterations)

    # Fill enclosed cavities
    grid = binary_fill_holes(grid)

    if grid.min() == grid.max():
        raise ValueError(
            "Voxel volume has no boundary. Try smaller closing/dilation settings."
        )
    solid = _voxel_matrix_to_world_mesh(
        grid,
        voxel_transform=vox.transform,
        pitch=pitch,
        input_padding=padding,
    )

    check = mesh_check(solid, min_volume=min_volume)
    if not check["valid_for_drake_inertia"]:
        raise ValueError(
            "Solid mesh repair did not produce a mesh with non-zero inertia: "
            f"{output_path}, check={check}"
        )

    if export:
        solid.export(output_path)

    return solid


def make_connected_watertight_mesh(
    input_path,
    output_path=None,
    pitch=0.02,
    pitch_divisions=None,
    closing_iterations=3,
    padding=2,
    min_volume=1e-14,
    max_voxels=25_000_000,
    auto_coarsen=False,
    export=True,
):
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_watertight.obj")
    else:
        output_path = Path(output_path)

    mesh = _load_mesh(input_path)
    if pitch is None:
        pitch = _resolve_pitch(
            mesh,
            MeshRepairConfig(pitch=None, pitch_divisions=pitch_divisions),
        )
    pitch = _guard_or_coarsen_pitch(
        mesh,
        pitch,
        padding,
        MeshRepairConfig(
            max_voxels=max_voxels,
            auto_coarsen=auto_coarsen,
        ),
    )
    print(f"pitch: {pitch}, padding: {padding}")

    vox = mesh.voxelized(pitch)
    grid = vox.matrix.astype(bool)

    if not grid.any():
        raise ValueError("Voxelization produced an empty grid. Try increasing pitch.")

    grid = binary_closing(grid, iterations=closing_iterations)
    grid = binary_fill_holes(grid)

    # Critical: make sure there is empty space around the solid
    grid = np.pad(grid, padding, mode="constant", constant_values=False)

    if grid.min() == grid.max():
        raise ValueError(
            "Voxel volume has no surface boundary. "
            "Try smaller closing_iterations, smaller pitch, or larger padding."
        )
    result = _voxel_matrix_to_world_mesh(
        grid,
        voxel_transform=vox.transform,
        pitch=pitch,
        input_padding=padding,
    )

    check = mesh_check(result, min_volume=min_volume)
    if not check["valid_for_drake_inertia"]:
        raise ValueError(
            "Watertight mesh repair did not produce a mesh with non-zero inertia: "
            f"{output_path}, check={check}"
        )

    if export:
        result.export(output_path)

    return result


def repair_mesh_file(
    input_path: str | Path,
    output_path: str | Path,
    cfg: MeshRepairConfig,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not cfg.overwrite:
        existing_check = check_mesh_path(output_path, min_volume=cfg.min_volume)
        if existing_check["valid_for_drake_inertia"]:
            return {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "status": "exists",
                "check": existing_check,
            }
        raise ValueError(
            f"Output already exists but is not valid for Drake inertia: {output_path}. "
            "Pass --overwrite to rebuild it."
        )

    input_check = check_mesh_path(input_path, min_volume=cfg.min_volume)
    if cfg.skip_valid and input_check["valid_for_drake_inertia"]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh = _load_mesh(input_path)
        mesh.export(output_path)
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "status": "copied_valid_input",
            "check": input_check,
        }

    mesh = _load_mesh(input_path)
    rewound_mesh = _rewind_positive_volume_mesh(mesh, min_volume=cfg.min_volume)
    if rewound_mesh is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rewound_mesh.export(output_path)
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "status": "rewound_valid_input",
            "input_check": input_check,
            "check": mesh_check(rewound_mesh, min_volume=cfg.min_volume),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired = make_connected_solid_mesh(
        input_path=input_path,
        output_path=output_path,
        pitch=cfg.pitch,
        pitch_divisions=cfg.pitch_divisions,
        closing_iterations=cfg.closing_iterations,
        dilation_iterations=cfg.dilation_iterations,
        padding=cfg.padding,
        min_volume=cfg.min_volume,
        max_voxels=cfg.max_voxels,
        auto_coarsen=cfg.auto_coarsen,
        export=True,
    )
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "repaired",
        "input_check": input_check,
        "check": mesh_check(repaired, min_volume=cfg.min_volume),
    }


def _checked_mesh_output_path(output_root: Path, category: str, job_id: str) -> Path:
    return output_root / category / _slug(job_id) / "target_checked.obj"


def _resolve_oakink_demo_mesh_path(demo_path: str | Path) -> Path:
    demo_path = _relative_or_abs_path(demo_path)
    if demo_path.name != "hand_param.pkl":
        raise ValueError(
            f"Expected OakInk target demo path ending in hand_param.pkl, got {demo_path}."
        )
    object_id = demo_path.parent.parent.name
    oakink_shape_root = DEXLINK_ROOT / "external_data/oakink/shape"
    for object_root_name in ("OakInkObjectsV2", "OakInkVirtualObjectsV2"):
        object_dir = oakink_shape_root / object_root_name / object_id
        for mesh_dir_name in ("align_ds", "align"):
            mesh_dir = object_dir / mesh_dir_name
            if not mesh_dir.exists():
                continue
            candidates = sorted(mesh_dir.glob("*.obj")) + sorted(mesh_dir.glob("*.ply"))
            if candidates:
                return candidates[0].resolve()
    # raise FileNotFoundError(
    print(
        f"Could not resolve OakInk object mesh for target demo {demo_path} "
        f"with object id {object_id!r}."
    )
    return None


def _parse_pairs_file(pairs_file: str | Path | None) -> set[str] | None:
    if pairs_file is None:
        return None
    if str(pairs_file).strip().lower() in {"none", "null", "all"}:
        return None
    pairs = set()
    for line in Path(pairs_file).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            pairs.add(value)
    return pairs


def _job_pair_ids(job) -> set[str]:
    base = f"{job.category}_s{job.source_index:02d}_t{job.target_index:02d}"
    ids = {base}
    if job.articulation_sample is None:
        return ids

    sample_id = job.articulation_sample.sample_id
    ids.add(f"{base}_{sample_id}")
    sample_aliases = {
        "trigger_open": "trigger_op",
        "trigger_pressed": "trigger_pr",
    }
    if sample_id in sample_aliases:
        ids.add(f"{base}_{sample_aliases[sample_id]}")
    return ids


def _category_jobs(args: argparse.Namespace):
    from dexlink.scripts.retargeting.run_experiment_category_retargetings import (
        _iter_jobs,
        _validate_job_inputs,
    )

    runner_args = SimpleNamespace(
        category=args.category,
        section=args.section,
        source_index=args.source_index,
        target_index=args.target_index,
        only_articulation_samples=args.only_articulation_samples,
        include_articulation_samples=args.include_articulation_samples,
        articulation_sample_id=args.articulation_sample_id,
    )
    selected_job_ids = _parse_pairs_file(args.pairs_file)
    for job in _iter_jobs(runner_args):
        if selected_job_ids is not None and _job_pair_ids(job).isdisjoint(
            selected_job_ids
        ):
            continue
        (
            target_obj_path,
            _target_bps_obj_path,
            target_demo_path,
            _target_transform_info,
            _target_scale,
        ) = _validate_job_inputs(job, require_transform=args.require_transform)
        if target_obj_path is None:
            if target_demo_path is None:
                raise ValueError(
                    f"{job.job_id} has neither target_obj_path nor target_demo_path."
                )
            target_obj_path = _resolve_oakink_demo_mesh_path(target_demo_path)
            if target_obj_path is None:
                continue
        yield job, Path(target_obj_path)


def _run_single(args: argparse.Namespace, cfg: MeshRepairConfig) -> int:
    input_path = _relative_or_abs_path(args.input)
    output_path = (
        Path(args.output)
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_solid_watertight.obj")
    )
    result = repair_mesh_file(input_path, output_path, cfg)
    print(json.dumps(result, indent=2))
    return 0


def _run_category_batch(args: argparse.Namespace, cfg: MeshRepairConfig) -> int:
    output_root = _relative_or_abs_path(args.output_root)
    manifest = []
    for job, input_path in _category_jobs(args):
        output_path = _checked_mesh_output_path(output_root, job.category, job.job_id)
        try:
            result = repair_mesh_file(input_path, output_path, cfg)
            result["category"] = job.category
            result["job_id"] = job.job_id
            result["pair_ids"] = sorted(_job_pair_ids(job))
            print(
                f"[{result['status']}] {result['pair_ids'][0]}: "
                f"{input_path} -> {output_path}"
            )
            manifest.append(result)
        except Exception as exc:
            if not args.keep_going:
                raise
            failure = {
                "category": job.category,
                "job_id": job.job_id,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "status": "failed",
                "error": str(exc),
            }
            print(f"[failed] {job.job_id}: {exc}")
            manifest.append(failure)

    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {"config": asdict(cfg), "meshes": manifest},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved manifest: {manifest_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert a disconnected mesh into a connected watertight mesh."
    )
    parser.add_argument("--input", default=None, help="Path to one input mesh file")
    parser.add_argument("--output", default=None, help="Output OBJ path for --input")
    parser.add_argument(
        "--output-root",
        default="logs/precomputed_checked_meshes",
        help="Root directory for category batch checked meshes.",
    )
    parser.add_argument("--pitch", type=float, default=None, help="Voxel size")
    parser.add_argument(
        "--pitch-divisions",
        type=int,
        default=600,
        help="Use mesh.scale / pitch_divisions when --pitch is omitted.",
    )
    parser.add_argument(
        "--closing-iterations",
        type=int,
        default=1,
        help="Number of morphological closing iterations",
    )
    parser.add_argument("--dilation-iterations", type=int, default=1)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--min-volume", type=float, default=1e-14)
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=25_000_000,
        help=(
            "Maximum estimated voxel grid cells allowed before repair. This prevents "
            "OOM kills during trimesh voxelization."
        ),
    )
    parser.add_argument(
        "--auto-coarsen",
        action="store_true",
        help="Increase pitch automatically when the estimated voxel grid is too large.",
    )
    parser.add_argument("--skip-valid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--section", action="append", default=[])
    parser.add_argument("--source-index", type=int, default=None)
    parser.add_argument("--target-index", type=int, default=None)
    parser.add_argument("--pairs-file", default=None)
    parser.add_argument("--require-transform", action="store_true")
    parser.add_argument("--include-articulation-samples", action="store_true")
    parser.add_argument("--only-articulation-samples", action="store_true")
    parser.add_argument("--articulation-sample-id", action="append", default=[])
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    cfg = MeshRepairConfig(
        pitch=args.pitch,
        pitch_divisions=args.pitch_divisions,
        closing_iterations=args.closing_iterations,
        dilation_iterations=args.dilation_iterations,
        padding=args.padding,
        min_volume=args.min_volume,
        max_voxels=args.max_voxels,
        auto_coarsen=args.auto_coarsen,
        skip_valid=args.skip_valid,
        overwrite=args.overwrite,
    )
    if args.input is not None:
        return _run_single(args, cfg)
    return _run_category_batch(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
