#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

DEFAULT_MAX_VERTICES = 100_000
MIN_FACE_COUNT = 4
DEFAULT_BOUNDARY_EDGE_TOLERANCE = 0.02


@dataclass(frozen=True)
class ReduceMeshConfig:
    mesh_path: Path
    max_vertices: int = DEFAULT_MAX_VERTICES
    output_path: Path | None = None
    force: bool = False
    boundary_edge_tolerance: float = DEFAULT_BOUNDARY_EDGE_TOLERANCE

    @property
    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        return self.mesh_path.with_name(f"{self.mesh_path.stem}_reduced.obj")


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh path does not exist: {mesh_path}")

    loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0
        ]
        if not geometries:
            raise ValueError(f"No triangle mesh geometry found in {mesh_path}.")
        mesh = trimesh.util.concatenate(geometries)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(
            f"Expected trimesh to load {mesh_path} as Trimesh or Scene, got {type(loaded)}."
        )

    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")
    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")
    return mesh.copy()


def _cleanup_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    cleaned = mesh.copy()
    cleaned.remove_unreferenced_vertices()
    cleaned.merge_vertices()
    cleaned.update_faces(cleaned.unique_faces())
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(cleaned)
    trimesh.repair.fix_normals(cleaned)
    return cleaned


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    edges = np.asarray(mesh.edges_sorted, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected sorted mesh edges shape (E, 2), got {edges.shape}.")
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.count_nonzero(counts == 1))


def _validate_no_new_holes(
    original: trimesh.Trimesh,
    reduced: trimesh.Trimesh,
    boundary_edge_tolerance: float,
) -> None:
    if boundary_edge_tolerance < 0.0:
        raise ValueError(
            f"`boundary_edge_tolerance` must be non-negative, got {boundary_edge_tolerance}."
        )

    if original.is_watertight:
        if not reduced.is_watertight:
            raise ValueError(
                "Reduction introduced holes: input mesh is watertight, but output is not."
            )
        return

    original_boundary_edges = _boundary_edge_count(original)
    reduced_boundary_edges = _boundary_edge_count(reduced)
    max_boundary_edges = int(original_boundary_edges * (1.0 + boundary_edge_tolerance))
    if reduced_boundary_edges > max_boundary_edges:
        raise ValueError(
            "Reduction introduced new open boundaries, which usually means holes. "
            f"Input boundary edges: {original_boundary_edges}; output boundary edges: "
            f"{reduced_boundary_edges}; allowed at most {max_boundary_edges}."
        )


def _simplify_to_vertex_cap(
    mesh: trimesh.Trimesh,
    max_vertices: int,
    boundary_edge_tolerance: float,
) -> trimesh.Trimesh:
    if max_vertices <= 0:
        raise ValueError(f"`max_vertices` must be positive, got {max_vertices}.")

    original = _cleanup_mesh(mesh)
    reduced = original.copy()
    if len(reduced.vertices) <= max_vertices:
        return reduced

    if not hasattr(reduced, "simplify_quadric_decimation"):
        raise AttributeError(
            "trimesh.Trimesh does not provide `simplify_quadric_decimation`. "
            "Install trimesh with simplification support before reducing meshes."
        )

    original_vertices = len(original.vertices)
    target_faces = max(
        MIN_FACE_COUNT,
        int(len(original.faces) * max_vertices / original_vertices),
    )
    last_error: ValueError | None = None

    while True:
        simplified = original.simplify_quadric_decimation(face_count=target_faces)
        if not isinstance(simplified, trimesh.Trimesh):
            raise TypeError(
                "Expected `simplify_quadric_decimation` to return trimesh.Trimesh, "
                f"got {type(simplified)}."
            )
        simplified = _cleanup_mesh(simplified)
        if len(simplified.vertices) <= max_vertices:
            try:
                _validate_no_new_holes(
                    original,
                    simplified,
                    boundary_edge_tolerance,
                )
            except ValueError as exc:
                last_error = exc
            else:
                return simplified

        if len(simplified.vertices) > max_vertices:
            next_target_faces = max(MIN_FACE_COUNT, int(target_faces * 0.8))
        else:
            next_target_faces = max(MIN_FACE_COUNT, int(target_faces * 0.9))

        if next_target_faces == target_faces:
            if last_error is not None:
                raise ValueError(
                    "Could not reduce the mesh below the requested vertex cap without "
                    "introducing holes. Try a higher --max-vertices value."
                ) from last_error
            raise ValueError(
                "Could not reduce mesh below the requested vertex cap: "
                f"got {len(simplified.vertices)} vertices, expected at most {max_vertices}."
            )
        if next_target_faces == MIN_FACE_COUNT and last_error is not None:
            raise ValueError(
                "Could not reduce the mesh below the requested vertex cap without "
                "introducing holes. Try a higher --max-vertices value."
            ) from last_error
        target_faces = next_target_faces


def reduce_mesh(config: ReduceMeshConfig) -> Path:
    mesh = _load_mesh(config.mesh_path)
    output_path = config.resolved_output_path
    if output_path.exists() and not config.force:
        raise FileExistsError(
            f"Output path already exists: {output_path}. Pass --force to overwrite it."
        )

    reduced = _simplify_to_vertex_cap(
        mesh,
        config.max_vertices,
        config.boundary_edge_tolerance,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reduced.export(str(output_path), file_type="obj")

    print(f"Input mesh: {config.mesh_path}")
    print(f"Input vertices: {len(mesh.vertices)}")
    print(f"Input faces: {len(mesh.faces)}")
    print(f"Input watertight: {_cleanup_mesh(mesh).is_watertight}")
    print(f"Input boundary edges: {_boundary_edge_count(_cleanup_mesh(mesh))}")
    print(f"Output mesh: {output_path}")
    print(f"Output vertices: {len(reduced.vertices)}")
    print(f"Output faces: {len(reduced.faces)}")
    print(f"Output watertight: {reduced.is_watertight}")
    print(f"Output boundary edges: {_boundary_edge_count(reduced)}")
    return output_path


def parse_args(argv: list[str] | None = None) -> ReduceMeshConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce a triangle mesh to at most the requested vertex count and "
            "write it as <mesh_stem>_reduced.obj."
        )
    )
    parser.add_argument("mesh_path", type=Path)
    parser.add_argument("--max-vertices", default=DEFAULT_MAX_VERTICES, type=int)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--boundary-edge-tolerance",
        default=DEFAULT_BOUNDARY_EDGE_TOLERANCE,
        type=float,
        help=(
            "Allowed fractional increase in open boundary edge count for non-watertight "
            "meshes. Keep this small for containers like cups."
        ),
    )
    args = parser.parse_args(argv)
    return ReduceMeshConfig(
        mesh_path=args.mesh_path,
        max_vertices=args.max_vertices,
        output_path=args.output,
        force=args.force,
        boundary_edge_tolerance=args.boundary_edge_tolerance,
    )


def main(argv: list[str] | None = None) -> int:
    reduce_mesh(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
