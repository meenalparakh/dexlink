from __future__ import annotations

import argparse
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class MeshInertia:
    mass: float
    com: np.ndarray
    inertia_at_com: np.ndarray


def _positive_mesh_volume(mesh: trimesh.Trimesh, mesh_path: str | Path) -> float:
    volume = float(mesh.volume)
    if volume > 0.0:
        return volume

    hull_volume = float(mesh.convex_hull.volume)
    if hull_volume > 0.0:
        warnings.warn(
            "Mesh produced non-positive signed volume; using convex hull volume "
            f"for inertia mass estimate: {mesh_path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return hull_volume
    raise ValueError(
        f"Mesh at {mesh_path} produced non-positive volume {volume} and convex "
        f"hull volume {hull_volume}. Check that the mesh has non-degenerate geometry."
    )


def _box_inertia_from_mesh_bounds(
    mesh: trimesh.Trimesh,
    mass: float,
    mesh_path: str | Path,
) -> MeshInertia:
    bounds = np.asarray(mesh.bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError(f"Mesh at {mesh_path} has invalid bounds: {bounds}.")

    extents = bounds[1] - bounds[0]
    if np.any(extents <= 0.0):
        raise ValueError(
            f"Mesh at {mesh_path} has non-positive bounding-box extents: {extents}."
        )
    x_extent, y_extent, z_extent = extents
    inertia_at_com = np.diag(
        [
            mass * (y_extent**2 + z_extent**2) / 12.0,
            mass * (x_extent**2 + z_extent**2) / 12.0,
            mass * (x_extent**2 + y_extent**2) / 12.0,
        ]
    )
    return MeshInertia(
        mass=mass,
        com=(bounds[0] + bounds[1]) / 2.0,
        inertia_at_com=inertia_at_com,
    )


def _valid_principal_moments(
    inertia_at_com: np.ndarray,
    mesh_path: str | Path,
    context: str,
) -> np.ndarray | None:
    inertia_at_com = np.asarray(inertia_at_com, dtype=float)
    if inertia_at_com.shape != (3, 3) or not np.all(np.isfinite(inertia_at_com)):
        warnings.warn(
            f"{context} inertia matrix is invalid for {mesh_path}; "
            f"matrix={inertia_at_com}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    inertia_at_com = 0.5 * (inertia_at_com + inertia_at_com.T)
    try:
        principal_moments = np.linalg.eigvalsh(inertia_at_com)
    except np.linalg.LinAlgError as exc:
        warnings.warn(
            f"{context} inertia eigenvalue decomposition failed for {mesh_path}: {exc}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if not np.all(np.isfinite(principal_moments)) or np.any(principal_moments <= 0.0):
        warnings.warn(
            f"{context} inertia is not physically valid for {mesh_path}: "
            f"principal moments={principal_moments.tolist()}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return principal_moments


def _mesh_inertia_or_none(
    mesh: trimesh.Trimesh,
    mesh_path: str | Path,
    density: float,
    mass: float,
    context: str,
) -> MeshInertia | None:
    com = np.asarray(mesh.center_mass, dtype=float)
    inertia_at_com = np.asarray(mesh.moment_inertia, dtype=float) * density
    if not np.all(np.isfinite(com)):
        warnings.warn(
            f"{context} center of mass is invalid for {mesh_path}: {com}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    inertia_at_com = 0.5 * (inertia_at_com + inertia_at_com.T)
    principal_moments = _valid_principal_moments(
        inertia_at_com,
        mesh_path,
        context,
    )
    if principal_moments is None:
        return None
    return MeshInertia(mass=mass, com=com, inertia_at_com=inertia_at_com)


def inertia_from_mesh(mesh_path: str | Path, density: float) -> MeshInertia:
    mesh = trimesh.load_mesh(mesh_path)
    volume = _positive_mesh_volume(mesh, mesh_path)
    mass = density * volume
    if mass <= 0.0:
        raise ValueError(
            f"Mesh at {mesh_path} produced non-positive mass {mass}. Check that the mesh is watertight and density is positive."
        )

    mesh_inertia = _mesh_inertia_or_none(
        mesh,
        mesh_path,
        density,
        mass,
        context="Mesh",
    )
    if mesh_inertia is not None:
        return mesh_inertia

    hull = mesh.convex_hull
    hull_inertia = _mesh_inertia_or_none(
        hull,
        mesh_path,
        density,
        mass,
        context="Convex hull",
    )
    if hull_inertia is not None:
        warnings.warn(
            f"Using convex-hull inertia fallback for {mesh_path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return hull_inertia

    warnings.warn(
        f"Using bounding-box inertia fallback for {mesh_path}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _box_inertia_from_mesh_bounds(mesh, mass, mesh_path)


def _resolve_mesh_path(urdf_dir: Path, mesh_filename: str) -> Path:
    if mesh_filename.startswith("package://"):
        mesh_filename = mesh_filename[len("package://") :]
    mesh_path = Path(mesh_filename)
    if not mesh_path.is_absolute():
        mesh_path = (urdf_dir / mesh_path).resolve()
    return mesh_path


def get_collision_mesh_path(link: ET.Element, urdf_dir: Path) -> Path | None:
    collision = link.find("collision")
    if collision is None:
        return None

    geometry = collision.find("geometry")
    if geometry is None:
        return None

    mesh = geometry.find("mesh")
    if mesh is None:
        return None

    filename = mesh.attrib.get("filename")
    if filename is None:
        return None
    return _resolve_mesh_path(urdf_dir, filename)


def write_inertia(link: ET.Element, mesh_inertia: MeshInertia) -> None:
    inertial = link.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(link, "inertial")

    mass = inertial.find("mass")
    if mass is None:
        mass = ET.SubElement(inertial, "mass")

    inertia = inertial.find("inertia")
    if inertia is None:
        inertia = ET.SubElement(inertial, "inertia")

    origin = inertial.find("origin")
    if origin is None:
        origin = ET.SubElement(inertial, "origin")

    total_mass = mesh_inertia.mass
    com = mesh_inertia.com
    inertia_matrix = mesh_inertia.inertia_at_com

    mass.attrib["value"] = str(total_mass)
    origin.attrib["xyz"] = f"{com[0]} {com[1]} {com[2]}"
    origin.attrib["rpy"] = "0 0 0"
    inertia.attrib["ixx"] = str(inertia_matrix[0, 0])
    inertia.attrib["ixy"] = str(inertia_matrix[0, 1])
    inertia.attrib["ixz"] = str(inertia_matrix[0, 2])
    inertia.attrib["iyy"] = str(inertia_matrix[1, 1])
    inertia.attrib["iyz"] = str(inertia_matrix[1, 2])
    inertia.attrib["izz"] = str(inertia_matrix[2, 2])


def recompute_urdf_inertia_from_density(
    input_urdf: str | Path,
    output_urdf: str | Path,
    density: float,
) -> Path:
    if density <= 0.0:
        raise ValueError(f"density must be positive, got {density}.")

    input_urdf = Path(input_urdf).resolve()
    output_urdf = Path(output_urdf).resolve()
    urdf_dir = input_urdf.parent

    tree = ET.parse(input_urdf)
    root = tree.getroot()

    for link in root.findall("link"):
        mesh_path = get_collision_mesh_path(link, urdf_dir)
        if mesh_path is None:
            continue
        spatial_inertia = inertia_from_mesh(mesh_path=mesh_path, density=density)
        write_inertia(link, spatial_inertia)

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_urdf)
    return output_urdf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute URDF inertial properties from collision meshes and a provided density."
    )
    parser.add_argument("--input-urdf", type=Path, required=True)
    parser.add_argument("--output-urdf", type=Path, required=True)
    parser.add_argument("--density", type=float, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = recompute_urdf_inertia_from_density(
        input_urdf=args.input_urdf,
        output_urdf=args.output_urdf,
        density=args.density,
    )
    print(f"Saved URDF with recomputed inertia to {output_path}")


if __name__ == "__main__":
    main()
