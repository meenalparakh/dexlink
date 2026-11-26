"""Procedural support geometry for ARCTIC reference preparation."""

from pathlib import Path
import numpy as np
import trimesh


def write_box_obj(path: Path, size: tuple[float, float, float]) -> Path:
    extents = np.asarray(size, dtype=np.float64).reshape(3)
    if np.any(extents <= 0.0):
        raise ValueError(f"Expected positive box extents, got {extents}.")
    if path.exists():
        existing = trimesh.load(path, force="mesh", process=False)
        if np.allclose(existing.extents, extents, atol=1e-6):
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=extents)
    mesh.export(path)
    return path
