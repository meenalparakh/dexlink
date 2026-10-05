"""Generate the visual-only checker plane used by Genesis scenes."""

from __future__ import annotations

from pathlib import Path


def write_plane(
    output_directory: Path,
    *,
    size: float = 10.0,
    squares: int = 20,
) -> None:
    """Write a triangulated checker plane without UVs or normals."""
    if size <= 0.0:
        raise ValueError(f"size must be positive, received {size}")
    if squares <= 0:
        raise ValueError(f"squares must be positive, received {squares}")

    output_directory.mkdir(parents=True, exist_ok=True)
    mesh_path = output_directory / "white_checkered_plane_safe.obj"
    material_path = output_directory / "white_checkered_plane_safe.mtl"

    half_size = size / 2.0
    square_size = size / squares
    vertices: list[tuple[float, float, float]] = []
    faces_by_material: dict[str, list[tuple[int, int, int]]] = {
        "white": [],
        "light_gray": [],
    }
    for row in range(squares):
        for column in range(squares):
            x_min = -half_size + column * square_size
            x_max = x_min + square_size
            y_min = -half_size + row * square_size
            y_max = y_min + square_size
            first_vertex = len(vertices) + 1
            vertices.extend(
                [
                    (x_min, y_min, 0.0),
                    (x_max, y_min, 0.0),
                    (x_max, y_max, 0.0),
                    (x_min, y_max, 0.0),
                ]
            )
            material = "white" if (row + column) % 2 == 0 else "light_gray"
            faces_by_material[material].extend(
                [
                    (first_vertex, first_vertex + 1, first_vertex + 2),
                    (first_vertex, first_vertex + 2, first_vertex + 3),
                ]
            )

    lines = [
        "# DexLink-generated Genesis checker plane",
        f"mtllib {material_path.name}",
    ]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    for material, faces in faces_by_material.items():
        lines.append(f"usemtl {material}")
        lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
    mesh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    material_path.write_text(
        """newmtl white
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks 0.0 0.0 0.0
d 1.0
illum 1

newmtl light_gray
Ka 0.86 0.86 0.86
Kd 0.86 0.86 0.86
Ks 0.0 0.0 0.0
d 1.0
illum 1
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_plane(Path(__file__).resolve().parent)
