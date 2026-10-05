"""
Create a flat writing board OBJ.

Geometry:
- Overall size: 50 cm x 30 cm
- Writing area thickness: 5 mm
- Boundary width: 1 cm
- Boundary thickness: 1 cm
- Bottom lies on z=0
- Writing surface top is z=0.005
- Boundary top is z=0.010

Output:
- writing_board.obj
- writing_board.mtl

This script writes a simple Genesis-friendly OBJ:
- triangle faces only
- no UVs
- no normals
- simple `f v1 v2 v3` syntax
"""

from pathlib import Path

# Optional import so this file is still useful in trimesh-based workflows.
# The final OBJ writer below is manual to keep the mesh parser-safe.
import trimesh  # noqa: F401

TOTAL_X = 0.50
TOTAL_Y = 0.30
WRITING_THICKNESS = 0.01
BORDER_WIDTH = 0.01
BORDER_THICKNESS = 0.02

INNER_X = TOTAL_X - 2 * BORDER_WIDTH
INNER_Y = TOTAL_Y - 2 * BORDER_WIDTH


def box_vertices_faces(cx, cy, zmin, sx, sy, sz):
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = zmin, zmin + sz

    verts = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]

    faces = [
        (1, 2, 3),
        (1, 3, 4),
        (5, 7, 6),
        (5, 8, 7),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 8),
        (3, 8, 4),
        (4, 8, 5),
        (4, 5, 1),
    ]

    return verts, faces


def main():
    out_dir = Path(".")
    obj_path = out_dir / "writing_board.obj"
    mtl_path = out_dir / "writing_board.mtl"

    parts = [
        (
            "writing_area_black",
            "black_writing_area",
            0.0,
            0.0,
            0.0,
            INNER_X,
            INNER_Y,
            WRITING_THICKNESS,
        ),
        (
            "border_top_wood",
            "wood_boundary",
            0.0,
            TOTAL_Y / 2 - BORDER_WIDTH / 2,
            0.0,
            TOTAL_X,
            BORDER_WIDTH,
            BORDER_THICKNESS,
        ),
        (
            "border_bottom_wood",
            "wood_boundary",
            0.0,
            -TOTAL_Y / 2 + BORDER_WIDTH / 2,
            0.0,
            TOTAL_X,
            BORDER_WIDTH,
            BORDER_THICKNESS,
        ),
        (
            "border_left_wood",
            "wood_boundary",
            -TOTAL_X / 2 + BORDER_WIDTH / 2,
            0.0,
            0.0,
            BORDER_WIDTH,
            INNER_Y,
            BORDER_THICKNESS,
        ),
        (
            "border_right_wood",
            "wood_boundary",
            TOTAL_X / 2 - BORDER_WIDTH / 2,
            0.0,
            0.0,
            BORDER_WIDTH,
            INNER_Y,
            BORDER_THICKNESS,
        ),
    ]

    mtl_path.write_text(
        """# Materials for writing_board.obj

newmtl black_writing_area
Ka 0.000 0.000 0.000
Kd 0.005 0.005 0.005
Ks 0.050 0.050 0.050
Ns 20.000

newmtl wood_boundary
Ka 0.250 0.120 0.040
Kd 0.550 0.280 0.100
Ks 0.120 0.080 0.040
Ns 12.000
"""
    )

    lines = [
        "# Genesis-safe writing board OBJ",
        "# Dimensions: 0.50 m x 0.30 m",
        "# Bottom lies on z=0",
        "mtllib writing_board.mtl",
    ]

    vertex_offset = 0
    for name, mat, cx, cy, zmin, sx, sy, sz in parts:
        verts, faces = box_vertices_faces(cx, cy, zmin, sx, sy, sz)
        lines.append(f"\no {name}")
        lines.append(f"usemtl {mat}")

        for x, y, z in verts:
            lines.append(f"v {x:.8f} {y:.8f} {z:.8f}")

        for face in faces:
            a, b, c = (idx + vertex_offset for idx in face)
            lines.append(f"f {a} {b} {c}")

        vertex_offset += len(verts)

    obj_path.write_text("\n".join(lines) + "\n")

    print(f"Saved {obj_path}")
    print(f"Saved {mtl_path}")


if __name__ == "__main__":
    main()
