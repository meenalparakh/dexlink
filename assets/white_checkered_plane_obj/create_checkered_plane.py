from pathlib import Path

# Generates a checkerboard OBJ plane on z=0 using geometry/materials only.
# No texture files required.

out_dir = Path(__file__).resolve().parent
obj_path = out_dir / "white_checkered_plane.obj"
mtl_path = out_dir / "white_checkered_plane.mtl"

size = 10.0  # total width/height of plane in world units
squares = 20  # number of squares along each axis
z = 0.0
half = size / 2.0
step = size / squares

vertices = []
faces = []
materials = []

# Make each square its own quad so alternating materials work everywhere.
for iy in range(squares):
    for ix in range(squares):
        x0 = -half + ix * step
        x1 = x0 + step
        y0 = -half + iy * step
        y1 = y0 + step

        base = len(vertices) + 1
        vertices.extend(
            [
                (x0, y0, z),
                (x1, y0, z),
                (x1, y1, z),
                (x0, y1, z),
            ]
        )
        mat = "white" if (ix + iy) % 2 == 0 else "very_light_gray"
        faces.append((mat, base, base + 1, base + 2, base + 3))

obj_lines = [
    "# White checkered plane on z=0",
    f"mtllib {mtl_path.name}",
    "o white_checkered_plane",
]
for v in vertices:
    obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
obj_lines.append("vn 0.000000 0.000000 1.000000")

current_mat = None
for mat, a, b, c, d in faces:
    if mat != current_mat:
        obj_lines.append(f"usemtl {mat}")
        current_mat = mat
    # quad face with shared upward normal
    obj_lines.append(f"f {a}//1 {b}//1 {c}//1 {d}//1")

mtl_lines = [
    "# Materials for white checkered plane",
    "newmtl white",
    "Ka 1.000 1.000 1.000",
    "Kd 1.000 1.000 1.000",
    "Ks 0.000 0.000 0.000",
    "d 1.0",
    "illum 1",
    "",
    "newmtl very_light_gray",
    "Ka 0.850 0.850 0.850",
    "Kd 0.850 0.850 0.850",
    "Ks 0.000 0.000 0.000",
    "d 1.0",
    "illum 1",
]

obj_path.write_text("\n".join(obj_lines) + "\n")
mtl_path.write_text("\n".join(mtl_lines) + "\n")
print(obj_path)
print(mtl_path)
