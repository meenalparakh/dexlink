from pathlib import Path


def write_plane(
    path="white_checkered_plane_safe.obj",
    mtl_path="white_checkered_plane_safe.mtl",
    size=10.0,
    squares=20,
):
    path = Path(path)
    mtl_path = Path(mtl_path)
    half = size / 2
    step = size / squares
    verts = []
    faces_by_mat = {"white": [], "light_gray": []}
    for iy in range(squares):
        for ix in range(squares):
            x0 = -half + ix * step
            x1 = x0 + step
            y0 = -half + iy * step
            y1 = y0 + step
            base = len(verts) + 1
            verts.extend([(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)])
            mat = "white" if (ix + iy) % 2 == 0 else "light_gray"
            faces_by_mat[mat].append((base, base + 1, base + 2))
            faces_by_mat[mat].append((base, base + 2, base + 3))

    lines = ["# Genesis-safe checker plane", f"mtllib {mtl_path.name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    for mat, faces in faces_by_mat.items():
        lines.append(f"usemtl {mat}")
        lines += [f"f {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")

    mtl_path.write_text(
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
"""
    )


if __name__ == "__main__":
    write_plane()
