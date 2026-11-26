"""Software z-buffer renderer and true-plane sections of URDF box geometry."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from kinematics import Model
from motion import Motion
from PIL import Image, ImageDraw, ImageFont

QUADS = np.array(
    [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5]]
)
BG = (246, 247, 247)
INK = (34, 44, 49)
MUTED = (102, 112, 116)


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def raster_triangle(
    image: np.ndarray,
    zbuffer: np.ndarray,
    triangle: np.ndarray,
    depth: np.ndarray,
    color: np.ndarray,
) -> None:
    height, width = zbuffer.shape
    xmin = max(0, int(math.floor(triangle[:, 0].min())))
    xmax = min(width - 1, int(math.ceil(triangle[:, 0].max())))
    ymin = max(0, int(math.floor(triangle[:, 1].min())))
    ymax = min(height - 1, int(math.ceil(triangle[:, 1].max())))
    if xmin > xmax or ymin > ymax:
        return
    x0, y0 = triangle[0]
    x1, y1 = triangle[1]
    x2, y2 = triangle[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denominator) < 1e-8:
        return
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5
    )
    u = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denominator
    v = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denominator
    w = 1 - u - v
    inside = (u >= -1e-8) & (v >= -1e-8) & (w >= -1e-8)
    z = u * depth[0] + v * depth[1] + w * depth[2]
    old = zbuffer[ymin : ymax + 1, xmin : xmax + 1]
    visible = inside & (z > old)
    old[visible] = z[visible]
    image[ymin : ymax + 1, xmin : xmax + 1][visible] = color


def overlay_edge(
    image: np.ndarray, depthbuffer: np.ndarray, pa, pb, za, zb, color, tol=3e-6
) -> None:
    length = int(np.linalg.norm(pb - pa) * 1.4) + 1
    u = np.linspace(0, 1, length)
    xy = pa[None, :] + u[:, None] * (pb - pa)[None, :]
    ij = np.round(xy).astype(int)
    z = za + u * (zb - za)
    h, w = depthbuffer.shape
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    ij, z = ij[ok], z[ok]
    visible = z >= depthbuffer[ij[:, 1], ij[:, 0]] - tol
    p = ij[visible]
    image[p[:, 1], p[:, 0]] = color


def scene_image(
    model: Model, cfg: dict[str, float], width: int, height: int
) -> Image.Image:
    eye = np.array([1.35, 1.75, 1.65])
    eye /= np.linalg.norm(eye)
    right = np.cross((0, 0, 1), eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    basis = np.column_stack((right, up, eye))
    records = []
    for box, T in model.box_poses(cfg):
        vertices = box.vertices() @ T[:3, :3].T + T[:3, 3]
        camera = vertices @ basis
        records.append((vertices, camera, box.color, box.link))
    points = np.concatenate([r[1] for r in records])
    low, high = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    center = (low + high) / 2
    scale = min((width - 100) / (high[0] - low[0]), (height - 65) / (high[1] - low[1]))
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = BG
    zbuffer = np.full((height, width), -np.inf)
    edges = []
    light = np.array([-0.35, 0.75, 1.0])
    light /= np.linalg.norm(light)
    for vertices, camera, rgba, link in records:
        xy = (camera[:, :2] - center) * scale
        xy[:, 0] += width / 2
        xy[:, 1] = height / 2 - xy[:, 1]
        for quad in QUADS:
            normal = np.cross(
                vertices[quad[1]] - vertices[quad[0]],
                vertices[quad[2]] - vertices[quad[0]],
            )
            normal /= np.linalg.norm(normal)
            if normal @ eye < -1e-10:
                continue
            # Flat illumination preserves a visible distinction between the
            # sheet faces and the 1 mm edges. Materials come from the URDF.
            strength = 0.64 + 0.32 * max(0.0, float(normal @ light))
            col = np.clip(rgba[:3] * 255 * strength + 8, 0, 255).astype(np.uint8)
            for triangle in (quad[[0, 1, 2]], quad[[0, 2, 3]]):
                raster_triangle(image, zbuffer, xy[triangle], camera[triangle, 2], col)
            edgecol = np.clip(col * 0.74, 0, 255).astype(np.uint8)
            for ia, ib in zip(quad, np.roll(quad, -1)):
                if link != "base":
                    edges.append(
                        (xy[ia], xy[ib], camera[ia, 2], camera[ib, 2], edgecol)
                    )
    for edge in edges:
        overlay_edge(image, zbuffer, *edge)
    return Image.fromarray(image)


def full_frame(model, motion, t, width=960, height=720, caption=None):
    q, label, k = motion.evaluate(t)
    if caption is not None:
        label = caption
    im = Image.new("RGB", (width, height), BG)
    dr = ImageDraw.Draw(im)
    dr.text((30, 20), "HINGED-LID MAILER", font=font(27, True), fill=INK)
    p = motion.p
    dims = " x ".join(f"{1000*p[n]:g}" for n in ("width", "depth", "height"))
    dr.text(
        (30, 59),
        f'{dims} mm nominal  /  {p["thickness"]*1000:g} mm board  /  procedural rigid-panel model',
        font=font(14),
        fill=MUTED,
    )
    dr.line((30, 86, width - 30, 86), fill=(217, 221, 222))
    im.paste(scene_image(model, q, width, height - 217), (0, 93))
    dr = ImageDraw.Draw(im)
    dr.text((30, height - 115), f"{k+1:02d}  {label}", font=font(19, True), fill=INK)
    dr.text(
        (width - 139, height - 111),
        f"{t:.1f} / {motion.total:.1f} s",
        font=font(13),
        fill=MUTED,
    )
    dr.rounded_rectangle(
        (30, height - 77, width - 30, height - 73), radius=2, fill=(218, 224, 225)
    )
    dr.rounded_rectangle(
        (30, height - 77, 30 + max(1, (width - 60) * t / motion.total), height - 73),
        radius=2,
        fill=(69, 100, 103),
    )
    x = 30
    for color, text in [
        ((199, 155, 96), "Panels"),
        ((199, 97, 43), "Margin strips"),
        ((82, 135, 133), "Liners / keeper"),
        ((232, 171, 74), "Tabs"),
    ]:
        dr.rectangle((x, height - 53, x + 11, height - 42), fill=color)
        dr.text((x + 18, height - 56), text, font=font(12), fill=MUTED)
        x += int(dr.textlength(text, font=font(12))) + 43
    dr.text(
        (30, height - 24),
        f"FK: {model.backend}  |  Base fixed  |  Scripted kinematics, not a contact-dynamics rollout",
        font=font(12),
        fill=MUTED,
    )
    return im


def plane_polygon(box, T, axis, coordinate):
    vertices = box.vertices() @ T[:3, :3].T + T[:3, 3]
    if (
        coordinate < vertices[:, axis].min() - 1e-9
        or coordinate > vertices[:, axis].max() + 1e-9
    ):
        return None
    edges = set()
    for quad in QUADS:
        for a, b in zip(quad, np.roll(quad, -1)):
            edges.add(tuple(sorted((int(a), int(b)))))
    points = []
    for a, b in edges:
        aa, bb = vertices[a], vertices[b]
        diff = bb[axis] - aa[axis]
        if abs(diff) < 1e-12:
            continue
        u = (coordinate - aa[axis]) / diff
        if -1e-9 <= u <= 1 + 1e-9:
            points.append(aa + u * (bb - aa))
    if len(points) < 3:
        return None
    points = np.unique(np.round(points, 12), axis=0)
    if len(points) < 3:
        return None
    dims = [i for i in range(3) if i != axis]
    v = points[:, dims]
    center = v.mean(axis=0)
    return points[np.argsort(np.arctan2(v[:, 1] - center[1], v[:, 0] - center[0]))]


def section_frame(model, motion, time, kind="base", width=880, height=690):
    q, label, k = motion.evaluate(time)
    p, d = motion.p, motion.d
    t = p["thickness"]
    W, D, H = p["width"], p["depth"], p["height"]
    if kind == "base":
        axis, coordinate = 1, d["lock_y_positions"][0]
        horizontal = lambda point: W / 2 - point[0]
        uc = d["rollover_spacing"] + d["foot_margin"]
        ulo, uhi = uc - 8 * t, uc + 8 * t
        zlo, zhi = -5 * t, 10 * t
        title = "FLOOR LOCK / TRUE GEOMETRIC SECTION"
        subtitle = "Right forward locking tongue. The floor aperture is absent from collision geometry."
        refs = [("Base", uc - 4 * t, 0), ("Locking tab", uc, -0.1 * t)]
        dim_z = -2.8 * t
        dim_start = uc - d["slot_clear_gap"] / 2
        dim_end = uc + d["slot_clear_gap"] / 2
        msg = f'{t*1000:g} mm tongue / {p["face_clearance"]*1000:g} mm per face / {t*500:g} mm below base underside'
    elif kind == "front":
        axis, coordinate = 0, 0
        horizontal = lambda point: D - point[1]
        uc = d["rollover_spacing"] / 2
        ulo, uhi = -7 * t, 12 * t
        zlo = H - d["front_keeper_length"] - 4 * t
        zhi = H + p["front_tab_length"] + 5 * t
        title = "FRONT TUCK / TRUE GEOMETRIC SECTION"
        subtitle = (
            "Central section: the two supporting spacer ears lie outside this plane."
        )
        refs = [
            ("Front wall", 0, H - p["front_tab_length"] * 0.7),
            ("Keeper", d["rollover_spacing"], H - p["front_tab_length"] * 0.7),
        ]
        dim_z = H - p["front_tab_length"] - 2.4 * t
        dim_start = t / 2
        dim_end = d["rollover_spacing"] - t / 2
        msg = f'{t*1000:g} mm flap / {p["face_clearance"]*1000:g} mm per face / {p["front_tab_length"]*1000:.3g} mm tip depth'
    elif kind == "wing":
        axis, coordinate = 1, D / 2
        horizontal = lambda point: W / 2 - point[0]
        ulo, uhi = -8 * t, 12 * t
        zlo = H - p["lid_wing_depth"] - 4 * t
        zhi = H + d["lid_lift"] + 5 * t
        title = "LID WING / OPEN SIDE CHANNEL"
        subtitle = (
            "Mid-depth section: no rollover bridge spans the wing insertion path."
        )
        refs = [
            ("Outer side wall", 0, H - p["lid_wing_depth"] * 0.6),
            ("Inner liner", d["rollover_spacing"], H - p["lid_wing_depth"] * 0.6),
        ]
        dim_z = H - p["lid_wing_depth"] - 1.5 * t
        dim_start = t / 2
        dim_end = d["rollover_spacing"] - t / 2
        msg = f'{t*1000:g} mm wing / {p["face_clearance"]*1000:g} mm face gap / side rollover channels remain open'
    else:
        raise ValueError("Unknown section kind")
    im = Image.new("RGB", (width, height), BG)
    dr = ImageDraw.Draw(im)
    dr.text((26, 21), title, font=font(23, True), fill=INK)
    dr.text((26, 58), subtitle, font=font(13), fill=MUTED)
    vw, vh = width - 52, height - 209
    view = Image.new("RGB", (vw, vh), (252, 252, 250))
    vd = ImageDraw.Draw(view)
    scale = min((vw - 55) / (uhi - ulo), (vh - 26) / (zhi - zlo))
    cx = (ulo + uhi) / 2
    cz = (zlo + zhi) / 2

    def xy(u, z):
        return (vw / 2 + (u - cx) * scale, vh / 2 - (z - cz) * scale)

    for box, T in model.box_poses(q):
        pts = plane_polygon(box, T, axis, coordinate)
        if pts is None:
            continue
        coords = [xy(horizontal(a), a[2]) for a in pts]
        col = tuple(np.clip(box.color[:3] * 255, 0, 255).astype(int))
        # Adjacent floor primitives form one rigid panel: do not draw spurious internal seams.
        vd.polygon(
            coords,
            fill=col,
            outline=None if box.link == "base" else tuple(int(c * 0.62) for c in col),
        )
    x0, yy = xy(dim_start, dim_z)
    x1, _ = xy(dim_end, dim_z)
    vd.line((x0, yy, x1, yy), fill=INK, width=2)
    vd.line((x0, yy - 6, x0, yy + 6), fill=INK)
    vd.line((x1, yy - 6, x1, yy + 6), fill=INK)
    vd.text(
        (x1 + 14, yy - 8),
        f'{d["slot_clear_gap"]*1000:g} mm air slot',
        font=font(13, True),
        fill=INK,
    )
    # Labels kept away from the exact-scale geometry.
    for j, (name, u, z) in enumerate(refs):
        sx, sy = xy(u, z)
        target = 35 if j == 0 else vw - 150
        vd.line((sx, sy, target + 10, sy), fill=INK, width=1)
        vd.text((target, sy - 23), name, font=font(13, True), fill=INK)
    bx, by = 24, vh - 25
    bar = 2 * t
    vd.line((bx, by, bx + bar * scale, by), fill=INK, width=3)
    vd.text((bx, by - 23), f"{bar*1000:g} mm", font=font(12), fill=MUTED)
    im.paste(view, (26, 93))
    dr = ImageDraw.Draw(im)
    dr.rectangle((26, 93, 26 + vw, 93 + vh), outline=(215, 221, 222))
    dr.text((26, height - 92), label, font=font(18, True), fill=INK)
    dr.text((26, height - 57), msg, font=font(13), fill=MUTED)
    dr.text(
        (26, height - 26),
        f"Actual transformed URDF geometry / FK: {model.backend} / No thickness exaggeration",
        font=font(12),
        fill=MUTED,
    )
    return im


def overview(model, motion, path):
    phases = [
        (0, "01  Flat simulation net"),
        (motion.bounds[4], "02  Walls and corner ears"),
        (motion.bounds[8], "03  Rollovers and floor locks"),
        (motion.open_time, "04  Assembled open tray"),
        (motion.bounds[14], "05  Lid wings entering channels"),
        (motion.total, "06  Front flap seated / closed"),
    ]
    tw, th = 480, 366
    im = Image.new("RGB", (3 * tw, 2 * th + 128), BG)
    dr = ImageDraw.Draw(im)
    dr.text((28, 20), "PROCEDURAL HINGED-LID MAILER", font=font(30, True), fill=INK)
    dr.text(
        (28, 65),
        f"{len(model.links)} links / {len(model.joints)} revolute joints / real floor slots and lid receivers / finite-thickness cardboard",
        font=font(16),
        fill=MUTED,
    )
    for j, (t, label) in enumerate(phases):
        x, y = j % 3 * tw, 100 + j // 3 * th
        im.paste(scene_image(model, motion.evaluate(float(t))[0], tw, th - 44), (x, y))
        dr.text((x + 20, y + th - 35), label, font=font(16, True), fill=INK)
    dr.text(
        (28, im.height - 25),
        f"FK: {model.backend}. Kinematic geometry preview, not dynamics. Each panel is fitted independently; views are not at a common scale.",
        font=font(13),
        fill=MUTED,
    )
    im.save(path)


def flat_net_svg(model, motion, path):
    """Top projection of actual solids and holes. Not a production cutting pattern."""
    import html

    rects = []
    for box, T in model.box_poses(motion.zero):
        vs = box.vertices() @ T[:3, :3].T + T[:3, 3]
        lo, hi = vs[:, :2].min(axis=0) * 1000, vs[:, :2].max(axis=0) * 1000
        rects.append((lo, hi, box))
    lows = np.array([a for a, _, _ in rects])
    highs = np.array([b for _, b, _ in rects])
    lo = lows.min(0) - 15
    hi = highs.max(0) + 15
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{lo[0]} {lo[1]} {hi[0]-lo[0]} {hi[1]-lo[1]}">',
        "<title>Hinged mailer: simulation geometry projection, not a manufacturing die line</title>",
        f'<rect x="{lo[0]}" y="{lo[1]}" width="{hi[0]-lo[0]}" height="{hi[1]-lo[1]}" fill="#f6f7f7"/>',
    ]
    for a, b, box in rects:
        color = "#" + "".join(f"{int(c*255):02x}" for c in box.color[:3])
        stroke = "none" if box.link == "base" else "#6c5437"
        lines.append(
            f'<rect x="{a[0]}" y="{a[1]}" width="{b[0]-a[0]}" height="{b[1]-a[1]}" fill="{color}" stroke="{stroke}" stroke-width="0.15"><title>{html.escape(box.link)}</title></rect>'
        )
    for x0, x1, y0, y1 in motion.meta["base_holes"]:
        lines.append(
            f'<rect x="{x0*1000}" y="{y0*1000}" width="{(x1-x0)*1000}" height="{(y1-y0)*1000}" fill="none" stroke="#243e45" stroke-width="0.2"/>'
        )
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines) + "\n")
