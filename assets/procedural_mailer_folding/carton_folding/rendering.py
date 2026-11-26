"""Headless, orthographic software rendering of actual URDF box geometry.

No OpenGL or display server is needed. FK is supplied by Model (urdfpy by
default). The section view clips the SAME transformed boxes at X=W/2;
it is not an artist's impression of a slot.
"""

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
    eye = np.array([0.8, 0.60, 1.6])
    eye /= np.linalg.norm(eye)
    right = np.cross((0, 1, 0), eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    basis = np.column_stack((right, up, eye))
    records = []
    for box, T in model.box_poses(cfg):
        vertices = box.vertices() @ T[:3, :3].T + T[:3, 3]
        camera = vertices @ basis
        records.append((vertices, camera, box.color))
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
    for vertices, camera, rgba in records:
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
                edges.append((xy[ia], xy[ib], camera[ia, 2], camera[ib, 2], edgecol))
    for edge in edges:
        overlay_edge(image, zbuffer, *edge)
    return Image.fromarray(image)


def legend(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 14) -> None:
    entries = [
        ((199, 163, 112), "Panels"),
        ((201, 89, 36), "Margin strips"),
        ((56, 122, 130), "Receiver keepers"),
        ((235, 178, 79), "Tuck tabs"),
    ]
    for color, label in entries:
        draw.rounded_rectangle((x, y + 2, x + 12, y + 14), radius=2, fill=color)
        draw.text((x + 20, y - 1), label, font=font(size), fill=MUTED)
        x += int(draw.textlength(label, font=font(size))) + 48


def full_frame(
    model: Model, motion: Motion, t: float, width=960, height=720
) -> Image.Image:
    cfg, label, step = motion.evaluate(t)
    im = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(im)
    draw.text((32, 24), "FOLDING CARTON", font=font(26, True), fill=INK)
    p = motion.p
    dims = " x ".join(f"{p[k]*1000:g}" for k in ("width", "depth", "height"))
    draw.text(
        (32, 61),
        f"Nominal {dims} mm  /  {p['thickness']*1000:.3g} mm board  /  open tuck slots",
        font=font(14),
        fill=MUTED,
    )
    draw.line((32, 88, width - 32, 88), fill=(218, 222, 223))
    scene = scene_image(model, cfg, width, height - 225)
    im.paste(scene, (0, 96))
    draw = ImageDraw.Draw(im)
    draw.text(
        (32, height - 124), f"{step+1:02d}   {label}", font=font(20, True), fill=INK
    )
    draw.text(
        (width - 151, height - 120),
        f"{t:04.1f} / {motion.total:.1f} s",
        font=font(14),
        fill=MUTED,
    )
    draw.rounded_rectangle(
        (32, height - 88, width - 32, height - 83), radius=2, fill=(219, 224, 225)
    )
    progress = max(1, int((width - 64) * t / motion.total))
    draw.rounded_rectangle(
        (32, height - 88, 32 + progress, height - 83), radius=2, fill=(65, 97, 104)
    )
    legend(draw, 32, height - 64, 13)
    draw.text(
        (32, height - 28),
        f"FK: {model.backend}  |  Root fixed  |  Kinematic preview, not contact dynamics",
        font=font(12),
        fill=MUTED,
    )
    return im


def section_frame(
    model: Model, motion: Motion, t: float, width=840, height=700
) -> Image.Image:
    cfg, label, _ = motion.evaluate(t)
    im = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(im)
    draw.text(
        (30, 22), "TUCK INSERTION / TRUE-SCALE SECTION", font=font(23, True), fill=INK
    )
    draw.text(
        (30, 58),
        "Top receiver, section through the tab center. Spacer ears lie outside this plane.",
        font=font(13),
        fill=MUTED,
    )
    p, d = motion.p, motion.d
    H, D, W, thickness = p["height"], p["depth"], p["width"], p["thickness"]
    left, top, vw, vh = 26, 93, width - 52, height - 208
    view = Image.new("RGB", (vw, vh), (252, 252, 250))
    vd = ImageDraw.Draw(view)
    ymin = H - p["keeper_length"] - 3 * thickness
    ymax = (
        H
        + p["tab_length"]
        + d["entry_tip_clearance"]
        + 2 * d["tuck_margin"]
        + 2 * thickness
    )
    zcenter = D - d["socket_axis_spacing"]
    scale = (vh - 28) / (ymax - ymin)

    def xy(z, y):
        return ((z - zcenter) * scale + vw / 2, (ymax - y) * scale + 14)

    for box, T in model.box_poses(cfg):
        vertices = box.vertices() @ T[:3, :3].T + T[:3, 3]
        # At this phase, the participating boxes have one axis parallel to
        # world X. General plane/edge intersection keeps the section exact.
        if vertices[:, 0].min() - 1e-9 > W / 2 or vertices[:, 0].max() + 1e-9 < W / 2:
            continue
        edges = set()
        for quad in QUADS:
            for ia, ib in zip(quad, np.roll(quad, -1)):
                edges.add(tuple(sorted((int(ia), int(ib)))))
        intersection = []
        for ia, ib in edges:
            a, b = vertices[ia], vertices[ib]
            dx = b[0] - a[0]
            if abs(dx) < 1e-12:
                continue
            u = (W / 2 - a[0]) / dx
            if -1e-9 <= u <= 1 + 1e-9:
                intersection.append(a + u * (b - a))
        if len(intersection) < 3:
            continue
        yz = np.unique(np.round(np.array(intersection)[:, [2, 1]], 12), axis=0)
        if len(yz) < 3:
            continue
        c = yz.mean(axis=0)
        yz = yz[np.argsort(np.arctan2(yz[:, 1] - c[1], yz[:, 0] - c[0]))]
        color = tuple(np.clip(box.color[:3] * 255, 0, 255).astype(int))
        coords = [xy(z, y) for z, y in yz]
        vd.polygon(coords, fill=color, outline=tuple(int(x * 0.65) for x in color))
    # Mouth reference line and labels, all geometry remains at true scale.
    _, mouth_y = xy(D, H)
    vd.line((35, mouth_y, vw - 35, mouth_y), fill=(194, 201, 202), width=1)
    vd.text((14, mouth_y - 21), "slot mouth", font=font(12), fill=MUTED)
    for name, z, offset in (
        ("Keeper", D - d["socket_axis_spacing"], -126),
        ("Receiving wall", D, 31),
    ):
        sx, sy = xy(z, H - 0.5 * p["tab_length"])
        vd.line((sx, sy, sx + offset, sy), fill=INK, width=1)
        vd.text(
            (sx + offset - (80 if offset < 0 else 0), sy - 20),
            name,
            font=font(13, True),
            fill=INK,
        )
    # Width dimension is below the seated tab tip, inside the long receiver.
    x0, yy = xy(
        D - d["socket_axis_spacing"] + thickness / 2,
        H - (p["tab_length"] + p["keeper_length"]) / 2,
    )
    x1, _ = xy(D - thickness / 2, H - (p["tab_length"] + p["keeper_length"]) / 2)
    vd.line((x0, yy, x1, yy), fill=INK, width=2)
    vd.line((x0, yy - 6, x0, yy + 6), fill=INK)
    vd.line((x1, yy - 6, x1, yy + 6), fill=INK)
    vd.text(
        (x1 + 18, yy - 8),
        f'{d["slot_clear_gap"]*1000:.3g} mm clear slot',
        font=font(13),
        fill=INK,
    )
    # Scale bar, so thickness is visibly not exaggerated.
    bx, by = 24, vh - 32
    target = 0.12 * (ymax - ymin)
    exponent = 10 ** math.floor(math.log10(target))
    bar = min(
        (1 * exponent, 2 * exponent, 5 * exponent, 10 * exponent),
        key=lambda v: abs(v - target),
    )
    vd.line((bx, by, bx + bar * scale, by), fill=INK, width=3)
    vd.text((bx, by - 22), f"{bar*1000:g} mm", font=font(12), fill=MUTED)
    im.paste(view, (left, top))
    draw = ImageDraw.Draw(im)
    draw.rectangle((left, top, left + vw, top + vh), outline=(219, 224, 225))
    draw.text((30, height - 92), label, font=font(19, True), fill=INK)
    draw.text(
        (30, height - 60),
        f'{thickness*1000:.3g} mm tab  |  {p["face_clearance"]*1000:.3g} mm clearance per face  |  {p["tab_length"]*1000:.3g} mm seated depth',
        font=font(14),
        fill=MUTED,
    )
    draw.text(
        (30, height - 29),
        f"Geometry and poses from the URDF  |  FK: {model.backend}  |  No thickness exaggeration",
        font=font(12),
        fill=MUTED,
    )
    return im


def contact_sheet(model: Model, motion: Motion, path: Path) -> None:
    # Each tile has its own fitted camera. They are not scale-comparison views.
    phases = [
        (0, "01  Flat net"),
        (motion.bounds[2], "02  Deploy receivers"),
        (motion.bounds[6], "03  Wall tube + offset seam"),
        (motion.bounds[10], "04  Bottom tab seated"),
        (motion.bounds[13], "05  Top tab aligned"),
        (motion.total, "06  Assembled carton"),
    ]
    tw, th = 480, 348
    canvas = Image.new("RGB", (3 * tw, 2 * th + 136), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 22), "FOLDING CARTON / GEOMETRY PREVIEW", font=font(29, True), fill=INK
    )
    draw.text(
        (28, 64),
        f"{len(model.links)} links   /   {len(model.joints)} revolute joints   /   {motion.p['thickness']*1000:.3g} mm cardboard   /   physical insertion gaps",
        font=font(16),
        fill=MUTED,
    )
    for i, (t, label) in enumerate(phases):
        x, y = (i % 3) * tw, 100 + (i // 3) * th
        tile = scene_image(model, motion.evaluate(float(t))[0], tw, th - 43)
        canvas.paste(tile, (x, y))
        draw.text((x + 24, y + th - 36), label, font=font(16, True), fill=INK)
    draw.text(
        (28, canvas.height - 28),
        f"FK backend: {model.backend}. Scripted rigid-body motion; not a dynamics simulation. Panels are fitted independently in each view.",
        font=font(13),
        fill=MUTED,
    )
    canvas.save(path)


def flat_net_svg(model: Model, motion: Motion, path: Path) -> None:
    """Export exact zero-pose panel projection and hinge centers, not a die line."""
    import xml.etree.ElementTree as ET

    records = []
    for box, pose in model.box_poses(motion.zero):
        verts = box.vertices() @ pose[:3, :3].T + pose[:3, 3]
        lo, hi = verts[:, :2].min(axis=0), verts[:, :2].max(axis=0)
        records.append((box, lo * 1000, hi * 1000))
    low = np.min([v[1] for v in records], axis=0)
    high = np.max([v[2] for v in records], axis=0)
    pad = 12.0
    vw, vh = high - low + 2 * pad
    root = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        viewBox=f"0 0 {vw} {vh}",
        width=f"{vw}mm",
        height=f"{vh}mm",
    )
    ET.SubElement(root, "title").text = (
        "Simulation carton: zero-pose geometry (not a manufacturing die line)"
    )
    ET.SubElement(
        root, "rect", x="0", y="0", width=str(vw), height=str(vh), fill="white"
    )

    def project(x, y):
        return x - low[0] + pad, high[1] - y + pad

    for box, lo, hi in records:
        x, y = project(lo[0], hi[1])
        color = "#" + "".join(f"{int(255*v):02x}" for v in box.color[:3])
        node = ET.SubElement(
            root,
            "rect",
            x=str(x),
            y=str(y),
            width=str(hi[0] - lo[0]),
            height=str(hi[1] - lo[1]),
            fill=color,
            stroke="#493827",
            **{"stroke-width": "0.12"},
        )
        ET.SubElement(node, "title").text = box.link
        if min(hi - lo) > 5:
            text = ET.SubElement(
                root,
                "text",
                x=str(x + (hi[0] - lo[0]) / 2),
                y=str(y + (hi[1] - lo[1]) / 2),
                fill="#222222",
                **{
                    "font-size": "2.2",
                    "font-family": "sans-serif",
                    "text-anchor": "middle",
                },
            )
            text.text = box.link
    fk = model.fk(motion.zero)
    for joint in model.joints:
        pose = fk[joint.parent] @ joint.pose
        center = pose[:3, 3] * 1000
        axis = pose[:3, :3] @ joint.axis
        A, B = center - 2 * axis, center + 2 * axis
        x1, y1 = project(A[0], A[1])
        x2, y2 = project(B[0], B[1])
        line = ET.SubElement(
            root,
            "line",
            x1=str(x1),
            y1=str(y1),
            x2=str(x2),
            y2=str(y2),
            stroke="#c12d20",
            **{"stroke-width": "0.35"},
        )
        ET.SubElement(line, "title").text = joint.name
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
