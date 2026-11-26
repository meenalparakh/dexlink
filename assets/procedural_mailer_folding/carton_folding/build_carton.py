#!/usr/bin/env python3
"""Generate a rigid-panel, finite-thickness folding carton (SI units).

The flat net has four wall panels, two opposite tuck lids, four dust flaps,
a lap seam, and two deployable receivers. Every receiver is a double crease:
wall --R-- two separated spacer ears --R-- keeper. The central mouth is EMPTY.
All collision objects are boxes; no concave collision meshes are required.
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Parameters:
    """Lengths are in metres; W/D/H are nominal fold-line dimensions.

    ``None`` detail parameters are automatically sized, not fixed to the
    default box. Call ``resolved()`` to obtain all final numeric values.
    Explicit overrides are checked rather than silently clipped.
    """

    width: float = 0.160
    depth: float = 0.130
    height: float = 0.280
    thickness: float = 0.002
    face_clearance: float | None = None
    lateral_clearance: float | None = None
    tab_width: float | None = None
    tab_length: float | None = None
    keeper_length: float | None = None
    ear_width: float | None = None
    glue_length: float | None = None
    density: float = 700.0  # Illustrative homogeneous-sheet density, not measured.

    def resolved(self) -> Parameters:
        for key, value in asdict(self).items():
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive")
        for required in ("width", "depth", "height", "thickness", "density"):
            if getattr(self, required) is None:
                raise ValueError(f"{required} cannot be null")
        W, D, H, t = self.width, self.depth, self.height, self.thickness
        c = 0.30 * t if self.face_clearance is None else self.face_clearance
        lateral = (
            max(t, 0.0125 * W)
            if self.lateral_clearance is None
            else self.lateral_clearance
        )
        a = 2 * t + 2 * c
        L = D - a / 2
        if L <= a:
            raise ValueError("Depth must exceed 1.5 times the socket axis spacing")
        # At the most open aligned pose alpha=pi/2, the tab root can reach
        # a + sqrt(2*a*L-a*a). Reserve 20% of that excursion for approach.
        reach = a + math.sqrt(2 * a * L - a * a)
        entry = t
        tab = min((14 / 65) * D, 0.22 * H, 0.80 * reach - entry)
        if self.tab_length is not None:
            tab = self.tab_length
        ear = max(2 * t, 0.0625 * W) if self.ear_width is None else self.ear_width
        keeper = (
            tab + max(5 * t, 0.075 * D)
            if self.keeper_length is None
            else self.keeper_length
        )
        return replace(
            self,
            face_clearance=c,
            lateral_clearance=lateral,
            tab_width=0.65 * W if self.tab_width is None else self.tab_width,
            tab_length=tab,
            keeper_length=keeper,
            ear_width=ear,
            glue_length=(
                min(0.20 * W, 0.25 * D)
                if self.glue_length is None
                else self.glue_length
            ),
        )

    def derived(self) -> dict[str, float]:
        p = self.resolved()
        t, c = p.thickness, p.face_clearance
        return {
            "slot_clear_gap": t + 2 * c,
            "socket_axis_spacing": 2 * t + 2 * c,
            "lid_root_margin": 2 * t + 2 * c,
            "tuck_margin": 2 * t + 2 * c,
            "hinge_relief": 0.55 * t,
            "side_end_inset": t + c,
            "dust_depth_notch": 2 * t + 2 * c + t + p.lateral_clearance,
            "dust_length": p.width / 2 - max(2 * t, 0.025 * p.width),
            "lid_length": p.depth - (2 * t + 2 * c) / 2,
            "mouth_width": p.tab_width + 2 * p.lateral_clearance,
            "seam_axis_spacing": t + c,
            "entry_tip_clearance": t,
        }

    def validate(self) -> None:
        p = self.resolved()
        for key, value in asdict(p).items():
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must resolve to a finite positive number")
        d = p.derived()
        if p.width <= d["mouth_width"] + 2 * p.ear_width + 4 * d["hinge_relief"]:
            raise ValueError(
                "Tab + mouth clearance + socket ears do not fit wall width; "
                "increase width or reduce tab-width / ear-width / thickness"
            )
        if p.depth <= 2 * d["dust_depth_notch"] + 6 * p.thickness:
            raise ValueError("Depth is too small for the dust-flap end notches")
        if p.height <= 2 * (p.keeper_length + d["side_end_inset"]):
            raise ValueError(
                "Height is too small for the receivers; reduce keeper-length / tab-length"
            )
        if p.keeper_length <= p.tab_length:
            raise ValueError("keeper_length must exceed tab_length")
        if (
            p.tab_length <= 2 * d["hinge_relief"]
            or d["dust_length"] <= 2 * d["hinge_relief"]
        ):
            raise ValueError(
                "A tab or dust flap is too short relative to cardboard thickness"
            )
        if not d["hinge_relief"] < p.glue_length < p.depth - 2 * p.thickness:
            raise ValueError("Glue flap length must fit inside the side-wall depth")
        if d["seam_axis_spacing"] <= 2 * d["hinge_relief"]:
            raise ValueError(
                "face_clearance must exceed 0.1 * thickness so the seam strip survives hinge relief"
            )
        a, L = d["tuck_margin"], d["lid_length"]
        theta_min = math.asin(1 - a / L)
        max_tip_height = a + L * math.cos(theta_min) - p.tab_length
        if max_tip_height <= d["entry_tip_clearance"]:
            raise ValueError(
                "Tab too long for this margin/depth: shorten tab or increase face-clearance"
            )


def numbers(values) -> str:
    return " ".join(f"{float(x):.12g}" for x in values)


class Writer:
    def __init__(self, p: Parameters):
        self.p = p
        self.root = ET.Element("robot", name="finite_thickness_tuck_carton")
        self.root.append(
            ET.Comment(
                " SI units. Root: back_wall. All zero joints: flat net. "
                "The slot ears are separate collision boxes: do not convex-hull them together. "
            )
        )
        self.materials = {
            "kraft": (0.69, 0.52, 0.34, 1),
            "lid": (0.84, 0.69, 0.49, 1),
            "dust": (0.75, 0.61, 0.43, 1),
            "margin": (0.79, 0.35, 0.14, 1),
            "keeper": (0.22, 0.48, 0.51, 1),
            "tab": (0.92, 0.70, 0.31, 1),
            "glue": (0.43, 0.52, 0.48, 1),
        }
        for name, rgba in self.materials.items():
            material = ET.SubElement(self.root, "material", name=name)
            ET.SubElement(material, "color", rgba=numbers(rgba))
        self.joints: dict[str, dict] = {}
        self.links: list[str] = []

    def link(self, name: str, boxes: list[tuple], material: str) -> None:
        """boxes: [(center_xyz, size_xyz)]. Mass/inertia include parallel-axis terms."""
        node = ET.SubElement(self.root, "link", name=name)
        self.links.append(name)
        centers, sizes = [], []
        for center, size in boxes:
            center, size = np.asarray(center, float), np.asarray(size, float)
            if np.any(size <= 0):
                raise ValueError(f"Nonpositive box dimension in {name}: {size}")
            centers.append(center)
            sizes.append(size)
        centers, sizes = np.array(centers), np.array(sizes)
        masses = np.prod(sizes, axis=1) * self.p.density
        mass = float(masses.sum())
        com = (centers * masses[:, None]).sum(axis=0) / mass
        inertia = np.zeros((3, 3))
        for c, size, m in zip(centers, sizes, masses):
            diag = m / 12 * (np.sum(size**2) - size**2)
            r = c - com
            inertia += np.diag(diag) + m * ((r @ r) * np.eye(3) - np.outer(r, r))
        inertial = ET.SubElement(node, "inertial")
        ET.SubElement(inertial, "origin", xyz=numbers(com), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{mass:.12g}")
        ET.SubElement(
            inertial,
            "inertia",
            **{
                "ixx": f"{inertia[0,0]:.12g}",
                "iyy": f"{inertia[1,1]:.12g}",
                "izz": f"{inertia[2,2]:.12g}",
                "ixy": f"{inertia[0,1]:.12g}",
                "ixz": f"{inertia[0,2]:.12g}",
                "iyz": f"{inertia[1,2]:.12g}",
            },
        )
        for i, (center, size) in enumerate(zip(centers, sizes)):
            for kind in ("visual", "collision"):
                shape = ET.SubElement(node, kind, name=f"{name}_{kind}_{i}")
                ET.SubElement(shape, "origin", xyz=numbers(center), rpy="0 0 0")
                geometry = ET.SubElement(shape, "geometry")
                ET.SubElement(geometry, "box", size=numbers(size))
                if kind == "visual":
                    ET.SubElement(shape, "material", name=material)

    def joint(
        self,
        name: str,
        parent: str,
        child: str,
        xyz=(0, 0, 0),
        rpy=(0, 0, 0),
        axis=(1, 0, 0),
        lower=0.0,
        upper=math.pi / 2,
        damping=0.08,
        friction=0.003,
    ) -> None:
        node = ET.SubElement(self.root, "joint", name=name, type="revolute")
        ET.SubElement(node, "parent", link=parent)
        ET.SubElement(node, "child", link=child)
        ET.SubElement(node, "origin", xyz=numbers(xyz), rpy=numbers(rpy))
        ET.SubElement(node, "axis", xyz=numbers(axis))
        ET.SubElement(
            node,
            "limit",
            lower=f"{lower:.12g}",
            upper=f"{upper:.12g}",
            effort="0.35",
            velocity="2.0",
        )
        ET.SubElement(
            node,
            "dynamics",
            damping=f"{damping:.12g}",
            friction=f"{friction:.12g}",
        )
        self.joints[name] = dict(parent=parent, child=child, lower=lower, upper=upper)


def generate(p: Parameters, out: Path) -> tuple[Path, Path]:
    requested = asdict(p)
    p.validate()
    p = p.resolved()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d, w = p.derived(), Writer(p)
    W, D, H, t = p.width, p.depth, p.height, p.thickness
    r, s = d["hinge_relief"], d["socket_axis_spacing"]
    a, L, B = d["tuck_margin"], d["lid_length"], p.tab_width

    def rect(x0, x1, y0, y1):
        return ((0.5 * (x0 + x1), 0.5 * (y0 + y1), 0), (x1 - x0, y1 - y0, t))

    # Back at z=0; assembled front at z=D; local Y is carton height.
    w.link("back_wall", [rect(r, W - r, r, H - r)], "kraft")
    w.link(
        "right_wall",
        [rect(r, D - r, d["side_end_inset"] + r, H - d["side_end_inset"] - r)],
        "kraft",
    )
    w.link("front_wall", [rect(r, W - r, r, H - r)], "kraft")
    w.link(
        "left_wall",
        [rect(-D + r, -r, d["side_end_inset"] + r, H - d["side_end_inset"] - r)],
        "kraft",
    )
    w.joint(
        "fold_right_wall", "back_wall", "right_wall", xyz=(W, 0, 0), axis=(0, -1, 0)
    )
    w.joint(
        "fold_front_wall", "right_wall", "front_wall", xyz=(D, 0, 0), axis=(0, -1, 0)
    )
    w.joint("fold_left_wall", "back_wall", "left_wall", axis=(0, 1, 0))

    # Two-crease lap seam: a narrow straight extension moves the glue flap
    # outside the left wall before that flap turns 90 degrees.
    sg = d["seam_axis_spacing"]
    w.link("seam_margin", [rect(r, sg - r, r, H - r)], "margin")
    w.link("glue_flap", [rect(r, p.glue_length, r, H - r)], "glue")
    w.joint(
        "seam_margin_fold",
        "front_wall",
        "seam_margin",
        xyz=(W, 0, 0),
        axis=(0, -1, 0),
        lower=-math.pi / 6,
        upper=math.pi / 6,
    )
    w.joint(
        "glue_flap_fold", "seam_margin", "glue_flap", xyz=(sg, 0, 0), axis=(0, -1, 0)
    )

    # Top receiver belongs to FRONT; bottom receiver to BACK. Both are flat
    # in the initial net. A pi rotation around local Z mirrors a bottom part.
    for end, parent, y, rz in (
        ("top", "front_wall", H, 0),
        ("bottom", "back_wall", 0, math.pi),
    ):
        mouth = d["mouth_width"]
        # Critical: no central rectangular collision slab across this link!
        ear_boxes = [
            rect(-mouth / 2 - p.ear_width, -mouth / 2, r, s - r),
            rect(mouth / 2, mouth / 2 + p.ear_width, r, s - r),
        ]
        w.link(f"{end}_socket_margin", ear_boxes, "margin")
        w.link(
            f"{end}_socket_keeper",
            [
                rect(
                    -mouth / 2 - p.ear_width,
                    mouth / 2 + p.ear_width,
                    r,
                    p.keeper_length,
                )
            ],
            "keeper",
        )
        w.joint(
            f"{end}_socket_base",
            parent,
            f"{end}_socket_margin",
            xyz=(W / 2, y, 0),
            rpy=(0, 0, rz),
        )
        w.joint(
            f"{end}_socket_return",
            f"{end}_socket_margin",
            f"{end}_socket_keeper",
            xyz=(0, s, 0),
        )

    # Each lid has TWO independently articulated margin strips, one at its
    # root (raises the lid over the receiver ears), and one before its tab.
    for end, parent, y, rz in (
        ("top", "back_wall", H, 0),
        ("bottom", "front_wall", 0, math.pi),
    ):
        w.link(f"{end}_lid_margin", [rect(-W / 2 + r, W / 2 - r, r, a - r)], "margin")
        w.link(f"{end}_lid", [rect(-W / 2 + r, W / 2 - r, r, L - r)], "lid")
        w.link(f"{end}_tuck_margin", [rect(-B / 2, B / 2, r, a - r)], "margin")
        w.link(f"{end}_tuck_tab", [rect(-B / 2, B / 2, r, p.tab_length)], "tab")
        w.joint(
            f"{end}_lid_margin_fold",
            parent,
            f"{end}_lid_margin",
            xyz=(W / 2, y, 0),
            rpy=(0, 0, rz),
            lower=-math.pi / 6,
            upper=math.pi / 6,
        )
        w.joint(f"{end}_lid_fold", f"{end}_lid_margin", f"{end}_lid", xyz=(0, a, 0))
        w.joint(
            f"{end}_tuck_margin_fold",
            f"{end}_lid",
            f"{end}_tuck_margin",
            xyz=(0, L, 0),
            upper=math.radians(105),
        )
        w.joint(
            f"{end}_tuck_tab_fold",
            f"{end}_tuck_margin",
            f"{end}_tuck_tab",
            xyz=(0, a, 0),
        )

    # Dust flaps are inset in depth to leave the receiver and tab corridors
    # empty. Side-wall ends are stepped inward to avoid lid/dust overlap.
    n, inset = d["dust_depth_notch"], d["side_end_inset"]
    for side, cx in (("right", D / 2), ("left", -D / 2)):
        for end, y, rz in (("top", H - inset, 0), ("bottom", inset, math.pi)):
            name = f"{end}_{side}_dust"
            w.link(name, [rect(-D / 2 + n, D / 2 - n, r, d["dust_length"])], "dust")
            w.joint(
                name + "_fold", side + "_wall", name, xyz=(cx, y, 0), rpy=(0, 0, rz)
            )

    ET.indent(w.root, space="  ")
    urdf_path = out / "carton.urdf"
    ET.ElementTree(w.root).write(urdf_path, encoding="utf-8", xml_declaration=True)
    metadata = {
        "schema_version": 2,
        "units": {"length": "m", "angle": "rad", "mass": "kg"},
        "requested_parameters": requested,
        "parameters": asdict(p),
        "derived": d,
        "dimension_convention": "Nominal W/D wall-midplane separation; H is main-wall fold-line height. Not exact exterior dimensions.",
        "automatic_sizing": "Null detail parameters are recomputed from W/D/H/thickness and analytic insertion reachability.",
        "root_link": "back_wall",
        "links": w.links,
        "joints": w.joints,
        "description": "Rigid-panel approximation inspired by the supplied carton net; not a traced die line.",
        "socket_mouth_note": "Two separate ear collision boxes, NEVER one full-span convex hull.",
        "dynamics_note": "URDF damping and friction match the Genesis cardboard baseline; Genesis adds a stateful plastic crease controller.",
    }
    meta_path = out / "carton.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return urdf_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    parser.add_argument(
        "--config", type=Path, help="JSON object of parameter names; lengths in metres"
    )
    for name in Parameters.__dataclass_fields__:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=float,
            default=None,
            help="SI units. Omitted detail dimensions are automatically sized.",
        )
    args = parser.parse_args()
    try:
        values = {}
        if args.config is not None:
            values = json.loads(args.config.read_text())
            if not isinstance(values, dict):
                raise ValueError("Config must be a JSON object")
            unknown = set(values) - set(Parameters.__dataclass_fields__)
            if unknown:
                raise ValueError(f"Unknown config parameters: {sorted(unknown)}")
        values.update(
            {
                name: getattr(args, name)
                for name in Parameters.__dataclass_fields__
                if getattr(args, name) is not None
            }
        )
        files = generate(Parameters(**values), args.out)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    for path in files:
        print(path)
    meta = json.loads(files[1].read_text())
    pp, dd = meta["parameters"], meta["derived"]
    print(
        f"Nominal W/D/H: {pp['width']*1000:g} / {pp['depth']*1000:g} / {pp['height']*1000:g} mm"
    )
    print(
        f"Slot: {dd['slot_clear_gap']*1000:g} mm; tab: {pp['tab_length']*1000:g} mm long"
    )


if __name__ == "__main__":
    main()
