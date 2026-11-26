#!/usr/bin/env python3
"""Procedural finite-thickness hinged-lid mailer. All distances are metres.

An approximate rigid-panel simulation net, not a manufacturer die line.
Side rollovers capture corner ears, four tongues enter holes in the base,
lid wings enter OPEN rollover channels, and a front tuck enters a keeper.
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from urdf_writer import Writer

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Parameters:
    width: float = 0.400
    depth: float = 0.300
    height: float = 0.090
    thickness: float = 0.003
    face_clearance: float | None = None
    lateral_clearance: float | None = None
    lock_width: float | None = None
    lid_wing_depth: float | None = None
    front_tab_width: float | None = None
    front_tab_length: float | None = None
    density: float = 450.0  # Illustrative bulk density, not measured corrugated stock.

    def resolved(self):
        for key, val in asdict(self).items():
            if val is not None and (not math.isfinite(val) or val <= 0):
                raise ValueError(f"{key} must be finite and positive")
        for key in ("width", "depth", "height", "thickness", "density"):
            if getattr(self, key) is None:
                raise ValueError(f"{key} cannot be null")
        W, D, H, t = self.width, self.depth, self.height, self.thickness
        c = 0.3 * t if self.face_clearance is None else self.face_clearance
        lat = t if self.lateral_clearance is None else self.lateral_clearance
        s = 2 * t + 2 * c
        L = D - s / 2
        if L <= s:
            raise ValueError("Depth must exceed 1.5 times the rollover spacing")
        reach = s + math.sqrt(2 * s * L - s * s)
        return replace(
            self,
            face_clearance=c,
            lateral_clearance=lat,
            lock_width=0.12 * D if self.lock_width is None else self.lock_width,
            lid_wing_depth=(
                0.60 * H if self.lid_wing_depth is None else self.lid_wing_depth
            ),
            front_tab_width=(
                0.70 * W if self.front_tab_width is None else self.front_tab_width
            ),
            front_tab_length=(
                min(0.52 * H, 0.72 * reach)
                if self.front_tab_length is None
                else self.front_tab_length
            ),
        )

    def derived(self):
        p = self.resolved()
        W, D, H, t = p.width, p.depth, p.height, p.thickness
        c, lat = p.face_clearance, p.lateral_clearance
        s = 2 * t + 2 * c
        dust = min(0.70 * H, 0.16 * D)
        mouth_end = dust + 2 * t
        wing_start = mouth_end + lat + t
        upper = D - mouth_end - lat - t
        r2 = upper * upper + s * s - p.lid_wing_depth**2
        if r2 <= 0:
            raise ValueError(
                "Lid wing depth is too large for the available side opening"
            )
        wing_end = math.sqrt(r2) - t
        return dict(
            slot_clear_gap=t + 2 * c,
            rollover_spacing=s,
            lid_lift=s,
            lid_front_margin=s,
            foot_margin=4 * t + 2 * c,
            hinge_relief=0.65 * t,
            foot_relief=t,
            side_end_inset=t + c,
            end_wall_width=W - s,
            liner_length=H - 2 * t,
            lock_axis_height=2 * t,
            lock_tab_length=3 * t,
            lock_tip_entry_height=1.1 * t,
            dust_length=dust,
            side_mouth_end=mouth_end,
            lid_length=D - s / 2,
            wing_start=wing_start,
            wing_end=wing_end,
            front_keeper_width=W - 2 * s - 2 * (t + c),
            front_keeper_length=p.front_tab_length + 4 * t,
            front_mouth_width=p.front_tab_width + 2 * lat,
            front_ear_width=max(3 * t, 0.045 * W),
            front_entry_tip_height=t,
            slot_x=W / 2 - s - (4 * t + 2 * c),
            lock_y_positions=[0.30 * D, 0.70 * D],
            lid_open_angle=math.radians(100),
        )

    def validate(self):
        p = self.resolved()
        d = p.derived()
        t = p.thickness
        if p.face_clearance <= 0.15 * t:
            raise ValueError(
                "face_clearance must exceed 0.15*thickness for hinge relief"
            )
        if p.height < 18 * t:
            raise ValueError("height must be at least 18*thickness")
        if (
            p.width
            <= 2 * (p.height + d["rollover_spacing"] + d["foot_margin"] + 3 * t) + 2 * t
        ):
            raise ValueError(
                "This shallow-mailer folding path requires width > "
                "2*(height + rollover_spacing + foot_margin + 3*thickness) + 2*thickness"
            )
        if d["front_keeper_length"] >= p.height - 4 * t:
            raise ValueError(
                "front_tab_length must leave at least 8*thickness below the keeper"
            )
        if (
            d["front_keeper_width"]
            <= d["front_mouth_width"] + 2 * d["front_ear_width"] + 2 * t
        ):
            raise ValueError(
                "Front tab, lateral clearances and receiver ears do not fit the width"
            )
        if d["wing_end"] <= d["wing_start"] + 8 * t:
            raise ValueError(
                "Depth too short / lid wings too deep for clear side sockets"
            )
        if p.lid_wing_depth >= p.height - 3 * t:
            raise ValueError("lid_wing_depth must be less than height - 3*thickness")
        if p.lock_width + 2 * p.lateral_clearance >= 0.32 * p.depth:
            raise ValueError(
                "lock_width plus clearances is too large for the two base slots"
            )
        if d["slot_x"] - (t + 2 * p.face_clearance) / 2 <= 2 * t:
            raise ValueError("Base slots do not fit")
        if d["side_mouth_end"] >= 0.40 * p.depth:
            raise ValueError("Corner ears occupy too much of the side opening")


def rect_box(x0, x1, y0, y1, t):
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid rectangle {(x0,x1,y0,y1)}")
    return ((0.5 * (x0 + x1), 0.5 * (y0 + y1), 0), (x1 - x0, y1 - y0, t))


def subtract_rectangles(rect, holes):
    """Disjoint positive-area rectangular partition of a sheet with true holes."""
    parts = [rect]
    for hx0, hx1, hy0, hy1 in holes:
        new = []
        for x0, x1, y0, y1 in parts:
            a, b = max(x0, hx0), min(x1, hx1)
            c, e = max(y0, hy0), min(y1, hy1)
            if a >= b or c >= e:
                new.append((x0, x1, y0, y1))
                continue
            for v in ((x0, a, y0, y1), (b, x1, y0, y1), (a, b, y0, c), (a, b, e, y1)):
                if v[1] - v[0] > 1e-12 and v[3] - v[2] > 1e-12:
                    new.append(v)
        parts = new
    return parts


def generate(params: Parameters, out: Path):
    requested = asdict(params)
    params.validate()
    p = params.resolved()
    d = p.derived()
    W, D, H, t = p.width, p.depth, p.height, p.thickness
    s, r, e = d["rollover_spacing"], d["hinge_relief"], d["side_end_inset"]
    F = d["end_wall_width"]
    A = d["foot_margin"]
    L = d["liner_length"]
    rf = d["foot_relief"]
    w = Writer(p)
    # Similar kraft hues for a readable physical preview; highlights for moving margins.
    colors = {
        "kraft": (0.71, 0.51, 0.29, 1),
        "lid": (0.81, 0.63, 0.39, 1),
        "dust": (0.74, 0.55, 0.32, 1),
        "margin": (0.78, 0.38, 0.17, 1),
        "keeper": (0.32, 0.53, 0.52, 1),
        "tab": (0.91, 0.67, 0.29, 1),
    }
    for el in w.root.findall("material"):
        if el.get("name") in colors:
            el.find("color").set("rgba", " ".join(map(str, colors[el.get("name")])))
    holes = []
    for sign in (-1, 1):
        for yy in d["lock_y_positions"]:
            xx = sign * d["slot_x"]
            g = d["slot_clear_gap"]
            b = p.lock_width + 2 * p.lateral_clearance
            holes.append((xx - g / 2, xx + g / 2, yy - b / 2, yy + b / 2))
    pieces = subtract_rectangles((-W / 2 + r, W / 2 - r, r, D - r), holes)
    w.link("base", [rect_box(*v, t) for v in pieces], "kraft")
    for end, y0, rz in [("rear", 0, math.pi), ("front", D, 0)]:
        name = f"{end}_wall"
        w.link(name, [rect_box(-F / 2 + r, F / 2 - r, r, H - r, t)], "kraft")
        w.joint(f"{end}_rise", "base", name, xyz=(0, y0, 0), rpy=(0, 0, rz))
        for label, sign in [("a", -1), ("b", 1)]:
            ear = f"{end}_corner_{label}"
            x0, x1 = sorted((sign * r, sign * d["dust_length"]))
            w.link(ear, [rect_box(x0, x1, e, H - e, t)], "dust")
            w.joint(
                f"{ear}_fold", name, ear, xyz=(sign * F / 2, 0, 0), axis=(0, -sign, 0)
            )
    for side, sign in [("left", -1), ("right", 1)]:
        outer = f"{side}_outer"
        cap = f"{side}_roll_margin"
        inner = f"{side}_liner"
        foot = f"{side}_foot_margin"
        x0, x1 = -D / 2 + e, D / 2 - e
        w.link(outer, [rect_box(x0, x1, r, H - r, t)], "kraft")
        w.joint(
            f"{side}_rise",
            "base",
            outer,
            xyz=(sign * W / 2, D / 2, 0),
            rpy=(0, 0, -sign * math.pi / 2),
        )
        cut = D / 2 - d["side_mouth_end"]
        # Two physical bridge ears, NOT a solid bridge across the lid-wing aperture.
        w.link(
            cap,
            [rect_box(x0, -cut, r, s - r, t), rect_box(cut, x1, r, s - r, t)],
            "margin",
        )
        w.joint(f"{side}_roll_base", outer, cap, xyz=(0, H, 0))
        w.link(inner, [rect_box(x0, x1, r, L - rf, t)], "keeper")
        w.joint(f"{side}_roll_return", cap, inner, xyz=(0, s, 0))
        w.link(foot, [rect_box(x0, x1, rf, A - rf, t)], "margin")
        w.joint(
            f"{side}_foot_fold", inner, foot, xyz=(0, L, 0), lower=-math.pi, upper=0
        )
        for k, yy in enumerate(d["lock_y_positions"]):
            tab = f"{side}_lock_{k}"
            # local hinge X maps to -sign * world Y in the all-zero orientation.
            local_x = -sign * (yy - D / 2)
            w.link(
                tab,
                [
                    rect_box(
                        -p.lock_width / 2, p.lock_width / 2, rf, d["lock_tab_length"], t
                    )
                ],
                "tab",
            )
            w.joint(f"{tab}_fold", foot, tab, xyz=(local_x, A, 0), upper=math.pi)
    # Front receiver: returning keeper hung from separated end ears.
    K = d["front_keeper_width"]
    M = d["front_mouth_width"]
    earw = d["front_ear_width"]
    w.link(
        "front_socket_margin",
        [
            rect_box(-M / 2 - earw, -M / 2, r, s - r, t),
            rect_box(M / 2, M / 2 + earw, r, s - r, t),
        ],
        "margin",
    )
    w.joint("front_socket_base", "front_wall", "front_socket_margin", xyz=(0, H, 0))
    w.link(
        "front_keeper",
        [rect_box(-K / 2, K / 2, r, d["front_keeper_length"], t)],
        "keeper",
    )
    w.joint("front_socket_return", "front_socket_margin", "front_keeper", xyz=(0, s, 0))
    # Rear spine lifts the lid clear of finite-thickness folded top edges.
    w.link("lid_spine_margin", [rect_box(-F / 2 + r, F / 2 - r, r, s - r, t)], "margin")
    w.joint(
        "lid_spine_fold",
        "rear_wall",
        "lid_spine_margin",
        xyz=(0, H, 0),
        lower=-0.3,
        upper=0.3,
    )
    lidL = d["lid_length"]
    w.link("lid", [rect_box(-F / 2 + r, F / 2 - r, r, lidL - r, t)], "lid")
    w.joint(
        "lid_fold",
        "lid_spine_margin",
        "lid",
        xyz=(0, s, 0),
        lower=-0.4,
        upper=math.pi / 2,
    )
    for side, sign in [("left", 1), ("right", -1)]:  # rear frame reverses global X
        wing = f"lid_{side}_wing"
        a, b = sorted((sign * r, sign * p.lid_wing_depth))
        w.link(wing, [rect_box(a, b, d["wing_start"], d["wing_end"], t)], "dust")
        w.joint(
            f"{wing}_fold", "lid", wing, xyz=(sign * F / 2, 0, 0), axis=(0, -sign, 0)
        )
    Tw = p.front_tab_width
    w.link("lid_front_margin", [rect_box(-Tw / 2, Tw / 2, r, s - r, t)], "margin")
    w.joint(
        "lid_front_margin_fold", "lid", "lid_front_margin", xyz=(0, lidL, 0), upper=2.2
    )
    w.link(
        "lid_front_tab", [rect_box(-Tw / 2, Tw / 2, r, p.front_tab_length, t)], "tab"
    )
    w.joint(
        "lid_front_tab_fold",
        "lid_front_margin",
        "lid_front_tab",
        xyz=(0, s, 0),
        upper=math.pi / 2,
    )
    meta = {
        "schema_version": 1,
        "model_type": "hinged_lid_rollover_mailer",
        "units": "m, kg, rad, s",
        "parameters": asdict(p),
        "requested_parameters": requested,
        "derived": d,
        "base_holes": [list(h) for h in holes],
        "joint_names": list(w.joints),
        "links": w.links,
        "joints": w.joints,
        "dimension_convention": "W and D are outer-wall hinge-midplane spans; H is outer-wall fold-line height. Not exact external or clear internal dimensions.",
        "approximation": "Rectangularized simulation net inspired by the photo; no manufacturer or FEFCO identity claimed. Not a fabrication die line.",
    }
    from motion import Motion

    Motion(meta)  # Reject unreachable paths before writing files.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ET.indent(w.root, space="  ")
    path = out / "mailer.urdf"
    ET.ElementTree(w.root).write(path, encoding="utf-8", xml_declaration=True)
    jsonpath = out / "mailer.json"
    jsonpath.write_text(json.dumps(meta, indent=2) + "\n")
    return path, jsonpath


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT)
    for k in Parameters.__dataclass_fields__:
        ap.add_argument("--" + k.replace("_", "-"), type=float)
    a = ap.parse_args()
    try:
        conf = {} if a.config is None else json.loads(a.config.read_text())
        unknown = set(conf) - set(Parameters.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        conf.update(
            {
                k: getattr(a, k)
                for k in Parameters.__dataclass_fields__
                if getattr(a, k) is not None
            }
        )
        paths = generate(Parameters(**conf), a.out)
    except (ValueError, TypeError, OSError) as ex:
        ap.exit(2, f"Error: {ex}\n")
    print("\n".join(str(p.resolve()) for p in paths))


if __name__ == "__main__":
    main()
