#!/usr/bin/env python3
"""Generate procedural countertop kitchen cabinets, counter, and pan trivet URDFs.

Every part is an axis-aligned box, so each visual/collision element is convex
and Genesis does not need a convex decomposition. Run from the repo root:

    conda run -n genesis python assets/procedural_kitchen_cabinet/generate.py
    conda run -n genesis python assets/procedural_kitchen_cabinet/generate.py \
        --preset double_door_drawer --carcass_width 0.5

Geometry is authored in a "cabinet frame" (origin at the center of the carcass
bottom, front opening facing -x, z up). The URDF root frame is then placed at
the closed primary door-handle grip center with the same axes, see README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ElementTree
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

ASSET_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class KitchenCabinetConfig:
    """Dimensions (meters/radians) of one procedural countertop cabinet."""

    name: str = "kitchen_cabinet_single_door"
    # Carcass outer extents: width along y, depth along x, height along z.
    carcass_width: float = 0.44
    carcass_depth: float = 0.40
    carcass_height: float = 0.50
    panel_thickness: float = 0.018
    back_thickness: float = 0.010
    # Shelf top height above the carcass bottom and setback from the front plane.
    shelf_top_height: float = 0.22
    shelf_front_setback: float = 0.147
    # Doors: 1 = one door hinged at +y, 2 = pair hinged at +y and -y.
    door_count: Literal[1, 2] = 1
    door_thickness: float = 0.018
    door_front_gap: float = 0.0015
    door_edge_gap: float = 0.0015
    door_center_gap: float = 0.003
    door_open_limit: float = math.radians(115.0)
    door_damping: float = 0.3
    door_friction: float = 0.05
    # Vertical bar pull: grip axis distance from the door face, distance from the
    # door free edge, and height of the grip center above the carcass bottom.
    handle_standoff: float = 0.060
    handle_edge_offset: float = 0.045
    handle_grip_height: float = 0.36
    handle_bar_size: tuple[float, float, float] = (0.010, 0.018, 0.18)
    handle_post_size_yz: tuple[float, float] = (0.016, 0.012)
    handle_post_inset: float = 0.006
    # Optional bottom drawer bay (prismatic joint along -x).
    include_drawer: bool = False
    drawer_bay_height: float = 0.14
    drawer_travel: float = 0.26
    drawer_wall_thickness: float = 0.010
    drawer_handle_standoff: float = 0.035
    # Wood-like material.
    density: float = 600.0

    def __post_init__(self) -> None:
        positive = {
            "carcass_width": self.carcass_width,
            "carcass_depth": self.carcass_depth,
            "carcass_height": self.carcass_height,
            "panel_thickness": self.panel_thickness,
            "back_thickness": self.back_thickness,
            "door_thickness": self.door_thickness,
            "handle_standoff": self.handle_standoff,
            "density": self.density,
        }
        bad = {key: value for key, value in positive.items() if value <= 0.0}
        if bad:
            raise ValueError(f"Expected positive dimensions, received {bad}.")
        if self.door_count not in (1, 2):
            raise ValueError(f"Expected door_count 1 or 2, received {self.door_count}.")
        interior_bottom = self.panel_thickness + (
            self.drawer_bay_height + self.panel_thickness if self.include_drawer else 0.0
        )
        if not interior_bottom + self.panel_thickness < self.shelf_top_height < (
            self.carcass_height - self.panel_thickness
        ):
            raise ValueError(
                f"Shelf top {self.shelf_top_height} must lie inside the door "
                f"compartment ({interior_bottom}, {self.carcass_height})."
            )
        bar_half_height = 0.5 * self.handle_bar_size[2]
        door_bottom = interior_bottom - self.panel_thickness + self.door_edge_gap
        if not door_bottom < self.handle_grip_height - bar_half_height and (
            self.handle_grip_height + bar_half_height < self.carcass_height
        ):
            raise ValueError(
                f"Handle bar centered at {self.handle_grip_height} does not fit "
                f"on the door spanning z=({door_bottom}, {self.carcass_height})."
            )
        if not 0.0 < self.door_open_limit < math.pi:
            raise ValueError(f"Expected door_open_limit in (0, pi), got {self.door_open_limit}.")


PRESETS: dict[str, KitchenCabinetConfig] = {
    "single_door": KitchenCabinetConfig(),
    "double_door_drawer": KitchenCabinetConfig(
        name="kitchen_cabinet_double_door_drawer",
        carcass_width=0.60,
        carcass_height=0.66,
        shelf_top_height=0.40,
        door_count=2,
        include_drawer=True,
        handle_grip_height=0.52,
    ),
}


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in its link frame."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass
class Link:
    name: str
    boxes: list[Box] = field(default_factory=list)
    material: str = "wood"


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: Literal["fixed", "revolute", "prismatic"]
    parent: str
    child: str
    origin: tuple[float, float, float]
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lower: float = 0.0
    upper: float = 0.0
    damping: float = 0.0
    friction: float = 0.0


def _box_from_bounds(name: str, lower: np.ndarray, upper: np.ndarray) -> Box:
    """Create a box from min/max corners (both in the same frame)."""

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if np.any(upper - lower <= 0.0):
        raise ValueError(f"Box {name!r} has non-positive extents {upper - lower}.")
    return Box(
        name=name,
        center=tuple(float(v) for v in 0.5 * (lower + upper)),
        size=tuple(float(v) for v in upper - lower),
    )


def _shift_box(box: Box, offset: np.ndarray) -> Box:
    return replace(box, center=tuple(float(v) for v in np.asarray(box.center) + offset))


def _composite_inertia(boxes: list[Box], density: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, center of mass, and inertia tensor about the COM of a union of boxes."""

    centers = np.asarray([box.center for box in boxes], dtype=np.float64)
    sizes = np.asarray([box.size for box in boxes], dtype=np.float64)
    masses = density * np.prod(sizes, axis=1)
    total_mass = float(masses.sum())
    com = (masses[:, None] * centers).sum(axis=0) / total_mass
    # Box inertia about its own center plus the parallel-axis term.
    squared = sizes**2
    diag = masses[:, None] / 12.0 * (squared[:, [1, 0, 0]] + squared[:, [2, 2, 1]])
    offsets = centers - com
    inertia = np.zeros((3, 3), dtype=np.float64)
    inertia[np.diag_indices(3)] = diag.sum(axis=0)
    inertia += (
        masses[:, None, None]
        * (
            (offsets**2).sum(axis=1)[:, None, None] * np.eye(3)[None]
            - offsets[:, :, None] * offsets[:, None, :]
        )
    ).sum(axis=0)
    return total_mass, com, inertia


class CabinetBuilder:
    """Build links/joints in the cabinet frame, then re-express them in the URDF root frame."""

    def __init__(self, config: KitchenCabinetConfig) -> None:
        self.cfg = config
        self.links: list[Link] = []
        self.joints: list[Joint] = []
        self.layout: dict[str, object] = {}

    # Carcass: bottom, top, sides, back, shelf, and the optional drawer divider.
    def _carcass_boxes(self) -> list[Box]:
        cfg = self.cfg
        t = cfg.panel_thickness
        half_w, half_d, height = 0.5 * cfg.carcass_width, 0.5 * cfg.carcass_depth, cfg.carcass_height
        front, back = -half_d, half_d
        boxes = [
            _box_from_bounds("bottom", [front, -half_w + t, 0.0], [back - cfg.back_thickness, half_w - t, t]),
            _box_from_bounds("top", [front, -half_w + t, height - t], [back - cfg.back_thickness, half_w - t, height]),
            _box_from_bounds("side_pos_y", [front, half_w - t, 0.0], [back, half_w, height]),
            _box_from_bounds("side_neg_y", [front, -half_w, 0.0], [back, -half_w + t, height]),
            _box_from_bounds("back", [back - cfg.back_thickness, -half_w + t, 0.0], [back, half_w - t, height]),
            _box_from_bounds(
                "shelf",
                [front + cfg.shelf_front_setback, -half_w + t, cfg.shelf_top_height - t],
                [back - cfg.back_thickness, half_w - t, cfg.shelf_top_height],
            ),
        ]
        if cfg.include_drawer:
            divider_top = t + cfg.drawer_bay_height + t
            boxes.append(
                _box_from_bounds(
                    "drawer_divider",
                    [front, -half_w + t, divider_top - t],
                    [back - cfg.back_thickness, half_w - t, divider_top],
                )
            )
        return boxes

    def _door_compartment_bottom(self) -> float:
        cfg = self.cfg
        if cfg.include_drawer:
            return cfg.panel_thickness + cfg.drawer_bay_height
        return 0.0

    def _add_door(self, index: int, hinge_side: Literal[1, -1], door_width: float) -> dict[str, object]:
        """Add one overlay door (hinge on the +y or -y carcass edge) with a vertical bar pull."""

        cfg = self.cfg
        suffix = "" if index == 0 else f"_{index + 1}"
        door_name, handle_name = f"door{suffix}", f"door{suffix}_handle"
        front = -0.5 * cfg.carcass_depth
        door_bottom = self._door_compartment_bottom() + cfg.door_edge_gap
        door_top = cfg.carcass_height - cfg.door_edge_gap
        # Hinge axis on the inner face of the door at its hinge-side edge.
        hinge = np.array(
            [front - cfg.door_front_gap, hinge_side * (0.5 * cfg.carcass_width - cfg.door_edge_gap), 0.0]
        )
        free_edge_y = hinge[1] - hinge_side * door_width
        door_box = _box_from_bounds(
            "panel",
            [-cfg.door_thickness, min(0.0, free_edge_y - hinge[1]), door_bottom],
            [0.0, max(0.0, free_edge_y - hinge[1]), door_top],
        )
        # Vertical bar pull near the free edge; frame at the grip center.
        grip = np.array(
            [
                hinge[0] - cfg.door_thickness - cfg.handle_standoff,
                free_edge_y + hinge_side * cfg.handle_edge_offset,
                cfg.handle_grip_height,
            ]
        )
        bar_x, bar_y, bar_z = cfg.handle_bar_size
        post_y, post_z = cfg.handle_post_size_yz
        post_x0 = 0.5 * bar_x
        post_x1 = cfg.handle_standoff
        post_zc = 0.5 * bar_z - cfg.handle_post_inset - 0.5 * post_z
        handle_boxes = [
            Box("bar", (0.0, 0.0, 0.0), (bar_x, bar_y, bar_z)),
            _box_from_bounds("post_top", [post_x0, -0.5 * post_y, post_zc - 0.5 * post_z], [post_x1, 0.5 * post_y, post_zc + 0.5 * post_z]),
            _box_from_bounds("post_bottom", [post_x0, -0.5 * post_y, -post_zc - 0.5 * post_z], [post_x1, 0.5 * post_y, -post_zc + 0.5 * post_z]),
        ]
        self.links.append(Link(door_name, [door_box], material="door"))
        self.links.append(Link(handle_name, handle_boxes, material="metal"))
        # Positive joint values swing the free edge outward (toward -x).
        self.joints.append(
            Joint(
                name=f"{door_name}_hinge",
                joint_type="revolute",
                parent="carcass",
                child=door_name,
                origin=tuple(hinge),
                axis=(0.0, 0.0, -float(hinge_side)),
                lower=0.0,
                upper=cfg.door_open_limit,
                damping=cfg.door_damping,
                friction=cfg.door_friction,
            )
        )
        self.joints.append(
            Joint(
                name=f"{handle_name}_fixed",
                joint_type="fixed",
                parent=door_name,
                child=handle_name,
                origin=tuple(grip - hinge),
            )
        )
        return {
            "link": door_name,
            "handle_link": handle_name,
            "joint": f"{door_name}_hinge",
            "hinge_side_sign": int(hinge_side),
            "hinge_cabinet_frame": hinge.tolist(),
            "axis": [0.0, 0.0, -float(hinge_side)],
            "width": door_width,
            "z_range": [door_bottom, door_top],
            "grip_center_cabinet_frame": grip.tolist(),
            "open_limit": cfg.door_open_limit,
        }

    def _add_drawer(self) -> dict[str, object]:
        """Add a box drawer with a horizontal bar pull in the bottom bay."""

        cfg = self.cfg
        t, wall = cfg.panel_thickness, cfg.drawer_wall_thickness
        front = -0.5 * cfg.carcass_depth
        half_inner_w = 0.5 * cfg.carcass_width - t
        bay_bottom, bay_top = t, t + cfg.drawer_bay_height
        clearance = 0.003
        body_depth = cfg.carcass_depth - cfg.back_thickness - 0.02
        # Drawer frame at the bay front-bottom-center; front panel overlays the carcass edge.
        origin = np.array([front, 0.0, bay_bottom])
        y0, y1 = -half_inner_w + clearance, half_inner_w - clearance
        z_top = cfg.drawer_bay_height - clearance
        boxes = [
            _box_from_bounds("front", [-cfg.door_thickness - cfg.door_front_gap, -0.5 * cfg.carcass_width + cfg.door_edge_gap, -bay_bottom + cfg.door_edge_gap], [-cfg.door_front_gap, 0.5 * cfg.carcass_width - cfg.door_edge_gap, cfg.drawer_bay_height - cfg.door_edge_gap]),
            _box_from_bounds("floor", [0.0, y0, clearance], [body_depth, y1, clearance + wall]),
            _box_from_bounds("side_pos_y", [0.0, y1 - wall, clearance], [body_depth, y1, z_top]),
            _box_from_bounds("side_neg_y", [0.0, y0, clearance], [body_depth, y0 + wall, z_top]),
            _box_from_bounds("back", [body_depth - wall, y0 + wall, clearance], [body_depth, y1 - wall, z_top]),
        ]
        grip = np.array([-cfg.door_thickness - cfg.door_front_gap - cfg.drawer_handle_standoff, 0.0, 0.5 * cfg.drawer_bay_height])
        bar_len = 0.16
        handle_boxes = [
            Box("bar", (0.0, 0.0, 0.0), (0.014, bar_len, 0.014)),
            _box_from_bounds("post_pos_y", [0.007, 0.5 * bar_len - 0.018, -0.006], [cfg.drawer_handle_standoff, 0.5 * bar_len - 0.006, 0.006]),
            _box_from_bounds("post_neg_y", [0.007, -0.5 * bar_len + 0.006, -0.006], [cfg.drawer_handle_standoff, -0.5 * bar_len + 0.018, 0.006]),
        ]
        self.links.append(Link("drawer", boxes, material="door"))
        self.links.append(Link("drawer_handle", handle_boxes, material="metal"))
        self.joints.append(
            Joint("drawer_slide", "prismatic", "carcass", "drawer", tuple(origin), (-1.0, 0.0, 0.0), 0.0, cfg.drawer_travel, damping=2.0, friction=0.5)
        )
        self.joints.append(Joint("drawer_handle_fixed", "fixed", "drawer", "drawer_handle", tuple(grip)))
        return {"link": "drawer", "handle_link": "drawer_handle", "joint": "drawer_slide", "travel": cfg.drawer_travel}

    def build(self) -> None:
        cfg = self.cfg
        self.links.append(Link("carcass", self._carcass_boxes(), material="wood"))
        if cfg.door_count == 1:
            doors = [self._add_door(0, 1, cfg.carcass_width - 2.0 * cfg.door_edge_gap)]
        else:
            width = 0.5 * (cfg.carcass_width - 2.0 * cfg.door_edge_gap - cfg.door_center_gap)
            doors = [self._add_door(0, 1, width), self._add_door(1, -1, width)]
        drawer = self._add_drawer() if cfg.include_drawer else None

        # Re-express everything with the URDF root at the primary grip center.
        root_offset = np.asarray(doors[0]["grip_center_cabinet_frame"], dtype=np.float64)
        carcass = self.links[0]
        carcass.boxes = [_shift_box(box, -root_offset) for box in carcass.boxes]
        self.joints = [
            replace(joint, origin=tuple(float(v) for v in np.asarray(joint.origin) - root_offset))
            if joint.parent == "carcass"
            else joint
            for joint in self.joints
        ]
        t = cfg.panel_thickness
        self.layout = {
            "config": asdict(cfg),
            "frame_note": (
                "Cabinet frame: origin at the carcass bottom center, front opening "
                "toward -x, z up. URDF root frame = cabinet frame translated by "
                "root_origin_in_cabinet_frame (the closed primary door grip center)."
            ),
            "root_origin_in_cabinet_frame": root_offset.tolist(),
            "carcass_front_x": -0.5 * cfg.carcass_depth,
            "interior": {
                "x": [-0.5 * cfg.carcass_depth, 0.5 * cfg.carcass_depth - cfg.back_thickness],
                "y": [-0.5 * cfg.carcass_width + t, 0.5 * cfg.carcass_width - t],
                "z": [self._door_compartment_bottom() + t, cfg.carcass_height - t],
            },
            "shelf": {
                "top_z": cfg.shelf_top_height,
                "front_x": -0.5 * cfg.carcass_depth + cfg.shelf_front_setback,
                "back_x": 0.5 * cfg.carcass_depth - cfg.back_thickness,
            },
            "doors": doors,
            "drawer": drawer,
            "joint_names": [joint.name for joint in self.joints if joint.joint_type != "fixed"],
        }

    # URDF serialization with explicit inertials and box visual/collision pairs.
    def to_urdf(self) -> ElementTree.ElementTree:
        cfg = self.cfg
        robot = ElementTree.Element("robot", name=cfg.name)
        robot.append(ElementTree.Comment(" Generated by assets/procedural_kitchen_cabinet/generate.py. SI units. Root frame: closed primary door grip center; front faces -x. "))
        colors = {"wood": "0.80 0.70 0.55 1", "door": "0.62 0.75 0.80 1", "metal": "0.35 0.35 0.38 1"}
        for material, rgba in colors.items():
            ElementTree.SubElement(ElementTree.SubElement(robot, "material", name=material), "color", rgba=rgba)
        density = {"wood": cfg.density, "door": cfg.density, "metal": 2700.0}
        for link in self.links:
            node = ElementTree.SubElement(robot, "link", name=link.name)
            mass, com, inertia = _composite_inertia(link.boxes, density[link.material])
            inertial = ElementTree.SubElement(node, "inertial")
            ElementTree.SubElement(inertial, "origin", xyz=_fmt(com), rpy="0 0 0")
            ElementTree.SubElement(inertial, "mass", value=f"{mass:.6g}")
            ElementTree.SubElement(
                inertial,
                "inertia",
                ixx=f"{inertia[0, 0]:.6g}", iyy=f"{inertia[1, 1]:.6g}", izz=f"{inertia[2, 2]:.6g}",
                ixy=f"{inertia[0, 1]:.6g}", ixz=f"{inertia[0, 2]:.6g}", iyz=f"{inertia[1, 2]:.6g}",
            )
            for box in link.boxes:
                for kind in ("visual", "collision"):
                    element = ElementTree.SubElement(node, kind, name=f"{link.name}_{box.name}_{kind}")
                    ElementTree.SubElement(element, "origin", xyz=_fmt(box.center), rpy="0 0 0")
                    ElementTree.SubElement(ElementTree.SubElement(element, "geometry"), "box", size=_fmt(box.size))
                    if kind == "visual":
                        ElementTree.SubElement(element, "material", name=link.material)
        for joint in self.joints:
            node = ElementTree.SubElement(robot, "joint", name=joint.name, type=joint.joint_type)
            ElementTree.SubElement(node, "parent", link=joint.parent)
            ElementTree.SubElement(node, "child", link=joint.child)
            ElementTree.SubElement(node, "origin", xyz=_fmt(joint.origin), rpy="0 0 0")
            if joint.joint_type != "fixed":
                ElementTree.SubElement(node, "axis", xyz=_fmt(joint.axis))
                ElementTree.SubElement(node, "limit", lower=f"{joint.lower:.6g}", upper=f"{joint.upper:.6g}", effort="50", velocity="2")
                ElementTree.SubElement(node, "dynamics", damping=f"{joint.damping:.6g}", friction=f"{joint.friction:.6g}")
        ElementTree.indent(robot, space="  ")
        return ElementTree.ElementTree(robot)

    def preview_mesh(self, joint_values: dict[str, float]) -> trimesh.Trimesh:
        """Combined mesh in the URDF root frame at the given joint values."""

        link_pose = {"carcass": np.eye(4)}
        for joint in self.joints:  # joints are appended parent-before-child
            local = np.eye(4)
            local[:3, 3] = joint.origin
            motion = np.eye(4)
            value = float(joint_values.get(joint.name, 0.0))
            if joint.joint_type == "revolute":
                motion[:3, :3] = trimesh.transformations.rotation_matrix(value, joint.axis)[:3, :3]
            elif joint.joint_type == "prismatic":
                motion[:3, 3] = value * np.asarray(joint.axis)
            link_pose[joint.child] = link_pose[joint.parent] @ local @ motion
        pieces = []
        for link in self.links:
            for box in link.boxes:
                mesh = trimesh.creation.box(extents=box.size)
                mesh.apply_translation(box.center)
                mesh.apply_transform(link_pose[link.name])
                pieces.append(mesh)
        return trimesh.util.concatenate(pieces)


def _fmt(values: tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _write_box_urdf(path: Path, name: str, size: tuple[float, float, float], top_frame: bool, color: str, mass: float) -> None:
    """Static support box with its frame on the top (counter) or bottom (trivet) face."""

    center_z = -0.5 * size[2] if top_frame else 0.5 * size[2]
    robot = ElementTree.Element("robot", name=name)
    ElementTree.SubElement(ElementTree.SubElement(robot, "material", name=f"{name}_color"), "color", rgba=color)
    link = ElementTree.SubElement(robot, "link", name=name)
    sx, sy, sz = size
    inertial = ElementTree.SubElement(link, "inertial")
    ElementTree.SubElement(inertial, "origin", xyz=f"0 0 {center_z:.6g}", rpy="0 0 0")
    ElementTree.SubElement(inertial, "mass", value=f"{mass:.6g}")
    ElementTree.SubElement(
        inertial, "inertia",
        ixx=f"{mass / 12 * (sy**2 + sz**2):.6g}", iyy=f"{mass / 12 * (sx**2 + sz**2):.6g}", izz=f"{mass / 12 * (sx**2 + sy**2):.6g}",
        ixy="0", ixz="0", iyz="0",
    )
    for kind in ("visual", "collision"):
        element = ElementTree.SubElement(link, kind)
        ElementTree.SubElement(element, "origin", xyz=f"0 0 {center_z:.6g}", rpy="0 0 0")
        ElementTree.SubElement(ElementTree.SubElement(element, "geometry"), "box", size=_fmt(size))
        if kind == "visual":
            ElementTree.SubElement(element, "material", name=f"{name}_color")
    ElementTree.indent(robot, space="  ")
    ElementTree.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


@dataclass(frozen=True)
class SupportConfig:
    counter_size: tuple[float, float, float] = (1.4, 1.6, 0.06)
    trivet_size: tuple[float, float, float] = (0.17, 0.16, 0.08)


def generate(config: KitchenCabinetConfig, output_dir: Path) -> dict[str, object]:
    """Write <name>/<name>.urdf, preview OBJs, and layout.json; return the layout."""

    builder = CabinetBuilder(config)
    builder.build()
    cabinet_dir = output_dir / config.name
    cabinet_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = cabinet_dir / f"{config.name}.urdf"
    builder.to_urdf().write(urdf_path, encoding="utf-8", xml_declaration=True)
    # Preview meshes at the closed and fully open configurations.
    closed = {name: 0.0 for name in builder.layout["joint_names"]}
    opened = {
        joint.name: joint.upper for joint in builder.joints if joint.joint_type != "fixed"
    }
    builder.preview_mesh(closed).export(cabinet_dir / "preview_closed.obj")
    builder.preview_mesh(opened).export(cabinet_dir / "preview_open.obj")
    layout = dict(builder.layout)
    layout["urdf_path"] = str(urdf_path.relative_to(ASSET_DIR.parents[1]))
    (cabinet_dir / "layout.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    return layout


def _parse_config() -> tuple[list[KitchenCabinetConfig], Path]:
    """Preset selection plus `--<field> value` overrides for every config field."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), action="append", default=None)
    parser.add_argument("--output_dir", type=Path, default=ASSET_DIR)
    for item in fields(KitchenCabinetConfig):
        default = getattr(KitchenCabinetConfig(), item.name)
        if isinstance(default, bool):
            parser.add_argument(f"--{item.name}", type=lambda v: v.lower() in ("1", "true", "yes"), default=None)
        elif isinstance(default, tuple):
            parser.add_argument(f"--{item.name}", type=float, nargs=len(default), default=None)
        else:
            parser.add_argument(f"--{item.name}", type=type(default), default=None)
    args = parser.parse_args()
    presets = args.preset or sorted(PRESETS)
    overrides = {
        item.name: (tuple(getattr(args, item.name)) if isinstance(getattr(args, item.name), list) else getattr(args, item.name))
        for item in fields(KitchenCabinetConfig)
        if getattr(args, item.name) is not None
    }
    if overrides and len(presets) != 1:
        raise ValueError(f"Field overrides need exactly one --preset, received {presets}.")
    return [replace(PRESETS[name], **overrides) for name in presets], args.output_dir


def main() -> None:
    configs, output_dir = _parse_config()
    for config in configs:
        layout = generate(config, output_dir)
        print(f"Wrote {layout['urdf_path']} with joints {layout['joint_names']}")
    supports = SupportConfig()
    support_dir = output_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    _write_box_urdf(support_dir / "counter.urdf", "kitchen_counter", supports.counter_size, True, "0.55 0.57 0.60 1", 80.0)
    _write_box_urdf(support_dir / "pan_trivet.urdf", "pan_trivet", supports.trivet_size, False, "0.25 0.25 0.27 1", 2.0)
    print(f"Wrote {support_dir}/counter.urdf and pan_trivet.urdf")


if __name__ == "__main__":
    main()
