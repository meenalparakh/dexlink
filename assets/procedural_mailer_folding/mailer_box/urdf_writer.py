"""URDF primitives with finite mass and compound inertia (SI units)."""

import math
import xml.etree.ElementTree as ET

import numpy as np


def numbers(values) -> str:
    return " ".join(f"{float(x):.12g}" for x in values)


class Writer:
    def __init__(self, p):
        self.p = p
        self.root = ET.Element("robot", name="procedural_hinged_mailer")
        self.root.append(
            ET.Comment(
                " SI units. Root: base. All zero joints: flat net. "
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
            velocity="2.5",
        )
        ET.SubElement(
            node,
            "dynamics",
            damping=f"{damping:.12g}",
            friction=f"{friction:.12g}",
        )
        self.joints[name] = dict(parent=parent, child=child, lower=lower, upper=upper)
