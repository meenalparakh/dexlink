"""Box-only URDF inspection and selectable FK backend.

The default backend is the REAL installed urdfpy package, never a replacement
module with that name. ``reference`` is an explicitly selected independent FK
implementation for offline inspection and numerical cross-checking.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def vector(value: str | None, default=(0, 0, 0)) -> np.ndarray:
    return np.array(default, float) if value is None else np.fromstring(value, sep=" ")


def rotate(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n <= 0:
        raise ValueError("Zero rotation axis")
    x, y, z = axis / n
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def origin(node: ET.Element | None) -> np.ndarray:
    if node is None:
        return np.eye(4)
    roll, pitch, yaw = vector(node.get("rpy"))
    T = rotate((0, 0, 1), yaw) @ rotate((0, 1, 0), pitch) @ rotate((1, 0, 0), roll)
    T[:3, 3] = vector(node.get("xyz"))
    return T


@dataclass
class Box:
    link: str
    name: str
    pose: np.ndarray
    size: np.ndarray
    color: np.ndarray

    def vertices(self) -> np.ndarray:
        return (
            np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ],
                float,
            )
            * self.size
            / 2
        )


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    pose: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float
    fixed: bool = False


class _NumpyCompatibility:
    """Local proxy for old urdfpy's np.float; does not modify global NumPy."""

    def __getattr__(self, name):
        legacy = {"float": float, "int": int, "bool": bool, "complex": complex}
        return legacy[name] if name in legacy else getattr(np, name)


def load_urdfpy(path: Path):
    try:
        import urdfpy.urdf as upstream
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "urdfpy could not be imported. Use a separate Python 3.10-3.11 "
            "environment and the README install commands. For the explicitly "
            "independent offline backend, pass --backend reference. "
            f"Original error: {exc}"
        ) from exc
    upstream.np = _NumpyCompatibility()
    return upstream.URDF.load(str(path))


class Model:
    def __init__(self, path: Path, backend: str = "urdfpy"):
        self.path = Path(path)
        self.backend = backend
        root = ET.parse(path).getroot()
        materials = {
            m.get("name"): vector(m.find("color").get("rgba"))
            for m in root.findall("material")
            if m.find("color") is not None
        }
        self.links: list[str] = []
        self.visuals: list[Box] = []
        self.collisions: list[Box] = []
        self.inertias: dict[str, tuple[float, np.ndarray]] = {}
        for link in root.findall("link"):
            name = link.get("name")
            if name in self.links:
                raise ValueError(f"Duplicate link {name}")
            self.links.append(name)
            inertial = link.find("inertial")
            if inertial is not None:
                mass = float(inertial.find("mass").get("value"))
                it = inertial.find("inertia")
                I = np.array(
                    [
                        [
                            float(it.get("ixx")),
                            float(it.get("ixy")),
                            float(it.get("ixz")),
                        ],
                        [
                            float(it.get("ixy")),
                            float(it.get("iyy")),
                            float(it.get("iyz")),
                        ],
                        [
                            float(it.get("ixz")),
                            float(it.get("iyz")),
                            float(it.get("izz")),
                        ],
                    ]
                )
                self.inertias[name] = mass, I
            for kind, collection in (
                ("visual", self.visuals),
                ("collision", self.collisions),
            ):
                for idx, shape in enumerate(link.findall(kind)):
                    box = shape.find("geometry/box")
                    if box is None:
                        raise ValueError(
                            "The reference checker supports box-only URDFs"
                        )
                    color = np.array([0.7, 0.6, 0.45, 1.0])
                    material = shape.find("material")
                    if material is not None:
                        color = materials.get(material.get("name"), color)
                        if material.find("color") is not None:
                            color = vector(material.find("color").get("rgba"))
                    collection.append(
                        Box(
                            name,
                            shape.get("name", f"{name}_{kind}_{idx}"),
                            origin(shape.find("origin")),
                            vector(box.get("size")),
                            color,
                        )
                    )
        self.joints: list[Joint] = []
        seen = set()
        for j in root.findall("joint"):
            if j.get("name") in seen:
                raise ValueError("Duplicate joint name")
            seen.add(j.get("name"))
            kind = j.get("type")
            if kind not in ("fixed", "revolute"):
                raise ValueError(f"Unsupported joint type for this checker: {kind}")
            lim, ax = j.find("limit"), j.find("axis")
            self.joints.append(
                Joint(
                    j.get("name"),
                    j.find("parent").get("link"),
                    j.find("child").get("link"),
                    origin(j.find("origin")),
                    vector(ax.get("xyz") if ax is not None else None, (1, 0, 0)),
                    float(lim.get("lower")) if lim is not None else 0,
                    float(lim.get("upper")) if lim is not None else 0,
                    float(lim.get("velocity")) if lim is not None else 0,
                    kind == "fixed",
                )
            )
        child_names = [j.child for j in self.joints]
        if len(child_names) != len(set(child_names)):
            raise ValueError("URDF is not a tree: a child has multiple parents")
        roots = set(self.links) - set(child_names)
        if len(roots) != 1:
            raise ValueError(f"Expected one root, got {roots}")
        self.root = roots.pop()
        available = {self.root}
        pending, ordered = self.joints.copy(), []
        while pending:
            ready = [
                j for j in pending if j.parent in available and j.child in self.links
            ]
            if not ready:
                raise ValueError("Disconnected or cyclic joint graph")
            for j in ready:
                ordered.append(j)
                pending.remove(j)
                available.add(j.child)
        if available != set(self.links):
            raise ValueError("Disconnected link graph")
        self.ordered = ordered
        self.joint_names = [j.name for j in self.joints if not j.fixed]
        self.robot = load_urdfpy(path) if backend == "urdfpy" else None
        if backend not in ("urdfpy", "reference"):
            raise ValueError("Unknown FK backend")

    def reference_fk(self, cfg: dict[str, float]) -> dict[str, np.ndarray]:
        poses = {self.root: np.eye(4)}
        for j in self.ordered:
            q = 0 if j.fixed else cfg.get(j.name, 0)
            poses[j.child] = poses[j.parent] @ j.pose @ rotate(j.axis, q)
        return poses

    def fk(self, cfg: dict[str, float]) -> dict[str, np.ndarray]:
        if self.robot is not None:
            return {
                link.name: pose for link, pose in self.robot.link_fk(cfg=cfg).items()
            }
        return self.reference_fk(cfg)

    def crosscheck_urdfpy(self, cfg: dict[str, float]) -> float | None:
        if self.robot is None:
            return None
        a, b = self.fk(cfg), self.reference_fk(cfg)
        return max(float(np.max(np.abs(a[k] - b[k]))) for k in a)

    def box_poses(self, cfg: dict[str, float], collision: bool = False):
        fk = self.fk(cfg)
        boxes = self.collisions if collision else self.visuals
        return [(b, fk[b.link] @ b.pose) for b in boxes]
