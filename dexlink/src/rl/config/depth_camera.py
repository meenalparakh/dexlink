"""Opt-in camera configuration for raw rendered-depth observations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass
class DepthCameraCfg:
    resolution: tuple[int, int] = (128, 128)
    position: tuple[float, float, float] = (0.65, -0.6, 0.7)
    lookat: tuple[float, float, float] = (-0.12, 0.0, 0.03)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    vertical_fov: float = 50.0
    near: float = 0.05
    far: float = 3.0

    def validate(self) -> None:
        if len(self.resolution) != 2 or any(
            type(v) is not int or v <= 0 for v in self.resolution
        ):
            raise ValueError(
                f"Expected positive integer camera (width,height), got {self.resolution}."
            )
        for name in ("position", "lookat", "up"):
            value = getattr(self, name)
            if len(value) != 3 or not all(isfinite(v) for v in value):
                raise ValueError(
                    f"Expected finite camera {name} XYZ triple, got {value}."
                )
        forward = [target - start for start, target in zip(self.position, self.lookat)]
        cross = (
            forward[1] * self.up[2] - forward[2] * self.up[1],
            forward[2] * self.up[0] - forward[0] * self.up[2],
            forward[0] * self.up[1] - forward[1] * self.up[0],
        )
        if sum(v * v for v in cross) <= 1e-12:
            raise ValueError(
                "Camera position/lookat/up must define nonparallel viewing axes."
            )
        if not 0 < self.vertical_fov < 180 or not 0 < self.near < self.far:
            raise ValueError("Expected 0 < vertical_fov < 180 and 0 < near < far.")

    def observation_contract(self) -> dict:
        self.validate()
        return {
            "resolution": list(self.resolution),
            "position": list(self.position),
            "lookat": list(self.lookat),
            "up": list(self.up),
            "vertical_fov": self.vertical_fov,
            "near": self.near,
            "far": self.far,
            "depth": "axial_m",
            "pixel_center_offset": 0.5,
            "renderer": "genesis_madrona_rasterizer",
            "robot_geometry": "included",
        }
