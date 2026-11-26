"""Reusable scene and rollout rendering utilities."""

from __future__ import annotations


from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CameraSpec:
    direction: tuple[float, float, float]
    margin: float
