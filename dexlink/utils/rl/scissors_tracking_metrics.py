"""Continuous hinge/pose tracking and geometric airborne retention for scissors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from urdfpy import URDF

from dexlink.utils.retargeting.error_functions import angle_error
from dexlink.utils.retargeting.math_utils import matrix_from_quat
from dexlink.utils.retargeting.path_utils import process_urdf

SCISSORS_METRIC_NAMES = (
    "scissors_articulation_error_rad", "scissors_position_error_m", "scissors_orientation_error_rad",
    "scissors_lowest_visual_height_above_table_m", "scissors_in_air",
    "scissors_mean_articulation_error_rad", "scissors_mean_position_error_m",
    "scissors_mean_orientation_error_rad", "scissors_airborne_fraction",
)


@dataclass(frozen=True)
class ScissorsSample:
    articulation_error_rad: torch.Tensor
    position_error_m: torch.Tensor
    orientation_error_rad: torch.Tensor
    lowest_height_m: torch.Tensor


class ScissorsTracking:
    def __init__(self, urdf: Path, num_envs: int, device: str, clearance_m: float) -> None:
        if num_envs <= 0 or clearance_m <= 0:
            raise ValueError("Scissors tracking requires positive environment count and airborne clearance")
        self.num_envs, self.device, self.clearance_m = num_envs, device, clearance_m
        model = URDF.load(str(process_urdf(urdf)))
        self.link_names = []
        points, indices = [], []
        # Linear height extrema are unchanged by taking each visual's convex
        # hull; open handles and every physical visual link remain represented.
        for link in model.links:
            pieces = []
            for visual in link.visuals:
                for mesh in visual.geometry.meshes:
                    transformed = mesh.copy()
                    transformed.apply_transform(visual.origin)
                    pieces.append(transformed)
            if pieces:
                vertices = trimesh.convex.convex_hull(trimesh.util.concatenate(pieces)).vertices
                indices.extend([len(self.link_names)]*len(vertices))
                points.append(vertices)
                self.link_names.append(link.name)
        if len(self.link_names) != 2:
            raise ValueError(f"Expected the two scissors visual links, received {self.link_names}")
        self.points = torch.as_tensor(np.concatenate(points), dtype=torch.float32, device=device)
        self.point_link_indices = torch.tensor(indices, dtype=torch.long, device=device)
        self.error_sum = torch.zeros((num_envs, 3), device=device)
        self.airborne_count = torch.zeros(num_envs, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def sample(self, entity: object, current: dict, reference: dict, table_height: float) -> ScissorsSample:
        if current["arti"].shape != (self.num_envs, 1) or reference["arti"].shape != (self.num_envs, 1):
            raise ValueError("Expected current and reference scissors articulation shape (N, 1)")
        links = [entity.get_link(name).idx_local for name in self.link_names]
        positions = entity.get_links_pos(links_idx_local=links)
        quaternions = entity.get_links_quat(links_idx_local=links)
        if positions.shape != (self.num_envs, 2, 3) or quaternions.shape != (self.num_envs, 2, 4):
            raise ValueError(f"Expected batched two-link poses, received {positions.shape}/{quaternions.shape}")
        rotation = matrix_from_quat(quaternions)
        world_z = (rotation[:, self.point_link_indices, 2, :]*self.points[None]).sum(-1)
        world_z += positions[:, self.point_link_indices, 2]
        return ScissorsSample(
            (current["arti"]-reference["arti"]).abs()[:, 0],
            torch.linalg.vector_norm(current["pos"]-reference["pos"], dim=-1),
            angle_error(current["quat"], reference["quat"], current_format="quat", target_format="quat"),
            world_z.amin(dim=1)-table_height,
        )

    def _errors(self, sample: ScissorsSample) -> torch.Tensor:
        values = (sample.articulation_error_rad, sample.position_error_m, sample.orientation_error_rad, sample.lowest_height_m)
        if any(value.shape != (self.num_envs,) for value in values):
            raise ValueError(f"Expected one scissors sample per environment, received {[value.shape for value in values]}")
        return torch.stack(values[:3], dim=1)

    def update(self, sample: ScissorsSample) -> None:
        self.error_sum += self._errors(sample)
        self.airborne_count += (sample.lowest_height_m > self.clearance_m).float()
        self.step_count += 1

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise ValueError("Expected rank-one long environment IDs for scissors metric reset")
        self.error_sum[env_ids] = 0
        self.airborne_count[env_ids] = 0
        self.step_count[env_ids] = 0

    def metrics(self, sample: ScissorsSample) -> dict[str, torch.Tensor]:
        self._errors(sample)
        count = self.step_count.clamp_min(1)
        mean = self.error_sum/count[:, None]
        values = (sample.articulation_error_rad, sample.position_error_m, sample.orientation_error_rad,
                  sample.lowest_height_m, (sample.lowest_height_m > self.clearance_m).float(),
                  mean[:, 0], mean[:, 1], mean[:, 2], self.airborne_count/count)
        return dict(zip(SCISSORS_METRIC_NAMES, values, strict=True))
