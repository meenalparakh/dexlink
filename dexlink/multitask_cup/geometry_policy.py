"""Geometry/program-conditioned architecture prototype; separate from the v1 trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.distributions import Normal

from .config import GeometryPolicyConfig
from .policy import AdaptiveRMSNorm, OFFSETS

STATE_SCHEMA = (
    "controlled_finger_positions:12",
    "controlled_finger_velocities:12",
    "cup_pose_in_current_wrist_xyz_rot6d:9",
    "cup_twist_relative_to_wrist:6",
    "wrist_world_twist_expressed_in_wrist:6",
    "gravity_direction_in_wrist:3",
    "wrist_height_above_horizontal_support:1",
    "active_phase_goal_in_wrist_xyz_rot6d:9",
    "final_goal_in_wrist_xyz_rot6d:9",
)
REFERENCE_SCHEMA = (
    "wrist_target_in_current_wrist_xyz_rot6d:9",
    "finger_targets:12",
    "cup_target_in_current_wrist_xyz_rot6d:9",
)
PHASE_KINDS = (
    "reach",
    "grasp",
    "lift",
    "hold",
    "rotate",
    "translate",
    "lower",
    "release",
    "retreat",
)
PROGRAM_SCHEMA = (
    "phase_kind_onehot:9",
    "duration_seconds:1",
    "cup_endpoint_in_initial_cup_xyz_rot6d:9",
    "finger_endpoint_targets:12",
    "signed_unwrapped_world_y_angle_rad:1",
)
GEOMETRY_SCHEMA = (
    "object_frame_surface_xyz_normals:Nx6",
    "extent_xyz_m:3",
    "bbox_center_xyz_m:3",
    "mass_kg:1",
    "com_xyz_m:3",
    "body_frame_inertia_xx_yy_zz_xy_xz_yz_kgm2:6",
)
PROGRESS_SCHEMA = (
    "phase_fraction",
    "phase_seconds_remaining",
    "program_seconds_remaining",
    "terminal_hold",
)


@dataclass(frozen=True)
class GeometryPolicySchema:
    state_dim: int = 67
    reference_dim: int = 30
    residual_dim: int = 18
    program_feature_dim: int = 32
    version: str = "c11.geometry_program.local10.v2"


@dataclass(frozen=True)
class GeometryPolicyInputs:
    state: torch.Tensor
    window: torch.Tensor
    window_mask: torch.Tensor
    geometry_points: torch.Tensor
    geometry_metadata: torch.Tensor
    geometry_index: torch.Tensor
    program_tokens: torch.Tensor
    program_mask: torch.Tensor
    program_index: torch.Tensor
    current_phase: torch.Tensor
    progress: torch.Tensor


class GeometryEncoder(nn.Module):
    """Object-frame XYZ/normals with symmetric pooling and explicit metric size/origin."""

    def __init__(self, config: GeometryPolicyConfig) -> None:
        super().__init__()
        if (
            config.point_count < 1
            or not config.geometry_hidden_dims
            or min(config.geometry_hidden_dims) < 1
        ):
            raise ValueError(
                "Expected a positive point count and nonempty positive geometry widths."
            )
        self.point_count = config.point_count
        self.register_buffer("metadata_scales", torch.tensor([0.1] * 10 + [0.001] * 6))
        blocks: list[nn.Module] = []
        for before, after in zip(
            (6, *config.geometry_hidden_dims[:-1]), config.geometry_hidden_dims
        ):
            blocks.extend((nn.Linear(before, after), nn.SiLU()))
        self.point_mlp = nn.Sequential(*blocks)
        self.projection = nn.Sequential(
            nn.Linear(
                2 * config.geometry_hidden_dims[-1] + 16, config.geometry_embed_dim
            ),
            nn.SiLU(),
        )

    def forward(self, points: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or tuple(points.shape[1:]) != (self.point_count, 6):
            raise ValueError(
                f"Expected geometry points (G,{self.point_count},6), received {tuple(points.shape)}."
            )
        if points.shape[0] < 1 or tuple(metadata.shape) != (points.shape[0], 16):
            raise ValueError(
                f"Expected geometry metadata (G,16) for a nonempty table, received {tuple(metadata.shape)}."
            )
        if (
            not torch.isfinite(points).all()
            or not torch.isfinite(metadata).all()
            or (metadata[:, :3] <= 0).any()
            or (metadata[:, 6] <= 0).any()
        ):
            raise ValueError(
                "Expected finite geometry, positive physical XYZ extents and positive mass."
            )
        xx, yy, zz, xy, xz, yz = metadata[:, 10:].unbind(-1)
        inertia = torch.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz), -1).reshape(
            -1, 3, 3
        )
        if (torch.linalg.eigvalsh(inertia) <= 0).any():
            raise ValueError("Expected a positive-definite body-frame inertia tensor.")
        if not torch.allclose(
            points[..., 3:].norm(dim=-1), torch.ones_like(points[..., 0]), atol=1e-3
        ):
            raise ValueError(
                "Expected unit object-frame surface normals in the last three channels."
            )
        # A single isotropic scale preserves shape ratios; metadata retains metres and root offsets.
        radius = metadata[:, :3].amax(-1) / 2
        xyz = (points[..., :3] - metadata[:, None, 3:6]) / radius[:, None, None]
        features = self.point_mlp(torch.cat((xyz, points[..., 3:]), dim=-1))
        return self.projection(
            torch.cat(
                (features.amax(1), features.mean(1), metadata / self.metadata_scales),
                dim=-1,
            )
        )


class ProgramEncoder(nn.Module):
    """Ordered phase tokens with explicit padding masks and no stochastic layers."""

    def __init__(
        self, schema: GeometryPolicySchema, config: GeometryPolicyConfig
    ) -> None:
        super().__init__()
        width = config.program_embed_dim
        if (
            width < 2
            or width % 2
            or config.program_heads < 1
            or width % config.program_heads
        ):
            raise ValueError(
                "Expected an even program width divisible by a positive head count."
            )
        if config.program_layers < 1 or config.max_program_phases < 1:
            raise ValueError("Expected positive program layer and phase limits.")
        self.width = width
        self.feature_dim = schema.program_feature_dim
        self.max_phases = config.max_program_phases
        self.projection = nn.Linear(self.feature_dim, width)
        layer = nn.TransformerEncoderLayer(
            width,
            config.program_heads,
            2 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, config.program_layers, enable_nested_tensor=False
        )

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            tokens.ndim != 3
            or tokens.shape[0] < 1
            or tokens.shape[2] != self.feature_dim
        ):
            raise ValueError(
                f"Expected program tokens (K,P,{self.feature_dim}), received {tuple(tokens.shape)}."
            )
        phases = tokens.shape[1]
        if (
            not 1 <= phases <= self.max_phases
            or mask.shape != tokens.shape[:2]
            or mask.dtype != torch.bool
        ):
            raise ValueError(
                f"Expected 1..{self.max_phases} phases with a boolean (K,P) validity mask."
            )
        if not mask.any(-1).all() or not torch.isfinite(tokens).all():
            raise ValueError(
                "Expected finite tokens and at least one valid phase per program."
            )
        # Explicit phase positions preserve order even when endpoint rotations coincide.
        positions = torch.arange(phases, dtype=tokens.dtype, device=tokens.device)
        frequencies = torch.exp(
            torch.arange(0, self.width, 2, dtype=tokens.dtype, device=tokens.device)
            * (-math.log(10000) / self.width)
        )
        angles = positions[:, None] * frequencies[None, :]
        encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
        embedded = self.projection(tokens.masked_fill(~mask[..., None], 0)) + encoding
        features = self.transformer(embedded, src_key_padding_mask=~mask)
        pooled = features.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(
            1, keepdim=True
        )
        return features, pooled


class GeometryCupPolicy(nn.Module):
    def __init__(
        self, schema: GeometryPolicySchema, config: GeometryPolicyConfig
    ) -> None:
        super().__init__()
        if (
            min(
                schema.state_dim,
                schema.reference_dim,
                schema.residual_dim,
                config.hidden_dim,
                config.geometry_embed_dim,
            )
            < 1
            or config.window_sigma <= 0
        ):
            raise ValueError(
                "Expected positive schema dimensions, widths and window sigma."
            )
        self.schema = schema
        self.config = config
        if (
            schema.state_dim,
            schema.reference_dim,
            schema.residual_dim,
            schema.program_feature_dim,
        ) != (67, 30, 18, 32):
            raise ValueError(
                "This input contract requires state/reference/residual/program dimensions 67/30/18/32."
            )
        width = config.hidden_dim
        self.geometry_encoder = GeometryEncoder(config)
        self.program_encoder = ProgramEncoder(schema, config)
        self.task_context = nn.Sequential(
            nn.Linear(2 * config.program_embed_dim + 4, config.program_embed_dim),
            nn.SiLU(),
            nn.Linear(config.program_embed_dim, config.program_embed_dim),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(config.geometry_embed_dim + config.program_embed_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(schema.state_dim, width), nn.SiLU()
        )
        self.reference_encoder = nn.Sequential(
            nn.Linear(10 * (schema.reference_dim + 1), width), nn.SiLU()
        )
        self.state_norm = AdaptiveRMSNorm(width, width)
        self.reference_norm = AdaptiveRMSNorm(width, width)
        fusion_dim = 2 * width + config.geometry_embed_dim + config.program_embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
        )
        self.reference_head = nn.Linear(width, 10 * schema.residual_dim)
        self.value_head = nn.Linear(width, 1)
        self.log_std = nn.Parameter(
            torch.full((schema.residual_dim,), config.initial_log_std)
        )
        nn.init.normal_(self.reference_head.weight, std=0.001)
        nn.init.zeros_(self.reference_head.bias)

    def checkpoint_schema(self) -> dict:
        return {
            "policy": asdict(self.schema),
            "model": asdict(self.config),
            "offsets": list(OFFSETS),
            "state_fields": STATE_SCHEMA,
            "reference_fields": REFERENCE_SCHEMA,
            "phase_kinds": PHASE_KINDS,
            "program_fields": PROGRAM_SCHEMA,
            "geometry_fields": GEOMETRY_SCHEMA,
            "progress_fields": PROGRESS_SCHEMA,
        }

    def forward(
        self, inputs: GeometryPolicyInputs
    ) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        batch = inputs.state.shape[0]
        if (
            batch < 1
            or inputs.geometry_points.ndim != 3
            or inputs.program_tokens.ndim != 3
        ):
            raise ValueError(
                "Expected a nonempty state batch, geometry table (G,N,6) and program table (K,P,32)."
            )
        expected = {
            "state": (batch, self.schema.state_dim),
            "window": (batch, 10, self.schema.reference_dim),
            "window_mask": (batch, 10),
            "geometry_index": (batch,),
            "program_index": (batch,),
            "current_phase": (batch,),
            "progress": (batch, 4),
        }
        for name, shape in expected.items():
            value = getattr(inputs, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"Expected {name} shape {shape}, received {tuple(value.shape)}."
                )
        if inputs.window_mask.dtype != torch.bool or not inputs.window_mask[:, 4].all():
            raise ValueError(
                "Expected a boolean window mask with the current slot valid."
            )
        if (
            inputs.window_mask.device != inputs.state.device
            or inputs.program_mask.device != inputs.state.device
        ):
            raise ValueError("Expected masks on the same device as the state.")
        for name in (
            "state",
            "window",
            "geometry_points",
            "geometry_metadata",
            "program_tokens",
            "progress",
        ):
            value = getattr(inputs, name)
            if (
                value.dtype != self.log_std.dtype
                or value.device != self.log_std.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"Expected finite {name} with policy dtype/device {self.log_std.dtype}/{self.log_std.device}."
                )
        for name, limit in (
            ("geometry_index", inputs.geometry_points.shape[0]),
            ("program_index", inputs.program_tokens.shape[0]),
            ("current_phase", inputs.program_tokens.shape[1]),
        ):
            value = getattr(inputs, name)
            if (
                value.dtype != torch.long
                or value.device != inputs.state.device
                or (value < 0).any()
                or (value >= limit).any()
            ):
                raise ValueError(
                    f"Expected {name} as int64 indices in [0,{limit - 1}]."
                )
        invalid_fraction = (
            (inputs.progress[:, 0] < 0) | (inputs.progress[:, 0] > 1)
        ).any()
        invalid_hold = not (
            (inputs.progress[:, 3] == 0) | (inputs.progress[:, 3] == 1)
        ).all()
        if invalid_fraction or (inputs.progress[:, 1:3] < 0).any() or invalid_hold:
            raise ValueError(
                "Expected phase fraction in [0,1], nonnegative remaining seconds and a binary terminal-hold flag."
            )
        geometry_table = self.geometry_encoder(
            inputs.geometry_points, inputs.geometry_metadata
        )
        program_table, global_program = self.program_encoder(
            inputs.program_tokens, inputs.program_mask
        )
        if not inputs.program_mask[inputs.program_index, inputs.current_phase].all():
            raise ValueError("Current phase must select a valid program token.")
        geometry = geometry_table[inputs.geometry_index]
        current = program_table[inputs.program_index, inputs.current_phase]
        task = self.task_context(
            torch.cat(
                (global_program[inputs.program_index], current, inputs.progress), -1
            )
        )
        condition = self.condition_encoder(torch.cat((geometry, task), -1))
        state = self.state_norm(self.state_encoder(inputs.state), condition)
        reference = torch.cat(
            (
                inputs.window.masked_fill(~inputs.window_mask[..., None], 0),
                inputs.window_mask[..., None],
            ),
            -1,
        ).flatten(1)
        reference = self.reference_norm(self.reference_encoder(reference), condition)
        hidden = self.trunk(torch.cat((state, reference, geometry, task), -1))
        candidates = self.reference_head(hidden).reshape(
            batch, 10, self.schema.residual_dim
        )
        logits = (
            -0.5
            * (inputs.state.new_tensor(OFFSETS) / self.config.window_sigma).square()
        )
        weights = (
            logits.expand(batch, -1)
            .masked_fill(~inputs.window_mask, -torch.inf)
            .softmax(-1)
        )
        mean = (candidates * weights[..., None]).sum(1)
        return (
            Normal(mean, self.log_std.exp().expand_as(mean)),
            self.value_head(hidden).squeeze(-1),
            weights,
        )

    def log_prob(
        self, inputs: GeometryPolicyInputs, residual: torch.Tensor
    ) -> torch.Tensor:
        distribution, _, _ = self(inputs)
        if residual.shape != distribution.mean.shape:
            raise ValueError(
                f"Expected residual shape {tuple(distribution.mean.shape)}, received {tuple(residual.shape)}."
            )
        return distribution.log_prob(residual).sum(-1)


def make_geometry_policy(
    schema: GeometryPolicySchema, config: GeometryPolicyConfig
) -> GeometryCupPolicy:
    with torch.device("cpu"):
        return GeometryCupPolicy(schema, config)
