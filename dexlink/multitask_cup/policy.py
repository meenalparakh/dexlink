"""Reference-length-independent policy and explicit residual-space likelihoods."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.distributions import Normal

from .config import ModelConfig

OFFSETS = tuple(range(-4, 6))
CONDITION_SCHEMA = (
    "geometry_extent_xyz:3",
    "initial_object_pose_wxyz:7",
    "goal_object_pose_wxyz:7",
    "phase_frames_div100:6",
    "signed_y_rotation_rad:1",
    "placement_required:1",
    "phase_onehot:6",
    "phase_fraction:1",
    "elapsed_fraction:1",
)
CONDITION_DIM = 33


def local_window(
    reference: torch.Tensor, timesteps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference.ndim != 2 or reference.shape[0] < 1:
        raise ValueError(
            f"Expected nonempty reference (T,F), got {tuple(reference.shape)}."
        )
    if timesteps.ndim != 1 or timesteps.dtype != torch.long:
        raise ValueError("Expected timesteps as a one-dimensional int64 tensor.")
    if torch.any(timesteps < 0) or torch.any(timesteps >= reference.shape[0]):
        raise ValueError(f"Expected timesteps in [0,{reference.shape[0] - 1}].")
    indices = timesteps[:, None] + timesteps.new_tensor(OFFSETS)
    valid = (indices >= 0) & (indices < reference.shape[0])
    # Clamped values keep the interface defined even for T < 10; masks exclude them.
    return reference[indices.clamp(0, reference.shape[0] - 1)], valid


@dataclass(frozen=True)
class PolicySchema:
    state_dim: int
    reference_dim: int
    residual_dim: int
    condition_dim: int = CONDITION_DIM
    version: str = "c11.adarms.local10.v1"


class AdaptiveRMSNorm(nn.Module):
    """Task-conditioned scale, shift and residual gate; no running statistics."""

    def __init__(self, width: int, condition_dim: int) -> None:
        super().__init__()
        self.modulation = nn.Linear(condition_dim, 3 * width)
        nn.init.normal_(self.modulation.weight, std=0.01)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, embedding: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift, gate = self.modulation(condition).chunk(3, dim=-1)
        normalized = embedding * torch.rsqrt(
            embedding.square().mean(-1, keepdim=True) + 1e-6
        )
        return embedding + torch.sigmoid(gate) * ((1 + scale) * normalized + shift)


class CupPolicy(nn.Module):
    def __init__(self, schema: PolicySchema, config: ModelConfig) -> None:
        super().__init__()
        if (
            min(
                schema.state_dim,
                schema.reference_dim,
                schema.residual_dim,
                config.hidden_dim,
            )
            < 1
        ):
            raise ValueError("Policy dimensions must be positive.")
        if schema.condition_dim != CONDITION_DIM or config.window_sigma <= 0:
            raise ValueError(
                "Expected the declared conditioning schema and positive window sigma."
            )
        self.schema = schema
        self.config = config
        width = config.hidden_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(schema.state_dim, width), nn.SiLU()
        )
        self.reference_encoder = nn.Sequential(
            nn.Linear(10 * (schema.reference_dim + 1), width), nn.SiLU()
        )
        self.state_norm = AdaptiveRMSNorm(width, schema.condition_dim)
        self.reference_norm = AdaptiveRMSNorm(width, schema.condition_dim)
        self.trunk = nn.Sequential(
            nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU()
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
            "conditioning": list(CONDITION_SCHEMA),
        }

    def forward(
        self,
        state: torch.Tensor,
        window: torch.Tensor,
        mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        expected = (
            (batch, self.schema.state_dim),
            (batch, 10, self.schema.reference_dim),
            (batch, 10),
            (batch, self.schema.condition_dim),
        )
        for name, tensor, shape in zip(
            ("state", "window", "mask", "condition"),
            (state, window, mask, condition),
            expected,
        ):
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"Expected {name} shape {shape}, got {tuple(tensor.shape)}."
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Expected finite {name} values.")
        if mask.dtype != torch.bool or not mask[:, 4].all():
            raise ValueError(
                "Expected boolean mask with the current-reference slot valid."
            )
        state_embed = self.state_norm(self.state_encoder(state), condition)
        masked_window = window.masked_fill(~mask[..., None], 0)
        reference_input = torch.cat((masked_window, mask[..., None]), dim=-1).flatten(1)
        reference_embed = self.reference_norm(
            self.reference_encoder(reference_input), condition
        )
        hidden = self.trunk(torch.cat((state_embed, reference_embed), dim=-1))
        candidates = self.reference_head(hidden).reshape(
            batch, 10, self.schema.residual_dim
        )
        offset = state.new_tensor(OFFSETS)
        logits = -0.5 * (offset / self.config.window_sigma).square()
        weights = logits.expand(batch, -1).masked_fill(~mask, -torch.inf).softmax(-1)
        mean = (candidates * weights[..., None]).sum(1)
        return (
            Normal(mean, self.log_std.exp().expand_as(mean)),
            self.value_head(hidden).squeeze(-1),
            weights,
        )

    def log_prob(
        self, inputs: dict[str, torch.Tensor], residual: torch.Tensor
    ) -> torch.Tensor:
        distribution, _, _ = self(**inputs)
        if residual.shape != distribution.mean.shape:
            raise ValueError(
                f"Expected residual shape {distribution.mean.shape}, got {residual.shape}."
            )
        return distribution.log_prob(residual).sum(-1)


def make_cpu_policy(schema: PolicySchema, config: ModelConfig) -> CupPolicy:
    """Keep inference/learning on CPU even after Genesis changes Torch's default device."""
    with torch.device("cpu"):
        return CupPolicy(schema, config)


def targets_from_residual(
    residual: torch.Tensor,
    window: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Single right-hand absolute target: blended position/fingers, current quaternion.

    Reference features start with wrist position(3), wxyz(4), fingers(J); object
    pose follows. The PPO action is the Gaussian residual, not this IK target.
    """
    if residual.ndim != 2:
        raise ValueError(f"Expected residual (B,6+J), got {tuple(residual.shape)}.")
    joints = residual.shape[1] - 6
    if joints < 1 or window.shape != (residual.shape[0], 10, 14 + joints):
        raise ValueError("Expected reference window (B,10,14+J) and residual (B,6+J).")
    if scales.shape != (6 + joints,) or torch.any(scales <= 0):
        raise ValueError("Expected positive scales (6+J,).")
    if weights.shape != (residual.shape[0], 10) or torch.any(weights < 0):
        raise ValueError("Expected nonnegative blending weights (B,10).")
    if not torch.allclose(weights.sum(-1), torch.ones_like(weights[:, 0]), atol=1e-6):
        raise ValueError("Expected blending weights summing to one per sample.")
    quaternion_norm = torch.linalg.vector_norm(window[:, 4, 3:7], dim=-1)
    if not torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), atol=1e-5):
        raise ValueError("Expected unit reference quaternions in wxyz order.")
    scaled = residual * scales
    base = (window * weights[..., None]).sum(1)
    angle = torch.linalg.vector_norm(scaled[:, 3:6], dim=-1, keepdim=True)
    xyz = scaled[:, 3:6] * 0.5 * torch.sinc(angle / (2 * torch.pi))
    delta = torch.cat((torch.cos(angle / 2), xyz), dim=-1)
    q = window[:, 4, 3:7]
    w = q[:, :1] * delta[:, :1] - (q[:, 1:] * delta[:, 1:]).sum(-1, keepdim=True)
    v = (
        q[:, :1] * delta[:, 1:]
        + delta[:, :1] * q[:, 1:]
        + torch.cross(q[:, 1:], delta[:, 1:], dim=-1)
    )
    return torch.cat(
        (base[:, :3] + scaled[:, :3], w, v, base[:, 7 : 7 + joints] + scaled[:, 6:]),
        dim=-1,
    )
