"""Generalized advantage estimation and stable parameter identities."""

from __future__ import annotations
import hashlib
import torch
from .config import PPOConfig


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    config: PPOConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = rewards.shape
    if (
        rewards.ndim != 2
        or min(shape) < 1
        or any(x.shape != shape for x in (values, next_values, terminated, truncated))
    ):
        raise ValueError("Expected all GAE inputs shaped (steps,environments).")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("Expected boolean termination/truncation masks.")
    # Time limits bootstrap from the pre-reset state; neither boundary links episodes.
    delta = rewards + config.gamma * next_values * ~terminated - values
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for step in range(shape[0] - 1, -1, -1):
        carry = (
            delta[step]
            + config.gamma
            * config.gae_lambda
            * ~(terminated[step] | truncated[step])
            * carry
        )
        advantages[step] = carry
    return advantages, advantages + values


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
