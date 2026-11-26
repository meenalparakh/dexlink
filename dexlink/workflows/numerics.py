"""Strict diagnostic instrumentation; no simulation or PPO equations are changed."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path
from typing import Any

import torch


def tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


def require_finite(value: Any, context: str) -> None:
    for tensor in tensors(value):
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f"Nonfinite diagnostic input: {context}, shape={tuple(tensor.shape)}."
            )


@dataclass
class NumericalAudit:
    environment_control_transitions: int = 0
    nonfinite_environment_control_transitions: int = 0
    optimizer_updates: int = 0
    loss_batches: int = 0
    losses: dict[str, float] = field(default_factory=dict)

    def attach(self, agent) -> None:
        environment = agent.env
        original_step = environment.step

        def observed_step(action, *args, **kwargs):
            require_finite(action, "environment actions")
            result = original_step(action, *args, **kwargs)
            observations, rewards = result[:2]
            count = environment.num_envs
            invalid = torch.zeros(count, device=environment.device, dtype=torch.bool)
            for tensor in tensors((observations, rewards)):
                if (
                    tensor.is_floating_point()
                    and tensor.ndim
                    and tensor.shape[0] == count
                ):
                    invalid |= ~torch.isfinite(tensor).reshape(count, -1).all(-1)
            self.environment_control_transitions += count
            self.nonfinite_environment_control_transitions += int(invalid.sum().item())
            if invalid.any():
                raise FloatingPointError(
                    "Nonfinite environment transition detected before insertion into the PPO buffer."
                )
            require_finite((observations, rewards), "learner observations/rewards")
            return result

        environment.step = observed_step
        for network in (
            agent.policy_net,
            agent.value_net,
            agent.policy_obs_processor,
            agent.value_obs_processor,
        ):
            network.register_forward_pre_hook(
                lambda module, inputs: require_finite(inputs, "network inputs")
            )

        def before_update(optimizer, args, kwargs):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    require_finite(parameter.grad, "optimizer gradients")

        def after_update(optimizer, args, kwargs):
            for group in optimizer.param_groups:
                require_finite(group["params"], "updated parameters")
            require_finite(optimizer.state, "optimizer state")
            self.optimizer_updates += 1

        agent.optimizer.register_step_pre_hook(before_update)
        agent.optimizer.register_step_post_hook(after_update)

    def record_losses(self, data: dict) -> None:
        losses = {
            name: float(value)
            for name, value in data.items()
            if "loss" in name and isinstance(value, (int, float))
        }
        for name, value in losses.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite learner loss: {name}={value}.")
        self.losses.update(losses)
        self.loss_batches += bool(losses)

    def save(self, output: Path, *, completed: bool, failure: str | None) -> None:
        result = asdict(self)
        result.update(
            completed=completed,
            failure=failure,
            nonfinite_fraction=self.nonfinite_environment_control_transitions
            / max(self.environment_control_transitions, 1),
            denominator="all parallel environment control transitions",
            diagnostic_only=True,
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
