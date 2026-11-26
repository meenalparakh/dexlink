"""Distributed PPO that preserves the native global minibatch and loss reductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.distributed as distributed

from dexlink.src.rl.config.base import AgentConfig

from .full_method_config import FullMethodRunConfig
from .full_method_policy import FullMethodInputs
from .learner import parameter_digest
from .task_data import TaskBank

FULL_BATCH_FIELDS = (
    "native_obs",
    "normalized_reference_time",
    "window",
    "window_mask",
    "geometry_index",
    "program_index",
    "current_phase",
    "progress",
)
TABLE_FIELDS = (
    "geometry_points",
    "geometry_metadata",
    "program_tokens",
    "program_mask",
)


@dataclass(frozen=True)
class PPOMinibatch:
    log_prob: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    entropy: torch.Tensor


def native_ppo_loss(
    batch: PPOMinibatch, agent: AgentConfig, world_size: int, global_samples: int
) -> tuple[torch.Tensor, dict]:
    """DDP averages gradients: compensate native actor sums and weight value/entropy means."""
    count = len(batch.log_prob)
    if count < 1 or global_samples < count or world_size < 1:
        raise ValueError(
            "Expected a positive local minibatch within the declared global sample count."
        )
    if any(
        value.shape != (count,)
        for value in (
            batch.old_log_prob,
            batch.advantages,
            batch.values,
            batch.returns,
            batch.entropy,
        )
    ):
        raise ValueError("Expected matching scalar PPO sample vectors.")
    ratio = (batch.log_prob - batch.old_log_prob).exp()
    actor = -torch.minimum(
        ratio * batch.advantages,
        ratio.clamp(1 - agent.clip_ratio, 1 + agent.clip_ratio) * batch.advantages,
    ).sum()
    critic = (batch.values - batch.returns).square().mean()
    entropy = batch.entropy.mean()
    mean_weight = world_size * count / global_samples
    loss = world_size * actor + mean_weight * (
        agent.vf_coef * critic - agent.entropy_coef * entropy
    )
    return loss, dict(actor=actor, critic=critic, entropy=entropy, ratio=ratio)


def local_global_minibatches(
    order: torch.Tensor, counts: tuple[int, ...], rank: int, batch_size: int
) -> Iterator[tuple[torch.Tensor, int]]:
    """Every rank takes its samples from the same shuffled global minibatch; no data replay or padding."""
    if rank < 0 or rank >= len(counts) or min(counts) < 1 or batch_size < 1:
        raise ValueError(
            "Expected positive rank counts, a valid rank and a positive global minibatch size."
        )
    if order.shape != (sum(counts),) or order.dtype != torch.long:
        raise ValueError(
            "Expected one int64 global permutation covering the complete fresh rollout."
        )
    start = sum(counts[:rank])
    for indices in order.split(batch_size):
        local = indices[(indices >= start) & (indices < start + counts[rank])] - start
        if not len(local):
            raise ValueError(
                "A global minibatch gave one learning rank no samples; use a batch with nonempty rank coverage."
            )
        yield local, len(indices)


def standardized_advantages(
    advantages: torch.Tensor, moments: torch.Tensor
) -> torch.Tensor:
    total, squares, count = moments
    if count <= 1:
        raise ValueError(
            "Native unbiased advantage normalization requires at least two global samples."
        )
    mean = total / count
    variance = ((squares - count * mean.square()) / (count - 1)).clamp_min(0)
    return (advantages - mean) / (variance.sqrt() + 1e-8)


def update_full_method(
    model,
    optimizer,
    batch: dict,
    bank: TaskBank,
    config: FullMethodRunConfig,
    version: int,
) -> dict[str, float]:
    batch["behavior_identity"].consume(version, parameter_digest(model.module))
    rank = distributed.get_rank()
    device = bank.wrist.device
    agent = config.method().agent
    count = len(batch["advantage"])
    all_counts = [None] * config.world_size
    distributed.all_gather_object(all_counts, count)
    counts = tuple(all_counts)
    moments = torch.stack(
        (
            batch["advantage"].sum(),
            batch["advantage"].square().sum(),
            batch["advantage"].new_tensor(count),
        )
    )
    distributed.all_reduce(moments)
    if agent.normalize_advantages:
        advantage = standardized_advantages(batch["advantage"], moments)
    else:
        advantage = batch["advantage"] - moments[0] / moments[2]
    diagnostics = torch.zeros(3, device=device)
    ratio_error = torch.zeros((), device=device)
    updates = 0
    first = True
    for _ in range(agent.update_epochs):
        # Native Memory.sample_indices uses NumPy permutation. Only rank0 advances its global sampler RNG.
        order = (
            torch.from_numpy(np.random.permutation(sum(counts))).to(device)
            if rank == 0
            else torch.empty(sum(counts), device=device, dtype=torch.long)
        )
        distributed.broadcast(order, src=0)
        for selected, global_count in local_global_minibatches(
            order, counts, rank, agent.batch_size
        ):
            inputs = FullMethodInputs(
                **{name: batch["inputs"][name][selected] for name in FULL_BATCH_FIELDS},
                **{name: getattr(bank, name) for name in TABLE_FIELDS},
            )
            distribution, value, _ = model(inputs)
            data = PPOMinibatch(
                distribution.log_prob(batch["residual"][selected]).sum(-1),
                batch["log_prob"][selected],
                advantage[selected],
                value,
                batch["return"][selected],
                distribution.entropy().sum(-1),
            )
            loss, metrics = native_ppo_loss(
                data, agent, config.world_size, global_count
            )
            if first:
                ratio_error = (metrics["ratio"].detach() - 1).abs().max()
                distributed.all_reduce(ratio_error, op=distributed.ReduceOp.MAX)
                first = False
            if not torch.isfinite(loss):
                raise RuntimeError("Native full-method PPO produced a nonfinite loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise RuntimeError(
                    "Native full-method PPO produced nonfinite gradients."
                )
            # Preserve the native AdamW update: no extra gradient clipping or loss rescaling.
            optimizer.step()
            diagnostics += torch.stack(
                (
                    metrics["actor"].detach(),
                    metrics["critic"].detach() * len(selected),
                    metrics["entropy"].detach() * len(selected),
                )
            )
            updates += 1
    distributed.all_reduce(diagnostics)
    rewards = torch.zeros((len(bank.lengths), 3), device=device)
    rewards[:, 0].index_add_(0, batch["task_id"], batch["reward"])
    rewards[:, 1].index_add_(0, batch["task_id"], torch.ones_like(batch["reward"]))
    rewards[:, 2].index_add_(0, batch["task_id"], batch["reference_clock_fraction"])
    distributed.all_reduce(rewards)
    completed = batch["episode_metrics"].clone()
    distributed.all_reduce(completed)
    result = dict(
        policy_loss=diagnostics[0].item() / updates,
        value_loss=diagnostics[1].item() / (sum(counts) * agent.update_epochs),
        entropy=diagnostics[2].item() / (sum(counts) * agent.update_epochs),
        optimizer_updates_this_rollout=updates,
        pre_update_ratio_max_error=ratio_error.item(),
        mean_reward=(rewards[:, 0].sum() / rewards[:, 1].sum()).item(),
        reference_clock_fraction=(rewards[:, 2].sum() / rewards[:, 1].sum()).item(),
        completed_episode_count=completed[:, 1].sum().item(),
    )
    progress = []
    for task in config.task_indices:
        result[f"task_index/{task}/mean_reward"] = (
            rewards[task, 0] / rewards[task, 1]
        ).item()
        result[f"task_index/{task}/reference_clock_fraction"] = (
            rewards[task, 2] / rewards[task, 1]
        ).item()
        result[f"task_index/{task}/completed_episodes"] = completed[task, 1].item()
        if completed[task, 1] > 0:
            value = (completed[task, 0] / completed[task, 1]).item()
            result[f"task_index/{task}/episode_progress"] = value
            progress.append(value)
    result["episode_progress_task_coverage"] = len(progress) / len(config.task_indices)
    if len(progress) == len(config.task_indices):
        result["episode_progress"] = sum(progress) / len(progress)
    return result
