"""Task-balanced fresh native rollouts, including completed-episode progress."""

from __future__ import annotations

import torch
from torch.distributions import Normal

from .config import PPOConfig
from .full_method_config import FullMethodRunConfig
from .full_method_learner import FULL_BATCH_FIELDS
from .learner import gae, parameter_digest
from .task_buffers import BehaviorPolicyIdentity, TaskExploration, TaskRolloutBuffer


def collect_full_method(
    scenes: list,
    policy,
    exploration: TaskExploration,
    horizon: int,
    version: int,
    config: FullMethodRunConfig,
) -> dict:
    identity = BehaviorPolicyIdentity(version, parameter_digest(policy))
    agent = config.method().agent
    gae_config = PPOConfig(gamma=agent.gamma, gae_lambda=agent.lam)
    pieces = []
    task_count = len(scenes[0].bank.lengths)
    completed = torch.zeros((task_count, 3), device=scenes[0].env.device)
    for scene in scenes:
        buffer = TaskRolloutBuffer(version, scene.task_ids, horizon)
        inputs_by_step = []
        clock_fraction = []
        for _ in range(horizon):
            inputs = scene.inputs()
            inputs_by_step.append(
                {
                    name: getattr(inputs, name).detach().clone()
                    for name in FULL_BATCH_FIELDS
                }
            )
            clock_fraction.append(inputs.normalized_reference_time.detach().clone())
            with torch.no_grad():
                distribution, value, _ = policy(inputs)
                behavior = Normal(distribution.mean.cpu(), distribution.stddev.cpu())
                residual = exploration.sample(behavior, scene.task_ids.cpu()).to(
                    scene.env.device
                )
                logp = distribution.log_prob(residual).sum(-1)
            episodes = scene.episode_ids.clone()
            reward, terminated, truncated = scene.step(residual)
            with torch.no_grad():
                next_value = policy(scene.inputs())[1]
            buffer.append(
                version,
                dict(
                    task_id=scene.task_ids,
                    episode_id=episodes,
                    residual=residual,
                    log_prob=logp,
                    reward=reward,
                    value=value,
                    next_value=next_value,
                    terminated=terminated,
                    truncated=truncated,
                ),
            )
            for row in scene.last_episode_metrics:
                completed[row["task_index"]] += completed.new_tensor(
                    (row["progress_sum"], row["episodes"], row["span_sum"])
                )
            done = (terminated | truncated).nonzero().flatten()
            if len(done):
                scene.reset(done, None)
        rows = buffer.consume(version)
        # The native method cuts both termination and time-limit boundaries in GAE.
        boundary = rows["terminated"] | rows["truncated"]
        advantages, returns = gae(
            rows["reward"],
            rows["value"],
            rows["next_value"],
            boundary,
            torch.zeros_like(boundary),
            gae_config,
        )
        pieces.append(
            dict(
                inputs={
                    name: torch.cat([row[name] for row in inputs_by_step])
                    for name in FULL_BATCH_FIELDS
                },
                residual=rows["residual"].flatten(0, 1),
                log_prob=rows["log_prob"].flatten(),
                advantage=advantages.flatten(),
                return_value=returns.flatten(),
                reward=rows["reward"].flatten(),
                task_id=rows["task_id"].flatten(),
                reference_clock_fraction=torch.cat(clock_fraction),
            )
        )
    batch = {
        name: (
            {
                field: torch.cat([piece[name][field] for piece in pieces])
                for field in FULL_BATCH_FIELDS
            }
            if name == "inputs"
            else torch.cat([piece[name] for piece in pieces])
        )
        for name in pieces[0]
    }
    batch["return"] = batch.pop("return_value")
    counts = torch.bincount(batch["task_id"], minlength=task_count)
    if not torch.equal(
        counts[counts > 0],
        torch.full_like(counts[counts > 0], horizon * config.environments_per_task),
    ):
        raise RuntimeError(
            "Every assigned task must provide equal fresh full-method rollout exposure."
        )
    if parameter_digest(policy) != identity.parameter_digest:
        raise RuntimeError(
            "Shared full-method policy changed during rollout collection."
        )
    batch["behavior_identity"] = identity
    batch["episode_metrics"] = completed
    return batch
