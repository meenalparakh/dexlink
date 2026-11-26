"""Strict current-policy PPO ownership and task-local exploration/reset history."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.distributions import Normal


@dataclass
class BehaviorPolicyIdentity:
    """Claim each fresh rollout once, against the exact policy that collected it."""

    policy_version: int
    parameter_digest: str
    consumed: bool = False

    def consume(self, version: int, digest: str) -> None:
        if (
            self.consumed
            or version != self.policy_version
            or digest != self.parameter_digest
        ):
            raise ValueError(
                "Expected unconsumed rollout data from the current policy version and parameters."
            )
        self.consumed = True


class TaskExploration:
    """Task-local Gaussian streams, independent of actor grouping and other task resets."""

    def __init__(self, task_count: int, seed: int = 42) -> None:
        if task_count < 1:
            raise ValueError("Expected a positive number of task exploration streams.")
        self.generators = [
            torch.Generator().manual_seed(seed + 7919 * task)
            for task in range(task_count)
        ]

    def sample(self, distribution: Normal, task_ids: torch.Tensor) -> torch.Tensor:
        if (
            task_ids.ndim != 1
            or task_ids.dtype != torch.long
            or task_ids.device.type != "cpu"
        ):
            raise ValueError(
                "Expected CPU int64 task labels for CPU behavior-policy sampling."
            )
        if (
            distribution.mean.shape != (len(task_ids), 18)
            or distribution.mean.device.type != "cpu"
        ):
            raise ValueError("Expected CPU Gaussian residuals (B,18).")
        if (task_ids < 0).any() or (task_ids >= len(self.generators)).any():
            raise ValueError("Task exploration index out of range.")
        noise = torch.empty_like(distribution.mean)
        # Separate generators are necessary for invariance to task sharding and resets.
        for task in task_ids.unique().tolist():
            slots = (task_ids == task).nonzero().flatten()
            noise[slots] = torch.randn(
                (len(slots), 18),
                generator=self.generators[task],
                device="cpu",
                dtype=noise.dtype,
            )
        return distribution.mean + distribution.stddev * noise

    def state_dict(self) -> dict:
        return {
            "generator_states": [generator.get_state() for generator in self.generators]
        }

    def load_state_dict(self, state: dict) -> None:
        if set(state) != {"generator_states"} or len(state["generator_states"]) != len(
            self.generators
        ):
            raise ValueError(
                "Exploration state does not match the declared task count."
            )
        for generator, value in zip(self.generators, state["generator_states"]):
            generator.set_state(value)


@dataclass
class TaskRolloutBuffer:
    policy_version: int
    task_ids: torch.Tensor
    steps: int
    rows: list[dict[str, torch.Tensor]] = field(default_factory=list)
    consumed: bool = False

    def __post_init__(self) -> None:
        if (
            self.policy_version < 0
            or self.steps < 1
            or self.task_ids.ndim != 1
            or self.task_ids.dtype != torch.long
            or not self.task_ids.numel()
        ):
            raise ValueError(
                "Expected a nonnegative policy version, positive rollout length and nonempty int64 task slots."
            )
        if (self.task_ids < 0).any():
            raise ValueError("Task IDs must be nonnegative.")
        self.task_ids = self.task_ids.clone()

    def append(self, version: int, row: dict[str, torch.Tensor]) -> None:
        if self.consumed or version != self.policy_version:
            raise ValueError("Cannot append consumed or stale policy experience.")
        if len(self.rows) >= self.steps:
            raise ValueError("The current on-policy rollout is already full.")
        required = {
            "task_id",
            "episode_id",
            "residual",
            "log_prob",
            "reward",
            "value",
            "next_value",
            "terminated",
            "truncated",
        }
        if set(row) != required or not torch.equal(row["task_id"], self.task_ids):
            raise ValueError(
                "Expected the declared fixed task slots and complete rollout fields."
            )
        batch = len(self.task_ids)
        for key, value in row.items():
            shape = (batch, 18) if key == "residual" else (batch,)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"Expected {key} shape {shape}, received {tuple(value.shape)}."
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"Expected finite rollout {key} values.")
        if row["episode_id"].dtype != torch.long or (row["episode_id"] < 0).any():
            raise ValueError("Expected nonnegative int64 episode IDs.")
        if (
            row["terminated"].dtype != torch.bool
            or row["truncated"].dtype != torch.bool
        ):
            raise ValueError("Expected boolean terminal and time-limit flags.")
        if self.rows:
            previous = self.rows[-1]
            expected = (
                previous["episode_id"]
                + (previous["terminated"] | previous["truncated"]).long()
            )
            if not torch.equal(row["episode_id"], expected):
                raise ValueError(
                    "Episode IDs must advance exactly at each slot's reset boundary."
                )
        self.rows.append({key: value.detach().clone() for key, value in row.items()})

    def consume(self, version: int) -> dict[str, torch.Tensor]:
        if (
            self.consumed
            or version != self.policy_version
            or len(self.rows) != self.steps
        ):
            raise ValueError(
                "Expected a complete, fresh, unconsumed rollout from this policy version."
            )
        result = {
            key: torch.stack([row[key] for row in self.rows]) for key in self.rows[0]
        }
        self.rows.clear()
        self.consumed = True
        return result


class TaskResetStatistics:
    """One masked frame-density table and independent generator per task."""

    def __init__(self, lengths: torch.Tensor, seed: int = 42) -> None:
        if (
            lengths.ndim != 1
            or lengths.dtype != torch.long
            or (lengths <= 0).any()
            or lengths.device.type != "cpu"
        ):
            raise ValueError("Expected positive CPU int64 task lengths.")
        self.lengths = lengths.clone()
        self.visits = torch.zeros(
            (len(lengths), int(lengths.max())), dtype=torch.long, device="cpu"
        )
        self.generators = [
            torch.Generator().manual_seed(seed + 1009 * i) for i in range(len(lengths))
        ]

    def record(self, task_ids: torch.Tensor, clocks: torch.Tensor) -> None:
        if (
            task_ids.shape != clocks.shape
            or task_ids.ndim != 1
            or task_ids.dtype != torch.long
            or clocks.dtype != torch.long
        ):
            raise ValueError("Expected CPU int64 task/frame vectors of equal length.")
        if task_ids.device.type != "cpu" or clocks.device.type != "cpu":
            raise ValueError("Reset statistics are checkpointed CPU state.")
        if (
            (task_ids < 0).any()
            or (task_ids >= len(self.lengths)).any()
            or (clocks < 0).any()
            or (clocks >= self.lengths[task_ids]).any()
        ):
            raise ValueError("Task/frame outside the approved valid range.")
        self.visits.index_put_(
            (task_ids, clocks), torch.ones_like(clocks), accumulate=True
        )

    def sample(self, task: int, count: int) -> torch.Tensor:
        if not 0 <= task < len(self.lengths) or count < 1:
            raise ValueError("Expected a valid task index and positive reset count.")
        # Rarely visited frames receive larger reset probability; tasks never share density.
        weights = 1 / (1 + self.visits[task, : self.lengths[task]].float())
        return torch.multinomial(
            weights, count, replacement=True, generator=self.generators[task]
        )

    def state_dict(self) -> dict:
        return {
            "lengths": self.lengths.clone(),
            "visits": self.visits.clone(),
            "generator_states": [
                generator.get_state() for generator in self.generators
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        if set(state) != {"lengths", "visits", "generator_states"} or not torch.equal(
            state["lengths"], self.lengths
        ):
            raise ValueError(
                "Reset-state task lineage differs from the current catalog."
            )
        if (
            state["visits"].shape != self.visits.shape
            or state["visits"].dtype != torch.long
            or len(state["generator_states"]) != len(self.generators)
        ):
            raise ValueError(
                "Reset-state table or RNG count differs from the task catalog."
            )
        self.visits.copy_(state["visits"])
        for generator, value in zip(self.generators, state["generator_states"]):
            generator.set_state(value)
