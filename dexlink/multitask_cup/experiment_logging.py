"""Explicit training/evaluation axes and complete-cohort success aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "dexlink.workflows")
SUCCESS_DEFINITIONS = {
    "G": "cup_goal_pose_hold_v1",
    "Q": "cup_goal_pose_hold_v1",
    "T": "cup_goal_pose_hold_v1",
    "S": "cup_ordered_orientation_holds_v1",
    "P": "cup_placement_release_contact_stability_v1",
}


def configure_wandb_storage(run_directory: Path) -> dict[str, str]:
    """Keep SDK cache/artifact staging within the run, independent of home quota."""
    root = run_directory.resolve(strict=True)
    directories = {
        "WANDB_CACHE_DIR": root / "wandb-cache",
        "WANDB_DATA_DIR": root / "wandb-data",
    }
    for variable, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["WANDB_DIR"] = str(root)
    return {key: os.environ[key] for key in (*directories, "WANDB_DIR")}


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    axis: str
    success_definition: str

    def __post_init__(self) -> None:
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]+", self.task_id)
            or not self.axis
            or not self.success_definition
        ):
            raise ValueError(
                "Expected a safe nonempty task ID, axis and declared success definition."
            )


@dataclass(frozen=True)
class TaskEvaluation:
    task_id: str
    control_step: int
    checkpoint_sha256: str
    successes: int
    episodes: int
    success_definition: str


def positive_checkpoint(control_step: int, checkpoint_sha256: str) -> None:
    if type(control_step) is not int or control_step <= 0:
        raise ValueError(
            f"Expected a strictly positive checkpoint control step, received {control_step}."
        )
    if (
        not isinstance(checkpoint_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None
    ):
        raise ValueError("Expected the checkpoint's lowercase SHA256 digest.")


class EvaluationRound:
    """Publish full-cohort means only after every task has the same checkpoint/cohort."""

    def __init__(
        self,
        tasks: tuple[TaskDefinition, ...],
        control_step: int,
        checkpoint_sha256: str,
        episodes_per_task: int = 50,
    ) -> None:
        positive_checkpoint(control_step, checkpoint_sha256)
        if not tasks or len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("Expected a nonempty unique task cohort.")
        if type(episodes_per_task) is not int or episodes_per_task < 50:
            raise ValueError(
                "Formal evaluation requires at least 50 complete episodes per task."
            )
        self.tasks = {task.task_id: task for task in tasks}
        self.control_step = control_step
        self.checkpoint_sha256 = checkpoint_sha256
        self.episodes_per_task = episodes_per_task
        self.results: dict[str, TaskEvaluation] = {}

    def add(self, result: TaskEvaluation) -> dict[str, int | float | str]:
        if result.task_id not in self.tasks:
            raise ValueError(
                f"Unexpected task {result.task_id}; expected one of {tuple(self.tasks)}."
            )
        if result.task_id in self.results:
            raise ValueError(
                f"Duplicate evaluation for {result.task_id} at control step {self.control_step}."
            )
        if (result.control_step, result.checkpoint_sha256) != (
            self.control_step,
            self.checkpoint_sha256,
        ):
            raise ValueError(
                "All tasks in an evaluation round must use the same positive checkpoint."
            )
        if result.success_definition != self.tasks[result.task_id].success_definition:
            raise ValueError(f"Success definition mismatch for {result.task_id}.")
        if (
            type(result.episodes) is not int
            or result.episodes != self.episodes_per_task
        ):
            raise ValueError(
                f"Expected {self.episodes_per_task} complete episodes for {result.task_id}, received {result.episodes}."
            )
        if (
            type(result.successes) is not int
            or not 0 <= result.successes <= result.episodes
        ):
            raise ValueError(
                "Expected an integer success count between zero and the episode count."
            )
        self.results[result.task_id] = result
        prefix = f"eval/task/{result.task_id}"
        rates = [row.successes / row.episodes for row in self.results.values()]
        metrics: dict[str, int | float | str] = {
            "eval/checkpoint_control_step": self.control_step,
            "eval/checkpoint_sha256": self.checkpoint_sha256,
            prefix + "/success_rate": result.successes / result.episodes,
            prefix + "/successes": result.successes,
            prefix + "/episodes": result.episodes,
            "eval/tasks_evaluated": len(self.results),
            "eval/tasks_expected": len(self.tasks),
            "eval/task_coverage": len(self.results) / len(self.tasks),
            "eval/evaluated_tasks_success_rate_macro": sum(rates) / len(rates),
        }
        if len(self.results) == len(self.tasks):
            metrics.update(
                {
                    "eval/success_rate_macro": sum(rates) / len(rates),
                    "eval/success_rate_micro": sum(
                        row.successes for row in self.results.values()
                    )
                    / sum(row.episodes for row in self.results.values()),
                    "eval/success_rate_worst_task": min(rates),
                }
            )
            for axis in sorted({task.axis for task in self.tasks.values()}):
                group = [
                    self.results[task.task_id]
                    for task in self.tasks.values()
                    if task.axis == axis
                ]
                metrics[f"eval/axis/{axis}/success_rate_macro"] = sum(
                    row.successes / row.episodes for row in group
                ) / len(group)
        return metrics


class MultitaskRunLogger:
    """One learner-owned SDK run; async evaluations use their checkpoint axis."""

    def __init__(
        self,
        run: Run,
        tasks: tuple[TaskDefinition, ...],
        *,
        logging_mode: str = "online",
        episode_progress_definition: str = "Initial-state greedy evaluation: contiguous prefix within3cm/15deg divided by authored transitions; initial frame and terminal padding excluded.",
    ) -> None:
        if logging_mode not in ("online", "offline"):
            raise ValueError(
                f"Expected an explicit online/offline SDK logging mode; received {logging_mode!r}."
            )
        if (
            run.project != WANDB_PROJECT
            or (WANDB_ENTITY and run.entity != WANDB_ENTITY)
        ) or run.settings.mode != logging_mode:
            raise ValueError(
                f"Expected {logging_mode} W&B {WANDB_ENTITY}/{WANDB_PROJECT}."
            )
        if not tasks or len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("Expected a nonempty unique task catalog.")
        self.run = run
        self.tasks = tasks
        self.training_step = 0
        self.rounds: dict[int, EvaluationRound] = {}
        run.config.update(
            {
                "multitask_logging": {
                    "schema": "c11.multitask.metrics.v2",
                    "evaluation_tasks": [asdict(task) for task in tasks],
                    "formal_episodes_per_task": 50,
                    "average": "arithmetic mean of per-task rates, complete cohort only",
                    "episode_progress": episode_progress_definition,
                    "reference_tracking_fraction": "Fraction of authored transitions within 3cm/15deg, including recovery after an earlier tracking error.",
                    "reference_clock_fraction": "Training reference-clock coverage only; intermediate resets prevent interpreting this as tracking success.",
                }
            }
        )
        for axis, pattern in (
            ("train/control_step", "train/*"),
            ("eval/checkpoint_control_step", "eval/*"),
            ("smoke/checkpoint_control_step", "smoke/*"),
        ):
            run.define_metric(axis, summary="last")
            run.define_metric(
                pattern, step_metric=axis, step_sync=False, summary="last"
            )

    def log_training(
        self, control_step: int, metrics: dict[str, int | float]
    ) -> dict[str, int | float]:
        if type(control_step) is not int or control_step <= self.training_step:
            raise ValueError(
                f"Expected training controls greater than {self.training_step}, received {control_step}."
            )
        if "control_step" in metrics or any(
            key.startswith("train/") for key in metrics
        ):
            raise ValueError(
                "Training metric names must be unprefixed and may not override control_step."
            )
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in metrics.values()
        ):
            raise ValueError("Expected finite numeric training metrics.")
        payload = {
            "train/control_step": control_step,
            **{"train/" + key: value for key, value in metrics.items()},
        }
        self.run.log(payload, commit=True)
        self.training_step = control_step
        return payload

    def log_evaluation(self, result: TaskEvaluation) -> dict[str, int | float | str]:
        if result.control_step not in self.rounds:
            self.rounds[result.control_step] = EvaluationRound(
                self.tasks, result.control_step, result.checkpoint_sha256
            )
        payload = self.rounds[result.control_step].add(result)
        self.run.log(payload, commit=True)
        return payload

    def log_pose_smoke(
        self,
        control_step: int,
        checkpoint_sha256: str,
        episodes_by_task: dict[str, tuple[bool, ...]],
    ) -> dict[str, int | float | str]:
        positive_checkpoint(control_step, checkpoint_sha256)
        if not episodes_by_task:
            raise ValueError("Expected measured positive-checkpoint smoke episodes.")
        metrics: dict[str, int | float | str] = {
            "smoke/checkpoint_control_step": control_step,
            "smoke/checkpoint_sha256": checkpoint_sha256,
        }
        rates = []
        for task_id, episodes in episodes_by_task.items():
            if (
                task_id not in {task.task_id for task in self.tasks}
                or not episodes
                or any(type(value) is not bool for value in episodes)
            ):
                raise ValueError(
                    "Expected known smoke tasks with nonempty boolean episode outcomes."
                )
            rate = sum(episodes) / len(episodes)
            rates.append(rate)
            metrics[f"smoke/task/{task_id}/pose_success_rate"] = rate
            metrics[f"smoke/task/{task_id}/episodes"] = len(episodes)
        metrics.update(
            {
                "smoke/pose_success_rate_macro": sum(rates) / len(rates),
                "smoke/tasks_evaluated": len(episodes_by_task),
            }
        )
        self.run.log(metrics, commit=True)
        return metrics
