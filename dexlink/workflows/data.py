"""Collect policy demonstrations and render saved motions as separate operations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from dexlink.utils.retargeting.path_utils import get_logs_dir

from .catalog import load_catalog, repository_path, task_by_id, validate_task


@dataclass(frozen=True)
class CollectionArguments:
    tasks: list[str]
    output: Path
    group: str | None = None
    episodes: int = 50
    parallel_envs: int = 10
    seed: int = 42


@dataclass(frozen=True)
class RenderArguments:
    output: Path
    task: str | None = None
    reference: Path | None = None
    eval_directory: Path | None = None
    reference_override: Path | None = None
    task_id: str = "custom"
    episode: int = 0
    width: int = 640
    height: int = 480
    frame_stride: int = 2
    fps: float = 15.0
    camera_mode: str = "tracked"


def index_rollout_chunks(directory: Path, expected_episodes: int) -> dict:
    """Index complete first episodes; post-reset/padded rows are excluded by length."""
    paths = sorted((directory / "rollouts").glob("chunk_*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"Expected saved rollout chunks in {directory / 'rollouts'}."
        )
    chunks = []
    episode_count = 0
    transition_count = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            done = arrays["done"]
            if done.ndim != 2 or done.dtype != np.bool_ or not done.any(axis=0).all():
                raise ValueError(
                    f"Expected boolean done(T,N) with a terminal transition in every episode: {path}."
                )
            lengths = done.argmax(axis=0) + 1
            valid = np.arange(done.shape[0])[:, None] < lengths[None, :]
            for key in ("policy_observation", "policy_action"):
                values = arrays[key]
                if (
                    values.ndim != 3
                    or values.shape[:2] != done.shape
                    or not np.isfinite(values[valid]).all()
                ):
                    raise ValueError(
                        f"Expected finite {key}(T,N,D) aligned with done{done.shape}, received {values.shape} at {path}."
                    )
            chunks.append(
                {
                    "path": str(path.relative_to(directory)),
                    "first_episode_index": episode_count,
                    "episode_lengths": lengths.tolist(),
                    "valid_transitions": int(valid.sum()),
                    "fields": {
                        key: {
                            "shape": list(arrays[key].shape),
                            "dtype": str(arrays[key].dtype),
                        }
                        for key in arrays.files
                    },
                }
            )
            episode_count += done.shape[1]
            transition_count += int(valid.sum())
    if episode_count != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} episodes, received {episode_count} in {directory}."
        )
    return {
        "episodes": episode_count,
        "valid_transitions": transition_count,
        "chunks": chunks,
    }


def collect(arguments: CollectionArguments) -> None:
    from .cli import RolloutArguments, rollout

    if bool(arguments.tasks) == (arguments.group is not None):
        raise ValueError("Specify task IDs or --group, exactly one selection route.")
    identities = arguments.tasks or [
        task["id"]
        for task in load_catalog().values()
        if task["group"] == arguments.group
    ]
    if (
        not identities
        or len(identities) != len(set(identities))
        or arguments.episodes < 1
        or arguments.parallel_envs < 1
    ):
        raise ValueError(
            "Expected unique task IDs and positive episode/environment counts."
        )
    tasks = [task_by_id(identity) for identity in identities]
    for task in tasks:
        validate_task(task)
    destination = arguments.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = destination / "dataset.json"
    manifest = {
        "schema": "dexlink_policy_demonstrations_v1",
        "seed": arguments.seed,
        "episodes_per_task": arguments.episodes,
        "parallel_envs": arguments.parallel_envs,
        "greedy_actions": True,
        "status": "collecting",
        "tasks": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for task in tasks:
            folder = destination / task["id"]
            rollout(
                RolloutArguments(
                    task=task["id"],
                    episodes=arguments.episodes,
                    parallel_envs=arguments.parallel_envs,
                    output=folder,
                    seed=arguments.seed,
                ),
                [],
                record_policy_inputs=True,
            )
            index = index_rollout_chunks(folder, arguments.episodes)
            (folder / "dataset-index.json").write_text(
                json.dumps(index, indent=2) + "\n"
            )
            manifest["tasks"].append(
                {
                    "task": task["id"],
                    "directory": task["id"],
                    "reference": task["reference"],
                    "reference_sha256": task["reference_sha256"],
                    "checkpoint": task["checkpoint"],
                    "episodes": index["episodes"],
                    "valid_transitions": index["valid_transitions"],
                    "metrics": json.loads(
                        (folder / "summary_metrics.json").read_text()
                    ),
                }
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


def render(arguments: RenderArguments) -> None:
    from .cli import run_module

    if (
        sum(
            value is not None
            for value in (arguments.task, arguments.reference, arguments.eval_directory)
        )
        != 1
    ):
        raise ValueError(
            "Choose exactly one of --task, --reference or --eval-directory."
        )
    if arguments.reference_override is not None and arguments.eval_directory is None:
        raise ValueError("--reference-override is only valid with --eval-directory.")
    output = arguments.output.resolve()
    if output.suffix != ".mp4" or output.exists():
        raise ValueError(f"Expected a new .mp4 output, received {output}.")
    command = [
        "--output-path",
        str(output),
        "--metadata-path",
        str(output.with_suffix(".json")),
        "--cache-dir",
        str(get_logs_dir() / "cache/rendering"),
        "--episode-index",
        str(arguments.episode),
        "--width",
        str(arguments.width),
        "--height",
        str(arguments.height),
        "--frame-stride",
        str(arguments.frame_stride),
        "--fps",
        str(arguments.fps),
        "--camera-mode",
        arguments.camera_mode,
    ]
    task_id = arguments.task_id
    if arguments.task is not None:
        task = task_by_id(arguments.task)
        validate_task(task, require_checkpoint=False)
        command.extend(["--reference-path", str(repository_path(task["reference"]))])
        task_id = task["id"]
    elif arguments.reference is not None:
        command.extend(
            ["--reference-path", str(arguments.reference.resolve(strict=True))]
        )
    else:
        command.extend(
            ["--eval-dir", str(arguments.eval_directory.resolve(strict=True))]
        )
        if arguments.reference_override is not None:
            command.extend(
                [
                    "--reference-path",
                    str(arguments.reference_override.resolve(strict=True)),
                ]
            )
    command.extend(["--task-id", task_id])
    run_module("dexlink.visualizations.rl.render_clean_policy_rollout_video", command)
