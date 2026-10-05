"""Load and validate public DexLink task specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from dexlink.utils.retargeting.path_utils import get_logs_dir, get_src


REQUIRED_TASK_KEYS = ("experiment_id", "keyframes", "trajectory")
PATH_KEY_SUFFIXES = ("_path", "_dir", "_root")


def load_task_spec(path: str | Path) -> dict[str, Any]:
    """Load one YAML task and reject malformed or machine-specific content."""

    task_path = Path(path)
    task_path = task_path if task_path.is_absolute() else get_src() / task_path
    if task_path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Expected a YAML task file, got {task_path}.")
    if not task_path.is_file():
        raise FileNotFoundError(f"Task file does not exist: {task_path}.")
    loaded = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Expected a mapping at the root of {task_path}, got "
            f"{type(loaded).__name__}."
        )
    missing = [key for key in REQUIRED_TASK_KEYS if key not in loaded]
    if missing:
        raise ValueError(f"Task {task_path} is missing required keys {missing}.")
    if not isinstance(loaded["experiment_id"], str) or not loaded["experiment_id"]:
        raise TypeError("Expected non-empty string 'experiment_id'.")
    if not isinstance(loaded["keyframes"], list) or not loaded["keyframes"]:
        raise TypeError("Expected 'keyframes' to be a non-empty list.")
    trajectory = loaded["trajectory"]
    if not isinstance(trajectory, dict):
        raise TypeError(
            f"Expected 'trajectory' to be a mapping, got "
            f"{type(trajectory).__name__}."
        )
    phases = trajectory.get("phases")
    if not isinstance(phases, list) or not phases:
        raise TypeError("Expected trajectory.phases to be a non-empty list.")
    for field_name in ("output_motion_path", "exported_rl_path"):
        field_value = trajectory.get(field_name)
        if not isinstance(field_value, str) or not field_value:
            raise TypeError(
                f"Expected trajectory.{field_name} to be a non-empty string."
            )
    _reject_unresolved_values(loaded, location=str(task_path))
    _reject_absolute_paths(loaded, location=str(task_path))
    return loaded


def task_output_dir(task: Mapping[str, Any]) -> Path:
    experiment_id = task.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise TypeError("Expected task['experiment_id'] to be a non-empty string.")
    return get_logs_dir() / experiment_id


def _reject_absolute_paths(value: Any, *, location: str, key: str = "") -> None:
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _reject_absolute_paths(
                child_value,
                location=f"{location}.{child_key}",
                key=str(child_key),
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child_value in enumerate(value):
            _reject_absolute_paths(
                child_value,
                location=f"{location}[{index}]",
                key=key,
            )
        return
    if (
        isinstance(value, str)
        and key.endswith(PATH_KEY_SUFFIXES)
        and Path(value).is_absolute()
    ):
        raise ValueError(
            f"Task paths must be repository-relative; found {value!r} at {location}."
        )


def _reject_unresolved_values(value: Any, *, location: str) -> None:
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _reject_unresolved_values(
                child_value,
                location=f"{location}.{child_key}",
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child_value in enumerate(value):
            _reject_unresolved_values(
                child_value,
                location=f"{location}[{index}]",
            )
        return
    if value == "TBD":
        raise ValueError(f"Task contains unresolved value 'TBD' at {location}.")
