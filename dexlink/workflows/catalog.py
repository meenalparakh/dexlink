"""Read the task catalog and validate exact deployment identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def repository_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"Expected a repository-relative path without '..', received {relative!r}."
        )
    return ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog() -> dict[str, dict[str, Any]]:
    document = json.loads((ROOT / "tasks/catalog.json").read_text())
    if document["schema"] != "dexlink_tasks_v1":
        raise ValueError(f"Expected dexlink_tasks_v1, received {document['schema']!r}.")
    tasks = document["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Expected a nonempty task list in tasks/catalog.json.")
    identities = [task["id"] for task in tasks]
    if len(set(identities)) != len(tasks):
        raise ValueError("Expected unique task IDs in tasks/catalog.json.")
    return {task["id"]: task for task in tasks}


def task_by_id(identity: str) -> dict[str, Any]:
    tasks = load_catalog()
    if identity not in tasks:
        raise ValueError(
            f"Unknown task {identity!r}. Run 'dexlink list' to list available IDs."
        )
    return tasks[identity]


def validate_task(
    task: dict[str, Any], *, require_checkpoint: bool = True
) -> dict[str, Any]:
    """Check that references, assets and learned state form a complete portable task."""
    reference = repository_path(task["reference"])
    if not reference.is_file():
        raise FileNotFoundError(f"Missing task reference: {reference}.")
    if sha256(reference) != task["reference_sha256"]:
        raise ValueError(f"Reference checksum mismatch for {task['id']}.")
    loaded = np.load(reference, allow_pickle=True)
    if task["reference_format"] == "native_xhand":
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError(
                f"Expected an NPZ native_xhand reference, received {reference}."
            )
        with loaded:
            arrays = {key: loaded[key] for key in loaded.files}
    elif task["reference_format"] == "oakink_sequence":
        if not isinstance(loaded, np.ndarray) or loaded.shape != ():
            raise ValueError(
                f"Expected a scalar dictionary NPY sequence, received {reference}."
            )
        arrays = loaded.item()
        if not isinstance(arrays, dict):
            raise ValueError(f"Expected a sequence dictionary in {reference}.")
    else:
        raise ValueError(f"Unsupported reference format: {task['reference_format']!r}.")
    for name, value in arrays.items():
        if (
            isinstance(value, np.ndarray)
            and np.issubdtype(value.dtype, np.number)
            and not np.isfinite(value).all()
        ):
            raise ValueError(f"Nonfinite values in {task['id']}/{name}.")
    paths = arrays["scene_object_urdf_paths"]
    if not isinstance(paths, np.ndarray) or paths.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional scene_object_urdf_paths, received {type(paths).__name__}."
        )
    for uri in paths.tolist():
        if not repository_path(uri).is_file():
            raise FileNotFoundError(f"Missing {task['id']} scene asset: {uri}.")
    checkpoint = task["checkpoint"]
    if checkpoint is None:
        if require_checkpoint:
            raise FileNotFoundError(
                f"No packaged policy for {task['id']}; generate a reference and train it first."
            )
    else:
        if type(checkpoint["timestep"]) is not int or checkpoint["timestep"] <= 0:
            raise ValueError(
                f"Expected a strictly positive trained checkpoint timestep for {task['id']}."
            )
        if require_checkpoint:
            folder = (
                repository_path(checkpoint["run_directory"])
                / "models"
                / str(checkpoint["timestep"])
            )
            for name, expected in checkpoint["files"].items():
                path = folder / name
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing {path}; extract the checkpoints ZIP into the repository root."
                    )
                if sha256(path) != expected:
                    raise ValueError(f"Checkpoint checksum mismatch: {path}.")
    return {
        "task": task["id"],
        "reference": "verified",
        "scene_assets": len(paths),
        "checkpoint": "verified"
        if checkpoint is not None and require_checkpoint
        else "not_requested",
    }
