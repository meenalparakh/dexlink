"""Validate the external data and in-repository assets required by one task."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexlink.scripts.retargeting.run_task_category_retargetings import (
    collect_category_retarget_requests,
)
from dexlink.src.reference.common.exp_info import (
    get_source_template,
    get_target_object,
)
from dexlink.src.reference.task_io import load_task_spec
from dexlink.utils.retargeting.path_utils import (
    get_external_data_dir,
    resolve_project_path,
)


INPUT_PATH_KEYS = {"mesh_path", "urdf_path", "target_demo_path"}


@dataclass(frozen=True)
class ValidationArgs:
    task: Path
    hand_type: str


def _parse_args(argv: list[str] | None = None) -> ValidationArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--hand-type", choices=("xhand",), default="xhand")
    parsed = parser.parse_args(argv)
    return ValidationArgs(task=parsed.task, hand_type=str(parsed.hand_type))


def _declared_input_paths(value: Any, *, key: str = "") -> set[Path]:
    paths: set[Path] = set()
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            paths.update(_declared_input_paths(child_value, key=str(child_key)))
        return paths
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child_value in value:
            paths.update(_declared_input_paths(child_value, key=key))
        return paths
    if isinstance(value, str) and key in INPUT_PATH_KEYS and value != "auto":
        path = Path(value)
        if not path.parts or path.parts[0] != "outputs":
            paths.add(resolve_project_path(path))
    return paths


def required_task_inputs(task: dict[str, Any], hand_type: str) -> tuple[Path, ...]:
    """Return every file that must exist before retargeting can begin."""

    required = _declared_input_paths(task)
    requests = collect_category_retarget_requests(task, hand_type)
    for request in requests:
        source = get_source_template(request.category, request.source_index)
        required.add(resolve_project_path(source.demo_path))

        target = get_target_object(request.category, request.target_index)
        if target.mesh_path is not None:
            required.add(resolve_project_path(target.mesh_path))
        if target.target_demo_path is not None:
            required.add(resolve_project_path(target.target_demo_path))
        if target.transform_path is not None:
            required.add(resolve_project_path(target.transform_path))

    # OakInk processing reconstructs human hand geometry from licensed MANO data.
    oakink_requests = tuple(
        request
        for request in requests
        if get_source_template(request.category, request.source_index).dataset
        == "oakink"
    )
    if oakink_requests:
        mano_models = get_external_data_dir() / "mano" / "mano_v1_2" / "models"
        required.add(mano_models / "MANO_RIGHT.pkl")
        if any(request.hand_side == "left" for request in oakink_requests):
            required.add(mano_models / "MANO_LEFT.pkl")
    return tuple(sorted(required))


def validate_task_inputs(args: ValidationArgs) -> tuple[Path, ...]:
    task_path = resolve_project_path(args.task)
    task = load_task_spec(task_path)
    required = required_task_inputs(task, args.hand_type)
    missing = tuple(path for path in required if not path.exists())
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Task {task['experiment_id']!r} is missing {len(missing)} required "
            f"input files:\n{formatted}\n"
            "Install or symlink the licensed datasets as described in "
            "external_data/README.md."
        )
    return required


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    required = validate_task_inputs(args)
    print(f"[dexlink] validated {len(required)} task inputs for {args.task}")


if __name__ == "__main__":
    main()
