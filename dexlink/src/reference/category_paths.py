"""Canonical paths for source-template to target-object retargeting artifacts."""

from __future__ import annotations

from pathlib import Path

from dexlink.src.reference.common.exp_info import (
    get_source_template,
    get_target_object,
)
from dexlink.utils.retargeting.path_utils import get_logs_dir


SUPPORTED_RETARGET_HAND_TYPES = ("xhand", "inspire_hand")


def normalize_retarget_hand_type(hand_type: str) -> str:
    aliases = {
        "xhand": "xhand",
        "inspire": "inspire_hand",
        "inspire_hand": "inspire_hand",
    }
    normalized = aliases.get(str(hand_type).lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported retarget hand type {hand_type!r}; expected one of "
            f"{SUPPORTED_RETARGET_HAND_TYPES}."
        )
    return normalized


def robot_model_for_hand_type(hand_type: str) -> str:
    normalized = normalize_retarget_hand_type(hand_type)
    return "xhand" if normalized == "xhand" else "inspire"


def category_retarget_job_id(
    category: str,
    source_index: int,
    target_index: int,
    articulation_sample_id: str | None = None,
) -> str:
    source = get_source_template(category, source_index)
    target = get_target_object(category, target_index)
    job_id = (
        f"source_{source_index:02d}_{source.template_id}"
        f"__target_{target_index:02d}_{target.target_id}"
    )
    if articulation_sample_id is not None:
        job_id += f"__articulation_{articulation_sample_id}"
    return job_id


def category_transform_dir(
    root: Path,
    category: str,
    source_index: int,
    target_index: int,
) -> Path:
    source = get_source_template(category, source_index)
    target = get_target_object(category, target_index)
    pair_id = (
        f"source_{source.category}_{source.source_object_id}"
        f"__target_{target_index:02d}_{target.target_id}"
    )
    return (
        root
        / "dexlink"
        / "src"
        / "reference"
        / "common"
        / "categories"
        / "transforms"
        / category
        / pair_id
    )


def category_retarget_run_dir(
    root: Path,
    hand_type: str,
    category: str,
    source_index: int,
    target_index: int,
    articulation_sample_id: str | None = None,
) -> Path:
    return (
        get_logs_dir()
        / "experiment_category_retargetings"
        / normalize_retarget_hand_type(hand_type)
        / category
        / category_retarget_job_id(
            category,
            source_index,
            target_index,
            articulation_sample_id,
        )
    )


def category_prepared_target_dir(
    root: Path,
    category: str,
    source_index: int,
    target_index: int,
) -> Path:
    """Return the generated-data location for URDF-derived target artifacts."""

    transform_dir = category_transform_dir(
        root, category, source_index, target_index
    )
    return get_logs_dir() / "prepared_targets" / category / transform_dir.name
