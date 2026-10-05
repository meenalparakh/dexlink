#!/usr/bin/env python3
"""Launch one seed-42 OakInk grasp-stability refinement from its manifest rank."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from dexlink.scripts.rl.train_rl import main as train_rl_main

PROJECT = "v3_template_diversity_grasp_refinement"
FULL_MANIFEST_SCHEMA = "oakink_grasp_stability_refinement_v1"
SUBSET_MANIFEST_SCHEMA = "oakink_grasp_stability_refinement_subset_v1"


@dataclass(frozen=True)
class RefinementProtocol:
    preset: str
    group: str
    run_name_prefix: str
    train_rollout_steps: int
    resumable: bool


PROTOCOLS = {
    "random_force_v1": RefinementProtocol(
        preset="seq_oakink_grasp_stability_refinement",
        group="full_object_aware_seed42",
        run_name_prefix="grasp_refine",
        train_rollout_steps=7_000,
        resumable=False,
    ),
    "random_force_2k_no_closeness_v1": RefinementProtocol(
        preset="seq_oakink_grasp_stability_refinement_random_force_2k",
        group="full_object_aware_random_force_2k_subset100_seed42",
        run_name_prefix="grasp_refine_random_force_2k",
        train_rollout_steps=2_000,
        resumable=True,
    ),
    "gravity_terminal_v1": RefinementProtocol(
        preset="seq_oakink_grasp_stability_refinement_gravity_terminal",
        group="full_object_aware_gravity_terminal_seed42",
        run_name_prefix="grasp_refine_gravity_terminal",
        train_rollout_steps=5_000,
        resumable=True,
    ),
}


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not result:
        raise ValueError(f"Cannot form a run-name component from {value!r}.")
    return result


def load_record(manifest_path: Path, rank: int) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version not in {FULL_MANIFEST_SCHEMA, SUBSET_MANIFEST_SCHEMA}:
        raise ValueError(
            f"Unexpected grasp-refinement manifest schema in {manifest_path}: "
            f"{schema_version!r}."
        )
    if manifest.get("objective_id") != "full_object_aware":
        raise ValueError(
            "Grasp refinement is restricted to the full object-aware objective; "
            f"got {manifest.get('objective_id')!r}."
        )
    if int(manifest.get("seed", -1)) != 42:
        raise ValueError(f"Expected manifest seed 42, got {manifest.get('seed')!r}.")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(
            f"Expected refinement records to be a list, got {type(records).__name__}."
        )
    if schema_version == FULL_MANIFEST_SCHEMA:
        expected_record_count = 287
    else:
        expected_record_count = int(manifest.get("subset_size", -1))
        if expected_record_count <= 0:
            raise ValueError(
                f"Expected a positive subset_size in {manifest_path}, got "
                f"{manifest.get('subset_size')!r}."
            )
    if len(records) != expected_record_count:
        raise ValueError(
            f"Expected {expected_record_count} refinement records for schema "
            f"{schema_version!r}, got {len(records)}."
        )
    if rank < 0 or rank >= len(records):
        raise IndexError(f"Rank {rank} is outside [0, {len(records)}).")
    record = records[rank]
    if not isinstance(record, dict) or int(record.get("rank", -1)) != rank:
        raise ValueError(f"Manifest record at rank {rank} is malformed: {record!r}.")
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=tuple(PROTOCOLS),
        default="random_force_v1",
    )
    raw = parser.parse_args(argv)

    manifest_path = raw.manifest.resolve()
    record = load_record(manifest_path, raw.rank)
    protocol = PROTOCOLS[raw.protocol]
    reference_path = Path(str(record["reference_path"])).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Missing grasp-refinement reference for rank {raw.rank}: "
            f"{reference_path}."
        )
    run_name = (
        f"{protocol.run_name_prefix}__{int(record['index']):03d}__"
        f"{_safe_name(str(record['category']))}__"
        f"{_safe_name(str(record['template_id']))}__seed42"
    )
    train_args = [
        "--preset",
        protocol.preset,
        "--env.task_cfg.oakink_exported_path",
        str(reference_path),
        "--env.seed",
        "42",
        "--agent.timesteps",
        str(protocol.train_rollout_steps),
        "--logger.logdir",
        str(raw.log_root.resolve()),
        "--logger.project",
        PROJECT,
        "--logger.group",
        protocol.group,
        "--logger.runname",
        run_name,
        "--logger.timestamp",
        "False",
        "--logger.seed",
        "False",
        "--logger.disable",
        "False",
    ]
    if protocol.resumable:
        train_args.extend(
            [
                "--agent.resume",
                "True",
                "--agent.resume_latest_fixed_budget",
                "True",
            ]
        )
    train_rl_main(train_args)


if __name__ == "__main__":
    main()
