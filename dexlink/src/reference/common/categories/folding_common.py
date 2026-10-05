from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from typing import Literal

from dexlink.src.reference.common.exp_schema import (
    ArticulationRetargetSample,
    TargetObject,
)
from dexlink.utils.retargeting.path_utils import get_src

REPO_ROOT = get_src()


@dataclass(frozen=True)
class FoldingStage:
    """One authored folding waypoint and the cardboard links grasped there."""

    stage_id: str
    label: str
    trajectory_stage: str
    left_links: tuple[str, ...]
    right_links: tuple[str, ...]
    sample_at: Literal["start", "end"] = "end"

    def __post_init__(self) -> None:
        if not self.stage_id or not self.label or not self.trajectory_stage:
            raise ValueError("Folding stages require non-empty ids and labels.")
        for side, links in (("left", self.left_links), ("right", self.right_links)):
            if len(links) not in (1, 2):
                raise ValueError(
                    f"Stage {self.stage_id!r} {side} hand must target one or two "
                    f"cardboard links, received {links}."
                )
            if len(set(links)) != len(links):
                raise ValueError(
                    f"Stage {self.stage_id!r} {side} links are not unique: {links}."
                )


@dataclass(frozen=True)
class FoldingStageSample:
    definition: FoldingStage
    joint_values: dict[str, float]
    source_duration_seconds: float


def load_folding_stage_samples(
    *,
    trajectory_path: str,
    urdf_path: str,
    stages: tuple[FoldingStage, ...],
) -> tuple[FoldingStageSample, ...]:
    """Load exact stage-boundary joint values from an asset's reference motion."""

    trajectory_file = REPO_ROOT / trajectory_path
    urdf_file = REPO_ROOT / urdf_path
    trajectory = json.loads(trajectory_file.read_text(encoding="utf-8"))
    joint_names = trajectory.get("joint_names")
    times = trajectory.get("times")
    positions = trajectory.get("positions")
    stage_labels = trajectory.get("stage", trajectory.get("stages"))

    if not isinstance(joint_names, list) or not all(
        isinstance(name, str) and name for name in joint_names
    ):
        raise ValueError(f"Expected string joint_names in {trajectory_file}.")
    if not isinstance(times, list) or not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in times
    ):
        raise ValueError(f"Expected finite numeric times in {trajectory_file}.")
    if (
        not isinstance(positions, list)
        or len(positions) != len(times)
        or not all(
            isinstance(row, list)
            and len(row) == len(joint_names)
            and all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in row
            )
            for row in positions
        )
    ):
        raise ValueError(
            f"Expected {len(times)} position rows with {len(joint_names)} finite "
            f"values each in {trajectory_file}."
        )
    if not isinstance(stage_labels, list) or len(stage_labels) != len(times):
        raise ValueError(
            f"Expected one stage label per trajectory sample in {trajectory_file}."
        )

    urdf_root = ElementTree.parse(urdf_file).getroot()
    urdf_joint_names = [
        str(joint.attrib["name"])
        for joint in urdf_root.findall("joint")
        if joint.attrib.get("type") != "fixed"
    ]
    if joint_names != urdf_joint_names:
        raise ValueError(
            f"Trajectory joints do not match {urdf_file}: trajectory={joint_names}, "
            f"urdf={urdf_joint_names}."
        )
    available_links = {str(link.attrib["name"]) for link in urdf_root.findall("link")}

    samples: list[FoldingStageSample] = []
    for stage in stages:
        selected_links = set(stage.left_links) | set(stage.right_links)
        missing_links = sorted(selected_links - available_links)
        if missing_links:
            raise ValueError(
                f"Stage {stage.stage_id!r} selects missing links {missing_links} from "
                f"{urdf_file}. Available links: {sorted(available_links)}."
            )
        indices = [
            index
            for index, label in enumerate(stage_labels)
            if label == stage.trajectory_stage
        ]
        if not indices:
            raise ValueError(
                f"Stage {stage.stage_id!r} references unknown trajectory stage "
                f"{stage.trajectory_stage!r} in {trajectory_file}."
            )
        expected_indices = list(range(indices[0], indices[-1] + 1))
        if indices != expected_indices:
            raise ValueError(
                f"Trajectory stage {stage.trajectory_stage!r} is not contiguous in "
                f"{trajectory_file}."
            )
        # Labels switch on exact motion boundaries. A stage therefore starts at
        # its first labeled sample and ends at the first sample of the next run.
        start_index = indices[0]
        end_index = indices[-1] + 1
        if end_index >= len(times):
            raise ValueError(
                f"Trajectory stage {stage.trajectory_stage!r} has no following "
                f"boundary sample in {trajectory_file}."
            )
        sample_index = start_index if stage.sample_at == "start" else end_index
        duration = float(times[end_index] - times[start_index])
        samples.append(
            FoldingStageSample(
                definition=stage,
                joint_values={
                    name: float(value)
                    for name, value in zip(
                        joint_names, positions[sample_index], strict=True
                    )
                },
                source_duration_seconds=duration,
            )
        )
    return tuple(samples)


def build_selective_link_targets(
    *,
    category: str,
    asset_id: str,
    urdf_path: str,
    samples: tuple[FoldingStageSample, ...],
) -> tuple[dict[int, TargetObject], dict[tuple[str, str], int]]:
    """Create one retargeting target for each stage/hand selected-link set."""

    targets: dict[int, TargetObject] = {}
    target_index_by_stage_side: dict[tuple[str, str], int] = {}
    for sample in samples:
        for side, link_names in (
            ("left", sample.definition.left_links),
            ("right", sample.definition.right_links),
        ):
            target_index = len(targets) + 1
            target_id = f"{asset_id}_{sample.definition.stage_id}_{side}"
            targets[target_index] = TargetObject(
                category=category,
                target_id=target_id,
                mesh_path=urdf_path,
                scale=1.0,
                bps_sampling_link_names=link_names,
                articulation_retarget_mode="per_articulation_retarget",
                articulation_retarget_samples=(
                    ArticulationRetargetSample(
                        sample_id=sample.definition.stage_id,
                        joint_values=sample.joint_values,
                    ),
                ),
                notes=(
                    f"{sample.definition.label}: {side} xHand holding retarget over "
                    f"only cardboard link(s) {link_names}."
                ),
            )
            target_index_by_stage_side[(sample.definition.stage_id, side)] = (
                target_index
            )
    return targets, target_index_by_stage_side
