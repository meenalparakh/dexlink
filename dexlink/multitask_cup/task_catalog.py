"""Physical-geometry groups and balanced GPU layouts for the approved source cohort."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def validate_training_catalog(manifest: dict) -> int:
    """Validate explicit cohort authorization without treating generated previews as approval."""
    variants = manifest["variants"]
    count = manifest["case_count"]
    if type(count) is not int or count < 1 or len(variants) != count:
        raise ValueError(
            f"Expected case_count matching a nonempty catalog, received {count} / {len(variants)}."
        )
    if len({row["id"] for row in variants}) != count:
        raise ValueError("Training catalog task IDs must be unique.")
    if manifest["schema"] != "dexlink_multitask_v1":
        raise ValueError(
            f"Expected dexlink_multitask_v1, received {manifest['schema']!r}."
        )
    for variant in variants:
        if variant["axis"] not in ("G", "Q", "T", "S", "P") or variant["frames"] < 1:
            raise ValueError(
                "Expected a declared cup task axis and positive reference frame count."
            )
    return count


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rigid_asset_signature(path: Path) -> str:
    """Ignore descriptive robot names/paths while retaining every physical XML field."""
    root = ET.parse(path).getroot()
    root.attrib.pop("name", None)
    for node in root.findall(".//mesh"):
        mesh = (path.parent / node.attrib["filename"]).resolve(strict=True)
        node.attrib["filename"] = "sha256:" + file_sha256(mesh)
    return hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest()


@dataclass(frozen=True)
class GeometryGroup:
    geometry_index: int
    physical_signature: str
    task_indices: tuple[int, ...]


@dataclass(frozen=True)
class GeometryShard:
    rank: int
    geometry_index: int
    task_indices: tuple[int, ...]


@dataclass(frozen=True)
class TaskLayout:
    world_size: int
    task_count: int
    groups: tuple[GeometryGroup, ...]
    shards: tuple[GeometryShard, ...]
    tasks_per_rank: tuple[int, ...]

    def rank_loss_weight(self, rank: int) -> float:
        return self.world_size * self.tasks_per_rank[rank] / self.task_count

    def to_dict(self) -> dict:
        return asdict(self)


def geometry_groups(variants: list[dict]) -> tuple[GeometryGroup, ...]:
    if not variants or len({row["id"] for row in variants}) != len(variants):
        raise ValueError("Expected a nonempty unique task catalog.")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(variants):
        if len(row["scene_urdfs"]) != 2:
            raise ValueError(f"Expected cup/table URDFs for {row['id']}.")
        contract = {
            "actors": [
                rigid_asset_signature(Path(path)) for path in row["scene_urdfs"]
            ],
            "extent_xyz": row["geometry_extent_xyz"],
        }
        signature = hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest()
        groups.setdefault(signature, []).append(index)
    return tuple(
        GeometryGroup(index, signature, tuple(indices))
        for index, (signature, indices) in enumerate(groups.items())
    )


def divide_tasks(variants: list[dict], world_size: int) -> TaskLayout:
    if world_size not in (1, 4, 8):
        raise ValueError(f"Expected 1, 4 or 8 coordinated GPUs, received {world_size}.")
    if len(variants) < world_size:
        raise ValueError("Each learning rank must own at least one task.")
    groups = geometry_groups(variants)
    loads = [0] * world_size
    axes = [Counter() for _ in range(world_size)]
    assignments: dict[tuple[int, int], list[int]] = {}
    # Share one scene per geometry/rank. Split the large base-cup group for load balance.
    for group in sorted(
        groups, key=lambda group: (-len(group.task_indices), group.geometry_index)
    ):
        ordered = sorted(
            group.task_indices,
            key=lambda index: (variants[index]["axis"], variants[index]["id"]),
        )
        for index in ordered:
            axis = variants[index]["axis"]
            rank = min(
                range(world_size),
                key=lambda rank: (loads[rank], axes[rank][axis], rank),
            )
            assignments.setdefault((rank, group.geometry_index), []).append(index)
            loads[rank] += 1
            axes[rank][axis] += 1
    shards = tuple(
        GeometryShard(rank, geometry, tuple(tasks))
        for (rank, geometry), tasks in sorted(assignments.items())
    )
    assigned = [index for shard in shards for index in shard.task_indices]
    if sorted(assigned) != list(range(len(variants))) or max(loads) - min(loads) > 1:
        raise RuntimeError(
            "GPU division must cover every task exactly once with balanced task counts."
        )
    return TaskLayout(world_size, len(variants), groups, shards, tuple(loads))
