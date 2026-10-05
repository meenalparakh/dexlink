from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SOURCE_DIR = "external_data/oakink/shape/oakink_shape_v2"
DEMO_SUFFIX = "hand_param.pkl"
SHAPENET_DIR = "external_data/shapenet"
SHAPENET_TRANSFORM_DIR = "external_data/shapenet/transforms"

ExperimentScale = Literal["minimum", "ideal"]
SourceDataset = Literal["oakink", "arctic"]
ArticulationRetargetMode = Literal["link_follow", "per_articulation_retarget"]


@dataclass(frozen=True)
class SourceTemplate:
    category: str
    template_id: str
    source_object_id: str
    demo_id: str
    behavior: str
    phase: str
    description: str
    dataset: SourceDataset = "oakink"
    demo_path_override: str | None = None
    arctic_target_object: str | None = None
    arctic_timestep: int | None = None
    arctic_start_timestep: int | None = None
    arctic_end_timestep: int | None = None
    arctic_skip: int = 1
    arctic_hand: str | None = None

    def __post_init__(self) -> None:
        if self.dataset == "oakink":
            if self.demo_path_override is not None:
                raise ValueError(
                    f"OakInk SourceTemplate {self.template_id} should not define "
                    "`demo_path_override`; use category/source_object_id/demo_id."
                )
            return

        if self.dataset != "arctic":
            raise ValueError(
                f"Unsupported SourceTemplate dataset {self.dataset!r}; expected "
                "'oakink' or 'arctic'."
            )
        if self.demo_path_override is None:
            raise ValueError(
                f"ARCTIC SourceTemplate {self.template_id} must define "
                "`demo_path_override`."
            )
        if self.arctic_target_object is None:
            raise ValueError(
                f"ARCTIC SourceTemplate {self.template_id} must define "
                "`arctic_target_object`."
            )
        if self.arctic_timestep is None:
            raise ValueError(
                f"ARCTIC SourceTemplate {self.template_id} must define "
                "`arctic_timestep`."
            )
        if self.arctic_start_timestep is None or self.arctic_end_timestep is None:
            raise ValueError(
                f"ARCTIC SourceTemplate {self.template_id} must define "
                "`arctic_start_timestep` and `arctic_end_timestep`."
            )
        if self.arctic_skip <= 0:
            raise ValueError(
                f"ARCTIC SourceTemplate {self.template_id} must have positive "
                f"`arctic_skip`, got {self.arctic_skip}."
            )
        if self.arctic_hand is not None and self.arctic_hand not in {"left", "right"}:
            raise ValueError(
                f"ARCTIC SourceTemplate {self.template_id} arctic_hand must be "
                f"'left' or 'right', got {self.arctic_hand!r}."
            )
        if not (
            self.arctic_start_timestep
            <= self.arctic_timestep
            < self.arctic_end_timestep
        ):
            raise ValueError(
                f"ARCTIC SourceTemplate {self.template_id} timestep "
                f"{self.arctic_timestep} is outside "
                f"[{self.arctic_start_timestep}, {self.arctic_end_timestep})."
            )
        if (self.arctic_timestep - self.arctic_start_timestep) % self.arctic_skip != 0:
            raise ValueError(
                f"ARCTIC SourceTemplate {self.template_id} timestep "
                f"{self.arctic_timestep} is not aligned with start "
                f"{self.arctic_start_timestep} and skip {self.arctic_skip}."
            )

    @property
    def demo_path(self) -> str:
        if self.demo_path_override is not None:
            return self.demo_path_override
        return (
            f"{SOURCE_DIR}/{self.category}/{self.source_object_id}/"
            f"{self.demo_id}/{DEMO_SUFFIX}"
        )


@dataclass(frozen=True)
class ArticulationRetargetSample:
    sample_id: str
    joint_values: dict[str, float]

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("ArticulationRetargetSample must define `sample_id`.")
        if not self.joint_values:
            raise ValueError(
                f"ArticulationRetargetSample {self.sample_id!r} must define "
                "`joint_values`."
            )


@dataclass(frozen=True)
class TargetObject:
    category: str
    target_id: str
    mesh_path: str | None
    scale: float
    transform_path: str | None = None
    notes: str = ""
    target_demo_path: str | None = None
    bps_sampling_link_names: tuple[str, ...] = ()
    same_as_source: bool = False
    articulation_retarget_mode: ArticulationRetargetMode = "link_follow"
    articulation_retarget_samples: tuple[ArticulationRetargetSample, ...] = ()

    def __post_init__(self) -> None:
        if self.articulation_retarget_mode == "per_articulation_retarget":
            if not self.articulation_retarget_samples:
                raise ValueError(
                    f"TargetObject {self.category}/{self.target_id} uses "
                    "`per_articulation_retarget` but defines no samples."
                )
        elif self.articulation_retarget_mode != "link_follow":
            raise ValueError(
                f"Unsupported articulation_retarget_mode "
                f"{self.articulation_retarget_mode!r}."
            )

        if self.same_as_source:
            if self.mesh_path is not None:
                raise ValueError(
                    f"Same-source TargetObject {self.category}/{self.target_id} "
                    "must not define `mesh_path`."
                )
            if self.target_demo_path is not None:
                raise ValueError(
                    f"Same-source TargetObject {self.category}/{self.target_id} "
                    "must not define `target_demo_path`."
                )
            if self.transform_path is not None:
                raise ValueError(
                    f"Same-source TargetObject {self.category}/{self.target_id} "
                    "must not define `transform_path`."
                )
            if self.scale != 1.0:
                raise ValueError(
                    f"Same-source TargetObject {self.category}/{self.target_id} "
                    f"must use scale 1.0, got {self.scale}."
                )
            if self.bps_sampling_link_names:
                raise ValueError(
                    f"Same-source TargetObject {self.category}/{self.target_id} "
                    "must not define `bps_sampling_link_names`."
                )
            if self.articulation_retarget_samples:
                raise ValueError(
                    f"Same-source TargetObject {self.category}/{self.target_id} "
                    "must not define `articulation_retarget_samples`."
                )
            return

        if self.mesh_path is None and self.target_demo_path is None:
            raise ValueError(
                f"TargetObject {self.category}/{self.target_id} must define "
                "`mesh_path` or `target_demo_path`."
            )


@dataclass(frozen=True)
class ObjectCategory:
    category: str
    display_name: str
    sections: tuple[str, ...]
    required_data: tuple[str, ...]
    density_range: tuple[float, float]
    notes: str

    def __post_init__(self) -> None:
        if len(self.density_range) != 2:
            raise ValueError(
                f"ObjectCategory {self.category} density_range must have two values, "
                f"got {self.density_range}."
            )
        density_min, density_max = self.density_range
        if density_min <= 0.0 or density_max <= 0.0 or density_min > density_max:
            raise ValueError(
                f"ObjectCategory {self.category} density_range must be positive "
                f"and ordered, got {self.density_range}."
            )


@dataclass(frozen=True)
class SectionCollectionRequirement:
    section: str
    purpose: str
    task_families: tuple[str, ...]
    categories: tuple[str, ...]
    source_templates_per_category: dict[str, int]
    target_instances_per_category: int
    notes: tuple[str, ...]


def shapenet_mesh(category: str, instance_id: str) -> str:
    return f"{SHAPENET_DIR}/{category}/{instance_id}/models/model_normalized_solid_mesh.ply"


def shapenet_transform(category: str, instance_id: str) -> str:
    return f"{SHAPENET_TRANSFORM_DIR}/{category}/{instance_id}.rotation.json"


def oakink_demo_path(category: str, object_id: str, demo_id: str) -> str:
    return f"{SOURCE_DIR}/{category}/{object_id}/{demo_id}/{DEMO_SUFFIX}"
