"""Category target selection for mesh validation and preparation."""

from __future__ import annotations


import argparse


import xml.etree.ElementTree as ET


from dataclasses import dataclass


from pathlib import Path


from dexlink.utils.retargeting.path_utils import get_src, resolve_project_path


from dexlink.src.reference.common.exp_info import (
    OBJECT_CATEGORIES,
    SOURCE_TEMPLATES,
    TARGET_OBJECTS,
)


from dexlink.src.reference.common.exp_schema import (
    ArticulationRetargetSample,
    SourceTemplate,
    TargetObject,
)


PROJECT_ROOT = get_src()


TRANSFORM_ROOT = PROJECT_ROOT / "dexlink/src/reference/common/categories/transforms"


@dataclass(frozen=True)
class CategoryRetargetJob:
    category: str
    source_index: int
    source: SourceTemplate
    target_index: int
    target: TargetObject
    articulation_sample: ArticulationRetargetSample | None = None

    @property
    def source_object_key(self) -> str:
        return f"{self.source.category}_{self.source.source_object_id}"

    @property
    def transform_pair_id(self) -> str:
        source_id = f"source_{self.source_object_key}"
        target_id = f"target_{self.target_index:02d}_{self.target.target_id}"
        return f"{source_id}__{target_id}"

    @property
    def job_id(self) -> str:
        job_id = (
            f"source_{self.source_index:02d}_{self.source.template_id}"
            f"__target_{self.target_index:02d}_{self.target.target_id}"
        )
        if self.articulation_sample is not None:
            job_id += f"__articulation_{self.articulation_sample.sample_id}"
        return job_id


def _relative_or_abs_path(path: str | Path) -> Path:
    return resolve_project_path(path)


def _transform_dir(job: CategoryRetargetJob) -> Path:
    return TRANSFORM_ROOT / job.category / job.transform_pair_id


def _rotation_path(job: CategoryRetargetJob) -> Path:
    return _transform_dir(job) / "rotation.json"


def _scale_path(job: CategoryRetargetJob) -> Path:
    return _transform_dir(job) / "scale.txt"


def _urdf_visualization_obj_path(job: CategoryRetargetJob) -> Path:
    suffix = _articulation_sample_suffix(job, default_suffix="_default")
    return _transform_dir(job) / f"{job.target.target_id}{suffix}_visual.obj"


def _urdf_bps_sampling_obj_path(job: CategoryRetargetJob) -> Path:
    suffix = _articulation_sample_suffix(job, default_suffix="")
    return _transform_dir(job) / f"{job.target.target_id}{suffix}_bps_sampling.obj"


def _urdfpy_compatible_urdf_path(job: CategoryRetargetJob) -> Path:
    return _transform_dir(job) / f"{job.target.target_id}_urdfpy.urdf"


def _is_unresolved(value: str | None) -> bool:
    return value is None or str(value).strip() in {"", "TBD", "None"}


def _articulation_sample_suffix(
    job: CategoryRetargetJob, *, default_suffix: str
) -> str:
    if job.articulation_sample is None:
        return default_suffix
    return f"_{job.articulation_sample.sample_id}"


def _articulation_joint_cfg(job: CategoryRetargetJob) -> dict[str, float]:
    if job.articulation_sample is None:
        return {}
    return {
        str(joint_name): float(value)
        for joint_name, value in job.articulation_sample.joint_values.items()
    }


def _resolve_rotation_path(job: CategoryRetargetJob) -> Path | None:
    rotation_path = _rotation_path(job)
    if rotation_path.exists():
        return rotation_path
    if not _is_unresolved(job.target.transform_path):
        fallback_path = _relative_or_abs_path(str(job.target.transform_path))
        if fallback_path.exists():
            return fallback_path
        raise FileNotFoundError(
            f"Target transform path does not exist for {job.job_id}: {fallback_path}"
        )
    return None


def _resolve_scale(job: CategoryRetargetJob) -> float:
    scale_path = _scale_path(job)
    if scale_path.exists():
        return float(scale_path.read_text(encoding="utf-8").strip())
    return float(job.target.scale)


def _materialize_urdfpy_compatible_urdf(
    job: CategoryRetargetJob,
    target_path: Path,
) -> Path:
    if target_path.suffix.lower() != ".urdf":
        raise ValueError(f"Expected a .urdf target path, got {target_path}.")
    if not target_path.exists():
        raise FileNotFoundError(f"URDF file not found: {target_path}")

    tree = ET.parse(target_path)
    root = tree.getroot()
    changed = False
    for joint in root.findall(".//joint"):
        if joint.attrib.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        if "effort" not in limit.attrib:
            limit.set("effort", "1.0")
            changed = True
        if "velocity" not in limit.attrib:
            limit.set("velocity", "1.0")
            changed = True
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename is None:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh.set("filename", str((target_path.parent / mesh_path).resolve()))
            changed = True

    if not changed:
        return target_path

    output_path = _urdfpy_compatible_urdf_path(job)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _materialize_urdf_visualization_obj(
    job: CategoryRetargetJob,
    target_path: Path,
) -> None:
    if target_path.suffix.lower() != ".urdf":
        raise ValueError(f"Expected a .urdf target path, got {target_path}.")
    visualization_obj_path = _urdf_visualization_obj_path(job)
    visualization_obj_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_path = _materialize_urdfpy_compatible_urdf(job, target_path)
    mesh = _load_urdf_combined_mesh(urdf_path, joint_cfg=_articulation_joint_cfg(job))
    mesh.export(visualization_obj_path)


def _materialize_urdf_bps_sampling_obj(
    job: CategoryRetargetJob,
    target_path: Path,
) -> None:
    if target_path.suffix.lower() != ".urdf":
        raise ValueError(f"Expected a .urdf target path, got {target_path}.")
    if not job.target.bps_sampling_link_names:
        raise ValueError(
            f"{job.job_id} does not define `bps_sampling_link_names`; "
            "cannot materialize a selected-link BPS sampling OBJ."
        )
    sampling_obj_path = _urdf_bps_sampling_obj_path(job)
    sampling_obj_path.parent.mkdir(parents=True, exist_ok=True)
    urdf_path = _materialize_urdfpy_compatible_urdf(job, target_path)
    mesh = _load_urdf_combined_mesh(
        urdf_path,
        job.target.bps_sampling_link_names,
        joint_cfg=_articulation_joint_cfg(job),
    )
    mesh.export(sampling_obj_path)


def _load_urdf_combined_mesh(
    urdf_path: Path,
    link_names: tuple[str, ...] = (),
    joint_cfg: dict[str, float] | None = None,
):
    import trimesh
    from urdfpy import URDF

    print(f"obtaining obj from urdf {urdf_path}")
    urdf_inst = URDF.load(str(urdf_path))
    if link_names:
        requested_links = set(link_names)
        available_links = {link.name for link in urdf_inst.links}
        missing_links = sorted(requested_links - available_links)
        if missing_links:
            raise ValueError(
                f"URDF {urdf_path} is missing requested BPS sampling links: "
                f"{missing_links}. Available links: {sorted(available_links)}."
            )
    cfg = {} if joint_cfg is None else dict(joint_cfg)
    if cfg:
        available_joints = {joint.name for joint in urdf_inst.actuated_joints}
        unknown_joints = sorted(set(cfg) - available_joints)
        if unknown_joints:
            raise ValueError(
                f"URDF {urdf_path} is missing requested articulation joints: "
                f"{unknown_joints}. Available actuated joints: {sorted(available_joints)}."
            )
    if len(link_names) == 0:
        mesh_fk = urdf_inst.visual_trimesh_fk(cfg=cfg, links=None)
    else:
        mesh_fk = urdf_inst.visual_trimesh_fk(cfg=cfg, links=list(link_names))
    pieces = []
    for mesh, pose in mesh_fk.items():
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(pose)
        pieces.append(transformed_mesh)
    if not pieces:
        link_msg = f" for links {list(link_names)}" if link_names else ""
        raise ValueError(f"No visual meshes found in URDF{link_msg}: {urdf_path}")
    return trimesh.util.concatenate(pieces)


def _resolve_target_obj_path(job: CategoryRetargetJob) -> Path | None:
    if job.target.same_as_source:
        return None
    if _is_unresolved(job.target.mesh_path):
        return None

    target_path = _relative_or_abs_path(str(job.target.mesh_path))
    if not target_path.exists():
        raise FileNotFoundError(f"Target object path does not exist: {target_path}")

    if target_path.suffix.lower() != ".urdf":
        return target_path

    visualization_obj_path = _urdf_visualization_obj_path(job)
    if visualization_obj_path.exists() and job.articulation_sample is None:
        return visualization_obj_path

    _materialize_urdf_visualization_obj(job, target_path)
    if not visualization_obj_path.exists():
        raise FileNotFoundError(
            f"Expected URDF visualization OBJ to be written to {visualization_obj_path}"
        )
    return visualization_obj_path


def _resolve_target_bps_obj_path(job: CategoryRetargetJob) -> Path | None:
    if job.target.same_as_source:
        return None
    if not job.target.bps_sampling_link_names:
        return None
    if _is_unresolved(job.target.mesh_path):
        raise ValueError(
            f"{job.job_id} defines `bps_sampling_link_names`, but has no resolved mesh_path."
        )

    target_path = _relative_or_abs_path(str(job.target.mesh_path))
    if not target_path.exists():
        raise FileNotFoundError(f"Target object path does not exist: {target_path}")
    if target_path.suffix.lower() != ".urdf":
        raise ValueError(
            f"{job.job_id} defines `bps_sampling_link_names`, but selective link "
            f"sampling requires a URDF target. Got: {target_path}"
        )

    sampling_obj_path = _urdf_bps_sampling_obj_path(job)
    _materialize_urdf_bps_sampling_obj(job, target_path)
    if not sampling_obj_path.exists():
        raise FileNotFoundError(
            f"Expected selected-link BPS OBJ to be written to {sampling_obj_path}"
        )
    return sampling_obj_path


def _resolve_target_demo_path(job: CategoryRetargetJob) -> Path | None:
    if job.target.same_as_source:
        return None
    if _is_unresolved(job.target.target_demo_path):
        return None
    target_demo_path = _relative_or_abs_path(str(job.target.target_demo_path))
    if not target_demo_path.exists():
        raise FileNotFoundError(f"Target demo path does not exist: {target_demo_path}")
    return target_demo_path


def _validate_job_inputs(
    job: CategoryRetargetJob,
    require_transform: bool,
) -> tuple[Path | None, Path | None, Path | None, Path | None, float]:
    source_demo_path = _relative_or_abs_path(job.source.demo_path)
    if not source_demo_path.exists():
        raise FileNotFoundError(f"Source demo path does not exist: {source_demo_path}")

    if job.source.dataset == "arctic":
        if job.source.arctic_target_object != job.target.target_id:
            raise ValueError(
                f"ARCTIC source {job.job_id} targets "
                f"{job.source.arctic_target_object!r}, but category target "
                f"{job.target_index} is {job.target.target_id!r}."
            )
    if job.articulation_sample is not None:
        if _is_unresolved(job.target.mesh_path):
            raise ValueError(
                f"{job.job_id} uses an articulation sample but has no target URDF."
            )
        target_mesh_path = _relative_or_abs_path(str(job.target.mesh_path))
        if target_mesh_path.suffix.lower() != ".urdf":
            raise ValueError(
                f"{job.job_id} uses an articulation sample, which requires a URDF "
                f"target. Got: {target_mesh_path}"
            )

    target_obj_path = _resolve_target_obj_path(job)
    target_bps_obj_path = _resolve_target_bps_obj_path(job)
    target_demo_path = _resolve_target_demo_path(job)
    if (
        target_obj_path is None
        and target_demo_path is None
        and not job.target.same_as_source
    ):
        raise ValueError(
            f"{job.job_id} must define a resolved target mesh path or target demo path."
        )

    target_transform_info = None
    if not job.target.same_as_source:
        target_transform_info = _resolve_rotation_path(job)
    if (
        require_transform
        and target_transform_info is None
        and not job.target.same_as_source
    ):
        raise FileNotFoundError(
            f"Missing rotation transform for {job.category}/{job.transform_pair_id}: "
            f"{_rotation_path(job)}"
        )
    target_scale = _resolve_scale(job)
    if target_scale <= 0.0:
        raise ValueError(
            f"Target scale must be positive for {job.job_id}: {target_scale}"
        )
    return (
        target_obj_path,
        target_bps_obj_path,
        target_demo_path,
        target_transform_info,
        target_scale,
    )


def _iter_jobs(args: argparse.Namespace) -> tuple[CategoryRetargetJob, ...]:
    selected_categories = args.category or sorted(SOURCE_TEMPLATES)
    jobs: list[CategoryRetargetJob] = []
    selected_sections = set(args.section)
    selected_articulation_samples = set(args.articulation_sample_id)
    for category in selected_categories:
        if category not in SOURCE_TEMPLATES:
            raise KeyError(f"Unknown source category {category!r}.")
        if category not in TARGET_OBJECTS:
            raise KeyError(f"Unknown target category {category!r}.")
        if selected_sections:
            sections = set(OBJECT_CATEGORIES[category].sections)
            if sections.isdisjoint(selected_sections):
                continue

        sources = SOURCE_TEMPLATES[category]
        targets = TARGET_OBJECTS[category]
        if args.source_index is not None:
            if args.source_index not in sources:
                raise KeyError(
                    f"Expected source index {args.source_index} in category {category!r}."
                )
            sources = {args.source_index: sources[args.source_index]}
        if args.target_index is not None:
            if args.target_index not in targets:
                raise KeyError(
                    f"Expected target index {args.target_index} in category {category!r}."
                )
            targets = {args.target_index: targets[args.target_index]}

        has_same_source_targets = any(
            target.same_as_source for target in targets.values()
        )
        has_regular_targets = any(
            not target.same_as_source for target in targets.values()
        )
        if has_same_source_targets and has_regular_targets:
            raise ValueError(
                f"Category {category!r} mixes same-source and regular targets; "
                "split them into separate categories."
            )

        for source_index, source in sources.items():
            for target_index, target in targets.items():
                if target.same_as_source and source_index != target_index:
                    continue
                if not args.only_articulation_samples:
                    jobs.append(
                        CategoryRetargetJob(
                            category=category,
                            source_index=source_index,
                            source=source,
                            target_index=target_index,
                            target=target,
                        )
                    )
                if args.include_articulation_samples or args.only_articulation_samples:
                    for sample in target.articulation_retarget_samples:
                        if (
                            selected_articulation_samples
                            and sample.sample_id not in selected_articulation_samples
                        ):
                            continue
                        jobs.append(
                            CategoryRetargetJob(
                                category=category,
                                source_index=source_index,
                                source=source,
                                target_index=target_index,
                                target=target,
                                articulation_sample=sample,
                            )
                        )
    return tuple(jobs)
