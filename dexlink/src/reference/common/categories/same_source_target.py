from __future__ import annotations

from pathlib import Path

from dexlink.src.reference.common.exp_schema import (
    DEMO_SUFFIX,
    SOURCE_DIR,
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)
from dexlink.utils.retargeting.path_utils import get_external_data_dir

OAKINK_SOURCE_ROOT = get_external_data_dir() / Path(SOURCE_DIR).relative_to(
    "external_data"
)

CATEGORY = ObjectCategory(
    category="same_source_target",
    display_name="same source target",
    sections=("I", "II", "III", "IV", "V"),
    required_data=("source_templates",),
    density_range=(700.0, 1400.0),
    notes=(
        "Retarget every OakInk source demo back to its own source object. "
        "No target mesh, target demo, transform, or scale is passed to trajopt."
    ),
)


def _iter_oakink_demo_paths() -> tuple[Path, ...]:
    if not OAKINK_SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"OakInk source root does not exist: {OAKINK_SOURCE_ROOT}"
        )
    return tuple(sorted(OAKINK_SOURCE_ROOT.glob(f"*/*/*/{DEMO_SUFFIX}")))


def _source_template(index: int, demo_path: Path) -> SourceTemplate:
    relative_parts = demo_path.relative_to(OAKINK_SOURCE_ROOT).parts
    if len(relative_parts) != 4:
        raise ValueError(
            "Expected OakInk demo path relative to source root to have form "
            f"category/instance/grasp_id/{DEMO_SUFFIX}, got {demo_path}."
        )
    category, source_object_id, demo_id, filename = relative_parts
    if filename != DEMO_SUFFIX:
        raise ValueError(
            f"Expected OakInk demo filename {DEMO_SUFFIX}, got {filename}."
        )

    template_id = f"{category}_{source_object_id}_{demo_id}"
    return SourceTemplate(
        category=category,
        template_id=template_id,
        source_object_id=source_object_id,
        demo_id=demo_id,
        behavior="same_source",
        phase="grasp",
        description=f"same-source retargeting for {template_id}",
    )


def _target_object(source: SourceTemplate) -> TargetObject:
    return TargetObject(
        category=CATEGORY.category,
        target_id=f"{source.category}_{source.source_object_id}_{source.demo_id}",
        mesh_path=None,
        scale=1.0,
        same_as_source=True,
    )


SOURCE_TEMPLATES = {
    index: _source_template(index, demo_path)
    for index, demo_path in enumerate(_iter_oakink_demo_paths(), start=1)
}

TARGET_OBJECTS = {
    index: _target_object(source) for index, source in SOURCE_TEMPLATES.items()
}
