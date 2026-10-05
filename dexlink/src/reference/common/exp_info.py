from __future__ import annotations

from importlib import import_module

from dexlink.src.reference.common.exp_schema import (
    DEMO_SUFFIX,
    SOURCE_DIR,
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)
CATEGORY_MODULES = (
    "banana_handover",
    "block",
    "bottle",
    "bowl",
    "bucket",
    "container",
    "cup",
    "dispenser",
    "donut",
    "drawer",
    "eraser",
    "faucet",
    "fruit",
    "folding_carton",
    "folding_mailer",
    "hammer",
    "headphones",
    "ketchup_seq",
    "lego_block",
    "marker",
    "mug",
    "mouse",
    "notebook",
    "scissor",
    "screwdriver",
    # "same_source_target",
    "spoon",
    "stapler",
    "table_setting_object",
    "template_transfer",
    "toothbrush",
    "top_lid",
    "trigger_sprayer",
    "v2_15_lift_asset",
    "wineglass",
    "ycb_cup",
    "ycb_pitcher",
    "ycb_rubiks_cube",
)


def _load_category_modules() -> tuple[object, ...]:
    return tuple(
        import_module(f"dexlink.src.reference.common.categories.{category}")
        for category in CATEGORY_MODULES
    )


_CATEGORY_MODULE_OBJECTS = _load_category_modules()

OBJECT_CATEGORIES: dict[str, ObjectCategory] = {
    module.CATEGORY.category: module.CATEGORY for module in _CATEGORY_MODULE_OBJECTS
}
OBJECT_CATEGORIES_BY_SECTION: dict[str, tuple[str, ...]] = {
    section: tuple(
        category
        for category, category_info in OBJECT_CATEGORIES.items()
        if section in category_info.sections
    )
    for section in ("I", "II", "III", "IV", "V")
}

SOURCE_TEMPLATES: dict[str, dict[int, SourceTemplate]] = {
    module.CATEGORY.category: module.SOURCE_TEMPLATES
    for module in _CATEGORY_MODULE_OBJECTS
}
TARGET_OBJECTS: dict[str, dict[int, TargetObject]] = {
    module.CATEGORY.category: module.TARGET_OBJECTS
    for module in _CATEGORY_MODULE_OBJECTS
}


def get_source_template(category: str, source_index: int) -> SourceTemplate:
    try:
        return SOURCE_TEMPLATES[category][source_index]
    except KeyError as exc:
        raise KeyError(
            f"Expected source template index {source_index!r} for category {category!r}."
        ) from exc


def get_target_object(category: str, target_index: int) -> TargetObject:
    try:
        return TARGET_OBJECTS[category][target_index]
    except KeyError as exc:
        raise KeyError(
            f"Expected target object index {target_index!r} for category {category!r}."
        ) from exc


def get_source_templates(category: str, behavior: str) -> tuple[SourceTemplate, ...]:
    if category not in SOURCE_TEMPLATES:
        raise KeyError(f"Expected source templates for category {category!r}.")

    templates = tuple(
        template
        for template in SOURCE_TEMPLATES[category].values()
        if template.behavior == behavior
    )
    if not templates:
        raise KeyError(
            f"Expected at least one {behavior!r} source template for category {category!r}."
        )
    return templates


def get_target_objects(category: str) -> tuple[TargetObject, ...]:
    if category not in TARGET_OBJECTS:
        raise KeyError(f"Expected target objects for category {category!r}.")

    targets = tuple(TARGET_OBJECTS[category].values())
    if not targets:
        raise KeyError(
            f"No target objects have been collected for category {category!r}."
        )
    return targets


# Compatibility aliases for existing ad-hoc spec drafting.
source_dir = f"{SOURCE_DIR}/"
demo_suffix = f"/{DEMO_SUFFIX}"

mug_source_templates = {
    template.template_id: template.demo_path
    for template in SOURCE_TEMPLATES["mug"].values()
}
mug_target_meshes = {
    index: target.mesh_path for index, target in TARGET_OBJECTS["mug"].items()
}
mug_target_transform_info = {
    index: target.transform_path for index, target in TARGET_OBJECTS["mug"].items()
}
mug_target_scales = {
    index: target.scale for index, target in TARGET_OBJECTS["mug"].items()
}

bowl_source_templates = {
    template.template_id: template.demo_path
    for template in SOURCE_TEMPLATES["bowl"].values()
}
bowl_target_meshes: dict[int, str] = {}
bowl_target_transform_info: dict[int, str | None] = {}
bowl_target_scales: dict[int, float] = {}
