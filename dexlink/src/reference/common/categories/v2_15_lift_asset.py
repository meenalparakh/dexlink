from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY_ID = "v2_15_lift_asset"
MANIFEST_PATH = Path(__file__).with_name("v2_15_lift_asset_manifest.json")

CATEGORY = ObjectCategory(
    category=CATEGORY_ID,
    display_name="V2_15 lift asset",
    sections=("V2_15",),
    required_data=("source_templates", "target_meshes", "target_transforms"),
    density_range=(500.0, 2500.0),
    notes=(
        "Manifest-backed object category for V2_15 lift and task diversity. "
        "Targets must be visually aligned to their source templates before "
        "retargeting."
    ),
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="apple",
        template_id="top_round",
        source_object_id="C90001",
        demo_id="4f267b3fba",
        behavior="lift",
        phase="grasp",
        description="Top grasp for round or compact objects.",
    ),
    2: SourceTemplate(
        category="cylinder_bottle",
        template_id="side_cylinder",
        source_object_id="A15027",
        demo_id="107c064718",
        behavior="lift",
        phase="grasp",
        description="Side grasp for cylindrical objects.",
    ),
    3: SourceTemplate(
        category="cylinder_bottle",
        template_id="top_cylinder",
        source_object_id="A15027",
        demo_id="24970fce56",
        behavior="lift",
        phase="grasp",
        description="Top grasp for cylindrical objects.",
    ),
    4: SourceTemplate(
        category="mug",
        template_id="side_open_container",
        source_object_id="C10001",
        demo_id="8f91a5a1be",
        behavior="lift",
        phase="grasp",
        description="Side grasp for open containers and box-like objects.",
    ),
    5: SourceTemplate(
        category="mug",
        template_id="top_open_container",
        source_object_id="S10001",
        demo_id="0e00a051e9",
        behavior="lift",
        phase="grasp",
        description="Top grasp for open containers.",
    ),
    6: SourceTemplate(
        category="bowl",
        template_id="rim_pinch",
        source_object_id="C12001",
        demo_id="67b27a5885",
        behavior="lift",
        phase="grasp",
        description="Rim pinch for bowls, baskets, and low containers.",
    ),
    7: SourceTemplate(
        category="bowl",
        template_id="full_support",
        source_object_id="C12001",
        demo_id="89af8e78ae",
        behavior="lift",
        phase="grasp",
        description="Full support grasp for larger containers.",
    ),
    8: SourceTemplate(
        category="screwdriver",
        template_id="long_axis_hold",
        source_object_id="O21001",
        demo_id="057b976ac2",
        behavior="lift",
        phase="grasp",
        description="Long-axis grasp for tools and bars.",
    ),
    9: SourceTemplate(
        category="screwdriver",
        template_id="long_axis_top",
        source_object_id="O21001",
        demo_id="f91cd4f21b",
        behavior="lift",
        phase="grasp",
        description="Top grasp for thin elongated objects.",
    ),
    10: SourceTemplate(
        category="toothbrush",
        template_id="flat_side",
        source_object_id="C25001",
        demo_id="3154da882c",
        behavior="lift",
        phase="grasp",
        description="Horizontal side grasp for flat elongated objects.",
    ),
    11: SourceTemplate(
        category="headphones",
        template_id="headphones_top",
        source_object_id="C41001",
        demo_id="27f7172256",
        behavior="lift",
        phase="grasp",
        description="Top grasp for deformable-looking or arched rigid objects.",
    ),
    12: SourceTemplate(
        category="headphones",
        template_id="headphones_ears",
        source_object_id="C41001",
        demo_id="7a531742d4",
        behavior="lift",
        phase="grasp",
        description="Side/ear grasp for arched objects.",
    ),
}


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing V2_15 asset manifest: {MANIFEST_PATH}. "
            "Prepare the separately licensed source assets described in "
            "external_data/README.md first."
        )
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "assets" not in payload or not isinstance(payload["assets"], list):
        raise ValueError(f"{MANIFEST_PATH}: expected top-level list field `assets`.")
    return payload


def _enabled_assets(payload: dict[str, Any]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    for raw in payload["assets"]:
        if not bool(raw.get("enabled", True)):
            continue
        if "target_index" not in raw:
            raise ValueError(f"V2_15 asset entry missing target_index: {raw}")
        if "asset_id" not in raw:
            raise ValueError(f"V2_15 asset entry missing asset_id: {raw}")
        if "processed_obj_path" not in raw:
            raise ValueError(f"V2_15 asset entry missing processed_obj_path: {raw}")
        if "source_index" not in raw:
            raise ValueError(f"V2_15 asset entry missing source_index: {raw}")
        target_index = int(raw["target_index"])
        asset_id = str(raw["asset_id"])
        if target_index in seen_indices:
            raise ValueError(f"Duplicate V2_15 target_index {target_index}.")
        if asset_id in seen_ids:
            raise ValueError(f"Duplicate V2_15 asset_id {asset_id!r}.")
        if int(raw["source_index"]) not in SOURCE_TEMPLATES:
            raise ValueError(
                f"{asset_id}: source_index {raw['source_index']} is not defined."
            )
        seen_indices.add(target_index)
        seen_ids.add(asset_id)
        assets.append(raw)
    return assets


_MANIFEST = _load_manifest()
_ASSETS = _enabled_assets(_MANIFEST)

TARGET_OBJECTS = {
    int(asset["target_index"]): TargetObject(
        category=CATEGORY_ID,
        target_id=str(asset["asset_id"]),
        mesh_path=str(asset["processed_obj_path"]),
        scale=1.0,
        notes=str(asset.get("notes", "")),
    )
    for asset in _ASSETS
}

TARGET_INDEX_BY_ID = {
    str(asset["asset_id"]): int(asset["target_index"]) for asset in _ASSETS
}
SOURCE_INDEX_BY_TARGET_INDEX = {
    int(asset["target_index"]): int(asset["source_index"]) for asset in _ASSETS
}
SOURCE_INDEX_BY_ID = {
    str(asset["asset_id"]): int(asset["source_index"]) for asset in _ASSETS
}
ASSET_KIND_BY_ID = {
    str(asset["asset_id"]): str(asset.get("kind", "lift")) for asset in _ASSETS
}


def target_index(asset_id: str) -> int:
    try:
        return TARGET_INDEX_BY_ID[asset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown V2_15 asset_id {asset_id!r}.") from exc


def source_index_for_target(target_index_: int) -> int:
    try:
        return SOURCE_INDEX_BY_TARGET_INDEX[int(target_index_)]
    except KeyError as exc:
        raise KeyError(f"Unknown V2_15 target_index {target_index_!r}.") from exc
