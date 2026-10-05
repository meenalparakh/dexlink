from __future__ import annotations

from dexlink.src.reference.common.categories import (
    bottle,
    drawer,
    fruit,
    hammer,
    mouse,
    mug,
    trigger_sprayer,
    wineglass,
)
from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
    oakink_demo_path,
)

CATEGORY = ObjectCategory(
    category="template_transfer",
    display_name="paired template transfer",
    sections=("IV",),
    required_data=("source_templates", "oakink_target_instances"),
    density_range=(500.0, 1500.0),
    notes=(
        "Gate 4C transferred-template and exact-target controls using OakInk "
        "source and target demonstrations."
    ),
)


# Reuse the established category templates whenever the requested interaction
# exists. The toothbrush template is the proxy already used for the drawer;
# entries 7, 9, and 11--13 are exact-target templates introduced for paired
# comparisons whose chosen OakInk demo was not already registered as a source.
SOURCE_TEMPLATES = {
    1: mug.SOURCE_TEMPLATES[7],  # mug top, S10001
    2: mug.SOURCE_TEMPLATES[3],  # mug top-diagonal, C10001
    3: mug.SOURCE_TEMPLATES[1],  # mug handle, C10001
    4: bottle.SOURCE_TEMPLATES[2],  # bottle top, A15027
    5: bottle.SOURCE_TEMPLATES[1],  # bottle side, A15027
    6: fruit.SOURCE_TEMPLATES[1],  # apple top, C90001
    7: SourceTemplate(
        category="cameras",
        template_id="apple_like_top_o50002",
        source_object_id="O50002",
        demo_id="e295428b5e",
        behavior="lift",
        phase="grasp",
        description=(
            "Exact-target O50002 camera grasp matching the apple-template "
            "approach direction."
        ),
    ),
    8: drawer.SOURCE_TEMPLATES[1],  # toothbrush handle proxy, C25001
    9: SourceTemplate(
        category="fryingpan",
        template_id="handle_c26001",
        source_object_id="C26001",
        demo_id="3a99a57b64",
        behavior="lift",
        phase="grasp",
        description="Exact-target handle grasp for the C26001 frying pan.",
    ),
    11: SourceTemplate(
        category="mug",
        template_id="handle_s10014",
        source_object_id="S10014",
        demo_id="3af983ec0e",
        behavior="lift",
        phase="grasp",
        description="Exact-target handle grasp for the S10014 mug instance.",
    ),
    12: SourceTemplate(
        category="cylinder_bottle",
        template_id="top_c15001",
        source_object_id="C15001",
        demo_id="16681bd36c",
        behavior="lift",
        phase="grasp",
        description="Exact-target top grasp for the C15001 bottle instance.",
    ),
    13: SourceTemplate(
        category="cylinder_bottle",
        template_id="side_a15015",
        source_object_id="A15015",
        demo_id="d02764ab7e",
        behavior="lift",
        phase="grasp",
        description="Exact-target side grasp for the A15015 bottle instance.",
    ),
    14: trigger_sprayer.SOURCE_TEMPLATES[2],  # A01008 spraying configuration
    15: trigger_sprayer.SOURCE_TEMPLATES[4],  # A01027 spraying configuration
    16: mouse.SOURCE_TEMPLATES[1],  # C51001 top grasp
    17: hammer.SOURCE_TEMPLATES[3],  # C22001 handle grasp
    18: wineglass.SOURCE_TEMPLATES[3],  # C14001 stem grasp
}


TARGET_OBJECTS = {
    1: TargetObject(
        category=CATEGORY.category,
        target_id="mug_c10001_top",
        mesh_path=None,
        target_demo_path=oakink_demo_path("mug", "C10001", "cc7dfee50b"),
        scale=1.0,
        notes="C10001 mug instance with its registered top-diagonal grasp.",
    ),
    2: TargetObject(
        category=CATEGORY.category,
        target_id="mug_s10014_handle",
        mesh_path=None,
        target_demo_path=oakink_demo_path("mug", "S10014", "3af983ec0e"),
        scale=1.0,
        notes="S10014 mug instance with a visually screened handle grasp.",
    ),
    3: TargetObject(
        category=CATEGORY.category,
        target_id="bottle_c15001_top",
        mesh_path=None,
        target_demo_path=oakink_demo_path("cylinder_bottle", "C15001", "16681bd36c"),
        scale=1.0,
        notes="C15001 bottle instance with a visually screened top grasp.",
    ),
    4: TargetObject(
        category=CATEGORY.category,
        target_id="bottle_a15015_side",
        mesh_path=None,
        target_demo_path=oakink_demo_path("cylinder_bottle", "A15015", "d02764ab7e"),
        scale=1.0,
        notes="A15015 bottle instance with a visually screened side grasp.",
    ),
    5: TargetObject(
        category=CATEGORY.category,
        target_id="camera_o50002_top",
        mesh_path=None,
        target_demo_path=SOURCE_TEMPLATES[7].demo_path,
        scale=1.0,
        notes=(
            "O50002 camera instance with the user-selected e295428b5e grasp, "
            "which matches the apple-template approach direction."
        ),
    ),
    6: TargetObject(
        category=CATEGORY.category,
        target_id="fryingpan_c26001_handle",
        mesh_path=None,
        target_demo_path=SOURCE_TEMPLATES[9].demo_path,
        scale=1.0,
        notes=(
            "C26001 frying-pan instance with the user-selected handle grasp "
            "from OakInk demo 3a99a57b64."
        ),
    ),
    7: TargetObject(
        category=CATEGORY.category,
        target_id="trigger_sprayer_a01027_spraying",
        mesh_path=None,
        target_demo_path=SOURCE_TEMPLATES[15].demo_path,
        scale=1.0,
        notes=(
            "A01027 trigger-sprayer instance with the selected spraying grasp "
            "used as the matched exact-target control."
        ),
    ),
    8: TargetObject(
        category=CATEGORY.category,
        target_id="mouse_c51001_top",
        mesh_path=None,
        target_demo_path=SOURCE_TEMPLATES[16].demo_path,
        scale=1.0,
        notes=(
            "C51001 mouse instance with its registered top grasp, used as the "
            "matched exact-target control for apple-to-mouse transfer."
        ),
    ),
    9: TargetObject(
        category=CATEGORY.category,
        target_id="wineglass_c14001_stem",
        mesh_path=None,
        target_demo_path=SOURCE_TEMPLATES[18].demo_path,
        scale=1.0,
        notes=(
            "C14001 wineglass instance with the selected stem grasp, used as "
            "the matched exact-target control for hammer-to-wineglass transfer."
        ),
    ),
}
