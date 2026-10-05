from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
    oakink_demo_path,
)

CATEGORY = ObjectCategory(
    category="toothbrush",
    display_name="toothbrush",
    sections=("III",),
    required_data=("source_templates", "target_meshes"),
    density_range=(700.0, 1400.0),
    notes="Long-horizon task object for placing into an inverted cup.",
)

# Sections: ideal III invert-cup-and-place-toothbrush task.
# Source templates needed:
# - Minimum: 0 unless selected as the representative compositional task.
# - Ideal: 2-3 phase templates for grasping and placing into the inverted cup.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="toothbrush",
        template_id="side_horizontal",
        source_object_id="C25001",
        demo_id="3154da882c",
        behavior="lift",
        phase="grasp",
        description="side grasp for lifting a horizontal toothbrush",
    ),
    2: SourceTemplate(
        category="toothbrush",
        template_id="handle_use",
        source_object_id="C25001",
        demo_id="7353c948eb",
        behavior="place_in_cup",
        phase="grasp",
        description="handle grasp for toothbrush use/placement",
    ),
    3: SourceTemplate(
        category="toothbrush",
        template_id="top",
        source_object_id="C25001",
        demo_id="94a15def16",
        behavior="lift",
        phase="grasp",
        description="top grasp",
    ),
}

# Target instances needed:
# - Minimum: 0 unless selected.
# - Ideal: 1 toothbrush target paired with the selected cup target for III.
TARGET_OBJECTS = {
    1: TargetObject(
        category="toothbrush",
        target_id="C25001",
        mesh_path=None,
        scale=2.0,
        target_demo_path=oakink_demo_path("toothbrush", "C25001", "3154da882c"),
        notes="OakInk target demo path; OakInkProcessor extracts object mesh/URDF from the demo.",
    ),
}
