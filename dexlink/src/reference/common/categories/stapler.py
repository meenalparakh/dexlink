from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="stapler",
    display_name="stapler",
    sections=("III",),
    required_data=("source_templates", "articulated_asset"),
    density_range=(700.0, 8000.0),
    notes="Ideal fine-grained articulation task.",
)

# Sections: ideal III fine-grained articulation.
# Source templates needed:
# - Minimum: 0; not part of the minimum plan.
# - Ideal: 2-3 phase templates for stapler articulation.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="stapler",
        template_id="stapling",
        source_object_id="C37001",
        demo_id="84c7c88b70",
        behavior="articulate",
        phase="staple",
        description="stapling instance",
    ),
    2: SourceTemplate(
        category="stapler",
        template_id="top",
        source_object_id="C37001",
        demo_id="31f300eb61",
        behavior="articulate",
        phase="grasp",
        description="top grasp",
    ),
    3: SourceTemplate(
        category="stapler",
        template_id="bottom",
        source_object_id="C37001",
        demo_id="7f19aaad99",
        behavior="articulate",
        phase="grasp",
        description="bottom grasp close to stapling instance",
    ),
}

# Target instances needed:
# - Minimum: 0; not part of the minimum plan.
# - Ideal: 1 articulated stapler asset for III.
TARGET_OBJECTS = {
    1: TargetObject(
        category="stapler",
        target_id="stapler_mobility",
        mesh_path="external_data/partnet/stapler/mobility.urdf",
        scale=1.0,
        notes="Articulated stapler URDF.",
    ),
}
