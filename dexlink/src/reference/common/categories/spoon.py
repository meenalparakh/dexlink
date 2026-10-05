from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="spoon",
    display_name="spoon",
    sections=("III", "V"),
    required_data=("source_templates", "target_meshes", "real_object_optional"),
    density_range=(2500.0, 8000.0),
    notes="Breakfast/table-setting object: pick spoon from holder/table and place it inside a bowl.",
)

# Sections: III/V simple breakfast place.
# Source templates needed:
# - Minimum: 1 spoon grasp template for placing the spoon into the bowl.
# - Ideal: 2 templates: one pickup grasp and one bowl-insertion/place relation.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="toothbrush",
        template_id="top_proxy",
        source_object_id="C25001",
        demo_id="94a15def16",
        behavior="place_in_bowl",
        phase="grasp",
        description="Top grasp proxy for spoon placement.",
    ),
}

# Target instances needed:
# - Minimum: 1 spoon target object for the breakfast task.
# - Ideal: 1 simulated target and optionally 1 real spoon target for Section V.
TARGET_OBJECTS = {
    1: TargetObject(
        category="spoon",
        target_id="spoon",
        mesh_path="external_data/sam3d/spoon/spoon/model.obj",
        scale=1.0,
    ),
}
