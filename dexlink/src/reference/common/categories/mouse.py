from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="mouse",
    display_name="mouse",
    sections=("III", "V"),
    required_data=("source_templates", "target_meshes", "real_object_optional"),
    density_range=(500.0, 1200.0),
    notes="Desk setup object: place mouse on the mousepad zone.",
)

# Sections: III/V tidy desk setup.
# Source templates needed:
# - Minimum: 1 mouse pickup/place template.
# - Ideal: 1-2 templates if you want top grasp and side grasp variants.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="mouse",
        template_id="lift",
        source_object_id="C51001",
        demo_id="453c469466",
        behavior="place",
        phase="grasp",
        description="lift configuration for desk setup",
    ),
}

# Target instances needed:
# - Minimum: 1 mouse target object for the desk task.
# - Ideal: 1 simulated target and optionally 1 real mouse object for Section V.
TARGET_OBJECTS = {
    1: TargetObject(
        category="mouse",
        target_id="mouse",
        mesh_path="external_data/sam3d/mouse/mouse/model.obj",
        scale=1.0,
    ),
}
