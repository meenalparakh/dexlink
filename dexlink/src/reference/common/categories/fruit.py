from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="fruit",
    display_name="fruit",
    sections=("III", "V"),
    required_data=("source_templates", "target_meshes", "real_object_optional"),
    density_range=(1000.0, 1500.0),
    notes="Breakfast/table-setting object: place a fruit item beside the bowl.",
)

# Sections: III/V simple breakfast place.
# Source templates needed:
# - Minimum: 1 pickup/place template for a fruit item beside the bowl.
# - Ideal: 1-2 templates if using different fruit shapes, e.g. apple and banana.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="apple",
        template_id="top",
        source_object_id="C90001",
        demo_id="4f267b3fba",
        behavior="place",
        phase="grasp",
        description="top grasp for placing fruit beside the bowl",
    ),
}

# Target instances needed:
# - Minimum: 1 fruit target object for the breakfast task.
# - Ideal: 1 simulated target and optionally 1 real fruit target for Section V.
TARGET_OBJECTS = {
    1: TargetObject(
        category="fruit",
        target_id="apple",
        mesh_path="assets/apple/apple.obj",
        scale=1.1,
    ),
    2: TargetObject(
        category="fruit",
        target_id="apple_reorient_z000",
        mesh_path="assets/apple/apple.obj",
        scale=1.1,
        notes="Apple reorientation waypoint with 0 degree target z rotation.",
    ),
    3: TargetObject(
        category="fruit",
        target_id="apple_reorient_z060",
        mesh_path="assets/apple/apple.obj",
        scale=1.1,
        notes="Apple reorientation waypoint with 60 degree target z rotation.",
    ),
    4: TargetObject(
        category="fruit",
        target_id="apple_reorient_z120",
        mesh_path="assets/apple/apple.obj",
        scale=1.1,
        notes="Apple reorientation waypoint with 120 degree target z rotation.",
    ),
    5: TargetObject(
        category="fruit",
        target_id="apple_reorient_z180",
        mesh_path="assets/apple/apple.obj",
        scale=1.1,
        notes="Apple reorientation waypoint with 180 degree target z rotation.",
    ),
}
