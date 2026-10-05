from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="screwdriver",
    display_name="screwdriver",
    sections=("I", "II"),
    required_data=("source_templates", "target_meshes"),
    density_range=(2500.0, 8000.0),
    notes="Ideal Section I category and alternative to marker in the marker/screwdriver reuse family.",
)

# Sections: ideal I and ideal II as marker/screwdriver family.
# Source templates needed:
# - Minimum: 0; not part of the minimum plan unless used instead of marker as an optional category.
# - Ideal: 3 lift + 2 tilt templates for I; 2 grasp templates for II if screwdriver is chosen over marker.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="screwdriver",
        template_id="holding",
        source_object_id="O21001",
        demo_id="057b976ac2",
        behavior="lift",
        phase="grasp",
        description="holding grasp",
    ),
    2: SourceTemplate(
        category="screwdriver",
        template_id="rotate_holding",
        source_object_id="O21001",
        demo_id="6798f10028",
        behavior="reorient",
        phase="grasp",
        description="holding grasp for trying to rotate",
    ),
    3: SourceTemplate(
        category="screwdriver",
        template_id="top",
        source_object_id="O21001",
        demo_id="f91cd4f21b",
        behavior="lift",
        phase="grasp",
        description="top grasp from the thin metallic cylinder of the screwdriver",
    ),
}

# Target instances needed:
# - Minimum: 0 unless selected as an optional category.
# - Ideal: 1 target for I; 4 targets for II if screwdriver is chosen as the marker/screwdriver reuse family.
TARGET_OBJECTS = {
    1: TargetObject(
        category="screwdriver",
        target_id="screwdriver1",
        mesh_path="external_data/sam3d/screwdrivers/s1/model.obj",
        scale=1.0,
    ),
    2: TargetObject(
        category="screwdriver",
        target_id="screwdriver2",
        mesh_path="external_data/sam3d/screwdrivers/s2/model.obj",
        scale=1.0,
    ),
    3: TargetObject(
        category="screwdriver",
        target_id="screwdriver3",
        mesh_path="external_data/sam3d/screwdrivers/s3/model.obj",
        scale=1.0,
    ),
    4: TargetObject(
        category="screwdriver",
        target_id="screwdriver4",
        mesh_path="external_data/sam3d/screwdrivers/s4/model.obj",
        scale=1.0,
    ),
}
