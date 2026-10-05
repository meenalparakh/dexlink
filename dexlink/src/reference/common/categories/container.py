from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="container",
    display_name="container",
    sections=("III", "V"),
    required_data=("source_templates", "target_meshes"),
    density_range=(700.0, 1400.0),
    notes="Box-with-lid container target using a bottom support grasp template.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="cylinder_bottle",
        template_id="bottom_support",
        source_object_id="A15015",
        demo_id="3541fe8b0e",
        behavior="support",
        phase="grasp",
        description="bottom support grasp proxy for holding a container.",
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="container",
        target_id="box_with_lid_container",
        mesh_path="assets/box_with_lid/container.obj",
        scale=1.0,
        notes="Container mesh from the box-with-lid asset.",
    ),
}
