from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="lego_block",
    display_name="lego block",
    sections=("V2_11",),
    required_data=("source_templates", "target_meshes"),
    density_range=(500.0, 900.0),
    notes="Custom 6.3 x 6.3 x 5 cm blocks for real-world stacking demonstrations.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="apple",
        template_id="top",
        source_object_id="C90001",
        demo_id="4f267b3fba",
        behavior="place",
        phase="grasp",
        description="Apple top grasp template used as the block pickup proxy.",
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="lego_block",
        target_id="red_lego_block",
        mesh_path="assets/lego_blocks/red.obj",
        scale=1.0,
        notes="6.3 x 6.3 x 5 cm red block.",
    ),
    2: TargetObject(
        category="lego_block",
        target_id="green_lego_block",
        mesh_path="assets/lego_blocks/green.obj",
        scale=1.0,
        notes="6.3 x 6.3 x 5 cm green block.",
    ),
    3: TargetObject(
        category="lego_block",
        target_id="blue_lego_block",
        mesh_path="assets/lego_blocks/blue.obj",
        scale=1.0,
        notes="6.3 x 6.3 x 5 cm blue block.",
    ),
}
