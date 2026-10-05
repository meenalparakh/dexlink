from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="block",
    display_name="block",
    sections=("V",),
    required_data=("source_templates", "target_meshes"),
    density_range=(500.0, 900.0),
    notes="Synthetic colored cube targets for block stacking tasks.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="apple",
        template_id="top",
        source_object_id="C90001",
        demo_id="4f267b3fba",
        behavior="place",
        phase="grasp",
        description="apple top grasp template used as the block pickup proxy",
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="block",
        target_id="red_4",
        mesh_path="assets/blocks/red_4.obj",
        scale=1.0,
        notes="4 cm red cube",
    ),
    2: TargetObject(
        category="block",
        target_id="green_4",
        mesh_path="assets/blocks/green_4.obj",
        scale=1.0,
        notes="4 cm green cube",
    ),
    3: TargetObject(
        category="block",
        target_id="blue_4",
        mesh_path="assets/blocks/blue_4.obj",
        scale=1.0,
        notes="4 cm blue cube",
    ),
    4: TargetObject(
        category="block",
        target_id="red_6",
        mesh_path="assets/blocks/red_6.obj",
        scale=1.0,
        notes="6 cm red cube",
    ),
    5: TargetObject(
        category="block",
        target_id="green_4_5",
        mesh_path="assets/blocks/green_4_5.obj",
        scale=1.0,
        notes="4.5 cm green cube",
    ),
    6: TargetObject(
        category="block",
        target_id="blue_3",
        mesh_path="assets/blocks/blue_3.obj",
        scale=1.0,
        notes="3 cm blue cube",
    ),
    7: TargetObject(
        category="block",
        target_id="yellow_7_5",
        mesh_path="assets/blocks/yellow_7_5.obj",
        scale=1.0,
        notes="7.5 cm yellow cube",
    ),
    8: TargetObject(
        category="block",
        target_id="purple_2",
        mesh_path="assets/blocks/purple_2.obj",
        scale=1.0,
        notes="2 cm purple cube",
    ),
}
