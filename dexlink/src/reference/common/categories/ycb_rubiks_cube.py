from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="ycb_rubiks_cube",
    display_name="YCB Rubik's cube",
    sections=("V2_11",),
    required_data=("source_templates", "target_meshes"),
    density_range=(500.0, 900.0),
    notes="YCB Rubik's cube target for real-world cube lift demonstrations.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="apple",
        template_id="top",
        source_object_id="C90001",
        demo_id="4f267b3fba",
        behavior="lift",
        phase="grasp",
        description="Apple top grasp used as a cube top grasp template.",
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="ycb_rubiks_cube",
        target_id="077_rubiks_cube",
        mesh_path="external_data/ycb/077_rubiks_cube/google_16k/textured.obj",
        scale=1.0,
        notes="YCB 077 Rubik's cube Google 16k mesh.",
    ),
}
