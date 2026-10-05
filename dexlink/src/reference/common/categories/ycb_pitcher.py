from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="ycb_pitcher",
    display_name="YCB pitcher",
    sections=("V2_11",),
    required_data=("source_templates", "target_meshes"),
    density_range=(700.0, 1400.0),
    notes="YCB pitcher target for real-world lift demonstrations.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="mug",
        template_id="mug_handle",
        source_object_id="C10001",
        demo_id="7a99e06f35",
        behavior="lift",
        phase="grasp",
        description="Mug handle grasp used as the pitcher handle grasp template.",
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="ycb_pitcher",
        target_id="019_pitcher_base",
        mesh_path="external_data/ycb/019_pitcher_base/google_16k/textured.obj",
        scale=1.0,
        notes="YCB 019 pitcher base Google 16k mesh.",
    ),
}
