from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="ycb_cup",
    display_name="YCB cup",
    sections=("V2_11",),
    required_data=("source_templates", "target_meshes"),
    density_range=(700.0, 1400.0),
    notes="YCB cups for real-world cup stacking demonstrations.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="mug",
        template_id="mug_top",
        source_object_id="S10001",
        demo_id="0e00a051e9",
        behavior="stack",
        phase="grasp",
        description="Mug top grasp used as the YCB cup stacking template.",
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="ycb_cup",
        target_id="065-a_cups_small",
        mesh_path="external_data/ycb/065-a_cups/google_16k/textured.obj",
        scale=1.0,
        notes="Smallest measured YCB cup from the downloaded cup set.",
    ),
    2: TargetObject(
        category="ycb_cup",
        target_id="065-j_cups_large",
        mesh_path="external_data/ycb/065-j_cups/google_16k/textured.obj",
        scale=1.0,
        notes="Largest measured YCB cup from the downloaded cup set.",
    ),
}
