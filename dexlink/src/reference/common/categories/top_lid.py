from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="top_lid",
    display_name="top lid",
    sections=("III", "V"),
    required_data=("source_templates", "target_meshes"),
    density_range=(700.0, 1400.0),
    notes="Box-with-lid top target using camera top-grasp source templates.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="cameras",
        template_id="top_grasp",
        source_object_id="O50001",
        demo_id="0fb59ca769",
        behavior="lift",
        phase="grasp",
        description="top grasp candidate for lifting a lid.",
    ),
    2: SourceTemplate(
        category="cameras",
        template_id="top_grasp_alt",
        source_object_id="C50001",
        demo_id="6ccd5a39a2",
        behavior="lift",
        phase="grasp",
        description="alternate top grasp candidate for lifting a lid.",
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="top_lid",
        target_id="box_with_lid_lid",
        mesh_path="assets/box_with_lid/lid.obj",
        scale=1.0,
        notes="Lid mesh from the box-with-lid asset.",
    ),
}
