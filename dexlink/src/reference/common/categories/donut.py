from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
    oakink_demo_path,
)

CATEGORY = ObjectCategory(
    category="donut",
    display_name="donut",
    sections=("I", "V"),
    required_data=("source_templates", "target_meshes"),
    density_range=(700.0, 1400.0),
    notes="Ring-shaped lift object for V2_10 category-diversity lift experiments.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="donut",
        template_id="top",
        source_object_id="O93001",
        demo_id="cb828e4611",
        behavior="lift",
        phase="grasp",
        description="top grasp for lifting a donut.",
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="donut",
        target_id="O93001",
        mesh_path=None,
        scale=0.8,
        target_demo_path=oakink_demo_path("donut", "O93001", "607ebd518b"),
        notes="OakInk donut target demo path; target scale is set to 0.8.",
    ),
}
