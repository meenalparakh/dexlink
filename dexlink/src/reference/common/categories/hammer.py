from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="hammer",
    display_name="hammer",
    sections=("V",),
    required_data=("source_templates", "target_meshes"),
    density_range=(2000.0, 4000.0),
    notes="Contact-rich hammering category with an OakInk hammer grasp and a hammer target asset.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="hammer",
        template_id="handle_grasp",
        source_object_id="C22001",
        demo_id="109708ca1a",
        behavior="grasp",
        phase="grasp",
        description="Handle grasp for hammer use and contact-rich hammering tasks.",
    ),
    2: SourceTemplate(
        category="hammer",
        template_id="handle_grasp_alt",
        source_object_id="C22001",
        demo_id="70bdfc2cc1",
        behavior="grasp",
        phase="grasp",
        description="Alternate handle grasp for contact-rich hammering tasks.",
    ),
    3: SourceTemplate(
        category="hammer",
        template_id="handle_grasp_transfer",
        source_object_id="C22001",
        demo_id="9011bb484a",
        behavior="grasp",
        phase="grasp",
        description=(
            "Handle grasp selected as the transferred template for a "
            "wineglass-stem target."
        ),
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="hammer",
        target_id="hammer",
        mesh_path="assets/hammer/hammer.urdf",
        scale=1.0,
        notes="Hammer target URDF.",
    ),
}
