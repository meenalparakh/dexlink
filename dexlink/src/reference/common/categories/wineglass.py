from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="wineglass",
    display_name="wineglass",
    sections=("I", "V"),
    required_data=("source_templates", "target_meshes", "real_object_optional"),
    density_range=(700.0, 1400.0),
    notes="Optional lift category for comparing stem grasp versus cup-part grasp.",
)

# Sections: optional I sparse-template lift task, optional V real-world lift.
# Source templates needed:
# - Minimum: 1 lift template if selected for I/V.
# - Ideal: 2 lift templates comparing stem grasp and cup-part grasp.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="wineglass",
        template_id="stem",
        source_object_id="C14001",
        demo_id="1c8bebc322",
        behavior="lift",
        phase="grasp",
        description="stem grasp for lifting a wineglass",
    ),
    2: SourceTemplate(
        category="wineglass",
        template_id="cup_part",
        source_object_id="C14001",
        demo_id="3f3a1477c6",
        behavior="lift",
        phase="grasp",
        description="cup-part grasp for lifting a wineglass",
    ),
    3: SourceTemplate(
        category="wineglass",
        template_id="stem_alt",
        source_object_id="C14001",
        demo_id="b1f9309805",
        behavior="lift",
        phase="grasp",
        description=(
            "Alternate stem grasp selected as the exact-target control for "
            "hammer-handle to wineglass transfer."
        ),
    ),
}

# Target instances needed:
# - Minimum: 1 simulation target for I.
# - Ideal: 1 simulation target plus 1 real object if selected for V.
TARGET_OBJECTS = {
    1: TargetObject(
        category="wineglass",
        target_id="infinigen_export_000",
        mesh_path="external_data/infinigen/infinigen_wineglass/wineglass_low_poly.obj",
        scale=1.0,
        notes="Guessed Infinigen wineglass export path; update if the asset lands elsewhere.",
    ),
}
