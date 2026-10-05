from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="cup",
    display_name="cup",
    sections=("II", "III", "V"),
    required_data=("source_templates", "target_meshes", "real_object_optional"),
    density_range=(700.0, 1400.0),
    notes="Alternative to bottle for reuse and real-world lift+tilt; also used for invert-cup long-horizon task.",
)

# Sections: ideal II as bottle/cup family, ideal III invert-cup task, V lift+tilt candidate.
# Source templates needed:
# - Minimum: 1 lift+tilt template if selected for V.
# - Ideal: 2 grasp templates for II if cup is chosen over bottle; 2-3 phase templates for invert-cup-and-place-toothbrush in III; 1 real-world lift+tilt template for V.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="cup",
        template_id="top",
        source_object_id="C11001",
        demo_id="a5465a7403",
        behavior="lift+tilt",
        phase="grasp",
        description="top grasp for lift and tilt",
    ),
    2: SourceTemplate(
        category="cup",
        template_id="bottom",
        source_object_id="C11001",
        demo_id="3760ccf4d8",
        behavior="invert_place",
        phase="grasp",
        description="bottom grasp for picking when cup is inverted",
    ),
}

# Target instances needed:
# - Minimum: 1 target if selected for V.
# - Ideal: 4 targets for II if cup is chosen as the bottle/cup reuse family; 1 target for III; 1 real object for V.
TARGET_OBJECTS = {
    1: TargetObject(
        category="cup",
        target_id="cup1",
        mesh_path="external_data/sam3d/cup/cup1/model.obj",
        scale=1.0,
    ),
    2: TargetObject(
        category="cup",
        target_id="cup2",
        mesh_path="external_data/sam3d/cup/cup2/model.obj",
        scale=1.0,
    ),
    3: TargetObject(
        category="cup",
        target_id="cup3",
        mesh_path="external_data/sam3d/cup/cup3/model.obj",
        scale=1.0,
    ),
}
