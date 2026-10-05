from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="marker",
    display_name="marker",
    sections=("I", "II", "III", "IV", "V"),
    required_data=("source_templates", "target_meshes", "real_object_optional"),
    density_range=(700.0, 1400.0),
    notes="Used for sparse-template learning, possible reuse family, reorientation, and optional real-world manipulation.",
)

# Sections: optional minimum I/IV/V, minimum III if chosen for post-grasp manipulation, ideal I/II/III.
# Source templates needed:
# - Minimum: 2 lift + 1-2 tilt templates if selected for I; 1 grasp template if selected for IV/V; 2-3 reorientation phase templates if selected for III.
# - Ideal: 3 lift + 2 tilt templates for I; 2 grasp templates for II if marker is chosen over screwdriver; 2-3 reorientation templates for III.

# not for I.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="pen",
        template_id="top",
        source_object_id="Y29040",
        demo_id="021b428f00",
        behavior="lift",
        phase="grasp",
        description="top grasp for lift",
    ),
    2: SourceTemplate(
        category="pen",
        template_id="writing_stance",
        source_object_id="Y29040",
        demo_id="ada64e4d8a",
        behavior="reorient",
        phase="writing_grasp",
        description="main writing stance",
    ),
}

# Additional candidate source templates, not used for now:
# - pen/Y29040/307d39f875: thumb and fist grasp for lift
# - pen/Y29040/05eca70e3d: body grasp for grasp refinement only
# - pen/Y29040/144d398c6c: two-finger grasp
# - pen/Y29040/21a825c216: power grasp
# - pen/Y29040/56d480cdfb: writing instance 1, almost
# - pen/Y29040/7ed6498b84: writing instance 2, almost

# Target instances needed:
# - Minimum: 1 target if selected for I/III/IV/V.
# - Ideal: 4 targets for II if marker is chosen as the marker/screwdriver reuse family; 1 target for I/III; optional 1 real object for V.
TARGET_OBJECTS = {
    1: TargetObject(
        category="marker",
        target_id="pen1",
        mesh_path="assets/marker_pen/pen1.obj",
        scale=1.0,
    ),
    2: TargetObject(
        category="marker",
        target_id="pen2",
        mesh_path="assets/marker_pen/pen2.obj",
        scale=1.0,
    ),
    3: TargetObject(
        category="marker",
        target_id="pen3",
        mesh_path="assets/marker_pen/pen3.obj",
        scale=1.0,
    ),
    4: TargetObject(
        category="marker",
        target_id="pen4",
        mesh_path="assets/marker_pen/pen4.obj",
        scale=1.0,
    ),
}
