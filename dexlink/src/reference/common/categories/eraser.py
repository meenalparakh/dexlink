from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="eraser",
    display_name="eraser",
    sections=("III",),
    required_data=("source_templates", "target_meshes"),
    density_range=(900.0, 1400.0),
    notes="Long-horizon board-erasing task object.",
)

# Sections: ideal III long-horizon board-erasing task.
# Source templates needed:
# - Minimum: 0 unless board erasing is selected as the representative compositional task.
# - Ideal: 2-3 phase templates for grasp/contact/erase phases.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="mouse",
        template_id="side_grasp_proxy",
        source_object_id="C51001",
        demo_id="453c469466",
        behavior="place",
        phase="grasp",
        description="lift configuration for desk setup",
    ),
}
# Target instances needed:
# - Minimum: 0 unless selected.
# - Ideal: 1 eraser target plus the paired board/environment asset for III.
TARGET_OBJECTS = {
    1: TargetObject(
        category="eraser",
        target_id="procedural_eraser",
        mesh_path="assets/eraser/eraser.obj",
        scale=1.0,
        notes="Generated cuboid target: height 2.7, width 4.8, length 12.8.",
    ),
}
