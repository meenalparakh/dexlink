from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
    shapenet_mesh,
)

CATEGORY = ObjectCategory(
    category="headphones",
    display_name="headphones",
    sections=("I",),
    required_data=("source_templates", "target_meshes"),
    density_range=(200.0, 900.0),
    notes="Ideal Section I category for showing distinct sparse-template-conditioned behaviors.",
)

# Sections: ideal I only.
# Source templates needed:
# - Minimum: 0; not part of the minimum plan.
# - Ideal: 3 lift templates and 2 lift+tilt templates for Section I.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="headphones",
        template_id="top",
        source_object_id="C41001",
        demo_id="27f7172256",
        behavior="lift",
        phase="grasp",
        description="top grasp",
    ),
    2: SourceTemplate(
        category="headphones",
        template_id="ears",
        source_object_id="C41001",
        demo_id="7a531742d4",
        behavior="lift",
        phase="grasp",
        description="ear grasp",
    ),
    3: SourceTemplate(
        category="headphones",
        template_id="top_one_finger_out",
        source_object_id="C41001",
        demo_id="ed97ac6c4a",
        behavior="lift",
        phase="grasp",
        description="top grasp with one finger out",
    ),
}

# Target instances needed:
# - Minimum: 0; not part of the minimum plan.
# - Ideal: 1 target instance for Section I.
TARGET_OBJECTS = {
    1: TargetObject(
        category="headphones",
        target_id="1757fe64e76a9630fc176230c2f2d294",
        mesh_path=shapenet_mesh("headphone", "1757fe64e76a9630fc176230c2f2d294"),
        scale=0.19,
        notes="ShapeNet synset 03261776.",
    ),
    2: TargetObject(
        category="headphones",
        target_id="1a5e2a7cddc8e46aa681aea7976a4565",
        mesh_path=shapenet_mesh("headphone", "1a5e2a7cddc8e46aa681aea7976a4565"),
        scale=0.22,
        notes="ShapeNet synset 03261776.",
    ),
    3: TargetObject(
        category="headphones",
        target_id="2b28e2a5080101d245af43a64155c221",
        mesh_path=shapenet_mesh("headphone", "2b28e2a5080101d245af43a64155c221"),
        scale=0.25,
        notes="ShapeNet synset 03261776.",
    ),
}
