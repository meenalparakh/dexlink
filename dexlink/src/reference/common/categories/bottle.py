from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
    shapenet_mesh,
)

CATEGORY = ObjectCategory(
    category="bottle",
    display_name="bottle",
    sections=("I", "II", "V"),
    required_data=("source_templates", "target_meshes", "real_object_optional"),
    density_range=(800.0, 1200.0),
    notes="Optional Section I category, bottle/cup reuse family, and candidate lift+tilt real-world task.",
)

# Sections: optional I, ideal II as bottle/cup family, V lift+tilt real-world candidate.
# Source templates needed:
# - Minimum: 1 lift+tilt template if selected for V, or 2 lift + 1-2 tilt templates if selected for I.
# - Ideal: 2 grasp templates for II if bottle is chosen over cup; 1 real-world lift+tilt template for V.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="cylinder_bottle",
        template_id="side",
        source_object_id="A15027",
        demo_id="107c064718",
        behavior="lift+tilt",
        phase="grasp",
        description="side grasp for lift and tilt",
    ),
    2: SourceTemplate(
        category="cylinder_bottle",
        template_id="top",
        source_object_id="A15027",
        demo_id="24970fce56",
        behavior="lift",
        phase="grasp",
        description="top grasp for lift",
    ),
    3: SourceTemplate(
        category="cylinder_bottle",
        template_id="diagonal",
        source_object_id="C15001",
        demo_id="f2632e5b80",
        behavior="lift",
        phase="grasp",
        description="diagonal grasp for lift",
    ),
    4: SourceTemplate(
        category="cylinder_bottle",
        template_id="side_b9142e8991",
        source_object_id="A15027",
        demo_id="b9142e8991",
        behavior="lift",
        phase="grasp",
        description=(
            "side grasp used to hold the V3.08 mounting cylinder and the lid "
            "task's container"
        ),
    ),
}

# Target instances needed:
# - Minimum: 1 target if selected for optional I/V.
# - Ideal: 4 targets for II if bottle is chosen as the bottle/cup reuse family; 1 real object for V.
TARGET_OBJECTS = {
    1: TargetObject(
        category="bottle",
        target_id="1071fa4cddb2da2fc8724d5673a063a6",
        mesh_path=shapenet_mesh("bottle", "1071fa4cddb2da2fc8724d5673a063a6"),
        scale=1.0,
    ),
    2: TargetObject(
        category="bottle",
        target_id="10f709cecfbb8d59c2536abb1e8e5eab",
        mesh_path=shapenet_mesh("bottle", "10f709cecfbb8d59c2536abb1e8e5eab"),
        scale=1.0,
    ),
    3: TargetObject(
        category="bottle",
        target_id="134c723696216addedee8d59893c8633",
        mesh_path=shapenet_mesh("bottle", "134c723696216addedee8d59893c8633"),
        scale=1.0,
    ),
    4: TargetObject(
        category="bottle",
        target_id="14588e373cdb9143ca972ee98a070b3d",
        mesh_path=shapenet_mesh("bottle", "14588e373cdb9143ca972ee98a070b3d"),
        scale=1.0,
    ),
    5: TargetObject(
        category="bottle",
        target_id="150108e481b52e41bfeb5b5bfc3dfc3b",
        mesh_path=shapenet_mesh("bottle", "150108e481b52e41bfeb5b5bfc3dfc3b"),
        scale=1.0,
    ),
    7: TargetObject(
        category="bottle",
        target_id="1071fa4cddb2da2fc8724d5673a063a6_scale_050",
        mesh_path=shapenet_mesh("bottle", "1071fa4cddb2da2fc8724d5673a063a6"),
        scale=1.0,
        notes="V2_06 geometry-size variant of bottle target 1; scale comes from transform scale.txt.",
    ),
    8: TargetObject(
        category="bottle",
        target_id="1071fa4cddb2da2fc8724d5673a063a6_scale_075",
        mesh_path=shapenet_mesh("bottle", "1071fa4cddb2da2fc8724d5673a063a6"),
        scale=1.0,
        notes="V2_06 geometry-size variant of bottle target 1; scale comes from transform scale.txt.",
    ),
    9: TargetObject(
        category="bottle",
        target_id="1071fa4cddb2da2fc8724d5673a063a6_scale_125",
        mesh_path=shapenet_mesh("bottle", "1071fa4cddb2da2fc8724d5673a063a6"),
        scale=1.0,
        notes="V2_06 geometry-size variant of bottle target 1; scale comes from transform scale.txt.",
    ),
    10: TargetObject(
        category="bottle",
        target_id="1071fa4cddb2da2fc8724d5673a063a6_scale_150",
        mesh_path=shapenet_mesh("bottle", "1071fa4cddb2da2fc8724d5673a063a6"),
        scale=1.0,
        notes="V2_06 geometry-size variant of bottle target 1; scale comes from transform scale.txt.",
    ),
    11: TargetObject(
        category="bottle",
        target_id="006_mustard_bottle",
        mesh_path="external_data/ycb/006_mustard_bottle/google_16k/textured.obj",
        scale=1.0,
        notes="YCB 006 mustard bottle Google 16k mesh for real-world lift demos.",
    ),
    12: TargetObject(
        category="bottle",
        target_id="1071fa4cddb2da2fc8724d5673a063a6_reorient_z000",
        mesh_path=shapenet_mesh("bottle", "1071fa4cddb2da2fc8724d5673a063a6"),
        scale=1.0,
        notes="Bottle side-grasp reorientation waypoint with 0 degree target z rotation.",
    ),
    13: TargetObject(
        category="bottle",
        target_id="1071fa4cddb2da2fc8724d5673a063a6_reorient_z060",
        mesh_path=shapenet_mesh("bottle", "1071fa4cddb2da2fc8724d5673a063a6"),
        scale=1.0,
        notes="Bottle side-grasp reorientation waypoint with 60 degree target z rotation.",
    ),
    14: TargetObject(
        category="bottle",
        target_id="1071fa4cddb2da2fc8724d5673a063a6_reorient_z120",
        mesh_path=shapenet_mesh("bottle", "1071fa4cddb2da2fc8724d5673a063a6"),
        scale=1.0,
        notes="Bottle side-grasp reorientation waypoint with 120 degree target z rotation.",
    ),
    15: TargetObject(
        category="bottle",
        target_id="1071fa4cddb2da2fc8724d5673a063a6_reorient_z180",
        mesh_path=shapenet_mesh("bottle", "1071fa4cddb2da2fc8724d5673a063a6"),
        scale=1.0,
        notes="Bottle side-grasp reorientation waypoint with 180 degree target z rotation.",
    ),
    16: TargetObject(
        category="bottle",
        target_id="1071fa4cddb2da2fc8724d5673a063a6_scale_070",
        mesh_path=shapenet_mesh("bottle", "1071fa4cddb2da2fc8724d5673a063a6"),
        scale=1.0,
        notes="V3_03 scale-reuse variant of bottle target 1 at 0.70x geometry scale.",
    ),
    17: TargetObject(
        category="bottle",
        target_id="1071fa4cddb2da2fc8724d5673a063a6_scale_085",
        mesh_path=shapenet_mesh("bottle", "1071fa4cddb2da2fc8724d5673a063a6"),
        scale=1.0,
        notes="V3_03 scale-reuse variant of bottle target 1 at 0.85x geometry scale.",
    ),
    # V3_08 mounting cylinders. They are bottle targets because the task grasps
    # them with the cylindrical bottle side template, and a source template and
    # its target are looked up in the same category. The two share an outside, so
    # the grasp, the reach and the lift are identical and only the bore differs.
    18: TargetObject(
        category="bottle",
        target_id="mounting_cylinder_clearance_005",
        mesh_path="assets/mounting/hollow_cylinder_clearance_005.obj",
        scale=1.0,
        notes="V3_08 hollow mounting cylinder, 0.030 m outer radius, 0.005 m bore clearance.",
    ),
    # The lid task's container, listed here as well as in the container category
    # so it can be paired with the cylindrical bottle side grasp: a source
    # template and its target are looked up in the same category.
    20: TargetObject(
        category="bottle",
        target_id="box_with_lid_container",
        mesh_path="assets/box_with_lid/container.obj",
        scale=1.0,
        notes="V3_08 lid-on-container container, side-grasped rather than held from underneath.",
    ),
    21: TargetObject(
        category="bottle",
        target_id="mounting_cylinder_clearance_003",
        mesh_path="assets/mounting/hollow_cylinder_clearance_003.obj",
        scale=1.0,
        notes="V3_08 hollow mounting cylinder, 0.030 m outer radius, 0.003 m bore clearance.",
    ),
    22: TargetObject(
        category="bottle",
        target_id="mounting_cylinder_clearance_007",
        mesh_path="assets/mounting/hollow_cylinder_clearance_007.obj",
        scale=1.0,
        notes="V3_08 hollow mounting cylinder, 0.030 m outer radius, 0.007 m bore clearance.",
    ),
}
