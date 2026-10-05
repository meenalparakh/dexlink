from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
    shapenet_mesh,
    shapenet_transform,
)

CATEGORY = ObjectCategory(
    category="mug",
    display_name="mug",
    sections=("I", "II", "IV", "V"),
    required_data=("source_templates", "target_meshes", "real_object_optional"),
    density_range=(700.0, 1400.0),
    notes="Core category: lift, lift+tilt, cross-geometry reuse, robustness, and real-world lift.",
)

# Sections: I, II, IV, V.
# Source templates needed:
# - Minimum: 2 lift templates for I; 1-2 tilt templates for I/V; 1 grasp template for II/IV.
# - Ideal: 3 lift + 2 tilt templates for I; 2 grasp templates for II; 1 template for IV/V.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="mug",
        template_id="handle",
        source_object_id="C10001",
        demo_id="7a99e06f35",
        behavior="lift",
        phase="grasp",
        description="handle grasp for lift/reuse",
    ),
    2: SourceTemplate(
        category="mug",
        template_id="side",
        source_object_id="C10001",
        demo_id="8f91a5a1be",
        behavior="lift",
        phase="grasp",
        description="side grasp for lift/reuse",
    ),
    3: SourceTemplate(
        category="mug",
        template_id="top_diagonal",
        source_object_id="C10001",
        demo_id="cc7dfee50b",
        behavior="lift",
        phase="grasp",
        description="Diagonal semi-top grasp for lift/reuse.",
    ),
    4: SourceTemplate(
        category="mug",
        template_id="handle_2",
        source_object_id="C10001",
        demo_id="d9152a3a5b",
        behavior="tilt",
        phase="grasp",
        description="handle grasp candidate for lift+tilt",
    ),
    5: SourceTemplate(
        category="mug",
        template_id="handle_3",
        source_object_id="C10001",
        demo_id="df53267320",
        behavior="tilt",
        phase="grasp",
        description="alternate handle grasp candidate for lift+tilt",
    ),
    6: SourceTemplate(
        category="cylinder_bottle",
        template_id="bottle_bottom_support",
        source_object_id="A15015",
        demo_id="3541fe8b0e",
        behavior="support",
        phase="grasp",
        description="Bottle bottom-support grasp used as a cross-category mug template.",
    ),
    7: SourceTemplate(
        category="mug",
        template_id="top",
        source_object_id="S10001",
        demo_id="0e00a051e9",
        behavior="lift",
        phase="grasp",
        description="Actual mug top grasp for lift.",
    ),
    8: SourceTemplate(
        category="mug",
        template_id="side_hang",
        source_object_id="C10001",
        demo_id="c8c634bd80",
        behavior="lift",
        phase="grasp",
        description="Side grasp used for hanging a mug on a stand.",
    ),
}

# Target instances needed:
# - Minimum: 3 simulation targets for II, of which one can also serve I/IV; 1 real object for V.
# - Ideal: 4 simulation targets for II, one selected target for I/IV, and 1 real object for V.
TARGET_OBJECTS = {
    1: TargetObject(
        category="mug",
        target_id="15bd6225c209a8e3654b0ce7754570c8",
        mesh_path=shapenet_mesh("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        transform_path=shapenet_transform("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        scale=0.2,
    ),
    2: TargetObject(
        category="mug",
        target_id="17952a204c0a9f526c69dceb67157a66",
        mesh_path=shapenet_mesh("mug", "17952a204c0a9f526c69dceb67157a66"),
        transform_path=shapenet_transform("mug", "17952a204c0a9f526c69dceb67157a66"),
        scale=0.2,
    ),
    3: TargetObject(
        category="mug",
        target_id="1ae1ba5dfb2a085247df6165146d5bbd",
        mesh_path=shapenet_mesh("mug", "1ae1ba5dfb2a085247df6165146d5bbd"),
        transform_path=shapenet_transform("mug", "1ae1ba5dfb2a085247df6165146d5bbd"),
        scale=0.2,
    ),
    4: TargetObject(
        category="mug",
        target_id="sam3d_jug_object_0",
        mesh_path="external_data/sam3d/jug/object_0/model.obj",
        scale=1.0,
        notes="SAM3D jug instance used as a handled mug-like target.",
    ),
    6: TargetObject(
        category="mug",
        target_id="15bd6225c209a8e3654b0ce7754570c8_scale_050",
        mesh_path=shapenet_mesh("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        transform_path=shapenet_transform("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        scale=1.0,
        notes="V2_06 geometry-size variant of mug target 1; scale comes from transform scale.txt.",
    ),
    7: TargetObject(
        category="mug",
        target_id="15bd6225c209a8e3654b0ce7754570c8_scale_075",
        mesh_path=shapenet_mesh("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        transform_path=shapenet_transform("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        scale=1.0,
        notes="V2_06 geometry-size variant of mug target 1; scale comes from transform scale.txt.",
    ),
    8: TargetObject(
        category="mug",
        target_id="15bd6225c209a8e3654b0ce7754570c8_scale_125",
        mesh_path=shapenet_mesh("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        transform_path=shapenet_transform("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        scale=1.0,
        notes="V2_06 geometry-size variant of mug target 1; scale comes from transform scale.txt.",
    ),
    9: TargetObject(
        category="mug",
        target_id="15bd6225c209a8e3654b0ce7754570c8_scale_150",
        mesh_path=shapenet_mesh("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        transform_path=shapenet_transform("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        scale=1.0,
        notes="V2_06 geometry-size variant of mug target 1; scale comes from transform scale.txt.",
    ),
    10: TargetObject(
        category="mug",
        target_id="b9f9f5b48ab1153626829c11d9aba173",
        mesh_path=shapenet_mesh("mug", "b9f9f5b48ab1153626829c11d9aba173"),
        transform_path=shapenet_transform("mug", "b9f9f5b48ab1153626829c11d9aba173"),
        scale=0.2,
    ),
    11: TargetObject(
        category="mug",
        target_id="025_mug",
        mesh_path="external_data/ycb/025_mug/google_16k/textured.obj",
        scale=1.0,
        notes="YCB 025 mug Google 16k mesh for real-world top-grasp lift demos.",
    ),
    12: TargetObject(
        category="mug",
        target_id="15bd6225c209a8e3654b0ce7754570c8_scale_070",
        mesh_path=shapenet_mesh("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        transform_path=shapenet_transform("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        scale=1.0,
        notes="V3_03 scale-reuse variant of mug target 1 at 0.70x geometry scale.",
    ),
    13: TargetObject(
        category="mug",
        target_id="15bd6225c209a8e3654b0ce7754570c8_scale_085",
        mesh_path=shapenet_mesh("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        transform_path=shapenet_transform("mug", "15bd6225c209a8e3654b0ce7754570c8"),
        scale=1.0,
        notes="V3_03 scale-reuse variant of mug target 1 at 0.85x geometry scale.",
    ),
}
