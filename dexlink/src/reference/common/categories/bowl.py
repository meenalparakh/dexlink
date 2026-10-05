from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
    shapenet_mesh,
)

CATEGORY = ObjectCategory(
    category="bowl",
    display_name="bowl",
    sections=("I", "II", "III", "IV", "V"),
    required_data=("source_templates", "target_meshes", "real_object_optional"),
    density_range=(1400.0, 2000.0),
    notes="Core category: lift/reuse plus beyond-grasping support/regrasp behavior.",
)

# Sections: I, II, III, IV, V.
# Source templates needed:
# - Minimum: 2 lift templates for I; 1 grasp template for II/IV; 2-3 phase templates if used for III; 1 real-world lift template for V.
# - Ideal: 3 lift + 2 tilt templates for I; 2 grasp templates for II; 2-3 support/regrasp phase templates for III; 1 template for IV/V.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="bowl",
        template_id="pinch",
        source_object_id="C12001",
        demo_id="67b27a5885",
        behavior="lift",
        phase="grasp",
        description="rim pinch grasp for lift/reuse",
    ),
    2: SourceTemplate(
        category="bowl",
        template_id="full",
        source_object_id="C12001",
        demo_id="89af8e78ae",
        behavior="lift",
        phase="grasp",
        description="full support grasp for lift/reuse",  # usable for tilt
    ),
    3: SourceTemplate(
        category="bowl",
        template_id="full",
        source_object_id="C12001",
        demo_id="89af8e78ae",
        behavior="lift",
        phase="grasp",
        description="full support grasp for lift/reuse",  # usable for tilt
    ),
    4: SourceTemplate(
        category="cameras",
        template_id="camera_top_support",
        source_object_id="O50001",
        demo_id="0fb59ca769",
        behavior="support",
        phase="grasp",
        description="Camera grasp used as a cross-category bowl support template.",
    ),
}

# Target instances needed:
# - Minimum: 3 simulation targets for II, of which one can also serve I/IV; 1 real object for V.
# - Ideal: 4 simulation targets for II, one selected target for I/III/IV, and 1 real object for V.
TARGET_OBJECTS = {
    1: TargetObject(
        category="bowl",
        target_id="11547e8d8f143557525b133235812833",
        mesh_path=shapenet_mesh("bowl", "11547e8d8f143557525b133235812833"),
        scale=1.0,
    ),
    # 2: TargetObject(         # not correct bowl mesh - faulty
    #     category="bowl",
    #     target_id="13e879cb517784a63a4b07a265cff347",
    #     mesh_path=shapenet_mesh("bowl", "13e879cb517784a63a4b07a265cff347"),
    #     scale=1.0,
    # ),
    3: TargetObject(
        category="bowl",
        target_id="1b4d7803a3298f8477bdcb8816a3fac9",
        mesh_path=shapenet_mesh("bowl", "1b4d7803a3298f8477bdcb8816a3fac9"),
        scale=1.0,
    ),
    4: TargetObject(
        category="bowl",
        target_id="34875f8448f98813a2c59a4d90e63212",
        mesh_path=shapenet_mesh("bowl", "34875f8448f98813a2c59a4d90e63212"),
        scale=1.0,
    ),
    # 5: TargetObject(     # faulty bowl mesh
    #     category="bowl",
    #     target_id="36ca3b684dbb9c159599371049c32d38",
    #     mesh_path=shapenet_mesh("bowl", "36ca3b684dbb9c159599371049c32d38"),
    #     scale=1.0,
    # ),
    6: TargetObject(
        category="bowl",
        target_id="ycb_bowl",
        mesh_path="external_data/ycb/bowl/textured.obj",
        scale=1.0,
        notes="Real-world/YCB bowl demo object for Section V.",
    ),
    7: TargetObject(
        category="bowl",
        target_id="sam3d_bowl_object",
        mesh_path="external_data/sam3d/bowl/object/model.obj",
        scale=1.0,
        notes="SAM3D bowl used for breakfast setup tasks.",
    ),
    9: TargetObject(
        category="bowl",
        target_id="sam3d_bowl_object_scale_050",
        mesh_path="external_data/sam3d/bowl/object/model.obj",
        scale=1.0,
        notes="V2_06 geometry-size variant of bowl target 7; scale comes from transform scale.txt.",
    ),
    10: TargetObject(
        category="bowl",
        target_id="sam3d_bowl_object_scale_075",
        mesh_path="external_data/sam3d/bowl/object/model.obj",
        scale=1.0,
        notes="V2_06 geometry-size variant of bowl target 7; scale comes from transform scale.txt.",
    ),
    11: TargetObject(
        category="bowl",
        target_id="sam3d_bowl_object_scale_125",
        mesh_path="external_data/sam3d/bowl/object/model.obj",
        scale=1.0,
        notes="V2_06 geometry-size variant of bowl target 7; scale comes from transform scale.txt.",
    ),
    12: TargetObject(
        category="bowl",
        target_id="sam3d_bowl_object_scale_150",
        mesh_path="external_data/sam3d/bowl/object/model.obj",
        scale=1.0,
        notes="V2_06 geometry-size variant of bowl target 7; scale comes from transform scale.txt.",
    ),
}
