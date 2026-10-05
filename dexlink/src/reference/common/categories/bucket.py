from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="bucket",
    display_name="bucket",
    sections=("III",),
    required_data=("source_templates", "articulated_asset"),
    density_range=(900.0, 1200.0),
    notes="PartNet bucket target with a hinged handle for handle grasping, articulation, and lifting.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="headphones",
        template_id="top",
        source_object_id="C41001",
        demo_id="27f7172256",
        behavior="lift",
        phase="grasp",
        description="Headphones top grasp used as a bucket handle proxy.",
    ),
    2: SourceTemplate(
        category="headphones",
        template_id="top_alt",
        source_object_id="C41001",
        demo_id="976a225479",
        behavior="lift",
        phase="grasp",
        description="Alternate headphones top grasp used as a bucket handle proxy.",
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="bucket",
        target_id="partnet_bucket",
        mesh_path="external_data/partnet/bucket/mobility.urdf",
        scale=1.0,
        bps_sampling_link_names=("link_0",),
        notes="PartNet bucket URDF; link_0 is the hinged handle used for BPS sampling.",
    ),
    2: TargetObject(
        category="bucket",
        target_id="partnet_bucket_2",
        mesh_path="external_data/partnet/bucket_2/mobility.urdf",
        scale=1.0,
        bps_sampling_link_names=("link_0",),
        notes="Second PartNet bucket URDF; link_0 is the hinged handle used for BPS sampling.",
    ),
}
