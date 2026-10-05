from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="faucet",
    display_name="faucet",
    sections=("III",),
    required_data=("source_templates", "articulated_asset"),
    density_range=(1000.0, 8000.0),
    notes="Articulated faucet switch task with a fixed spout and movable switch.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="toothbrush",
        template_id="top_proxy",
        source_object_id="C25001",
        demo_id="94a15def16",
        behavior="articulation_proxy",
        phase="grasp",
        description="Toothbrush proxy grasp used for faucet switch articulation.",
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="faucet",
        target_id="partnet_faucet",
        mesh_path="external_data/partnet/faucet/mobility.urdf",
        scale=1.0,
        bps_sampling_link_names=("link_0",),
        notes="PartNet faucet URDF; link_0 is the hinged switch used for BPS sampling.",
    ),
}
