from dexlink.src.reference.common.exp_schema import (
    ArticulationRetargetSample,
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)


def _scissor_samples(
    closed_value: float,
    full_open_value: float,
) -> tuple[ArticulationRetargetSample, ...]:
    open_value = closed_value + (full_open_value - closed_value) / 3.0
    samples = [ArticulationRetargetSample("scissor_closed", {"joint_0": closed_value})]
    samples.extend(
        ArticulationRetargetSample(
            f"scissor_interp_{idx:02d}",
            {"joint_0": closed_value + (open_value - closed_value) * idx / 6.0},
        )
        for idx in range(1, 6)
    )
    samples.append(ArticulationRetargetSample("scissor_open", {"joint_0": open_value}))
    return tuple(samples)


CATEGORY = ObjectCategory(
    category="scissor",
    display_name="scissor",
    sections=("III",),
    required_data=("source_templates", "articulated_assets"),
    density_range=(2500.0, 8000.0),
    notes="Optional beyond-grasping category for scissor use, handoff, and articulated target demos.",
)

# Sections: optional III use/articulation and handoff behavior.
# Source templates needed:
# - Minimum: 1 grasp/use template if selected for a qualitative demo.
# - Ideal: 2-3 grasp/use templates plus 1 handoff template.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="scissors",
        template_id="use_cfg",
        source_object_id="C35001",
        demo_id="15bb15b73b",
        behavior="use_cfg_motion",
        phase="use",
        description="finger configuration to use scissors after grasping",
    ),
    2: SourceTemplate(
        category="scissors",
        template_id="outside_handle_cover",
        source_object_id="C35001",
        demo_id="e2f276d8f3",
        behavior="grasp",
        phase="grasp",
        description="outside grasp covering the handles",
    ),
    3: SourceTemplate(
        category="scissors",
        template_id="use_cfg_alt",
        source_object_id="C35001",
        demo_id="eec38a53ac",
        behavior="use_cfg_motion",
        phase="use",
        description="alternate use configuration",
    ),
    4: SourceTemplate(
        category="scissors",
        template_id="handoff_point_side",
        source_object_id="C35001",
        demo_id="f61d14a5c0",
        behavior="handoff",
        phase="grasp",
        description="handoff configuration grasped from the point side",
    ),
    5: SourceTemplate(
        category="scissors",
        template_id="outside_other_instance",
        source_object_id="Y35037",
        demo_id="c9065998f7",
        behavior="grasp",
        phase="grasp",
        description="outside grasp for a second scissor instance",
    ),
}

# Target instances needed:
# - Minimum: 1 articulated scissor asset if selected for III.
# - Ideal: 3-4 articulated scissor assets for visual/geometry variation.
TARGET_OBJECTS = {
    1: TargetObject(
        category="scissor",
        target_id="partnet_scissors1",
        mesh_path="external_data/partnet/scissors1/mobility.urdf",
        scale=1.0,
        articulation_retarget_mode="per_articulation_retarget",
        articulation_retarget_samples=_scissor_samples(-0.6094689748, 0.8230972752),
        notes="PartNet articulated scissor URDF.",
    ),
    2: TargetObject(
        category="scissor",
        target_id="partnet_scissors2",
        mesh_path="external_data/partnet/scissors2/mobility.urdf",
        scale=1.0,
        articulation_retarget_mode="per_articulation_retarget",
        articulation_retarget_samples=_scissor_samples(-1.4388494353, 0.0),
        notes="PartNet articulated scissor URDF.",
    ),
    3: TargetObject(
        category="scissor",
        target_id="partnet_scissors3",
        mesh_path="external_data/partnet/scissors3/mobility.urdf",
        scale=1.0,
        articulation_retarget_mode="per_articulation_retarget",
        articulation_retarget_samples=_scissor_samples(0.0, 0.8545132018),
        notes="PartNet articulated scissor URDF.",
    ),
    4: TargetObject(
        category="scissor",
        target_id="partnet_scissors4",
        mesh_path="external_data/partnet/scissors4/mobility.urdf",
        scale=1.0,
        articulation_retarget_mode="per_articulation_retarget",
        articulation_retarget_samples=_scissor_samples(-0.2450442270, 0.7853981634),
        notes="PartNet articulated scissor URDF.",
    ),
}
