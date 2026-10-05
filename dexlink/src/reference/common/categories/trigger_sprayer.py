from dexlink.src.reference.common.exp_schema import (
    ArticulationRetargetSample,
    ObjectCategory,
    SourceTemplate,
    TargetObject,
    oakink_demo_path,
)


def _trigger_press_samples(
    pressed_value: float,
    joint_name: str = "joint_0",
) -> tuple[ArticulationRetargetSample, ...]:
    samples = [ArticulationRetargetSample("trigger_open", {joint_name: 0.0})]
    samples.extend(
        ArticulationRetargetSample(
            f"trigger_interp_{idx:02d}",
            {joint_name: float(pressed_value) * idx / 6.0},
        )
        for idx in range(1, 6)
    )
    samples.append(
        ArticulationRetargetSample("trigger_pressed", {joint_name: pressed_value})
    )
    return tuple(samples)


CATEGORY = ObjectCategory(
    category="trigger_sprayer",
    display_name="trigger sprayer",
    sections=("II", "III", "IV"),
    required_data=("source_templates", "target_meshes"),
    density_range=(700.0, 1400.0),
    notes="Reuse and robustness category; beyond-grasping task is a trigger-reaching/grasp sequence.",
)

# Sections: optional minimum III, ideal II/III/IV.
# Source templates needed:
# - Minimum: 2-3 phase templates if selected for the compositional/grasp-transition task in III.
# - Ideal: 2 grasp templates for II; 2-3 trigger-reaching sequence templates for III; 1 template for IV.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="trigger_sprayer",
        template_id="base_cylinder",
        source_object_id="A01008",
        demo_id="d4cec13792",
        behavior="lift",
        phase="grasp",
        description="grasp from base cylinder",
    ),
    2: SourceTemplate(
        category="trigger_sprayer",
        template_id="spraying_configuration",
        source_object_id="A01008",
        demo_id="c914fb458b",
        behavior="trigger_reach",
        phase="transition_2",
        description="spraying configuration for grasp-transition phase 2",
    ),
    3: SourceTemplate(
        category="trigger_sprayer",
        template_id="neck_grasp",
        source_object_id="A01008",
        demo_id="b26cac2ce0",
        behavior="trigger_reach",
        phase="transition_1",
        description="grasp from neck for grasp-transition phase 1",
    ),
    4: SourceTemplate(
        category="trigger_sprayer",
        template_id="spraying_configuration_a01027",
        source_object_id="A01027",
        demo_id="6c268d9aae",
        behavior="trigger_reach",
        phase="transition_2",
        description=(
            "Spraying configuration on A01027 used as the exact-target "
            "control for instance transfer."
        ),
    ),
}

# Target instances needed:
# - Minimum: 1 target if selected for III.
# - Ideal: 4 targets for II, one selected target for III/IV.
TARGET_OBJECTS = {
    1: TargetObject(
        category="trigger_sprayer",
        target_id="A01001",
        mesh_path=None,
        scale=1.0,
        target_demo_path=oakink_demo_path("trigger_sprayer", "A01001", "11aebd1062"),
        notes="OakInk target demo path; OakInkProcessor extracts object mesh/URDF from the demo.",
    ),
    2: TargetObject(
        category="trigger_sprayer",
        target_id="A01002",
        mesh_path=None,
        scale=1.0,
        target_demo_path=oakink_demo_path("trigger_sprayer", "A01002", "03268f0c14"),
        notes="OakInk target demo path; OakInkProcessor extracts object mesh/URDF from the demo.",
    ),
    3: TargetObject(
        category="trigger_sprayer",
        target_id="A01005",
        mesh_path=None,
        scale=1.0,
        target_demo_path=oakink_demo_path("trigger_sprayer", "A01005", "18f54cec1f"),
        notes="OakInk target demo path; OakInkProcessor extracts object mesh/URDF from the demo.",
    ),
    4: TargetObject(
        category="trigger_sprayer",
        target_id="A01010",
        mesh_path=None,
        scale=1.0,
        target_demo_path=oakink_demo_path("trigger_sprayer", "A01010", "00045cc573"),
        notes="OakInk target demo path; OakInkProcessor extracts object mesh/URDF from the demo.",
    ),
    5: TargetObject(
        category="trigger_sprayer",
        target_id="A01023",
        mesh_path=None,
        scale=1.0,
        target_demo_path=oakink_demo_path("trigger_sprayer", "A01023", "3ee06e9344"),
        notes="OakInk target demo path; OakInkProcessor extracts object mesh/URDF from the demo.",
    ),
    6: TargetObject(
        category="trigger_sprayer",
        target_id="A01026",
        mesh_path=None,
        scale=1.0,
        target_demo_path=oakink_demo_path("trigger_sprayer", "A01026", "01d073ebb9"),
        notes="OakInk target demo path; OakInkProcessor extracts object mesh/URDF from the demo.",
    ),
    7: TargetObject(
        category="trigger_sprayer",
        target_id="A01027",
        mesh_path=None,
        scale=1.0,
        target_demo_path=oakink_demo_path("trigger_sprayer", "A01027", "02c2620b55"),
        notes="OakInk target demo path; OakInkProcessor extracts object mesh/URDF from the demo.",
    ),
    8: TargetObject(
        category="trigger_sprayer",
        target_id="partnet_mobility",
        mesh_path="external_data/partnet/trigger_sprayer/mobility_single_articulation.urdf",
        scale=1.0,
        articulation_retarget_mode="per_articulation_retarget",
        articulation_retarget_samples=_trigger_press_samples(0.7853981634),
        notes="PartNet trigger sprayer URDF with only the trigger revolute joint articulated.",
    ),
    9: TargetObject(
        category="trigger_sprayer",
        target_id="partnet_mobility",
        mesh_path="external_data/partnet/trigger_sprayer_2/mobility_single_articulation.urdf",
        scale=1.0,
        articulation_retarget_mode="per_articulation_retarget",
        articulation_retarget_samples=_trigger_press_samples(0.3490658504),
        notes="PartNet trigger sprayer URDF with only the trigger revolute joint articulated.",
    ),
    10: TargetObject(
        category="trigger_sprayer",
        target_id="partnet_mobility",
        mesh_path="external_data/partnet/trigger_sprayer_3/mobility_single_articulation.urdf",
        scale=1.0,
        articulation_retarget_mode="per_articulation_retarget",
        articulation_retarget_samples=_trigger_press_samples(1.0471975512),
        notes="PartNet trigger sprayer URDF with only the trigger revolute joint articulated.",
    ),
    11: TargetObject(
        category="trigger_sprayer",
        target_id="primitive_trigger_sprayer_final",
        mesh_path="assets/trigger_sprayer_final/trigger_sprayer_v9.urdf",
        scale=1.0,
        articulation_retarget_mode="per_articulation_retarget",
        articulation_retarget_samples=_trigger_press_samples(
            0.523598776,
            joint_name="trigger_joint",
        ),
        notes="Primitive-geometry trigger sprayer URDF with one revolute trigger joint.",
    ),
    13: TargetObject(
        category="trigger_sprayer",
        target_id="A01001_scale_050",
        mesh_path=None,
        scale=1.0,
        target_demo_path=oakink_demo_path("trigger_sprayer", "A01001", "11aebd1062"),
        notes="V2_06 geometry-size variant of OakInk trigger sprayer target A01001; scale comes from transform scale.txt.",
    ),
    14: TargetObject(
        category="trigger_sprayer",
        target_id="A01001_scale_075",
        mesh_path=None,
        scale=1.0,
        target_demo_path=oakink_demo_path("trigger_sprayer", "A01001", "11aebd1062"),
        notes="V2_06 geometry-size variant of OakInk trigger sprayer target A01001; scale comes from transform scale.txt.",
    ),
    15: TargetObject(
        category="trigger_sprayer",
        target_id="A01001_scale_125",
        mesh_path=None,
        scale=1.0,
        target_demo_path=oakink_demo_path("trigger_sprayer", "A01001", "11aebd1062"),
        notes="V2_06 geometry-size variant of OakInk trigger sprayer target A01001; scale comes from transform scale.txt.",
    ),
    16: TargetObject(
        category="trigger_sprayer",
        target_id="A01001_scale_150",
        mesh_path=None,
        scale=1.0,
        target_demo_path=oakink_demo_path("trigger_sprayer", "A01001", "11aebd1062"),
        notes="V2_06 geometry-size variant of OakInk trigger sprayer target A01001; scale comes from transform scale.txt.",
    ),
    # 9: TargetObject(
    #     category="trigger_sprayer",
    #     target_id="ycb_windex_bottle",
    #     mesh_path="external_data/ycb/windex_bottle/textured.obj",
    #     scale=1.0,
    #     notes="YCB Windex bottle mesh.",
    # ),
}
