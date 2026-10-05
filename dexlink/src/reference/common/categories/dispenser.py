from dexlink.src.reference.common.exp_schema import (
    ArticulationRetargetSample,
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

DISPENSER_TARGET_SCALE = 0.12
DISPENSER_PUMP_UP = 0.0
DISPENSER_PUMP_DOWN = 0.189 * DISPENSER_TARGET_SCALE
FINAL_DISPENSER_PUMP_UP = 0.0
FINAL_DISPENSER_PUMP_DOWN = -0.01


def _dispenser_press_samples(
    joint_name: str = "joint_3",
    pump_up: float = DISPENSER_PUMP_UP,
    pump_down: float = DISPENSER_PUMP_DOWN,
) -> tuple[ArticulationRetargetSample, ...]:
    samples = [ArticulationRetargetSample("pump_up", {joint_name: pump_up})]
    samples.extend(
        ArticulationRetargetSample(
            f"pump_interp_{idx:02d}",
            {joint_name: pump_up + (pump_down - pump_up) * idx / 6.0},
        )
        for idx in range(1, 6)
    )
    samples.append(ArticulationRetargetSample("pump_down", {joint_name: pump_down}))
    return tuple(samples)


CATEGORY = ObjectCategory(
    category="dispenser",
    display_name="dispenser",
    sections=("III", "V"),
    required_data=("source_templates", "articulated_asset"),
    density_range=(700.0, 1400.0),
    notes="PartNet dispenser target with side-grasp and top-spraying source templates.",
)

SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="lotion_pump",
        template_id="spraying_configuration",
        source_object_id="A02014",
        demo_id="e758576bbf",
        behavior="press",
        phase="grasp",
        description="top spraying configuration for dispenser pressing.",
    ),
    2: SourceTemplate(
        category="lotion_pump",
        template_id="side_grasp",
        source_object_id="A02015",
        demo_id="8aa3f37a23",
        behavior="grasp",
        phase="grasp",
        description="side grasp candidate for holding a dispenser.",
    ),
}

TARGET_OBJECTS = {
    1: TargetObject(
        category="dispenser",
        target_id="partnet_dispenser_1",
        mesh_path="external_data/partnet/dispenser_1/mobility_single_articulation.urdf",
        scale=1.0,
        articulation_retarget_mode="per_articulation_retarget",
        articulation_retarget_samples=_dispenser_press_samples(),
        notes="PartNet dispenser URDF with only joint_3, the top pipe prismatic joint, articulated.",
    ),
    2: TargetObject(
        category="dispenser",
        target_id="primitive_dispenser_final",
        mesh_path="assets/dispenser_final/dispenser_variation_prismatic_v5.urdf",
        scale=1.0,
        articulation_retarget_mode="per_articulation_retarget",
        articulation_retarget_samples=_dispenser_press_samples(
            joint_name="neck_to_head_prismatic",
            pump_up=FINAL_DISPENSER_PUMP_UP,
            pump_down=FINAL_DISPENSER_PUMP_DOWN,
        ),
        notes="Primitive-geometry dispenser URDF with a prismatic pump joint.",
    ),
}
