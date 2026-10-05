from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

ARCTIC_DEMO_PATH = "external_data/arctic/seqs/s01/ketchup_use_01.npy"
ARCTIC_TARGET_OBJECT = "banana_handover"
ARCTIC_START_TIMESTEP = 215
ARCTIC_END_TIMESTEP = 245
ARCTIC_SKIP = 1
ARCTIC_TIMESTEPS = tuple(range(ARCTIC_START_TIMESTEP, ARCTIC_END_TIMESTEP, ARCTIC_SKIP))

CATEGORY = ObjectCategory(
    category="banana_handover",
    display_name="banana handover",
    sections=("III",),
    required_data=("source_templates", "target_meshes"),
    density_range=(600.0, 1000.0),
    notes="ARCTIC ketchup handover/opening motion retargeted onto a rigid banana OBJ.",
)


def _source_templates() -> dict[int, SourceTemplate]:
    templates: dict[int, SourceTemplate] = {}
    index = 1
    for hand in ("right", "left"):
        for timestep in ARCTIC_TIMESTEPS:
            templates[index] = SourceTemplate(
                category="banana_handover",
                template_id=f"ketchup_use_01_{hand}_t{timestep:03d}",
                source_object_id="ketchup_use_01",
                demo_id=f"{hand}_t{timestep:03d}",
                behavior="handover",
                phase="handover_sequence",
                description=f"ARCTIC ketchup_use_01 {hand}-hand frame {timestep}.",
                dataset="arctic",
                demo_path_override=ARCTIC_DEMO_PATH,
                arctic_target_object=ARCTIC_TARGET_OBJECT,
                arctic_timestep=timestep,
                arctic_start_timestep=ARCTIC_START_TIMESTEP,
                arctic_end_timestep=ARCTIC_END_TIMESTEP,
                arctic_skip=ARCTIC_SKIP,
                arctic_hand=hand,
            )
            index += 1
    return templates


SOURCE_TEMPLATES = _source_templates()

TARGET_OBJECTS = {
    1: TargetObject(
        category="banana_handover",
        target_id=ARCTIC_TARGET_OBJECT,
        mesh_path="assets/banana/banana.obj",
        scale=1.5,
        notes=(
            "Rigid external OBJ target for ARCTIC ketchup_use_01. "
            "process_arctic applies any saved category transform to the OBJ before "
            "wrapping it in a fixed URDF."
        ),
    ),
}
