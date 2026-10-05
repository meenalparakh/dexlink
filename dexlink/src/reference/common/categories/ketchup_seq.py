from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

ARCTIC_DEMO_PATH = "external_data/arctic/seqs/s01/ketchup_use_01.npy"
ARCTIC_TARGET_OBJECT = "proc_ketchup_2"
ARCTIC_START_TIMESTEP = 219
ARCTIC_END_TIMESTEP = 240
ARCTIC_SKIP = 3

CATEGORY = ObjectCategory(
    category="ketchup",
    display_name="ketchup",
    sections=("III",),
    required_data=("source_templates", "articulated_asset"),
    density_range=(600.0, 1000.0),
    notes="Ideal beyond-grasping articulation task.",
)

# Sections: ideal III articulation.
# Source templates needed:
# - Minimum: 0; not part of the minimum plan.
# - Ideal: opening sequence templates from ARCTIC ketchup_use_01 frames 219-240,
#   sampled with skip 3.
SOURCE_TEMPLATES = {
    index: SourceTemplate(
        category="ketchup",
        template_id=f"ketchup_use_01_t{timestep:03d}",
        source_object_id="ketchup_use_01",
        demo_id=f"t{timestep:03d}",
        behavior="open",
        phase="opening_sequence",
        description=f"ARCTIC ketchup_use_01 opening frame {timestep}.",
        dataset="arctic",
        demo_path_override=ARCTIC_DEMO_PATH,
        arctic_target_object=ARCTIC_TARGET_OBJECT,
        arctic_timestep=timestep,
        arctic_start_timestep=ARCTIC_START_TIMESTEP,
        arctic_end_timestep=ARCTIC_END_TIMESTEP,
        arctic_skip=ARCTIC_SKIP,
    )
    for index, timestep in enumerate(
        range(ARCTIC_START_TIMESTEP, ARCTIC_END_TIMESTEP, ARCTIC_SKIP),
        start=1,
    )
}

# Target instances needed:
# - Minimum: 0; not part of the minimum plan.
# - Ideal: 1 articulated ketchup asset for III.
TARGET_OBJECTS = {
    1: TargetObject(
        category="ketchup",
        target_id=ARCTIC_TARGET_OBJECT,
        mesh_path="assets/proc_ketchup/2/2.urdf",
        scale=1.0,
        notes=(
            "Procedural ketchup target used by the ARCTIC ketchup_use_01 source "
            "templates; process_arctic owns the source-target transform logic."
        ),
    ),
}
