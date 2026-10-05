from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

ARCTIC_DEMO_PATH = "external_data/arctic/seqs/s01/notebook_use_01.npy"
ARCTIC_TARGET_OBJECT = "proc_notebook_5"
ARCTIC_START_TIMESTEP = 77
ARCTIC_END_TIMESTEP = 114
ARCTIC_SKIP = 1

CATEGORY = ObjectCategory(
    category="notebook",
    display_name="notebook",
    sections=("III",),
    required_data=("source_templates", "articulated_asset"),
    density_range=(500.0, 1000.0),
    notes="Ideal beyond-grasping articulation task.",
)

# Sections: ideal III articulation.
# Source templates needed:
# - Minimum: 0; not part of the minimum plan.
# - Ideal: opening sequence templates from ARCTIC notebook_use_01 frames 77-114.
SOURCE_TEMPLATES = {
    index: SourceTemplate(
        category="notebook",
        template_id=f"notebook_use_01_t{timestep:03d}",
        source_object_id="notebook_use_01",
        demo_id=f"t{timestep:03d}",
        behavior="open",
        phase="opening_sequence",
        description=f"ARCTIC notebook_use_01 opening frame {timestep}.",
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
# - Ideal: 1 articulated notebook asset for III.
TARGET_OBJECTS = {
    1: TargetObject(
        category="notebook",
        target_id=ARCTIC_TARGET_OBJECT,
        mesh_path="assets/proc_notebook/5/5.urdf",
        scale=1.0,
        notes=(
            "Procedural notebook target used by the ARCTIC notebook_use_01 source "
            "templates; process_arctic owns the source-target transform logic."
        ),
    ),
}
