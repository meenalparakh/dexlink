from dexlink.src.reference.common.categories.folding_common import (
    FoldingStage,
    build_selective_link_targets,
    load_folding_stage_samples,
)
from dexlink.src.reference.common.exp_schema import ObjectCategory, SourceTemplate

CATEGORY_NAME = "folding_carton"
URDF_PATH = "assets/procedural_mailer_folding/carton_folding/carton.urdf"
TRAJECTORY_PATH = (
    "assets/procedural_mailer_folding/carton_folding/output/trajectory.json"
)

CATEGORY = ObjectCategory(
    category=CATEGORY_NAME,
    display_name="folding carton",
    sections=("V",),
    required_data=("source_templates", "articulated_asset"),
    density_range=(500.0, 900.0),
    notes="Bimanual xHand folding of the finite-thickness tuck carton.",
)

# The hammer handle grasp is a compact power-hold proxy. Every target below
# restricts BPS sampling to the one or two cardboard links named for that hand.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="hammer",
        template_id="cardboard_holding",
        source_object_id="C22001",
        demo_id="109708ca1a",
        behavior="holding",
        phase="folding_contact",
        description="Hammer-style holding template reused for a cardboard panel.",
    )
}


def _operation(
    stage_id: str,
    label: str,
    trajectory_stage: str,
    left_links: tuple[str, ...],
    right_links: tuple[str, ...],
) -> tuple[FoldingStage, FoldingStage]:
    return (
        FoldingStage(
            stage_id=f"{stage_id}_ready",
            label=f"Ready: {label}",
            trajectory_stage=trajectory_stage,
            left_links=left_links,
            right_links=right_links,
            sample_at="start",
        ),
        FoldingStage(
            stage_id=f"{stage_id}_done",
            label=label,
            trajectory_stage=trajectory_stage,
            left_links=left_links,
            right_links=right_links,
            sample_at="end",
        ),
    )


STAGES = sum(
    (
        _operation(
            "deploy_receivers",
            "Deploy open-mouth receivers",
            "Deploy open-mouth receivers",
            ("bottom_socket_keeper",),
            ("top_socket_keeper",),
        ),
        _operation(
            "raise_left_wall",
            "Raise left wall",
            "Raise left wall",
            ("left_wall",),
            ("back_wall",),
        ),
        _operation(
            "raise_right_wall",
            "Raise right wall",
            "Raise right wall",
            ("back_wall",),
            ("right_wall",),
        ),
        _operation(
            "close_front_wall",
            "Close front wall",
            "Close front wall",
            ("front_wall",),
            ("right_wall",),
        ),
        _operation(
            "fold_glue_seam",
            "Fold offset glue seam",
            "Fold offset glue seam",
            ("glue_flap",),
            ("front_wall",),
        ),
        _operation(
            "fold_bottom_dust",
            "Fold bottom dust flaps",
            "Fold bottom dust flaps",
            ("bottom_left_dust",),
            ("bottom_right_dust",),
        ),
        _operation(
            "approach_bottom_receiver",
            "Approach bottom receiver",
            "Approach bottom receiver",
            ("bottom_lid",),
            ("back_wall",),
        ),
        _operation(
            "align_bottom_tab",
            "Align bottom tuck tab",
            "Align bottom tuck tab",
            ("bottom_tuck_tab",),
            ("bottom_lid",),
        ),
        _operation(
            "insert_bottom_tab",
            "Insert bottom tab",
            "Insert bottom tab",
            ("bottom_tuck_tab",),
            ("bottom_socket_keeper",),
        ),
        _operation(
            "fold_top_dust",
            "Fold top dust flaps",
            "Fold top dust flaps",
            ("top_left_dust",),
            ("top_right_dust",),
        ),
        _operation(
            "approach_top_receiver",
            "Approach top receiver",
            "Approach top receiver",
            ("top_lid",),
            ("back_wall",),
        ),
        _operation(
            "align_top_tab",
            "Align top tuck tab",
            "Align top tuck tab",
            ("top_tuck_tab",),
            ("top_lid",),
        ),
        _operation(
            "insert_top_tab",
            "Insert top tab",
            "Insert top tab",
            ("top_tuck_tab",),
            ("top_socket_keeper",),
        ),
    ),
    (),
)

STAGE_SAMPLES = load_folding_stage_samples(
    trajectory_path=TRAJECTORY_PATH,
    urdf_path=URDF_PATH,
    stages=STAGES,
)
TARGET_OBJECTS, TARGET_INDEX_BY_STAGE_SIDE = build_selective_link_targets(
    category=CATEGORY_NAME,
    asset_id="carton",
    urdf_path=URDF_PATH,
    samples=STAGE_SAMPLES,
)
