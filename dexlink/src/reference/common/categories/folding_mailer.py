from dexlink.src.reference.common.categories.folding_common import (
    FoldingStage,
    build_selective_link_targets,
    load_folding_stage_samples,
)
from dexlink.src.reference.common.exp_schema import ObjectCategory, SourceTemplate

CATEGORY_NAME = "folding_mailer"
URDF_PATH = "assets/procedural_mailer_folding/mailer_box/mailer.urdf"
TRAJECTORY_PATH = "assets/procedural_mailer_folding/mailer_box/output/trajectory.json"

CATEGORY = ObjectCategory(
    category=CATEGORY_NAME,
    display_name="folding hinged-lid mailer",
    sections=("V",),
    required_data=("source_templates", "articulated_asset"),
    density_range=(350.0, 650.0),
    notes="Bimanual xHand folding of the hinged-lid rollover mailer.",
)

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
            "raise_end_walls",
            "Raise front wall and rear lid spine",
            "Raise front wall and rear lid spine",
            ("rear_wall",),
            ("front_wall",),
        ),
        _operation(
            "fold_corner_ears",
            "Fold four corner ears inward",
            "Fold four corner ears inward",
            ("rear_corner_a", "rear_corner_b"),
            ("front_corner_a", "front_corner_b"),
        ),
        _operation(
            "raise_side_walls",
            "Raise tray side walls",
            "Raise the tray side walls",
            ("left_outer",),
            ("right_outer",),
        ),
        _operation(
            "fold_rollover_strips",
            "Fold rollover clearance strips",
            "Fold the two rollover clearance strips",
            ("left_roll_margin",),
            ("right_roll_margin",),
        ),
        _operation(
            "prebend_locking_strips",
            "Pre-bend lower locking strips",
            "Pre-bend the lower locking strips",
            ("left_lock_0", "left_lock_1"),
            ("right_lock_0", "right_lock_1"),
        ),
        _operation(
            "align_floor_tabs",
            "Roll liners down and align floor tabs",
            "Roll liners down and align the floor tabs",
            ("left_liner",),
            ("right_liner",),
        ),
        _operation(
            "insert_floor_tabs",
            "Insert four locking tabs into base slots",
            "Insert four locking tabs into base slots",
            ("left_lock_0", "left_lock_1"),
            ("right_lock_0", "right_lock_1"),
        ),
        _operation(
            "deploy_front_receiver",
            "Deploy front receiver spacing ears",
            "Deploy front receiver spacing ears",
            ("front_socket_margin",),
            ("front_wall",),
        ),
        _operation(
            "return_front_keeper",
            "Return front keeper into tray",
            "Return the front keeper into the tray",
            ("front_keeper",),
            ("front_wall",),
        ),
        _operation(
            "fold_lid_wings",
            "Fold lid side wings",
            "Fold the lid side wings",
            ("lid_left_wing",),
            ("lid_right_wing",),
        ),
        _operation(
            "prefold_front_tuck",
            "Pre-fold front tuck margin and flap",
            "Pre-fold the front tuck margin and flap",
            ("lid_front_tab",),
            ("lid",),
        ),
        _operation(
            "lower_lid",
            "Lower lid into side channels",
            "Lower the lid and enter the side channels",
            ("lid_left_wing",),
            ("lid_right_wing",),
        ),
        _operation(
            "insert_front_flap",
            "Insert front flap into receiver",
            "Insert the front flap into its open receiver",
            ("lid_front_tab",),
            ("front_keeper",),
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
    asset_id="mailer",
    urdf_path=URDF_PATH,
    samples=STAGE_SAMPLES,
)
