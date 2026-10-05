from dexlink.src.reference.common.exp_schema import (
    ExperimentScale,
    SectionCollectionRequirement,
)

SECTION_COLLECTION_REQUIREMENTS: dict[
    ExperimentScale, dict[str, SectionCollectionRequirement]
] = {
    "minimum": {
        "I": SectionCollectionRequirement(
            section="I",
            purpose="template-conditioned behavior and relational fidelity",
            task_families=("grasp+lift", "grasp+lift+tilt"),
            categories=("mug", "bowl", "marker_or_bottle"),
            source_templates_per_category={"lift": 2, "tilt": 1},
            target_instances_per_category=1,
            notes=(
                "Use one target object per category.",
                "The same first target can also seed Section II/IV target pools.",
                "Include no-template and reward-only/no-scaffold comparisons on one representative category.",
            ),
        ),
        "II": SectionCollectionRequirement(
            section="II",
            purpose="cross-geometry template reuse",
            task_families=("grasp retention under gravity/noise",),
            categories=("mug", "bowl"),
            source_templates_per_category={"grasp": 1},
            target_instances_per_category=3,
            notes=(
                "Collect target pools, not new source templates, for each category.",
                "Compare raw retargeting against RL-refined grasps for at least one category.",
            ),
        ),
        "III": SectionCollectionRequirement(
            section="III",
            purpose="breadth beyond static grasping",
            task_families=(
                "post-grasp manipulation",
                "articulation",
                "compositional/grasp-transition",
            ),
            categories=("marker_or_bowl", "drawer", "trigger_sprayer_or_bowl"),
            source_templates_per_category={"phase_templates": 2},
            target_instances_per_category=1,
            notes=(
                "Collect one target setup per task.",
                "Use two to three templates only when the hand-object relation changes by phase.",
            ),
        ),
        "IV": SectionCollectionRequirement(
            section="IV",
            purpose="robustness under imperfect templates and geometry",
            task_families=("grasp retention under pose/mesh noise",),
            categories=("mug", "bowl", "marker"),
            source_templates_per_category={"grasp": 1},
            target_instances_per_category=1,
            notes=(
                "Reuse Section II grasp source templates.",
                "Run three pose-noise levels and three mesh-noise levels.",
                "Compare raw/scaffolded execution against RL-refined execution at each noise level.",
            ),
        ),
        "V": SectionCollectionRequirement(
            section="V",
            purpose="real-world viability",
            task_families=("lift", "optional lift+tilt"),
            categories=("mug", "bowl", "bottle_or_marker"),
            source_templates_per_category={"lift": 1, "tilt": 1},
            target_instances_per_category=1,
            notes=(
                "Keep this as a small real-object validation set.",
                "Prioritize success and qualitative fidelity over broad coverage.",
            ),
        ),
    },
    "ideal": {
        "I": SectionCollectionRequirement(
            section="I",
            purpose="broad template-conditioned behavior evidence",
            task_families=("grasp+lift", "grasp+lift+tilt"),
            categories=("mug", "bowl", "headphones", "screwdriver", "marker"),
            source_templates_per_category={"lift": 3, "tilt": 2},
            target_instances_per_category=1,
            notes=(
                "This is source-template heavy but target-light.",
                "One target instance per category is enough for the template-conditioning claim.",
            ),
        ),
        "II": SectionCollectionRequirement(
            section="II",
            purpose="strong cross-geometry template reuse",
            task_families=("grasp retention under gravity/noise",),
            categories=(
                "mug",
                "bowl",
                "trigger_sprayer",
                "bottle_or_cup",
                "marker_or_screwdriver",
            ),
            source_templates_per_category={"grasp": 2},
            target_instances_per_category=4,
            notes=(
                "This is target-pool heavy: 5 categories x 2 templates x 4 targets = 40 grasp refinements.",
                "Do retargeting objective ablations on one category.",
            ),
        ),
        "III": SectionCollectionRequirement(
            section="III",
            purpose="broad manipulation, articulation, and long-horizon capability suite",
            task_families=(
                "sprayer sequence",
                "bowl support regrasp",
                "marker reorientation",
                "drawer/notebook/stapler articulation",
                "long-horizon composition",
            ),
            categories=(
                "trigger_sprayer",
                "bowl",
                "marker",
                "drawer",
                "notebook",
                "stapler",
                "eraser",
                "cup",
                "toothbrush",
            ),
            source_templates_per_category={"phase_templates": 2},
            target_instances_per_category=1,
            notes=(
                "Breadth matters more than uniform metrics in this section.",
                "Collect extra phase templates only when the interaction mode changes substantially.",
            ),
        ),
        "IV": SectionCollectionRequirement(
            section="IV",
            purpose="robustness trends across perturbation types",
            task_families=("grasp retention under pose/mesh noise",),
            categories=("mug", "bowl", "trigger_sprayer"),
            source_templates_per_category={"grasp": 1},
            target_instances_per_category=1,
            notes=(
                "Reuse Section II source templates and one selected target per category.",
                "Run before-vs-after RL at every noise level.",
            ),
        ),
        "V": SectionCollectionRequirement(
            section="V",
            purpose="real-world transfer across increasing complexity",
            task_families=("lift", "lift+tilt", "optional long-horizon"),
            categories=("mug", "bowl", "bottle_or_cup", "drawer_plus_table_objects"),
            source_templates_per_category={"task_template": 1},
            target_instances_per_category=1,
            notes=(
                "Minimum real-world target is two lift objects, one lift+tilt task, and one long-horizon demo.",
            ),
        ),
    },
}
