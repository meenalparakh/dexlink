from dexlink.src.reference.common.exp_schema import ObjectCategory

TBD = "TBD"

CATEGORY = ObjectCategory(
    category="table_setting_object",
    display_name="table-setting object",
    sections=("III", "V"),
    required_data=("source_templates", "target_meshes", "real_object_optional"),
    density_range=(500.0, 1400.0),
    notes="Placeholder for the three dinner-table setup objects; split into concrete categories once chosen.",
)

# Sections: ideal III and optional V dinner-table long-horizon task.
# Source templates needed:
# - Minimum: 0 unless dinner-table setup is selected as the minimum real-world long-horizon demo.
# - Ideal: 1-3 task templates per chosen table object, depending on pick/place/contact phases.

SOURCE_TEMPLATES = {
    # cleaning up a dinner table
    # dinner table: taking out a plate from a rack and laying it down.
    # placing a cup that was inverted next to the plate
    # lifting a bowl and placing it next to the plate
    # lifting a spoon from a holder and putting it in bowl.
}

# Target instances needed:
# - Minimum: 3 real or simulated table-setting objects if used for the minimum dinner-table demo.
# - Ideal: concrete target entries for each selected table object; split this placeholder into specific categories once chosen.
TARGET_OBJECTS = {}


# Concrete long-horizon task setups. Existing entries use
# (category, source_template_index, target_object_index). Missing entries are
# deliberately marked TBD so they fail visibly during spec generation until
# you fill the corresponding category files.
LONG_HORIZON_TASK_SETUPS = {
    "simple_breakfast_place": {
        "goal": "Set one breakfast place with a bowl, a drinking vessel, a spoon in the bowl, and fruit on the side.",
        "sections": ("III", "V"),
        "objects": {
            "bowl": {
                "source_target": ("bowl", 2, 6),
                "role": "place bowl at center of setting",
                "notes": "Uses bowl full-support source and YCB bowl target.",
            },
            "drinking_vessel": {
                "source_target": ("cup", 1, 1),
                "role": "place cup beside bowl",
                "notes": "Can be swapped for mug with ('mug', 3, 1) if cup is harder.",
            },
            "spoon": {
                "source_target": ("spoon", TBD, TBD),
                "role": "place spoon inside bowl",
                "fill": "Fill the spoon category with one source template and one target.",
            },
            "fruit": {
                "source_target": ("fruit", TBD, TBD),
                "role": "place fruit on the side of the bowl",
                "fill": "Fill the fruit category with one source template and one target.",
            },
        },
        "template_sequence": (
            "bowl.pick",
            "bowl.place_at_setting_center",
            "drinking_vessel.pick",
            "drinking_vessel.place_to_right_of_bowl",
            "spoon.pick",
            "spoon.place_inside_bowl",
            "fruit.pick",
            "fruit.place_to_side_of_bowl",
        ),
        "missing": (
            "Spoon source/target category entries.",
            "Fruit source/target category entries.",
            "Scene-level target poses for the place setting.",
        ),
    },
    "prepare_toothbrushing_cup": {
        "goal": "Prepare a toothbrush holder by placing a toothbrush into a cup.",
        "sections": ("III",),
        "objects": {
            "cup_upright_or_inverted": {
                "source_target": ("cup", 1, 1),
                "role": "pick and place cup in holder position",
                "notes": "Use cup source 2 instead when starting from an inverted cup.",
            },
            "cup_bottom_grasp": {
                "source_target": ("cup", 2, 1),
                "role": "alternate grasp for inverted-cup pickup",
            },
            "toothbrush": {
                "source_target": ("toothbrush", TBD, TBD),
                "role": "pick toothbrush and place it into the cup",
                "fill": "Fill the toothbrush category with one horizontal-pick source template and one target mesh.",
            },
        },
        "template_sequence": (
            "cup.pick",
            "cup.place_or_invert_at_holder_location",
            "toothbrush.pick_horizontal",
            "toothbrush.insert_into_cup",
        ),
        "missing": (
            "Toothbrush source template index.",
            "Toothbrush target object index.",
            "Cup insertion pose / toothbrush final pose relative to cup.",
        ),
    },
    "tidy_desk_setup": {
        "goal": "Organize a desk by placing writing, compute, and drink objects into semantic zones.",
        "sections": ("III", "V"),
        "objects": {
            "pen": {
                "source_target": ("marker", 2, 1),
                "role": "place pen in writing zone",
                "notes": "Use ('marker', 1, target) if a simpler lift grasp is preferred.",
            },
            "cup": {
                "source_target": ("cup", 1, 1),
                "role": "place cup away from electronics",
            },
            "mouse": {
                "source_target": ("mouse", TBD, TBD),
                "role": "place mouse on mousepad zone",
                "fill": "Fill the mouse category with one source template and one target.",
            },
        },
        "template_sequence": (
            "mouse.pick",
            "mouse.place_on_mousepad",
            "pen.pick",
            "pen.place_in_writing_zone",
            "cup.pick",
            "cup.place_in_drink_zone",
        ),
        "missing": (
            "Mouse source/target category entries.",
            "Scene-level target zones for dock, mousepad, writing area, and drink area.",
        ),
    },
}
