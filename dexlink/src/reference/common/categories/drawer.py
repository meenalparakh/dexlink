from dexlink.src.reference.common.exp_schema import (
    ObjectCategory,
    SourceTemplate,
    TargetObject,
)

CATEGORY = ObjectCategory(
    category="drawer",
    display_name="drawer",
    sections=("III", "V"),
    required_data=("source_templates", "articulated_asset", "real_object_optional"),
    density_range=(1200.0, 1500.0),
    notes="Articulation task and optional real-world long-horizon setup.",
)

# Sections: minimum/ideal III articulation, optional ideal V long-horizon real-world setup.
# Source templates needed:
# - Minimum: 2-3 phase templates for drawer opening in III.
# - Ideal: 2-3 articulation templates for III; 1 real-world-compatible template sequence if used in V.
SOURCE_TEMPLATES = {
    1: SourceTemplate(
        category="toothbrush",
        template_id="drawer_proxy_grasp",
        source_object_id="C25001",
        demo_id="7c163ff232",
        behavior="articulation_proxy",
        phase="grasp",
        description="Proxy grasp template currently selected for drawer articulation setup.",
    ),
}

# Target instances needed:
# - Minimum: 1 articulated drawer asset for III.
# - Ideal: 1 articulated drawer asset for III and optionally 1 real drawer setup for V.
TARGET_OBJECTS = {
    1: TargetObject(
        category="drawer",
        target_id="drawer_58",
        mesh_path="assets/drawers/58/drawer.urdf",
        scale=1.0,
        bps_sampling_link_names=("handle_front",),
        notes="Articulated drawer URDF.",
    ),
}
