"""Which tasks keep two objects active at once, and which hand supports each.

The default multi-object environment expects one active object per frame: one
object moves, one hand works on it, and the other hand, if there is one, steadies
something that never moves. A task where both hands lift their own object breaks
that assumption, so it opts in here and the environment factory hands it the
derived classes in `dexlink.src.rl.dual_active`.

Everything absent from this table keeps the default classes and the default
behaviour, which is what `tests/test_dual_active_world.py` holds it to.
"""

from __future__ import annotations

# Experiment id -> the object each hand supports for the whole episode.
DUAL_ACTIVE_HAND_OBJECTS: dict[str, dict[str, str]] = {
    "v3_08_lid_on_container": {"left": "container", "right": "lid"},
}


def dual_active_experiment_ids() -> frozenset[str]:
    """The experiments that opt in to two simultaneously active objects."""

    return frozenset(DUAL_ACTIVE_HAND_OBJECTS)


def experiment_id_for_exported_path(exported_path: str) -> str | None:
    """Name the experiment an exported reference belongs to, if it names one.

    An exported reference lives at `<pipeline root>/<experiment id>/exports/...`,
    so the component after the pipeline directory is the experiment. This reads
    it directly rather than through the preset helper, which keeps the opt-in
    table independent of the preset table: a task can be listed here before it
    has any RL settings at all.
    """

    if not exported_path:
        return None
    parts = [part for part in str(exported_path).split("/") if part]
    for index, part in enumerate(parts[:-1]):
        if part.endswith("_pipeline"):
            return parts[index + 1]
    return None


def hand_objects_for_exported_path(exported_path: str) -> dict[str, str] | None:
    """Return the hand-to-object assignment for a reference, or None by default.

    None is the answer for every task that has not opted in, and it is what makes
    the derived environment fall back to the default behaviour rather than guess.
    """

    experiment_id = experiment_id_for_exported_path(exported_path)
    if experiment_id is None:
        return None
    assignment = DUAL_ACTIVE_HAND_OBJECTS.get(experiment_id)
    if assignment is None:
        return None
    return dict(assignment)


def is_dual_active_exported_path(exported_path: str) -> bool:
    """Whether a reference belongs to a task that keeps two objects active."""

    return hand_objects_for_exported_path(exported_path) is not None
