from __future__ import annotations

from pathlib import Path

from dexlink.src.reference.task_io import load_task_spec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPOSITORY_ROOT / "tasks"


def test_every_public_task_is_valid_and_uniquely_named() -> None:
    task_paths = sorted(TASK_ROOT.rglob("*.yaml"))
    assert len(task_paths) == 586

    experiment_ids: dict[str, Path] = {}
    for task_path in task_paths:
        task = load_task_spec(task_path)
        experiment_id = task["experiment_id"]
        assert experiment_id not in experiment_ids, (
            f"Duplicate experiment_id {experiment_id!r}: "
            f"{experiment_ids.get(experiment_id)} and {task_path}"
        )
        experiment_ids[experiment_id] = task_path

        trajectory = task["trajectory"]
        assert isinstance(trajectory, dict)
        assert isinstance(trajectory.get("phases"), list)
        assert trajectory["phases"]
        assert str(trajectory["output_motion_path"]).startswith("outputs/")
        assert str(trajectory["exported_rl_path"]).startswith("outputs/")


def test_public_tasks_have_no_unresolved_or_private_paths() -> None:
    forbidden = ("TBD", "/" + "home/", "/" + "mnt/", "pvl" + "-exptrack")
    for task_path in TASK_ROOT.rglob("*.yaml"):
        task_text = task_path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in task_text, f"Found {token!r} in {task_path}"
