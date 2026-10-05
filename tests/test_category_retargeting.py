from __future__ import annotations

from pathlib import Path

from dexlink.scripts.retargeting.run_task_category_retargetings import (
    collect_category_retarget_requests,
)
from dexlink.src.reference.category_paths import category_retarget_run_dir
from dexlink.src.reference.task_io import load_task_spec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPOSITORY_ROOT / "tasks"


def _requests(relative_path: str):
    spec = load_task_spec(TASK_ROOT / relative_path)
    return collect_category_retarget_requests(spec, "xhand")


def test_collects_simple_scene_object_retarget() -> None:
    requests = _requests(
        "v3_03_reusability/v3_03_reuse_bottle_side_scale_070.yaml"
    )
    assert len(requests) == 1
    request = requests[0]
    assert request.category == "bottle"
    assert request.source_index == 1
    assert request.target_index == 16
    assert request.hand_side == "right"
    assert request.articulation_sample_id is None


def test_collects_all_articulation_samples_without_duplicates() -> None:
    requests = _requests(
        "v2_03_beyond_grasp_div/v2_03_dispenser_press_twice.yaml"
    )
    assert len(requests) == 7
    assert {request.articulation_sample_id for request in requests} == {
        "pump_up",
        "pump_interp_01",
        "pump_interp_02",
        "pump_interp_03",
        "pump_interp_04",
        "pump_interp_05",
        "pump_down",
    }


def test_collects_bimanual_retargets() -> None:
    requests = _requests(
        "v3_08_lid_on_container/v3_08_lid_on_container.yaml"
    )
    assert {(request.category, request.source_index, request.target_index) for request in requests} == {
        ("bottle", 4, 20),
        ("top_lid", 1, 1),
    }


def test_all_public_tasks_have_unambiguous_category_retargets() -> None:
    for task_path in TASK_ROOT.rglob("*.yaml"):
        spec = load_task_spec(task_path)
        requests = collect_category_retarget_requests(spec, "xhand")
        assert requests, task_path
        identities = [(request.hand_type, request.job_id) for request in requests]
        assert len(identities) == len(set(identities)), task_path


def test_runner_and_reference_builder_share_output_path_contract() -> None:
    request = _requests(
        "v3_03_reusability/v3_03_reuse_bottle_side_scale_070.yaml"
    )[0]
    expected = (
        REPOSITORY_ROOT
        / "outputs"
        / "experiment_category_retargetings"
        / "xhand"
        / "bottle"
        / request.job_id
    )
    assert request.run_dir(REPOSITORY_ROOT) == expected
    assert category_retarget_run_dir(
        REPOSITORY_ROOT,
        "xhand",
        request.category,
        request.source_index,
        request.target_index,
    ) == expected
