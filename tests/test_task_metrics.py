from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dexlink.src.rl.config.base import EnvConfig, TaskMetricCfg, load_experiment_config
from dexlink.src.rl.config.seq_oakink_final_experiment_presets import (
    OUR_METHOD_PRESET_NAME,
)
from dexlink.src.rl.multi_object.task_manager import MultiObjectTaskManager
from dexlink.utils.rl.final_rollout_metrics import FinalRolloutMetrics
from dexlink.src.rl.world.contact_info_sim import (
    get_expanded_mask_and_force_sign,
    get_filtered_contact_force_from_mask,
)


class FakeObject:
    def __init__(
        self,
        *,
        pos: torch.Tensor,
        arti: torch.Tensor,
        quat: torch.Tensor | None = None,
        mass_kg: float = 1.0,
    ) -> None:
        self.pos = pos
        self.arti = arti
        self.quat = (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=pos.dtype).repeat(
                pos.shape[0], 1
            )
            if quat is None
            else quat
        )
        self.env_link_mass = np.full((pos.shape[0], 1), mass_kg, dtype=np.float32)

    def get_ground_truth_info(self) -> dict[str, torch.Tensor]:
        return {"pos": self.pos, "quat": self.quat, "arti": self.arti}


class FakeHand:
    def __init__(self) -> None:
        self.robot = object()
        self.wrist_link_name = "wrist"


def test_contact_force_aggregation_returns_force_on_requested_entity_a() -> None:
    n_contacts = torch.tensor([2])
    pair_a = torch.tensor([[10], [20]])
    pair_b = torch.tensor([[20], [10]])
    requested_a = torch.tensor([10])
    requested_b = torch.tensor([20])
    raw_force_on_stored_link_b = torch.tensor([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])

    mask, force_sign = get_expanded_mask_and_force_sign(
        n_contacts,
        pair_a,
        pair_b,
        requested_a,
        requested_b,
    )
    force_on_requested_a = get_filtered_contact_force_from_mask(
        raw_force_on_stored_link_b,
        mask,
        force_sign,
    )

    assert force_on_requested_a.shape == (1, 1, 1, 3)
    assert force_on_requested_a[0, 0, 0].tolist() == [1.0, 0.0, 0.0]


class FakePhysics:
    def __init__(
        self,
        *,
        object_entities: dict[str, FakeObject],
        contact_signal: torch.Tensor | None = None,
        source_contact_signal: torch.Tensor | None = None,
        receiving_net_force: torch.Tensor | None = None,
        source_net_force: torch.Tensor | None = None,
        hammer_impact: torch.Tensor | None = None,
        hammer_contact: torch.Tensor | None = None,
        wrist_position: torch.Tensor | None = None,
        wrist_quaternion: torch.Tensor | None = None,
        scene_table_top_height: float = 0.75,
    ) -> None:
        self.object_entities = object_entities
        self.scene_entities = {"left": FakeHand(), "right": FakeHand()}
        self.contact_signals = {
            "left": contact_signal,
            "right": source_contact_signal,
        }
        self.net_forces = {
            "left": receiving_net_force,
            "right": source_net_force,
        }
        self.hammer_impact = hammer_impact
        self.hammer_contact = hammer_contact
        self.wrist_position = wrist_position
        self.wrist_quaternion = wrist_quaternion
        self.scene_table_top_height = scene_table_top_height

    def get_scene_table_top_height(self) -> float:
        return self.scene_table_top_height

    def get_hand_object_binary_contact_signal(
        self,
        hand: object,
        object_entity: FakeObject,
        *,
        threshold: float,
    ) -> torch.Tensor:
        assert object_entity is self.object_entities["banana"]
        assert threshold == pytest.approx(1.0e-4)
        matching_sides = [
            side for side, entity in self.scene_entities.items() if hand is entity
        ]
        if len(matching_sides) != 1:
            raise ValueError(
                f"Expected exactly one matching hand, got {matching_sides}."
            )
        contact_signal = self.contact_signals[matching_sides[0]]
        if contact_signal is None:
            raise ValueError(
                f"Expected a contact signal for {matching_sides[0]!r} hand."
            )
        return contact_signal

    def get_hand_object_contact_support_diagnostics(
        self,
        hand: object,
        object_entity: FakeObject,
        *,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contact_signal = self.get_hand_object_binary_contact_signal(
            hand,
            object_entity,
            threshold=threshold,
        )
        matching_sides = [
            side for side, entity in self.scene_entities.items() if hand is entity
        ]
        if len(matching_sides) != 1:
            raise ValueError(
                f"Expected exactly one matching hand, got {matching_sides}."
            )
        net_force = self.net_forces[matching_sides[0]]
        if net_force is None:
            raise ValueError(f"Expected net force for {matching_sides[0]!r} hand.")
        return contact_signal, net_force

    def get_hammer_peg_contact_impact(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hammer_impact is None or self.hammer_contact is None:
            raise ValueError("Expected hammer impact data for this test.")
        return self.hammer_impact, self.hammer_contact

    def get_entity_link_pos(
        self, entity: object, *, link_name: str, ref: str
    ) -> torch.Tensor:
        assert link_name == "wrist"
        assert ref == "link_origin"
        if self.wrist_position is None:
            raise ValueError("Expected wrist-position data for this test.")
        return self.wrist_position[:, None, :]

    def get_entity_link_quat(self, entity: object, *, link_name: str) -> torch.Tensor:
        assert link_name == "wrist"
        if self.wrist_quaternion is None:
            raise ValueError("Expected wrist-quaternion data for this test.")
        return self.wrist_quaternion[:, None, :]


def make_manager(
    *,
    metric_cfg: TaskMetricCfg,
    object_entities: dict[str, FakeObject],
    scene_object_arti_vals: np.ndarray | None = None,
    scene_object_arti_valid_mask: np.ndarray | None = None,
    scene_object_arti_names: list[list[str]] | None = None,
    contact_signal: torch.Tensor | None = None,
    source_contact_signal: torch.Tensor | None = None,
    receiving_net_force: torch.Tensor | None = None,
    source_net_force: torch.Tensor | None = None,
    hammer_impact: torch.Tensor | None = None,
    hammer_contact: torch.Tensor | None = None,
    wrist_position: torch.Tensor | None = None,
    wrist_quaternion: torch.Tensor | None = None,
) -> MultiObjectTaskManager:
    manager = MultiObjectTaskManager.__new__(MultiObjectTaskManager)
    manager.task_cfg = SimpleNamespace(metric_cfg=metric_cfg)
    manager.scene_object_ids = list(object_entities)
    manager.scene_table_height = 0.75
    manager.scene_object_arti_vals = (
        np.zeros((1, len(object_entities), 0), dtype=np.float32)
        if scene_object_arti_vals is None
        else scene_object_arti_vals
    )
    manager.scene_object_arti_valid_mask = (
        np.zeros((len(object_entities), 0), dtype=bool)
        if scene_object_arti_valid_mask is None
        else scene_object_arti_valid_mask
    )
    manager.scene_object_arti_names = (
        [[] for _ in object_entities]
        if scene_object_arti_names is None
        else scene_object_arti_names
    )
    manager.total_steps = int(manager.scene_object_arti_vals.shape[0])
    manager.scene_object_pos = np.stack(
        [obj.pos[0].numpy() for obj in object_entities.values()], axis=0
    )[None, ...]
    manager.scene_object_quat = np.stack(
        [obj.quat[0].numpy() for obj in object_entities.values()], axis=0
    )[None, ...]
    manager.env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        cfg=SimpleNamespace(
            decimation=8,
            reset_cfg=SimpleNamespace(eval_start_timestep=0),
        ),
        sim_cfg=SimpleNamespace(gravity=-9.81, dt=1.0 / 120.0),
        rewards_cfg=SimpleNamespace(task_success_pos_thd=0.03),
        physics_manager=FakePhysics(
            object_entities=object_entities,
            contact_signal=contact_signal,
            source_contact_signal=source_contact_signal,
            receiving_net_force=receiving_net_force,
            source_net_force=source_net_force,
            hammer_impact=hammer_impact,
            hammer_contact=hammer_contact,
            wrist_position=wrist_position,
            wrist_quaternion=wrist_quaternion,
        ),
        mode="train",
        start_timestep=torch.zeros(2, dtype=torch.long),
    )
    return manager


def test_bottle_interaction_metric_conditions_grasp_match_on_lift_success() -> None:
    cfg = TaskMetricCfg(
        metric_id="bottle_interaction",
        bottle_interaction_object_id="bottle",
        bottle_interaction_hand_side="right",
        bottle_relative_wrist_position_tolerance_m=0.05,
        bottle_relative_wrist_orientation_tolerance_rad=np.deg2rad(30.0),
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "bottle": FakeObject(
                pos=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
                arti=torch.zeros((2, 0)),
            )
        },
        wrist_position=torch.tensor(
            [[0.0, 0.0, 1.15], [0.15, 0.0, 1.0]], dtype=torch.float32
        ),
        wrist_quaternion=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    manager.scene_object_pos = np.asarray(
        [[[0.0, 0.0, 0.8]], [[0.0, 0.0, 1.0]]], dtype=np.float32
    )
    manager.scene_object_quat = np.asarray(
        [[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    manager.env.reset_manager = SimpleNamespace(
        frame_right_wrist_pos=torch.tensor([[0.0, 0.0, 1.15]]),
        frame_right_wrist_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )

    metrics = manager.compute_task_metrics()

    assert metrics["object_lift_success"].tolist() == [1.0, 1.0]
    assert metrics["bottle_requested_interaction_match"].tolist() == [1.0, 0.0]
    assert metrics["bottle_interaction_success"].tolist() == [1.0, 0.0]
    assert manager.primary_task_metric_name == "object_lift_success"

    episode_metrics = manager.compute_episode_end_task_metrics(
        torch.tensor([0, 1]), torch.tensor([True, False])
    )
    assert episode_metrics["object_lift_success"].tolist() == [0.0, 1.0]
    assert episode_metrics["bottle_interaction_success"].tolist() == [0.0, 0.0]


def test_multi_object_environment_is_default() -> None:
    assert EnvConfig().env_class == "multi_object"


def test_open_notebook_task_metric() -> None:
    cfg = TaskMetricCfg(
        metric_id="open_notebook",
        notebook_object_id="notebook",
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "notebook": FakeObject(
                pos=torch.zeros((2, 3)),
                arti=torch.tensor([[1.10], [0.60]]),
            )
        },
        scene_object_arti_vals=np.asarray([[[1.0]]], dtype=np.float32),
        scene_object_arti_valid_mask=np.asarray([[True]], dtype=bool),
    )

    metrics = manager.compute_task_metrics()

    assert metrics["notebook_final_angle_error_rad"].tolist() == pytest.approx(
        [0.10, 0.40]
    )
    assert metrics["notebook_final_angle_within_15deg"].tolist() == [1.0, 0.0]
    assert manager.primary_task_metric_name == "notebook_final_angle_within_15deg"


def test_early_termination_forces_primary_task_metric_failure() -> None:
    cfg = TaskMetricCfg(
        metric_id="open_notebook",
        notebook_object_id="notebook",
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "notebook": FakeObject(
                pos=torch.zeros((2, 3)),
                arti=torch.tensor([[1.00], [1.00]]),
            )
        },
        scene_object_arti_vals=np.asarray([[[1.0]]], dtype=np.float32),
        scene_object_arti_valid_mask=np.asarray([[True]], dtype=bool),
    )

    metrics = manager.compute_episode_end_task_metrics(
        torch.tensor([0, 1]),
        torch.tensor([True, False]),
    )

    assert metrics["notebook_final_angle_error_rad"].tolist() == [0.0, 0.0]
    assert metrics["notebook_final_angle_within_15deg"].tolist() == [0.0, 1.0]


def test_block_stacking_task_metric() -> None:
    cfg = TaskMetricCfg(
        metric_id="block_stacking",
        stacking_object_ids=("red", "green", "blue"),
        stacking_min_center_height_offsets_m=(0.03, 0.0225),
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "red": FakeObject(
                pos=torch.tensor([[0.0, 0.0, 0.80], [0.0, 0.0, 0.80]]),
                arti=torch.zeros((2, 0)),
            ),
            "green": FakeObject(
                pos=torch.tensor([[0.0, 0.0, 0.84], [0.0, 0.0, 0.82]]),
                arti=torch.zeros((2, 0)),
            ),
            "blue": FakeObject(
                pos=torch.tensor([[0.0, 0.0, 0.87], [0.0, 0.0, 0.86]]),
                arti=torch.zeros((2, 0)),
            ),
        },
    )

    metrics = manager.compute_task_metrics()

    assert metrics["stack_min_center_height_margin_m"].tolist() == pytest.approx(
        [0.0075, -0.01],
        abs=1.0e-6,
    )
    assert metrics["stack_success"].tolist() == [1.0, 0.0]


def test_object_lift_uses_reference_height_and_task_success_tolerance() -> None:
    cfg = TaskMetricCfg(
        metric_id="object_lift",
        object_lift_object_id="asset",
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "asset": FakeObject(
                pos=torch.tensor([[0.0, 0.0, 0.275], [0.0, 0.0, 0.235]]),
                arti=torch.zeros((2, 0)),
            )
        },
    )
    manager.scene_object_pos = np.asarray(
        [[[0.0, 0.0, 0.075]], [[0.0, 0.0, 0.275]]],
        dtype=np.float32,
    )

    manager._initialize_task_metric()
    metrics = manager.compute_task_metrics()

    assert metrics["object_lift_height_m"].tolist() == pytest.approx([0.20, 0.16])
    assert metrics["object_lift_reference_height_m"].tolist() == pytest.approx(
        [0.20, 0.20]
    )
    assert metrics["object_lift_height_error_m"].tolist() == pytest.approx([0.0, 0.04])
    assert metrics["object_lift_success"].tolist() == [1.0, 0.0]
    assert manager.primary_task_metric_name == "object_lift_success"


def test_precision_mounting_allows_clearance_limited_lean() -> None:
    clearance_m = 0.003
    cylinder_height_m = 0.09
    cfg = TaskMetricCfg(
        metric_id="precision_mounting",
        mounting_cylinder_object_id="cylinder",
        mounting_post_object_id="post",
        mounting_cylinder_height_m=cylinder_height_m,
        mounting_bore_clearance_m=clearance_m,
        mounting_max_origin_height_above_post_m=0.018,
        mounting_max_axis_tilt_rad=float(
            np.arcsin(2.0 * clearance_m / cylinder_height_m)
        ),
    )
    tilt = torch.deg2rad(torch.tensor([3.0, 5.0]))
    cylinder_quat = torch.stack(
        (
            torch.cos(tilt / 2.0),
            torch.zeros_like(tilt),
            torch.sin(tilt / 2.0),
            torch.zeros_like(tilt),
        ),
        dim=-1,
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "cylinder": FakeObject(
                pos=torch.tensor([[-0.002, 0.0, 0.012], [0.0, 0.0, 0.012]]),
                quat=cylinder_quat,
                arti=torch.zeros((2, 0)),
            ),
            "post": FakeObject(
                pos=torch.tensor([[0.0, 0.0, 0.002], [0.0, 0.0, 0.002]]),
                arti=torch.zeros((2, 0)),
            ),
        },
    )

    manager._initialize_task_metric()
    metrics = manager.compute_task_metrics()

    assert metrics["mounting_origin_height_above_post_m"].tolist() == pytest.approx(
        [0.01, 0.01]
    )
    assert metrics["mounting_bottom_axis_radial_error_m"].tolist() == pytest.approx(
        [0.002, 0.0]
    )
    assert metrics["mounting_top_axis_radial_error_m"].tolist() == pytest.approx(
        [0.002710, 0.007844], abs=1.0e-6
    )
    assert metrics["mounting_axis_tilt_rad"].tolist() == pytest.approx(
        np.deg2rad([3.0, 5.0]), abs=1.0e-6
    )
    assert metrics["mounting_success"].tolist() == [1.0, 0.0]
    assert manager.primary_task_metric_name == "mounting_success"


def test_dispenser_pressed_metric_preserves_press_after_spring_return() -> None:
    cfg = TaskMetricCfg(
        metric_id="dispenser_pressed",
        dispenser_object_id="dispenser",
        dispenser_joint_name="neck_to_head_prismatic",
        dispenser_pressed_tolerance_m=0.002,
    )
    dispenser = FakeObject(
        pos=torch.zeros((2, 3)),
        arti=torch.zeros((2, 1)),
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={"dispenser": dispenser},
        scene_object_arti_vals=np.asarray(
            [[[0.0]], [[-0.005]], [[-0.01]], [[0.0]]],
            dtype=np.float32,
        ),
        scene_object_arti_valid_mask=np.asarray([[True]], dtype=bool),
        scene_object_arti_names=[["neck_to_head_prismatic"]],
    )
    manager._initialize_task_metric()
    manager.reset_task_metric_accumulators(np.asarray([0, 1]))

    # Only the first environment reaches within 2 mm of the -10 mm reference
    # press. Returning both heads to the raised state must not erase that event.
    dispenser.arti = torch.tensor([[-0.0085], [-0.0079]])
    manager.update_task_metric_accumulators()
    dispenser.arti = torch.zeros((2, 1))
    manager.update_task_metric_accumulators()
    metrics = manager.compute_task_metrics()

    assert metrics["dispenser_min_head_joint_position_m"].tolist() == pytest.approx(
        [-0.0085, -0.0079]
    )
    assert metrics["dispenser_pressed_state_achieved"].tolist() == [1.0, 0.0]
    assert manager.primary_task_metric_name == "dispenser_pressed_state_achieved"


def test_drawer_metric_uses_one_centimeter_final_articulation_tolerance() -> None:
    cfg = TaskMetricCfg(
        metric_id="drawer_articulation",
        drawer_object_id="drawer",
        drawer_joint_name="drawer_slide",
        drawer_final_articulation_tolerance_m=0.01,
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "drawer": FakeObject(
                pos=torch.zeros((2, 3)),
                arti=torch.tensor([[0.091], [0.089]]),
            )
        },
        scene_object_arti_vals=np.asarray([[[0.0]], [[0.1]]], dtype=np.float32),
        scene_object_arti_valid_mask=np.asarray([[True]], dtype=bool),
        scene_object_arti_names=[["drawer_slide"]],
    )
    manager._initialize_task_metric()

    metrics = manager.compute_task_metrics()

    assert metrics["drawer_final_articulation_error_m"].tolist() == pytest.approx(
        [0.009, 0.011]
    )
    assert metrics["drawer_articulation_success"].tolist() == [1.0, 0.0]
    assert manager.primary_task_metric_name == "drawer_articulation_success"


def test_eraser_metric_requires_complete_erase_phase_below_two_centimeters() -> None:
    cfg = TaskMetricCfg(
        metric_id="eraser_tracking",
        eraser_object_id="eraser",
        eraser_phase_names=("erase_1",),
        eraser_max_position_error_m=0.02,
    )
    eraser = FakeObject(
        pos=torch.zeros((2, 3)),
        arti=torch.zeros((2, 0)),
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={"eraser": eraser},
    )
    manager.total_steps = 4
    manager.phase_names = ["reach", "erase_1", "hold"]
    manager.phase_lengths = np.asarray([1, 2, 1], dtype=np.int64)
    manager.phase_index_by_timestep = torch.tensor([0, 1, 1, 2])
    manager.scene_object_pos = np.asarray(
        [
            [[0.00, 0.0, 0.0]],
            [[0.01, 0.0, 0.0]],
            [[0.02, 0.0, 0.0]],
            [[0.02, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    manager.env.command_time_step = torch.zeros(2, dtype=torch.long)
    manager._initialize_task_metric()
    manager.reset_task_metric_accumulators(np.asarray([0, 1]))

    manager.env.command_time_step[:] = 1
    eraser.pos = torch.tensor([[0.020, 0.0, 0.0], [0.031, 0.0, 0.0]])
    manager.update_task_metric_accumulators()
    manager.env.command_time_step[:] = 2
    eraser.pos = torch.tensor([[0.020, 0.0, 0.0], [0.020, 0.0, 0.0]])
    manager.update_task_metric_accumulators()

    # Errors outside the named erase phases do not affect the criterion.
    manager.env.command_time_step[:] = 3
    eraser.pos = torch.full((2, 3), 1.0)
    manager.update_task_metric_accumulators()
    metrics = manager.compute_task_metrics()

    assert metrics["eraser_erase_phase_max_position_error_m"].tolist() == (
        pytest.approx([0.01, 0.021])
    )
    assert metrics["eraser_erase_phase_step_count"].tolist() == [2.0, 2.0]
    assert metrics["eraser_erase_phase_complete"].tolist() == [1.0, 1.0]
    assert metrics["eraser_erase_phase_success"].tolist() == [1.0, 0.0]
    assert manager.primary_task_metric_name == "eraser_erase_phase_success"

    manager._eraser_max_position_error_m = torch.tensor([0.019999, 0.02])
    boundary_metrics = manager.compute_task_metrics()
    assert boundary_metrics["eraser_erase_phase_success"].tolist() == [1.0, 0.0]


def test_drawer_final_scene_articulation_uses_task_metric_tolerance() -> None:
    metrics = FinalRolloutMetrics.__new__(FinalRolloutMetrics)
    metrics.env = SimpleNamespace(
        task_manager=SimpleNamespace(
            task_cfg=SimpleNamespace(
                metric_cfg=TaskMetricCfg(
                    metric_id="drawer_articulation",
                    drawer_object_id="drawer",
                    drawer_final_articulation_tolerance_m=0.01,
                )
            )
        )
    )

    threshold = metrics._final_scene_arti_thresholds(
        object_id="drawer",
        valid_arti_count=1,
    )

    assert threshold.tolist() == pytest.approx([0.01])


def test_banana_handover_task_metric() -> None:
    cfg = TaskMetricCfg(
        metric_id="banana_handover",
        banana_object_id="banana",
        banana_source_hand="right",
        banana_receiving_hand="left",
    )
    contact_signal = torch.zeros((2, 5, 3))
    contact_signal[0, 0, 0] = 1.0
    contact_signal[1, 0, 0] = 1.0
    source_contact_signal = torch.zeros((2, 5, 3))
    source_contact_signal[0, 0, 0] = 1.0
    source_contact_signal[1, 0, 0] = 1.0
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "banana": FakeObject(
                pos=torch.tensor([[0.0, 0.0, 0.90], [0.0, 0.0, 0.90]]),
                arti=torch.zeros((2, 0)),
            )
        },
        contact_signal=contact_signal,
        source_contact_signal=source_contact_signal,
        receiving_net_force=torch.tensor([[0.0, 0.0, 6.0], [0.0, 0.0, 6.0]]),
        source_net_force=torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 6.0]]),
    )

    metrics = manager.compute_task_metrics()

    assert metrics["banana_in_air"].tolist() == [1.0, 1.0]
    assert metrics["banana_not_fallen"].tolist() == [1.0, 1.0]
    assert metrics["banana_source_hand_contact_link_count"].tolist() == [1.0, 1.0]
    assert metrics["banana_source_hand_released"].tolist() == [1.0, 0.0]
    assert metrics["banana_receiving_hand_contact_link_count"].tolist() == [1.0, 1.0]
    assert metrics["banana_receiving_hand_upward_support_force_n"].tolist() == [
        6.0,
        6.0,
    ]
    assert metrics["banana_receiving_hand_support"].tolist() == [1.0, 1.0]
    assert metrics["banana_handover_success"].tolist() == [1.0, 0.0]


def test_banana_handover_rejects_fallen_banana() -> None:
    cfg = TaskMetricCfg(
        metric_id="banana_handover",
        banana_object_id="banana",
        banana_source_hand="right",
        banana_receiving_hand="left",
    )
    receiving_contact_signal = torch.zeros((2, 5, 3))
    receiving_contact_signal[:, 0, 0] = 1.0
    receiving_contact_signal[:, 2, 1] = 1.0
    source_contact_signal = torch.zeros((2, 5, 3))
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "banana": FakeObject(
                pos=torch.tensor([[0.0, 0.0, 0.79], [0.0, 0.0, 0.90]]),
                arti=torch.zeros((2, 0)),
            )
        },
        contact_signal=receiving_contact_signal,
        source_contact_signal=source_contact_signal,
        receiving_net_force=torch.tensor([[0.0, 0.0, 6.0], [0.0, 0.0, 6.0]]),
        source_net_force=torch.zeros((2, 3)),
    )

    metrics = manager.compute_task_metrics()

    assert metrics["banana_receiving_hand_support"].tolist() == [1.0, 1.0]
    assert metrics["banana_source_hand_released"].tolist() == [1.0, 1.0]
    assert metrics["banana_not_fallen"].tolist() == [0.0, 1.0]
    assert metrics["banana_handover_success"].tolist() == [0.0, 1.0]


def test_hammer_impact_averages_only_contact_steps() -> None:
    cfg = TaskMetricCfg(
        metric_id="hammer_peg_impact",
        hammer_object_id="hammer",
        hammer_peg_object_id="peg",
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            object_id: FakeObject(
                pos=torch.zeros((2, 3)),
                arti=torch.zeros((2, 0)),
            )
            for object_id in ("hammer", "peg")
        },
        hammer_impact=torch.tensor([8.0, 0.0]),
        hammer_contact=torch.tensor([True, False]),
    )
    manager._hammer_contact_impact_sum = torch.zeros(2)
    manager._hammer_contact_step_count = torch.zeros(2, dtype=torch.long)

    manager.update_task_metric_accumulators()
    manager.env.physics_manager.hammer_impact = torch.tensor([4.0, 0.0])
    manager.env.physics_manager.hammer_contact = torch.tensor([True, False])
    manager.update_task_metric_accumulators()
    metrics = manager.compute_task_metrics()

    assert metrics["hammer_peg_average_contact_impact_n"].tolist() == [6.0, 0.0]
    assert metrics["hammer_peg_contact_step_count"].tolist() == [2.0, 0.0]


def test_hammer_single_swing_uses_first_contact_in_strike_window() -> None:
    cfg = TaskMetricCfg(
        metric_id="hammer_single_swing",
        hammer_object_id="hammer",
        hammer_peg_object_id="peg",
        hammer_strike_phase_names=("strike", "hold_impact"),
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            object_id: FakeObject(
                pos=torch.zeros((2, 3)),
                arti=torch.zeros((2, 0)),
            )
            for object_id in ("hammer", "peg")
        },
        hammer_impact=torch.tensor([8.0, 4.0]),
        hammer_contact=torch.tensor([True, True]),
    )
    manager.phase_names = ["hold_ready", "strike", "hold_impact"]
    manager.phase_index_by_timestep = torch.tensor([0, 1, 2])
    manager._hammer_strike_phase_indices = torch.tensor([1, 2])
    manager._hammer_contact_impact_sum = torch.zeros(2)
    manager._hammer_contact_step_count = torch.zeros(2, dtype=torch.long)
    manager._hammer_strike_contact_timestep = torch.full((2,), -1, dtype=torch.long)

    # Contact before the strike window contributes to force, but not completion.
    manager.env.command_time_step = torch.tensor([0, 0])
    manager.env.episode_length = torch.tensor([40, 40])
    manager.update_task_metric_accumulators()
    manager.env.physics_manager.hammer_impact = torch.tensor([6.0, 0.0])
    manager.env.physics_manager.hammer_contact = torch.tensor([True, False])
    manager.env.command_time_step = torch.tensor([1, 1])
    manager.env.episode_length = torch.tensor([55, 55])
    manager.update_task_metric_accumulators()

    metrics = manager.compute_task_metrics()

    assert metrics["hammer_strike_contacted"].tolist() == [1.0, 0.0]
    assert metrics["hammer_single_swing_completed"].tolist() == [1.0, 0.0]
    assert metrics["hammer_single_swing_completion_time_steps"][0].item() == 55.0
    assert torch.isnan(metrics["hammer_single_swing_completion_time_steps"][1])
    assert metrics["hammer_single_swing_completion_time_s"][0].item() == pytest.approx(
        55.0 / 15.0
    )
    assert metrics["hammer_peg_average_contact_impact_n"].tolist() == [7.0, 4.0]
    assert manager.primary_task_metric_name == "hammer_single_swing_completed"


def test_hammer_two_swing_metric_uses_actual_second_strike_contact_time() -> None:
    cfg = TaskMetricCfg(
        metric_id="hammer_two_swing",
        hammer_object_id="hammer",
        hammer_peg_object_id="peg",
        hammer_first_strike_phase_names=("strike", "hold_impact"),
        hammer_second_strike_phase_names=("strike_again", "hold_second_impact"),
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            object_id: FakeObject(
                pos=torch.zeros((2, 3)),
                arti=torch.zeros((2, 0)),
            )
            for object_id in ("hammer", "peg")
        },
        hammer_impact=torch.tensor([8.0, 0.0]),
        hammer_contact=torch.tensor([True, False]),
    )
    manager.phase_names = [
        "strike",
        "hold_impact",
        "lift_back",
        "strike_again",
        "hold_second_impact",
    ]
    manager.phase_index_by_timestep = torch.tensor([0, 2, 3])
    manager._hammer_first_strike_phase_indices = torch.tensor([0, 1])
    manager._hammer_second_strike_phase_indices = torch.tensor([3, 4])
    manager._hammer_contact_impact_sum = torch.zeros(2)
    manager._hammer_contact_step_count = torch.zeros(2, dtype=torch.long)
    manager._hammer_first_contact_timestep = torch.full((2,), -1, dtype=torch.long)
    manager._hammer_second_contact_timestep = torch.full((2,), -1, dtype=torch.long)

    manager.env.command_time_step = torch.tensor([0, 0])
    manager.env.episode_length = torch.tensor([10, 10])
    manager.update_task_metric_accumulators()
    manager.env.physics_manager.hammer_impact = torch.tensor([4.0, 5.0])
    manager.env.physics_manager.hammer_contact = torch.tensor([True, True])
    manager.env.command_time_step = torch.tensor([2, 2])
    manager.env.episode_length = torch.tensor([30, 30])
    manager.update_task_metric_accumulators()

    metrics = manager.compute_task_metrics()

    assert metrics["hammer_first_strike_contacted"].tolist() == [1.0, 0.0]
    assert metrics["hammer_second_strike_contacted"].tolist() == [1.0, 1.0]
    assert metrics["hammer_two_swings_completed"].tolist() == [1.0, 0.0]
    assert metrics["hammer_two_swing_completion_time_steps"][0].item() == 30.0
    assert torch.isnan(metrics["hammer_two_swing_completion_time_steps"][1])
    assert metrics["hammer_two_swing_completion_time_s"][0].item() == pytest.approx(2.0)
    assert metrics["hammer_peg_average_contact_impact_n"].tolist() == [6.0, 5.0]
    assert manager.primary_task_metric_name == "hammer_two_swings_completed"


def test_hammer_checkpoint_metric_excludes_random_phase_resets() -> None:
    cfg = TaskMetricCfg(
        metric_id="hammer_two_swing",
        hammer_object_id="hammer",
        hammer_peg_object_id="peg",
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            object_id: FakeObject(
                pos=torch.zeros((2, 3)),
                arti=torch.zeros((2, 0)),
            )
            for object_id in ("hammer", "peg")
        },
    )
    manager.env.start_timestep = torch.tensor([0, 12], dtype=torch.long)
    episode_metrics = {
        "hammer_two_swings_completed": torch.tensor([1.0, 1.0]),
        "hammer_peg_average_contact_impact_n": torch.tensor([2.0, 20.0]),
    }

    checkpoint_values = manager.compute_episode_end_checkpoint_metric(
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([False, False]),
        episode_metrics,
    )

    assert manager.checkpoint_metric_name == "hammer_two_swings_completed"
    assert checkpoint_values is not None
    assert checkpoint_values.tolist() == [1.0]


def test_hammer_single_swing_checkpoint_metric_excludes_random_phase_resets() -> None:
    cfg = TaskMetricCfg(
        metric_id="hammer_single_swing",
        hammer_object_id="hammer",
        hammer_peg_object_id="peg",
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            object_id: FakeObject(
                pos=torch.zeros((2, 3)),
                arti=torch.zeros((2, 0)),
            )
            for object_id in ("hammer", "peg")
        },
    )
    manager.env.start_timestep = torch.tensor([0, 12], dtype=torch.long)
    episode_metrics = {
        "hammer_single_swing_completed": torch.tensor([1.0, 1.0]),
        "hammer_peg_average_contact_impact_n": torch.tensor([2.0, 20.0]),
    }

    checkpoint_values = manager.compute_episode_end_checkpoint_metric(
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([False, False]),
        episode_metrics,
    )

    assert manager.checkpoint_metric_name == "hammer_single_swing_completed"
    assert checkpoint_values is not None
    assert checkpoint_values.tolist() == [1.0]


def test_object_pose_metric_uses_frozen_position_and_orientation_thresholds() -> None:
    cfg = TaskMetricCfg(
        metric_id="object_pose",
        object_pose_object_id="cup",
        object_pose_position_tolerance_m=0.03,
        object_pose_orientation_tolerance_rad=np.deg2rad(15.0),
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "cup": FakeObject(
                pos=torch.tensor([[0.20, 0.0, 0.90], [0.24, 0.0, 0.90]]),
                quat=torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.9238795, 0.0, 0.3826834, 0.0],
                    ]
                ),
                arti=torch.zeros((2, 0)),
            )
        },
    )
    manager.scene_object_pos[-1, 0] = np.asarray([0.20, 0.0, 0.90])
    manager.scene_object_quat[-1, 0] = np.asarray([1.0, 0.0, 0.0, 0.0])

    metrics = manager.compute_task_metrics()

    assert metrics["object_final_position_error_m"].tolist() == pytest.approx(
        [0.0, 0.04]
    )
    assert metrics["object_final_orientation_error_rad"].tolist() == pytest.approx(
        [0.0, np.pi / 4.0], abs=1.0e-6
    )
    assert metrics["object_pose_success"].tolist() == [1.0, 0.0]


@pytest.mark.parametrize(
    ("expected_lift", "expected_success"),
    ((True, [1.0, 0.0]), (False, [0.0, 1.0])),
)
def test_cube_strategy_combines_final_pose_with_lift_classifier(
    expected_lift: bool,
    expected_success: list[float],
) -> None:
    cfg = TaskMetricCfg(
        metric_id="cube_strategy",
        cube_strategy_object_id="red",
        cube_strategy_expected_lift=expected_lift,
        cube_strategy_min_lift_height_change_m=0.02,
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "red": FakeObject(
                pos=torch.tensor([[0.18, 0.0, 0.022], [0.18, 0.0, 0.022]]),
                arti=torch.zeros((2, 0)),
            )
        },
    )
    manager.scene_object_pos[-1, 0] = np.asarray([0.18, 0.0, 0.022])
    manager._cube_initial_center_z = torch.tensor([0.022, 0.022])
    manager._cube_max_center_z = torch.tensor([0.162, 0.022])

    metrics = manager.compute_task_metrics()

    assert metrics["cube_final_pose_success"].tolist() == [1.0, 1.0]
    assert metrics["cube_max_center_height_change_m"].tolist() == pytest.approx(
        [0.14, 0.0]
    )
    assert metrics["cube_lifted"].tolist() == [1.0, 0.0]
    assert metrics["cube_strategy_success"].tolist() == expected_success


def test_mug_checkpoint_metric_uses_episode_tracking_average() -> None:
    cfg = TaskMetricCfg(
        eval_metric_id="mug_grasp_change",
        mug_object_id="mug",
        mug_hand_side="right",
        mug_reference_grasp_phase_names=("top_grasp", "side_grasp"),
    )
    manager = make_manager(
        metric_cfg=cfg,
        object_entities={
            "mug": FakeObject(
                pos=torch.zeros((2, 3)),
                arti=torch.zeros((2, 0)),
            )
        },
    )
    manager._mug_hand_tracking_score_sum = torch.tensor([1.5, 3.0])
    manager._mug_hand_tracking_step_count = torch.tensor([2, 4])

    values = manager.compute_episode_end_checkpoint_metric(
        torch.tensor([0, 1]),
        torch.tensor([True, False]),
        {},
    )

    assert manager.checkpoint_metric_name == "mug_hand_reference_tracking_score"
    assert values.tolist() == [0.0, 0.75]


@pytest.mark.parametrize(
    ("experiment_id", "metric_id"),
    (
        ("v2_03_open_notebook", "open_notebook"),
        ("v2_03_stack_rgb_blocks_different_sizes", "block_stacking"),
        ("v2_03_banana_handover", "banana_handover"),
        ("v2_03_mug_grasp_change_sequence", None),
        ("v2_03_hammer_strike_peg", "hammer_peg_impact"),
        ("v2_03_dispenser_press_twice", "dispenser_pressed"),
        ("v2_03_erase_board", "eraser_tracking"),
        ("v2_03_invert_cup", "object_pose"),
        ("v2_03_open_drawer", "drawer_articulation"),
    ),
)
def test_key_task_protocol_settings(
    experiment_id: str,
    metric_id: str | None,
) -> None:
    cfg = load_experiment_config(
        preset=OUR_METHOD_PRESET_NAME,
        overrides=[
            "env.task_cfg.oakink_exported_path="
            f"logs/v2_03_pipeline/{experiment_id}/exports/task.npy"
        ],
    )

    assert cfg.agent.timesteps == 50_000
    assert cfg.agent.normalize_advantages is True
    assert cfg.agent.policy_lr == pytest.approx(1.0e-4)
    assert cfg.agent.value_lr == pytest.approx(1.0e-4)
    assert cfg.env.key_task_protocol_version == "gate0_v0.5"
    assert cfg.env.sim_cfg.num_envs == 1024
    assert cfg.agent.eval_freq == 0
    assert cfg.agent.eval_metrics_freq == 0
    assert cfg.agent.eval_video_freq == 0
    assert cfg.env.task_cfg.metric_cfg.metric_id == metric_id
    if experiment_id == "v2_03_banana_handover":
        assert cfg.env.task_cfg.metric_cfg.banana_source_hand == "right"
        assert cfg.env.task_cfg.metric_cfg.banana_receiving_hand == "left"
    if experiment_id == "v2_03_mug_grasp_change_sequence":
        assert cfg.env.task_cfg.metric_cfg.eval_metric_id == "mug_grasp_change"
    if experiment_id == "v2_03_dispenser_press_twice":
        assert cfg.env.task_cfg.metric_cfg.dispenser_object_id == "dispenser"
        assert (
            cfg.env.task_cfg.metric_cfg.dispenser_joint_name == "neck_to_head_prismatic"
        )
        assert (
            cfg.env.task_cfg.metric_cfg.dispenser_pressed_tolerance_m
            == pytest.approx(0.002)
        )
    if experiment_id == "v2_03_erase_board":
        assert cfg.env.task_cfg.metric_cfg.eraser_object_id == "eraser"
        assert cfg.env.task_cfg.metric_cfg.eraser_phase_names == tuple(
            f"erase_{phase_index}" for phase_index in range(1, 8)
        )
        assert cfg.env.task_cfg.metric_cfg.eraser_max_position_error_m == (
            pytest.approx(0.02)
        )
    if experiment_id == "v2_03_invert_cup":
        assert cfg.env.task_cfg.metric_cfg.object_pose_object_id == "cup"
        assert cfg.env.task_cfg.metric_cfg.object_pose_position_tolerance_m == (
            pytest.approx(0.03)
        )
        assert cfg.env.task_cfg.metric_cfg.object_pose_orientation_tolerance_rad == (
            pytest.approx(np.deg2rad(15.0))
        )
    if experiment_id == "v2_03_open_drawer":
        assert cfg.env.task_cfg.metric_cfg.drawer_object_id == "drawer"
        assert cfg.env.task_cfg.metric_cfg.drawer_joint_name == "drawer_slide"
        assert cfg.env.task_cfg.metric_cfg.drawer_final_articulation_tolerance_m == (
            pytest.approx(0.01)
        )


@pytest.mark.parametrize(
    ("experiment_id", "metric_id", "expected_lift"),
    (
        ("v3_01_hammer_timing_slow", "hammer_two_swing", None),
        ("v3_01_hammer_timing_medium", "hammer_two_swing", None),
        ("v3_01_hammer_timing_fast", "hammer_two_swing", None),
        ("v3_01_cup_pose_tilt_045deg", "object_pose", None),
        ("v3_01_cup_pose_tilt_090deg", "object_pose", None),
        ("v3_01_cup_pose_tilt_120deg", "object_pose", None),
        ("v3_01_cube_strategy_pick", "cube_strategy", True),
        ("v3_01_cube_strategy_push", "cube_strategy", False),
    ),
)
def test_v3_01_controllability_protocol_settings(
    experiment_id: str,
    metric_id: str,
    expected_lift: bool | None,
) -> None:
    cfg = load_experiment_config(
        preset=OUR_METHOD_PRESET_NAME,
        overrides=[
            "env.task_cfg.oakink_exported_path="
            "logs/v3_01_controllability_pipeline/"
            f"{experiment_id}/exports/task.npy"
        ],
    )

    assert cfg.env.key_task_protocol_version == "v3_01_controllability_v0.3"
    assert cfg.env.task_cfg.metric_cfg.metric_id == metric_id
    assert cfg.agent.timesteps == 50_000
    assert cfg.agent.eval_freq == 0
    if expected_lift is not None:
        assert cfg.env.task_cfg.metric_cfg.cube_strategy_expected_lift is expected_lift


@pytest.mark.parametrize(
    ("exported_path", "expected_object_id"),
    (
        (
            "logs/v3_03_reusability_pipeline/"
            "v3_03_reuse_mug_top_scale_070/exports/"
            "reach_grasp_lift_oakink_sequence.npy",
            "mug",
        ),
        (
            "logs/v3_03_reusability_pipeline/"
            "v3_03_reuse_bottle_top_scale_125/exports/"
            "reach_grasp_lift_oakink_sequence.npy",
            "bottle",
        ),
        (
            "logs/v3_05_template_transfer_pipeline/"
            "v3_05_mug_top_transferred/exports/"
            "reach_grasp_lift_oakink_sequence.npy",
            "template_transfer",
        ),
    ),
)
def test_gate4_lift_protocol_settings(
    exported_path: str,
    expected_object_id: str,
) -> None:
    cfg = load_experiment_config(
        preset=OUR_METHOD_PRESET_NAME,
        overrides=[f"env.task_cfg.oakink_exported_path={exported_path}"],
    )

    assert cfg.env.key_task_protocol_version == "gate4_v0.1"
    assert cfg.env.task_cfg.metric_cfg.metric_id == "object_lift"
    assert cfg.env.task_cfg.metric_cfg.object_lift_object_id == expected_object_id
    assert cfg.env.rewards_cfg.task_success_pos_thd == pytest.approx(0.03)
    assert cfg.agent.timesteps == 50_000
    assert cfg.agent.normalize_advantages is True
    assert cfg.agent.policy_lr == pytest.approx(1.0e-4)
    assert cfg.agent.value_lr == pytest.approx(1.0e-4)
    assert cfg.agent.eval_freq == 0


@pytest.mark.parametrize("grasp", ("side", "top"))
def test_normal_scale_bottle_protocol_logs_interaction_control(grasp: str) -> None:
    experiment_id = f"v3_03_reuse_bottle_{grasp}_scale_100"
    cfg = load_experiment_config(
        preset=OUR_METHOD_PRESET_NAME,
        overrides=[
            "env.task_cfg.oakink_exported_path="
            "logs/v3_03_reusability_pipeline/"
            f"{experiment_id}/exports/reach_grasp_lift_oakink_sequence.npy"
        ],
    )
    metric_cfg = cfg.env.task_cfg.metric_cfg

    assert cfg.env.key_task_protocol_version == "gate4_bottle_interaction_v0.1"
    assert metric_cfg.metric_id == "bottle_interaction"
    assert metric_cfg.bottle_interaction_object_id == "bottle"
    assert metric_cfg.bottle_interaction_hand_side == "right"
    assert metric_cfg.bottle_relative_wrist_position_tolerance_m == pytest.approx(0.05)
    assert metric_cfg.bottle_relative_wrist_orientation_tolerance_rad == (
        pytest.approx(np.deg2rad(30.0))
    )


@pytest.mark.parametrize(
    ("clearance_suffix", "clearance_m"),
    (("03", 0.003), ("05", 0.005), ("07", 0.007)),
)
def test_v3_08_precision_mounting_protocol_settings(
    clearance_suffix: str,
    clearance_m: float,
) -> None:
    cfg = load_experiment_config(
        preset=OUR_METHOD_PRESET_NAME,
        overrides=[
            "env.task_cfg.oakink_exported_path="
            "logs/v3_08_precision_mounting_pipeline/"
            f"v3_08_mounting_{clearance_suffix}/exports/task.npy"
        ],
    )
    metric_cfg = cfg.env.task_cfg.metric_cfg

    assert cfg.env.key_task_protocol_version == "v3_08_precision_mounting_v0.1"
    assert metric_cfg.metric_id == "precision_mounting"
    assert metric_cfg.mounting_cylinder_object_id == "cylinder"
    assert metric_cfg.mounting_post_object_id == "post"
    assert metric_cfg.mounting_cylinder_height_m == pytest.approx(0.09)
    assert metric_cfg.mounting_bore_clearance_m == pytest.approx(clearance_m)
    assert metric_cfg.mounting_max_origin_height_above_post_m == pytest.approx(0.018)
    assert metric_cfg.mounting_max_axis_tilt_rad == pytest.approx(
        np.arcsin(2.0 * clearance_m / 0.09)
    )
    assert cfg.agent.timesteps == 50_000
    assert cfg.agent.eval_freq == 0


@pytest.mark.parametrize("bowl_count", (2, 3, 5))
def test_gate4_bowl_composition_protocol_settings(bowl_count: int) -> None:
    experiment_id = f"v3_04_stack_{bowl_count}_bowls_normal"
    cfg = load_experiment_config(
        preset=OUR_METHOD_PRESET_NAME,
        overrides=[
            "env.task_cfg.oakink_exported_path="
            "logs/v3_04_compositionality_pipeline/"
            f"{experiment_id}/exports/stack_bowls_oakink_sequence.npy"
        ],
    )

    assert cfg.env.key_task_protocol_version == "gate4_v0.1"
    assert cfg.env.task_cfg.metric_cfg.metric_id == "block_stacking"
    assert cfg.env.task_cfg.metric_cfg.stacking_object_ids == tuple(
        f"bowl_{index}" for index in range(1, bowl_count + 1)
    )
    assert cfg.env.task_cfg.metric_cfg.stacking_min_center_height_offsets_m == (
        pytest.approx((0.05711411,) * (bowl_count - 1))
    )
    assert cfg.agent.timesteps == 50_000
