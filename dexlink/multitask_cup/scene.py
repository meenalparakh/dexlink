"""multi-task-only seeded simulator subclasses; the single-task factories remain intact."""

from __future__ import annotations

import genesis as gs

from dexlink.src.rl.multi_object.env import MultiObjectEnv
from dexlink.src.rl.multi_object.physics_sim import MultiObjectPhysicsSim
from dexlink.src.rl.multi_object.task_manager import MultiObjectTaskManager
from dexlink.src.retargeting.data.oakink_processor_sequence import (
    OakinkProcessorSequence,
)
from dexlink.src.rl.world.target_action_manager import AbsoluteTargetActionManager

_genesis_configuration: tuple[str, int] | None = None


class CupAbsoluteTargetActionManager(AbsoluteTargetActionManager):
    def _compute_position_targets(self, *, spec, action):
        targets = super()._compute_position_targets(spec=spec, action=action)
        if self.env.passive_mask is not None:
            # Completed evaluation slots keep their final joint command during passive observation.
            targets[0][self.env.passive_mask] = self.env.passive_joint_targets[
                self.env.passive_mask
            ]
        return targets


class CupSequenceTaskManager(MultiObjectTaskManager):
    """Use the approved support frame explicitly; generic oakink loading drops this setting."""

    def initialize(self) -> None:
        if self.task_cfg.object_only_scene or self.task_cfg.grasp_refinement_mode:
            raise ValueError(
                "multi-task sequence scenes require a controlled hand and immutable sequence references."
            )
        self.original_demo = OakinkProcessorSequence(
            self.task_cfg.oakink_exported_path,
            start_timestep=self.task_cfg.start_timestep,
            end_timestep=self.task_cfg.end_timestep,
            skip=self.task_cfg.skip,
            support_height=self.task_cfg.lift_support_height,
            support_margin=self.task_cfg.lift_support_margin,
        )
        self.hand_side_available = self._resolve_hand_side_availability()
        self.total_steps = self.original_demo.n_frames
        self.env.sim_cfg.apply_random_force = False
        self._initialize_phase_metadata()
        self._initialize_scene_object_metadata()
        self._initialize_task_metric()


class CupPhysicsSim(MultiObjectPhysicsSim):
    def initialize(self, camera_lookat: tuple[float, float, float]) -> None:
        global _genesis_configuration
        self.camera_lookat = camera_lookat
        configuration = (self.env.device, self.env.cfg.seed)
        if _genesis_configuration is None:
            gs.init(
                backend=gs.cpu if self.env.device == "cpu" else gs.cuda,
                seed=self.env.cfg.seed,
            )
            _genesis_configuration = configuration
        elif _genesis_configuration != configuration:
            raise ValueError(
                "All multi-task scenes in one process must share the same Genesis device and seed."
            )
        self.scene = gs.Scene(**self._get_scene_cfg())
        self._initialize_cameras()
        self._populate_scene()
        self._build()
        self.env.task_manager.process_retargeted_motion()


class CupEnv(MultiObjectEnv):
    def _create_physics_manager(self) -> CupPhysicsSim:
        if self.cfg.backend != "sim":
            raise ValueError("multi-task CupEnv requires backend='sim'.")
        return CupPhysicsSim(self.cfg.sim_cfg, self)


class CupSequenceEnv(CupEnv):
    def __init__(self, cfg):
        self.passive_mask = None
        self.passive_joint_targets = None
        super().__init__(cfg)

    def _create_task_manager(self) -> CupSequenceTaskManager:
        return CupSequenceTaskManager(self.cfg.task_cfg, self)

    def _create_action_manager(self) -> CupAbsoluteTargetActionManager:
        if self.cfg.action_cfg.action_mode != "absolute_target":
            raise ValueError(
                "multi-task sequence scenes require absolute wrist/finger targets."
            )
        return CupAbsoluteTargetActionManager(self.cfg.action_cfg, self)
