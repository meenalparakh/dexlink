from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from dexlink.src.retargeting.data.demo_loader import is_oakink_dataset, load_demo_processor
from dexlink.src.retargeting.data.oakink_processor_grasp import OakinkProcessorGrasp
from dexlink.src.rl.world.art_object import ArtObject
from dexlink.src.rl.world.robot import Hand

if TYPE_CHECKING:
    from dexlink.src.retargeting.data.exported_oakink_processor import ExportedOakInkProcessor
    from dexlink.src.retargeting.data.process_arctic import ArcticProcessor, RandomTrajProcessor
    from dexlink.src.rl.config.base import TaskCfg
    from dexlink.src.rl.world.env import Env


class TaskManager:
    def __init__(self, task_cfg: TaskCfg, env: Env):
        self.task_cfg = task_cfg
        self.env = env
        self.object_only = task_cfg.object_only_scene
        self.hand_side_available = {"left": False, "right": False}
        self.original_demo: (
            ArcticProcessor | ExportedOakInkProcessor | RandomTrajProcessor
        ) | None = None
        self.total_steps = 0

    def initialize(self) -> None:
        self.original_demo = load_demo_processor(
            self.task_cfg,
            start_timestep=self.task_cfg.start_timestep,
            end_timestep=self.task_cfg.end_timestep,
        )
        self.hand_side_available = self._resolve_hand_side_availability()
        self.env.sim_cfg.apply_random_force = (
            self.env.sim_cfg.apply_random_force
            and (
                isinstance(self.original_demo, OakinkProcessorGrasp)
                or self.task_cfg.grasp_refinement_mode
            )
        )
        if self.task_cfg.grasp_refinement_mode:
            self._validate_grasp_refinement_configuration()

        if is_oakink_dataset(self.task_cfg.demo_dataset):
            print(
                f"Using {self.task_cfg.demo_dataset} demo from: {self.task_cfg.oakink_exported_path}"
            )
        elif self.task_cfg.demo_dataset == "arctic":
            print(f"Using arctic demo from: {self.task_cfg.original_demo_path}")
        else:
            print(
                f"Using random trajectory results from: {self.original_demo.robot_obj_traj_path}"
            )

        self.total_steps = self.original_demo.n_frames

    def _validate_grasp_refinement_configuration(self) -> None:
        """Require a dynamic, unsupported object with a physical disturbance."""
        if self.task_cfg.object_fixed:
            raise ValueError(
                "Grasp refinement requires a dynamic object; "
                "task_cfg.object_fixed must be False."
            )
        gravity = float(self.env.sim_cfg.gravity)
        if gravity > 0.0:
            raise ValueError(
                "Grasp refinement gravity must be non-positive, got "
                f"sim_cfg.gravity={gravity}."
            )
        if self.env.sim_cfg.apply_random_force:
            if self.env.sim_cfg.random_force_mag <= 0.0:
                raise ValueError(
                    "Random-force grasp refinement requires "
                    "sim_cfg.random_force_mag > 0, got "
                    f"{self.env.sim_cfg.random_force_mag}."
                )
            return
        if gravity == 0.0:
            raise ValueError(
                "Grasp refinement requires either downward gravity "
                "(sim_cfg.gravity < 0) or a positive random external force."
            )

    def process_retargeted_motion(self) -> None:
        if self.object_only:
            return
        assert self.original_demo is not None
        self.original_demo.process_retargeted_motion(
            left_hand=self.env.physics_manager.scene_entities.get("left"),
            right_hand=self.env.physics_manager.scene_entities.get("right"),
        )

    def _resolve_hand_side_availability(self) -> dict[str, bool]:
        if self.object_only:
            return {"left": False, "right": False}
        assert self.original_demo is not None
        availability = getattr(self.original_demo, "hand_side_available", None)
        if availability is None:
            return {"left": True, "right": True}
        return {
            "left": bool(availability.get("left", False)),
            "right": bool(availability.get("right", False)),
        }

    def has_hand(self, side: str) -> bool:
        return bool(self.hand_side_available.get(side, False))

    @property
    def object_raised_by(self) -> float:
        assert self.original_demo is not None
        value = getattr(self.original_demo, "object_raised_by", None)
        if value is None:
            return 0.1
        return float(value)

    @property
    def active_hand_sides(self) -> list[str]:
        return [side for side in ("left", "right") if self.has_hand(side)]

    @property
    def task_metric_names(self) -> tuple[str, ...]:
        return ()

    @property
    def primary_task_metric_name(self) -> str | None:
        return None

    @property
    def checkpoint_metric_name(self) -> str | None:
        return self.primary_task_metric_name

    def compute_task_metrics(self) -> dict[str, torch.Tensor]:
        return {}

    def update_task_metric_accumulators(self) -> None:
        pass

    def reset_task_metric_accumulators(self, env_ids: np.ndarray) -> None:
        pass

    def compute_episode_end_task_metrics(
        self,
        env_ids: torch.Tensor,
        early_terminated: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        metrics = self.compute_task_metrics()
        if not metrics:
            return {}
        if env_ids.ndim != 1 or early_terminated.shape != env_ids.shape:
            raise ValueError(
                "Expected rank-1 env_ids and a matching early_terminated mask, got "
                f"env_ids={tuple(env_ids.shape)}, "
                f"early_terminated={tuple(early_terminated.shape)}."
            )
        if env_ids.dtype != torch.long or early_terminated.dtype != torch.bool:
            raise ValueError(
                "Expected env_ids dtype torch.long and early_terminated dtype "
                f"torch.bool, got {env_ids.dtype} and {early_terminated.dtype}."
            )
        primary_name = self.primary_task_metric_name
        if primary_name not in metrics:
            raise KeyError(
                "Configured primary task metric is absent from terminal task metrics: "
                f"primary={primary_name!r}, available={sorted(metrics)}."
            )
        episode_metrics = {
            name: values[env_ids].clone() for name, values in metrics.items()
        }
        episode_metrics[primary_name][early_terminated] = 0.0
        return episode_metrics

    def compute_episode_end_checkpoint_metric(
        self,
        env_ids: torch.Tensor,
        early_terminated: torch.Tensor,
        episode_task_metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        checkpoint_name = self.checkpoint_metric_name
        if checkpoint_name is None:
            return None
        if checkpoint_name not in episode_task_metrics:
            raise KeyError(
                "Checkpoint metric is absent from terminal task metrics: "
                f"checkpoint={checkpoint_name!r}, "
                f"available={sorted(episode_task_metrics)}."
            )
        return episode_task_metrics[checkpoint_name]

    def compute_camera_lookat(self) -> tuple[float, float, float]:
        assert self.original_demo is not None
        motion = self.original_demo.object_reference_motion
        obj_pos = motion.get("obj_pos", None)
        if obj_pos is None or len(obj_pos) == 0:
            return self.env.sim_cfg.camera_look_at

        center = np.asarray(obj_pos).mean(axis=0)
        return tuple(float(v) for v in center.tolist())

    def verify_target_object_motion(self, mode: str = "sim") -> None:

        if mode == "real":
            return

        if not self.task_cfg.save_reference_video:
            return
        assert self.original_demo is not None

        reference = self.original_demo.object_reference_motion
        retargeted_motion = self.original_demo.retargeted_motion

        pos = torch.from_numpy(reference["obj_pos"]).to(self.env.device)
        quat = torch.from_numpy(reference["obj_quat"]).to(self.env.device)
        if reference["obj_arti"] is None:
            n_frames = pos.shape[0]
            n_obj_joints = len(
                self.env.physics_manager.scene_entities["object"].dof_map
            )
            arti = torch.zeros((n_frames, n_obj_joints), dtype=torch.float32).to(
                self.env.device
            )
        else:
            arti = torch.from_numpy(reference["obj_arti"]).to(self.env.device)

        if retargeted_motion:
            reset_mgr = self.env.reset_manager
            pos = reset_mgr.frame_pos.to(self.env.device)
            quat = reset_mgr.frame_quat.to(self.env.device)
            arti = (
                None
                if reset_mgr.frame_arti is None
                else reset_mgr.frame_arti.to(self.env.device)
            )

            hand_jvals = {
                side: getattr(reset_mgr, f"frame_{side}_reset_jvals").to(
                    self.env.device
                )
                for side in self.active_hand_sides
            }
        else:
            hand_jvals = {}

        if mode != "sim":
            return

        physics_manager = self.env.physics_manager
        asset: ArtObject = self.env.physics_manager.scene_entities["object"]

        while True:
            physics_manager.start_recording()

            for frame_idx in range(self.total_steps):
                _pos = pos[frame_idx]
                _quat = quat[frame_idx]
                _arti = None if arti is None else arti[frame_idx]

                asset.set_state(
                    pos=_pos,
                    quat=_quat,
                    jpos=_arti,
                    env_idx=np.arange(self.env.num_envs),
                )

                for side in self.active_hand_sides:
                    robot: Hand = self.env.physics_manager.scene_entities[side]
                    hand_jpos = (
                        hand_jvals[side][frame_idx]
                        .unsqueeze(0)
                        .repeat(self.env.num_envs, 1)
                    )
                    robot.set_robot_state(
                        jpos=hand_jpos,
                        env_idx=np.arange(self.env.num_envs),
                    )

                physics_manager.step()
                physics_manager.video_recording_hook(timestep=0)

            fname = f"{physics_manager.logs_dir}/reference_motion.mp4"
            physics_manager.stop_recording(save_to_filename=fname, fps=30)

            replay = "n"
            if replay.lower() != "y":
                break
