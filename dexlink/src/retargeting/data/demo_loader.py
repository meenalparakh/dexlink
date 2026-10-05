from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from dexlink.src.retargeting.data.oakink_processor_grasp import OakinkProcessorGrasp
from dexlink.src.retargeting.data.oakink_processor_lift import OakinkProcessorLift
from dexlink.src.retargeting.data.process_arctic import RandomTrajProcessor, get_processor_cfg
from dexlink.utils.retargeting.path_utils import get_abs_cluster_path, get_logs_dir

if TYPE_CHECKING:
    from dexlink.src.rl.config.base import SceneCfg

MotionProcessor = Any

OAKINK_DATASETS = {"oakink", "oakink_grasp", "oakink_lift"}
SUPPORTED_DEMO_DATASETS = OAKINK_DATASETS | {"arctic", "random_traj"}


def is_oakink_dataset(demo_dataset: str) -> bool:
    return demo_dataset in OAKINK_DATASETS


def _validate_oakink_lift_payload_has_reach_frames(exported_path: str) -> None:
    raw = np.load(exported_path, allow_pickle=True)
    if not isinstance(raw, np.ndarray) or raw.shape != ():
        raise ValueError(
            f"Expected OakInk lift export at {exported_path!r} to contain a scalar "
            f"payload dict, got object of type {type(raw)} with shape "
            f"{getattr(raw, 'shape', None)}."
        )
    payload = raw.item()
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected OakInk lift export at {exported_path!r} to contain a dict, "
            f"got {type(payload)}."
        )
    if "n_reach_frames" not in payload:
        raise ValueError(
            f"OakInk lift export at {exported_path!r} is missing required key "
            "'n_reach_frames'. Re-export the lift payload with reach phase metadata."
        )


def load_demo_processor(
    scene_cfg: SceneCfg,
    *,
    start_timestep: int,
    end_timestep: int,
    ref_motion_noise_sig: tuple[float, float] | None = None,
) -> MotionProcessor:
    demo_dataset = scene_cfg.demo_dataset
    if demo_dataset not in SUPPORTED_DEMO_DATASETS:
        raise ValueError(
            f"Unsupported demo_dataset={demo_dataset!r}. "
            f"Expected one of {sorted(SUPPORTED_DEMO_DATASETS)}."
        )

    if demo_dataset in {"oakink", "oakink_grasp"}:
        processor_cls = OakinkProcessorGrasp if demo_dataset == "oakink_grasp" else None
        if processor_cls is None:
            from dexlink.src.retargeting.data.exported_oakink_processor import (
                ExportedOakInkProcessor,
            )

            processor_cls = ExportedOakInkProcessor
        return processor_cls(
            scene_cfg.oakink_exported_path,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            skip=scene_cfg.skip,
        )

    if demo_dataset == "oakink_lift":
        _validate_oakink_lift_payload_has_reach_frames(scene_cfg.oakink_exported_path)
        return OakinkProcessorLift(
            scene_cfg.oakink_exported_path,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            skip=scene_cfg.skip,
            support_height=scene_cfg.lift_support_height,
            support_margin=scene_cfg.lift_support_margin,
        )

    if demo_dataset == "arctic":
        processor_cfg = get_processor_cfg(scene_cfg.target_object)
        processor_cls = processor_cfg["processor_cls"]
        reference_urdf = (
            processor_cfg["reference_urdf"] or scene_cfg.reference_object_urdf
        )
        target_urdf = processor_cfg["target_urdf_path"] or scene_cfg.target_object_urdf
        return processor_cls(
            scene_cfg.original_demo_path,
            scene_cfg.target_object,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            skip=scene_cfg.skip,
            reference_urdf=reference_urdf,
            target_urdf=target_urdf,
            retargeted_results=scene_cfg.retargeted_results,
            retargeted_results_upsample=scene_cfg.retargeted_results_upsample,
            ref_motion_noise_sig=ref_motion_noise_sig or scene_cfg.ref_motion_noise_sig,
        )

    if scene_cfg.retargeted_results is None:
        raise ValueError(
            "SceneCfg.retargeted_results must be set when demo_dataset='random_traj'."
        )

    random_traj_path = get_logs_dir() / "trajopt_results" / scene_cfg.retargeted_results
    random_traj_path = get_abs_cluster_path(random_traj_path)
    return RandomTrajProcessor(
        retargeted_results=str(random_traj_path),
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        skip=scene_cfg.skip,
    )
