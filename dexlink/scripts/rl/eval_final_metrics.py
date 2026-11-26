#!/usr/bin/env python
"""Run batch simulation evaluation and compute final experiment rollout metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import setuptools  # noqa: F401
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from dexlink.src.rl.core_agent.wandb_logger import WandbLogger
from dexlink.utils.retargeting.path_utils import get_logs_dir
from dexlink.scripts.rl.eval_sim import (
    _get_eval_preset_overrides,
    _get_override_value,
    _infer_run_dir_from_policy_checkpoint,
    _load_training_run_config,
    _resolve_policy_checkpoint,
    convert_to_local_paths,
)
from dexlink.src.rl.agent.ppo_eval import PPOEval
from dexlink.src.rl.config.base import (
    PRESETS,
    apply_overrides,
    apply_overrides_from_dict,
    extract_overrides,
)
from dexlink.src.rl.config.final_eval_wandb import FinalEvalWandbUploadCfg
from dexlink.src.rl.config.seq_oakink_presets import required_seq_task_settings
from dexlink.src.rl.multi_object.env_factory import create_env
from dexlink.utils.rl.final_eval_wandb import upload_final_eval_metrics
from dexlink.utils.rl.final_rollout_metrics import (
    FinalRolloutMetrics,
    FinalRolloutMetricsConfig,
)


@dataclass(frozen=True)
class FinalMetricEvalArgs:
    preset: str
    rl_run_dir: str | None
    ckpt_timestep: int | None
    best_checkpoint: bool
    num_episodes: int
    max_parallel_envs: int
    output_dir: Path | None
    seed: int
    greedy_actions: bool
    max_steps: int | None
    lift_success_threshold_m: float
    dtw_window_fraction: float
    compute_template_metrics: bool
    record_video: bool
    wandb_mode: Literal["disabled", "offline", "online"]
    wandb_project: str | None
    wandb_entity: str | None
    wandb_run_name: str | None
    wandb_group: str | None
    record_policy_inputs: bool = False


def parse_args(argv: list[str] | None = None) -> tuple[FinalMetricEvalArgs, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained simulation policy and compute final metrics."
    )
    parser.add_argument("--preset", type=str, default="seq_oakink_eval_sim")
    parser.add_argument("--rl_run_dir", type=str, default=None)
    parser.add_argument("--ckpt_timestep", type=int, default=None)
    parser.add_argument(
        "--best_checkpoint",
        action="store_true",
        help="Evaluate <rl_run_dir>/models/best/policy.pt.",
    )
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_parallel_envs", type=int, default=50)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record_policy_inputs", action="store_true", help="Save pre-step policy observations and actions in rollout chunks.")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lift_success_threshold_m", type=float, default=0.15)
    parser.add_argument("--dtw_window_fraction", type=float, default=0.10)
    parser.add_argument(
        "--compute_template_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help=(
            "Record environment zero to rollout.mp4 in the output directory. "
            "All requested episodes must fit in one parallel rollout chunk."
        ),
    )
    parser.add_argument(
        "--wandb_mode",
        choices=("disabled", "offline", "online"),
        default="disabled",
        help="Upload aggregate metrics, per-episode metrics, and metadata to W&B.",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        help="W&B project; defaults to the training run's logger project.",
    )
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default=None)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--greedy_actions",
        action="store_true",
        dest="greedy_actions",
        default=True,
    )
    action_group.add_argument(
        "--sample_actions",
        action="store_false",
        dest="greedy_actions",
    )
    parsed, unknown = parser.parse_known_args(argv)
    args = FinalMetricEvalArgs(
        preset=str(parsed.preset),
        rl_run_dir=parsed.rl_run_dir,
        ckpt_timestep=parsed.ckpt_timestep,
        best_checkpoint=bool(parsed.best_checkpoint),
        num_episodes=int(parsed.num_episodes),
        max_parallel_envs=int(parsed.max_parallel_envs),
        output_dir=parsed.output_dir,
        seed=int(parsed.seed),
        greedy_actions=bool(parsed.greedy_actions),
        max_steps=parsed.max_steps,
        lift_success_threshold_m=float(parsed.lift_success_threshold_m),
        dtw_window_fraction=float(parsed.dtw_window_fraction),
        compute_template_metrics=bool(parsed.compute_template_metrics),
        record_video=bool(parsed.record_video),
        record_policy_inputs=parsed.record_policy_inputs,
        wandb_mode=parsed.wandb_mode,
        wandb_project=parsed.wandb_project,
        wandb_entity=parsed.wandb_entity,
        wandb_run_name=parsed.wandb_run_name,
        wandb_group=parsed.wandb_group,
    )
    return args, unknown


def _validate_args(args: FinalMetricEvalArgs) -> None:
    if args.best_checkpoint and args.rl_run_dir is None:
        raise ValueError("--best_checkpoint requires --rl_run_dir.")
    if args.best_checkpoint and args.ckpt_timestep is not None:
        raise ValueError(
            "--best_checkpoint and --ckpt_timestep are mutually exclusive."
        )
    if args.ckpt_timestep is not None and args.ckpt_timestep <= 0:
        raise ValueError(
            "Evaluation checkpoints must have a positive training timestep, got "
            f"--ckpt_timestep={args.ckpt_timestep}."
        )
    if args.num_episodes <= 0:
        raise ValueError(f"Expected --num_episodes > 0, got {args.num_episodes}.")
    if args.max_parallel_envs <= 0:
        raise ValueError(
            f"Expected --max_parallel_envs > 0, got {args.max_parallel_envs}."
        )
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError(f"Expected --max_steps > 0, got {args.max_steps}.")
    if args.lift_success_threshold_m <= 0.0:
        raise ValueError(
            "Expected --lift_success_threshold_m > 0, got "
            f"{args.lift_success_threshold_m}."
        )
    if not (0.0 <= args.dtw_window_fraction <= 1.0):
        raise ValueError(
            "Expected --dtw_window_fraction in [0, 1], got "
            f"{args.dtw_window_fraction}."
        )
    if args.record_video and args.num_episodes > args.max_parallel_envs:
        raise ValueError(
            "--record_video requires all episodes to fit in one rollout chunk, "
            "but got --num_episodes "
            f"{args.num_episodes} and --max_parallel_envs {args.max_parallel_envs}."
        )
    if args.wandb_project is not None and not args.wandb_project.strip():
        raise ValueError("Expected --wandb_project to be non-empty when provided.")
    if args.wandb_entity is not None and not args.wandb_entity.strip():
        raise ValueError("Expected --wandb_entity to be non-empty when provided.")
    if args.wandb_run_name is not None and not args.wandb_run_name.strip():
        raise ValueError("Expected --wandb_run_name to be non-empty when provided.")
    if args.wandb_group is not None and not args.wandb_group.strip():
        raise ValueError("Expected --wandb_group to be non-empty when provided.")


def _default_output_dir(run_dir: Path, checkpoint_label: str) -> Path:
    return run_dir / "final_eval_metrics" / f"ckpt_{checkpoint_label}"


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _step_ground_truth(info: dict[str, Any]) -> dict[str, Any]:
    pre_reset = info.get("pre_reset_gt_info")
    if pre_reset is not None:
        return pre_reset
    gt_info = info.get("gt_info")
    if gt_info is None:
        raise ValueError("Expected env.step(..., compute_gt=True) to return gt_info.")
    return gt_info


def _extract_object_gt(
    *,
    gt: dict[str, Any],
    object_ids: tuple[str, ...],
    max_arti_dim: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    object_pos = []
    object_quat = []
    object_arti = []
    for object_id in object_ids:
        if "objects" in gt:
            object_gt = gt["objects"][object_id]
        elif object_id == "object":
            object_gt = gt["object"]
        else:
            raise ValueError(
                f"Ground-truth info is missing object {object_id!r}; keys={list(gt)}."
            )
        pos = _to_numpy(object_gt["pos"])[:count]
        quat = _to_numpy(object_gt["quat"])[:count]
        arti_raw = _to_numpy(object_gt["arti"])[:count]
        if pos.shape != (count, 3):
            raise ValueError(
                f"Expected object pos shape {(count, 3)}, got {pos.shape}."
            )
        if quat.shape != (count, 4):
            raise ValueError(
                f"Expected object quat shape {(count, 4)}, got {quat.shape}."
            )
        if arti_raw.ndim != 2 or arti_raw.shape[0] != count:
            raise ValueError(
                f"Expected object arti shape ({count}, A), got {arti_raw.shape}."
            )
        if arti_raw.shape[1] > max_arti_dim:
            raise ValueError(
                f"Object articulation dim {arti_raw.shape[1]} exceeds reference dim {max_arti_dim}."
            )
        arti = np.zeros((count, max_arti_dim), dtype=np.float32)
        arti[:, : arti_raw.shape[1]] = arti_raw.astype(np.float32, copy=False)
        object_pos.append(pos)
        object_quat.append(quat)
        object_arti.append(arti)
    return (
        np.stack(object_pos, axis=1).astype(np.float32, copy=False),
        np.stack(object_quat, axis=1).astype(np.float32, copy=False),
        np.stack(object_arti, axis=1).astype(np.float32, copy=False),
    )


def _extract_hand_gt(
    *,
    gt: dict[str, Any],
    active_hand_sides: tuple[str, ...],
    count: int,
) -> dict[str, np.ndarray]:
    out = {}
    for side in active_hand_sides:
        if side not in gt:
            raise ValueError(f"Ground-truth info is missing active hand {side!r}.")
        hand_gt = gt[side]
        for field, dim in (("wrist_pos", 3), ("wrist_quat", 4)):
            value = _to_numpy(hand_gt[field])[:count]
            if value.shape != (count, dim):
                raise ValueError(
                    f"Expected {side}.{field} shape {(count, dim)}, got {value.shape}."
                )
            out[f"{side}_{field}"] = value.astype(np.float32, copy=False)
        jvals = _to_numpy(hand_gt["jvals"])[:count]
        if jvals.ndim != 2 or jvals.shape[0] != count:
            raise ValueError(
                f"Expected {side}.jvals shape ({count}, D), got {jvals.shape}."
            )
        out[f"{side}_jvals"] = jvals.astype(np.float32, copy=False)
    return out


def _collect_rollout_chunk(
    *,
    agent: PPOEval,
    metrics: FinalRolloutMetrics,
    active_count: int,
    seed: int,
    greedy_actions: bool,
    max_steps: int,
    video_path: Path | None = None,
    record_policy_inputs: bool = False,
) -> dict[str, np.ndarray]:
    env = agent.env
    obs, _ = env.reset(seed=seed, options={"gt_info": True})
    obs = agent._policy_obs(obs)
    if agent.history_len > 0:
        agent.reset_history_buffer()

    start_timestep = env.start_timestep.detach().cpu().numpy()[:active_count].copy()
    done_seen = np.zeros(active_count, dtype=bool)
    records: dict[str, list[np.ndarray]] = {
        "reward": [],
        "done": [],
        "terminated": [],
        "truncated": [],
        "action_command_timestep": [],
        "command_timestep": [],
        "object_pos": [],
        "object_quat": [],
        "object_arti": [],
    }
    for side in metrics.active_hand_sides:
        records[f"{side}_wrist_pos"] = []
        records[f"{side}_wrist_quat"] = []
        records[f"{side}_jvals"] = []
    if record_policy_inputs:
        records["policy_observation"] = []
        records["policy_action"] = []
    task_metric_names = tuple(env.task_manager.task_metric_names)
    for name in task_metric_names:
        records[f"task_metric__{name}"] = []

    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        env.physics_manager.start_recording()

    try:
        with torch.no_grad():
            for _step_idx in range(max_steps):
                action_command_timestep = _to_numpy(env.command_time_step)[
                    :active_count
                ].astype(np.int64)
                history = None
                if agent.history_len > 0:
                    history_obs, _, history_dones = (
                        agent.history.get_previous_timesteps_data(
                            "obs", "actions", "dones", device=agent.device
                        )
                    )
                    history = agent._get_history_inputs(
                        history_obs=history_obs,
                        history_dones=history_dones,
                        obs_processor=agent.policy_obs_processor,
                    )
                action = agent.eval_policy_action(
                    obs,
                    history=history,
                    greedy_actions=greedy_actions,
                )
                # Capture inputs before stepping: environments can reset in-place.
                if record_policy_inputs:
                    for name, tensor in (("policy_observation", obs), ("policy_action", action)):
                        if tensor.ndim != 2 or tensor.shape[0] != env.num_envs or not torch.isfinite(tensor).all():
                            raise ValueError(f"Expected finite {name} with shape ({env.num_envs}, D), received {tuple(tensor.shape)}.")
                        records[name].append(tensor[:active_count].detach().cpu().numpy().copy())
                next_obs, reward, terminated, truncated, info = env.step(
                    action,
                    compute_gt=True,
                )
                if video_path is not None:
                    env.physics_manager.video_recording_hook(0)
                next_obs = agent._policy_obs(next_obs)
                done = terminated | truncated
                gt = _step_ground_truth(info)
                object_pos, object_quat, object_arti = _extract_object_gt(
                    gt=gt,
                    object_ids=metrics.object_ids,
                    max_arti_dim=metrics.ref_object_arti.shape[-1],
                    count=active_count,
                )
                hand_records = _extract_hand_gt(
                    gt=gt,
                    active_hand_sides=metrics.active_hand_sides,
                    count=active_count,
                )
                task_metric_gt = gt.get("task_metrics", {})
                if task_metric_names and not isinstance(task_metric_gt, dict):
                    raise ValueError(
                        "Ground-truth task_metrics must be a dict for configured task "
                        f"metrics {task_metric_names}, got {type(task_metric_gt).__name__}."
                    )

                records["reward"].append(
                    _to_numpy(reward)[:active_count].astype(np.float32)
                )
                records["done"].append(_to_numpy(done)[:active_count].astype(bool))
                records["terminated"].append(
                    _to_numpy(terminated)[:active_count].astype(bool)
                )
                records["truncated"].append(
                    _to_numpy(truncated)[:active_count].astype(bool)
                )
                records["action_command_timestep"].append(action_command_timestep)
                records["command_timestep"].append(
                    _to_numpy(gt["command_timestep"])[:active_count].astype(np.int64)
                )
                records["object_pos"].append(object_pos)
                records["object_quat"].append(object_quat)
                records["object_arti"].append(object_arti)
                for key, value in hand_records.items():
                    records[key].append(value)
                for name in task_metric_names:
                    if name not in task_metric_gt:
                        raise KeyError(
                            f"Ground-truth task_metrics is missing {name!r}; available "
                            f"keys are {sorted(task_metric_gt)}."
                        )
                    value = _to_numpy(task_metric_gt[name])[:active_count]
                    if value.shape != (active_count,):
                        raise ValueError(
                            f"Expected task metric {name!r} shape {(active_count,)}, "
                            f"got {value.shape}."
                        )
                    records[f"task_metric__{name}"].append(
                        value.astype(np.float32, copy=False)
                    )

                if agent.history_len > 0:
                    agent.history.add_samples(
                        obs=obs,
                        actions=action,
                        dones=done.unsqueeze(-1),
                    )
                done_seen |= _to_numpy(done)[:active_count].astype(bool)
                obs = next_obs
                if np.all(done_seen):
                    break
    finally:
        if video_path is not None and env.physics_manager.is_recording:
            # Genesis 0.3.3's MoviePy encoder omits the terminal image from the
            # decodable frame sequence. Append the unchanged terminal scene so
            # the saved video retains one frame for every recorded rollout step.
            env.physics_manager.video_recording_hook(0)
            env.physics_manager.stop_recording(
                save_to_filename=str(video_path),
                fps=30,
            )

    chunk = {key: np.stack(values, axis=0) for key, values in records.items()}
    chunk["start_timestep"] = start_timestep
    return chunk


def _episodes_from_chunk(
    chunk: dict[str, np.ndarray], count: int
) -> list[dict[str, Any]]:
    done = chunk["done"]
    if done.ndim != 2 or done.shape[1] < count:
        raise ValueError(f"Expected done shape (T, >= {count}), got {done.shape}.")
    episodes = []
    for env_idx in range(count):
        done_indices = np.where(done[:, env_idx])[0]
        length = int(done_indices[0] + 1) if done_indices.size else int(done.shape[0])
        episode = {
            key: value[:length, env_idx].copy()
            for key, value in chunk.items()
            if key != "start_timestep"
        }
        episode["start_timestep"] = int(chunk["start_timestep"][env_idx])
        episode["length"] = length
        episodes.append(episode)
    return episodes


def _write_episode_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Expected at least one episode metric row.")
    fieldnames = ["episode_index", *sorted({key for row in rows for key in row})]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for episode_index, row in enumerate(rows):
            writer.writerow({"episode_index": episode_index, **row})


def _write_summary_csv(path: Path, summary: dict[str, float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(summary):
            writer.writerow({"metric": key, "value": summary[key]})


def _write_reference_trajectory(
    path: Path,
    metrics: FinalRolloutMetrics,
) -> None:
    """Save the exact reference arrays used by the evaluation environment."""
    task_manager = metrics.env.task_manager
    phase_names = tuple(str(name) for name in getattr(task_manager, "phase_names", ()))
    phase_lengths = np.asarray(
        getattr(task_manager, "phase_lengths", np.empty(0, dtype=np.int64)),
        dtype=np.int64,
    )
    if phase_lengths.shape != (len(phase_names),):
        raise ValueError(
            "Expected phase_lengths to match phase_names, got "
            f"{phase_lengths.shape} and {len(phase_names)} names."
        )
    if (
        phase_lengths.size > 0
        and int(phase_lengths.sum()) != metrics.ref_object_pos.shape[0]
    ):
        raise ValueError(
            "Expected phase lengths to span the reference trajectory, got "
            f"{int(phase_lengths.sum())} frames for a "
            f"{metrics.ref_object_pos.shape[0]}-frame reference."
        )

    arrays: dict[str, np.ndarray] = {
        "object_ids": np.asarray(metrics.object_ids, dtype=str),
        "object_static": metrics.object_static.astype(bool, copy=False),
        "object_active_mask": metrics.object_active_mask.astype(bool, copy=False),
        "object_pos": metrics.ref_object_pos.astype(np.float32, copy=False),
        "object_quat": metrics.ref_object_quat.astype(np.float32, copy=False),
        "object_arti": metrics.ref_object_arti.astype(np.float32, copy=False),
        "active_hand_sides": np.asarray(metrics.active_hand_sides, dtype=str),
        "phase_names": np.asarray(phase_names, dtype=str),
        "phase_lengths": phase_lengths,
    }
    for side in metrics.active_hand_sides:
        hand = metrics.env.physics_manager.scene_entities[side]
        arrays[f"{side}_joint_names"] = np.asarray(hand.joint_names, dtype=str)
        arrays[f"{side}_finger_joint_indices"] = metrics.finger_joint_indices[side]
        arrays[f"{side}_jvals"] = metrics.ref_hand_jvals[side].astype(
            np.float32, copy=False
        )
        arrays[f"{side}_wrist_pos"] = metrics.ref_hand_wrist_pos[side].astype(
            np.float32, copy=False
        )
        arrays[f"{side}_wrist_quat"] = metrics.ref_hand_wrist_quat[side].astype(
            np.float32, copy=False
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _load_eval_config(args: FinalMetricEvalArgs, unknown: list[str]):
    overrides = extract_overrides(args, unknown)
    preset_overrides = _get_eval_preset_overrides(args.preset)
    resume_from_ckpt_override = _get_override_value(overrides, "agent.resume_from_ckpt")
    preset_resume_from_ckpt = PRESETS[args.preset].get("agent.resume_from_ckpt")
    if args.best_checkpoint:
        policy_ckpt = Path(args.rl_run_dir) / "models" / "best" / "policy.pt"
        if not policy_ckpt.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {policy_ckpt}")
        selection_metadata_path = policy_ckpt.parent / "metadata.json"
        if not selection_metadata_path.exists():
            raise FileNotFoundError(
                "Best checkpoint selection metadata not found: "
                f"{selection_metadata_path}"
            )
        selection_metadata = json.loads(selection_metadata_path.read_text())
        policy_timestep = selection_metadata.get("policy_timestep")
        if not isinstance(policy_timestep, int) or policy_timestep <= 0:
            raise ValueError(
                "Refusing to evaluate an untrained best checkpoint: expected a "
                "positive integer policy_timestep in "
                f"{selection_metadata_path}, got {policy_timestep!r}. Use an "
                "explicit positive --ckpt_timestep instead."
            )
    else:
        policy_ckpt = _resolve_policy_checkpoint(
            rl_run_dir=args.rl_run_dir,
            ckpt_timestep=args.ckpt_timestep,
            resume_from_ckpt=resume_from_ckpt_override or preset_resume_from_ckpt,
            logger_logdir=str(get_logs_dir()),
        )
    run_dir = _infer_run_dir_from_policy_checkpoint(policy_ckpt)
    exp_cfg = _load_training_run_config(run_dir)
    convert_to_local_paths(exp_cfg)
    exp_cfg = apply_overrides_from_dict(exp_cfg, preset_overrides)
    exp_cfg = apply_overrides_from_dict(
        exp_cfg,
        required_seq_task_settings(
            exp_cfg.env.task_cfg.oakink_exported_path,
            preset_name=exp_cfg.preset,
        ),
    )
    exp_cfg = apply_overrides_from_dict(
        exp_cfg,
        {
            "env.backend": "sim",
            "env.sim_cfg.num_envs": int(args.max_parallel_envs),
            "env.sim_cfg.show_viewer": False,
            "env.sim_cfg.apply_random_force": False,
            "env.rewards_cfg.last_reference_hold_steps": 10,
            "agent.resume": False,
            "agent.resume_start_timestep": -1,
            "agent.resume_latest_fixed_budget": False,
            "logger.disable": True,
            "logger.timestamp": False,
        },
    )
    if args.record_video:
        # Headless Genesis scenes only allocate camera buffers for explicitly
        # rendered environments. Without this, recording succeeds nominally but
        # produces no frames.
        exp_cfg = apply_overrides_from_dict(
            exp_cfg,
            {
                "env.sim_cfg.n_rendered_envs": 1,
                # PhysicsSim.video_recording_hook renders frames only when its
                # periodic recording path is enabled. The evaluator owns start
                # and stop timing, so a frequency of one simply enables the hook.
                "env.sim_cfg.video_save_freq": 1,
            },
        )
    if overrides:
        exp_cfg = apply_overrides(exp_cfg, overrides)
    exp_cfg = apply_overrides_from_dict(
        exp_cfg,
        {"env.rewards_cfg.last_reference_hold_steps": 10},
    )
    exp_cfg.agent.resume_from_ckpt = str(policy_ckpt)
    return exp_cfg, policy_ckpt, run_dir


def main(argv: list[str] | None = None) -> int:
    args, unknown = parse_args(argv)
    _validate_args(args)
    exp_cfg, policy_ckpt, run_dir = _load_eval_config(args, unknown)
    checkpoint_label = policy_ckpt.parent.name
    selection_metadata_path = policy_ckpt.parent / "metadata.json"
    checkpoint_selection = (
        json.loads(selection_metadata_path.read_text())
        if selection_metadata_path.exists()
        else None
    )
    if checkpoint_selection is not None and not isinstance(checkpoint_selection, dict):
        raise ValueError(
            f"Expected checkpoint selection metadata dict at {selection_metadata_path}, "
            f"got {type(checkpoint_selection).__name__}."
        )
    output_dir = args.output_dir or _default_output_dir(run_dir, checkpoint_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = exp_cfg.env
    if env_cfg.backend != "sim":
        raise ValueError(f"Expected sim eval backend, got {env_cfg.backend!r}.")
    logger_cfg = exp_cfg.logger
    agent_cfg = exp_cfg.agent
    eval_cfg = exp_cfg.eval
    eval_cfg.logdir = str(output_dir)
    agent_cfg.seed = env_cfg.seed
    agent_cfg.device = env_cfg.device

    logger = WandbLogger(
        logger_cfg,
        exp_seed=env_cfg.seed,
        experiment_dir=output_dir,
    )
    env = create_env(env_cfg)
    try:
        env.task_manager.verify_target_object_motion(mode=env_cfg.backend)
        agent_cfg.use_discriminator = env.use_discriminator
        logger.add_config({"experiment": asdict(exp_cfg)})
        logger.save_config(asdict(exp_cfg), name="final_eval_exp_cfg.json")
        agent = PPOEval(env, logger=logger, agent_cfg=agent_cfg, eval_cfg=eval_cfg)
        agent.load(policy_ckpt)
        agent.policy_net.eval()
        agent.policy_obs_processor.eval()
        agent.value_obs_processor.eval()
        agent.value_net.eval()
        if agent.use_discriminator:
            agent.discriminator.eval()

        env.set_mode("eval")
        metrics = FinalRolloutMetrics(
            env,
            FinalRolloutMetricsConfig(
                lift_success_threshold_m=args.lift_success_threshold_m,
                compute_template_metrics=args.compute_template_metrics,
                dtw_window_fraction=args.dtw_window_fraction,
                dtw_plot_dir=output_dir / "dtw_alignment_plots",
                dtw_plot_max_episodes=args.num_episodes,
            ),
        )
        reference_path = output_dir / "reference_trajectory.npz"
        _write_reference_trajectory(reference_path, metrics)
        default_max_steps = (
            env.task_manager.total_steps
            + int(env.cfg.rewards_cfg.last_reference_hold_steps)
            + 2
        )
        max_steps = args.max_steps or default_max_steps
        episode_rows: list[dict[str, float]] = []
        remaining = args.num_episodes
        chunk_idx = 0
        while remaining > 0:
            active_count = min(int(env.sim_cfg.num_envs), remaining)
            print(
                f"[final-eval] chunk={chunk_idx} active_envs={active_count} "
                f"remaining={remaining} max_steps={max_steps}"
            )
            chunk = _collect_rollout_chunk(
                agent=agent,
                metrics=metrics,
                active_count=active_count,
                seed=args.seed + chunk_idx,
                greedy_actions=args.greedy_actions,
                record_policy_inputs=args.record_policy_inputs,
                max_steps=max_steps,
                video_path=(output_dir / "rollout.mp4") if args.record_video else None,
            )
            chunk_path = output_dir / "rollouts" / f"chunk_{chunk_idx:04d}.npz"
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(chunk_path, **chunk)
            for episode in _episodes_from_chunk(chunk, active_count):
                episode_rows.append(
                    metrics.compute_episode_metrics(
                        episode,
                        episode_index=len(episode_rows),
                    )
                )
            remaining -= active_count
            chunk_idx += 1

        summary = FinalRolloutMetrics.summarize(episode_rows)
        metrics_csv = output_dir / "episode_metrics.csv"
        summary_json = output_dir / "summary_metrics.json"
        summary_csv = output_dir / "summary_metrics.csv"
        _write_episode_metrics(metrics_csv, episode_rows)
        summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _write_summary_csv(summary_csv, summary)
        serialized_args = asdict(args)
        serialized_args["output_dir"] = (
            str(args.output_dir) if args.output_dir is not None else None
        )
        metadata = {
            "evaluation_protocol_version": env.cfg.key_task_protocol_version,
            "uncertainty": {
                "estimand": "mean over evaluation episodes",
                "confidence_level": FinalRolloutMetrics.CONFIDENCE_LEVEL,
                "method": "percentile bootstrap over episodes",
                "bootstrap_samples": (FinalRolloutMetrics.CONFIDENCE_BOOTSTRAP_SAMPLES),
                "bootstrap_seed": FinalRolloutMetrics.CONFIDENCE_BOOTSTRAP_SEED,
            },
            "args": serialized_args,
            "policy_checkpoint": str(policy_ckpt),
            "run_dir": str(run_dir),
            "checkpoint_label": checkpoint_label,
            "checkpoint_selection": checkpoint_selection,
            "output_dir": str(output_dir),
            "num_episode_rows": len(episode_rows),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        print(f"[final-eval] wrote episode metrics -> {metrics_csv}")
        print(f"[final-eval] wrote summary metrics -> {summary_json}")
        print(f"[final-eval] wrote reference trajectory -> {reference_path}")
    finally:
        env.close()
    if args.wandb_mode != "disabled":
        wandb_project = args.wandb_project or str(logger_cfg.project)
        marker = upload_final_eval_metrics(
            FinalEvalWandbUploadCfg(
                output_dir=output_dir,
                project=wandb_project,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                run_name=(
                    args.wandb_run_name or f"{run_dir.name}__eval_{checkpoint_label}"
                ),
                group=args.wandb_group or str(logger_cfg.group),
            )
        )
        print(f"[final-eval] uploaded W&B metrics -> {marker['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
