"""Fixed positive-checkpoint cohorts with native episode progress and frozen functional predicates."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as distributed

from .episode_progress import episode_tracking_progress, progress_metrics
from .experiment_logging import EvaluationRound, SUCCESS_DEFINITIONS, TaskEvaluation
from .source_metrics import CupTrace, task_success
from .lenient_metrics import RelaxedCriteria, score_relaxed
from .training_state import atomic_json, runtime_state, restore_application_rng


@dataclass(frozen=True)
class EvaluationRecording:
    directory: Path
    robot_urdf: Path


def evaluate_native_scene(
    scene,
    policy,
    variants: list[dict],
    episodes: int,
    *,
    transfer_metrics: bool = False,
    recording: EvaluationRecording | None = None,
) -> list[dict]:
    labels = scene.task_ids
    tasks = labels.unique().tolist()
    if any(int((labels == task).sum()) != episodes for task in tasks):
        raise ValueError(
            "Dedicated evaluation scenes must contain exactly the declared episodes per task."
        )
    scene.env.passive_mask = None
    scene.env.passive_joint_targets = None
    scene.prepare_evaluation()
    horizon = max(
        variants[task]["frames"] + 10 + (15 if variants[task]["axis"] == "P" else 0)
        for task in tasks
    )
    passive_after = scene.bank.lengths[labels] + 10
    passive = torch.zeros_like(labels, dtype=torch.bool)
    held = torch.zeros((len(labels), 18), device=scene.env.device)
    traces = {
        name: []
        for name in (
            "pose",
            "velocity",
            "hand_contact",
            "table_contact",
            "table_upward_force",
        )
    }
    episode_returns = torch.zeros(len(labels), dtype=torch.float64, device="cpu")
    motion = {name: [] for name in ("joints", "wrist", "table", "base")}
    first_slots = torch.stack([(labels == task).nonzero()[0, 0] for task in tasks])
    contact_start = min(
        (
            variants[task]["frames"] + 6
            for task in tasks
            if variants[task]["axis"] == "P"
        ),
        default=horizon + 1,
    )
    for step in range(horizon + 1):
        state = scene.physical_state()
        if transfer_metrics or step >= contact_start:
            hand_contact, table_contact, force = scene.placement_contacts()
        else:
            hand_contact = torch.zeros_like(labels, dtype=torch.bool)
            table_contact = torch.zeros_like(hand_contact)
            force = torch.zeros_like(labels, dtype=torch.float32)
        for name, value in zip(
            traces, (state.cup, state.cup_velocity, hand_contact, table_contact, force)
        ):
            traces[name].append(value.detach().cpu())
        if recording is not None:
            joints = scene.env.physics_manager.get_entity_dofs_position(
                scene.hand.robot, scene.hand.dof_map
            )
            table = scene.env.observation_manager.object_entities["table"].get_obs(
                rot_format="quat"
            )
            base = torch.cat(
                (scene.hand.robot.get_pos(), scene.hand.robot.get_quat()), -1
            )
            if joints.shape != (len(labels), 18) or base.shape != (len(labels), 7):
                raise ValueError(
                    "Recorded robot state must have 18 joints and a world base pose per environment."
                )
            for name, value in zip(motion, (joints, state.wrist, table, base)):
                motion[name].append(value[first_slots].detach().cpu())
        if step == horizon:
            break
        newly_passive = (scene.clocks >= passive_after) & ~passive
        held[newly_passive] = scene.env.previous_position_targets_by_side["right"][
            newly_passive
        ]
        passive |= newly_passive
        scene.env.passive_mask, scene.env.passive_joint_targets = passive, held
        with torch.no_grad():
            residual = policy(scene.inputs())[0].mean
        reward, _, _ = scene.step(residual)
        if transfer_metrics:
            if reward.shape != (len(labels),) or not torch.isfinite(reward).all():
                raise ValueError(
                    "Transfer evaluation requires one finite native reward per environment control."
                )
            episode_returns += (
                torch.where(step < passive_after, reward, torch.zeros_like(reward))
                .detach()
                .cpu()
            )
    stacked = {name: torch.stack(values) for name, values in traces.items()}
    results = []
    for recorded_task, task in enumerate(tasks):
        selected = (labels.cpu() == task).nonzero().flatten()
        trace = CupTrace(
            **{name: value[:, selected] for name, value in stacked.items()}
        )
        reference = scene.bank.cup[task, : variants[task]["frames"]].cpu()
        mass = scene.bank.geometry_metadata[scene.bank.geometry_index[task], 6].item()
        success = task_success(variants[task], trace, reference, mass)
        tracking = episode_tracking_progress(trace.pose, reference)
        context = scene.contexts[task]
        task_slots = selected.to(scene.env.device)
        recorded = context.eval_would_terminate_timestep[task_slots] >= 0
        native_progress = torch.where(
            recorded,
            context.eval_would_terminate_progress[task_slots],
            torch.ones(len(task_slots), device=scene.env.device),
        ).clamp(0, 1)
        results.append(
            dict(
                task_id=variants[task]["id"],
                successes=int(success.sum()),
                episodes=episodes,
                success_definition=SUCCESS_DEFINITIONS[variants[task]["axis"]],
                episode_progress=native_progress.mean().item(),
                pose_tracking_prefix_fraction=tracking["episode_progress"],
                reference_tracking_fraction=tracking["reference_tracking_fraction"],
            )
        )
        if transfer_metrics:
            relaxed = score_relaxed(
                variants[task], trace, reference, mass, RelaxedCriteria()
            )
            results[-1].update(
                lenient_successes=int(relaxed["relaxed_success"].sum()),
                episode_return_mean=episode_returns[selected].mean().item(),
                episode_returns=episode_returns[selected].tolist(),
                relaxed_tracking_fraction=relaxed["relaxed_tracking_fraction"]
                .mean()
                .item(),
                final_position_error_m=relaxed["final_position_error_m"].mean().item(),
                final_orientation_error_deg=relaxed["final_orientation_error_deg"]
                .mean()
                .item(),
            )
        if recording is not None:
            folder = recording.directory / variants[task]["id"]
            folder.mkdir(parents=True, exist_ok=False)
            count = (
                variants[task]["frames"]
                + 11
                + (15 if variants[task]["axis"] == "P" else 0)
            )
            payload = np.load(variants[task]["reference"], allow_pickle=True).item()
            values = {
                name: value[:count, 0].numpy() for name, value in vars(trace).items()
            }
            values.update(
                {
                    name: torch.stack(value)[:count, recorded_task].numpy()
                    for name, value in motion.items()
                }
            )
            values.update(
                reference=reference.numpy(),
                joint_names=payload["right_jnames"],
                scene_urdfs=np.array(variants[task]["scene_urdfs"]),
                robot_urdf=np.array(str(recording.robot_urdf)),
            )
            np.savez_compressed(folder / "episode0.npz", **values)
    scene.env.passive_mask = None
    scene.env.passive_joint_targets = None
    return results


def evaluate_full_method_checkpoint(
    training_scenes: list,
    evaluation_scenes: list,
    policy,
    exploration,
    variants: list[dict],
    config,
    seal: dict,
    logger,
    run,
) -> None:
    if seal["control_step"] <= 0:
        raise ValueError("Never evaluate a timestep-zero full-method policy.")
    saved = runtime_state(training_scenes, exploration)
    transfer = config.transfer.arm != "source"
    outcomes = [
        row
        for scene in evaluation_scenes
        for row in evaluate_native_scene(
            scene,
            policy,
            variants,
            config.evaluation_episodes,
            transfer_metrics=transfer,
        )
    ]
    # Evaluation has its own physical scenes. Restore application RNGs and verify training state remains intact.
    restore_application_rng(saved)
    collected = [None] * config.world_size
    distributed.all_gather_object(collected, outcomes)
    if distributed.get_rank() == 0:
        rows = [row for rank_rows in collected for row in rank_rows]
        cohort = EvaluationRound(
            logger.tasks,
            seal["control_step"],
            seal["files"]["shared.pt"],
            config.evaluation_episodes,
        )
        payload = {}
        for row in rows:
            payload.update(
                cohort.add(
                    TaskEvaluation(
                        control_step=seal["control_step"],
                        checkpoint_sha256=seal["files"]["shared.pt"],
                        **{
                            name: row[name]
                            for name in (
                                "task_id",
                                "successes",
                                "episodes",
                                "success_definition",
                            )
                        },
                    )
                )
            )
        payload.update(
            progress_metrics(rows, tuple(task.task_id for task in logger.tasks))
        )
        for row in rows:
            payload[f"eval/task/{row['task_id']}/pose_tracking_prefix_fraction"] = row[
                "pose_tracking_prefix_fraction"
            ]
        payload["eval/pose_tracking_prefix_fraction"] = sum(
            row["pose_tracking_prefix_fraction"] for row in rows
        ) / len(rows)
        if transfer:
            payload["eval/lenient_success_rate_macro"] = sum(
                row["lenient_successes"] / row["episodes"] for row in rows
            ) / len(rows)
            for metric in (
                "episode_return_mean",
                "relaxed_tracking_fraction",
                "final_position_error_m",
                "final_orientation_error_deg",
            ):
                payload["eval/" + metric] = sum(row[metric] for row in rows) / len(rows)
            for row in rows:
                payload[f"eval/task/{row['task_id']}/lenient_success_rate"] = (
                    row["lenient_successes"] / row["episodes"]
                )
        directory = Path(config.run_directory) / "evaluations"
        directory.mkdir(exist_ok=True)
        atomic_json(
            directory / f"{seal['control_step']}.json",
            dict(
                control_step=seal["control_step"],
                checkpoint_sha256=seal["files"]["shared.pt"],
                results=rows,
                metrics=payload,
                evaluation_seed=42,
                seed_sd=None,
                episode_progress_definition="Native eval would-termination progress; capped at1 before appended passive frames.",
                success_predicates="Unchanged frozen multi-task/C02 source_metrics.py",
            ),
        )
        run.log(payload, commit=True)
    distributed.barrier()
