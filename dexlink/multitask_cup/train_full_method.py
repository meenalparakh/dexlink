"""Synchronous source PPO inheriting the native full-method recipe."""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import time
import uuid
import xml.etree.ElementTree as ET

import setuptools  # Match the canonical train_rl import order in spawned learning ranks.
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

from .experiment_logging import (
    MultitaskRunLogger,
    SUCCESS_DEFINITIONS,
    TaskDefinition,
    configure_wandb_storage,
)
from .full_method_checkpoint import save_full_method_checkpoint
from .full_method_config import (
    FULL_METHOD_PRESET,
    FullMethodRunConfig,
    evaluation_due,
    next_checkpoint_control,
)
from .full_method_evaluation import evaluate_full_method_checkpoint
from .full_method_learner import update_full_method
from .full_method_policy import NativeObservationLayout, make_full_method_policy
from .full_method_rollout import collect_full_method
from .learner import parameter_digest
from .task_buffers import TaskExploration
from .task_catalog import divide_tasks, file_sha256, validate_training_catalog
from .task_data import TaskBank
from .remote_logging import verify_remote
from .training_state import (
    PHYSICS_RESUME_LIMITATION,
    atomic_json,
    load_latest,
    prepare_run,
    restore_runtime,
)
from .transfer_initialization import initialize_fresh_policy

_stop_requested = False


def request_stop(_signal, _frame) -> None:
    global _stop_requested
    _stop_requested = True


def stop_requested(device: torch.device) -> bool:
    value = torch.tensor(int(_stop_requested), device=device, dtype=torch.long)
    distributed.all_reduce(value, op=distributed.ReduceOp.MAX)
    return bool(value.item())


def verify_frozen_inputs(config: FullMethodRunConfig) -> dict:
    source_path = Path(config.data) / "source-manifest.json"
    source = json.loads(source_path.read_text())
    for relative, expected in source["source_files"].items():
        if file_sha256(Path(__file__).resolve().parents[2] / relative) != expected:
            raise ValueError(f"Frozen full-method executable changed:{relative}.")
    asset_path = source_path.parent / "multitask-assets.json"
    if file_sha256(asset_path) != source["asset_manifest_sha256"]:
        raise ValueError("Frozen native asset manifest changed.")
    assets = json.loads(asset_path.read_text())
    for name, expected in {**assets["urdfs"], **assets["meshes"]}.items():
        if file_sha256(Path(name)) != expected:
            raise ValueError(f"Frozen robot/scene asset changed:{name}.")
    manifest = json.loads((Path(config.data) / "manifest.json").read_text())
    validate_training_catalog(manifest)
    for row in manifest["variants"]:
        if file_sha256(Path(row["reference"])) != row["reference_sha256"]:
            raise ValueError(f"Approved reference changed:{row['id']}.")
    for asset in manifest["assets"].values():
        path = Path(asset["export_urdf"])
        if file_sha256(path) != asset["export_urdf_sha256"]:
            raise ValueError(f"Approved cup/table URDF changed:{path}.")
        for mesh in ET.parse(path).getroot().findall(".//mesh"):
            if (
                file_sha256(
                    (path.parent / mesh.attrib["filename"]).resolve(strict=True)
                )
                not in asset["source_dependencies"].values()
            ):
                raise ValueError(f"Approved physical mesh changed:{path}.")
    return dict(
        source_manifest_sha256=file_sha256(source_path),
        code_files={
            path.name: file_sha256(path) for path in Path(__file__).parent.glob("*.py")
        },
    )


def finish_run(config: FullMethodRunConfig, counters: dict, run) -> None:
    root = Path(config.run_directory)
    if not config.validation_mode:
        final = load_latest(config)
        if (
            counters["control_step"] != config.control_steps
            or not (root / "evaluations" / f"{config.control_steps}.json").exists()
        ):
            raise RuntimeError(
                "Run completion requires its positive original endpoint and complete evaluation."
            )
        artifact = (
            "frozen-source-checkpoint.json"
            if config.transfer.arm == "source"
            else "completed-transfer-checkpoint.json"
        )
        atomic_json(
            root / artifact,
            dict(
                control_step=config.control_steps,
                checkpoint=str(Path(final[0]["directory"]) / "shared.pt"),
                checkpoint_sha256=final[0]["files"]["shared.pt"],
                config_sha256=config.digest(),
                source_manifest_sha256=file_sha256(Path(config.data) / "manifest.json"),
                source_task_count=len(config.task_indices),
                seed=config.seed,
                target_authoring_released=config.transfer.arm == "source",
                initialization=asdict(config.transfer),
                wandb_url=run.url,
                method="native full-method plus author-declared multi-task conditioning/local10 architecture",
            ),
        )
    atomic_json(
        root / "status.json", dict(state="complete", **counters, wandb_url=run.url)
    )


def train_full_rank(
    rank: int, config: FullMethodRunConfig, options: dict, port: int
) -> None:
    visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(visible) != config.world_size:
        raise ValueError("Expected one allocated CUDA device per full-method rank.")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[rank]
    cpus = sorted(os.sched_getaffinity(0))
    width = min(8, len(cpus) // config.world_size)
    if width < 1:
        raise ValueError("Every full-method GPU rank requires an allocated CPU core.")
    os.sched_setaffinity(0, cpus[rank * width : (rank + 1) * width])
    cache = Path(config.cache_directory)
    for name in ("genesis", "gstaichi", f"numba-rank-{rank}"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(
        GS_CACHE_FILE_PATH=str(cache / "genesis"),
        TI_OFFLINE_CACHE_FILE_PATH=str(cache / "gstaichi"),
        TI_CPU_MAX_NUM_THREADS=str(width),
        NUMBA_CACHE_DIR=str(cache / f"numba-rank-{rank}"),
    )
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    distributed.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=config.world_size,
        timeout=timedelta(minutes=90),
    )
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    root = Path(config.run_directory)
    run, logger, lease = None, None, None
    try:
        if rank == 0:
            lease, identity = prepare_run(config, options["resume"])
            frozen = verify_frozen_inputs(config)
            runtime_path = root / "runtime.json"
            if (
                runtime_path.exists()
                and json.loads(runtime_path.read_text())["code_files"]
                != frozen["code_files"]
            ):
                raise ValueError(
                    "Full-method resume code differs from the original frozen run."
                )
            configure_wandb_storage(root)
            import wandb

            variants = json.loads((Path(config.data) / "manifest.json").read_text())[
                "variants"
            ]
            name = (
                Path(config.data).name
                + "__"
                + ("validation" if config.validation_mode else config.transfer.arm)
                + "__seed42__"
                + root.name
                if config.transfer.arm == "source"
                else "transfer6__"
                + variants[config.task_indices[0]]["id"]
                + "__"
                + config.transfer.arm
                + "__"
                + ("validation" if config.validation_mode else "seed42")
                + "__"
                + root.name
            )
            run = wandb.init(
                entity=config.wandb_entity or None,
                project=config.wandb_project,
                mode=config.wandb_mode(),
                id=config.run_id,
                resume="allow" if options["resume"] else "never",
                dir=str(root),
                name=name,
                group=Path(config.data).name
                if config.transfer.arm == "source"
                else "transfer6__" + str(config.task_indices[0]),
                job_type="full-method-validation"
                if config.validation_mode
                else (
                    "source-pretraining"
                    if config.transfer.arm == "source"
                    else "target-adaptation"
                ),
                config={
                    **asdict(config),
                    "full_method_preset": FULL_METHOD_PRESET,
                    "resolved_native_method": asdict(config.method()),
                    "identity": identity,
                    "seed_sd": "unavailable; single seed",
                    "physics_resume_limitation": PHYSICS_RESUME_LIMITATION,
                    "architecture_changes": "Geometry/program conditioning, AdaRMSNorm and fixed local10; native observation/backbone/loss/optimizer retained.",
                    "reset_infrastructure_change": "Partial resets use native sampling/noise/direct state writes without stepping other tasks.",
                    "global_minibatch_size": config.method().agent.batch_size,
                },
            )
            variants = json.loads((Path(config.data) / "manifest.json").read_text())[
                "variants"
            ]
            definitions = tuple(
                TaskDefinition(
                    variants[index]["id"],
                    variants[index]["axis"],
                    SUCCESS_DEFINITIONS[variants[index]["axis"]],
                )
                for index in config.task_indices
            )
            logger = MultitaskRunLogger(
                run,
                definitions,
                logging_mode=config.wandb_mode(),
                episode_progress_definition="Native completed-episode span/remaining horizon during training; initial-state native would-termination progress during evaluation.",
            )
            run.config.update(
                {
                    "episode_progress_definition": "Native completed-episode fraction during training; initial-state native would-termination progress during evaluation.",
                    "pose_tracking_prefix_fraction": "Supplemental3cm/15deg contiguous tracking prefix; distinct from native episode progress.",
                }
            )
            if not runtime_path.exists():
                atomic_json(
                    runtime_path,
                    dict(
                        **frozen,
                        world_size=config.world_size,
                        wandb_url=run.url,
                        full_method_preset=FULL_METHOD_PRESET,
                        physics_resume_limitation=PHYSICS_RESUME_LIMITATION,
                    ),
                )
                atomic_json(
                    root / "resolved-native-method.json", asdict(config.method())
                )
        distributed.barrier()
        if options["exit_before_initialization"]:
            if rank == 0:
                atomic_json(
                    root / "status.json",
                    dict(state="interrupted_before_first_checkpoint", control_step=0),
                )
            return
        latest = load_latest(config) if options["resume"] else None
        counters = dict(
            control_step=0,
            policy_version=0,
            optimizer_updates=0,
            aggregate_transitions=0,
        )
        if latest is not None:
            counters.update({name: latest[1][name] for name in counters})
        if counters["control_step"] > config.control_steps:
            raise ValueError(
                "Resume exceeded the original full-method absolute endpoint."
            )
        evaluation_path = root / "evaluations" / f"{counters['control_step']}.json"
        if counters["control_step"] == config.control_steps and (
            not config.evaluation_interval or evaluation_path.exists()
        ):
            if rank == 0:
                finish_run(config, counters, run)
            return
        variants = json.loads((Path(config.data) / "manifest.json").read_text())[
            "variants"
        ]
        layout = divide_tasks(
            [variants[index] for index in config.task_indices], config.world_size
        )
        shards = [
            tuple(config.task_indices[index] for index in shard.task_indices)
            for shard in layout.shards
            if shard.rank == rank
        ]
        bank = TaskBank.load(Path(config.data) / "task-bank.pt", device)
        scenes, evaluation_scenes = [], []
        for owner in range(config.world_size):
            if owner == rank:
                from .full_method_environment import FullMethodGeometryEnv

                cpu_bank = bank.to("cpu")
                for tasks in shards:
                    scene = FullMethodGeometryEnv(cpu_bank, variants, tasks, config)
                    scene.reset_probe()
                    scenes.append(scene)
                    if config.evaluation_interval:
                        evaluation_scenes.append(
                            FullMethodGeometryEnv(
                                cpu_bank, variants, tasks, config, evaluation=True
                            )
                        )
            distributed.barrier()
            if rank == 0:
                run.log({"setup/ranks_ready": owner + 1}, commit=True)
        native_layout = NativeObservationLayout(
            tuple(scenes[0].env.observation_manager.obs_terms),
            tuple(scenes[0].env.observation_manager.term_dims),
        )
        layouts = [None] * config.world_size
        distributed.all_gather_object(layouts, asdict(native_layout))
        if any(item != layouts[0] for item in layouts):
            raise ValueError(
                "Native policy observation layouts differ across geometry/rank shards."
            )
        torch.manual_seed(config.seed)
        policy = make_full_method_policy(
            native_layout, config.method(), config.architecture
        ).to(device)
        initialization = (
            initialize_fresh_policy(policy, config) if latest is None else None
        )
        optimizer = policy.optimizer()
        if latest is not None:
            if latest[1]["schema"] != policy.checkpoint_schema():
                raise ValueError(
                    "Full-method policy input/backbone schema differs from its checkpoint."
                )
            policy.load_state_dict(latest[1]["policy"], strict=True)
            optimizer.load_state_dict(latest[1]["optimizer"])
        model = DistributedDataParallel(policy, device_ids=[0])
        exploration = TaskExploration(len(variants), config.seed)
        if latest is not None:
            state = torch.load(
                Path(latest[0]["directory"]) / f"rank-{rank}.pt",
                map_location="cpu",
                weights_only=True,
            )
            restore_runtime(scenes, exploration, state)
        if rank == 0:
            if initialization is not None:
                atomic_json(root / "initialization.json", initialization)
            logger.training_step = counters["control_step"]
            events = root / "startup-events"
            events.mkdir(exist_ok=True)
            atomic_json(
                events / f"{uuid.uuid4().hex}.json",
                dict(
                    resume=options["resume"],
                    **counters,
                    config_sha256=config.digest(),
                    original_endpoint=config.control_steps,
                    restored_policy_digest=parameter_digest(policy),
                    optimizer_steps_before=sorted(
                        {
                            int(value["step"].item())
                            for value in optimizer.state.values()
                        }
                    ),
                    pre_first_checkpoint_restart=options["resume"] and latest is None,
                    restored_seal=str(Path(latest[0]["directory"]) / "checkpoint.json")
                    if latest
                    else None,
                ),
            )
            atomic_json(root / "layout.json", layout.to_dict())
            atomic_json(
                root / "status.json",
                dict(state="training", **counters, wandb_url=run.url),
            )
        if (
            latest is not None
            and not evaluation_path.exists()
            and evaluation_due(config, counters["control_step"])
        ):
            evaluate_full_method_checkpoint(
                scenes,
                evaluation_scenes,
                policy,
                exploration,
                variants,
                config,
                latest[0],
                logger,
                run,
            )
        first = True
        while counters["control_step"] < config.control_steps:
            step = counters["control_step"]
            boundary = next_checkpoint_control(config, step)
            horizon = min(config.rollout_steps, boundary - step)
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            batch = collect_full_method(
                scenes, policy, exploration, horizon, counters["policy_version"], config
            )
            collected = time.monotonic()
            metrics = update_full_method(
                model, optimizer, batch, bank, config, counters["policy_version"]
            )
            counters["control_step"] += horizon
            counters["policy_version"] += 1
            counters["optimizer_updates"] += metrics["optimizer_updates_this_rollout"]
            counters["aggregate_transitions"] += (
                horizon * len(config.task_indices) * config.environments_per_task
            )
            step = counters["control_step"]
            requested = stop_requested(device) or (
                options["stop_after_control"] and step >= options["stop_after_control"]
            )
            save_now = first or requested or step == boundary
            seal = (
                save_full_method_checkpoint(
                    config, policy, optimizer, counters, scenes, exploration
                )
                if save_now
                else None
            )
            free, total = torch.cuda.mem_get_info(device)
            memory = torch.tensor(
                [torch.cuda.max_memory_allocated(device), total - free],
                device=device,
                dtype=torch.float64,
            )
            distributed.all_reduce(memory, op=distributed.ReduceOp.MAX)
            if rank == 0:
                for name, value in list(metrics.items()):
                    if name.startswith("task_index/"):
                        _, task, metric = name.split("/")
                        metrics[f"task/{variants[int(task)]['id']}/{metric}"] = value
                metrics.update(
                    dict(
                        aggregate_transitions=counters["aggregate_transitions"],
                        rollout_seconds=collected - started,
                        update_seconds=time.monotonic() - collected,
                        task_coverage=len(config.task_indices),
                        world_size=config.world_size,
                        cuda_peak_allocated_gib_max=memory[0].item() / 1024**3,
                        cuda_device_used_gib_max=memory[1].item() / 1024**3,
                    )
                )
                logger.log_training(step, metrics)
                atomic_json(
                    root / "status.json",
                    dict(
                        state="training",
                        **counters,
                        wandb_url=run.url,
                        latest_seal=str(Path(seal["directory"]) / "checkpoint.json")
                        if seal
                        else json.loads((root / "latest.json").read_text())["seal"],
                    ),
                )
                if first and config.wandb_mode() == "online":
                    verify_remote(
                        f"{config.wandb_entity}/{config.wandb_project}/{config.run_id}",
                        step,
                    )
                elif first:
                    atomic_json(
                        root / "offline-logging.json",
                        dict(
                            mode="offline",
                            pending_sync=True,
                            run_id=config.run_id,
                            entity=config.wandb_entity or None,
                            project=config.wandb_project,
                            control_step=step,
                        ),
                    )
            distributed.barrier()
            first = False
            if requested:
                if rank == 0:
                    atomic_json(
                        root / "status.json",
                        dict(state="paused_at_checkpoint_boundary", **counters),
                    )
                return
            if evaluation_due(config, step):
                evaluate_full_method_checkpoint(
                    scenes,
                    evaluation_scenes,
                    policy,
                    exploration,
                    variants,
                    config,
                    seal,
                    logger,
                    run,
                )
            if stop_requested(device):
                if rank == 0:
                    atomic_json(
                        root / "status.json",
                        dict(state="paused_at_checkpoint_boundary", **counters),
                    )
                return
        if rank == 0:
            finish_run(config, counters, run)
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
            run = None
        raise
    finally:
        if run is not None:
            run.finish()
        if lease is not None:
            lease.close()
        distributed.destroy_process_group()
