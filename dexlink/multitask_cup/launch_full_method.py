"""Full-method supervisor preserving one run identity and the original absolute endpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket

import setuptools  # Match the canonical train_rl entry point before Torch/Triton imports.
import torch.multiprocessing as multiprocessing

from .full_method_config import parse_full_config


def main(argv: list[str] | None = None) -> int:
    config, options = parse_full_config(argv)
    cache = Path(config.cache_directory)
    (cache / "genesis").mkdir(parents=True, exist_ok=True)
    (cache / "gstaichi").mkdir(parents=True, exist_ok=True)
    os.environ.update(
        GS_CACHE_FILE_PATH=str(cache / "genesis"),
        TI_OFFLINE_CACHE_FILE_PATH=str(cache / "gstaichi"),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    os.environ.update(
        WANDB_ENTITY=config.wandb_entity, WANDB_PROJECT=config.wandb_project
    )
    from .train_full_method import train_full_rank

    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    operational = {
        name: getattr(options, name)
        for name in ("resume", "stop_after_control", "exit_before_initialization")
    }
    context = None
    pending_stop = False

    def stop(_signal, _frame) -> None:
        nonlocal pending_stop
        if context is None:
            pending_stop = True
        else:
            for process in context.processes:
                if process.is_alive():
                    os.kill(process.pid, signal.SIGUSR1)

    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    context = multiprocessing.start_processes(
        train_full_rank,
        args=(config, operational, port),
        nprocs=config.world_size,
        join=False,
        start_method="spawn",
    )
    if pending_stop:
        stop(signal.SIGUSR1, None)
    while not context.join(timeout=1):
        pass
    state = json.loads((Path(config.run_directory) / "status.json").read_text())
    return 0 if state["state"] == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
