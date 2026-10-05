from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE
from dexlink.src.reference.arctic.runtime import materialize_runtime_transfer_sequence
from dexlink.src.reference.arctic.train import ARCTIC_EVAL_PRESET, register_training_preset
from dexlink.src.reference.arctic.transfer_config import (
    ARCTIC_TRANSFER_SPECS,
    get_transfer,
)
from dexlink.src.rl.config.base import load_experiment_config


@dataclass(frozen=True)
class RenderTransferArgs:
    transfers: tuple[str, ...]
    workspace: Path
    output_root: Path
    worker_transfer: str | None


def parse_args(argv: list[str] | None = None) -> RenderTransferArgs:
    parser = argparse.ArgumentParser(
        description="Render corrected ARCTIC transfer references in Genesis."
    )
    parser.add_argument(
        "--transfer",
        action="append",
        choices=tuple(ARCTIC_TRANSFER_SPECS),
        dest="transfers",
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("logs/arctic_evals/transfer_reference_videos"),
    )
    parser.add_argument(
        "--worker-transfer",
        choices=tuple(ARCTIC_TRANSFER_SPECS),
        help=argparse.SUPPRESS,
    )
    raw = parser.parse_args(argv)
    transfers = tuple(raw.transfers or ARCTIC_TRANSFER_SPECS)
    return RenderTransferArgs(
        transfers=transfers,
        workspace=raw.workspace.resolve(),
        output_root=raw.output_root.resolve(),
        worker_transfer=raw.worker_transfer,
    )


def render_one(args: RenderTransferArgs, transfer_id: str) -> Path:
    from dexlink.src.rl.multi_object.env_factory import create_env

    transfer = get_transfer(transfer_id)
    sequence = materialize_runtime_transfer_sequence(transfer, args.workspace)
    register_training_preset(
        exported_path=sequence,
        seed=42,
        runname=f"reference_render__{transfer_id}",
        timesteps=8,
    )
    cfg = load_experiment_config(preset=ARCTIC_EVAL_PRESET, overrides=[])
    output_dir = args.output_root / transfer_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.env.device = "cuda"
    cfg.env.sim_cfg.num_envs = 1
    cfg.env.sim_cfg.show_viewer = False
    cfg.env.sim_cfg.n_rendered_envs = 1
    cfg.env.sim_cfg.apply_random_force = False
    cfg.env.sim_cfg.video_save_freq = 1
    cfg.env.sim_cfg.video_save_dir = str(output_dir)
    cfg.env.task_cfg.save_reference_video = True

    env = create_env(cfg.env)
    try:
        env.task_manager.verify_target_object_motion(mode="sim")
    finally:
        env.close()
    output = output_dir / "reference_motion.mp4"
    if not output.is_file():
        raise FileNotFoundError(f"Genesis did not write reference video: {output}.")
    print(f"[arctic-transfer-render] {transfer_id} -> {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_transfer is not None:
        render_one(args, args.worker_transfer)
        return 0

    script = Path(__file__).resolve()
    for transfer_id in args.transfers:
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--workspace",
                str(args.workspace),
                "--output-root",
                str(args.output_root),
                "--worker-transfer",
                transfer_id,
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
