from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE
from dexlink.src.reference.arctic.runtime import materialize_runtime_transfer_sequence
from dexlink.src.reference.arctic.train import RewardVariant, register_training_preset
from dexlink.src.reference.arctic.transfer_config import get_transfer
from dexlink.scripts.rl.train_rl import main as train_rl


@dataclass(frozen=True)
class TrainTransferArgs:
    transfer: str
    workspace: Path
    seed: int
    runname: str | None
    local: bool
    reward_variant: RewardVariant
    timesteps: int


def parse_args(argv: list[str] | None = None) -> tuple[TrainTransferArgs, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train the residual RL method on one assembled ARCTIC transfer."
    )
    parser.add_argument("--transfer", required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runname")
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--reward-variant",
        choices=("our_method", "no_template_reward"),
        default="our_method",
    )
    parser.add_argument("--timesteps", type=int, default=50_000)
    raw, remaining = parser.parse_known_args(argv)
    if raw.timesteps <= 0:
        raise ValueError(f"Expected --timesteps > 0, got {raw.timesteps}.")
    return (
        TrainTransferArgs(
            transfer=str(raw.transfer),
            workspace=raw.workspace.resolve(),
            seed=int(raw.seed),
            runname=raw.runname,
            local=bool(raw.local),
            reward_variant=raw.reward_variant,
            timesteps=int(raw.timesteps),
        ),
        remaining,
    )


def main(argv: list[str] | None = None) -> int:
    args, overrides = parse_args(argv)
    transfer = get_transfer(args.transfer)
    runname = args.runname or (
        f"{transfer.transfer_id}__{args.reward_variant}__"
        f"{args.timesteps}__seed{args.seed}"
    )
    runtime_sequence = materialize_runtime_transfer_sequence(
        transfer, args.workspace
    )
    preset = register_training_preset(
        exported_path=runtime_sequence,
        seed=args.seed,
        runname=runname,
        reward_variant=args.reward_variant,
        timesteps=args.timesteps,
    )
    command = ["--preset", preset]
    if args.local:
        command.append("--local")
    train_rl([*command, *overrides])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
