from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from dexlink.src.reference.arctic.assemble_dense_motion import main as assemble_dense_motion
from dexlink.src.reference.arctic.config import DEFAULT_WORKSPACE, get_clip
from dexlink.src.reference.arctic.prepare import materialize_support_block


@dataclass(frozen=True)
class AssembleSceneVariantArgs:
    clip: str
    workspace: Path
    overwrite: bool


def parse_args(argv: list[str] | None = None) -> AssembleSceneVariantArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble an ARCTIC scene variant while reusing its frozen hand "
            "retargeting outputs."
        )
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--overwrite", action="store_true")
    raw = parser.parse_args(argv)
    return AssembleSceneVariantArgs(
        clip=str(raw.clip),
        workspace=raw.workspace.resolve(),
        overwrite=bool(raw.overwrite),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    clip = get_clip(args.clip)
    if clip.retargeting_source_clip_id is None:
        raise ValueError(
            f"Clip {clip.clip_id!r} is not a scene variant: "
            "retargeting_source_clip_id is unset."
        )

    # Scene variants reuse the frozen robot motion but materialize independent geometry.
    materialize_support_block(clip, args.workspace)
    command = ["--clip", clip.clip_id, "--workspace", str(args.workspace)]
    if args.overwrite:
        command.append("--overwrite")
    return assemble_dense_motion(command)


if __name__ == "__main__":
    raise SystemExit(main())
