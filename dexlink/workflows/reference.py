"""Build a reference from portable YAML, with optional packaged template reuse."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import yaml
from .catalog import ROOT, load_catalog
from dexlink.src.reference.task_io import load_task_spec
from dexlink.utils.retargeting.path_utils import get_logs_dir


@dataclass(frozen=True)
class ReferenceArguments:
    specification: Path
    output: Path
    density: float
    max_workers: int = 4
    skip_retargeting: bool = False
    refine_collisions: bool = False


def generate_reference(arguments: ReferenceArguments) -> None:
    from dexlink.scripts.reference.run_task_pipeline import PipelineArgs, run_pipeline

    specification = arguments.specification.resolve(strict=True)
    output = arguments.output.resolve()
    output_root = get_logs_dir().resolve()
    if output.suffix != ".npy" or arguments.max_workers < 1 or arguments.density <= 0:
        raise ValueError(
            "Expected a .npy output, positive density and max-workers >= 1."
        )
    if not output.is_relative_to(output_root):
        raise ValueError(
            f"Reference outputs must be inside DEXLINK_OUTPUTS ({output_root}); received {output}."
        )
    if output.exists():
        raise FileExistsError(f"Reference output already exists: {output}.")
    prepared = deepcopy(load_task_spec(specification))
    logical_output = Path("outputs") / output.relative_to(output_root)
    prepared["trajectory"]["output_motion_path"] = str(
        logical_output.with_suffix(".npz")
    )
    prepared["trajectory"]["exported_rl_path"] = str(logical_output)
    if arguments.skip_retargeting and specification.is_relative_to(ROOT):
        matches = [
            task
            for task in load_catalog().values()
            if task.get("spec") == str(specification.relative_to(ROOT))
        ]
        if matches:
            templates = matches[0].get("templates", {})
            for keyframe in prepared["keyframes"]:
                if keyframe["id"] in templates:
                    keyframe["from_retarget"]["motion_path"] = templates[keyframe["id"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared_path = output.with_suffix(".spec.yaml")
    prepared_path.write_text(yaml.safe_dump(prepared, sort_keys=False))
    result = run_pipeline(
        PipelineArgs(
            task=prepared_path,
            max_workers=arguments.max_workers,
            retarget_only_frames=20,
            density=arguments.density,
            skip_retargeting=arguments.skip_retargeting,
            skip_reference_build=False,
            refine_collisions=arguments.refine_collisions,
            allow_unresolved_collisions=False,
            retarget_hand_type="xhand",
            refined_output=None,
        )
    )
    print(result)
