"""Run the category retargetings required by one or more task YAML files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import trimesh
from urdfpy import URDF

from dexlink.src.reference.category_paths import (
    category_prepared_target_dir,
    category_retarget_job_id,
    category_retarget_run_dir,
    category_transform_dir,
    normalize_retarget_hand_type,
    robot_model_for_hand_type,
)
from dexlink.src.reference.common.exp_info import (
    get_source_template,
    get_target_object,
)
from dexlink.src.reference.common.exp_schema import ArticulationRetargetSample
from dexlink.src.reference.task_io import load_task_spec
from dexlink.utils.retargeting.path_utils import (
    get_src,
    process_urdf,
    resolve_project_path,
)
from dexlink.utils.retargeting.urdf_utils import materialize_scaled_urdf


MOTION_FILENAME = "sequence_motion_lineseq.npz"


@dataclass(frozen=True)
class CategoryRetargetRequest:
    category: str
    source_index: int
    target_index: int
    hand_side: str
    hand_type: str
    articulation_sample_id: str | None = None
    retarget_run_dir: str | None = None

    @property
    def job_id(self) -> str:
        return category_retarget_job_id(
            self.category,
            self.source_index,
            self.target_index,
            self.articulation_sample_id,
        )

    def run_dir(self, root: Path) -> Path:
        if self.retarget_run_dir is not None:
            path = Path(self.retarget_run_dir)
            return path.resolve() if path.is_absolute() else (root / path).resolve()
        return category_retarget_run_dir(
            root,
            self.hand_type,
            self.category,
            self.source_index,
            self.target_index,
            self.articulation_sample_id,
        )


@dataclass(frozen=True)
class RunnerArgs:
    specs: tuple[Path, ...]
    max_workers: int
    retarget_only_frames: int
    hand_type: str
    preset: str
    dry_run: bool
    skip_existing: bool
    save_retarget_probes: bool


@dataclass(frozen=True)
class PreparedTarget:
    target_obj_path: Path | None
    target_bps_obj_path: Path | None
    target_demo_path: Path | None
    target_transform_path: Path | None
    target_scale: float
    articulated_urdf_path: Path | None
    articulation_sample: ArticulationRetargetSample | None


def _parse_args(argv: list[str] | None = None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        description="Run all category retargetings referenced by task YAML files."
    )
    parser.add_argument("--spec", type=Path, action="append", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--retarget-only-frames", type=int, default=20)
    parser.add_argument(
        "--hand-type", choices=("xhand", "inspire_hand"), default="xhand"
    )
    parser.add_argument("--preset", default="oakink_default_2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--save-retarget-probes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parsed = parser.parse_args(argv)
    return RunnerArgs(
        specs=tuple(path.resolve() for path in parsed.spec),
        max_workers=int(parsed.max_workers),
        retarget_only_frames=int(parsed.retarget_only_frames),
        hand_type=str(parsed.hand_type),
        preset=str(parsed.preset),
        dry_run=bool(parsed.dry_run),
        skip_existing=bool(parsed.skip_existing),
        save_retarget_probes=bool(parsed.save_retarget_probes),
    )


def _scene_object_source_target(spec: dict[str, Any], object_id: str) -> dict:
    matches = [
        item
        for item in spec.get("scene_objects", [])
        if str(item.get("id")) == object_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one scene object with id {object_id!r}, "
            f"found {len(matches)}."
        )
    source_target = matches[0].get("source_target")
    if not isinstance(source_target, dict):
        raise ValueError(
            f"Scene object {object_id!r} must define a source_target mapping."
        )
    return dict(source_target)


def _source_target_for_template(
    spec: dict[str, Any], template: dict[str, Any], object_id: str
) -> dict:
    if "source_target" in template:
        source_target = dict(template["source_target"])
    elif object_id != "asset" and "scene_objects" in spec:
        source_target = _scene_object_source_target(spec, object_id)
    elif "source_target" in spec:
        source_target = dict(spec["source_target"])
    else:
        raise ValueError(
            f"Template {template.get('id')!r} uses motion_path='auto' but has "
            "no source_target."
        )
    from_retarget = template["from_retarget"]
    if "articulation_sample_id" in from_retarget:
        source_target["articulation_sample_id"] = str(
            from_retarget["articulation_sample_id"]
        )
    return source_target


def _request_from_template(
    spec: dict[str, Any],
    template: dict[str, Any],
    *,
    object_id: str,
    default_hand_side: str,
    hand_type: str,
) -> CategoryRetargetRequest | None:
    from_retarget = template.get("from_retarget")
    if not isinstance(from_retarget, dict):
        return None
    if str(from_retarget.get("motion_path", "auto")) != "auto":
        return None
    if "retarget_job_id" in from_retarget:
        return None

    source_target = _source_target_for_template(spec, template, object_id)
    expected_keys = {"category", "source_index", "target_index"}
    missing = expected_keys - set(source_target)
    if missing:
        raise ValueError(
            f"Template {template.get('id')!r} source_target is missing "
            f"{sorted(missing)}."
        )
    hand_side = str(template.get("source_hand_side", default_hand_side))
    if hand_side not in {"left", "right"}:
        raise ValueError(
            f"Expected source hand side to be left or right, got {hand_side!r}."
        )
    retarget_run_dir = source_target.get("retarget_run_dir")
    return CategoryRetargetRequest(
        category=str(source_target["category"]),
        source_index=int(source_target["source_index"]),
        target_index=int(source_target["target_index"]),
        hand_side=hand_side,
        hand_type=normalize_retarget_hand_type(hand_type),
        articulation_sample_id=(
            None
            if source_target.get("articulation_sample_id") is None
            else str(source_target["articulation_sample_id"])
        ),
        retarget_run_dir=(
            None if retarget_run_dir is None else str(retarget_run_dir)
        ),
    )


def collect_category_retarget_requests(
    spec: dict[str, Any], hand_type: str
) -> tuple[CategoryRetargetRequest, ...]:
    """Collect and deduplicate all automatic category retargets in a task."""

    requests_by_job: dict[tuple[str, str], CategoryRetargetRequest] = {}
    default_hand = str(spec.get("hand_side", "right"))
    for keyframe in spec["keyframes"]:
        if "bimanual_from_retarget" in keyframe:
            parent_object_id = str(keyframe.get("object_id", "asset"))
            bimanual = keyframe["bimanual_from_retarget"]
            if not isinstance(bimanual, dict):
                raise TypeError(
                    f"Expected bimanual_from_retarget mapping in "
                    f"{keyframe.get('id')!r}."
                )
            for side, raw_side_template in bimanual.items():
                if not isinstance(raw_side_template, dict):
                    raise TypeError(
                        f"Expected bimanual side {side!r} to be a mapping."
                    )
                side_template = dict(raw_side_template)
                if "from_retarget" not in side_template:
                    continue
                object_id = str(side_template.get("object_id", parent_object_id))
                request = _request_from_template(
                    spec,
                    side_template,
                    object_id=object_id,
                    default_hand_side=str(side),
                    hand_type=hand_type,
                )
                if request is not None:
                    _insert_unique_request(requests_by_job, request)
            continue

        object_id = str(keyframe.get("object_id", "asset"))
        hand_role = str(keyframe.get("hand_role", default_hand))
        request = _request_from_template(
            spec,
            keyframe,
            object_id=object_id,
            default_hand_side=hand_role,
            hand_type=hand_type,
        )
        if request is not None:
            _insert_unique_request(requests_by_job, request)
    return tuple(requests_by_job.values())


def _insert_unique_request(
    requests: dict[tuple[str, str], CategoryRetargetRequest],
    request: CategoryRetargetRequest,
) -> None:
    key = (request.hand_type, request.job_id)
    previous = requests.get(key)
    if previous is not None and previous != request:
        raise ValueError(
            f"Conflicting retarget requests share output {request.job_id!r}: "
            f"{previous} and {request}."
        )
    requests[key] = request


def _articulation_sample(
    request: CategoryRetargetRequest,
) -> ArticulationRetargetSample | None:
    if request.articulation_sample_id is None:
        return None
    target = get_target_object(request.category, request.target_index)
    matches = [
        sample
        for sample in target.articulation_retarget_samples
        if sample.sample_id == request.articulation_sample_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Target {request.category}/{target.target_id} has no unique "
            f"articulation sample {request.articulation_sample_id!r}."
        )
    return matches[0]


def _resolve_existing_path(raw_path: str | Path, *, label: str) -> Path:
    path = resolve_project_path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Expected {label} at {path}.")
    return path


def _transform_and_scale(
    root: Path, request: CategoryRetargetRequest
) -> tuple[Path | None, float]:
    target = get_target_object(request.category, request.target_index)
    transform_dir = category_transform_dir(
        root, request.category, request.source_index, request.target_index
    )
    checked_rotation = transform_dir / "rotation.json"
    if checked_rotation.exists():
        transform_path = checked_rotation
    elif target.transform_path is not None:
        transform_path = _resolve_existing_path(
            target.transform_path, label="target rotation transform"
        )
    else:
        transform_path = None

    scale_path = transform_dir / "scale.txt"
    scale = (
        float(scale_path.read_text(encoding="utf-8").strip())
        if scale_path.exists()
        else float(target.scale)
    )
    if scale <= 0.0:
        raise ValueError(f"Expected positive target scale, got {scale}.")
    return transform_path, scale


def _combined_urdf_mesh(
    urdf_path: Path,
    *,
    joint_values: dict[str, float],
    link_names: tuple[str, ...] = (),
) -> trimesh.Trimesh:
    urdf = URDF.load(str(urdf_path))
    available_joints = {joint.name for joint in urdf.actuated_joints}
    unknown_joints = sorted(set(joint_values) - available_joints)
    if unknown_joints:
        raise ValueError(
            f"URDF {urdf_path} does not contain articulation joints "
            f"{unknown_joints}; available joints are {sorted(available_joints)}."
        )
    links = None
    if link_names:
        available_links = {link.name for link in urdf.links}
        missing_links = sorted(set(link_names) - available_links)
        if missing_links:
            raise ValueError(
                f"URDF {urdf_path} does not contain links {missing_links}; "
                f"available links are {sorted(available_links)}."
            )
        links = list(link_names)
    mesh_fk = urdf.visual_trimesh_fk(cfg=joint_values, links=links)
    pieces: list[trimesh.Trimesh] = []
    for mesh, pose in mesh_fk.items():
        transformed = mesh.copy()
        transformed.apply_transform(pose)
        pieces.append(transformed)
    if not pieces:
        raise ValueError(f"URDF {urdf_path} contains no selected visual meshes.")
    return trimesh.util.concatenate(pieces)


def _prepare_urdf_target(
    root: Path,
    request: CategoryRetargetRequest,
    source_urdf: Path,
    scale: float,
    sample: ArticulationRetargetSample | None,
) -> tuple[Path, Path | None, Path]:
    target = get_target_object(request.category, request.target_index)
    output_dir = category_prepared_target_dir(
        root, request.category, request.source_index, request.target_index
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    absolute_urdf = process_urdf(source_urdf)
    sample_suffix = "default" if sample is None else sample.sample_id
    joint_values = {} if sample is None else dict(sample.joint_values)
    visual_path = output_dir / f"{target.target_id}_{sample_suffix}_visual.obj"
    _combined_urdf_mesh(
        absolute_urdf, joint_values=joint_values
    ).export(visual_path)

    bps_path = None
    if target.bps_sampling_link_names:
        bps_path = output_dir / f"{target.target_id}_{sample_suffix}_bps.obj"
        _combined_urdf_mesh(
            absolute_urdf,
            joint_values=joint_values,
            link_names=target.bps_sampling_link_names,
        ).export(bps_path)

    scaled_urdf = materialize_scaled_urdf(
        absolute_urdf,
        output_dir,
        scale,
        output_name=f"{target.target_id}_scaled.urdf",
    )
    return visual_path, bps_path, scaled_urdf


def prepare_target(root: Path, request: CategoryRetargetRequest) -> PreparedTarget:
    source = get_source_template(request.category, request.source_index)
    target = get_target_object(request.category, request.target_index)
    _resolve_existing_path(source.demo_path, label="source demonstration")
    sample = _articulation_sample(request)
    transform_path, scale = _transform_and_scale(root, request)

    target_obj_path = None
    target_bps_obj_path = None
    articulated_urdf_path = None
    if target.mesh_path is not None:
        mesh_path = _resolve_existing_path(target.mesh_path, label="target object")
        if mesh_path.suffix.lower() == ".urdf":
            (
                target_obj_path,
                target_bps_obj_path,
                articulated_urdf_path,
            ) = _prepare_urdf_target(root, request, mesh_path, scale, sample)
        else:
            if sample is not None:
                raise ValueError(
                    f"Articulation sample {sample.sample_id!r} requires a URDF "
                    f"target, got {mesh_path}."
                )
            target_obj_path = mesh_path
    target_demo_path = (
        None
        if target.target_demo_path is None
        else _resolve_existing_path(
            target.target_demo_path, label="target demonstration"
        )
    )
    if (
        not target.same_as_source
        and target_obj_path is None
        and target_demo_path is None
    ):
        raise ValueError(
            f"Target {request.category}/{target.target_id} has neither a mesh "
            "nor a target demonstration."
        )
    return PreparedTarget(
        target_obj_path=target_obj_path,
        target_bps_obj_path=target_bps_obj_path,
        target_demo_path=target_demo_path,
        target_transform_path=transform_path,
        target_scale=scale,
        articulated_urdf_path=articulated_urdf_path,
        articulation_sample=sample,
    )


def build_command(
    root: Path,
    request: CategoryRetargetRequest,
    target: PreparedTarget,
    *,
    preset: str,
    retarget_only_frames: int,
    save_retarget_probes: bool,
) -> list[str]:
    source = get_source_template(request.category, request.source_index)
    run_dir = request.run_dir(root)
    robot_model = robot_model_for_hand_type(request.hand_type)
    if source.dataset == "arctic":
        required = {
            "arctic_target_object": source.arctic_target_object,
            "arctic_timestep": source.arctic_timestep,
            "arctic_start_timestep": source.arctic_start_timestep,
            "arctic_end_timestep": source.arctic_end_timestep,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"ARCTIC source {source.template_id!r} is missing {missing}."
            )
        hand = source.arctic_hand or request.hand_side
        command = [
            sys.executable,
            "-m",
            "dexlink.scripts.retargeting.run_trajopt_arctic",
            "--preset",
            "arctic_default" if preset.startswith("oakink") else preset,
            "--cfg.arctic.demo_path",
            str(resolve_project_path(source.demo_path)),
            "--cfg.arctic.target_object",
            str(source.arctic_target_object),
            "--cfg.arctic.timestep",
            str(source.arctic_timestep),
            "--cfg.arctic.start_timestep",
            str(source.arctic_start_timestep),
            "--cfg.arctic.end_timestep",
            str(source.arctic_end_timestep),
            "--cfg.arctic.skip",
            str(source.arctic_skip),
            "--cfg.arctic.hand",
            hand,
            "--cfg.arctic.robot_model",
            robot_model,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "dexlink.scripts.retargeting.run_trajopt_oakink",
            "--preset",
            preset,
            "--cfg.oakink.demo_path",
            str(resolve_project_path(source.demo_path)),
            "--cfg.oakink.hand",
            request.hand_side,
            "--cfg.oakink.oakink_hand_side",
            request.hand_side,
            "--cfg.oakink.robot_model",
            robot_model,
        ]
    prefix = "arctic" if source.dataset == "arctic" else "oakink"
    command.extend(
        [
            f"--cfg.{prefix}.target_scale",
            str(target.target_scale),
            "--cfg.run_dir",
            str(run_dir),
            "--cfg.trajopt.visualize",
            "False",
            "--cfg.trajopt.pause_after_visualize",
            "False",
        ]
    )
    if source.dataset == "oakink":
        command.extend(
            [
                "--cfg.trajopt_seq.visualize",
                "False",
                "--cfg.trajopt_seq.pause_after_visualize",
                "False",
            ]
        )
    if target.target_obj_path is not None:
        command.extend(
            [f"--cfg.{prefix}.target_obj_path", str(target.target_obj_path)]
        )
    if target.target_demo_path is not None:
        if source.dataset == "arctic":
            raise ValueError("ARCTIC retargeting does not accept target_demo_path.")
        command.extend(
            ["--cfg.oakink.target_demo_path", str(target.target_demo_path)]
        )
    if target.target_bps_obj_path is not None:
        if source.dataset == "arctic":
            raise ValueError("ARCTIC retargeting does not accept target_bps_obj_path.")
        command.extend(
            ["--cfg.oakink.target_bps_obj_path", str(target.target_bps_obj_path)]
        )
    if target.target_transform_path is not None:
        command.extend(
            [
                f"--cfg.{prefix}.target_transform_info",
                str(target.target_transform_path),
            ]
        )
    if target.articulation_sample is not None:
        if source.dataset != "oakink" or target.articulated_urdf_path is None:
            raise ValueError(
                "Articulation samples require OakInk retargeting and a target URDF."
            )
        target_object = get_target_object(request.category, request.target_index)
        command.extend(
            [
                "--object-articulation-joint-values-json",
                json.dumps(target.articulation_sample.joint_values, sort_keys=True),
                "--target-articulated-urdf-path",
                str(target.articulated_urdf_path),
                "--articulation-retarget-mode",
                target_object.articulation_retarget_mode,
                "--articulation-sample-id",
                target.articulation_sample.sample_id,
            ]
        )
    command.extend(
        ["--retarget-only", "--retarget-only-frames", str(retarget_only_frames)]
    )
    if save_retarget_probes:
        command.append("--save-retarget-probes")
    return command


def _write_manifest(
    request: CategoryRetargetRequest,
    target: PreparedTarget,
    run_dir: Path,
) -> None:
    payload = {
        "request": asdict(request),
        "job_id": request.job_id,
        "run_dir": str(run_dir),
        "target": {
            "target_obj_path": (
                None if target.target_obj_path is None else str(target.target_obj_path)
            ),
            "target_bps_obj_path": (
                None
                if target.target_bps_obj_path is None
                else str(target.target_bps_obj_path)
            ),
            "target_demo_path": (
                None
                if target.target_demo_path is None
                else str(target.target_demo_path)
            ),
            "target_transform_path": (
                None
                if target.target_transform_path is None
                else str(target.target_transform_path)
            ),
            "target_scale": target.target_scale,
            "articulated_urdf_path": (
                None
                if target.articulated_urdf_path is None
                else str(target.articulated_urdf_path)
            ),
            "articulation_sample_id": (
                None
                if target.articulation_sample is None
                else target.articulation_sample.sample_id
            ),
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "retarget_job_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_one(command: list[str], root: Path, run_dir: Path) -> int:
    print("[dexlink]", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode == 0 and not (run_dir / MOTION_FILENAME).is_file():
        raise FileNotFoundError(
            f"Retargeting command succeeded but did not create "
            f"{run_dir / MOTION_FILENAME}."
        )
    return int(completed.returncode)


def run(args: RunnerArgs) -> tuple[Path, ...]:
    if args.max_workers <= 0:
        raise ValueError(f"Expected max_workers > 0, got {args.max_workers}.")
    if args.retarget_only_frames <= 0:
        raise ValueError(
            "Expected retarget_only_frames > 0, got "
            f"{args.retarget_only_frames}."
        )
    root = get_src()
    requests: dict[tuple[str, str], CategoryRetargetRequest] = {}
    for spec_path in args.specs:
        spec = load_task_spec(spec_path)
        for request in collect_category_retarget_requests(spec, args.hand_type):
            _insert_unique_request(requests, request)

    jobs: list[tuple[list[str], Path]] = []
    completed_paths: list[Path] = []
    for request in requests.values():
        run_dir = request.run_dir(root)
        motion_path = run_dir / MOTION_FILENAME
        if args.skip_existing and motion_path.is_file():
            completed_paths.append(motion_path)
            continue
        target = prepare_target(root, request)
        command = build_command(
            root,
            request,
            target,
            preset=args.preset,
            retarget_only_frames=args.retarget_only_frames,
            save_retarget_probes=args.save_retarget_probes,
        )
        _write_manifest(request, target, run_dir)
        if args.dry_run:
            print("[dry-run]", " ".join(command))
        else:
            jobs.append((command, run_dir))

    failures: list[tuple[Path, int]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_jobs = {
            executor.submit(_run_one, command, root, run_dir): run_dir
            for command, run_dir in jobs
        }
        for future in as_completed(future_jobs):
            run_dir = future_jobs[future]
            return_code = future.result()
            if return_code != 0:
                failures.append((run_dir, return_code))
            else:
                completed_paths.append(run_dir / MOTION_FILENAME)
    if failures:
        details = ", ".join(f"{path} (exit {code})" for path, code in failures)
        raise RuntimeError(f"Category retargeting jobs failed: {details}.")
    return tuple(completed_paths)


def main(argv: list[str] | None = None) -> None:
    outputs = run(_parse_args(argv))
    print(f"[dexlink] completed category retargetings: {len(outputs)}")


if __name__ == "__main__":
    main()
