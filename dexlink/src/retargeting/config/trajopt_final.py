from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from dexlink.utils.retargeting.path_utils import get_external_data_dir, get_tmp_dir


@dataclass
class TrajOptSeqCfg:
    sequence_source: str = "single_step"
    solver_tolerance: float = 1e-4
    log_cost_terms: bool = True
    plot_cost_terms: bool = True
    cost_log_every: int = 1
    visualize: bool = True
    pause_after_visualize: bool = True
    min_optimizer_iterations: int = 100
    n_control_pts: int = 30
    closeness_cost_weight: float = 1e-1

    bps_cost_weight: float = 20.0
    distance_cost_weight: float = 10.0

    seed: int = 42


@dataclass
class TrajoptCfg:
    num_pts_sample: int = 256
    bps_beta: float = 20.0
    bps_beta_range: tuple[float, float] = (1.0, 20.0)
    # bps_beta_range: tuple[float, float] | None = None
    rotational_freedom: bool = True
    bps_beta_anneal_steps: int = 100
    bps_beta_update_every: int = 1
    bps_nocs_features: bool = True
    bps_exp_coeff: float = 1e-3
    hand_cfg_weight: float = 1.0
    hand_object_bps_weight: float = 1.0
    distance_weight: float = 1.0
    penetration_constraint: bool = False
    penetration_constraint_mode: str = "distance_constraint"
    penetration_bound: float = -0.02
    penetration_distance_upper: float = 1.0
    influence_distance_offset: float = 1e-1
    solver_tolerance: float = 1e-4
    log_cost_terms: bool = True
    plot_cost_terms: bool = True
    cost_log_every: int = 1
    visualize: bool = True
    pause_after_visualize: bool = True
    non_penetration_refinement: bool = False
    sequential_finger_refinement: bool = False
    sequential_finger_use_masked_cost: bool = True
    sequential_line_search_refinement: bool = False
    sequential_line_search_samples: int = 21
    sequential_line_search_radius: float = 0.1
    sequential_line_search_passes: int = 2
    sequential_finger_refinement_order: tuple[str, ...] = (
        "wrist",
        "thumb",
        "index",
        "mid",
        "ring",
        "pinky",
    )
    sequential_finger_nominal_weight: float = 10.0
    sequential_wrist_nominal_weight: float = 10.0
    sequential_finger_penetration_bound: float = 0.001
    sequential_inter_stage_penetration_bound: float = 0.001
    sequential_finger_contact_distance: float = 0.01
    sequential_finger_contact_tolerance: float = 0.005
    initial_solution_npz: str | None = None
    initial_solution_frame: int = -1
    contact_target_refinement: bool = False
    contact_target_distance: float = 0.04
    contact_target_position_weight: float = 1.0
    contact_target_normal_weight: float = 0.01
    contact_target_nominal_weight: float = 1e-3
    min_optimizer_iterations: int = 100

    seed: int = 42


@dataclass
class ArcticDemoCfg:
    demo_path: str = str(
        get_external_data_dir() / "arctic" / "seqs" / "s01" / "waffleiron_use_01.npy"
    )
    target_object: str = "proc_waffleiron_3"
    timestep: int = 400
    start_timestep: int = 0
    end_timestep: int = -1
    skip: int = 1
    hand: str = "right"  # left | right
    robot_model: str = "xhand"  # inspire | xhand
    mesh_type: str = "visual"  # visual | collision
    target_obj_path: str | None = None
    target_transform_info: str | None = None
    target_scale: float = 1.0


@dataclass
class OakinkDemoCfg:
    demo_path: str | None = None
    target_demo_path: str | None = None
    target_obj_path: str | None = None
    target_bps_obj_path: str | None = None
    target_transform_info: str | None = None  # path to json file
    target_scale: float = 1.0
    timestep: int = 0
    start_timestep: int = 0
    end_timestep: int = -1
    skip: int = 1
    hand: str = "right"
    robot_model: str = "xhand"  # inspire | xhand
    mesh_type: str = "visual"  # visual | collision
    oakink_hand_side: str = "right"
    oakink_smplx_model_path: str | None = None
    oakink_mano_assets_root: str | None = None


@dataclass
class ExperimentConfig:
    preset: str = "default"
    trajopt: TrajoptCfg = field(default_factory=TrajoptCfg)
    trajopt_seq: TrajOptSeqCfg = field(default_factory=TrajOptSeqCfg)
    arctic: ArcticDemoCfg = field(default_factory=ArcticDemoCfg)
    oakink: OakinkDemoCfg = field(default_factory=OakinkDemoCfg)
    run_dir: str = str(get_tmp_dir() / "trajopt_final")


PRESETS: dict[str, dict[str, Any]] = {
    "arctic_default": {
        "trajopt.min_optimizer_iterations": 100,
        # "trajopt.penetration_constraint": False,
        # "trajopt.non_penetration_refinement": True,
        "trajopt.bps_beta_range": None,
        "trajopt.hand_object_bps_weight": 10.0,
        "trajopt.bps_exp_coeff": 1e-3,
        "trajopt.num_pts_sample": 512,
        "trajopt.penetration_constraint": False,
        "trajopt.non_penetration_refinement": False,
        "trajopt.rotational_freedom": False,
        "trajopt.sequential_finger_refinement": False,
        "trajopt.sequential_line_search_refinement": True,
        "trajopt.sequential_line_search_samples": 31,
        "trajopt.sequential_finger_use_masked_cost": False,
    },
    "oakink_default": {
        "trajopt.min_optimizer_iterations": 20,
        "trajopt.num_pts_sample": 512,
        "trajopt.penetration_constraint": False,
        "trajopt.non_penetration_refinement": False,
        "trajopt.bps_beta_range": None,
        "trajopt.bps_beta": 20.0,
        "trajopt.bps_exp_coeff": 1e-3,
        "trajopt.rotational_freedom": False,
        "trajopt.sequential_finger_refinement": True,
    },
    "oakink_default_2": {
        "trajopt.min_optimizer_iterations": 20,
        "trajopt.num_pts_sample": 512,
        "trajopt.penetration_constraint": False,
        "trajopt.non_penetration_refinement": False,
        "trajopt.bps_beta_range": None,
        "trajopt.bps_beta": 20.0,
        "trajopt.bps_exp_coeff": 1e-3,
        "trajopt.rotational_freedom": False,
        "trajopt.sequential_finger_refinement": False,
        "trajopt.sequential_line_search_refinement": True,
        "trajopt.sequential_line_search_samples": 31,
        "trajopt.sequential_finger_use_masked_cost": False,
    },
}


def _get_nested_attr(obj: Any, path: list[str]) -> Any:
    current = obj
    for key in path:
        if not hasattr(current, key):
            raise KeyError(f"Unknown config key '{'.'.join(path)}'.")
        current = getattr(current, key)
    return current


def _set_nested_attr(obj: Any, path: list[str], value: Any) -> None:
    if len(path) == 1:
        key = path[0]
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key '{key}'.")
        setattr(obj, key, value)
        return

    head = path[0]
    if not hasattr(obj, head):
        raise KeyError(f"Unknown config key '{'.'.join(path)}'.")
    _set_nested_attr(getattr(obj, head), path[1:], value)


def _cast_to_type(value_str: str, existing_value: Any) -> Any:
    if value_str.lower() in {"none", "null"}:
        return None
    if isinstance(existing_value, bool):
        return value_str.lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(existing_value, int) and not isinstance(existing_value, bool):
        return int(value_str)
    if isinstance(existing_value, float):
        return float(value_str)
    if isinstance(existing_value, str):
        return value_str
    if isinstance(existing_value, (list, tuple)):
        parts = [part.strip() for part in value_str.split(",") if part.strip()]
        if not existing_value:
            return parts
        elem_type = type(existing_value[0])
        return [elem_type(part) for part in parts]
    return value_str


def apply_overrides_from_dict(
    cfg: ExperimentConfig, overrides: dict[str, Any]
) -> ExperimentConfig:
    for key, value in overrides.items():
        _set_nested_attr(cfg, key.split("."), value)
    return cfg


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must be in key=val format.")
        key, value_str = override.split("=", 1)
        path = key.split(".")
        existing_value = _get_nested_attr(cfg, path)
        value = _cast_to_type(value_str, existing_value)
        print(f"Overriding config key '{key}' from {existing_value!r} to {value!r}.")
        _set_nested_attr(cfg, path, value)
    return cfg


def load_experiment_config(
    preset: str | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    cfg = ExperimentConfig()
    preset_name = preset or cfg.preset
    if preset_name != "default":
        preset_dict = PRESETS.get(preset_name)
        if preset_dict is None:
            raise ValueError(f"Unknown trajopt preset '{preset_name}'.")
        apply_overrides_from_dict(cfg, preset_dict)
    cfg.preset = preset_name
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


def load_trajopt_config(
    preset: str | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    return load_experiment_config(preset=preset, overrides=overrides)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def print_cfg_summary(cfg: ExperimentConfig) -> None:
    from pprint import pprint

    pprint(config_to_dict(cfg))


def extract_overrides(unknown_tokens: list[str]) -> list[str]:
    overrides = []
    it = iter(unknown_tokens)
    for token in it:
        if not token.startswith("--cfg."):
            continue
        key = token[len("--cfg.") :]
        try:
            value = next(it)
        except StopIteration as exc:
            raise ValueError(f"Missing value for override '{token}'.") from exc
        overrides.append(f"{key}={value}")
    return overrides


__all__ = [
    "ArcticDemoCfg",
    "ExperimentConfig",
    "OakinkDemoCfg",
    "PRESETS",
    "TrajOptSeqCfg",
    "TrajoptCfg",
    "apply_overrides",
    "apply_overrides_from_dict",
    "config_to_dict",
    "extract_overrides",
    "load_experiment_config",
    "load_trajopt_config",
    "print_cfg_summary",
]
