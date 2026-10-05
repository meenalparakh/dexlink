from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextmanager
def prepended_sys_path(*paths: Path):
    original = list(sys.path)
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = original


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root() / path


@contextmanager
def redirect_standard_fds_to_path(log_path: Path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    target_fd = os.open(
        str(log_path),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.dup2(target_fd, 1)
        os.dup2(target_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(target_fd)


def load_robot_control_cfg(config_path: str):
    cfg_path = resolve_repo_path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    try:
        import attridict
    except ImportError as exc:
        raise ImportError(
            "Failed to import attridict for real robot config loading."
        ) from exc
    return attridict.AttriDict(cfg_dict), cfg_path


def create_franka_xhand_robot_base(
    config_path: str, *, real_cfg=None, output_log_path: str | Path | None = None
):
    cfg, cfg_path = load_robot_control_cfg(config_path)
    if real_cfg is not None:
        if real_cfg.franka_control_mode not in {"osc_absolute_pose", "joint_impedance"}:
            raise ValueError(
                "Expected real_cfg.franka_control_mode to be 'osc_absolute_pose' or "
                f"'joint_impedance', "
                f"got {real_cfg.franka_control_mode!r}."
            )
        if real_cfg.xhand_control_mode != "pos":
            raise ValueError(
                "Expected real_cfg.xhand_control_mode to be 'pos', "
                f"got {real_cfg.xhand_control_mode!r}."
            )
        cfg.franka.control_mode = real_cfg.franka_control_mode
        cfg.franka.joint_controller_cfg = real_cfg.franka_joint_controller_cfg
        cfg.franka.osc_controller_cfg = real_cfg.franka_osc_controller_cfg
        cfg.xhand.ctrl_type = real_cfg.xhand_control_mode
        if real_cfg.control_period_s <= 0.0:
            raise ValueError(
                f"Expected real_cfg.control_period_s > 0, got {real_cfg.control_period_s}."
            )
        cfg.control_freq = max(1, int(round(1.0 / float(real_cfg.control_period_s))))
    control_root = cfg_path.parent.parent
    module_path = control_root / "env" / "control_base.py"
    module_spec = spec_from_file_location("dexlink_real_control_base", module_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}.")
    module = module_from_spec(module_spec)
    with prepended_sys_path(control_root, repo_root()), pushd(control_root):
        module_spec.loader.exec_module(module)
        FrankaXHandRobotBase = module.FrankaXHandRobotBase
        if output_log_path is None:
            robot = FrankaXHandRobotBase(cfg)
        else:
            log_path = Path(output_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with redirect_standard_fds_to_path(log_path):
                robot = FrankaXHandRobotBase(cfg)
    return robot, str(cfg_path)
