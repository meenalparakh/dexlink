from __future__ import annotations

from pathlib import Path

import pytest

from dexlink.scripts.rl.eval_sim import (
    _infer_run_dir_from_policy_checkpoint,
    _resolve_policy_checkpoint,
)


def _checkpoint(tmp_path: Path, timestep: int) -> Path:
    checkpoint = tmp_path / "run" / "models" / str(timestep) / "policy.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"test checkpoint")
    return checkpoint


def test_evaluation_rejects_timestep_zero(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 0)
    with pytest.raises(ValueError, match="untrained policy"):
        _infer_run_dir_from_policy_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="strictly positive"):
        _resolve_policy_checkpoint(
            rl_run_dir=str(tmp_path / "run"),
            ckpt_timestep=0,
            resume_from_ckpt=None,
            logger_logdir=str(tmp_path),
        )


def test_evaluation_accepts_positive_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 5000)
    assert _infer_run_dir_from_policy_checkpoint(checkpoint) == tmp_path / "run"
    assert (
        _resolve_policy_checkpoint(
            rl_run_dir=str(tmp_path / "run"),
            ckpt_timestep=5000,
            resume_from_ckpt=None,
            logger_logdir=str(tmp_path),
        )
        == checkpoint
    )
