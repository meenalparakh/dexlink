"""Initialize target policies from a positive source without transferring training state."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from .full_method_config import FullMethodRunConfig
from .learner import parameter_digest
from .task_catalog import file_sha256


def initialize_fresh_policy(policy, config: FullMethodRunConfig) -> dict:
    initialization = config.transfer
    receipt = dict(
        arm=initialization.arm,
        optimizer_transferred=False,
        runtime_transferred=False,
        common_initial_log_std=-1.0,
        initial_parameter_digest=parameter_digest(policy),
    )
    if initialization.arm in ("source", "scratch"):
        if (
            initialization.checkpoint
            or initialization.checkpoint_sha256
            or initialization.source_control_step
        ):
            raise ValueError(
                "Fresh source/scratch policies cannot load a learned checkpoint."
            )
        return receipt
    if initialization.arm != "warm_start":
        raise ValueError(f"Unknown initialization arm: {initialization.arm!r}.")
    study_path = Path(initialization.study)
    if file_sha256(study_path) != initialization.study_sha256:
        raise ValueError("Warm-start study manifest changed.")
    study = json.loads(study_path.read_text())
    selected = study["selected_source"]
    path = Path(initialization.checkpoint)
    if (
        file_sha256(path) != initialization.checkpoint_sha256
        or initialization.checkpoint_sha256 != selected["shared_sha256"]
    ):
        raise ValueError(
            "Warm-start weights differ from the source-only selected checkpoint."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint["control_step"] <= 0
        or checkpoint["control_step"] != initialization.source_control_step
        or checkpoint["control_step"] != selected["control_step"]
        or checkpoint["optimizer_updates"] <= 0
    ):
        raise ValueError(
            "Warm initialization requires the exact selected positive trained source checkpoint."
        )
    # Only the declared target budget may differ in the native policy schema.
    source_schema = deepcopy(checkpoint["schema"])
    source_budget = source_schema["agent"]["timesteps"]
    source_schema["agent"]["timesteps"] = policy.checkpoint_schema()["agent"][
        "timesteps"
    ]
    if source_schema != policy.checkpoint_schema():
        raise ValueError(
            "Source/target policy schemas differ beyond the declared target budget."
        )
    initial_log_std = policy.actor.network.log_std.detach().clone()
    if not torch.equal(initial_log_std, torch.full_like(initial_log_std, -1.0)):
        raise ValueError(
            "Both transfer arms require the validated common initial exploration log standard deviation -1."
        )
    policy.load_state_dict(checkpoint["policy"], strict=True)
    loaded_digest = parameter_digest(policy)
    if loaded_digest != selected["parameter_digest"]:
        raise ValueError("Loaded policy digest differs from the selected source seal.")
    with torch.no_grad():
        policy.actor.network.log_std.copy_(initial_log_std)
    if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
        raise ValueError("Warm policy contains nonfinite parameters.")
    return dict(
        receipt,
        source_control_step=checkpoint["control_step"],
        source_shared_sha256=initialization.checkpoint_sha256,
        source_parameter_digest=loaded_digest,
        source_optimizer_updates=checkpoint["optimizer_updates"],
        source_control_budget=source_budget,
        target_control_budget=config.control_steps,
        reset_parameter="actor.network.log_std",
        initial_parameter_digest=parameter_digest(policy),
    )
