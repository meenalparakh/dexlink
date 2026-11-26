"""Verify publication of an online training history."""

import time


def verify_remote(run_path: str, step: int) -> None:
    import wandb

    deadline = time.monotonic() + 90
    while True:
        rows = (
            wandb.Api(timeout=20)
            .run(run_path)
            .scan_history(keys=["train/control_step"])
        )
        if any(row["train/control_step"] >= step for row in rows):
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                "Online W&B history did not publish the positive training step; checkpoint is preserved."
            )
        time.sleep(3)
