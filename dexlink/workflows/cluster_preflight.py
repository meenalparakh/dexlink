"""Require a completed finite PPO optimizer audit before a training submission."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def check_report(path: Path) -> None:
    audit = json.loads(path.read_text())
    if not audit["completed"] or audit["failure"] is not None:
        raise ValueError("The supplied PPO smoke did not complete successfully.")
    if (
        audit["optimizer_updates"] < 1
        or audit["loss_batches"] < 1
        or audit["environment_control_transitions"] < 1
    ):
        raise ValueError(
            "Expected actual optimizer updates, losses and environment transitions."
        )
    if audit["nonfinite_environment_control_transitions"] != 0:
        raise ValueError(
            "The smoke contains nonfinite transitions; diagnose and review before submission."
        )
    print(
        "Completed finite PPO optimizer audit verified; confirm its profile matches this job."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    check_report(parser.parse_args().report)


if __name__ == "__main__":
    main()
