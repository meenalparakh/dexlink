from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class FinalEvalWandbUploadCfg:
    output_dir: Path
    project: str
    entity: str | None
    mode: Literal["online", "offline"]
    run_name: str
    group: str
    job_type: str = "checkpoint_eval"
    run_id: str | None = None
