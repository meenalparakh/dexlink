---
name: dexlink-troubleshoot
description: >-
  Diagnose missing data, simulator dependency, checkpoint, numerical learning or rendering failures in DexLink. Use for a failed workflow with its logs and resolved configuration.
---

# Diagnose DexLink workflows

Start with `AGENTS.md`, the relevant workflow skill and the actual traceback.
Check the pinned Genesis fork/revision, selected environment and paths before
changing code. Validate frozen tasks using `python -m dexlink check TASK`;
validate generation inputs using `dexlink-validate-task TASK.yaml`.
Use `docs/resources.md` for ZIP/checksum links. Google Drive downloads need
the `downloads` extra and public sharing; never bypass a checksum mismatch.
Do not replace missing checkpoint preprocessors with fresh state or evaluate
a timestep-0 alias. Review `launch.json`, `training-result.json`, W&B and
`numerical-audit.json` for learning; report any NaN/Inf and its transition
denominator, and keep invalid samples out of learning. Do not retune full-method
settings merely to make a smoke pass. Saved data/rendering can be diagnosed
without starting another training run. Preserve run lineage and avoid retries
that create duplicate jobs or overwrite outputs.
