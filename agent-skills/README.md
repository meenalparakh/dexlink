# Agent skills

These packages teach coding agents how to operate DexLink's public workflows:

- [`dexlink-install`](dexlink-install/SKILL.md): install the package and connect
  separately licensed datasets and models.
- [`dexlink-create-task`](dexlink-create-task/SKILL.md): author and validate a
  portable task YAML from an existing task family.
- [`dexlink-run-task`](dexlink-run-task/SKILL.md): generate a reference and
  train a policy for an existing task.
- [`dexlink-evaluate-policy`](dexlink-evaluate-policy/SKILL.md): select a
  trained checkpoint and run reproducible simulation evaluation.

Each directory is a self-contained agent skill with a `SKILL.md` entry point
and product metadata under `agents/`. Install or link only the skills needed by
your agent runtime.

The task-specification UI and its usage skill belong to the separate UI
repository. DexLink only consumes the YAML exported by that interface.

Additional workflow skills:

- [dexlink-datasets](dexlink-datasets/SKILL.md): provider data and runtime geometry.
- [dexlink-generate-data](dexlink-generate-data/SKILL.md): policy demonstrations.
- [dexlink-render-data](dexlink-render-data/SKILL.md): separate saved-motion rendering.
- [dexlink-multitask](dexlink-multitask/SKILL.md): conditional cup task-bank/learning.
- [dexlink-submit-cluster](dexlink-submit-cluster/SKILL.md): portable Slurm examples.
- [dexlink-troubleshoot](dexlink-troubleshoot/SKILL.md): input/runtime/checkpoint diagnosis.

Point an agent at the repository-root [AGENTS.md](../AGENTS.md) first. To make
these skills discoverable by Codex, link the desired skill directories into
your agent's skills directory. Other agents can read SKILL.md directly.
