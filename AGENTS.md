# Start here

DexLink builds dexterous robot reference motions from task specifications,
then trains residual policies that execute them in Genesis. Read
[README.md](README.md), [the task catalog](tasks/README.md), and
[agent-skills/README.md](agent-skills/README.md). The skills route installation,
datasets, authoring, training, evaluation, data collection and separate rendering.

- Use the pinned public Genesis fork and the environment guides in
  [docs/installation.md](docs/installation.md).
- For pretrained demonstrations, install runtime assets and checkpoints with
  [docs/resources.md](docs/resources.md). Raw MANO/OakInk/ARCTIC data and W&B
  are unnecessary for default packaged policy rollouts.
- For new references, read [docs/new-task.md](docs/new-task.md) and
  [external_data/README.md](external_data/README.md). Cached templates still
  need the source demonstration and MANO during native RL export.
- Use portable YAML tasks and the existing typed category registrations.
  Reject unexpected inputs clearly instead of guessing alternate paths/shapes.
- Start fresh training in a new persistent directory, use the established
  full-method preset and online W&B, and save complete positive checkpoints.
  Never evaluate timestep 0, including a `best` alias that points to zero.
- Run an actual finite PPO optimizer smoke test before expensive new training.
  Record environment-control transition counts and learner checks. A clean
  representative profile can qualify equivalent tasks; retest material changes.
- Keep datasets, models, archives, generated data and caches out of Git, on
  storage sized for them. Set DEXLINK_EXTERNAL_DATA and DEXLINK_OUTPUTS.
- Collection and rendering are separate operations; see
  [docs/data-generation.md](docs/data-generation.md). Preserve episode lengths,
  failures, pre-step observations/actions and post-step physical-state timing.
- Cluster scripts are examples. Set your site's account/resources and persistent
  paths explicitly; never submit duplicate fresh run identities.

The task-authoring UI is in a separate repository. This checkout consumes its
YAML exports. Paper planning, result aggregation, figures and ongoing research
generators are outside the framework distribution.
