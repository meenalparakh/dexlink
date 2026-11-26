---
name: dexlink-run-task
description: >-
  Run an existing DexLink task end to end: validate its inputs, retarget
  interaction templates, build and inspect the reference trajectory, and launch
  RL training with a public preset. Use when reproducing a catalog task,
  smoke-testing the reference-to-policy pipeline, or training a task YAML
  already present in the repository.
---

# Run an Existing DexLink Task

Use this workflow only for an existing YAML under `tasks/`. For a new or
modified task specification, use `$dexlink-create-task` first.

## 1. Resolve and validate the task

Read the task YAML and record its `experiment_id`,
`trajectory.output_motion_path`, and `trajectory.exported_rl_path`. Confirm
that every path is repository-relative. Validate all declared assets before
starting optimization:

```bash
dexlink-validate-task tasks/path/to/task.yaml
```

Stop on validation errors. Correct the external-data mount or task declaration;
do not add filename fallbacks or copy licensed files into the repository.

## 2. Generate the reference

Run the public pipeline from the repository root:

```bash
dexlink-reference tasks/path/to/task.yaml \
  --max-workers 8 \
  --refine-collisions
```

Select `--max-workers` from the actual CPU and memory limits. Retargeting is
CPU-heavy; avoid running a large worker pool on a constrained workstation.
Collision refinement can be expensive and must not be replaced by an
unrecorded shortcut. Use `--allow-unresolved-collisions` only when the user
explicitly accepts unresolved contacts, and report that choice.

The command prints the RL-readable `.npy` path. Verify that file and the
corresponding `.npz` and manifest exist under `DEXLINK_OUTPUTS`. If the task has
already been retargeted, `--skip-retargeting` is valid only after confirming the
declared retargeting outputs exist.

## 3. Inspect the reference

Open the assembled motion in MeshCat before spending GPU time:

```bash
python -m dexlink.visualizations.retargeting.vis_robot_retgt_motion \
  --retargeted_results outputs/path/to/reference.npz \
  --mesh_type visual \
  --port 6001
```

Check the source/target alignment, object placement, support surfaces,
articulation convention, hand-object contact, multi-object visibility, and the
full ordered sequence. A syntactically valid reference is not evidence that
the task semantics are correct.

## 4. Train the policy

Use the public full-method preset unless the experiment explicitly calls for a
different ablation:

```bash
dexlink-train \
  --preset seq_oakink_residual_target_trajectory_weighted_policy_v3 \
  --env.task_cfg.oakink_exported_path outputs/path/to/reference.npy
```

Dot-notation overrides are supported, for example `--env.seed 23` or
`--env.sim_cfg.num_envs 4`. Preserve the preset's optimizer and PPO settings
for a compatibility smoke test unless the user requests an experiment change.
Use `--local` only for an interactive local smoke test; it enables the viewer
and changes the environment count, so it is not a paper training command.

The multi-object environment is the default. Do not add alternate-environment
routing for a task-specific issue. New task metrics belong in the existing
task-conditioned metric configuration and task manager.

## 5. Confirm useful progress

Confirm that PPO reaches at least one update, the configured task-specific
train metric is present, and periodic positive-timestep checkpoints are
written. Report the exact task YAML, generated reference, preset, overrides,
seed, run directory, and last positive checkpoint. Do not claim success from a
timestep-zero checkpoint or from reference visualization alone.

## Packaged workflow entry points

Start with `AGENTS.md`, `docs/installation.md`, `docs/resources.md`, and
`tasks/catalog.json`. The RL extra pins the tested public Genesis fork;
updating that revision requires validation. Use `python -m dexlink list`,
`check`, `train`, `smoke`, `rollout`, `collect` and `render` for catalog tasks.
Install the `downloads` extra, then run `dexlink-resources assets` and
`dexlink-resources checkpoints`; both use the public Google Drive links and
pinned checksums. For a manual download use `--archive PATH`.
Use `$dexlink-datasets` for provider datasets. Fresh learning/smoke requires a W&B account and online
logging; default pretrained rollouts and separate mesh rendering do not.
`docs/data-generation.md` defines collection timing and episode boundaries.
