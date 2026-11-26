---
name: dexlink-create-task
description: >-
  Author or modify a portable DexLink task YAML, including source interaction
  templates, target objects, alignment transforms, scene placement, ordered
  keyframes, multi-stage trajectories, and task-conditioned metrics. Use when
  adding a task to the public catalog or adapting an existing task family. This
  skill covers the YAML/code workflow, not the separately hosted
  task-specification UI.
---

# Create a DexLink Task

The browser task-specification UI is maintained in a separate repository. In
this repository, author the YAML and the smallest task-specific integration
needed to consume it.

## 1. Start from the nearest semantic template

Search `tasks/` for a task with the same interaction structure, not merely the
same object category. Prefer, in order:

1. the same number of hands and active objects;
2. the same task type, such as lift, place, articulate, or sequential contact;
3. the same source-template dataset and target-object representation;
4. the same support-surface and articulation requirements.

Read the selected YAML completely and inspect its category definition under
`dexlink/src/reference/common/categories/`. Source and target indices are
explicit catalog entries; never infer an index from a filename.

## 2. Define a portable specification

Every task must have a unique non-empty `experiment_id`, a non-empty
`keyframes` list, and a `trajectory` mapping with non-empty `phases`,
`output_motion_path`, and `exported_rl_path`.

Keep all declared paths repository-relative:

- project-authored assets use `assets/...`;
- licensed datasets and models use `external_data/...`;
- generated artifacts use `outputs/...`;
- `from_retarget.motion_path: auto` requests the pipeline-derived retargeting.

Do not write absolute workstation or cluster paths, unresolved `TBD` values,
or alternate field names. The loader rejects malformed data deliberately.

Declare the scene and motion in this order:

1. source/target template selection for each active object;
2. static support and scene objects with explicit poses;
3. named keyframes derived from retargetings or earlier keyframes;
4. explicit transforms for alignment, translation, rotation, or pose setting;
5. ordered trajectory phases with exact frame counts;
6. output paths and experiment metadata.

Use only transform and phase types already implemented by
`dexlink/scripts/retargeting/build_sequence_from_spec.py`. If the desired
operation is absent, add a clear typed implementation and a focused test; do
not encode it through an unrelated transform that happens to produce a similar
pose.

For multi-object tasks, assign stable object IDs and specify `active_object`
where the phase changes ownership. Keep support objects separate from active
objects. Place resting meshes using the existing lowest-mesh-point alignment
transforms rather than hand-tuned visual offsets when possible.

## 3. Add task-conditioned RL semantics

If an existing task criterion applies, register the `experiment_id` with its
settings in `dexlink/src/rl/config/seq_oakink_presets.py`. If a genuinely new
criterion is required, extend `TaskMetricCfg` in
`dexlink/src/rl/config/base.py` and the metric dispatch in
`dexlink/src/rl/multi_object/task_manager.py`, then add unit tests for success,
failure, early termination, and episode reset. Do not change global reward or
metric behavior to accommodate one task.

## 4. Validate in increasing-cost order

Run these checks from the repository root:

```bash
python -c "from dexlink.src.reference.task_io import load_task_spec; load_task_spec('tasks/path/to/task.yaml')"
dexlink-validate-task tasks/path/to/task.yaml
pytest tests/test_task_catalog.py tests/test_task_metrics.py
```

Then generate and visually inspect the reference with `$dexlink-run-task`.
For a dynamic object, use the existing Genesis scene-loading path to verify
that its initial support pose remains stable for several simulation steps.
Visual appearance alone can hide a small floating offset.

Report the source task used as the schema template, source/target catalog
entries, alignment choice, phase sequence, metric registration, validation
results, and any unresolved semantic choice that needs user judgment.

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
