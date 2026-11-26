# Cup multi-task policies

The final conditional cup-policy implementation is in
`dexlink/multitask_cup/`. It combines task-program and physical-geometry
conditioning with the native full-method residual policy, critic, rollout
collection, PPO updates and synchronized checkpoints. Its current task
representation is specific to the cup task family.

The included 30 task programs cover geometry, goal pose, timing, ordered
orientation holds and placement/release. Prepare a local absolute-path task
bank and immutable input hashes:

```bash
conda activate genesis
python -m dexlink.workflows.multitask --output outputs/cup30-data
```

This creates `manifest.json`, `task-bank.pt`, `training-config.json`, and
source/asset hashes. Recreate the preparation directory after changing the
code, references or assets; training rejects hash mismatches. Unavailable
human-hand keypoints retain the source's masked missing-data placeholders;
the robot/object features consumed by the bank must be finite.

The generated configuration retains the established production resource
defaults: eight GPUs and 2,048 environments per task. Select resources that
fit the actual machine explicitly. One-GPU execution is supported, but the
full 30-task cohort can still require substantial memory.

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_ENTITY=YOUR_WANDB_ENTITY
python -m dexlink.multitask_cup.launch_full_method \
  --config outputs/cup30-data/training-config.json \
  --world_size 1 --environments_per_task 32 \
  --wandb_entity YOUR_WANDB_ENTITY
```

All settings use the dataclass configuration and dotted CLI overrides,
including `--architecture.program_layers 2`. `storage_root` and
`run_directory` must identify persistent project storage. Fresh run identities
are exclusive; `--resume` validates the original data/configuration lineage.
Task evaluation requires positive saved checkpoints and complete 50-episode
cohorts. The multi-task training run itself has not been repeated as part of
the initial export; bank construction and the policy/learner tests are separate
validation steps.

For cluster runs, follow [cluster examples](../cluster/README.md) and your
site's account, queue and persistent-storage rules. Data preparation submits
no jobs. The provided checkpoint ZIP contains single-task policies; it does
not contain a conditional multi-task policy checkpoint.
