---
name: dexlink-generate-data
description: >-
  Generate policy observation/action demonstrations for existing DexLink tasks using verified trained checkpoints. Use for numerical datasets and episode indexing, before video rendering.
---

# Collect DexLink demonstrations

Read `docs/data-generation.md` and `docs/resources.md`. Install the public
checkpoint/archive resources and validate each task before spending GPU time.
Use `python -m dexlink collect invert_cup t047_a --episodes 10 --parallel-envs 5
--output outputs/data/my-dataset`. Choose a fresh persistent output directory.
For a newly trained run use `dexlink-eval-metrics --rl_run_dir RUN
--ckpt_timestep POSITIVE_STEP --record_policy_inputs` with explicit episode
counts/output, rather than silently switching the catalog's checkpoint.
Keep failed episodes and provenance. Read `dataset-index.json`: only each
environment's first complete episode counts; exclude later reset/padding rows.
Policy observations/actions are pre-step; physical states are post-step. Do
not infer an aligned next observation or action convention from vector width.
Collect numeric trajectories first; use `$dexlink-render-data` for videos.
Default collection needs no W&B login. Never evaluate timestep 0.
