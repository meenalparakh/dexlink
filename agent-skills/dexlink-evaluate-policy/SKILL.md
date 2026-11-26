---
name: dexlink-evaluate-policy
description: >-
  Load a trained DexLink policy checkpoint, audit checkpoint selection, run
  deterministic simulation rollouts, compute task success and secondary
  trajectory metrics, optionally render video, and record reproducible
  evaluation artifacts. Use for final 50-episode evaluations, checkpoint
  comparisons, or policy visualization.
---

# Evaluate a DexLink Policy

Evaluation is deterministic by default: use policy-mean (greedy) actions so
results remain comparable across runs. Use stochastic actions only when the
experiment explicitly studies policy sampling and label those results
separately.

## 1. Audit checkpoint selection

Require the standard layout:

```text
<run_dir>/
├── exp_cfg.json
└── models/<positive_timestep>/policy.pt
```

List available integer checkpoint directories and select the checkpoint using
the task's declared training metric or an explicitly requested periodic/final
step. Never evaluate or report timestep 0. It is the untrained initialization.

If `models/best/` is considered, read `models/best/metadata.json` and require a
strictly positive integer `policy_timestep`. For a reproducible command, pass
that positive timestep explicitly and record any substitution from a stale or
invalid best checkpoint.

## 2. Run the standard aggregate evaluation

Run 50 episodes with task-conditioned success metrics and secondary DTW
metrics:

```bash
dexlink-eval-metrics \
  --rl_run_dir outputs/runs/example \
  --ckpt_timestep 50000 \
  --num_episodes 50 \
  --max_parallel_envs 50 \
  --dtw_window_fraction 0.10
```

Do not add `--sample_actions` for standard reporting. The evaluator defaults to
greedy actions, restores the training configuration from `exp_cfg.json`, and
writes aggregate metrics, per-episode metrics, rollout arrays, the resolved
evaluation config, and checkpoint metadata under its output directory.

Use `--record_video` when a visual audit is needed. All requested episodes must
fit in one parallel chunk for recording, so keep
`max_parallel_envs >= num_episodes` or run a separate small video evaluation.

W&B upload is optional and must not be the only result record. To upload after
local artifacts are written, add `--wandb_mode online` and explicit project,
entity, run-name, and group values appropriate to the user's account.

## 3. Run an interactive rollout only when needed

For continuous Genesis visualization rather than aggregate metrics:

```bash
dexlink-eval \
  --rl_run_dir outputs/runs/example \
  --ckpt_timestep 50000
```

This command is for inspecting behavior; it is not a replacement for the
50-episode metric evaluation.

## 4. Check and report outputs

Confirm that exactly the requested number of episodes were recorded, early
terminations count as failures, the configured task-specific metric exists,
and DTW uses the 0.10 window. Report:

- run directory, seed, preset, and positive checkpoint timestep;
- selection rule and any checkpoint substitution;
- number of episodes and greedy/stochastic action mode;
- task success with an episode-level confidence interval;
- relevant DTW or task-specific secondary metrics;
- artifact and video paths.

Do not infer missing task metrics from reward values. If the task has no valid
criterion, stop and add a task-conditioned metric through
`$dexlink-create-task` before reporting a success rate.

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
