---
name: dexlink-submit-cluster
description: >-
  Prepare and submit DexLink Slurm learning/evaluation jobs with explicit site resources, persistent outputs and W&B. Use when the user requests cluster execution.
---

# Submit DexLink cluster jobs

Read `cluster/README.md` and the selected `.sbatch` example. Set the site's
account/partition/GPU, memory/time and persistent checkout/output/data paths;
public templates contain no lab accounts or cluster mounts. Use the user's
authorized account and check existing jobs before a retry to avoid duplicates.
Training requires online W&B and an actual finite local PPO smoke record.
Use a representative profile for equivalent tasks; retest material runtime
changes. Keep checkpoints in persistent storage and use new run directories
for fresh training. The invert-cup example uses a four-hour allocation and
stops after a complete PPO update before its wall limit, preserving a positive
checkpoint. Source bashrc and activate Conda before enabling nounset.
Record submitted job ID, effective resources, resolved paths and W&B URL;
do not claim training/learning passed from submission or a running job alone.
