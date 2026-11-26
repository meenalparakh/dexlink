# Cluster examples

These Slurm examples use your own cluster account and storage. Install the
public checkout, pinned Genesis fork and [resources](../docs/resources.md) on
the cluster's persistent project filesystem. The public package requires no
lab mounts. Set your site's account/partition/GPU arguments explicitly.

Before new learning, run a representative local PPO optimizer smoke using
`python -m dexlink smoke invert_cup --run-directory outputs/runs/smoke
--control-steps 64 --num-envs 32 --entity YOUR_ENTITY`. Review
`numerical-audit.json`, actual finite updates and the matching runtime profile.
Changing physics, learner or memory capacity requires a new qualification.
Copy the clean report and configuration with your job inputs. Do not submit
on the basis of a zero-action environment test alone.

```bash
export DEXLINK_ROOT="$PWD"
export DEXLINK_RUN_DIRECTORY=/path/to/persistent/runs/invert-seed42
export DEXLINK_SMOKE_REPORT=/path/to/qualified-smoke/numerical-audit.json
export DEXLINK_OUTPUTS=/path/to/persistent/dexlink-outputs
export DEXLINK_EXTERNAL_DATA=/path/to/external-data
export WANDB_ENTITY=YOUR_ENTITY
export WANDB_PROJECT=dexlink
# Run wandb login on the compute-visible account; do not put API keys in scripts.
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION cluster/train_invert_cup.sbatch
```

The four-hour example stops after a complete PPO update at approximately
3h50m, leaving time to save all positive policy/processor/value/optimizer
state. Check `training-result.json` for the actual endpoint and W&B URL;
the wall-clock setting is not a guarantee of any particular success rate.
Use a fresh run directory and inspect existing jobs before retrying. Native
periodic checkpoints remain enabled; keep them out of node-local scratch.

To collect numerical rollouts from a bundled policy:

```bash
export DEXLINK_TASK=invert_cup
export DEXLINK_EVAL_DIRECTORY=/path/to/persistent/data/invert-evaluation
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION cluster/evaluate_policy.sbatch
```

For your newly trained run, use `dexlink-eval-metrics` with the explicit run
directory and a strictly positive saved step. The catalog collection example
continues to select the bundled policy. Render saved trajectories separately
using [the rendering guide](../docs/data-generation.md). Evaluation and mesh
rendering do not need W&B authentication by default.

The existing `slurm_job.sh` accepts `DEXLINK_COMMAND` for other supported
commands. All examples source bashrc and activate Conda before enabling
nounset. No submission, account allocation or authentication is performed by
installation, data preparation or reading these examples.
