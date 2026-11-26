# DexLink

DexLink turns sparse hand–object interaction templates and object-motion
specifications into robot reference trajectories, then learns residual policies
that make those references physically executable in simulation.

This repository contains framework source, task YAML files, frozen reference
trajectories, procedural task assets and robot descriptions. Dataset and
model dependencies that are not distributed with DexLink are installed under
`external_data/`; see [external_data/README.md](external_data/README.md).

Start an agent at [AGENTS.md](AGENTS.md), which links to all workflow skills.

## Try a trained policy and generate data

Install [runtime assets and checkpoints](docs/resources.md). Checkpoints are
available at [this Google Drive link](https://drive.google.com/file/d/1eBwPPSprh8zNtIJUVWrqJyna6K07ZqTb/view?usp=sharing).
Runtime assets are available from [Google Drive](https://drive.google.com/file/d/1O4BduFPzy9-XJU65-MPxcDxnaM-Q5mgr/view?usp=sharing),
with a separate [checksum file](https://drive.google.com/file/d/1NjfltcXXbDj-YF9b2ty1FYVsMMfGjQVh/view?usp=sharing).

```bash
dexlink-resources checkpoints
dexlink-resources assets
python -m dexlink check all
python -m dexlink rollout invert_cup --episodes 5 --parallel-envs 5
python -m dexlink collect invert_cup t047_a --episodes 10 --parallel-envs 5 \
  --output outputs/data/examples
python -m dexlink render --eval-directory outputs/data/examples/invert_cup \
  --output outputs/videos/invert-cup.mp4
```

The catalog contains 40 frozen tasks plus three tutorials, with 42 selected
trained policies. Default pretrained execution needs no raw source datasets
or W&B account. Fresh training needs online W&B; new reference generation
needs the relevant [source data](external_data/README.md).
See [data generation](docs/data-generation.md), [new tasks](docs/new-task.md),
[multi-task policies](docs/multitask.md), and [cluster jobs](cluster/README.md).

## Repository layout

- [`tasks/`](tasks/README.md): one portable YAML specification per experiment task.
- `assets/`: project-authored task assets and the XHand model.
- `external_data/`: ignored local mount points for external datasets and models.
- `dexlink/src/retargeting/`: hand–object template retargeting.
- `dexlink/src/reference/`: reference-motion construction.
- `dexlink/src/rl/`: multi-object Genesis environments and PPO agents.
- `dexlink/scripts/`: supported command-line entry points.
- `dexlink/visualizations/`: MeshCat, Genesis, and offline renderers.
- `cluster/`: scheduler-neutral Slurm templates.
- `outputs/`: generated references, checkpoints, metrics, and videos (ignored).

## Installation

Use Python 3.10 and the pinned public [Genesis fork](https://github.com/meenalparakh/Genesis)
at `65883d3e9ad1c6704dc5217be742ba63e0275a73`. Follow the
[installation guide](docs/installation.md) for separate RL and retargeting
environments, compatible PyTorch3D and tested dependency constraints.

Configure local data either by symlinking the documented subdirectories under
`external_data/`, or by pointing DexLink at a separate directory:

```bash
export DEXLINK_EXTERNAL_DATA=/path/to/dexlink_external_data
export DEXLINK_OUTPUTS=/path/to/dexlink_outputs
```

Validate a task before starting an expensive retargeting run. Missing external
inputs are reported together, with their resolved locations:

```bash
dexlink-validate-task \
  tasks/v3_03_reusability/v3_03_reuse_bottle_side_scale_070.yaml
```

## Generate a reference trajectory

The supported end-to-end command is:

```bash
dexlink-reference tasks/v3_03_reusability/v3_03_reuse_bottle_side_scale_070.yaml \
  --max-workers 8 \
  --refine-collisions
```

It runs independent retargeting jobs in parallel, builds the task reference,
exports the RL-readable sequence, and optionally applies Drake collision
refinement. Every output is written under `outputs/` unless
`DEXLINK_OUTPUTS` is set.

## Train and evaluate

The multi-object environment is the default public simulation environment:

```bash
dexlink-train --preset seq_oakink_residual_target_trajectory_weighted_policy_v3 \
  --env.task_cfg.oakink_exported_path outputs/path/to/reference.npy
```

Evaluation requires an explicitly positive checkpoint timestep. Timestep zero
is rejected because it is the untrained initialization:

```bash
dexlink-eval \
  --rl_run_dir outputs/runs/example \
  --ckpt_timestep 50000
```

Use `--sample_actions` only when stochastic evaluation is intentionally desired;
the default evaluation action is the policy mean.

## Visualize generated artifacts

Inspect a retargeting or assembled reference interactively in MeshCat:

```bash
python -m dexlink.visualizations.retargeting.vis_robot_retgt_motion \
  --retargeted_results outputs/path/to/reference.npz \
  --mesh_type visual \
  --port 6001
```

Render references or saved evaluation episodes with `python -m dexlink render`.
This standalone mesh renderer consumes saved motion and geometry; it does not
rerun policy inference. See [the data guide](docs/data-generation.md).

## Tests

```bash
pytest
```

Unit tests do not require external datasets. Tests marked `integration` require
the data and simulator assets described under `external_data/`.

## Public release checklist

- [x] Make the [tested Genesis fork](https://github.com/meenalparakh/Genesis) publicly accessible.
- [ ] Update the Genesis dependency to the most recent fork commit, rerun reference,
  positive-checkpoint rollout, data collection/rendering and finite PPO-update tests,
  then update the installation pin. The current tested pin is
  `65883d3e9ad1c6704dc5217be742ba63e0275a73`.
- [x] Provide task specifications, training/evaluation, data collection/rendering,
  multi-task learning and agent skills.
- [x] Publish the project asset archive and checksum, add their download URLs,
  and verify anonymous download against the pinned SHA-256.
- [x] Download and install hosted assets into an empty destination; verify all
  1,536 installed file checksums.
- [x] Link upstream download instructions for MANO, OakInk and ARCTIC;
  keep licensed source datasets/models out of the source repository.
- [x] Verify anonymous checkpoint download, archive/file hashes and positive
  checkpoint selection for all 42 packaged policies.
- [x] Run public unit tests, cached lift rebuilding, standalone rendering and
  cup task-bank construction; see [validation](docs/validation.md).
- [x] Verify a clean base/download/visualization installation, task checks and
  headless rendering; resolve the complete RL dependency set.
- [ ] Repeat the full CUDA/retargeting installation on another clean machine.
- [x] Repeat positive-checkpoint rollout, observation/action collection, saved
  episode rendering and finite PPO optimizer smoke on the public layout.
