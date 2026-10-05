# DexLink

DexLink turns sparse hand–object interaction templates and object-motion
specifications into robot reference trajectories, then learns residual policies
that make those references physically executable in simulation.

This repository contains the source code, task YAML files, procedural task
assets, and robot descriptions needed by the published tasks. Dataset and
model dependencies that are not distributed with DexLink are installed under
`external_data/`; see [external_data/README.md](external_data/README.md).

The task-specification web interface is maintained in a separate repository.
DexLink consumes the YAML files exported by that interface but does not contain
the interface itself.

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

Python 3.10 is the tested version. Install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[rl,retargeting,visualization,test]'
```

Retargeting additionally requires a PyTorch3D build compatible with the
installed PyTorch and CUDA versions; follow the
[official PyTorch3D installation instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
The optional real-robot modules use `pip install -e '.[real]'` and require the
appropriate robot/camera SDKs. Install MANO and the other external inputs under
`external_data/` rather than copying them into the package.

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
dexlink-train --preset seq_oakink_v3_02_full_method \
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

Render a saved policy rollout with Genesis using the scripts under
`dexlink/visualizations/rl/`. Rendering and evaluation consume the same task
reference and checkpoint; evaluation still requires a strictly positive
checkpoint timestep.

## Tests

```bash
pytest
```

Unit tests do not require external datasets. Tests marked `integration` require
the data and simulator assets described under `external_data/`.
