# Generate data, evaluate policies and render separately

The package supports two outputs: authored kinematic references, and executed
policy demonstrations. Generate numerical data first, then render the saved
motion separately. Video appearance does not establish physical task success.

## Build task references

Follow [new-task authoring](new-task.md) and [data setup](../external_data/README.md).
In `hpart`, build a portable YAML recipe using cached templates or new retargeting:

```bash
python -m dexlink reference tasks/v2_03_beyond_grasp_div/v2_03_lift_mug.yaml \
  --output outputs/data/my-lift/reference.npy --density 1000 --skip-retargeting
```

This writes the scalar-dictionary RL NPY, native NPZ, resolved specification
and build manifest. Cached retargeting skips optimization, but export still
needs the source human demo/MANO. New objects or source interactions require
their explicit registrations, geometry and alignments. Use `--refine-collisions`
when desired and use its `_refined.npy` downstream.

The 40 library tasks already include frozen NPZ references, geometry, goals and
task metadata; they can be trained, executed and rendered directly. Their
separate ongoing authoring generator is excluded. New task authoring uses the
included portable YAML pipeline rather than regenerating library40's source.

## Collect policy demonstrations

Install `genesis` and extract the checkpoint ZIP. Raw source datasets, MANO and
W&B credentials are unnecessary for the default collection/evaluation workflow.
Use a fresh destination on persistent storage:

```bash
python -m dexlink collect invert_cup t047_a \
  --episodes 10 --parallel-envs 5 --seed 42 --output outputs/data/examples

# Explicitly collect the full trained library; this can take substantial GPU time.
python -m dexlink collect --group library40 \
  --episodes 50 --parallel-envs 10 --output outputs/data/library40
```

Collection validates positive checkpoint hashes, runs complete initial-state
episodes with greedy actions, and computes the native evaluation metrics.
Every episode is retained, including failures. A task with no checkpoint, such
as `lift_mug`, must be trained first. Collection creates:

| File | Contents |
|---|---|
| `dataset.json` | Selection, seed, status, reference/checkpoint hashes, counts and per-task metrics |
| `TASK/rollouts/chunk_NNNN.npz` | Numerical trajectories batched by parallel environment |
| `TASK/dataset-index.json` | Chunk fields/shapes/dtypes, first-episode lengths and valid-transition counts |
| `TASK/final_eval_exp_cfg.json` | Resolved simulator, observation/action and object/joint definitions |
| `TASK/metadata.json` | Policy identity, seed, episode count and evaluation arguments |
| `TASK/episode_metrics.csv`, `summary_metrics.json` | Per-episode results and aggregate success/errors |
| `TASK/reference_trajectory.npz` | Numerical target trajectory used in evaluation |

These runtime metrics are normal framework outputs; paper result ledgers and
figure/table assembly workflows are excluded from the distribution.

Chunks use time-major arrays `(T, N, ...)`. The new collection route saves:

- `policy_observation`: pre-step raw observation passed to the evaluator's policy,
  before its saved observation preprocessor; `(T,N,D_obs)`.
- `policy_action`: exactly the pre-step action vector passed to `env.step`;
  `(T,N,D_action)`. Interpret it using `env.action_cfg` and the policy config.
- `object_pos`, `object_quat`, `object_arti`: post-step/pre-reset object state,
  with object order in the saved reference/config. Quaternions are WXYZ.
- Active-hand `*_jvals`, `*_wrist_pos`, `*_wrist_quat`: post-step robot state.
- `reward`, `terminated`, `truncated`, `done`: transition labels.
- `action_command_timestep`, `command_timestep`, `start_timestep`: reference
  timing. Registered task metrics have the `task_metric__` prefix.

Each parallel environment contributes its **first complete episode**. Some
chunks contain extra rows after that environment ended while others finish.
Use the recorded lengths to exclude those padded/reset rows:

Each chunk's `first_episode_index` gives its starting row in
`episode_metrics.csv`; environment columns preserve the following episode order.

```python
import json
from pathlib import Path
import numpy as np

task_dir = Path('outputs/data/examples/invert_cup')
index = json.loads((task_dir / 'dataset-index.json').read_text())
chunk = index['chunks'][0]
with np.load(task_dir / chunk['path']) as data:
    lengths = np.asarray(chunk['episode_lengths'])
    valid = np.arange(data['done'].shape[0])[:, None] < lengths[None, :]
    observations = data['policy_observation'][valid]
    actions = data['policy_action'][valid]
```

The first terminal transition is included. These observation/action pairs can
support imitation learning. A raw next observation is not exported as a generic
RL transition field: simulator autoreset needs explicit terminal-state handling.
The physical state arrays are post-step, not an aligned pre-step state label.
Do not mix the two timings or infer action semantics from vector width.

An interrupted collection leaves a `failed` manifest and completed task outputs.
Inspect the error and choose a fresh destination; the command does not guess
resume lineage or silently merge datasets.

## Render saved references or policy episodes

These commands use the packaged URDF/mesh renderer and EGL. They require no
policy inference or simulator stepping, so change views without collecting
the trajectories again:

```bash
conda activate genesis
python -m dexlink render --task t047_a \
  --output outputs/videos/t047-reference.mp4
python -m dexlink render --reference outputs/data/my-lift/reference.npy \
  --task-id my_lift --output outputs/videos/my-lift-reference.mp4
python -m dexlink render --eval-directory outputs/data/examples/invert_cup \
  --task-id invert_cup --episode 0 --camera-mode fixed \
  --output outputs/videos/invert-episode0.mp4
```

Native NPZ and scalar-dictionary NPY references are supported explicitly. For
data moved to another machine, add `--reference-override` with the exact local
copy identified by the collection manifest. The adjacent render JSON records
source kind, episode, frame selection, reference and camera settings. A preview
renders target kinematics; a saved episode renders the actual executed states.

Defaults are 640×480, frame stride 2 and 15 FPS, preserving 30 Hz source timing.
All declared scene objects, including static fixtures, remain visible. The
tracked camera focuses on the hand and task-relevant objects; fixed camera
framing covers the full scene.
For other strides, set FPS accordingly to retain real-time playback. Width,
height, stride, FPS and camera mode are explicit CLI options. Video and JSON
outputs must be new paths.

## Evaluate a newly trained policy

The native evaluator also accepts an explicit saved run and positive timestep:

```bash
python -m dexlink.scripts.rl.eval_final_metrics \
  --preset seq_oakink_eval_sim --rl_run_dir outputs/runs/my-task \
  --ckpt_timestep 50000 --num_episodes 50 --max_parallel_envs 10 \
  --greedy_actions --record_policy_inputs --no-compute_template_metrics \
  --output_dir outputs/eval/my-task
```

Replace 50000 with an actually saved positive checkpoint. The same standalone
renderer reads that output. Evaluate using the task's declared success predicate,
count all requested episodes, and keep frame-level training signals separate
from complete-episode success rates. Training/PPO smoke require W&B login;
default collection, evaluation and rendering do not.
