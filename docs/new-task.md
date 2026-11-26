# Author a task

A portable YAML specification names source/target interactions, retargeted
keyframes, scene geometry and timed trajectory phases. Start with a YAML in
`tasks/`; `tasks/v2_03_beyond_grasp_div/v2_03_lift_mug.yaml` is a short lift
example. Give a new recipe a unique `experiment_id` and output paths under
`outputs/`. Positions are metres; quaternions are WXYZ.

Source/target registrations live in `dexlink/src/reference/common/categories/`.
The typed `SourceTemplate`/`TargetObject` contracts are in `exp_schema.py`.
A new target needs valid mesh/URDF geometry, metric scale, alignment, collision
geometry, joint limits and support surfaces. Do not replace missing source
files with similarly named objects. Follow the [dataset guide](../external_data/README.md)
and [task-authoring skill](../agent-skills/dexlink-create-task/SKILL.md).

```bash
conda activate hpart
dexlink-validate-task tasks/custom/my_task.yaml
dexlink-reference tasks/custom/my_task.yaml --max-workers 4 --refine-collisions
```

For a catalog tutorial, the wrapper can reuse its included cached retargetings:

```bash
python -m dexlink reference tasks/v2_03_beyond_grasp_div/v2_03_lift_mug.yaml \
  --output outputs/my-lift/reference.npy --density 1000 --skip-retargeting
```

Cached retargeting skips optimization; the native RL exporter still reads
the original human demonstration and MANO. New source/target pairs should
use the full pipeline. Inspect the generated motion before expensive learning;
a kinematic reference alone does not establish physical success.

```bash
conda activate genesis
python -m dexlink train outputs/my-lift/reference.npy \
  --run-directory outputs/runs/my-lift-seed42 --num-envs 1024 \
  --control-steps 50000 --entity YOUR_ENTITY --project dexlink
```

Use the output filename actually printed by the builder/refiner. To evaluate
your own newly trained run, use `dexlink-eval-metrics --rl_run_dir RUN_DIR
--ckpt_timestep POSITIVE_STEP --num_episodes 50 --max_parallel_envs 10`.
Add `--record_policy_inputs` for observation/action trajectories, then use
`python -m dexlink render --eval-directory EVAL_DIR --output outputs/video.mp4`.
The catalog `rollout`/`collect` commands select the bundled policies;
they do not automatically replace them with your newly trained run.

The 40-task library provides native references and goals for training and
rollout. Use the YAML workflow above to author additional tasks.
