# DexLink implementation modules

The public implementation is divided by the three stages of the method:

- `retargeting/` transfers a human hand–object interaction template to XHand
  and a selected target object. Licensed human/object data are read from
  `DEXLINK_EXTERNAL_DATA`; generated optimization artifacts go to
  `DEXLINK_OUTPUTS`.
- `reference/` turns the retargeted configurations and the phases declared in
  a task YAML into one dense, RL-readable reference trajectory.
- `rl/` contains the Genesis multi-object environment, task-conditioned
  metrics, PPO implementation, and public method/ablation presets.

Supported entry points are listed in the root `README.md`. Malformed inputs
fail with the expected shape, field, or path in the error message.
