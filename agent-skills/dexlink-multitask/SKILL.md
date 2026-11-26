---
name: dexlink-multitask
description: >-
  Prepare the included 30 cup task programs and run the conditional full-method multi-task learner. Use for task-bank construction, geometry/program conditioning or synchronized checkpoint workflows.
---

# Run conditional cup policies

Read `docs/multitask.md` and `docs/installation.md`. Run
`python -m dexlink.workflows.multitask --output outputs/cup30-data` to build
the tensor bank and immutable manifests. Select explicit GPU/environment
resources for the actual host; production defaults are eight GPUs and 2,048
environments per task. Launch `python -m dexlink.multitask_cup.launch_full_method
--config outputs/cup30-data/training-config.json` with the desired typed overrides.
Keep full-method hyperparameters, task identities and equal task exposure;
use fresh run IDs or explicit validated resume lineage. Training needs a
writable W&B account/entity/project. Run the optimizer smoke before an
expensive submission. Rebuild the bank after source or asset changes. The
downloaded checkpoint ZIP contains single-task policies, not a cup conditional
policy. Do not claim multi-task GPU validation from bank/unit tests.
