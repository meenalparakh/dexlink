# Task catalog

This directory contains the portable task specifications used by DexLink. Each
YAML file is a declarative input to `dexlink-reference`; generated
retargetings, reference trajectories, and policy artifacts are written under
`outputs/` and are not stored beside the specification.

Task directories preserve the experiment group names used in the paper. A
typical specification defines:

- the source hand-object interaction template;
- the target object mesh or URDF and its alignment to the template;
- scene objects and their initial poses;
- ordered task stages and object-motion keyframes; and
- task-specific simulation metadata used by reference generation and RL.

Paths beginning with `assets/` resolve inside this repository. Licensed inputs
such as OakInk, ARCTIC, MANO, and ShapeNet resolve through
`DEXLINK_EXTERNAL_DATA`; see [`external_data/README.md`](../external_data/README.md).
An `auto` motion path asks the reference pipeline to derive the generated path
under `DEXLINK_OUTPUTS` rather than embedding a machine-specific location.

Validate a specification before running optimization:

```bash
dexlink-validate-task tasks/v3_03_reusability/v3_03_reuse_bottle_side_scale_070.yaml
```

Generate its reference trajectory with:

```bash
dexlink-reference tasks/v3_03_reusability/v3_03_reuse_bottle_side_scale_070.yaml \
  --max-workers 8 \
  --refine-collisions
```

The browser-based task authoring interface is intentionally maintained in a
separate repository. DexLink only defines and consumes the exported YAML
format.
