# Task catalog

This directory contains the portable task specifications used by DexLink. Each
YAML file is a declarative input to `dexlink-reference`; generated
retargetings, reference trajectories, and policy artifacts are written under
`outputs/` and are not stored beside the specification.

Tasks are grouped into directories by task family and experiment. A typical
specification defines:

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

## Packaged task library and policies

The frozen catalog provides 40 native XHand tasks and three tutorials, with
42 selected positive policy checkpoints. Success varies across tasks. Use the
included references for training and rollout, and follow the YAML pipeline in
[new-task authoring](../docs/new-task.md) to create a task.

```bash
python -m dexlink list
python -m dexlink check all
python -m dexlink rollout invert_cup --episodes 5 --parallel-envs 5
python -m dexlink collect t047_a --episodes 10 --parallel-envs 5 --output outputs/data/stir
```

Install [assets/checkpoints](../docs/resources.md) first. The lift tutorial
needs fresh training; the other catalog rows have selected weights. The
[30 cup multi-task programs](../docs/multitask.md) are included separately.

| ID | Description | Positive checkpoint |
|---|---|---:|
| [t012_a](library40/t012_a/task.json) | Insert and twist a keyed part: pause at two engagement depths before completing the task | 30000 |
| [t009_a](library40/t009_a/task.json) | Insert and twist a keyed part: complete the fit | 12000 |
| [t001_a](library40/t001_a/task.json) | Insert a long peg: complete the fit | 10000 |
| [t001_b](library40/t001_b/task.json) | Insert a long peg: complete the fit | 6000 |
| [t017_a](library40/t017_a/task.json) | Seat a broad part in a recess: complete the fit | 48000 |
| [t002_b](library40/t002_b/task.json) | Insert a long peg: complete the fit and return to the original state | 30000 |
| [t002_a](library40/t002_a/task.json) | Insert a long peg: complete the fit and return to the original state | 6000 |
| [t006_b](library40/t006_b/task.json) | Extract a deeply seated peg: complete the fit and return to the original state | 50000 |
| [t006_a](library40/t006_a/task.json) | Extract a deeply seated peg: complete the fit and return to the original state | 16000 |
| [t004_a](library40/t004_a/task.json) | Insert a long peg: pause at two engagement depths before completing the task | 46000 |
| [t004_b](library40/t004_b/task.json) | Insert a long peg: pause at two engagement depths before completing the task | 10000 |
| [t005_a](library40/t005_a/task.json) | Extract a deeply seated peg: complete the fit | 40000 |
| [t005_b](library40/t005_b/task.json) | Extract a deeply seated peg: complete the fit | 50000 |
| [t007_b](library40/t007_b/task.json) | Extract a deeply seated peg: pause halfway through engagement before completing the task | 50000 |
| [t008_a](library40/t008_a/task.json) | Extract a deeply seated peg: pause at two engagement depths before completing the task | 16000 |
| [t010_b](library40/t010_b/task.json) | Insert and twist a keyed part: complete the fit and return to the original state | 42000 |
| [t011_a](library40/t011_a/task.json) | Insert and twist a keyed part: pause halfway through engagement before completing the task | 28000 |
| [t011_b](library40/t011_b/task.json) | Insert and twist a keyed part: pause halfway through engagement before completing the task | 28000 |
| [t056_a](library40/t056_a/task.json) | Turn a held plate by a quarter turn with both hands | 2000 |
| [t057_a](library40/t057_a/task.json) | Present a tilted plate toward the viewer | 2000 |
| [t058_a](library40/t058_a/task.json) | Raise a plate and lower it onto a marked support while holding its rim | 2000 |
| [t023_a](library40/t023_a/task.json) | Clean the same board strip with three return strokes | 2000 |
| [t024_a](library40/t024_a/task.json) | Trace a closed rectangle on the board with the eraser | 38000 |
| [t025_a](library40/t025_a/task.json) | Wipe a zigzag across a board patch | 50000 |
| [t047_a](library40/t047_a/task.json) | Stir a figure-eight through two regions of the bowl | 2000 |
| [t048_a](library40/t048_a/task.json) | Sweep the spoon across the bowl floor and lift it clear | 24000 |
| [t062_a](library40/t062_a/task.json) | Press the left and right bowl regions in order | 50000 |
| [t064_a](library40/t064_a/task.json) | Make two presses at increasing depths | 50000 |
| [t065_a](library40/t065_a/task.json) | Press in one bowl and then a second bowl | 46000 |
| [t066_a](library40/t066_a/task.json) | Turn the masher head before pressing the bowl center | 34000 |
| [t067_a](library40/t067_a/task.json) | Trace a short floor line with the masher and then retract | 10000 |
| [t069_a](library40/t069_a/task.json) | Lift the mouse for inspection and replace it | 4000 |
| [t038_a](library40/t038_a/task.json) | Place the mug inside a wide cup holder | 50000 |
| [t041_a](library40/t041_a/task.json) | Turn the mug handle toward the user before placing it | 50000 |
| [t042_a](library40/t042_a/task.json) | Lift and present the mug with its opening visible | 50000 |
| [t080_a](library40/t080_a/task.json) | Stand the remote upright inside a wide holder | 36000 |
| [t083_a](library40/t083_a/task.json) | Present the remote after a quarter turn in the hand | 26000 |
| [t033_b](library40/t033_b/task.json) | Square the keyboard against two desk alignment markers | 6000 |
| [t088_a](library40/t088_a/task.json) | Make two pour motions toward two separate receivers | 10000 |
| [t090_a](library40/t090_a/task.json) | Perform a pour motion and return the carton upright | 50000 |
| [lift_mug](tutorials/v2_03_lift_mug/task.json) | Reach, grasp and lift the mug 20 cm. | train first |
| [open_notebook](tutorials/v2_03_open_notebook/task.json) | Open the articulated notebook cover. | 29248 |
| [invert_cup](tutorials/v2_03_invert_cup/task.json) | Grasp and turn the cup over. | 50000 |
