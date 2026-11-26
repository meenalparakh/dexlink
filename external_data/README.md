# External data setup

This directory contains only instructions and ignored mount points. Download
each resource from its official provider and keep its original license separate
from DexLink. Do not commit dataset files, MANO parameters, third-party object
meshes, or vendor robot assets.

You may either place data in the expected directories or symlink each directory
to storage outside the repository. A single external root may instead be set
with `DEXLINK_EXTERNAL_DATA`.

Expected structure:

```text
external_data/
├── arctic/
├── dexmachina/
├── mano/
│   ├── mano_v1_2/models/{MANO_LEFT.pkl,MANO_RIGHT.pkl}
│   └── arctic_faces/{left_faces.npy,right_faces.npy}
├── oakink/
├── partnet/
├── robots/
│   ├── inspire_hand/       # optional; obtain from its official provider
│   └── frankaX_hand/       # optional Franka-XHand model
├── sam3d/          # project targets from the DexLink asset ZIP
├── shapenet/
└── ycb/
```

The XHand-only task suite does not require either optional robot directory:
its separately licensed model is bundled under `assets/robots/xhand/`. All
other robot descriptions remain external and retain their upstream terms.

See the source links and licensing boundary in `../THIRD_PARTY.md`.

After installing data, use `dexlink-validate-task path/to/task.yaml` to list
all missing inputs for one task before launching retargeting. The validator
honors `DEXLINK_EXTERNAL_DATA`.

## Choose inputs by operation

| Operation | Required downloads |
|---|---|
| Packaged policy rollout/collection or reference rendering | Runtime asset ZIP and, for policies, checkpoint ZIP; no raw datasets |
| Rebuild a tutorial from cached grasps | Its source human demo, MANO models, required source/target meshes and cached templates |
| New interaction retargeting | Registered human/object demo, MANO, target geometry and alignment/scale files |
| Human-template evaluation metrics | The explicitly configured human data/models |

Install [runtime resources](../docs/resources.md) before using the frozen
catalog. These are separate from the provider datasets below.

## Provider downloads and preprocessing

- **MANO:** register at [the official site](https://mano.is.tue.mpg.de/), accept
  its terms, and install v1.2 `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` under
  `mano/mano_v1_2/models/`. Never package these model files with DexLink.
- **OakInk:** follow [the official dataset instructions](https://github.com/oakink/OakInk/blob/main/docs/datasets.md).
  Obtain the interaction annotations and object meshes needed by your selected
  category. Unpack the provider layout under `oakink/`; do not rename objects
  to approximate missing IDs. Category registrations in
  `dexlink/src/reference/common/categories/` declare the exact demo and geometry
  paths consumed by the existing source/target processors. In particular,
  Shape annotations belong at
  `oakink/shape/oakink_shape_v2/CATEGORY/OBJECT_ID/DEMO_ID/hand_param.pkl`.
  Keep the provider object-mesh directories alongside the Shape data. DexLink's
  `run_sequence_retargetings`, category retargeting and
  `export_oakink_sequence_data` modules perform the supported conversion into
  native robot/hand/object arrays; do not fabricate processed NPZ files.
  The project asset ZIP also installs SAM3D targets and project-authored
  ShapeNet alignment JSONs into their registered external-data paths.
- **ARCTIC:** follow [the official data guide](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md)
  and the upstream `scripts_data/process_seqs.py` preprocessing. DexLink's
  adapter is `python -m dexlink.src.reference.arctic.prepare --help`; it checks
  processed sequence vertices and source object assets and writes a preparation
  manifest. Use its explicit `--arctic-root`, `--dexmachina-root` and output
  arguments for your downloaded installations.
- **Other objects:** use the ShapeNet, PartNet and YCB links in
  [THIRD_PARTY.md](../THIRD_PARTY.md) only if your category requires them.
  Runtime task geometry is supplied separately; raw providers are needed for
  new source data or target geometry outside that pack.

`DEXLINK_EXTERNAL_DATA` redirects the root, preserving the tree above. Do not
use cluster paths as defaults. Run `dexlink-validate-task TASK.yaml` before
retargeting; its missing-input report gives the exact local paths to prepare.
`python -m dexlink check all` verifies the frozen catalog and selected policies,
not the completeness of arbitrary new raw datasets. Dataset access, upstream
terms, raw-to-processed conversions and task validation are distinct steps.

MANO-based retargeting uses [manotorch](https://github.com/lixiny/manotorch),
pinned in the retargeting extra. Semantic hand-keypoint face caches are
generated from your installed MANO models by
`dexlink.src.retargeting.data.human_hand_kpts`; no model parameters are
distributed in the source or asset archive.
