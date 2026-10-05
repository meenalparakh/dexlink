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
