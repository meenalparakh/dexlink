# Third-party resources

The DexLink MIT license covers only DexLink-authored code and project-authored
assets. It does not grant rights to datasets, body models, third-party robot
descriptions, object meshes, or comparison-method implementations.

## Redistributed robot model

The base XHand URDFs and meshes under `assets/robots/xhand/` were obtained from
[DexMachina](https://github.com/MandiZhao/dexmachina/tree/adae5bf620c57723d185b2757ee3ce9656927c20/dexmachina/assets/xhand),
which distributes its repository under MIT. The exact source commit, copied
files, DexLink-derived variants, upstream license, and attribution are recorded
in that directory's `NOTICE.md` and `LICENSE`.

## Adapted source code

The point-to-triangle distance helpers in
`dexlink/utils/retargeting/math_utils.py` are adapted from PyTorch3D's
point-to-mesh distance tests. PyTorch3D is distributed under a BSD license; the
copyright notice and license terms are reproduced in
`third_party_licenses/PYTORCH3D-BSD.txt`.

PyTorch3D is also an optional runtime dependency for mesh sampling, Chamfer
distance, and grasp metrics. DexLink does not redistribute the PyTorch3D
package itself.

DexLink does not redistribute the following resources:

| Resource | Obtain from | Expected local location |
|---|---|---|
| OakInk / OakInk-Shape | <https://github.com/oakink/OakInk/blob/main/docs/datasets.md> | `external_data/oakink/` |
| ARCTIC | <https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md> | `external_data/arctic/` |
| MANO v1.2 | <https://mano.is.tue.mpg.de/> | `external_data/mano/` |
| DexMachina comparison code/assets | the authors' official release | `external_data/dexmachina/` |
| ShapeNet | <https://shapenet.org/> | `external_data/shapenet/` |
| PartNet | <https://partnet.cs.stanford.edu/> | `external_data/partnet/` |
| YCB object models | <https://www.ycbbenchmarks.com/> | `external_data/ycb/` |
| Additional robot descriptions | the applicable robot vendor or official model repository | `external_data/robots/` |

Users are responsible for accepting upstream terms, preserving attribution,
and ensuring their intended use is permitted. In particular, MANO files are
license-gated and must not be committed, packaged, or redistributed with
DexLink.
