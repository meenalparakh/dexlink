# XHand model notice

The base XHand URDFs, meshes, colored meshes, and retargeting configuration in
this directory were obtained from the
[`MandiZhao/dexmachina`](https://github.com/MandiZhao/dexmachina/tree/adae5bf620c57723d185b2757ee3ce9656927c20/dexmachina/assets/xhand)
repository at commit `adae5bf620c57723d185b2757ee3ce9656927c20`.
DexMachina distributes its repository under the MIT terms reproduced in
`LICENSE`.

DexLink preserves 68 files from that source byte-for-byte. DexLink additionally
contains:

- `xhand_*_6dof_abs*.urdf` variants derived for the DexLink simulation and
  keypoint conventions; and
- OBJ conversions under `meshes/` generated from the corresponding source
  robot meshes.

These additions and modifications do not remove the upstream copyright and
license notice. DexLink's root MIT license covers DexLink-authored changes;
the upstream MIT notice is retained here for the source material.
