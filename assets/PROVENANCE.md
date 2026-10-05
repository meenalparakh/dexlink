# Asset provenance

Except for the attributed XHand package described below, committed assets are
project-authored procedural meshes, primitive URDFs, or scene fixtures
generated for DexLink tasks. They contain no MANO parameters, OakInk or ARCTIC
meshes, or ShapeNet/PartNet/YCB geometry. `PROVENANCE.yaml` is the
machine-checked inventory of every committed asset group.

`robots/xhand/` was sourced from DexMachina under MIT. Its exact source commit,
license, attribution, and the DexLink-derived variants are documented inside
that directory.

Dataset-derived object models and any additional robot descriptions required at
runtime must be installed separately under `external_data/`; see
`../THIRD_PARTY.md`.
