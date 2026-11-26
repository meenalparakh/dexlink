---
name: dexlink-install
description: >-
  Install DexLink, select optional dependency groups, connect licensed external
  datasets and models without redistributing them, and verify a checkout. Use
  when setting up DexLink on a workstation or cluster, repairing missing-data
  errors, or preparing an environment for retargeting, RL, evaluation, or
  visualization.
---

# Install DexLink

Work from the repository root. Do not copy MANO, OakInk, ARCTIC, ShapeNet,
PartNet, YCB, DexMachina, or vendor assets into version control. Read
`THIRD_PARTY.md` before acquiring them and preserve every upstream license.

## 1. Inspect the target environment

Check Python, CUDA, the intended package stages, and available storage before
installing. Python 3.10 is the tested version. Reuse the user's requested Conda
or virtual environment; do not silently create a second environment.

Choose only the extras required by the requested workflow:

- `retargeting`: trajectory optimization and reference generation;
- `rl`: Genesis training and evaluation;
- `visualization`: MeshCat and offline rendering;
- `real`: optional real-robot dependencies;
- `test`: repository tests.

For a complete development environment, run:

```bash
python -m pip install -e '.[rl,retargeting,visualization,downloads,test]' --find-links third_party/wheels
```

Install a PyTorch3D build matching the installed PyTorch and CUDA versions by
following its official installation instructions. Do not guess a wheel URL or
change PyTorch versions merely to make dependency resolution pass.

## 2. Connect external data

Read `external_data/README.md` and `THIRD_PARTY.md`. Obtain each required
resource from its official provider after the user accepts its terms. Either
symlink the documented subdirectories under `external_data/`, or configure
separate roots:

```bash
export DEXLINK_ROOT=/path/to/dexlink
export DEXLINK_EXTERNAL_DATA=/path/to/dexlink_external_data
export DEXLINK_OUTPUTS=/path/to/dexlink_outputs
```

The external root must use the layout documented in
`external_data/README.md`. In particular, OakInk-backed tasks require MANO
model files at
`mano/mano_v1_2/models/{MANO_LEFT.pkl,MANO_RIGHT.pkl}`. Never fabricate an
alternate filename or search unrelated directories as a fallback; report the
exact missing path.

## 3. Verify the installation

Run the unit suite first:

```bash
pytest
```

Then validate one task that represents the intended dataset:

```bash
dexlink-validate-task tasks/v3_03_reusability/v3_03_reuse_bottle_side_scale_070.yaml
```

Treat a missing input as an installation issue. Fix its documented mount or
environment variable instead of editing task YAML to embed a machine-specific
absolute path.

Report the Python environment, installed extras, configured data/output roots,
validation command, and any resource that remains unavailable.

## Packaged workflow entry points

Start with `AGENTS.md`, `docs/installation.md`, `docs/resources.md`, and
`tasks/catalog.json`. The RL extra pins the tested public Genesis fork;
updating that revision requires validation. Use `python -m dexlink list`,
`check`, `train`, `smoke`, `rollout`, `collect` and `render` for catalog tasks.
Install the `downloads` extra, then run `dexlink-resources assets` and
`dexlink-resources checkpoints`; both use the public Google Drive links and
pinned checksums. For a manual download use `--archive PATH`.
Use `$dexlink-datasets` for provider datasets. Fresh learning/smoke requires a W&B account and online
logging; default pretrained rollouts and separate mesh rendering do not.
`docs/data-generation.md` defines collection timing and episode boundaries.
