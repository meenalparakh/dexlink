---
name: dexlink-datasets
description: >-
  Install or locate MANO, OakInk, ARCTIC and task geometry for a specified DexLink operation. Use for dataset access, preprocessing contracts or missing task inputs.
---

# Set up DexLink datasets

Read `external_data/README.md`, `THIRD_PARTY.md` and `docs/resources.md`.
Determine whether the request is packaged policy execution or new reference
generation. Packaged rollout requires runtime geometry and positive weights,
not raw datasets. Cached retargeting still needs the source demo/MANO at export.
Use official provider downloads and preserve their folder/ID contracts. Never
substitute similarly named files or fall back to lab/cluster directories.
Use `dexlink-validate-task TASK.yaml` for authoring inputs; use
`python -m dexlink check TASK_ID` for the packaged reference/weight identities.
Install the `downloads` extra, then run `dexlink-resources assets` for the
public runtime geometry and `dexlink-resources checkpoints` for trained weights.
The Google Drive URLs and pinned archive/file hashes are in
`resources/manifest.json`. For a manually downloaded ZIP, use `--archive PATH`.
Do not claim arbitrary raw downloads recreate the registered processed demos.
Report exact missing files and the applicable preparation step.
