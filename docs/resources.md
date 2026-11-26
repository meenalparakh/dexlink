# Runtime assets and checkpoints

The source includes task definitions, references and small project assets. The
runtime asset ZIP adds the exact geometry used by the 40-task library, three
tutorials and 30 cup programs, plus the project SAM3D authoring targets
and ShapeNet alignment metadata. It contains no raw MANO/OakInk/ARCTIC dataset.
Dataset-derived runtime meshes retain their provider terms; see THIRD_PARTY.md.

## Checkpoints

Download [the checkpoint ZIP from Google Drive](https://drive.google.com/file/d/1eBwPPSprh8zNtIJUVWrqJyna6K07ZqTb/view?usp=sharing).
It contains 42 selected positive checkpoint sets: 40 library policies plus
notebook-open and cup-inversion policies. The lift tutorial provides a
reference for training, without pretrained weights.

```bash
conda activate genesis
# The downloads extra supplies gdown; no Drive login is needed.
dexlink-resources checkpoints
# Or install a manually downloaded archive:
dexlink-resources checkpoints --archive /path/to/dexlink-checkpoints.zip
```

Archive SHA-256:
`733121a1a549d5863248666d5f43d4d0ecd68f671e06548249d803879cb19920`.
The installer also validates each file and the task catalog's trained-state
hashes. It accepts identical existing files and refuses conflicting content.

## Runtime geometry

Download the [runtime asset ZIP](https://drive.google.com/file/d/1O4BduFPzy9-XJU65-MPxcDxnaM-Q5mgr/view?usp=sharing) and its
[SHA-256 checksum file](https://drive.google.com/file/d/1NjfltcXXbDj-YF9b2ty1FYVsMMfGjQVh/view?usp=sharing) from Google Drive.
Both are accessible without authentication. With the `downloads` extra installed:

```bash
dexlink-resources assets
python -m dexlink check all
```

The installer downloads to `outputs/downloads/` (or the corresponding
`DEXLINK_OUTPUTS` directory), verifies the pinned archive SHA-256 and every
installed file, and refuses conflicting existing content. To use a manually
downloaded archive instead:

```bash
dexlink-resources assets --archive /path/to/dexlink-runtime-assets.zip
```

Archive SHA-256:
`11f3aab41921aa7ca19af80ef084e60bea3a6d61e3946183826c54c0dca1afbd`.
The download links and archive/file checksums are recorded in
`resources/manifest.json`. No cluster access is required. The prepared local
copy remains `bundles/dexlink-runtime-assets.zip`.

Checkpoints and downloaded runtime geometry are ignored by Git. Do not use
`unzip` to overwrite existing checkpoints/configurations; use the verified
installer. Raw source data are only needed when authoring/exporting new
references or explicitly requesting human-template metrics; see
[the dataset guide](../external_data/README.md).
