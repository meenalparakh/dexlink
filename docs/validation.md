# Validation

The following checks were run on 2026-11-25 using Python 3.10 in the
`genesis` and `hpart` environments.

| Check | Outcome |
|---|---|
| Public tests, including full-method multi-task policy/learner regressions and resource installation protections | 125 passed |
| Catalog references and selected positive checkpoint hashes | 43 tasks; 42 policies verified |
| Native task full-method configurations | 40 resolved successfully |
| Downloaded learned-state tensors | 209 policy/value/preprocessor/optimizer files checked finite |
| Anonymous Google Drive downloads | Checkpoint and asset ZIPs match their pinned SHA-256 values; both hosted checksum files match too |
| Hosted asset installer | Downloaded anonymously into an empty destination; all 1,536 installed files verified |
| Hosted checkpoint installer | Downloaded archive matches the prepared public bundle; 251 files and all 42 configurations verified in an empty destination |
| Resource and setup-skill checks after connecting Drive assets | Eight targeted tests passed; all six updated skills passed the skill validator |
| Cached lift tutorial reference build using locally installed licensed source data | Native NPZ, RL NPY and build manifest produced |
| Standalone reference and saved-policy rendering | Native stirring reference and saved cup-policy episode MP4s/metadata produced |
| Public policy data collection | Two complete episodes each for invert_cup and t047_a, with finite observations/actions and episode indexes |
| Actual PPO optimizer smoke | 64 controls, 32 optimizer updates, zero nonfinite transitions out of 2,048; positive checkpoint saved |
| Online PPO logging | Eight remote history rows verified at controls 8 through 64; short audit runs also log every update |
| Clean installation | Fresh Python environment installed base, downloads and visualization extras; pip check clean; URDF/material/FK/EGL rendering and task checks passed |
| Full RL dependency resolution | Documented RL extras and pinned public Genesis fork resolved successfully with the local dependency wheels |
| Cup task-bank construction | 30 programs, seven geometries, maximum 124 frames |
| Public agent skills | All ten packages passed the skill validator |
| Slurm examples and whitespace checks | Shell syntax and git diff checks passed |
| Genesis source | Tested public pin matches fork HEAD; no local runtime source modifications |

The renderer supports native NPZ references, exported NPY sequences and saved
policy episodes. Run it with `python -m dexlink render`. The task catalog
records reference and checkpoint hashes for input validation.

The asset archive also supplies the project's SAM3D target geometry and
ShapeNet alignment JSONs used by authoring. Official provider instructions
remain required for licensed source datasets and MANO models. No raw MANO
model parameters are included.

## Remaining scope and prerequisites

- Full learning/collection runs need an NVIDIA GPU, driver and compatible
  Torch/CUDA/PyTorch3D installation. The existing validated CUDA environment
  was used for the GPU tests above; a second clean machine's complete CUDA
  build and full-budget retraining have not been repeated.
- MANO, OakInk and ARCTIC source data must be obtained from the providers for
  new human-template retargeting. Default packaged policy execution uses the
  downloadable runtime geometry and checkpoints without those raw datasets.
- Training needs an online W&B account/entity/project; evaluation, collection
  and offline mesh rendering do not need W&B authentication by default.
- Conditional multi-task bank construction and policy/learner regression tests
  passed. Full multi-task GPU training has not been validated by these checks.
- Revalidate after any future Genesis pin update. The current tested pin
  already matched the public fork HEAD when checked.

Two clean-install blockers were fixed by shipping small attributed URDFpy and
Pyrender wheels: the tested parser's NumPy/NetworkX fixes and compatible OpenGL
metadata. Follow the documented `--find-links third_party/wheels` option;
see [dependency details](../third_party/wheels/README.md).
