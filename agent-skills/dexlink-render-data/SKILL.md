---
name: dexlink-render-data
description: >-
  Render saved DexLink reference trajectories or policy episodes as mesh videos without rerunning policy inference or physics. Use for separate dataset rendering and camera changes.
---

# Render saved DexLink motions

Read the rendering section of `docs/data-generation.md`. In the visualization
environment use `python -m dexlink render --task t047_a --output outputs/stir.mp4`
or `--reference PATH.npy` / `--eval-directory EVAL_DIR`, exactly one source.
Use a fresh `.mp4` path. `--episode`, camera mode, dimensions, stride and FPS
control the saved view; metadata is written beside it. The renderer accepts
native NPZ and scalar-dictionary NPY references and includes static fixtures.
An explicit `--reference-override` is only valid for a saved evaluation.
Rendering does not run a policy, prove physical success or need W&B. Preserve
the saved numeric dataset and source/hash metadata when changing the view.
