# Public port changes

- `dexlink/workflows/`: task catalog/validation, typed training and rollout
  wrappers, finite PPO auditing, numerical demonstration collection, separate
  rendering, resource installation and multi-task preparation. Entry points:
  `cli.main`, `training.train_task`, `data.collect`, `data.render`,
  `resources.install` and `multitask.prepare`.
- `dexlink/src/rl/library/`: native task environment, full-method task preset
  and collision-capacity checks. `multi_object.env_factory.create_env` now
  constructs this environment while retaining public dual-active support.
- `dexlink/src/rl/agent/ppo.py`: `train` supports stopping after a complete
  update and returns its actual positive checkpoint timestep;
  `_load_checkpoint_dir` enforces trained observation state for scene clouds.
- `dexlink/src/rl/agent/agent_utils.py`: native trajectory-weighted policies
  accept derived multi-object environments; observation processors preserve
  and validate their semantic checkpoint contracts.
- Shared RL config, observation/reset/task/physics and object modules:
  internal task settings, metrics, passive mechanisms and scene-cloud support;
  strict public CLI parsing and portable overheat handling are retained.
- `dexlink/scripts/rl/eval_final_metrics.py`: `parse_args`,
  `_collect_rollout_chunk` and `main` add pre-step policy observation/action
  recording for demonstration generation.
- `dexlink/visualizations/rl/`: native NPZ and NPY reference loading,
  saved-episode rendering, static fixtures, tracked cameras and source metadata.
  Removed two obsolete wrappers requiring the excluded paper renderer.
- `dexlink/multitask_cup/`: dependency closure of the conditional full-method
  policy, learner, task bank, synchronized checkpoints and evaluation. Private
  campaign launch infrastructure and result assembly are excluded.
- `dexlink/src/reference/arctic/geometry.py`: extracted reusable support-box
  helper fixes the public ARCTIC adapter. `category_mesh_inputs.py` provides
  the mesh utility's missing job/input helpers without the legacy launcher.
- `tasks/`: 40 frozen native references/goals and three tutorial references/
  cached templates, catalog checkpoint hashes and 30 cup programs, alongside
  the existing portable YAML recipes. Numeric arrays are preserved.
- `assets/`: remaining small project-authored procedural variants, with
  provenance. Runtime assets, SAM3D targets and alignment metadata are supplied
  in the separate verified ZIP rather than committed datasets/checkpoints.
- `AGENTS.md`, `agent-skills/`, `docs/`, `external_data/README.md` and root
  `README.md`: ten workflow skills, source-provider instructions, public
  resource links, install constraints and release checklist. HOT3D was left out.
- `cluster/`: four-hour invert-cup learning and policy-data collection examples
  with explicit site account/resources, persistent paths and clean-smoke checks.
- `tests/`: public portability/provenance checks, data timing/episode bounds,
  multi-task learner/policy regression tests and archive installation protections.

See [validation](validation.md) for completed checks and remaining release gates.
The original source-of-truth and internal checkouts were left unchanged.

The final release also includes the public asset/checksum/checkpoint URLs,
small attributed URDFpy/Pyrender dependency wheels with rebuild instructions,
and per-update logging for short numerical PPO smoke runs.
