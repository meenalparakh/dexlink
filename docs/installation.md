# Installation

Clone the public source and run the commands from its root:

```bash
git clone https://github.com/meenalparakh/dexlink.git
cd dexlink
```

Use Python 3.10. GPU policy learning and rollout use an NVIDIA CUDA GPU. Create
separate environments for RL and retargeting to keep their dependencies clear:

```bash
conda create -n genesis python=3.10
conda activate genesis
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e '.[rl,visualization,downloads,test]' -c environments/genesis-constraints.txt --find-links third_party/wheels

conda create -n hpart python=3.10
conda activate hpart
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e '.[retargeting,visualization,test]' -c environments/hpart-constraints.txt --find-links third_party/wheels
```

The `--find-links third_party/wheels` option is required: the repository
includes small, attributed URDFpy/Pyrender wheels matching the tested parser
source and resolving their outdated NetworkX/NumPy/OpenGL dependencies.
Their exact patches, upstream hashes, licenses and rebuild command are in
[the wheelhouse guide](../third_party/wheels/README.md).

The RL extra installs the public [Genesis fork](https://github.com/meenalparakh/Genesis)
at `65883d3e9ad1c6704dc5217be742ba63e0275a73`. This commit is required by the
tested runtime, including its solver/contact diagnostics. Do not substitute a
PyPI release or a newer commit without validation. Updating this pin is an
explicit item in the root README release checklist.

Install a compatible [PyTorch3D source build](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
against the installed Torch/CUDA in each environment that imports it. The
tested build uses PyTorch3D 0.7.9. Retargeting also uses MANO and
SMPL-X; obtain the licensed models separately using the dataset guide.

The constraints record the tested dependency versions, including Torch 2.6.0,
NumPy 2.2.6, SciPy 1.15.3 and the URDFpy/Pyrender wheels described above.
Headless rendering uses EGL:

```bash
export PYOPENGL_PLATFORM=egl
export DEXLINK_OUTPUTS=/path/to/persistent/dexlink-outputs
export DEXLINK_EXTERNAL_DATA=/path/to/dexlink-external-data
python -m dexlink list
```

Keep the checkout and large outputs on adequate persistent storage. Install the
[runtime assets/checkpoints](resources.md) next. Use `pip install -e . --no-deps`
only when reusing an already configured compatible environment.

Training and strict PPO smoke require a W&B account and a writable entity/project:

```bash
conda activate genesis
wandb login
python -m dexlink smoke invert_cup --run-directory outputs/runs/invert-smoke \
  --control-steps 64 --num-envs 32 --entity YOUR_ENTITY --project dexlink
```

The wrapper verifies online logging and saves the actual run URL. Keep API keys
out of Git and manifests. Default reference generation, pretrained rollout,
collection and offline mesh rendering do not require W&B authentication.
