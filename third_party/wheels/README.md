# Runtime dependency wheels

Use `--find-links third_party/wheels` in the installation commands. These
small pure-Python wheels preserve the tested runtime while resolving concrete
upstream installation problems.

| Package | Upstream source | Changes |
|---|---|---|
| `urdfpy==0.0.22+dexlink.1` | [mmatl/urdfpy](https://github.com/mmatl/urdfpy/tree/5466842899b33bd549e8f9e2a9a987bd5e37373b) | Pin tested NetworkX 3.3 instead of Python-incompatible 2.2; replace the removed `np.float` alias with `float` in material colors; declare runtime requirements only at runtime |
| `pyrender==0.1.45+dexlink.1` | [Pyrender 0.1.45](https://pypi.org/project/pyrender/0.1.45/) | Relax the old PyOpenGL 3.1.0 pin to >=3.1.4, satisfying the pinned Genesis dependency; renderer source is unchanged |

Both upstream packages are MIT licensed. Their notices/licenses are preserved
in the wheels and in `third_party_licenses/URDFPY-MIT.txt` and
`third_party_licenses/PYRENDER-MIT.txt`. `SHA256SUMS` records the shipped wheels.
The local version suffix identifies these explicit patches.

A maintainer can rebuild these wheels from checksum-pinned source with Python 3.10,
`setuptools==80.9.0` and `wheel==0.45.1`:

```bash
python tools/build_runtime_wheels.py --output-directory outputs/rebuilt-wheels
```

The script checks upstream hashes and patch locations, uses a fixed build
timestamp, and refuses to replace different output wheels. Do not replace
these files with unmodified PyPI distributions under the same local version.
