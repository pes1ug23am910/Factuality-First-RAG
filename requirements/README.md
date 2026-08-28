# Target dependency locks

These repository-only locks describe the target Windows x86-64, CPython
3.10.11, CUDA 12.6 research environment. They are intentionally excluded from
the platform-neutral source distribution.

- `locks/windows-cp310-cu126-build.txt` authenticates the installer, the
  `build==1.5.0` front-end, and the complete build bootstrap. Install it first.
- `locks/windows-cp310-cu126.txt` authenticates the runtime, quantization, and
  offline-verification dependency graph.
- Java 21 remains an external Pyserini prerequisite and is not represented by
  Python package metadata.

The runtime lock binds Torch directly to the official Windows CPython 3.10 CUDA
12.6 wheel and hash. It never exposes unrelated packages to a PyTorch secondary
index. Pyserini 1.2.0 is source-only, so the bootstrap must be installed first
and the runtime lock installed with `--no-build-isolation`:

```powershell
python -m pip install --require-hashes -r requirements/locks/windows-cp310-cu126-build.txt
python -m pip install --require-hashes --no-build-isolation -r requirements/locks/windows-cp310-cu126.txt
python -m build --no-isolation
```

Use a new virtual environment without `--system-site-packages`, editable
installs, or dependency-borrowing `.pth` files. Install the audited project
wheel or sdist afterward with `--no-deps`, then run `pip check` and the
hostile-working-directory resource/CLI smokes.

The checked-in `.in` files record the direct roots used for the target
resolution. Regeneration must use `uv==0.12.5`, the exact target tuple above,
hash generation, the PyPI primary index, and no local paths or annotations.
The final runtime prologue must retain `--only-binary :all:` before the sole
`--no-binary pyserini` exception because pip processes those options in order.
