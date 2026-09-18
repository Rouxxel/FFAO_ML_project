# Platforms and environments

## Python

- **Supported:** Python **3.11** or **3.12** (see `requires-python` in `pyproject.toml`).
- Use a project-local virtual environment (`.venv`).

## Operating systems

| Workload | Windows (native) | WSL2 / Linux |
|----------|-------------------|--------------|
| `ffaoml` core + custom finite-difference CFD | Supported (NumPy/SciPy stack) | Supported |
| PyTorch training (CPU or GPU) | Supported; install CUDA build from [pytorch.org](https://pytorch.org) when using GPU | Supported |
| **Dedalus** (`cfd-dedalus` extra) | Not recommended; use **WSL2 Ubuntu** | Preferred |
| **OpenFOAM** (future adapter) | Use WSL2 or Docker | Supported |

Primary CFD path for early development: **custom FD solver on Windows**; enable Dedalus on Linux/WSL2 when needed.

## Lockfiles and pins

- **Application dependencies:** optional groups in `pyproject.toml` (`core`, `ml`, `dev`, …) use minimum versions, not exact pins.
- **Reproducible environments:** generate a lockfile locally (`uv lock` / `pip freeze` / conda export) and store it as `requirements-lock.txt` or commit `uv.lock` only when the team agrees—do not put secrets in lockfiles.
- **PyTorch + CUDA:** pin the wheel index (CUDA version) in setup notes or lockfile, not in shared config with credentials.
- **Dedalus:** pin only after a working install recipe is recorded in this folder.

See also [TECH_STACK.md](../TECH_STACK.md) (version pinning policy).
