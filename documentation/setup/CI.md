# Continuous integration

GitHub Actions workflows follow the patterns in the repository root guide. This project
is a **single Python package** at the repo root — no frontend/backend split.

## Layout

```text
.github/
  workflows/
    ci.yml          # lint, format, tests, wheel build, pip audit
    security.yml    # gitleaks + dependency review on PRs
  dependabot.yml    # weekly pip + Actions updates
```

## What runs on every PR

| Job | Workflow | Purpose |
|-----|----------|---------|
| `quality` | `ci.yml` | `ruff check`, `ruff format --check`, `pytest` (excludes `slow`, `gpu`, `cfd`) |
| `package` | `ci.yml` | `python -m build` after quality passes |
| `supply-chain` | `ci.yml` | `pip audit` (informational until tightened), no `.env`, secret grep |
| `secrets` | `security.yml` | gitleaks |
| `dependency-review` | `security.yml` | Optional; needs Dependency graph + GHAS (non-blocking) |

Install set in CI: `pip install -e ".[core,dev]"` — **not** `ml` or `cfd-dedalus`
(heavy / platform-specific). Local ML/CFD installs use extras from `pyproject.toml`.

## Pytest markers

| Marker | CI default | Where to run |
|--------|------------|----------------|
| (none) | Yes | Unit tests on every PR |
| `slow` | No | CFD generation, long rollouts — `workflow_dispatch` or nightly (future) |
| `gpu` | No | CUDA training — self-hosted or manual |
| `cfd` | No | Dedalus/OpenFOAM — Linux/WSL runners (future job) |

Example local full suite:

```bash
pytest
pytest -m slow    # when CFD exists
```

## Branch protection

On GitHub: **Settings → Branches →** require checks such as:

- `CI / quality`
- `CI / package`
- `Security / secrets`

Optional: require `supply-chain` once `pip audit` runs with `continue-on-error: false`.

## Troubleshooting

### `Could not find 'dataset/stage1_zenodo'`

Commit the Hydra dataset group files:

- `configs/dataset/stage1_zenodo.yaml`
- `configs/dataset/splits.yaml`

### Dependency review unsupported

On personal repos without GitHub Advanced Security, the `dependency-review` job
logs a message and exits non-zero but is marked **`continue-on-error: true`** so
it does not fail the workflow. Enable [Dependency graph](https://github.com/Rouxxel/FFAO_ML_project/settings/security_analysis)
if you add GHAS later.

### gitleaks `Resource not accessible by integration`

`security.yml` sets `pull-requests: read`. If a fork PR still fails, run gitleaks
locally or rely on the `supply-chain` grep step in `ci.yml`.

### Node 20 deprecation notice

GitHub Actions may print a notice while third-party actions move to Node 24; it is
informational and does not fail the build.

## Adding workflows later

|------|----------------------------------|
| Nightly CFD smoke | `schedule` + `pytest -m cfd` on `ubuntu-24.04` |
| ML training smoke | Separate job with `pip install -e ".[core,ml,dev]"`, CPU-only torch |
| Monorepo split | Not needed unless repo structure changes |

## Secrets

No production secrets in workflow YAML. Use GitHub **Secrets** for tokens if
tracking or deployment is added later (`track` extra / MLflow).
