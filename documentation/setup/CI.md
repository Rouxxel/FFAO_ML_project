# Continuous integration

GitHub Actions workflows follow the patterns in the repository root guide. This project
is a **single Python package** at the repo root — no frontend/backend split.

## Layout

```text
.github/
  workflows/
    ci.yml          # lint, format, tests, wheel build, pip audit
    security.yml    # gitleaks secret scan
  dependabot.yml    # weekly pip + Actions updates
```

## What runs on every PR

| Job | Workflow | Purpose |
|-----|----------|---------|
| `quality` | `ci.yml` | `ruff check`, `ruff format --check`, `pytest` (excludes `slow`, `gpu`, `cfd`) |
| `package` | `ci.yml` | `python -m build` after quality passes |
| `supply-chain` | `ci.yml` | `pip audit` (informational until tightened), no `.env`, secret grep |
| `secrets` | `security.yml` | gitleaks |

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

### Dependency review (not used)

`actions/dependency-review-action` requires **Dependency graph** plus **GitHub
Advanced Security**, which many personal repos do not have. This project relies
on **`pip audit`** and **Dependabot** in `ci.yml` / `dependabot.yml` instead. To
add dependency review later, restore a job from [GitHub’s
docs](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/about-dependency-review)
after enabling security analysis.

### gitleaks `Resource not accessible by integration`

`security.yml` sets `pull-requests: read`. If a fork PR still fails, run gitleaks
locally or rely on the `supply-chain` grep step in `ci.yml`.

### Node 20 deprecation notice

GitHub may log that **Node 20 is deprecated** while an action’s bundle still targets
Node 20; the runner often uses **Node 24** anyway. That line is **informational**
(not a failed step). It should disappear as action authors publish Node-24-based
releases (e.g. newer `gitleaks-action` / `checkout` versions).

## Adding workflows later

|------|----------------------------------|
| Nightly CFD smoke | `schedule` + `pytest -m cfd` on `ubuntu-24.04` |
| ML training smoke | Separate job with `pip install -e ".[core,ml,dev]"`, CPU-only torch |
| Monorepo split | Not needed unless repo structure changes |

## Secrets

No production secrets in workflow YAML. Use GitHub **Secrets** for tokens if
tracking or deployment is added later (`track` extra / MLflow).
