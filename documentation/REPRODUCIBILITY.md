# Reproducibility and manifests

Conventions for dataset provenance and ML run artifacts. Implemented in
`src/ffaoml/manifests.py`.

---

## Software vs published artifacts

| Material | License | Where declared |
|----------|---------|----------------|
| Source code in this repository | Apache-2.0 | `LICENSE`, `pyproject.toml` |
| Downloaded or generated datasets, checkpoints, figures | CC BY-4.0 | `LICENSE-DATA`, `DatasetManifest.license_spdx` |

When you publish results, cite upstream data (e.g. Zenodo Stage 1) in addition to
this project — see [ATTRIBUTION.md](./ATTRIBUTION.md) and [DATA_SOURCES.md](./DATA_SOURCES.md).

---

## Dataset manifest (`dataset/manifest.json`)

Written when data is **imported or generated** (not committed to git; `dataset/` is
ignored). Minimum fields:

| Field | Description |
|-------|-------------|
| `schema_version` | Currently `1` |
| `stage` | `1`, `2`, or `3` (see DATA_SOURCES) |
| `source_id` | e.g. `zenodo_re100` |
| `source_url` | Canonical download or project page |
| `created_at` | UTC ISO-8601 timestamp |
| `git_commit` | Short SHA at import time (optional) |
| `config_hash` | Hash of resolved Hydra config used for import |
| `license_spdx` | Default `CC-BY-4.0` for artifacts you redistribute |

Example (factory in code):

```python
from ffaoml.manifests import build_dataset_manifest, dataset_manifest_path, hash_config
from hydra import compose, initialize_config_dir
from ffaoml.config import config_dir

# ... compose cfg ...
manifest = build_dataset_manifest(
    stage=1,
    source_id="zenodo_re100",
    source_url="https://zenodo.org/records/18669296",
    config_hash=hash_config(cfg),
)
manifest.write_json(dataset_manifest_path("dataset"))
```

---

## Training checkpoint bundle (`results/runs/<run_id>/`)

Each training run should persist:

| File | Role |
|------|------|
| `model.pt` | Weights |
| `config.yaml` | Resolved training configuration |
| `preprocess_stats.json` | Normalization fit on train split only |
| `dataset_manifest.json` | Copy or snapshot reference of the dataset manifest used |

Optional metadata file: `bundle_manifest.json` (`CheckpointBundleManifest`) with
`dataset_manifest_hash`, `config_hash`, and `git_commit`.

Validate before publishing a run:

```python
from ffaoml.manifests import validate_checkpoint_bundle

missing = validate_checkpoint_bundle("results/runs/my_run")
```

---

## Versioning and citations

- Bump `version` in `pyproject.toml` and [CITATION.cff](../CITATION.cff) together when
  tagging releases (`foundation-v0.1`, `data-stage1-v0.1`, etc.).
- Record `git_commit` and `config_hash` in manifests so papers can reference exact
  code + config (PRD §17).

---

## Related documents

- [CONTRACTS.md](./CONTRACTS.md) — `metadata.csv`, tensor layout
- [DATA_SOURCES.md](./DATA_SOURCES.md) — staged external data
- [ARCHITECTURE.md](./ARCHITECTURE.md) §5–7 — data flow and checkpoint bundle
