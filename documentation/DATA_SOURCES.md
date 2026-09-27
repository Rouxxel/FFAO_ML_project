# External dataset sources (staged)

Canonical URLs and how each stage fits the project. **Stage 1** and **Stage 2**
have import/train paths; **Stage 3** (CFDBench) is **planned** — config and contracts
land first; import adapter follows.

Import adapters should normalize all sources into the repository layout described
in [CONTRACTS.md](./CONTRACTS.md) and [ARCHITECTURE.md](./ARCHITECTURE.md) §5.1,
with `dataset/manifest.json` recording `stage`, `source_id`, and download provenance.

---

## Scientific progression

```text
                    PHYSICS
                       │
                       ▼
         Cylinder flow (data or simulation)
                       │
                       ▼
                Normalized dataset
                       │
                       ▼
            ┌──────────────────────┐
            │  ML flow predictor   │
            └──────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
   Short-horizon accuracy      Long-horizon stability
         │                           │
         └─────────────┬─────────────┘
                       ▼
         Physical quantities (ω, Cd, Cl, St when available)
                       │
                       ▼
   (Later) Generalization across Re / BC / geometry — Stage 3 or own CFD
```

| Stage | Purpose | Storage (order of magnitude) | Status |
|-------|---------|------------------------------|--------|
| **1** | End-to-end pipeline; temporal evolution at **Re ≈ 100** | ~1 MB | **Active** — import + validation implemented |
| **2** | Unstructured mesh; MeshGraphNet-style models | Multi-GB TFRecords | **Active** — import + train (see § Stage 2) |
| **3** | Generalization across physical conditions (CFDBench) | ~13.4 GB interpolated | **Planned** — `dataset=stage3_cfdbench` (import adapter next) |

**Stage 1 first research question:** Can a model learn the **temporal evolution**
of vortex shedding (e.g. flow(t) → flow(t+Δt), rolled out to t+50)?

**Later research question (Stage 3 / multi-Re own CFD):** How does error grow when
evaluating **outside** the training distribution of physical conditions?

---

## Stage 1 — Small Re = 100 prototype (active)

Classic 2D cylinder wake at **Re = 100**. The Zenodo PO-CAE release documents
**`cylinder_re100_grid64_last100.h5`**: ~**100** time snapshots on a **64×64**
structured grid. The canonical HDF5 layout is:

| Dataset | Shape / role |
|---------|----------------|
| `fields` | `(N_t, 3, 64, 64)` — channel 0 = **u**, 1 = **v**, 2 = vorticity (ω) |
| `grid_x`, `grid_y` | 1D axes for the structured grid |

The importer also accepts separate `u`, `v`, and optional `vorticity` datasets
`(N_t, 64, 64)` and maps them into the same pipeline.

**Zenodo gap:** record [18669296](https://zenodo.org/records/18669296) may list
**training scripts only**, not the HDF5. **Pipeline fallback:** `python main.py --run`
tries Zenodo/cache and imports to **`dataset/zenodo_data/`**; on failure it runs
LBM (~1–3 min CPU) and imports to **`dataset/generated_data/`**. Force generation
with `python main.py --run --generate-data` or `--only generate`. Cite Addiucci
(2026) for the benchmark definition (see [ATTRIBUTION.md](./ATTRIBUTION.md)).

**Import adapter:** `src/ffaoml/data/sources/zenodo_re100.py`

**How to import (pick one):**

| Entry | Command |
|-------|---------|
| Generate HDF5 (fallback) | `python scripts/generate_stage1_cylinder_h5.py` |
| Import to `dataset/` | `python scripts/download_stage1_zenodo.py` |
| Pipeline | `python main.py --run --only import` |
| Pipeline + local HDF5 | `python main.py --run --local-file path/to/cylinder_re100_grid64_last100.h5` |

Hydra dataset config: `configs/dataset/stage1_zenodo.yaml`. After import, CFD
validation: `python scripts/validate_stage1_zenodo.py` or
`python main.py --run --from cfd_validation` (with import already done).

| Resource | URL |
|----------|-----|
| Zenodo dataset (physics-constrained autoencoders release) | https://zenodo.org/records/18669296 |
| DOI | https://doi.org/10.5281/zenodo.18669296 |
| Related code (Physics-Constrained Convolutional Autoencoders) | https://github.com/LucaAddiucci/Physics-Constrained-Convolutional-Autoencoders |

**Rights (upstream deposit):** Creative Commons Attribution 4.0 International
(CC BY 4.0). **Copyright © 2026 Luca Addiucci.**

**Citation:**

> Addiucci, L. (2026). *Physics-Constrained Convolutional Autoencoders for 2D
> Cylinder Flow at Re=100* (Version 1.0) [Dataset]. Zenodo.
> https://doi.org/10.5281/zenodo.18669296

Full BibTeX and credit lines: [ATTRIBUTION.md](./ATTRIBUTION.md#stage-1--zenodo-re--100-flow-fields).

**Project config:** `configs/dataset/stage1_zenodo.yaml`  
**On-disk target:** `dataset/simulations/re_100_zenodo/` (+ `metadata.csv` row,
manifest `stage: 1`). Default cache:
`.cache/zenodo_stage1/cylinder_re100_grid64_last100.h5` (from the generator or a
manual copy).

**Splits:** Use **time-based** train/val/test within the trajectory — not the
multi-Re lists in `configs/dataset/splits.yaml` (those apply to Stage 3 / own
multi-Re CFD). ML Re-conditioning (`model=cnn_re`, `dataset=splits`) needs
**multiple** `metadata.csv` rows (own CFD or future Stage 3); Stage 1 alone is
single-Re temporal learning only.

**Attribution:** Cite the DOI and Addiucci (2026) when publishing figures or
derivatives; respect **CC BY 4.0** and the upstream copyright in addition to this
repo’s [LICENSE-DATA](../LICENSE-DATA).

---

## Stage 2 — DeepMind MeshGraphNets `cylinder_flow`

Unstructured triangular mesh; node features (`mesh_pos`, `cells`, `node_type`,
`velocity`, `pressure`); **600** steps, **dt = 0.01**. Enables GNN / MeshGraphNet
research — not a broad Reynolds-number benchmark.

| Resource | URL |
|----------|-----|
| Official MeshGraphNets (DeepMind Research) | https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets |
| Training TFRecord | https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord |
| Validation TFRecord | https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/valid.tfrecord |
| Test TFRecord | https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord |
| Hugging Face mirror (`cylinder_flow`) | https://huggingface.co/datasets/OneScience-Group/cylinder_flow |
| PyTorch MeshGraphNets reference | https://github.com/echowve/meshGraphNets_pytorch |

**Reference implementation / tutorial:** NVIDIA PhysicsNeMo MeshGraphNet cylinder-flow
example (download + training walkthrough).

**Project note:** Requires mesh graph data pipeline and GNN models — planned after
Stage 1 grid-based pipeline is stable.

**Hydra (config skeleton):** `configs/dataset/stage2_meshgraphnets.yaml`,
`configs/model/meshgraphnet.yaml`; mesh tensor layout in [CONTRACTS.md](./CONTRACTS.md).

**Import (local, requires TensorFlow):**

```bash
python scripts/download_stage2_meshgraphnets.py --split train --max-trajectories 1
python scripts/inspect_meshgraphnets_tfrecord.py .cache/meshgraphnets_cylinder/train.tfrecord
python scripts/validate_stage2_meshgraphnets.py
```

**Train (CPU smoke by default; use `--device cuda` for the full train shard):**

```bash
python scripts/train_meshgraphnet.py --run-id stage2_meshgn
python scripts/evaluate_mesh_model.py --run-dir results/runs/stage2_meshgn
python scripts/run_stage2_pipeline.py --run --max-trajectories 2
```

---

## Stage 3 — CFDBench generalization (planned)

Regular **64×64** interpolated grids; cylinder case; varying boundary conditions,
geometry, and physical properties. Intended for **condition generalization**
experiments (e.g. train on Re ∈ {100,200,300,400}, test on unseen Re).

| Resource | URL |
|----------|-----|
| CFDBench GitHub | https://github.com/luo-yining/CFDBench |
| CFDBench on Hugging Face | https://huggingface.co/datasets/luoyining/CFDBench |

**Storage:** ~**13.4 GB** interpolated subset (do **not** download raw ~460 GB for
this project).

**Hydra (config skeleton):** `configs/dataset/stage3_cfdbench.yaml`,
`configs/model/cnn_re.yaml`; multi-Re grid layout in [CONTRACTS.md](./CONTRACTS.md).
Use `dataset=splits` for the same Re lists with stub/own-CFD data during development.

---

## Own CFD simulations (parallel / long-term)

In-house **finite-difference** or **Dedalus** solvers remain valuable for:

* Full control of Re sweeps aligned with [PRD.md](./PRD.md) §11
* Matching export channels (`u_x`, `u_y`, `p`, `ω`) exactly
* Publishing new simulation data under [LICENSE-DATA](../LICENSE-DATA)

Own-CFD generation is **not** required to complete Stage 1. Solver modules under
`src/ffaoml/cfd/` are mostly **placeholders** today (`stub` writes contract-shaped
Zarr for tests; `fd` / `dedalus` / OpenFOAM raise or defer). See the CFD/data
task checklist in the repo root for Track B vs import adapters.

**CI note:** GitHub Actions does **not** download Zenodo or CFDBench; tests use
`tests/fixtures/` and small synthetic imports.

---

## Manifest fields (all stages)

Implementation: `ffaoml.manifests` — full notes in [REPRODUCIBILITY.md](./REPRODUCIBILITY.md).

`dataset/manifest.json` should include at minimum:

| Field | Example |
|-------|---------|
| `schema_version` | `1` |
| `stage` | `1`, `2`, or `3` |
| `source_id` | `zenodo_re100`, `meshgraphnets_cylinder_flow`, `cfdbench_cylinder` |
| `source_url` | Zenodo or dataset homepage |
| `created_at` | ISO timestamp |
| `git_commit` | Short SHA at import time |
| `config_hash` | Hash of Hydra config used for import |
