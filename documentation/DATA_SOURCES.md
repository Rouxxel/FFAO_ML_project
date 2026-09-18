# External dataset sources (staged)

Canonical URLs and how each stage fits the project. **Active work uses Stage 1 only**
(small download). Stage 2 and Stage 3 are deferred until storage and prior stages
are complete.

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
| **1** | End-to-end pipeline; temporal evolution at **Re ≈ 100** | ~1 MB | **Active** |
| **2** | Unstructured mesh; MeshGraphNet-style models | Multi-GB TFRecords | Deferred |
| **3** | Generalization across physical conditions (CFDBench) | ~13.4 GB interpolated | Deferred |

**Stage 1 first research question:** Can a model learn the **temporal evolution**
of vortex shedding (e.g. flow(t) → flow(t+Δt), rolled out to t+50)?

**Later research question (Stage 3 / multi-Re own CFD):** How does error grow when
evaluating **outside** the training distribution of physical conditions?

---

## Stage 1 — Small Re = 100 prototype (active)

Classic 2D cylinder wake at **Re = 100**; ~**151** time snapshots; high spatial
resolution (e.g. 449×199); **vorticity** (Kármán shedding). Suitable for CNN /
ConvLSTM on regular grids after optional downsampling.

| Resource | URL |
|----------|-----|
| Zenodo dataset (physics-constrained autoencoders release) | https://zenodo.org/records/18669296 |
| Related code (Physics-Constrained Convolutional Autoencoders) | https://github.com/LucaAddiucci/Physics-Constrained-Convolutional-Autoencoders |

**Project config:** `configs/dataset/stage1_zenodo.yaml`  
**On-disk target:** `dataset/simulations/re_100/` (+ `metadata.csv` row, manifest `stage: 1`).

**Splits:** Use **time-based** train/val/test within the trajectory — not the
multi-Re lists in `configs/dataset/splits.yaml` (those apply to Stage 3 / own
multi-Re CFD).

**Attribution:** Cite Zenodo record and upstream authors when publishing figures
or derivatives; respect upstream license terms in addition to this repo’s
[LICENSE-DATA](../LICENSE-DATA).

---

## Stage 2 — DeepMind MeshGraphNets `cylinder_flow` (deferred)

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

---

## Stage 3 — CFDBench generalization (deferred)

Regular **64×64** interpolated grids; cylinder case; varying boundary conditions,
geometry, and physical properties. Intended for **condition generalization**
experiments (e.g. train on Re ∈ {100,200,300,400}, test on unseen Re).

| Resource | URL |
|----------|-----|
| CFDBench GitHub | https://github.com/luo-yining/CFDBench |
| CFDBench on Hugging Face | https://huggingface.co/datasets/luoyining/CFDBench |

**Storage:** ~**13.4 GB** interpolated subset (do **not** download raw ~460 GB for
this project). Use `configs/dataset/splits.yaml` Re-style splits when this stage
is enabled.

---

## Own CFD simulations (parallel / long-term)

In-house **finite-difference** or **Dedalus** solvers remain valuable for:

* Full control of Re sweeps aligned with [PRD.md](./PRD.md) §11
* Matching export channels (`u_x`, `u_y`, `p`, `ω`) exactly
* Publishing new simulation data under [LICENSE-DATA](../LICENSE-DATA)

Own-CFD generation is **not** required to complete Stage 1. See CFD/data implementation
checklist for solver vs import tracks.

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
