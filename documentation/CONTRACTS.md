# Interface contracts

Frozen conventions for CFD export, on-disk datasets, and ML tensors. Code
constants live in `src/ffaoml/contracts.py`.

---

## Field tensor layout (ML)

| Property | Convention |
|----------|------------|
| Shape | `(batch, channels, height, width)` |
| Channel order (default) | `velocity_x`, `velocity_y`, `pressure`, `vorticity` |
| Aliases | `u_x`, `u_y`, `p`, `ω` / `omega` map to the names above |

Coordinates: store grid spacing `dx`, `dy` and origin in per-simulation
metadata (not inside the tensor).

### Cylinder mask

Boolean array with shape `(height, width)` aligned with spatial dimensions:

| Value | Meaning |
|-------|---------|
| `True` | Solid cylinder cell; velocity should be zero; typically **excluded** from training losses |
| `False` | Fluid cell; included in field losses and divergence penalties |

---

## Simulation export (CFD)

Each timestep must provide at least:

- `velocity_x`, `velocity_y`
- `pressure`
- `vorticity`

Preferred on-disk representation: chunked **Zarr** stores with **xarray**
semantics (see [TECH_STACK.md](./TECH_STACK.md)).

**Code:** `ffaoml.data.io` (read/write), `ffaoml.data.catalog` (paths from config),
`ffaoml.data.loading` (`(T, C, H, W)` windows via `dataset.temporal_split`).

Directory layout (see [PRD.md](./PRD.md) §7):

```text
dataset/
├── simulations/
│   ├── re_050/
│   ├── re_100/
│   └── ...
└── metadata.csv
```

---

## `metadata.csv` columns

One row per simulation. Required columns:

| Column | Description |
|--------|-------------|
| `sim_id` | Unique id (e.g. `re_100_seed0`) |
| `re` | Reynolds number |
| `split` | `train`, `val`, or `test` (simulation-level split by Re) |
| `path` | Relative path to simulation store under `dataset/` |
| `u_inlet` | Inlet velocity scale U |
| `nu` | Kinematic viscosity ν |
| `diameter` | Cylinder diameter D |
| `nx` | Grid points in x |
| `ny` | Grid points in y |
| `dt` | Time step |
| `n_steps` | Number of stored timesteps |
| `seed` | RNG seed for the simulation |

Optional columns (e.g. solver name, git commit) may be appended later; document
them in `dataset/manifest.json` when introduced.

External imports must also record `stage`, `source_id`, and `source_url` — see
[DATA_SOURCES.md](./DATA_SOURCES.md). JSON schema and checkpoint layout:
[REPRODUCIBILITY.md](./REPRODUCIBILITY.md).

**Train/val/test:** assign by **Reynolds number**, not by random frames (PRD §11).

For **Stage 1** grid imports with a single trajectory, time blocks in
`dataset.temporal_split` override the `split` column for ML.

---

## Mesh graph layout (Stage 2 — unstructured)

Stage 2 uses **one graph per trajectory** (DeepMind MeshGraphNets `cylinder_flow`).
Splits are **trajectory-level** (`train` / `val` / `test` rows in `metadata.csv`),
not time blocks inside one simulation.

### Static mesh (time-invariant per trajectory)

| Array | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `mesh_pos` | `(N_nodes, 2)` | float | Node coordinates in the simulation plane |
| `node_type` | `(N_nodes,)` | int | Boundary / fluid typing (see below) |
| `cells` | `(N_faces, 3)` | int | Triangle vertex indices into `mesh_pos` |

**Edges for message passing:** derive **undirected** edges from triangle sides:
for each triangle `(i, j, k)`, add edges `(i,j)`, `(j,k)`, `(k,i)` and deduplicate.
Store `edge_index` as `(2, N_edges)` with PyTorch Geometric convention
(`edge_index[0]` = source, `edge_index[1]` = target) at load time; optional
precompute on disk in a later import version.

### Time-varying node features

| Array | Shape | Description |
|-------|-------|-------------|
| `velocity` | `(T, N_nodes, 2)` | Planar velocity components |
| `pressure` | `(T, N_nodes, 1)` or `(T, N_nodes)` | Pressure at nodes |

Default **one-step ML target:** predict next-step node features
`[velocity, pressure]` from the current step (configurable in `model.meshgraphnet`).

### `node_type` (MeshGraphNets convention)

| Value | Meaning |
|-------|---------|
| 0 | Normal / interior fluid |
| 1 | Obstacle (e.g. cylinder surface) |
| 2 | Airfoil |
| 3 | Handle |
| 4 | Inflow |
| 5 | Outflow |
| 6 | Wall |

Losses should **mask or down-weight** obstacle and boundary nodes when the task
is interior flow propagation; document the mask rule in the training config.

### On-disk layout (import target)

```text
dataset/meshgraphnets_data/
├── manifest.json
├── metadata.csv
└── trajectories/
    └── <sim_id>/
        ├── graph.zarr/     # mesh_pos, node_type, cells, velocity, pressure, time
        └── meta.json       # dt, n_steps, upstream split name
```

**Code (planned):** `ffaoml.data.mesh_io`, `ffaoml.ml.mesh_dataset`,
`ffaoml.models.meshgraphnet`. Constants: `ffaoml.contracts` (`MESH_*`).

### `metadata.csv` extensions (Stage 2 rows)

Required Stage 1 columns still apply where meaningful (`sim_id`, `split`, `path`,
`dt`, `n_steps`, `seed`). Mesh trajectories additionally use:

| Column | Description |
|--------|-------------|
| `n_nodes` | Node count |
| `n_cells` | Triangle count |
| `re` | Use `0` or sentinel if upstream has no single Re label |

---

## Checkpoint bundle (ML runs)

Each training run should persist:

- `model.pt`
- `config.yaml` (resolved configuration)
- `preprocess_stats.json`
- Reference to dataset manifest or `metadata.csv` hash

See [ARCHITECTURE.md](./ARCHITECTURE.md) §7.
