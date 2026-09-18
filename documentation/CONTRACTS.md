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

---

## Checkpoint bundle (ML runs)

Each training run should persist:

- `model.pt`
- `config.yaml` (resolved configuration)
- `preprocess_stats.json`
- Reference to dataset manifest or `metadata.csv` hash

See [ARCHITECTURE.md](./ARCHITECTURE.md) §7.
