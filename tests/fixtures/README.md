# Test fixtures

## `stage1_mini.h5`

Tiny synthetic upstream file (6 time steps, 10×12 grid) matching the Addiucci HDF5
layout (`fields`, `grid_x`, `grid_y`). Used by CI for import and loading tests without
downloading Zenodo data.

Regenerate:

```bash
python -c "from pathlib import Path; import runpy; runpy.run_path('tests/fixtures/build_stage1_mini.py')"
```

Or run `tests/fixtures/build_stage1_mini.py` if present.

## `meshgraphnets_mini/`

Synthetic mesh trajectory (24 nodes, 8 timesteps) for Stage 2 mesh I/O tests without
downloading MeshGraphNets TFRecords.

Regenerate:

```bash
python tests/fixtures/build_meshgraphnets_mini.py
```
