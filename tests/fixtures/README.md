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

## `cfdbench_mini/` and `cfdbench_upstream_case/`

Stage 3 multi-Re Zarr layout (three Reynolds numbers, 16×16 grid, 8 timesteps) and a
synthetic upstream CFDBench case folder (`u.npy`, `v.npy`, `case.json`) for probe tests.
No CFDBench download in CI.

Regenerate:

```bash
python tests/fixtures/build_cfdbench_mini.py
```
