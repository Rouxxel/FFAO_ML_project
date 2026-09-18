# Stage 1 Zenodo import notes

## Command

```bash
pip install -e ".[core,dev]"
python scripts/download_stage1_zenodo.py
```

If Zenodo record [18669296](https://zenodo.org/records/18669296) does not yet list the
HDF5 asset, place `cylinder_re100_grid64_last100.h5` in `.cache/zenodo_stage1/` or pass
`--local-file /path/to/cylinder_re100_grid64_last100.h5`.

Optional: set `dataset.import.upstream_data_url` in `configs/dataset/stage1_zenodo.yaml`
to a direct HTTPS link.

## Upstream size

The documented HDF5 is on the order of **~1 MB** (64×64 grid, ~100 time steps, float32
fields). The Zenodo deposit may currently ship **code only**; the README in that record
names the HDF5 file expected alongside the training scripts.

## Attribution

- Zenodo: Addiucci, L. (2026). *Physics-Constrained Convolutional Autoencoders for 2D
  Cylinder Flow at Re=100*. https://zenodo.org/records/18669296 (CC BY 4.0).
- Related code: https://github.com/LucaAddiucci/Physics-Constrained-Convolutional-Autoencoders

See also `documentation/ATTRIBUTION.md` and `LICENSE-DATA`.

## Outputs

| Path | Role |
|------|------|
| `dataset/simulations/re_100_zenodo/fields.zarr` | Time-chunked CONTRACT channels |
| `dataset/metadata.csv` | Single row, `split=full` (ML uses `temporal_split` in config) |
| `dataset/manifest.json` | `stage: 1`, `source_id: zenodo_re100`, provenance hashes |

## Validation (Phase 3)

After import:

```bash
python scripts/validate_stage1_zenodo.py
```

Writes `results/cfd_validation/stage1_zenodo/` (`SUMMARY.md`, figures, GIF, `metrics.json`).
Cd/Cl plots are documented as deferred when forces are not in the dataset (PRD §8).
