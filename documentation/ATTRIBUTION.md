# Attribution and credit

This project is **open source**. You may use the **code**, **documentation**, **datasets**, and **published results**, provided you **give appropriate credit** to the author and project as described below.

Legal terms are defined in:

| Artifact | License file |
|----------|----------------|
| Source code, configs, tests | [LICENSE](../LICENSE) (Apache 2.0) |
| Data, simulations, checkpoints, figures, metrics | [LICENSE-DATA](../LICENSE-DATA) (CC BY 4.0) |
| Redistribution requirements | [NOTICE](../NOTICE) |

When in doubt, **attribute both** the software and the data/results you used.

---

## Who to credit

- **Author:** Sebastian Russo  
- **Project:** FFAO ML Project (*Fluid Flow Around an Obstacle* — ML for 2D cylinder flow)  
- **Repository:** https://github.com/Rouxxel/FFAO_ML_project  

Update this document if maintainers or canonical URLs change (e.g. after a rename or org transfer).

---

## Recommended credit lines

### Software (code, configs, training scripts)

Use in README, documentation, “About”, or license screens:

```text
Based on the FFAO ML Project (Fluid Flow Around an Obstacle) by Sebastian Russo
(https://github.com/Rouxxel/FFAO_ML_project), licensed under Apache-2.0.
```

If you redistribute source, keep the [LICENSE](../LICENSE) and [NOTICE](../NOTICE) files and comply with Apache 2.0 section 4.

### Data, model weights, figures, and numerical results

Use in papers, posters, slides, datasets, and model cards:

```text
Data [or: Model / Figures / Results] from the FFAO ML Project (Fluid Flow Around an Obstacle)
by Sebastian Russo (https://github.com/Rouxxel/FFAO_ML_project), licensed under CC BY 4.0.
```

Link to the **specific release, commit, or dataset manifest** when possible (see [PRD.md](./PRD.md) §17 reproducibility).

---

## Academic citation

Prefer the machine-readable citation in [CITATION.cff](../CITATION.cff) (GitHub “Cite this repository”).

**BibTeX (update version/year when you tag releases):**

```bibtex
@software{ffao_ml_project,
  author    = {Russo, Sebastian},
  title     = {{FFAO ML Project}: Machine Learning for Fluid Flow Around an Obstacle},
  year      = {2026},
  url       = {https://github.com/Rouxxel/FFAO_ML_project},
  license   = {Apache-2.0},
  note      = {Data and published results under CC BY 4.0; see LICENSE-DATA}
}
```

For a formal publication by the author, add a `@article` entry here and in `CITATION.cff` when available.

---

## Examples by use case

| Use case | What to do |
|----------|------------|
| Fork or copy code into another repo | Keep LICENSE + NOTICE; state modification; link to upstream |
| Import as a Python dependency / submodule | Credit in README and docs; retain notices in distributions |
| Train a new model starting from project checkpoints | Credit project + CC BY for weights; describe your changes |
| Publish plots from `results/` | Credit in figure caption; link to repo or release |
| Ship a subset of Zarr/NetCDF simulations | Credit + CC BY notice; include link and license text |
| Compare your method to ours in a paper | Cite repository (and release version); do not imply endorsement |
| Use only ideas/algorithms (no copy of code or data) | Citation is appreciated but not a license requirement |

---

## What this project publishes (for clarity)

Aligned with [PRD.md](./PRD.md) and [ARCHITECTURE.md](./ARCHITECTURE.md):

- **Code** — physics utilities, CFD adapters, datasets, models, training, evaluation  
- **Documentation** — PRD, architecture, tech stack, this file  
- **Data** — time-resolved flow fields, metadata, manifests under `dataset/`  
- **Results** — metrics, figures, animations, checkpoints under `results/`  

Third-party **dependencies** (PyTorch, Dedalus, etc.) remain under their own licenses; see [NOTICE](../NOTICE).

---

## Maintainer checklist (when the project evolves)

Update as needed when you:

- [ ] Tag a release → set version in `CITATION.cff` and optional Zenodo DOI  
- [ ] Add co-maintainers → copyright line in LICENSE appendix, NOTICE, this file  
- [ ] Publish external datasets → note URL and license in `metadata.csv` / manifest  
- [ ] Bundle vendored code → add section to NOTICE  
- [ ] Change default license for a subtree → state it in that directory’s README or LICENSE file  

---

## Related documents

- [LEGAL.md](./LEGAL.md) — index of legal and attribution files  
- [PRD.md](./PRD.md) — project scope and reproducibility  
- [ARCHITECTURE.md](./ARCHITECTURE.md) — artifact locations (`dataset/`, `results/`)
