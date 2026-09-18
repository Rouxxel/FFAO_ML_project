# Legal and licensing

Summary of how the **FFAO ML Project** is licensed and how to give credit. This is an overview only; the **LICENSE**, **LICENSE-DATA**, and **NOTICE** files are authoritative.

---

## Open source commitment

The project is intended to remain **open and reusable** for research, education, and derivative work, with **clear attribution** to the author and project.

---

## Two licenses (by material type)

| Material | License | File |
|----------|---------|------|
| **Software** — `src/`, `scripts/`, `tests/`, `configs/`, tooling | Apache License 2.0 | [LICENSE](../LICENSE) |
| **Research outputs** — CFD fields, datasets, checkpoints, plots, metrics, tables | CC BY 4.0 | [LICENSE-DATA](../LICENSE-DATA) |
| **Attribution text** — required notices for redistributions | (informational) | [NOTICE](../NOTICE) |

**Documentation** in `documentation/` is provided under **Apache 2.0** with the source code, unless a file explicitly states otherwise.

---

## Your obligations (short)

1. **Using code** — Follow Apache 2.0; preserve LICENSE and NOTICE; credit the project (see [ATTRIBUTION.md](./ATTRIBUTION.md)).  
2. **Using data or results** — Follow CC BY 4.0; credit Sebastian Russo and the FFAO ML Project; link to the repo or release.  
3. **Academic use** — Cite via [CITATION.cff](../CITATION.cff) and/or BibTeX in [ATTRIBUTION.md](./ATTRIBUTION.md).

---

## Relationship to project docs

| Document | Relevance |
|----------|-----------|
| [PRD.md](./PRD.md) | Defines datasets, experiments, and artifacts that may be shared (§7, §17) |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Where `dataset/` and `results/` live; checkpoint bundles |
| [TECH_STACK.md](./TECH_STACK.md) | Third-party libraries with separate licenses |

---

## Updates

Copyright years, maintainers, release URLs, and third-party notices should be updated in **NOTICE**, **LICENSE** (appendix), **CITATION.cff**, and **ATTRIBUTION.md** as the project grows. Review this index when tagging releases.
