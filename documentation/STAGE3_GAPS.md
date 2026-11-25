# Stage 3 — remaining gaps and follow-ups

Stage 3 implementation is **feature-complete for v1** (import → validate → train → eval,
orchestration, CI fixtures). This file replaces the checklist role of `STAGE3_TASKS.md`
after that plan is removed. Canonical runbooks:

| Topic | Document |
|-------|----------|
| Commands, PRD experiments 1–3 | [experiments/stage3_re_generalization.md](../experiments/stage3_re_generalization.md) |
| Data URLs, layout, status | [DATA_SOURCES.md](./DATA_SOURCES.md) § Stage 3 |
| Expected artifacts | [EXPECTED_RESULTS.md](./EXPECTED_RESULTS.md) § Stage 3 |
| Local release + git tags | [STAGE3_RELEASE.md](./STAGE3_RELEASE.md) |
| Grid / manifest contract | [CONTRACTS.md](./CONTRACTS.md) § Multi-Re |

---

## Operator / release (not automated)

| Gap | Notes |
|-----|--------|
| **Disk & HF policy** | Confirm ~15–20 GB free and org policy allows Hugging Face download of the **interpolated** subset only (not raw ~460 GB). |
| **Real-case probe log** | Run `python scripts/inspect_cfdbench_sample.py <path/to/cylinder/case>` on one upstream case and paste shapes/Re/channels into `experiments/stage3_re_generalization.md` (or a lab notebook) for reproducibility. |
| **Git tags** | `data-stage3-v0.1` and `ml-stage3-v0.1` are **documented** in [STAGE3_RELEASE.md](./STAGE3_RELEASE.md) but not created by the repo; apply after `verify_stage3_local.py` passes. |
| **Full E2E on imported CFDBench** | No CI job runs HF download + full train + full `test_re` eval. Use `main.py --stage3 --run` (or step scripts) locally; gate with `scripts/verify_stage3_local.py`. |
| **Training quality bar on real data** | Stub/mini tests assert `beats_persistence`; there is **no** automated check that a full `stage3_cnn_re` run beats persistence on **val** Re after CFDBench import. |

---

## CI vs local coverage

| In CI | Local / manual only |
|-------|---------------------|
| Hydra compose `dataset=stage3_cfdbench` | HF download and multi-GB import |
| `cfdbench_mini` fixture, stub multi-Re | Full `train_re` / `val_re` / `test_re` coverage on disk |
| `test_stage3_pipeline.py` dry-run | `verify_stage3_local.py` after real import |
| Optional `@pytest.mark.slow` import test when cache exists | GPU training on full subset |

---

## Small product / API gaps (optional polish)

| Gap | Workaround |
|-----|------------|
| **`main.py --stage3` has no `--re`** | Use `scripts/run_stage3_pipeline.py --re 100 --re 200` or `scripts/download_stage3_cfdbench.py --re …`. |
| **`stage3_baseline` run id** | Mentioned in early layout docs only; pipeline trains `stage3_cnn_re`. Use `scripts/evaluate.py` per sim or add a dedicated baseline script if needed. |
| **`cfdbench_mini` + `FlowMultiReDataset`** | Mini fixture is covered for tensor load and training tests; an explicit `FlowMultiReDataset` length/shape test on `cfdbench_mini` is still optional (stub path is the canonical dataset test). |
| **`clean --preset pipeline`** | Wipes Zenodo/generated/mesh cache and runs; it does **not** remove `dataset/cfdbench_data` or `.cache/cfdbench`. Use `--preset stage3` for Stage 3 only. |
| **Stage 1 vs Stage 3 numeric comparison** | No script aligns Zenodo Re≈100 vs CFDBench Re≈100; compare in prose only (different source/grid). |

---

## Deferred (explicitly out of Stage 3 v1)

- CFDBench families beyond cylinder **`prop`** Re sweep (airfoil, BC/geometry grids).
- MeshGraphNet Reynolds sweeps (Stage 2 extension or new stage).
- Full raw CFDBench mirror (~460 GB).
- Re-conditioned **FNO** / **ConvLSTM** (CNN+Re is v1; see PRD stretch).
- Physics-informed multi-Re divergence losses.
- Single `main.py` command chaining Stage 1 + 2 + 3.
- Own multi-Re LBM/FD export path (Track B): same contract, import adapter not required if CFDBench suffices.

---

## Cross-plan references (unchanged scope)

- [PRD.md](./PRD.md) §11 (Re splits), §14 (experiments 1–3).
- [ML_EVAL_TASKS.md](../ML_EVAL_TASKS.md) — Re figures (heatmap, horizon-by-Re implemented).
- Stage 2 remains separate storage and metrics (`STAGE2_TASKS.md` if present locally).

---

## Suggested order if you continue Stage 3 work

1. Local import with `--max-cases` until `verify_stage3_local.py` passes (dataset + eval).
2. Record one real `inspect_cfdbench_sample.py` transcript in the experiment doc.
3. Train/eval on GPU if needed; tag `data-stage3-v0.1` / `ml-stage3-v0.1` per [STAGE3_RELEASE.md](./STAGE3_RELEASE.md).
4. Optional: add `FlowMultiReDataset` test on `cfdbench_mini`; wire `--re` on `main.py --stage3`; implement `stage3_baseline` or Stage-1 comparison script only if research needs them.
