"""
#############################################################################
### ConvLSTM training runner
###
### @file train_convlstm.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Train ``FlowConvLSTM`` with optional truncated unroll (``train.unroll_steps``).
"""

# Native imports
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party imports
from omegaconf import DictConfig

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]

# Project imports
from ffaoml.config import write_resolved_config
from ffaoml.data.loading import load_pooled_split_tensor
from ffaoml.evaluation.rollout import one_step_baseline_metrics
from ffaoml.manifests import (
    RUN_DATASET_MANIFEST_SNAPSHOT_FILENAME,
    git_short_commit,
    hash_config,
    hash_file,
)
from ffaoml.ml.dataset import build_flow_unroll_datasets
from ffaoml.ml.preprocessing import (
    fit_preprocess_stats,
    normalize_fields,
    save_preprocess_stats,
)
from ffaoml.models.convlstm import build_flow_convlstm
from ffaoml.training.bundle import (
    copy_dataset_manifest_snapshot,
    finalize_training_bundle,
)
from ffaoml.training.losses import build_training_loss
from ffaoml.training.progress import log_training_epoch, log_training_start
from ffaoml.training.train import MODEL_FILENAME, TRAINING_SUMMARY_FILENAME

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class ConvLSTMTrainingResult:
    """Artifacts from ``run_convlstm_training``."""

    output_dir: Path
    model_path: Path
    summary_path: Path
    best_val_mse: float
    persistence_val_mse: float
    beats_persistence: bool


"""HELPERS-----------------------------------------------------------"""


def _require_torch() -> None:
    if torch is None or DataLoader is None:
        raise ImportError("torch is required; install with pip install -e '.[ml]'")


def _collate_unroll(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch as th

    return {
        "input": th.stack([th.from_numpy(s["input"]) for s in samples]),
        "targets": th.stack([th.from_numpy(s["targets"]) for s in samples]),
        "time_index": [s["time_index"] for s in samples],
        "re": [s["re"] for s in samples],
    }


def _training_loss(
    model,
    batch: dict[str, Any],
    device: torch.device,
    loss_fn,
    unroll_steps: int,
    teacher_forcing: bool,
) -> torch.Tensor:
    x0 = batch["input"].to(device)
    targets = batch["targets"].to(device)
    if unroll_steps <= 1:
        pred, _ = model(x0)
        return loss_fn(pred, targets[:, 0])
    preds = model.forward_unroll(
        x0,
        unroll_steps,
        teacher_forcing=teacher_forcing,
        target_seq=targets if teacher_forcing else None,
    )
    step_losses = [loss_fn(preds[:, k], targets[:, k]) for k in range(unroll_steps)]
    return sum(step_losses) / len(step_losses)


def evaluate_convlstm_mse(
    model,
    loader,
    device,
    loss_fn,
) -> float:
    """One-step (first frame) validation MSE for checkpoint selection."""
    _require_torch()
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x0 = batch["input"].to(device)
            targets = batch["targets"].to(device)
            pred, _ = model(x0)
            loss = loss_fn(pred, targets[:, 0])
            total += float(loss.item()) * x0.size(0)
            count += x0.size(0)
    return total / max(count, 1)


"""TRAINING-----------------------------------------------------------"""


def run_convlstm_training(
    cfg: DictConfig,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> ConvLSTMTrainingResult:
    """
    Train ``FlowConvLSTM`` and export a checkpoint bundle.

    Parameters:
        cfg (DictConfig): Composed Hydra config (``model=convlstm``).
        output_dir (str | Path): Run directory under ``results/runs/``.
        repo_root (str | Path | None): For git metadata in summaries.

    Returns:
        ConvLSTMTrainingResult: Paths and validation comparison vs persistence.
    """
    _require_torch()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    unroll_steps = int(cfg.train.get("unroll_steps", 1))
    teacher_forcing = bool(cfg.train.get("teacher_forcing", True))
    device = torch.device(str(cfg.train.device))
    stats = fit_preprocess_stats(cfg)
    save_preprocess_stats(out / "preprocess_stats.json", stats)
    datasets = build_flow_unroll_datasets(cfg, unroll_steps=unroll_steps, stats=stats)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=_collate_unroll,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=_collate_unroll,
    )

    model = build_flow_convlstm(cfg).to(device)
    loss_fn = build_training_loss(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
    )

    best_val = float("inf")
    best_path = out / MODEL_FILENAME
    epochs = int(cfg.train.epochs)
    checkpoint_every = int(cfg.train.checkpoint_every)
    model_name = str(cfg.model.name)
    log_training_start(
        model_name=model_name,
        epochs=epochs,
        device=str(device),
        output_dir=str(out),
    )

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            loss = _training_loss(
                model,
                batch,
                device,
                loss_fn,
                unroll_steps,
                teacher_forcing,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mse = evaluate_convlstm_mse(model, val_loader, device, loss_fn)
        is_best = val_mse < best_val
        if is_best:
            best_val = val_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": val_mse,
                },
                best_path,
            )
        log_training_epoch(
            model_name=model_name,
            epoch=epoch,
            epochs=epochs,
            val_mse=val_mse,
            best_val_mse=best_val,
            is_best=is_best,
        )
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            torch.save(model.state_dict(), out / f"checkpoint_epoch_{epoch}.pt")

    if best_path.is_file():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    val_raw = load_pooled_split_tensor(cfg, "val")
    val_norm = normalize_fields(val_raw, stats)
    persistence_mse = one_step_baseline_metrics(val_norm, "persistence")["mse"]
    beats = best_val < persistence_mse

    write_resolved_config(cfg, out)
    copy_dataset_manifest_snapshot(cfg, out)
    ds_manifest_path = out / RUN_DATASET_MANIFEST_SNAPSHOT_FILENAME
    dataset_manifest_hash = (
        hash_file(ds_manifest_path) if ds_manifest_path.is_file() else None
    )
    summary = {
        "schema_version": 1,
        "model": str(cfg.model.name),
        "epochs": epochs,
        "seed": int(cfg.seed),
        "unroll_steps": unroll_steps,
        "teacher_forcing": teacher_forcing,
        "best_val_mse": best_val,
        "persistence_val_mse": persistence_mse,
        "beats_persistence": beats,
        "config_hash": hash_config(cfg),
        "dataset_manifest_hash": dataset_manifest_hash,
        "git_commit": git_short_commit(repo_root),
    }
    summary_path = out / TRAINING_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    finalize_training_bundle(cfg, out, repo_root=repo_root)

    return ConvLSTMTrainingResult(
        output_dir=out,
        model_path=best_path,
        summary_path=summary_path,
        best_val_mse=best_val,
        persistence_val_mse=persistence_mse,
        beats_persistence=beats,
    )
