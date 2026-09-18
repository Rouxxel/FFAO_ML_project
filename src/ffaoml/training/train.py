"""
#############################################################################
### CNN training runner
###
### @file train.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Train the one-step CNN on Stage 1 temporal splits and write ARCHITECTURE §7.3
checkpoint bundles.
"""

# Native imports
from __future__ import annotations

import json
import shutil
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
from ffaoml.data.catalog import dataset_root_from_config
from ffaoml.data.loading import load_split_tensor
from ffaoml.evaluation.rollout import one_step_baseline_metrics
from ffaoml.manifests import (
    DATASET_MANIFEST_FILENAME,
    dataset_manifest_path,
    git_short_commit,
    hash_config,
)
from ffaoml.ml.dataset import build_flow_datasets
from ffaoml.ml.preprocessing import (
    fit_preprocess_stats,
    normalize_fields,
    save_preprocess_stats,
)
from ffaoml.models.cnn import build_flow_cnn
from ffaoml.training.losses import build_loss_fn

"""CONSTANTS-----------------------------------------------------------"""
MODEL_FILENAME = "model.pt"
TRAINING_SUMMARY_FILENAME = "training_summary.json"

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class TrainingResult:
    """Artifacts from ``run_cnn_training``."""

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


def _collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch as th

    return {
        "input": th.stack([th.from_numpy(s["input"]) for s in samples]),
        "target": th.stack([th.from_numpy(s["target"]) for s in samples]),
        "time_index": [s["time_index"] for s in samples],
        "re": [s["re"] for s in samples],
    }


def evaluate_loader_mse(model, loader, device, loss_fn) -> float:
    """Mean validation MSE over a DataLoader."""
    _require_torch()
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total += float(loss.item()) * x.size(0)
            count += x.size(0)
    return total / max(count, 1)


def _copy_dataset_manifest(cfg: DictConfig, run_dir: Path) -> None:
    root = dataset_root_from_config(cfg)
    src = dataset_manifest_path(root)
    if src.is_file():
        shutil.copy2(src, run_dir / DATASET_MANIFEST_FILENAME)


"""TRAINING-----------------------------------------------------------"""


def run_cnn_training(
    cfg: DictConfig,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> TrainingResult:
    """
    Train ``FlowCNN`` and export a checkpoint bundle.

    Parameters:
        cfg (DictConfig): Composed Hydra config.
        output_dir (str | Path): Run directory under ``results/runs/``.
        repo_root (str | Path | None): For git metadata in summaries.

    Returns:
        TrainingResult: Paths and validation comparison vs persistence.
    """
    _require_torch()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(cfg.train.device))
    stats = fit_preprocess_stats(cfg)
    save_preprocess_stats(out / "preprocess_stats.json", stats)
    datasets = build_flow_datasets(cfg, stats=stats)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=_collate_batch,
    )

    model = build_flow_cnn(cfg).to(device)
    loss_fn = build_loss_fn(field_mse=bool(cfg.train.loss.field_mse))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
    )

    best_val = float("inf")
    best_path = out / MODEL_FILENAME
    epochs = int(cfg.train.epochs)
    checkpoint_every = int(cfg.train.checkpoint_every)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mse = evaluate_loader_mse(model, val_loader, device, loss_fn)
        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_mse": val_mse,
                },
                best_path,
            )
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            torch.save(model.state_dict(), out / f"checkpoint_epoch_{epoch}.pt")

    if best_path.is_file():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    val_raw = load_split_tensor(cfg, "val")
    val_norm = normalize_fields(val_raw, stats)
    persistence_mse = one_step_baseline_metrics(val_norm, "persistence")["mse"]
    beats = best_val < persistence_mse

    summary = {
        "schema_version": 1,
        "model": str(cfg.model.name),
        "epochs": epochs,
        "best_val_mse": best_val,
        "persistence_val_mse": persistence_mse,
        "beats_persistence": beats,
        "config_hash": hash_config(cfg),
        "git_commit": git_short_commit(repo_root),
    }
    summary_path = out / TRAINING_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    write_resolved_config(cfg, out)
    _copy_dataset_manifest(cfg, out)

    return TrainingResult(
        output_dir=out,
        model_path=best_path,
        summary_path=summary_path,
        best_val_mse=best_val,
        persistence_val_mse=persistence_mse,
        beats_persistence=beats,
    )
