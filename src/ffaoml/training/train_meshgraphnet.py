"""
#############################################################################
### MeshGraphNet training (Stage 2)
###
### @file train_meshgraphnet.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Train one-step mesh node prediction and write the same checkpoint bundle layout
as Stage 1 grid models.
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
from ffaoml.manifests import (
    RUN_DATASET_MANIFEST_SNAPSHOT_FILENAME,
    git_short_commit,
    hash_config,
    hash_file,
)
from ffaoml.ml.mesh_dataset import build_mesh_datasets, collate_mesh_graph_batch
from ffaoml.ml.mesh_preprocessing import (
    fit_mesh_preprocess_stats,
    save_mesh_preprocess_stats,
)
from ffaoml.models.meshgraphnet import build_meshgraphnet
from ffaoml.training.bundle import (
    copy_dataset_manifest_snapshot,
    finalize_training_bundle,
)
from ffaoml.training.checkpointing import load_model_weights
from ffaoml.training.losses import build_training_loss
from ffaoml.training.progress import log_training_epoch, log_training_start

"""CONSTANTS-----------------------------------------------------------"""
MODEL_FILENAME = "model.pt"
TRAINING_SUMMARY_FILENAME = "training_summary.json"
PREPROCESS_STATS_FILENAME = "preprocess_stats.json"

"""TYPES-----------------------------------------------------------"""


@dataclass(frozen=True)
class MeshTrainingResult:
    """Artifacts from ``run_meshgraphnet_training``."""

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


def _collate_mesh_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    import torch as th

    batch = collate_mesh_graph_batch(samples)
    return {
        "input": th.from_numpy(batch["input"]),
        "target": th.from_numpy(batch["target"]),
        "edge_index": th.from_numpy(batch["edge_index"]).long(),
        "node_type": th.from_numpy(batch["node_type"]).long(),
        "mesh_pos": th.from_numpy(batch["mesh_pos"].astype("float32")),
        "batch": th.from_numpy(batch["batch"]).long(),
    }


def _forward_batch(model, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    return model(
        batch["input"].to(device),
        batch["mesh_pos"].to(device),
        batch["edge_index"].to(device),
        batch["node_type"].to(device),
    )


def evaluate_mesh_loader_mse(
    model,
    loader,
    device: torch.device,
    loss_fn,
) -> float:
    """Mean batch MSE over a mesh DataLoader."""
    _require_torch()
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            pred = _forward_batch(model, batch, device)
            y = batch["target"].to(device)
            loss = loss_fn(pred, y)
            total += float(loss.item())
            count += 1
    return total / max(count, 1)


def evaluate_mesh_persistence_mse(
    loader,
    device: torch.device,
    loss_fn,
) -> float:
    """Persistence baseline: predict ``input`` as next step."""
    _require_torch()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            loss = loss_fn(x, y)
            total += float(loss.item())
            count += 1
    return total / max(count, 1)


"""TRAINING-----------------------------------------------------------"""


def run_meshgraphnet_training(
    cfg: DictConfig,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> MeshTrainingResult:
    """
    Train MeshGraphNet on trajectory splits and export a checkpoint bundle.

    Parameters:
        cfg (DictConfig): Composed Hydra config (``dataset=stage2_meshgraphnets``).
        output_dir (str | Path): Run directory under ``results/runs/``.
        repo_root (str | Path | None): For git metadata in summaries.

    Returns:
        MeshTrainingResult: Paths and validation comparison vs persistence.
    """
    _require_torch()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(cfg.train.device))
    stats = fit_mesh_preprocess_stats(cfg)
    save_mesh_preprocess_stats(out / PREPROCESS_STATS_FILENAME, stats)
    datasets = build_mesh_datasets(cfg, stats)

    batch_size = int(cfg.train.batch_size)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=_collate_mesh_batch,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=_collate_mesh_batch,
    )

    model = build_meshgraphnet(cfg).to(device)
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
            pred = _forward_batch(model, batch, device)
            y = batch["target"].to(device)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mse = evaluate_mesh_loader_mse(model, val_loader, device, loss_fn)
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
        load_model_weights(model, checkpoint)

    persistence_mse = evaluate_mesh_persistence_mse(val_loader, device, loss_fn)
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

    return MeshTrainingResult(
        output_dir=out,
        model_path=best_path,
        summary_path=summary_path,
        best_val_mse=best_val,
        persistence_val_mse=persistence_mse,
        beats_persistence=beats,
    )
