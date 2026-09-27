"""
#############################################################################
### MeshGraphNet (Stage 2) - simplified v1
###
### @file meshgraphnet.py
### @author Sebastian Russo
### @date 2026
#############################################################################

Encoder / message-passing processor / decoder following the MeshGraphNets
layout (Pfaff et al.). Edge features are relative mesh displacements; node
inputs are normalized ``[velocity, pressure]`` plus a ``node_type`` embedding.
"""

# Native imports
from __future__ import annotations

# Third-party imports
from omegaconf import DictConfig

"""CONSTANTS-----------------------------------------------------------"""
NUM_NODE_TYPES = 7
EDGE_FEATURE_DIM = 3

try:
    import torch
    from torch import nn

    def _mlp(
        in_dim: int, hidden_dim: int, out_dim: int, *, layers: int = 2
    ) -> nn.Sequential:
        """Two-layer MLP with LayerNorm + ReLU between hidden blocks."""
        if layers < 1:
            raise ValueError("layers must be >= 1")
        sizes = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        modules: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            modules.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                modules.append(nn.LayerNorm(sizes[i + 1]))
                modules.append(nn.ReLU())
        return nn.Sequential(*modules)

    def _scatter_add(
        src: torch.Tensor,
        index: torch.Tensor,
        dim_size: int,
    ) -> torch.Tensor:
        """Sum ``src`` rows into ``out[index]`` (0-dim aggregation)."""
        out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out

    class MeshGraphNet(nn.Module):
        """
        One-step mesh node predictor.

        Forward inputs are disjoint-union batches (single graph when ``batch_size=1``).
        """

        def __init__(
            self,
            *,
            node_in_dim: int,
            node_out_dim: int,
            hidden_dim: int,
            num_message_passing_steps: int,
            node_type_embed_dim: int,
        ) -> None:
            super().__init__()
            self.node_type_embed = nn.Embedding(NUM_NODE_TYPES, node_type_embed_dim)
            node_enc_in = node_in_dim + node_type_embed_dim
            self.node_encoder = _mlp(node_enc_in, hidden_dim, hidden_dim)
            self.edge_encoder = _mlp(EDGE_FEATURE_DIM, hidden_dim, hidden_dim)

            self.edge_processors = nn.ModuleList(
                [
                    _mlp(hidden_dim * 3, hidden_dim, hidden_dim)
                    for _ in range(num_message_passing_steps)
                ]
            )
            self.node_processors = nn.ModuleList(
                [
                    _mlp(hidden_dim * 2, hidden_dim, hidden_dim)
                    for _ in range(num_message_passing_steps)
                ]
            )
            self.decoder = _mlp(hidden_dim, hidden_dim, node_out_dim, layers=2)

        def _edge_features(
            self,
            mesh_pos: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> torch.Tensor:
            senders = edge_index[0]
            receivers = edge_index[1]
            disp = mesh_pos[senders] - mesh_pos[receivers]
            length = torch.linalg.norm(disp, dim=-1, keepdim=True).clamp_min(1e-8)
            return torch.cat([disp, length], dim=-1)

        def forward(
            self,
            node_features: torch.Tensor,
            mesh_pos: torch.Tensor,
            edge_index: torch.Tensor,
            node_type: torch.Tensor,
        ) -> torch.Tensor:
            """
            Predict normalized next-step node features.

            Parameters:
                node_features (torch.Tensor): ``(N, node_in_dim)``.
                mesh_pos (torch.Tensor): ``(N, 2)`` world positions.
                edge_index (torch.Tensor): ``(2, E)`` int64 COO edges.
                node_type (torch.Tensor): ``(N,)`` int64 MeshGraphNets types.

            Returns:
                torch.Tensor: ``(N, node_out_dim)``.
            """
            n_type = self.node_type_embed(node_type.long().clamp(0, NUM_NODE_TYPES - 1))
            node_h = self.node_encoder(torch.cat([node_features, n_type], dim=-1))
            edge_attr = self._edge_features(mesh_pos, edge_index)
            edge_h = self.edge_encoder(edge_attr)

            senders = edge_index[0]
            receivers = edge_index[1]
            n_nodes = node_features.size(0)

            for edge_mlp, node_mlp in zip(
                self.edge_processors, self.node_processors, strict=True
            ):
                edge_in = torch.cat(
                    [node_h[senders], node_h[receivers], edge_h],
                    dim=-1,
                )
                edge_h = edge_mlp(edge_in)
                agg = _scatter_add(edge_h, receivers, n_nodes)
                node_in = torch.cat([node_h, agg], dim=-1)
                node_h = node_mlp(node_in)

            return self.decoder(node_h)

except ImportError:  # pragma: no cover

    class MeshGraphNet:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "torch is required for MeshGraphNet; pip install -e '.[ml]'"
            )


def build_meshgraphnet(cfg: DictConfig) -> MeshGraphNet:
    """Instantiate ``MeshGraphNet`` from Hydra ``model`` group."""
    return MeshGraphNet(
        node_in_dim=int(cfg.model.node_in_dim),
        node_out_dim=int(cfg.model.node_out_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_message_passing_steps=int(cfg.model.num_message_passing_steps),
        node_type_embed_dim=int(cfg.model.node_type_embed_dim),
    )
