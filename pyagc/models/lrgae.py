"""Left-right Graph Auto-Encoder (lrGAE).

This module follows the structure-based ``lrGAE-vu`` implementation from
Li et al., "Revisiting and Benchmarking Graph Autoencoders: A Contrastive
Learning Perspective".  It deliberately does not use spectral low-rank
targets: ``lr`` means *left-right*, not *low-rank*.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import BatchNorm, GCNConv
from torch_geometric.utils import negative_sampling


class LayerwiseGCNEncoder(nn.Module):
    """GCN encoder returning the input and every receptive-field output."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.8,
        norm: str = "none",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if norm not in {"none", "batch_norm"}:
            raise ValueError("norm must be 'none' or 'batch_norm'")

        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = in_channels if layer == 0 else hidden_channels
            self.convs.append(GCNConv(in_dim, hidden_channels))
            self.norms.append(
                BatchNorm(hidden_channels) if norm == "batch_norm" else nn.Identity()
            )

    @property
    def num_layers(self) -> int:
        return len(self.convs)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> List[Tensor]:
        outputs = [x]
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(norm(x))
            outputs.append(x)
        return outputs


class EdgeDecoder(nn.Module):
    """Official-style MLP decoder over element-wise left/right products."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("decoder num_layers must be >= 1")

        layers: List[nn.Module] = []
        current = in_channels
        for _ in range(num_layers - 1):
            layers.extend(
                [nn.Dropout(dropout), nn.Linear(current, hidden_channels), nn.ELU()]
            )
            current = hidden_channels
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def reset_parameters(self) -> None:
        for module in self.network:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, left: Tensor, right: Tensor, pairs: Tensor) -> Tensor:
        pair_features = left[pairs[0]] * right[pairs[1]]
        return self.network(pair_features).view(-1)


def _mask_undirected_edges(
    edge_index: Tensor, num_nodes: int, ratio: float
) -> Tuple[Tensor, Tensor]:
    """Mask reciprocal directions together to prevent reverse-edge leakage."""
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {ratio}")

    src, dst = edge_index
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    pair_key = lo * num_nodes + hi
    _, inverse = torch.unique(pair_key, return_inverse=True)
    pair_mask = torch.rand(
        int(inverse.max().item()) + 1, device=edge_index.device
    ) < ratio
    masked = pair_mask[inverse]

    # Avoid degenerate views on tiny sampled subgraphs.
    if bool(masked.all()):
        masked[0] = False
    elif not bool(masked.any()):
        masked[0] = True
    return edge_index[:, ~masked], edge_index[:, masked]


class LRGAE(nn.Module):
    """Structure-based left-right GAE with ``vu`` node pairs.

    Variants follow the official node-classification protocol:

    * 5: AA, left=2, right=2
    * 6: AA, left=2, right=1
    * 7: AB, left=2, right=2
    * 8: AB, left=2, right=1
    """

    VARIANTS = {
        5: ("AA", 2, 2),
        6: ("AA", 2, 1),
        7: ("AB", 2, 2),
        8: ("AB", 2, 1),
    }

    def __init__(
        self,
        encoder: LayerwiseGCNEncoder,
        decoder: EdgeDecoder,
        variant: int = 8,
        mask_ratio: float = 0.7,
        neg_ratio: float = 1.0,
        decoder_batch_size: int = 131072,
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VARIANTS)}")
        self.encoder = encoder
        self.decoder = decoder
        self.variant = int(variant)
        self.view, self.left_layer, self.right_layer = self.VARIANTS[self.variant]
        if max(self.left_layer, self.right_layer) > encoder.num_layers:
            raise ValueError(
                f"lrGAE-{variant} needs at least "
                f"{max(self.left_layer, self.right_layer)} encoder layers"
            )
        self.mask_ratio = float(mask_ratio)
        self.neg_ratio = float(neg_ratio)
        self.decoder_batch_size = int(decoder_batch_size)
        if self.decoder_batch_size < 1:
            raise ValueError("decoder_batch_size must be >= 1")

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)[-1]

    def loss(self, x: Tensor, edge_index: Tensor) -> Tensor:
        remaining_edges, masked_edges = _mask_undirected_edges(
            edge_index, x.size(0), self.mask_ratio
        )

        left_outputs = self.encoder(x, remaining_edges)
        right_outputs = (
            self.encoder(x, masked_edges) if self.view == "AB" else left_outputs
        )
        left = left_outputs[self.left_layer]
        right = right_outputs[self.right_layer]

        num_neg = max(1, int(masked_edges.size(1) * self.neg_ratio))
        neg_edges = negative_sampling(
            edge_index=edge_index,
            num_nodes=x.size(0),
            num_neg_samples=num_neg,
            method="sparse",
        )
        return self._edge_bce(left, right, masked_edges, 1.0) + self._edge_bce(
            left, right, neg_edges, 0.0
        )

    def _edge_bce(
        self, left: Tensor, right: Tensor, pairs: Tensor, label: float
    ) -> Tensor:
        """Memory-bounded decoder loss for million-edge graphs.

        Non-reentrant checkpointing prevents the decoder from retaining an
        ``edges x embedding_dim`` product tensor until backward.
        """
        total = left.new_zeros(())
        for pair_batch in pairs.split(self.decoder_batch_size, dim=1):
            logits = checkpoint(
                self.decoder, left, right, pair_batch, use_reentrant=False
            )
            targets = torch.full_like(logits, label)
            total = total + F.binary_cross_entropy_with_logits(
                logits, targets, reduction="sum"
            )
        return total / max(pairs.size(1), 1)

    def extra_repr(self) -> str:
        return (
            f"variant={self.variant}, view={self.view}, "
            f"left={self.left_layer}, right={self.right_layer}, "
            f"mask_ratio={self.mask_ratio:.2f}, neg_ratio={self.neg_ratio:.2f}, "
            f"decoder_batch_size={self.decoder_batch_size}"
        )
