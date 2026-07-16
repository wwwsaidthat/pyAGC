"""Graph Auto-Encoder (GAE) model.

Reference:
    Kipf & Welling, "Variational Graph Auto-Encoders", NeurIPS 2016.
    https://arxiv.org/abs/1611.07308
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling

from pyagc.models import TrainableModel, LossOutput
from pyagc.utils import filter_kwargs

EPS = 1e-12


def _sample_neg_edges(
    num_nodes: int,
    num_pos: int,
    device: torch.device,
    pos_edge_index: Tensor,
    neg_ratio: float = 1.0,
) -> Tensor:
    """Sample negative edges using PyG's :func:`negative_sampling`.

    Args:
        num_nodes: Number of nodes in the (sub)graph.
        num_pos: Number of positive edges (scaled by ``neg_ratio``).
        device: Target device (ignored; output follows ``pos_edge_index`` device).
        pos_edge_index: Positive edge_index of shape ``(2, E_pos)``.
        neg_ratio: Fraction of positive count to sample as negatives (default 1.0).

    Returns:
        Negative edge_index of shape ``(2, num_neg)``.
    """
    num_neg = max(1, int(num_pos * neg_ratio))
    return negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method="sparse",
    )


class GAE(TrainableModel):
    r"""The Graph Auto-Encoder (GAE) model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper (Kipf & Welling, NeurIPS 2016).

    GAE learns node embeddings by reconstructing the graph adjacency matrix
    through an inner-product decoder:

    .. math::
        \hat{A} = \sigma(Z Z^\top), \quad Z = \text{GCN}(X, A)

    Args:
        encoder: The encoder (e.g. GCN) that produces node embeddings Z.
        neg_ratio: Fraction of positive edges to sample as negatives (default 1.0).
            Use 0.25--0.5 to reduce CPU RAM on large graphs.
    """

    def __init__(self, encoder: nn.Module, neg_ratio: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.neg_ratio = float(neg_ratio)

    def reset_parameters(self):
        r"""Resets all learnable parameters."""
        reset(self.encoder)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Encodes the graph into node embeddings Z.

        Returns:
            Node embeddings of shape ``(N, D)``.
        """
        return self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        r"""Computes edge probabilities via inner product + sigmoid.

        .. math::
            p(A_{ij}=1) = \sigma(z_i^\top z_j / \sqrt{d})

        Scaling by :math:`1/\sqrt{d}` prevents logit variance from growing
        with embedding dimension.

        Args:
            z: Node embeddings of shape ``(N, D)``.
            edge_index: Edge indices of shape ``(2, E)``.

        Returns:
            Edge probabilities of shape ``(E,)``.
        """
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1) / math.sqrt(z.size(-1)))

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Binary cross-entropy reconstruction loss.

        Args:
            z: Node embeddings of shape ``(N, D)``.
            pos_edge_index: Positive edges of shape ``(2, E_pos)``.
            neg_edge_index: Negative edges of shape ``(2, E_neg)``.
                If None, random negatives are sampled using ``neg_ratio``.

        Returns:
            Scalar reconstruction loss.
        """
        pos_score = self.decode(z, pos_edge_index)
        pos_loss = -torch.log(pos_score + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = _sample_neg_edges(
                z.size(0), pos_edge_index.size(1), z.device, pos_edge_index,
                neg_ratio=self.neg_ratio,
            )

        neg_score = self.decode(z, neg_edge_index)
        neg_loss = -torch.log(1 - neg_score + EPS).mean()

        return pos_loss + neg_loss

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Computes node embeddings for downstream evaluation."""
        return self.encode(*args, **kwargs)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        r"""Encodes the graph and returns node embeddings Z."""
        return self.encode(x, edge_index)

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""Computes the GAE training loss (full graph)."""
        z = self.forward(x, edge_index)
        recon = self.recon_loss(z, edge_index)
        return LossOutput(total=recon, components={'recon': recon.item()})

    def loss_batch(self, batch: Data) -> LossOutput:
        r"""Computes loss for a mini-batch from NeighborLoader."""
        z = self.forward(batch.x, batch.edge_index)
        recon = self.recon_loss(z, batch.edge_index)
        return LossOutput(total=recon, components={'recon': recon.item()})

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(encoder={self.encoder}, "
                f"neg_ratio={self.neg_ratio:.2f})")
