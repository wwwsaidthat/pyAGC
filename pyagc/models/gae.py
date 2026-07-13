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
) -> Tensor:
    """Sample negative edges using PyG's :func:`negative_sampling`.

    Uses the sparse-matrix method to guarantee sampled edges are NOT in
    ``pos_edge_index`` — no self-loops, no duplicate positives, and no
    reverse-direction positives in undirected graphs.

    Args:
        num_nodes: Number of nodes in the (sub)graph.
        num_pos: Number of negative edges to sample.
        device: Target device (ignored; output is on the same device as
            ``pos_edge_index``).
        pos_edge_index: Positive edge_index of shape (2, E_pos) used to
            exclude true edges from the negative sample.

    Returns:
        Negative edge_index of shape (2, num_pos).
    """
    return negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_pos,
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

    The model is trained by minimizing the binary cross-entropy between
    the input adjacency and the reconstructed adjacency.

    Args:
        encoder (torch.nn.Module): The encoder (e.g. GCN) that produces node
            embeddings Z from (X, A).
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def reset_parameters(self):
        r"""Resets all learnable parameters."""
        reset(self.encoder)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Encodes the graph into node embeddings Z.

        Returns:
            Node embeddings of shape (N, D).
        """
        return self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        r"""Computes edge probabilities via scaled inner product + sigmoid.

        .. math::
            s_{ij} = \frac{z_i^\top z_j}{\sqrt{d}}, \quad
            p(A_{ij}=1) = \sigma(s_{ij})

        Scaling by :math:`1/\sqrt{d}` prevents logit variance from growing
        with embedding dimension, avoiding sigmoid saturation at high ``d``.

        Args:
            z: Node embeddings of shape (N, D).
            edge_index: Edge indices of shape (2, E).

        Returns:
            Edge probabilities of shape (E,).
        """
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1) / math.sqrt(z.size(-1)))

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Binary cross-entropy reconstruction loss on positive and negative edges.

        .. math::
            \mathcal{L}_{\text{recon}} =
            -\frac{1}{|E^+|}\sum_{(i,j)\in E^+} \log \sigma(z_i^\top z_j)
            -\frac{1}{|E^-|}\sum_{(i,j)\in E^-} \log (1 - \sigma(z_i^\top z_j))

        Args:
            z: Node embeddings of shape (N, D).
            pos_edge_index: Positive edges (existing edges) of shape (2, E_pos).
            neg_edge_index: Negative edges (non-edges) of shape (2, E_neg).
                If None, random negatives are sampled.

        Returns:
            Scalar reconstruction loss.
        """
        pos_score = self.decode(z, pos_edge_index)
        pos_loss = -torch.log(pos_score + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = _sample_neg_edges(
                z.size(0), pos_edge_index.size(1), z.device, pos_edge_index,
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
        r"""Computes the GAE training loss (full graph).

        Args:
            x: Node features of shape (N, F).
            edge_index: Edge indices of shape (2, E).

        Returns:
            LossOutput with total loss and reconstruction component.
        """
        z = self.forward(x, edge_index)
        recon = self.recon_loss(z, edge_index)

        return LossOutput(
            total=recon,
            components={'recon': recon.item()}
        )

    def loss_batch(self, batch: Data) -> LossOutput:
        r"""Computes loss for a mini-batch from NeighborLoader.

        Uses the subgraph edge_index for negative sampling.  In the
        NeighborLoader setting, all edges among the sampled nodes are
        guaranteed to be present in ``batch.edge_index``, so
        :func:`negative_sampling` correctly avoids them.

        Args:
            batch: A mini-batch subgraph.

        Returns:
            LossOutput with total loss and reconstruction component.
        """
        z = self.forward(batch.x, batch.edge_index)
        recon = self.recon_loss(z, batch.edge_index)

        return LossOutput(
            total=recon,
            components={'recon': recon.item()}
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(encoder={self.encoder})"
