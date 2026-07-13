"""Graph Auto-Encoder (GAE) model.

Reference:
    Kipf & Welling, "Variational Graph Auto-Encoders", NeurIPS 2016.
    https://arxiv.org/abs/1611.07308
"""

from typing import Optional

import torch
import torch.nn.functional as F
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
        \hat{A} = \sigma(\gamma \cdot \tilde{Z} \tilde{Z}^\top), \quad
        \tilde{Z} = \text{L2-Norm}(\text{GCN}(X, A))

    Embeddings are L2-normalized so the inner product is bounded in
    :math:`[-1, 1]` regardless of dimensionality.  A learnable temperature
    :math:`\gamma` (initialised at 1.0) controls sigmoid sharpness.

    Args:
        encoder (torch.nn.Module): The encoder (e.g. GCN) that produces node
            embeddings Z from (X, A).
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        r"""Resets all learnable parameters."""
        reset(self.encoder)
        self.gamma.data.fill_(1.0)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Encodes the graph into **L2-normalized** node embeddings Z.

        Returns:
            Node embeddings of shape (N, D), each row has unit L2 norm.
        """
        z = self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))
        return F.normalize(z, p=2, dim=-1)

    def decode(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Computes (scaled) inner-product logits or sigmoid probabilities.

        .. math::
            s_{ij} = \gamma \cdot \tilde{z}_i^\top \tilde{z}_j, \quad
            p(A_{ij}=1) = \sigma(s_{ij})

        Because embeddings are L2-normalised the raw dot product is in
        :math:`[-1, 1]`; the learnable :obj:`gamma` parameter restores the
        dynamic range lost by normalisation.

        Args:
            z: L2-normalised node embeddings of shape (N, D).
            edge_index: Edge indices of shape (2, E).
            sigmoid: If True, return probabilities; otherwise return raw logits.

        Returns:
            Edge probabilities/logits of shape (E,).
        """
        src, dst = edge_index
        logits = self.gamma * (z[src] * z[dst]).sum(dim=-1)
        return torch.sigmoid(logits) if sigmoid else logits

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Binary cross-entropy reconstruction loss using numerically-stable
        :func:`F.logsigmoid`.

        .. math::
            \mathcal{L}_{\text{recon}} =
            -\frac{1}{|E^+|}\sum_{(i,j)\in E^+} \log \sigma(s_{ij})
            -\frac{1}{|E^-|}\sum_{(i,j)\in E^-} \log (1 - \sigma(s_{ij}))

        Args:
            z: L2-normalised node embeddings of shape (N, D).
            pos_edge_index: Positive edges of shape (2, E_pos).
            neg_edge_index: Negative edges of shape (2, E_neg).
                If None, random negatives are sampled.

        Returns:
            Scalar reconstruction loss.
        """
        pos_logits = self.decode(z, pos_edge_index, sigmoid=False)
        pos_loss = -F.logsigmoid(pos_logits).mean()

        if neg_edge_index is None:
            neg_edge_index = _sample_neg_edges(
                z.size(0), pos_edge_index.size(1), z.device, pos_edge_index,
            )

        neg_logits = self.decode(z, neg_edge_index, sigmoid=False)
        neg_loss = -F.logsigmoid(-neg_logits).mean()

        return pos_loss + neg_loss

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Computes L2-normalised node embeddings for downstream evaluation."""
        return self.encode(*args, **kwargs)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        r"""Encodes the graph and returns L2-normalised node embeddings Z."""
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
        return f"{self.__class__.__name__}(encoder={self.encoder}, gamma={self.gamma.item():.3f})"
