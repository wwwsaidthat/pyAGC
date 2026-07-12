"""Graph Auto-Encoder (GAE) and Variational Graph Auto-Encoder (VGAE) models.

Reference:
    Kipf & Welling, "Variational Graph Auto-Encoders", NeurIPS 2016.
    https://arxiv.org/abs/1611.07308
"""

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset

from pyagc.models import TrainableModel, LossOutput
from pyagc.utils import filter_kwargs

EPS = 1e-12


def _sample_neg_edges(num_nodes: int, num_pos: int, device: torch.device) -> Tensor:
    """Sample random negative edges (non-edges) within the graph.

    Args:
        num_nodes: Number of nodes in the (sub)graph.
        num_pos: Number of positive edges to match.
        device: Target device.

    Returns:
        Negative edge_index of shape (2, num_pos).
    """
    neg_src = torch.randint(0, num_nodes, (num_pos,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_pos,), device=device)
    return torch.stack([neg_src, neg_dst], dim=0)


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
        r"""Computes edge probabilities via inner product + sigmoid.

        .. math::
            p(A_{ij}=1) = \sigma(z_i^\top z_j)

        Args:
            z: Node embeddings of shape (N, D).
            edge_index: Edge indices of shape (2, E).

        Returns:
            Edge probabilities of shape (E,).
        """
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

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
                z.size(0), pos_edge_index.size(1), z.device
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


class VGAE(TrainableModel):
    r"""The Variational Graph Auto-Encoder (VGAE) model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper (Kipf & Welling, NeurIPS 2016).

    VGAE extends GAE by introducing a probabilistic latent variable model:

    .. math::
        q(Z | X, A) &= \prod_i \mathcal{N}(z_i | \mu_i, \text{diag}(\sigma_i^2)) \\
        p(A | Z)   &= \prod_{i,j} \sigma(z_i^\top z_j)

    The inference model (encoder) outputs :math:`\mu` and :math:`\log\sigma^2`.
    Latent variables are sampled via the reparameterization trick.

    The loss is the negative variational lower bound (ELBO):

    .. math::
        \mathcal{L} = -\mathbb{E}_{q}[\log p(A|Z)] + \text{KL}[q(Z|X,A) \| p(Z)]

    where :math:`p(Z) = \mathcal{N}(0, I)`.

    Args:
        encoder (torch.nn.Module): Encoder that returns a tuple ``(mu, logvar)``
            of shape ``(N, D)`` each.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def reset_parameters(self):
        r"""Resets all learnable parameters."""
        reset(self.encoder)

    def encode(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        r"""Encodes the graph and returns distribution parameters.

        Returns:
            Tuple of (mu, logvar), each of shape (N, D).
        """
        return self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        r"""Samples latent Z via the reparameterization trick.

        .. math::
            z = \mu + \epsilon \odot \sigma, \quad \epsilon \sim \mathcal{N}(0, I)

        During evaluation (:obj:`self.training == False`), returns mu directly.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        r"""Computes edge probabilities via inner product + sigmoid."""
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=-1))

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Binary cross-entropy reconstruction loss (same as GAE)."""
        pos_score = self.decode(z, pos_edge_index)
        pos_loss = -torch.log(pos_score + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = _sample_neg_edges(
                z.size(0), pos_edge_index.size(1), z.device
            )

        neg_score = self.decode(z, neg_edge_index)
        neg_loss = -torch.log(1 - neg_score + EPS).mean()

        return pos_loss + neg_loss

    def kl_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        r"""Kullback-Leibler divergence between q(Z|X,A) and the prior p(Z).

        .. math::
            \text{KL} = -\frac{1}{2N}\sum_i \sum_j
            \big(1 + \log\sigma_{ij}^2 - \mu_{ij}^2 - \sigma_{ij}^2\big)

        Args:
            mu: Mean vectors of shape (N, D).
            logvar: Log-variance vectors of shape (N, D).

        Returns:
            Scalar KL divergence averaged over nodes and dimensions.
        """
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        )

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Computes deterministic node embeddings (mu) for downstream evaluation."""
        mu, _ = self.encode(*args, **kwargs)
        return mu

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Forward pass returning latent Z and distribution parameters.

        Returns:
            Tuple of (z, mu, logvar), each of shape (N, D).
        """
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""Computes the VGAE training loss (ELBO).

        Args:
            x: Node features of shape (N, F).
            edge_index: Edge indices of shape (2, E).

        Returns:
            LossOutput with total, reconstruction, and KL components.
        """
        z, mu, logvar = self.forward(x, edge_index)
        recon = self.recon_loss(z, edge_index)
        kl = self.kl_loss(mu, logvar)
        total = recon + kl

        return LossOutput(
            total=total,
            components={'recon': recon.item(), 'kl': kl.item()}
        )

    def loss_batch(self, batch: Data) -> LossOutput:
        r"""Computes VGAE loss for a mini-batch from NeighborLoader.

        Args:
            batch: A mini-batch subgraph.

        Returns:
            LossOutput with total, reconstruction, and KL components.
        """
        z, mu, logvar = self.forward(batch.x, batch.edge_index)
        recon = self.recon_loss(z, batch.edge_index)
        kl = self.kl_loss(mu, logvar)
        total = recon + kl

        return LossOutput(
            total=total,
            components={'recon': recon.item(), 'kl': kl.item()}
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(encoder={self.encoder})"
