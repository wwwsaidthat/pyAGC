from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.transforms import BaseTransform

from pyagc.models import TrainableModel, LossOutput
from pyagc.utils import filter_kwargs


class SSGE(TrainableModel):
    r"""
    The SSGE (Negative-Free Self-Supervised Gaussian Embedding) model is proposed in the
    `Negative-Free Self-Supervised Gaussian Embedding of Graphs
    <https://www.sciencedirect.com/science/article/pii/S0893608024000334>`_ paper
    (Liu et al., Neural Networks 2024).

    SSGE learns node representations by maximizing the agreement between embeddings from
    two augmented graph views while encouraging the embedding distribution to be uniform
    (close to a standard Gaussian :math:`\mathcal{N}(0, I)` via the 2-Wasserstein distance).

    Compared to CCA-SSG, SSGE replaces the decorrelation (off-diagonal penalty) term
    with a uniformity term based on the Wasserstein distance between the normalized
    embeddings and :math:`\mathcal{N}(0, I)`.

    The loss function combines an invariance term and a uniformity term:

    .. math::
        \mathcal{L} = \underbrace{-\frac{1}{N}\sum_{i} \tilde{z}_i^{(1)} \cdot \tilde{z}_i^{(2)}}_{\text{invariance}} +
        \lambda \cdot \frac{1}{2}\Big(\mathcal{W}_2(\tilde{Z}^{(1)}, \mathcal{N}(0,I)) + \mathcal{W}_2(\tilde{Z}^{(2)}, \mathcal{N}(0,I))\Big)

    where:

    - :math:`\tilde{Z}^{(i)}` is the batch-normalized embedding of view :math:`i`;
    - :math:`\mathcal{W}_2(Z, \mathcal{N}(0,I)) = -2 \sum_i \sqrt{\lambda_i}` for eigenvalues
      :math:`\lambda_i` of the covariance matrix :math:`Z^T Z / (n-1)`;
    - :math:`\lambda` controls the strength of uniformity regularization.

    Args:
        encoder (torch.nn.Module): The encoder shared across both views.
        transform1 (torch_geometric.transforms.BaseTransform): The 1-st graph view transformation.
        transform2 (torch_geometric.transforms.BaseTransform): The 2-nd graph view transformation.
        lam (float): Weight of uniformity loss (default: :obj:`0.1`).
    """

    def __init__(self, encoder: nn.Module, transform1: BaseTransform, transform2: BaseTransform, lam: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.transform1 = transform1
        self.transform2 = transform2
        self.lam = lam

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Computes node embeddings."""
        return self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        r"""Generates embeddings from two graph augmentations."""
        data1 = self.transform1(*args, **kwargs)
        data2 = self.transform2(*args, **kwargs)
        z1 = self.encoder(**data1)
        z2 = self.encoder(**data2)
        return z1, z2

    def _uniformity(self, Z: Tensor) -> Tensor:
        r"""
        Computes the 2-Wasserstein distance between the batch-normalized Z and N(0, I).

        Following the SSGE formulation:

        .. math::
            \mathcal{W}_2(Z, \mathcal{N}(0,I)) = -2 \sum_i \sqrt{\lambda_i}

        where :math:`\lambda_i` are the eigenvalues of the covariance matrix
        :math:`C = Z^T Z / (n-1)`.

        Args:
            Z (torch.Tensor): Batch-normalized embeddings of shape (N, D).

        Returns:
            Scalar uniformity loss.
        """
        n, d = Z.shape
        C = Z.T @ Z / (n - 1)
        L, _ = torch.linalg.eigh(C)
        uni = -2 * torch.clamp(L, min=1e-8).sqrt().sum()
        return uni

    def _compute_loss(self, z1: Tensor, z2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Computes the SSGE loss.

        Args:
            z1 (torch.Tensor): First view embeddings.
            z2 (torch.Tensor): Second view embeddings.

        Returns:
            Tuple of (total_loss, invariance_term, uniformity_term).
        """
        # Batch normalization (column-wise z-score)
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-12)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-12)

        inv = -(z1 * z2).sum() / z1.shape[0]  # invariance loss

        uni = 0.5 * (self._uniformity(z1) + self._uniformity(z2))  # uniformity loss

        loss = inv + self.lam * uni
        return loss, inv, uni

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""
        Computes the SSGE loss with multiple components.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z1, z2 = self(x, edge_index, **kwargs)
        loss, inv, uni = self._compute_loss(z1, z2)

        return LossOutput(
            total=loss,
            components={
                'inv': inv.item(),
                'uni': uni.item()
            }
        )

    def loss_batch(self, batch: Data) -> LossOutput:
        r"""
        Computes loss for a mini-batch with seed node slicing.

        Args:
            batch (Data): A mini-batch from the loader.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z1, z2 = self(batch.x, batch.edge_index)
        z1 = z1[:batch.batch_size]
        z2 = z2[:batch.batch_size]
        loss, inv, uni = self._compute_loss(z1, z2)

        return LossOutput(
            total=loss,
            components={
                'inv': inv.item(),
                'uni': uni.item()
            }
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(encoder={self.encoder})"
