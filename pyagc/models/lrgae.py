"""Low-Rank Graph Auto-Encoder (LRGAE) model.

LRGAE extends GAE by reconstructing a low-rank denoised version of the adjacency
matrix instead of the raw binary adjacency.  The low-rank target is obtained via
truncated SVD of the normalised adjacency Â = D^{-1/2} A D^{-1/2}.

Key differences from standard GAE:
  - **Soft, continuous targets** from the low-rank approximation instead of
    hard 0/1 labels.
  - **MSE loss** instead of BCE — the model learns to match the low-rank
    structure of the graph.
  - The low-rank approximation acts as a denoising mechanism: high-frequency
    noise edges are filtered out, giving a cleaner training signal.

Reference:
  Salha et al., "Simple and Effective Graph Autoencoders", 2019.
  (Extended here with learnable GCN encoder + low-rank denoising.)
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling, to_scipy_sparse_matrix

EPS = 1e-12


def _build_normalised_adj(edge_index: Tensor, num_nodes: int) -> "scipy.sparse.spmatrix":
    """Build the normalised adjacency matrix Â = D^{-1/2} A D^{-1/2}.

    Args:
        edge_index: PyG edge_index of shape (2, E).
        num_nodes: Total number of nodes.

    Returns:
        Sparse scipy CSR matrix of shape (num_nodes, num_nodes).
    """
    import scipy.sparse as sp

    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    # Make symmetric (undirected) — some graphs may have directed edges
    adj = adj.maximum(adj.T)

    # D^{-1/2}
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    return D_inv_sqrt @ adj @ D_inv_sqrt


def compute_low_rank_targets(
    edge_index: Tensor,
    num_nodes: int,
    rank: int,
    logger=None,
) -> Tensor:
    """Compute low-rank target embeddings via truncated SVD of Â.

    Â ≈ U @ diag(S) @ V^T

    For undirected graphs Â is symmetric, so U ≈ V.  We store
    T = U @ diag(sqrt(|S|))  of shape (num_nodes, rank).

    The low-rank approximation of Â[i,j] is then T[i] · T[j].

    Args:
        edge_index: PyG edge_index (2, E).
        num_nodes: Total number of nodes.
        rank: Truncated SVD rank (k).
        logger: Optional logger for progress messages.

    Returns:
        Target embeddings T of shape (num_nodes, rank), float32.
    """
    try:
        from scipy.sparse.linalg import svds
    except ImportError:
        raise ImportError("lrGAE requires scipy for sparse SVD.  Install with: pip install scipy")

    if logger is not None:
        logger.info("lrGAE: building normalised adjacency (%d nodes) …", num_nodes)

    adj_norm = _build_normalised_adj(edge_index, num_nodes)

    if logger is not None:
        logger.info("lrGAE: computing rank-%d SVD …", rank)

    # svds returns singular vectors sorted by decreasing singular value
    k_eff = min(rank, num_nodes - 2, adj_norm.shape[0] - 2)
    if k_eff < rank and logger is not None:
        logger.warning("lrGAE: requested rank %d but only %d feasible — clamping.", rank, k_eff)

    try:
        U, S, Vt = svds(adj_norm, k=k_eff, which='LM', return_singular_vectors='u')
    except Exception as exc:
        if logger is not None:
            logger.warning("lrGAE: scipy svds failed (%s), falling back to sklearn TruncatedSVD.", exc)
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=k_eff, random_state=0)
        svd.fit(adj_norm)
        U = svd.components_.T  # TruncatedSVD returns V^T, so U = components_.T
        S = svd.singular_values_
        Vt = svd.components_

    # svds returns in ascending order; TruncatedSVD returns descending.
    # Normalise: make sure singular values are descending.
    if S[0] < S[-1]:
        S = S[::-1]
        U = U[:, ::-1]

    # Target embeddings: T = U @ diag(sqrt(abs(S)))
    # Use abs(S) for numerical stability (S should be non-negative for PSD Â)
    S_sqrt = np.sqrt(np.maximum(S, 0.0))
    T = U * S_sqrt[None, :]  # broadcasting: (N, k) * (k,) → (N, k)

    if logger is not None:
        explained = float(np.sum(S)) / float(np.sum(adj_norm.diagonal()))
        logger.info(
            "lrGAE: SVD done — rank=%d, top singular value=%.4f, explained var≈%.4f",
            k_eff, float(S[0]), min(explained, 1.0),
        )

    return torch.from_numpy(T.astype(np.float32))


def _sample_neg_edges(
    num_nodes: int,
    num_pos: int,
    device: torch.device,
    pos_edge_index: Tensor,
    neg_ratio: float = 1.0,
) -> Tensor:
    """Sample negative edges using PyG's :func:`negative_sampling`."""
    num_neg = max(1, int(num_pos * neg_ratio))
    return negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method="sparse",
    )


class LRGAE(nn.Module):
    r"""Low-Rank Graph Auto-Encoder.

    Learns node embeddings Z = GCN(X, A) such that the inner-product decoder
    matches a low-rank, denoised version of the adjacency matrix:

    .. math::
        \hat{A}_k[i,j] = t_i^\top t_j
        \quad\text{where}\quad
        A \approx U \Sigma V^\top,\; T = U \Sigma^{1/2}

    Loss (per edge):

    .. math::
        \mathcal{L}_{pos} = \text{MSE}\!\left(\frac{z_i^\top z_j}{\sqrt{d}},\; t_i^\top t_j\right)

    Args:
        encoder: GCN encoder that produces node embeddings Z.
        target_embeddings: Precomputed T of shape (N, rank) — the low-rank
            target embeddings from :func:`compute_low_rank_targets`.
        neg_ratio: Fraction of positive edges to sample as negatives (default 1.0).
    """

    def __init__(
        self,
        encoder: nn.Module,
        target_embeddings: Tensor,
        neg_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.register_buffer("target_embeddings", target_embeddings)
        self.neg_ratio = float(neg_ratio)

    @property
    def rank(self) -> int:
        return self.target_embeddings.size(1)

    def reset_parameters(self) -> None:
        reset(self.encoder)

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Raw inner-product scores (no sigmoid) for edge_index.

        Returns shape (E,).
        """
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1) / math.sqrt(z.size(-1))

    def _lookup_targets(
        self, z: Tensor, edge_index: Tensor, node_map: Optional[Tensor] = None
    ) -> Tensor:
        """Look up low-rank target weights for the given edges.

        Args:
            z: Node embeddings, used only for device/dtype info.
            edge_index: (2, E) edge indices (local to subgraph if node_map given).
            node_map: If provided, maps local indices → global indices (e.g.
                ``batch.n_id`` from NeighborLoader).

        Returns:
            Target weights of shape (E,), same device/dtype as z.
        """
        src, dst = edge_index
        if node_map is not None:
            src = node_map[src]
            dst = node_map[dst]
        t = self.target_embeddings.to(z.device)
        return (t[src] * t[dst]).sum(dim=-1).to(dtype=z.dtype)

    def recon_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Optional[Tensor] = None,
        node_map: Optional[Tensor] = None,
    ) -> Tensor:
        r"""MSE reconstruction loss against low-rank targets.

        Args:
            z: Node embeddings (N, D).
            pos_edge_index: Positive edges (2, E_pos).
            neg_edge_index: Negative edges (2, E_neg).  Sampled if None.
            node_map: Global node indices for subgraph mapping.

        Returns:
            Scalar loss.
        """
        # Positive loss
        pos_pred = self.decode(z, pos_edge_index)
        pos_target = self._lookup_targets(z, pos_edge_index, node_map)
        pos_loss = ((pos_pred - pos_target) ** 2).mean()

        # Negative loss — target is 0 for negative edges
        if neg_edge_index is None:
            neg_edge_index = _sample_neg_edges(
                z.size(0), pos_edge_index.size(1), z.device,
                pos_edge_index, neg_ratio=self.neg_ratio,
            )
        neg_pred = self.decode(z, neg_edge_index)
        neg_loss = (neg_pred ** 2).mean()

        return pos_loss + neg_loss

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encode(x, edge_index)

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> "LossOutput":
        """Full-graph training loss."""
        from pyagc.models.base import LossOutput
        z = self.forward(x, edge_index)
        recon = self.recon_loss(z, edge_index)
        return LossOutput(total=recon, components={'recon': recon.item()})

    def loss_batch(self, batch: Data) -> "LossOutput":
        """Mini-batch training loss (NeighborLoader)."""
        from pyagc.models.base import LossOutput
        z = self.forward(batch.x, batch.edge_index)
        node_map = getattr(batch, 'n_id', None)
        recon = self.recon_loss(z, batch.edge_index, node_map=node_map)
        return LossOutput(total=recon, components={'recon': recon.item()})

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(encoder={self.encoder}, "
            f"rank={self.rank}, neg_ratio={self.neg_ratio:.2f})"
        )
