"""Low-Rank Graph Auto-Encoder (LRGAE) model.

LRGAE learns node embeddings by reconstructing a **PSD low-rank approximation**
of the normalised adjacency matrix.  Instead of raw binary edges (GAE) or an
SVD-based approximation (which fails for indefinite matrices), we compute::

    Â = D^{-1/2} A D^{-1/2}
    Â ≈ Q_+ Λ_+ Q_+^⊤          (only positive eigenvalues)
    T  = Q_+ √Λ_+

and then minimise::

    MSE( z_i·z_j / √d ,  t_i·t_j )

for every supervised edge (positive and negative).  The inner-product decoder
z_i·z_j can represent any PSD Gram matrix, so the target is chosen to be PSD
as well — this makes the optimisation well-posed.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.utils import (
    negative_sampling,
    to_scipy_sparse_matrix,
    to_undirected,
)

EPS = 1e-12


# ============================================================================
# 1. Normalised adjacency
# ============================================================================


def _build_normalised_adj(
    edge_index: Tensor, num_nodes: int, add_self_loops: bool = False
) -> "scipy.sparse.spmatrix":
    """Build Â = D^{-1/2} A D^{-1/2} (optionally with self-loops).

    Edges are made undirected and binarised so the eigendecomposition
    target is well-defined.
    """
    import scipy.sparse as sp

    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    # binarise — remove duplicate edge weights
    adj.data[:] = 1.0
    adj.eliminate_zeros()

    if add_self_loops:
        adj = adj + sp.eye(num_nodes, dtype=adj.dtype, format="csr")

    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg, dtype=np.float64)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    return D_inv_sqrt @ adj @ D_inv_sqrt


# ============================================================================
# 2. PSD low-rank target computation
# ============================================================================


def compute_low_rank_targets(
    edge_index: Tensor,
    num_nodes: int,
    rank: int,
    *,
    add_self_loops: bool = False,
    logger=None,
) -> Tensor:
    """Compute PSD low-rank target embeddings for LRGAE.

    Uses :func:`scipy.sparse.linalg.eigsh` to find the *largest* eigenvalues
    of the normalised adjacency Â.  Only **positive** eigenvalues are kept.
    Target:  T = Q_+  diag(√λ_+)   so that  T[i]·T[j] = (Â_psd,k)[i,j].

    Args:
        edge_index: PyG edge_index (2, E).
        num_nodes: Total number of nodes.
        rank: Maximum number of eigencomponents to keep.
        add_self_loops: Add I before normalisation (GCN-style Â).
        logger: Optional logger.

    Returns:
        T of shape (num_nodes, effective_rank), float32, on CPU.
    """
    try:
        from scipy.sparse.linalg import eigsh
    except ImportError:
        raise ImportError(
            "lrGAE requires scipy for sparse eigendecomposition.  "
            "Install with: pip install scipy"
        )

    if num_nodes < 2:
        raise ValueError(f"lrGAE requires at least 2 nodes, got {num_nodes}.")

    k_eff = min(int(rank), num_nodes - 1)
    if k_eff < 1:
        raise ValueError(f"Invalid effective rank {k_eff} for num_nodes={num_nodes}.")

    if logger is not None:
        msg = "lrGAE: building normalised adjacency"
        if add_self_loops:
            msg += " (with self-loops)"
        logger.info("%s (%d nodes) …", msg, num_nodes)

    adj_norm = _build_normalised_adj(
        edge_index, num_nodes, add_self_loops=add_self_loops
    ).astype(np.float64)

    if logger is not None:
        logger.info("lrGAE: computing top-%d eigenvalues …", k_eff)

    eigenvalues, eigenvectors = eigsh(adj_norm, k=k_eff, which="LA")

    # eigsh returns ASCENDING — reverse to descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Keep only positive eigenvalues (PSD decoder requirement)
    pos_mask = eigenvalues > 1e-10
    eigenvalues = eigenvalues[pos_mask]
    eigenvectors = eigenvectors[:, pos_mask]

    if eigenvalues.size == 0:
        raise RuntimeError(
            "No positive eigenvalues found in the normalised adjacency.  "
            "Try --lrgae-add-self-loops or reduce the embedding dimension."
        )

    # T = Q_+  √Λ_+
    target = eigenvectors * np.sqrt(eigenvalues)[None, :]

    # Frobenius energy captured
    total_energy = float(adj_norm.multiply(adj_norm).sum())
    captured = float(np.sum(eigenvalues ** 2))
    energy_pct = captured / max(total_energy, EPS)

    if logger is not None:
        logger.info(
            "lrGAE PSD target: requested_rank=%d  effective_rank=%d  "
            "λ_max=%.6f  λ_min_pos=%.6f  Frobenius_energy=%.4f",
            rank,
            target.shape[1],
            float(eigenvalues[0]),
            float(eigenvalues[-1]),
            energy_pct,
        )

    return torch.from_numpy(target.astype(np.float32))


# ============================================================================
# 3. LRGAE model
# ============================================================================


class LRGAE(nn.Module):
    r"""Low-Rank Graph Auto-Encoder with PSD spectral targets.

    Encoder:  Z = GCN(X, A)
    Decoder:  p_ij = z_i·z_j / √d
    Target:   y_ij = t_i·t_j    (PSD eigendecomposition of Â)

    Loss (per edge, pos & neg):  MSE(p_ij, y_ij)

    .. important::
        ``target_embeddings_cpu`` is a **plain attribute** (not a PyTorch
        buffer).  It stays on CPU regardless of ``model.to(device)`` calls.
        This is intentional — for large graphs (e.g. ogbn-products) the
        target matrix can be 5+ GB, and only the needed rows are moved to
        GPU on-the-fly in :meth:`_lookup_targets`.

    Args:
        encoder: GCN encoder.
        target_embeddings: Precomputed T (N, rank).  May be ``None`` and
            set later via :meth:`set_targets`.
        neg_ratio: Negative / positive edge ratio for sampling.
    """

    def __init__(
        self,
        encoder: nn.Module,
        target_embeddings: Optional[Tensor] = None,
        neg_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.neg_ratio = float(neg_ratio)
        # Plain CPU attribute — NEVER moved by model.to(device)
        self.target_embeddings_cpu: Optional[Tensor] = None
        if target_embeddings is not None:
            self.set_targets(target_embeddings)

    # -- properties ----------------------------------------------------------

    @property
    def rank(self) -> int:
        if self.target_embeddings_cpu is None:
            return 0
        return self.target_embeddings_cpu.size(1)

    @property
    def has_targets(self) -> bool:
        return self.target_embeddings_cpu is not None

    def set_targets(self, targets: Tensor) -> None:
        """Store targets as a CPU attribute (never moved to GPU)."""
        if targets.dim() != 2:
            raise ValueError(
                f"targets must have shape [num_nodes, rank], got {targets.shape}"
            )
        self.target_embeddings_cpu = (
            targets.detach().to(device="cpu", dtype=torch.float32).contiguous()
        )

    def reset_parameters(self) -> None:
        reset(self.encoder)

    # -- encode / decode -----------------------------------------------------

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Raw dot-product scores, scaled by 1/√d.  Returns (E,)."""
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1) / math.sqrt(z.size(-1))

    # -- target lookup (memory-efficient) ------------------------------------

    def _lookup_targets(
        self,
        z: Tensor,
        edge_index: Tensor,
        node_map: Optional[Tensor] = None,
    ) -> Tensor:
        """Look up low-rank target weights for edges.

        Only the *needed rows* of ``target_embeddings_cpu`` are moved to
        GPU — never the full (N × rank) matrix.

        Args:
            z: Node embeddings (for device/dtype info).
            edge_index: (2, E) edge indices.  Must be **global** indices
                if *node_map* is None; local indices otherwise.
            node_map: Maps local → global indices (``batch.n_id``).

        Returns:
            Target weights (E,), same device/dtype as *z*.
        """
        if self.target_embeddings_cpu is None:
            raise RuntimeError(
                "LRGAE targets are not initialised.  "
                "Call set_targets() or provide them at construction time."
            )

        src, dst = edge_index
        if node_map is not None:
            node_map_cpu = node_map.detach().cpu()
            src = node_map_cpu[src.detach().cpu()]
            dst = node_map_cpu[dst.detach().cpu()]
        else:
            src = src.detach().cpu()
            dst = dst.detach().cpu()

        t = self.target_embeddings_cpu
        t_src = t.index_select(0, src).to(
            device=z.device, dtype=z.dtype, non_blocking=True,
        )
        t_dst = t.index_select(0, dst).to(
            device=z.device, dtype=z.dtype, non_blocking=True,
        )
        return (t_src * t_dst).sum(dim=-1)

    # -- reconstruction loss -------------------------------------------------

    def recon_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Optional[Tensor] = None,
        node_map: Optional[Tensor] = None,
        *,
        negative_exclusion_edge_index: Optional[Tensor] = None,
    ) -> Tensor:
        r"""MSE reconstruction loss against low-rank targets.

        Both positive AND negative edges use soft low-rank targets.

        Args:
            z: Node embeddings (N_sub, D).
            pos_edge_index: Positive edges (2, E_pos).
            neg_edge_index: Pre-sampled negative edges (2, E_neg).
                If None, sampled from the local complement.
            node_map: Global node indices for subgraph mapping.
            negative_exclusion_edge_index: Edges to exclude from negative
                sampling (e.g. all local true edges in the subgraph).
                Defaults to *pos_edge_index*.

        Returns:
            Scalar loss.
        """
        if pos_edge_index.numel() == 0:
            raise RuntimeError(
                "LRGAE received a batch with no positive supervision edges."
            )

        pos_pred = self.decode(z, pos_edge_index)
        pos_target = self._lookup_targets(z, pos_edge_index, node_map)
        pos_loss = F.mse_loss(pos_pred, pos_target)

        if neg_edge_index is None:
            exclusion = (
                negative_exclusion_edge_index
                if negative_exclusion_edge_index is not None
                else pos_edge_index
            )
            neg_edge_index = _sample_neg_edges_local(
                z.size(0), pos_edge_index.size(1), z.device,
                exclusion, neg_ratio=self.neg_ratio,
            )

        if neg_edge_index.numel() == 0:
            return pos_loss

        neg_pred = self.decode(z, neg_edge_index)
        neg_target = self._lookup_targets(z, neg_edge_index, node_map)
        neg_loss = F.mse_loss(neg_pred, neg_target)

        return pos_loss + neg_loss

    # -- forward / loss ------------------------------------------------------

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encode(x, edge_index)

    def loss(
        self, x: Tensor, edge_index: Tensor, **kwargs
    ) -> "LossOutput":  # noqa: F821
        from pyagc.models.base import LossOutput

        z = self.forward(x, edge_index)
        recon = self.recon_loss(z, edge_index)
        return LossOutput(total=recon, components={"recon": recon.item()})

    def loss_batch(self, batch: Data) -> "LossOutput":  # noqa: F821
        """Mini-batch loss.

        LinkNeighborLoader (preferred):
            Uses ``batch.edge_label_index`` / ``batch.edge_label`` to
            separate positive and negative supervision edges.

        NeighborLoader (fallback, biased):
            Supervises edges whose **source** is a seed node.  Local
            negative sampling excludes all subgraph edges but may still
            include global true edges as false negatives.
        """
        from pyagc.models.base import LossOutput

        z = self.forward(batch.x, batch.edge_index)
        node_map = getattr(batch, "n_id", None)

        edge_label_index = getattr(batch, "edge_label_index", None)

        if edge_label_index is not None:
            # ---- LinkNeighborLoader ----
            edge_label = getattr(batch, "edge_label", None)
            if edge_label is not None:
                pos_mask = edge_label > 0
                neg_mask = edge_label == 0
                pos_edges = edge_label_index[:, pos_mask]
                neg_edges = (
                    edge_label_index[:, neg_mask]
                    if bool(neg_mask.any().item()) else None
                )
            else:
                pos_edges = edge_label_index
                neg_edges = None

            recon = self.recon_loss(
                z, pos_edges,
                neg_edge_index=neg_edges,
                node_map=node_map,
                negative_exclusion_edge_index=batch.edge_index,
            )
        else:
            # ---- NeighborLoader (approximate) ----
            n_seed = int(batch.batch_size)
            ei = batch.edge_index
            # Supervise edges whose source is a seed node.
            seed_mask = ei[0] < n_seed
            pos_edges = ei[:, seed_mask]

            recon = self.recon_loss(
                z, pos_edges,
                neg_edge_index=None,
                node_map=node_map,
                negative_exclusion_edge_index=batch.edge_index,
            )

        return LossOutput(total=recon, components={"recon": recon.item()})

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(encoder={self.encoder}, "
            f"rank={self.rank}, neg_ratio={self.neg_ratio:.2f})"
        )


# ============================================================================
# 4. Local negative sampling helper
# ============================================================================


def _sample_neg_edges_local(
    num_nodes: int,
    num_pos: int,
    device: torch.device,
    pos_edge_index: Tensor,
    neg_ratio: float = 1.0,
) -> Tensor:
    """Sample negative edges from the local complement.

    NOTE: In mini-batch (NeighborLoader) mode this only excludes edges
    present in *pos_edge_index* — other true global edges may be sampled
    as negatives.  For unbiased training use LinkNeighborLoader.
    """
    num_neg = int(num_pos * neg_ratio)
    if num_neg <= 0:
        return pos_edge_index.new_empty((2, 0))
    return negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method="sparse",
    )
