"""BES: Bootstrapped Embedding Selection for Graph Self-Supervised Learning.

Reference:
  "Bootstrapped Embedding Selection for Graph Representation Learning", ICML 2026.

Key components:
  - BESEncoder: Learnable GCN encoder (replaces original frozen dual-encoder + attention)
  - detect_boundary_nodes(): Mahalanobis-based boundary node detection
  - compute_repulsion_loss(): InfoNCE-style gravitational repulsion for boundary nodes
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


# ============================================================================
# 1. BES Encoder
# ============================================================================


class BESEncoder(nn.Module):
    """Learnable GCN encoder for BES.

    Architecture:
        num_layers × (GCNConv → BatchNorm1d → ReLU → Dropout)

    This replaces the original dual frozen encoder + MultiHeadSelfAttention stack.
    The core BES contribution (boundary detection + repulsion) is orthogonal to
    the encoder choice; a standard GCN is dataset-agnostic and sufficient.

    Args:
        in_dim: Input feature dimension.
        hidden_dim: Hidden/output embedding dimension.
        num_layers: Number of GCNConv layers (default 2).
        dropout: Dropout probability (default 0.5).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        for i in range(num_layers):
            in_c = in_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_c, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Forward pass returning [N, hidden_dim] embeddings."""
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.drop(x)
        return x


# ============================================================================
# 2. Boundary Detection (standalone function)
# ============================================================================


def detect_boundary_nodes(
    h: Tensor,
    labels: Tensor,
    train_mask: Tensor,
    num_classes: int,
    delta: float = 5.0,
) -> Tuple[Tensor, Tensor]:
    """Identify boundary nodes using Mahalanobis slab distance.

    Algorithm (faithful to original BES train_eval.py):
      1. Extract training node embeddings: h_train = h[train_mask]
      2. Compute per-class centroids (mean embedding per class)
      3. Compute global covariance matrix Sigma on h_train (with regularization)
      4. For each training node i with label m:
         For each other class n ≠ m:
           slab = | (μ_m - μ_n)ᵀ Σ⁻¹ h_i |
           If slab ≤ delta → boundary candidate
      5. Return boundary indices and class centroids

    Args:
        h: [N, D] all node embeddings.
        labels: [N] node labels (long).
        train_mask: [N] boolean mask for training nodes.
        num_classes: Total number of classes.
        delta: Mahalanobis threshold (default 5.0).

    Returns:
        boundary_indices: LongTensor of candidate boundary node indices (may be empty).
        class_centroids: [num_classes, D] per-class mean embeddings.
    """
    device = h.device
    train_h = h[train_mask]                     # [N_train, D]
    train_labels = labels[train_mask]           # [N_train]
    train_indices = train_mask.nonzero(as_tuple=False).squeeze(-1)  # [N_train] original indices

    # --- Class centroids ---
    class_centroids_list = []
    for c in range(num_classes):
        c_mask = (train_labels == c)
        if c_mask.sum() > 0:
            class_centroids_list.append(train_h[c_mask].mean(dim=0))
        else:
            class_centroids_list.append(torch.zeros(h.size(1), device=device))
    class_centroids = torch.stack(class_centroids_list)  # [num_classes, D]

    # --- Covariance matrix (regularized) ---
    centered = train_h - train_h.mean(dim=0, keepdim=True)
    n_train = train_h.size(0)
    Sigma = (centered.t() @ centered) / (n_train - 1 + 1e-8)
    Sigma = Sigma + 1e-4 * torch.eye(Sigma.size(0), device=device)

    # Stable inverse (fallback to pseudo-inverse)
    try:
        Sigma_inv = torch.linalg.inv(Sigma)
    except RuntimeError:
        Sigma_inv = torch.linalg.pinv(Sigma)

    # --- Boundary detection (vectorized) ---
    # Project all training embeddings: proj[i] = Sigma_inv @ h_i  [N_train, D]
    proj = (Sigma_inv @ train_h.t()).t()  # [N_train, D]

    # For each node i with class m, slab[i, n] = |(μ_m - μ_n)ᵀ Σ⁻¹ h_i|
    # = |μ_mᵀ Σ⁻¹ h_i - μ_nᵀ Σ⁻¹ h_i| = |dot(train_centroids[i], proj[i]) - dot(centroids[n], proj[i])|
    train_centroids_h = class_centroids[train_labels]  # [N_train, D]
    dot_own = (train_centroids_h * proj).sum(dim=1)    # [N_train]
    dot_all = proj @ class_centroids.t()                # [N_train, C]
    slab = torch.abs(dot_own.unsqueeze(1) - dot_all)   # [N_train, C]

    # Mask out own class
    own_mask = torch.zeros(train_h.size(0), num_classes, dtype=torch.bool, device=device)
    own_mask[torch.arange(train_h.size(0), device=device), train_labels] = True
    slab.masked_fill_(own_mask, float('inf'))

    # Boundary condition: any n ≠ m has slab <= delta
    boundary_mask = (slab <= delta).any(dim=1)  # [N_train]
    boundary_indices = train_indices[boundary_mask]

    return boundary_indices, class_centroids


# ============================================================================
# 3. Repulsion Loss (standalone function)
# ============================================================================


def compute_repulsion_loss(
    h: Tensor,
    boundary_indices: Tensor,
    labels: Tensor,
    class_centroids: Tensor,
    num_classes: int,
    tau: float = 1.0,
    beta_size: int = 256,
) -> Tensor:
    """Gravitational repulsion loss for boundary nodes.

    Algorithm (faithful to original BES train_eval.py):
      1. Among boundary candidates, find k=5 nearest neighbors in embedding space.
      2. A node is "active" if >50% of its k-NN have a DIFFERENT class label.
         This filters for nodes truly straddling decision boundaries.
      3. Randomly sample up to beta_size active boundary nodes.
      4. InfoNCE-style loss with class centroids as anchors:
         - Positive: exp(-max(0, dist_to_own_centroid - min_dist_to_other)^2 / tau)
         - Negative: Σ_{c≠m} exp(-dist_to_centroid_c^2 / tau)
         - L = -log(pos / (pos + neg_sum))

    Args:
        h: [N, D] all node embeddings.
        boundary_indices: [B] candidate boundary node indices.
        labels: [N] node labels (long).
        class_centroids: [num_classes, D] pre-computed centroids.
        num_classes: Total number of classes.
        tau: Temperature for InfoNCE loss (default 1.0).
        beta_size: Max boundary nodes per loss computation (default 256).

    Returns:
        Scalar repulsion loss. Returns 0.0 (with grad) if no active boundary nodes.
    """
    if boundary_indices.numel() < 2:
        return torch.tensor(0.0, device=h.device, requires_grad=True)

    device = h.device

    # --- k-NN filtering: keep nodes whose neighbors have mixed labels ---
    B_Phi = h[boundary_indices]   # [B, D]
    B_labels = labels[boundary_indices]  # [B]

    dist_B = torch.cdist(B_Phi, B_Phi)  # [B, B]
    k_nn = min(5, B_Phi.size(0) - 1)
    _, knn_idx = torch.topk(dist_B, k=k_nn + 1, dim=1, largest=False)
    knn_idx = knn_idx[:, 1:]  # exclude self (index 0)

    knn_labels = B_labels[knn_idx]  # [B, k]
    mismatch = (knn_labels != B_labels.unsqueeze(1)).float()  # [B, k]
    S_v = mismatch.mean(dim=1)  # [B] shift score

    boundary_nodes = boundary_indices[S_v > 0.5]  # active boundary nodes

    if boundary_nodes.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # --- Sample up to beta_size active boundary nodes ---
    if boundary_nodes.size(0) > beta_size:
        perm = torch.randperm(boundary_nodes.size(0), device=device)[:beta_size]
        batch_idx = boundary_nodes[perm]
    else:
        batch_idx = boundary_nodes

    Z_batch = h[batch_idx]                  # [beta, D]
    y_batch = labels[batch_idx]             # [beta]

    # --- Repulsion loss (InfoNCE with class centroids as anchors) ---
    dist_to_pos = torch.norm(Z_batch - class_centroids[y_batch], dim=1)  # [beta]
    dist_to_all = torch.cdist(Z_batch, class_centroids)                  # [beta, num_classes]

    # Mask out own class for negative computation
    neg_mask = torch.ones((Z_batch.size(0), num_classes), dtype=torch.bool, device=device)
    neg_mask[torch.arange(Z_batch.size(0), device=device), y_batch] = False

    dist_to_all_masked = dist_to_all.clone()
    dist_to_all_masked[~neg_mask] = float('inf')
    min_dist_to_neg, _ = torch.min(dist_to_all_masked, dim=1)  # [beta]

    # Positive: margin-based similarity (node closer to own centroid than others → high sim)
    sim_pos = -torch.clamp(dist_to_pos - min_dist_to_neg, min=0.0).pow(2)  # [beta]
    # Negative: squared distance to all centroids
    sim_neg = -dist_to_all.pow(2)  # [beta, num_classes]

    numerator = torch.exp(sim_pos / tau)  # [beta]
    exp_sim_neg = torch.exp(sim_neg / tau) * neg_mask.float()  # [beta, num_classes]
    denominator = numerator + exp_sim_neg.sum(dim=1)  # [beta]

    gravity_loss = -torch.log(numerator / (denominator + 1e-8)).mean()

    return gravity_loss
