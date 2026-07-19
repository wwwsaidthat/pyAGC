"""Paper-faithful components for Boundary Embedding Shaping (BES).

Reference: Chen et al., "Boundary Embedding Shaping with Adaptive Contrastive
Learning for Graph Structural Disentanglement", ICML 2026.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch_geometric.nn import GCNConv, SAGEConv


class BESGraphEncoder(nn.Module):
    """One pretrained graph view used by the BES plug-in."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, kind: str) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers 必须 >= 1")
        conv_cls = GCNConv if kind == "gcn" else SAGEConv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            in_channels = in_dim if layer == 0 else hidden_dim
            self.convs.append(conv_cls(in_channels, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = float(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = F.relu(norm(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class BESMultiViewBackbone(nn.Module):
    """GCN + GraphSAGE views, replacing dataset-specific pretrained pairs.

    The paper uses two pretrained encoders selected per dataset.  GCN and
    GraphSAGE provide a reproducible dataset-independent pair for OGB datasets.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gcn = BESGraphEncoder(in_dim, hidden_dim, num_layers, dropout, "gcn")
        self.sage = BESGraphEncoder(in_dim, hidden_dim, num_layers, dropout, "sage")

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return torch.stack((self.gcn(x, edge_index), self.sage(x, edge_index)), dim=1)


class BoundaryAttentionLayer(nn.Module):
    """Multi-head attention over encoder views with the paper's residual path."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} 必须能被 bes_num_heads={num_heads} 整除")
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, views: Tensor) -> Tensor:
        shaped, _ = self.attention(views, views, views, need_weights=False)
        return views + shaped


class BoundaryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 2, num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            BoundaryAttentionLayer(hidden_dim, num_heads) for _ in range(num_layers)
        )

    def forward_views(self, views: Tensor, num_layers: Optional[int] = None) -> Tensor:
        limit = len(self.layers) if num_layers is None else int(num_layers)
        for layer in self.layers[:limit]:
            views = layer(views)
        return views

    def forward(self, views: Tensor, num_layers: Optional[int] = None) -> Tensor:
        return self.forward_views(views, num_layers=num_layers).mean(dim=1)


@dataclass
class BESBoundaryState:
    nodes: Tensor
    centroids: Tensor


@torch.no_grad()
def detect_boundary_nodes(
    h: Tensor,
    labels: Tensor,
    train_indices: Tensor,
    num_classes: int,
    delta: float = 5.0,
    knn_k: int = 5,
    covariance_reg: float = 1e-4,
    max_candidates: int = 0,
) -> BESBoundaryState:
    """Paper Eq. (4-5): global Mahalanobis slab followed by k-NN shift score.

    Statistics use training nodes only.  k-NN is executed with sklearn so the
    implementation does not materialize the former B x B tensor on the GPU.
    ``max_candidates=0`` is exact; a positive value enables an explicitly
    bounded large-graph approximation.
    """
    device = h.device
    train_indices = train_indices.to(device=device, dtype=torch.long)
    train_h = h[train_indices]
    train_y = labels.to(device)[train_indices].long()
    centroids = h.new_zeros((num_classes, h.size(1)))
    present: List[int] = []
    for class_id in range(num_classes):
        mask = train_y == class_id
        if bool(mask.any()):
            centroids[class_id] = train_h[mask].mean(dim=0)
            present.append(class_id)
    if len(present) < 2 or train_h.size(0) < 2:
        return BESBoundaryState(train_indices.new_empty(0), centroids)

    centered = train_h - train_h.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(train_h.size(0) - 1, 1)
    covariance.diagonal().add_(float(covariance_reg))
    try:
        covariance_inv = torch.linalg.inv(covariance)
    except RuntimeError:
        covariance_inv = torch.linalg.pinv(covariance)

    # Keep memory O(N + D^2): evaluate each present class pair independently.
    in_slab = torch.zeros(train_h.size(0), dtype=torch.bool, device=device)
    for class_id in present:
        own = (train_y == class_id).nonzero(as_tuple=False).flatten()
        if own.numel() == 0:
            continue
        own_h = train_h[own]
        for other_id in present:
            if other_id == class_id:
                continue
            direction = (centroids[class_id] - centroids[other_id]) @ covariance_inv
            slab = (own_h * direction).sum(dim=1).abs()
            in_slab[own] |= slab <= float(delta)

    candidates = train_indices[in_slab]
    if candidates.numel() <= 1:
        return BESBoundaryState(train_indices.new_empty(0), centroids)
    if max_candidates > 0 and candidates.numel() > max_candidates:
        perm = torch.randperm(candidates.numel(), device=device)[:max_candidates]
        candidates = candidates[perm]

    candidate_h = h[candidates].detach().float().cpu().numpy()
    candidate_y = labels[candidates.cpu()].detach().cpu().numpy() if labels.device.type == "cpu" else labels[candidates].detach().cpu().numpy()
    k = min(int(knn_k), candidates.numel() - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    neighbor_ids = nbrs.fit(candidate_h).kneighbors(
        candidate_h, return_distance=False
    )[:, 1:]
    mismatch = (candidate_y[neighbor_ids] != candidate_y[:, None]).mean(axis=1)
    active = torch.from_numpy(np.flatnonzero(mismatch > 0.5)).to(device)
    return BESBoundaryState(candidates[active], centroids)


def compute_repulsion_loss(
    h: Tensor,
    boundary_nodes: Tensor,
    labels: Tensor,
    class_centroids: Tensor,
    tau: float = 1.0,
    beta_size: int = 256,
) -> Tensor:
    """Official center-based gravity loss (paper Eq. 9-10)."""
    if tau <= 0:
        raise ValueError("bes_tau 必须 > 0")
    if boundary_nodes.numel() == 0:
        return h.sum() * 0.0
    if boundary_nodes.numel() > beta_size:
        perm = torch.randperm(boundary_nodes.numel(), device=h.device)[:beta_size]
        boundary_nodes = boundary_nodes[perm]

    z = h[boundary_nodes]
    y = labels.to(h.device)[boundary_nodes].long()
    return compute_gravity_loss(z, y, class_centroids, tau)


def compute_gravity_loss(
    z: Tensor,
    labels: Tensor,
    class_centroids: Tensor,
    tau: float = 1.0,
) -> Tensor:
    """Gravity loss for an already sampled boundary-node batch."""
    if tau <= 0:
        raise ValueError("bes_tau 必须 > 0")
    if z.numel() == 0:
        return z.sum() * 0.0
    y = labels.to(z.device).long()
    centers = class_centroids.to(z.device).detach()
    dist_pos = torch.norm(z - centers[y], dim=1)
    dist_all = torch.cdist(z, centers)
    negative_mask = torch.ones_like(dist_all, dtype=torch.bool)
    negative_mask[torch.arange(z.size(0), device=z.device), y] = False
    min_negative = dist_all.masked_fill(~negative_mask, float("inf")).min(dim=1).values
    sim_pos = -torch.clamp(dist_pos - min_negative, min=0.0).square()
    sim_neg = -dist_all.square()
    numerator = torch.exp(sim_pos / float(tau))
    denominator = numerator + (torch.exp(sim_neg / float(tau)) * negative_mask).sum(dim=1)
    return -torch.log(numerator / denominator.clamp_min(1e-8)).mean()
