"""PaGCL: Graph Contrastive Learning with Progressive Augmentations.

This is a node-level, large-graph adaptation of Zhao et al. (KDD 2025).
The paper is defined for a sequence of progressively augmented graphs.  For a
single OGB graph, NeighborLoader seed nodes are the learning instances and all
progressive views retain stable node identities.

The exact paper timestamp requires repeated matrix-exponential line searches
over graph Laplacians.  That preprocessing is intractable for million-node OGB
graphs and stochastic neighbor batches, so ``change`` mode uses a scalable
piecewise-smooth proxy based on relative feature and degree changes.  ``index``
mode uses the progressive augmentation step directly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected

from pyagc.encoders import GCN
from .base_method import BaseMethod


PAGCL_AUGMENTATIONS = {
    "none",
    "edge_perturb",
    "node_drop",
    "feature_mask",
    "graph_sampling",
    "random",
}


class PaGCLAugmentor:
    """The four progressive augmentations listed in PaGCL Sec. 3.2."""

    def __init__(self, augmentation: str, ratio: float) -> None:
        if augmentation not in PAGCL_AUGMENTATIONS:
            raise ValueError(
                f"unknown PaGCL augmentation={augmentation}; "
                f"choose from {sorted(PAGCL_AUGMENTATIONS)}"
            )
        if not 0.0 <= ratio < 1.0:
            raise ValueError("pagcl_augmentation_ratio must be in [0, 1)")
        self.augmentation = augmentation
        self.ratio = float(ratio)

    def _resolve(self) -> str:
        if self.augmentation != "random":
            return self.augmentation
        choices = ("edge_perturb", "node_drop", "feature_mask", "graph_sampling")
        return choices[int(torch.randint(len(choices), ()).item())]

    @staticmethod
    def _induced_view(x: Tensor, edge_index: Tensor, keep: Tensor) -> Tuple[Tensor, Tensor]:
        # Stable indices are required for node-level positives.  Removed nodes
        # are zeroed rather than physically reindexed.
        x_aug = x.clone()
        x_aug[~keep] = 0
        edge_keep = keep[edge_index[0]] & keep[edge_index[1]]
        return x_aug, edge_index[:, edge_keep]

    def __call__(
        self,
        x: Tensor,
        edge_index: Tensor,
        anchor_size: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        augmentation = self._resolve()
        num_nodes = int(x.size(0))
        anchor_keep = torch.ones(anchor_size, dtype=torch.bool, device=x.device)
        if augmentation == "none" or self.ratio == 0.0:
            return x, edge_index, anchor_keep

        if augmentation == "feature_mask":
            mask = torch.rand_like(x, dtype=torch.float32) < self.ratio
            source = x.float()
            mean = source.mean(dim=0, keepdim=True)
            std = source.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            gaussian = mean + torch.randn_like(source) * std
            x_aug = torch.where(mask, gaussian, source).to(dtype=x.dtype)
            return x_aug, edge_index, anchor_keep

        if augmentation == "edge_perturb":
            # Preserve the undirected structure used by OGB node datasets.
            # Perturb one canonical copy of each edge, then restore both
            # directions and coalesce duplicates with to_undirected.
            canonical = edge_index[:, edge_index[0] < edge_index[1]]
            num_edges = int(canonical.size(1))
            change = min(num_edges, int(round(num_edges * self.ratio)))
            if change == 0:
                return x, edge_index, anchor_keep
            permutation = torch.randperm(num_edges, device=edge_index.device)
            retained = canonical[:, permutation[change:]]
            additions = torch.randint(
                num_nodes,
                (2, change),
                device=edge_index.device,
                dtype=edge_index.dtype,
            )
            additions = additions[:, additions[0] != additions[1]]
            additions = torch.stack(
                (
                    torch.minimum(additions[0], additions[1]),
                    torch.maximum(additions[0], additions[1]),
                ),
                dim=0,
            )
            perturbed = torch.cat((retained, additions), dim=1)
            perturbed = to_undirected(perturbed, num_nodes=num_nodes)
            return x, perturbed, anchor_keep

        # Node dropping and graph sampling both produce an induced subgraph in
        # the paper.  They differ operationally in motivation, while stable
        # node-level adaptation uses the same keep-mask representation.
        keep = torch.rand(num_nodes, device=x.device) >= self.ratio
        if int(keep.sum().item()) < 2:
            keep[: min(2, num_nodes)] = True
        x_aug, edge_aug = self._induced_view(x, edge_index, keep)
        return x_aug, edge_aug, keep[:anchor_size]


class FourierTimeEncoder(nn.Module):
    """Learnable Fourier timestamp encoder from PaGCL Eq. (6)--(8)."""

    def __init__(self, output_dim: int, num_frequencies: int = 16) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError("pagcl_time_frequencies must be >= 1")
        self.num_frequencies = int(num_frequencies)
        self.frequencies = nn.Parameter(torch.linspace(1.0, float(num_frequencies), num_frequencies))
        basis_dim = 1 + 2 * num_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(basis_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, timestamp: Tensor) -> Tensor:
        timestamp = timestamp.reshape(-1, 1)
        phase = 2.0 * math.pi * timestamp * self.frequencies.reshape(1, -1)
        scale = 1.0 / math.sqrt(float(self.num_frequencies))
        constant = torch.ones_like(timestamp) * scale
        basis = torch.cat((constant, scale * torch.cos(phase), scale * torch.sin(phase)), dim=-1)
        return self.mlp(basis)


class PaGCLCore(nn.Module):
    """Progressive view generator plus spatial/temporal encoder."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        proj_dim: int,
        tau: float,
        augmentation: str,
        augmentation_ratio: float,
        sequence_length: int,
        rho: float,
        temporal_eta: float,
        time_frequencies: int,
        time_mode: str,
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("pagcl_tau must be > 0")
        if sequence_length < 1:
            raise ValueError("pagcl_sequence_length must be >= 1")
        if rho < 0:
            raise ValueError("pagcl_rho must be >= 0")
        if not 0.0 <= temporal_eta <= 1.0:
            raise ValueError("pagcl_temporal_eta must be in [0, 1]")
        if time_mode not in {"change", "index"}:
            raise ValueError("pagcl_time_mode must be 'change' or 'index'")

        # PaGCL optimizes the fused representation Delta_G directly. Use the
        # node-level message-passing GCN family instead of GraphCL's GIN and
        # avoid an additional contrastive projection head.
        self.encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm="batch_norm",
        )
        # Compatibility attribute for existing method code. It intentionally
        # has no parameters and is not a separate training representation.
        self.projector = nn.Identity()
        self.time_encoder = FourierTimeEncoder(hidden_dim, time_frequencies)
        self.augmentor = PaGCLAugmentor(augmentation, augmentation_ratio)
        self.sequence_length = int(sequence_length)
        self.tau = float(tau)
        self.rho = float(rho)
        self.temporal_eta = float(temporal_eta)
        self.time_mode = time_mode
        self.augmentation_ratio = float(augmentation_ratio)

    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def fuse_time(self, spatial: Tensor, timestamp: Tensor) -> Tensor:
        temporal = self.time_encoder(timestamp.to(device=spatial.device, dtype=spatial.dtype))
        if temporal.size(0) == 1:
            temporal = temporal.expand(spatial.size(0), -1)
        return (1.0 - self.temporal_eta) * spatial + self.temporal_eta * temporal

    @staticmethod
    @torch.no_grad()
    def _change_interval(x1: Tensor, e1: Tensor, x2: Tensor, e2: Tensor) -> Tensor:
        """Scalable proxy for the paper's diffusion-distance line search."""
        eps = torch.finfo(torch.float32).eps
        feature_delta = (x1.float() - x2.float()).pow(2).mean().sqrt()
        feature_scale = x1.float().pow(2).mean().sqrt().clamp_min(eps)
        feature_delta = feature_delta / feature_scale

        num_nodes = int(x1.size(0))
        degree1 = torch.bincount(e1[0], minlength=num_nodes).float()
        degree2 = torch.bincount(e2[0], minlength=num_nodes).float()
        degree_delta = (degree1 - degree2).pow(2).mean().sqrt()
        degree_scale = degree1.pow(2).mean().sqrt().clamp_min(1.0)
        degree_delta = degree_delta / degree_scale
        return (feature_delta + degree_delta).clamp_min(1e-6)

    def progressive_views(
        self,
        x: Tensor,
        edge_index: Tensor,
        anchor_size: int,
    ) -> Tuple[List[Tuple[Tensor, Tensor]], Tensor, List[Tensor]]:
        views: List[Tuple[Tensor, Tensor]] = [(x, edge_index)]
        common = torch.ones(anchor_size, dtype=torch.bool, device=x.device)
        timestamps: List[Tensor] = [x.new_zeros(())]
        current_x, current_edge = x, edge_index
        cumulative = x.new_zeros(())

        for step in range(1, self.sequence_length + 1):
            next_x, next_edge, keep = self.augmentor(current_x, current_edge, anchor_size)
            common = common & keep
            if self.time_mode == "change":
                interval = self._change_interval(current_x, current_edge, next_x, next_edge)
                interval = interval.to(device=x.device, dtype=x.dtype)
            else:
                interval = x.new_tensor(max(self.augmentation_ratio, 1.0 / self.sequence_length))
            cumulative = cumulative + interval
            timestamps.append(cumulative)
            views.append((next_x, next_edge))
            current_x, current_edge = next_x, next_edge
        return views, common, timestamps

    def projected_sequence(
        self,
        views: Sequence[Tuple[Tensor, Tensor]],
        common: Tensor,
        timestamps: Sequence[Tensor],
        anchor_size: int,
    ) -> List[Tensor]:
        projected: List[Tensor] = []
        for (view_x, view_edge), timestamp in zip(views, timestamps):
            spatial = self.embed(view_x, view_edge)[:anchor_size][common]
            fused = self.fuse_time(spatial, timestamp.reshape(1))
            projected.append(fused)
        return projected

    @staticmethod
    def _pairwise_similarity_mean(
        left: Tensor,
        right: Tensor,
        include_matched: bool,
        chunk_size: int = 1024,
    ) -> Tensor:
        """Mean BxB cosine similarity for the PaGCL variant objective.

        Off-diagonal entries are completely different seed-node instances.
        Matched entries are included only when the two progressive views are
        sufficiently separated. Chunking preserves the paper's pairwise
        objective without materializing every BxB matrix at once.
        """
        left = F.normalize(left, dim=-1)
        right = F.normalize(right, dim=-1)
        batch_size = int(left.size(0))
        total = left.new_zeros(())
        count = 0
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            similarity = left[start:stop] @ right.T
            if include_matched:
                total = total + similarity.sum()
                count += int(similarity.numel())
            else:
                local = torch.arange(stop - start, device=left.device)
                global_index = torch.arange(start, stop, device=left.device)
                total = total + similarity.sum() - similarity[local, global_index].sum()
                count += int(similarity.numel()) - (stop - start)
        if count <= 0:
            raise ValueError("PaGCL pairwise variant objective needs B >= 2")
        return total / count

    def invariance_objective(self, projected: Sequence[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """PaGCL Eq. (9)--(11), adapted from graphs to seed-node instances."""
        if len(projected) < 2:
            raise ValueError("PaGCL needs at least two progressive views")
        invariant_terms = [
            -F.cosine_similarity(left, right, dim=-1).mean() / self.tau
            for left, right in zip(projected[:-1], projected[1:])
        ]
        # The paper sums the consecutive-view terms in Eq. (9).
        invariant = torch.stack(invariant_terms).sum()

        # Eq. (10) permits a sufficiently distant view or a completely
        # different instance. In a single large graph, seed nodes are the
        # learning instances, so use their full BxB relation structure rather
        # than assigning only one randomly rolled negative to each anchor.
        num_nodes = int(projected[0].size(0))
        if num_nodes < 2:
            raise ValueError("PaGCL requires at least two active seed nodes")
        farthest = projected[-1]
        last_index = len(projected) - 1
        variant_terms = []
        for index, current in enumerate(projected[:-1]):
            # The matched pair is variant only if an intermediate view lies
            # between the current and final views. Off-diagonal pairs always
            # correspond to completely different seed-node instances.
            include_matched = (last_index - index) > 1
            similarity = self._pairwise_similarity_mean(
                current,
                farthest,
                include_matched=include_matched,
            )
            variant_terms.append(similarity / self.tau)
        # Eq. (10) likewise sums the sufficiently distant negative terms.
        variant = torch.stack(variant_terms).sum()
        total = invariant + self.rho * variant
        return total, invariant, variant


class PaGCLMethod(BaseMethod):
    """Large-graph node representation implementation of PaGCL."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        proj_dim: int,
        tau: float = 0.5,
        augmentation: str = "edge_perturb",
        augmentation_ratio: float = 0.2,
        sequence_length: int = 2,
        rho: float = 1.0,
        temporal_eta: float = 0.3,
        time_frequencies: int = 16,
        time_mode: str = "change",
    ) -> None:
        super().__init__(method_name="pagcl", is_supervised=False)
        self.hidden_dim = int(hidden_dim)
        self.model = PaGCLCore(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            proj_dim=proj_dim,
            tau=tau,
            augmentation=augmentation,
            augmentation_ratio=augmentation_ratio,
            sequence_length=sequence_length,
            rho=rho,
            temporal_eta=temporal_eta,
            time_frequencies=time_frequencies,
            time_mode=time_mode,
        )
        self.last_pagcl_losses: Dict[str, float] = {}

    def output_dim(self) -> int:
        return self.hidden_dim

    def _projected_sequence_loss(self, projected: Sequence[Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        total, invariant, variant = self.model.invariance_objective(projected)
        return total, {
            "pagcl_invariant_loss": float(invariant.detach().item()),
            "pagcl_variant_loss": float(variant.detach().item()),
            "pagcl_loss": float(total.detach().item()),
        }

    def _train_batch(
        self,
        x: Tensor,
        edge_index: Tensor,
        anchor_size: int,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, Dict[str, float]]:
        if anchor_size < 2:
            raise ValueError("PaGCL requires at least two seed nodes")
        optimizer.zero_grad()
        for _ in range(5):
            views, common, timestamps = self.model.progressive_views(x, edge_index, anchor_size)
            if int(common.sum().item()) >= 2:
                break
        else:
            raise RuntimeError("PaGCL progressive augmentation retained fewer than two seed nodes")
        projected = self.model.projected_sequence(views, common, timestamps, anchor_size)
        loss, details = self._projected_sequence_loss(projected)
        loss.backward()
        optimizer.step()
        details["active_anchors"] = float(common.sum().item())
        details["pagcl_last_timestamp"] = float(timestamps[-1].detach().item())
        return float(loss.item()), details

    def ssl_train_step_full(
        self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer
    ) -> float:
        self.train()
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        loss, details = self._train_batch(x, edge_index, int(x.size(0)), optimizer)
        self.last_pagcl_losses = details
        return loss

    def ssl_train_step_neighbor(
        self,
        data: Data,
        input_nodes: Optional[Tensor],
        num_neighbors: Sequence[int],
        batch_size: int,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.train()
        loader = NeighborLoader(
            data,
            input_nodes=input_nodes,
            num_neighbors=list(num_neighbors),
            batch_size=batch_size,
            shuffle=True,
        )
        total = 0.0
        count = 0
        detail_sums: Dict[str, float] = {}
        for batch in loader:
            batch = batch.to(device)
            active = int(batch.batch_size)
            if active < 2:
                continue
            loss, details = self._train_batch(batch.x, batch.edge_index, active, optimizer)
            total += loss * active
            count += active
            for key, value in details.items():
                detail_sums[key] = detail_sums.get(key, 0.0) + value * active
        if count:
            self.last_pagcl_losses = {key: value / count for key, value in detail_sums.items()}
        return total / max(count, 1)

    @torch.no_grad()
    def infer_embeddings(
        self,
        data: Data,
        mode: str,
        device: torch.device,
        eval_num_neighbors: Sequence[int],
        eval_batch_size: int,
    ) -> Tensor:
        self.eval()
        timestamp = torch.zeros(1, device=device)
        if mode == "full":
            spatial = self.model.embed(data.x.to(device), data.edge_index.to(device))
            return self.model.fuse_time(spatial, timestamp).cpu()
        loader = NeighborLoader(
            data,
            input_nodes=None,
            num_neighbors=list(eval_num_neighbors),
            batch_size=eval_batch_size,
            shuffle=False,
        )
        outputs: List[Tensor] = []
        for batch in loader:
            batch = batch.to(device)
            spatial = self.model.embed(batch.x, batch.edge_index)[: batch.batch_size]
            outputs.append(self.model.fuse_time(spatial, timestamp).cpu())
        return torch.cat(outputs, dim=0)

    def get_last_mrl_dim_losses(self) -> Dict[str, float]:
        return dict(self.last_pagcl_losses)
