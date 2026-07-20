"""Node-level large-graph adaptation of GraphCL (NeurIPS 2020).

The official GraphCL implementation contrasts graph-level GIN embeddings from
small-graph batches.  OGB node classification instead provides one large graph,
so this adapter treats NeighborLoader seed nodes as contrastive instances while
retaining GraphCL's GIN backbone, augmentation family, projection head and
positive-excluded contrastive denominator.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GINConv

from .base_method import BaseMethod


GRAPHCL_AUGMENTATIONS = {
    "none",
    "node_drop",
    "edge_perturb",
    "subgraph",
    "attr_mask",
    "random",
}


class GraphCLAugmentor:
    """GraphCL augmentation family with stable node identities.

    Nodes are not physically re-indexed: node/subgraph removal zeros their
    features and deletes incident edges.  The returned mask tells the loss
    which seed nodes survived, preserving valid positive-pair identities.
    """

    def __init__(self, augmentation: str, ratio: float) -> None:
        if augmentation not in GRAPHCL_AUGMENTATIONS:
            raise ValueError(
                f"未知 GraphCL augmentation={augmentation}; "
                f"可选 {sorted(GRAPHCL_AUGMENTATIONS)}"
            )
        if not 0.0 <= ratio < 1.0:
            raise ValueError("graphcl augmentation ratio 必须位于 [0, 1)")
        self.augmentation = augmentation
        self.ratio = float(ratio)

    def _resolve(self) -> str:
        if self.augmentation != "random":
            return self.augmentation
        choices = ("node_drop", "edge_perturb", "subgraph", "attr_mask")
        return choices[int(torch.randint(len(choices), ()).item())]

    @staticmethod
    def _apply_node_keep(x: Tensor, edge_index: Tensor, keep: Tensor) -> Tuple[Tensor, Tensor]:
        x_aug = x.clone()
        x_aug[~keep] = 0
        edge_keep = keep[edge_index[0]] & keep[edge_index[1]]
        return x_aug, edge_index[:, edge_keep]

    def _connected_keep(self, num_nodes: int, edge_index: Tensor) -> Tensor:
        target = min(num_nodes, max(2, int(round(num_nodes * (1.0 - self.ratio)))))
        keep = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        keep[torch.randint(num_nodes, (), device=edge_index.device)] = True

        # Vectorized frontier growth. NeighborLoader batches are already unions
        # of connected ego-graphs, so this produces a connected retained core in
        # normal cases without materializing a dense adjacency matrix.
        for _ in range(32):
            current = int(keep.sum().item())
            if current >= target:
                break
            frontier = edge_index[1, keep[edge_index[0]] & ~keep[edge_index[1]]]
            frontier = torch.unique(frontier)
            if frontier.numel() == 0:
                break
            need = target - current
            if frontier.numel() > need:
                order = torch.randperm(frontier.numel(), device=frontier.device)[:need]
                frontier = frontier[order]
            keep[frontier] = True

        missing = target - int(keep.sum().item())
        if missing > 0:
            candidates = (~keep).nonzero(as_tuple=False).flatten()
            order = torch.randperm(candidates.numel(), device=candidates.device)[:missing]
            keep[candidates[order]] = True
        return keep

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

        if augmentation == "attr_mask":
            x_aug = x.clone()
            mask = torch.rand(num_nodes, device=x.device) < self.ratio
            x_aug[mask] = 0
            return x_aug, edge_index, anchor_keep

        if augmentation == "edge_perturb":
            num_edges = int(edge_index.size(1))
            change = min(num_edges, int(round(num_edges * self.ratio)))
            if change == 0:
                return x, edge_index, anchor_keep
            permutation = torch.randperm(num_edges, device=edge_index.device)
            retained = edge_index[:, permutation[change:]]
            additions = torch.randint(
                num_nodes,
                (2, change),
                device=edge_index.device,
                dtype=edge_index.dtype,
            )
            non_loop = additions[0] != additions[1]
            edge_aug = torch.cat((retained, additions[:, non_loop]), dim=1)
            return x, edge_aug, anchor_keep

        if augmentation == "node_drop":
            keep = torch.rand(num_nodes, device=x.device) >= self.ratio
            if int(keep.sum().item()) < 2:
                keep[: min(2, num_nodes)] = True
        elif augmentation == "subgraph":
            keep = self._connected_keep(num_nodes, edge_index)
        else:  # guarded in __init__
            raise RuntimeError(f"未处理的 GraphCL augmentation: {augmentation}")

        x_aug, edge_aug = self._apply_node_keep(x, edge_index, keep)
        return x_aug, edge_aug, keep[:anchor_size]


class GraphCLNodeEncoder(nn.Module):
    """GIN encoder following GraphCL, adapted to a fixed node output width."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("GraphCL num_layers 必须 >= 1")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = in_dim if layer == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = float(dropout)
        # Original GraphCL concatenates all GIN layers. This learned fusion keeps
        # the public node embedding width equal to --hidden-dim for MRL slicing.
        self.fusion = nn.Linear(hidden_dim * num_layers, hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        outputs = []
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(conv(x, edge_index))
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            outputs.append(x)
        return self.fusion(torch.cat(outputs, dim=-1))


class GraphCLCore(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        proj_dim: int,
        tau: float,
        aug_1: str,
        aug_2: str,
        aug_ratio_1: float,
        aug_ratio_2: float,
        symmetric_loss: bool,
    ) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("graphcl_tau 必须 > 0")
        self.encoder = GraphCLNodeEncoder(in_dim, hidden_dim, num_layers, dropout)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.tau = float(tau)
        self.symmetric_loss = bool(symmetric_loss)
        self.augmentor_1 = GraphCLAugmentor(aug_1, aug_ratio_1)
        self.augmentor_2 = GraphCLAugmentor(aug_2, aug_ratio_2)

    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    @staticmethod
    def _directional_loss(z1: Tensor, z2: Tensor, tau: Union[float, Tensor]) -> Tensor:
        if z1.size(0) < 2:
            raise ValueError("GraphCL InfoNCE 至少需要 2 个有效 anchor")
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.T / tau
        positive = logits.diag()
        diagonal = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        negative_logsumexp = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
        return (-positive + negative_logsumexp).mean()

    def contrastive_loss(
        self,
        z1: Tensor,
        z2: Tensor,
        tau: Optional[Union[float, Tensor]] = None,
    ) -> Tensor:
        temperature = self.tau if tau is None else tau
        forward = self._directional_loss(z1, z2, temperature)
        if not self.symmetric_loss:
            return forward
        backward = self._directional_loss(z2, z1, temperature)
        return 0.5 * (forward + backward)


class GraphCLMethod(BaseMethod):
    """GraphCL baseline for node embeddings on arxiv/products/MAG."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        proj_dim: int,
        tau: float = 0.2,
        aug_1: str = "edge_perturb",
        aug_2: str = "attr_mask",
        aug_ratio_1: float = 0.2,
        aug_ratio_2: float = 0.2,
        symmetric_loss: bool = False,
    ) -> None:
        super().__init__(method_name="graphcl", is_supervised=False)
        self.hidden_dim = int(hidden_dim)
        self.model = GraphCLCore(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            proj_dim=proj_dim,
            tau=tau,
            aug_1=aug_1,
            aug_2=aug_2,
            aug_ratio_1=aug_ratio_1,
            aug_ratio_2=aug_ratio_2,
            symmetric_loss=symmetric_loss,
        )
        self.last_graphcl_losses: Dict[str, float] = {}

    def output_dim(self) -> int:
        return self.hidden_dim

    def _projected_loss(self, z1: Tensor, z2: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        loss = self.model.contrastive_loss(z1, z2)
        return loss, {"graphcl_loss": float(loss.detach().item())}

    def _train_batch(
        self,
        x: Tensor,
        edge_index: Tensor,
        anchor_size: int,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, Dict[str, float]]:
        if anchor_size < 2:
            raise ValueError("GraphCL 对比批次至少需要 2 个 seed nodes")
        optimizer.zero_grad()
        # Node-drop/subgraph can theoretically leave fewer than two common
        # seed nodes across the two random views. Resample instead of silently
        # treating a removed seed as a valid positive pair.
        for _ in range(5):
            x1, e1, keep1 = self.model.augmentor_1(x, edge_index, anchor_size)
            x2, e2, keep2 = self.model.augmentor_2(x, edge_index, anchor_size)
            common = keep1 & keep2
            if int(common.sum().item()) >= 2:
                break
        else:
            x1, e1 = x, edge_index
            x2, e2 = x, edge_index
            common = torch.ones(anchor_size, dtype=torch.bool, device=x.device)
        h1 = self.model.embed(x1, e1)[:anchor_size][common]
        h2 = self.model.embed(x2, e2)[:anchor_size][common]
        z1 = self.model.projector(h1)
        z2 = self.model.projector(h2)
        loss, details = self._projected_loss(z1, z2)
        loss.backward()
        optimizer.step()
        details["active_anchors"] = float(common.sum().item())
        return float(loss.item()), details

    def ssl_train_step_full(
        self,
        data: Data,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.train()
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        loss, details = self._train_batch(x, edge_index, int(x.size(0)), optimizer)
        self.last_graphcl_losses = details
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
            self.last_graphcl_losses = {key: value / count for key, value in detail_sums.items()}
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
        if mode == "full":
            return self.model.embed(data.x.to(device), data.edge_index.to(device)).cpu()
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
            outputs.append(self.model.embed(batch.x, batch.edge_index)[: batch.batch_size].cpu())
        return torch.cat(outputs, dim=0)

    def get_last_mrl_dim_losses(self) -> Dict[str, float]:
        # Reuse the unified trainer's detailed-loss logging hook.
        return dict(self.last_graphcl_losses)
