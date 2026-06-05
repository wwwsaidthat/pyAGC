"""GRACE 方法类。"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.encoders import GCN
from pyagc.transforms import GSSLTransform

from .base_method import BaseMethod


class GRACECore(nn.Module):
    """GRACE 核心模型。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        proj_dim: int,
        tau: float,
        p_feat_mask_1: float,
        p_edge_drop_1: float,
        p_feat_mask_2: float,
        p_edge_drop_2: float,
    ) -> None:
        super().__init__()
        self.encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm="batch_norm",
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.tau = tau
        self.t1 = GSSLTransform(p_feat_mask_1, p_edge_drop_1, node_attrs=["x"], edge_attrs=[])
        self.t2 = GSSLTransform(p_feat_mask_2, p_edge_drop_2, node_attrs=["x"], edge_attrs=[])

    @staticmethod
    def nt_xent(z1: Tensor, z2: Tensor, tau: float) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim = torch.mm(z1, z2.t()) / tau
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))

    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x=x, edge_index=edge_index)

    def loss(self, x: Tensor, edge_index: Tensor, seed_size: Optional[int] = None) -> Tensor:
        v1 = self.t1(x, edge_index)
        v2 = self.t2(x, edge_index)
        h1 = self.embed(v1["x"], v1["edge_index"])
        h2 = self.embed(v2["x"], v2["edge_index"])
        if seed_size is not None:
            h1 = h1[:seed_size]
            h2 = h2[:seed_size]
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return self.nt_xent(z1, z2, self.tau)


class GRACEMethod(BaseMethod):
    """GRACE 自监督方法。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        proj_dim: int,
        tau: float,
        p_feat_mask_1: float,
        p_edge_drop_1: float,
        p_feat_mask_2: float,
        p_edge_drop_2: float,
    ) -> None:
        super().__init__(method_name="grace", is_supervised=False)
        self.model = GRACECore(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            proj_dim=proj_dim,
            tau=tau,
            p_feat_mask_1=p_feat_mask_1,
            p_edge_drop_1=p_edge_drop_1,
            p_feat_mask_2=p_feat_mask_2,
            p_edge_drop_2=p_edge_drop_2,
        )
        self.hidden_dim = hidden_dim

    def output_dim(self) -> int:
        return self.hidden_dim

    def ssl_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        loss = self.model.loss(data.x.to(device), data.edge_index.to(device))
        loss.backward()
        optimizer.step()
        return float(loss.item())

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
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = self.model.loss(batch.x, batch.edge_index, seed_size=int(batch.batch_size))
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(batch.batch_size)
            count += int(batch.batch_size)
        return total / max(count, 1)

    @torch.no_grad()
    def infer_embeddings(self, data: Data, mode: str, device: torch.device, eval_num_neighbors: Sequence[int], eval_batch_size: int) -> Tensor:
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
        out: List[Tensor] = []
        for batch in loader:
            batch = batch.to(device)
            h = self.model.embed(batch.x, batch.edge_index)[: batch.batch_size]
            out.append(h.cpu())
        return torch.cat(out, dim=0)
