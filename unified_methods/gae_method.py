"""GAE 和 VGAE 方法类。"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.encoders import GCN
from pyagc.models.gae import GAE, VGAE

from .base_method import BaseMethod


class VGAEEncoder(nn.Module):
    """VGAE 编码器：GCN 骨干 + μ/log σ² 双头输出。

    对应论文中共享第一层 GCN、第二层分离为 GCN_μ 和 GCN_σ 的设计。
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.backbone = GCN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm="batch_norm",
        )
        self.mu_head = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.backbone(x, edge_index)
        return self.mu_head(h), self.logvar_head(h)

    def reset_parameters(self) -> None:
        self.backbone.reset_parameters()
        self.mu_head.reset_parameters()
        self.logvar_head.reset_parameters()


class GAEMethod(BaseMethod):
    """GAE 自监督方法。

    使用 GCN 编码器 + 内积解码器，通过重建图邻接矩阵来学习节点表示。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(method_name="gae", is_supervised=False)
        encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm="batch_norm",
        )
        self.model = GAE(encoder=encoder)
        self.hidden_dim = hidden_dim

    def output_dim(self) -> int:
        return self.hidden_dim

    def ssl_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        loss = self.model.loss(x=data.x.to(device), edge_index=data.edge_index.to(device)).total
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
            loss = self.model.loss_batch(batch).total
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


class VGAEMethod(BaseMethod):
    """VGAE 自监督方法。

    使用 GCN 骨干 + μ/log σ² 双头编码器，学习节点表示的变分分布。
    损失函数为证据下界（ELBO）：重建损失 + KL 散度。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(method_name="vgae", is_supervised=False)
        encoder = VGAEEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.model = VGAE(encoder=encoder)
        self.hidden_dim = hidden_dim

    def output_dim(self) -> int:
        return self.hidden_dim

    def ssl_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        loss = self.model.loss(x=data.x.to(device), edge_index=data.edge_index.to(device)).total
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
            loss = self.model.loss_batch(batch).total
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
