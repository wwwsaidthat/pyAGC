"""GCN 监督训练方法类。"""

from typing import List, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.encoders import GCN

from .base_method import BaseMethod


class SupervisedGCNMethod(BaseMethod):
    """GCN 监督训练方法。"""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_classes: int) -> None:
        super().__init__(method_name="gcn_supervised", is_supervised=True)
        self.encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.hidden_dim = hidden_dim

    def output_dim(self) -> int:
        return self.hidden_dim

    def forward_logits(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self.encoder(x=x, edge_index=edge_index)
        return self.classifier(h)

    def supervised_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        logits = self.forward_logits(data.x.to(device), data.edge_index.to(device))
        loss = self.criterion(logits[data.train_idx.to(device)], data.y[data.train_idx].to(device))
        loss.backward()
        optimizer.step()
        return float(loss.item())

    def supervised_train_step_neighbor(
        self,
        data: Data,
        train_idx: Tensor,
        num_neighbors: Sequence[int],
        batch_size: int,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.train()
        loader = NeighborLoader(
            data,
            input_nodes=train_idx,
            num_neighbors=list(num_neighbors),
            batch_size=batch_size,
            shuffle=True,
        )
        total = 0.0
        count = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = self.forward_logits(batch.x, batch.edge_index)[: batch.batch_size]
            y = batch.y[: batch.batch_size]
            loss = self.criterion(logits, y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(batch.batch_size)
            count += int(batch.batch_size)
        return total / max(count, 1)

    @torch.no_grad()
    def supervised_predict(self, data: Data, mode: str, device: torch.device, eval_num_neighbors: Sequence[int], eval_batch_size: int) -> Tensor:
        self.eval()
        if mode == "full":
            return self.forward_logits(data.x.to(device), data.edge_index.to(device)).cpu()
        loader = NeighborLoader(
            data,
            input_nodes=None,
            num_neighbors=list(eval_num_neighbors),
            batch_size=eval_batch_size,
            shuffle=False,
        )
        all_logits: List[Tensor] = []
        for batch in loader:
            batch = batch.to(device)
            logits = self.forward_logits(batch.x, batch.edge_index)[: batch.batch_size]
            all_logits.append(logits.cpu())
        return torch.cat(all_logits, dim=0)
