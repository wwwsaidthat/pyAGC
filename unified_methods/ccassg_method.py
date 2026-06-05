"""CCA-SSG 方法类。"""

from typing import List, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.encoders import GCN
from pyagc.models import CCASSG
from pyagc.transforms import GSSLTransform

from .base_method import BaseMethod


class CCASSGMethod(BaseMethod):
    """CCA-SSG 自监督方法。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        lam: float,
        p_feat_mask_1: float,
        p_edge_drop_1: float,
        p_feat_mask_2: float,
        p_edge_drop_2: float,
    ) -> None:
        super().__init__(method_name="ccassg", is_supervised=False)
        encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm="batch_norm",
        )
        t1 = GSSLTransform(
            p_feat_mask=p_feat_mask_1,
            p_edge_drop=p_edge_drop_1,
            node_attrs=["x"],
            edge_attrs=[],
        )
        t2 = GSSLTransform(
            p_feat_mask=p_feat_mask_2,
            p_edge_drop=p_edge_drop_2,
            node_attrs=["x"],
            edge_attrs=[],
        )
        self.model = CCASSG(encoder=encoder, transform1=t1, transform2=t2, lam=lam)
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
