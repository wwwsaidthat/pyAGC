"""GAE 方法类。"""

from typing import List, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.encoders import GCN
from pyagc.models.gae import GAE

from .base_method import BaseMethod


class GAEMethod(BaseMethod):
    """GAE 自监督方法。

    使用 GCN 编码器 + 内积解码器，通过重建图邻接矩阵来学习节点表示。

    Args:
        in_dim: 输入特征维度。
        hidden_dim: 隐藏/输出维度。
        num_layers: GCN 层数。
        dropout: Dropout 比率。
        neg_ratio: 负边采样比例（相对正边数），默认 1.0。
            大图建议 0.25--0.5 以控制内存。
        use_amp: 是否启用自动混合精度（AMP），可节省约 40% 显存。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        neg_ratio: float = 1.0,
        use_amp: bool = False,
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
        self.model = GAE(encoder=encoder, neg_ratio=neg_ratio)
        self.hidden_dim = hidden_dim
        self.use_amp = bool(use_amp)

    def output_dim(self) -> int:
        return self.hidden_dim

    def ssl_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp and device.type == "cuda"):
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
            with torch.cuda.amp.autocast(enabled=self.use_amp and device.type == "cuda"):
                loss = self.model.loss_batch(batch).total
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(batch.batch_size)
            count += int(batch.batch_size)
            del batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
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

    # NOTE: diagnose() removed — it was only diagnostic logging, not needed for
    # training → inference flow.
