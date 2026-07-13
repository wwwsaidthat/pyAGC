"""GAE 方法类。"""

import math
from typing import List, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import negative_sampling

from pyagc.encoders import GCN
from pyagc.models.gae import GAE

from .base_method import BaseMethod


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

    @torch.no_grad()
    def diagnose(
        self,
        data: Data,
        mode: str,
        device: torch.device,
        eval_num_neighbors: Sequence[int],
        eval_batch_size: int,
        logger,
    ) -> None:
        """打印诊断信息：embedding 范数、logit 尺度、Sigmoid 饱和度。

        在训练结束后调用一次，用于排查高维下的表示坍缩或 logit 爆炸。
        """
        self.eval()
        z = self.infer_embeddings(data, mode, device, eval_num_neighbors, eval_batch_size)
        e = data.edge_index
        n = z.size(0)
        d = z.size(-1)

        # 负边采样（CPU 上完成，避免 GPU OOM）
        neg_e = negative_sampling(edge_index=e, num_nodes=n,
                                  num_neg_samples=e.size(1), method="sparse")

        # 正边 logits（已缩放）
        src_p, dst_p = e
        pos_logits = (z[src_p] * z[dst_p]).sum(dim=-1) / math.sqrt(d)

        # 负边 logits（已缩放）
        src_n, dst_n = neg_e
        neg_logits = (z[src_n] * z[dst_n]).sum(dim=-1) / math.sqrt(d)

        z_norms = z.norm(dim=-1)

        logger.info(
            "DIAG | dim=%d | z mean=%.4f std=%.4f norm_mean=%.2f norm_max=%.2f | "
            "pos_logit mean=%.2f std=%.2f | neg_logit mean=%.2f std=%.2f | "
            "pos_sat=%.4f neg_sat=%.4f",
            d,
            z.mean().item(), z.std().item(),
            z_norms.mean().item(), z_norms.max().item(),
            pos_logits.mean().item(), pos_logits.std().item(),
            neg_logits.mean().item(), neg_logits.std().item(),
            (pos_logits.abs() > 10).float().mean().item(),
            (neg_logits.abs() > 10).float().mean().item(),
        )
