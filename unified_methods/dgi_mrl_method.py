"""DGI + MRL 方法类。"""

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .dgi_method import DGIMethod
from pyagc.models.dgi import EPS


class DGIWithMRLMethod(DGIMethod):
    """DGI + MRL 融合方法。"""

    def __init__(self, *args, mrl_dims: Sequence[int], mrl_weight: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.method_name = "dgi_mrl"
        dims = sorted({int(d) for d in mrl_dims})
        if len(dims) == 0:
            raise ValueError("mrl_dims 不能为空")
        if any(d <= 0 for d in dims):
            raise ValueError("mrl_dims 中所有维度必须 > 0")
        self.mrl_dims = dims
        self.mrl_weight = float(mrl_weight)
        if self.hidden_dim < int(max(self.mrl_dims)):
            raise ValueError(
                f"DGI+MRL 需要 --hidden-dim >= max(mrl_dims)={max(self.mrl_dims)}, "
                f"因为下游评估使用 encoder 输出（dim={self.hidden_dim}）切片, "
                f"当前 hidden-dim={self.hidden_dim} 不足"
            )
        self.last_mrl_dim_losses: Dict[str, float] = {}

    def output_dim(self) -> int:
        return self.hidden_dim

    def _dgi_prefix_loss_with_details(self, pos_z_full: Tensor, neg_z_full: Tensor) -> Tensor:
        w = self.model.weight
        losses = []
        for dim in self.mrl_dims:
            pos_z = pos_z_full[:, :dim]
            neg_z = neg_z_full[:, :dim]
            summary = self.model.summary(pos_z)
            w_dim = w[:dim, :dim]

            summary_vec = summary.t() if summary.dim() > 1 else summary
            pos_score = torch.matmul(pos_z, torch.matmul(w_dim, summary_vec))
            neg_score = torch.matmul(neg_z, torch.matmul(w_dim, summary_vec))

            pos_prob = torch.sigmoid(pos_score)
            neg_prob = torch.sigmoid(neg_score)

            pos_loss = -torch.log(pos_prob + EPS).mean()
            neg_loss = -torch.log(1 - neg_prob + EPS).mean()
            losses.append(pos_loss + neg_loss)

        return self.mrl_weight * torch.stack(losses, dim=0).mean()

    def ssl_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        x = data.x.to(device)
        e = data.edge_index.to(device)
        pos_z, neg_z, _ = self.model(x=x, edge_index=e)
        loss = self._dgi_prefix_loss_with_details(pos_z, neg_z)
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
            pos_z, neg_z, _ = self.model(batch.x, batch.edge_index)
            pos_z = pos_z[: batch.batch_size]
            neg_z = neg_z[: batch.batch_size]
            loss = self._dgi_prefix_loss_with_details(pos_z, neg_z)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(batch.batch_size)
            count += int(batch.batch_size)
        return total / max(count, 1)

    @torch.no_grad()
    def infer_embeddings(self, data: Data, mode: str, device: torch.device, eval_num_neighbors: Sequence[int], eval_batch_size: int) -> Tensor:
        return super().infer_embeddings(data, mode, device, eval_num_neighbors, eval_batch_size)

    def get_last_mrl_dim_losses(self) -> Dict[str, float]:
        """返回最近一个 epoch 的各维度 MRL 损失。"""
        return dict(self.last_mrl_dim_losses)
