"""SSGE + MRL 方法类。"""

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .ssge_method import SSGEMethod


class SSGEWithMRLMethod(SSGEMethod):
    """SSGE + MRL 融合方法。

    在 SSGE 的基础上加入 MRL（Multi-Representation Learning），
    对每个指定的维度前缀独立计算 SSGE 损失（不变性 + 均匀性），
    然后对各个维度的损失取平均。
    """

    def __init__(self, *args, mrl_dims: Sequence[int], mrl_weight: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.method_name = "ssge_mrl"
        self.mrl_dims = sorted({int(d) for d in mrl_dims})
        if len(self.mrl_dims) == 0:
            raise ValueError("mrl_dims 不能为空")
        if any(d <= 0 for d in self.mrl_dims):
            raise ValueError("mrl_dims 中所有维度必须 > 0")
        self.mrl_weight = float(mrl_weight)
        if self.hidden_dim < int(max(self.mrl_dims)):
            raise ValueError(
                f"SSGE+MRL 需要 --hidden-dim >= max(mrl_dims)={max(self.mrl_dims)}, "
                f"因为下游评估使用 encoder 输出（dim={self.hidden_dim}）切片, "
                f"当前 hidden-dim={self.hidden_dim} 不足"
            )
        self.last_mrl_dim_losses: Dict[str, float] = {}

    def output_dim(self) -> int:
        return self.hidden_dim

    def _ssge_mrl_loss(self, z1_full: Tensor, z2_full: Tensor) -> Tensor:
        losses = []
        for dim in self.mrl_dims:
            z1 = z1_full[:, :dim]
            z2 = z2_full[:, :dim]
            z1 = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + 1e-12)
            z2 = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + 1e-12)

            inv = -(z1 * z2).sum() / z1.shape[0]

            c1 = z1.T @ z1 / (z1.shape[0] - 1)
            c2 = z2.T @ z2 / (z2.shape[0] - 1)
            L1, _ = torch.linalg.eigh(c1)
            L2, _ = torch.linalg.eigh(c2)
            uni = -(torch.clamp(L1, min=1e-8).sqrt().sum() + torch.clamp(L2, min=1e-8).sqrt().sum())

            losses.append(inv + self.model.lam * uni)

        return self.mrl_weight * torch.stack(losses, dim=0).mean()

    def ssl_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        x = data.x.to(device)
        e = data.edge_index.to(device)
        z1, z2 = self.model(x, e)
        loss = self._ssge_mrl_loss(z1, z2)
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
            z1, z2 = self.model(batch.x, batch.edge_index)
            z1 = z1[: batch.batch_size]
            z2 = z2[: batch.batch_size]
            loss = self._ssge_mrl_loss(z1, z2)
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
