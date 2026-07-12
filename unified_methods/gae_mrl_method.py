"""GAE + MRL 和 VGAE + MRL 方法类。

MRL (Multi-Representation Learning) 对每个维度前缀独立计算损失，
然后对各维度损失取平均，使得不同维度的前缀也能独立地保留图结构信息。
"""

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.models.gae import EPS, _sample_neg_edges

from .gae_method import GAEMethod, VGAEMethod, VGAEEncoder


# ============================================================================
# GAE + MRL
# ============================================================================


class GAEWithMRLMethod(GAEMethod):
    """GAE + MRL 融合方法。

    对每个指定的维度前缀独立计算重建损失（BCE），然后对各个维度的损失取平均。
    """

    def __init__(self, *args, mrl_dims: Sequence[int], mrl_weight: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.method_name = "gae_mrl"
        self.mrl_dims = sorted({int(d) for d in mrl_dims})
        if len(self.mrl_dims) == 0:
            raise ValueError("mrl_dims 不能为空")
        if any(d <= 0 for d in self.mrl_dims):
            raise ValueError("mrl_dims 中所有维度必须 > 0")
        self.mrl_weight = float(mrl_weight)
        if self.hidden_dim < int(max(self.mrl_dims)):
            raise ValueError(
                f"GAE+MRL 需要 --hidden-dim >= max(mrl_dims)={max(self.mrl_dims)}, "
                f"因为下游评估使用 encoder 输出（dim={self.hidden_dim}）切片, "
                f"当前 hidden-dim={self.hidden_dim} 不足"
            )
        self.last_mrl_dim_losses: Dict[str, float] = {}

    def output_dim(self) -> int:
        return self.hidden_dim

    def _gae_mrl_loss(self, z_full: Tensor, pos_edge_index: Tensor) -> Tensor:
        """对每个维度前缀独立计算 GAE 重建损失。"""
        losses = []
        n = z_full.size(0)
        num_pos = pos_edge_index.size(1)

        # 共享同一组负边采样
        neg_edge_index = _sample_neg_edges(n, num_pos, z_full.device)

        for dim in self.mrl_dims:
            z = z_full[:, :dim]

            # 正边损失
            src, dst = pos_edge_index
            pos_score = torch.sigmoid((z[src] * z[dst]).sum(dim=-1))
            pos_loss = -torch.log(pos_score + EPS).mean()

            # 负边损失
            n_src, n_dst = neg_edge_index
            neg_score = torch.sigmoid((z[n_src] * z[n_dst]).sum(dim=-1))
            neg_loss = -torch.log(1 - neg_score + EPS).mean()

            losses.append(pos_loss + neg_loss)

        return self.mrl_weight * torch.stack(losses, dim=0).mean()

    def ssl_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        x = data.x.to(device)
        e = data.edge_index.to(device)
        z = self.model(x, e)
        loss = self._gae_mrl_loss(z, e)
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
            z = self.model(batch.x, batch.edge_index)
            loss = self._gae_mrl_loss(z, batch.edge_index)
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


# ============================================================================
# VGAE + MRL
# ============================================================================


class VGAEWithMRLMethod(VGAEMethod):
    """VGAE + MRL 融合方法。

    对每个指定的维度前缀独立计算 VGAE 损失（重建 + KL 散度），
    然后对各个维度的损失取平均。
    """

    def __init__(self, *args, mrl_dims: Sequence[int], mrl_weight: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.method_name = "vgae_mrl"
        self.mrl_dims = sorted({int(d) for d in mrl_dims})
        if len(self.mrl_dims) == 0:
            raise ValueError("mrl_dims 不能为空")
        if any(d <= 0 for d in self.mrl_dims):
            raise ValueError("mrl_dims 中所有维度必须 > 0")
        self.mrl_weight = float(mrl_weight)
        if self.hidden_dim < int(max(self.mrl_dims)):
            raise ValueError(
                f"VGAE+MRL 需要 --hidden-dim >= max(mrl_dims)={max(self.mrl_dims)}, "
                f"因为下游评估使用 encoder 输出（dim={self.hidden_dim}）切片, "
                f"当前 hidden-dim={self.hidden_dim} 不足"
            )
        self.last_mrl_dim_losses: Dict[str, float] = {}

    def output_dim(self) -> int:
        return self.hidden_dim

    def _vgae_mrl_loss(self, mu_full: Tensor, logvar_full: Tensor, z_full: Tensor, pos_edge_index: Tensor) -> Tensor:
        """对每个维度前缀独立计算 VGAE ELBO 损失。"""
        losses = []
        n = z_full.size(0)
        num_pos = pos_edge_index.size(1)

        # 共享同一组负边采样
        neg_edge_index = _sample_neg_edges(n, num_pos, z_full.device)

        for dim in self.mrl_dims:
            mu = mu_full[:, :dim]
            logvar = logvar_full[:, :dim]
            z = z_full[:, :dim]

            # 重建损失
            src, dst = pos_edge_index
            pos_score = torch.sigmoid((z[src] * z[dst]).sum(dim=-1))
            pos_loss = -torch.log(pos_score + EPS).mean()

            n_src, n_dst = neg_edge_index
            neg_score = torch.sigmoid((z[n_src] * z[n_dst]).sum(dim=-1))
            neg_loss = -torch.log(1 - neg_score + EPS).mean()

            recon = pos_loss + neg_loss

            # KL 散度
            kl = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            )

            losses.append(recon + kl)

        return self.mrl_weight * torch.stack(losses, dim=0).mean()

    def ssl_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        x = data.x.to(device)
        e = data.edge_index.to(device)
        z, mu, logvar = self.model(x, e)
        loss = self._vgae_mrl_loss(mu, logvar, z, e)
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
            z, mu, logvar = self.model(batch.x, batch.edge_index)
            loss = self._vgae_mrl_loss(mu, logvar, z, batch.edge_index)
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
