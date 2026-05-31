"""MRL 融合模块。"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pyagc.transforms import GSSLTransform


class MRLFusionModule(nn.Module):
    """MRL 融合模块：一次训练联合学习多个维度表示。"""

    def __init__(
        self,
        hidden_dim: int,
        dims: Sequence[int],
        tau: float,
        weight: float,
        p_feat_mask_1: float,
        p_edge_drop_1: float,
        p_feat_mask_2: float,
        p_edge_drop_2: float,
    ) -> None:
        super().__init__()
        cleaned = [int(d) for d in dims]
        if len(cleaned) == 0:
            raise ValueError("MRL dims 不能为空")
        if any(d <= 0 for d in cleaned):
            raise ValueError("MRL dims 中所有维度必须 > 0")
        self.dims = sorted(set(cleaned))
        self.max_dim = int(max(self.dims))
        self.projector = nn.Identity() if hidden_dim == self.max_dim else nn.Linear(hidden_dim, self.max_dim)
        self.tau = tau
        self.weight = weight
        self.t1 = GSSLTransform(
            p_feat_mask=p_feat_mask_1,
            p_edge_drop=p_edge_drop_1,
            node_attrs=["x"],
            edge_attrs=[],
        )
        self.t2 = GSSLTransform(
            p_feat_mask=p_feat_mask_2,
            p_edge_drop=p_edge_drop_2,
            node_attrs=["x"],
            edge_attrs=[],
        )

    def output_dim(self) -> int:
        """MRL 输出维度（最大维度）。"""
        return int(self.max_dim)

    @staticmethod
    def _nt_xent(z1: Tensor, z2: Tensor, tau: float) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim = torch.mm(z1, z2.t()) / tau
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))

    def mrl_loss_with_details(
        self,
        encode_fn,
        x: Tensor,
        edge_index: Tensor,
        seed_size: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """计算 MRL 融合损失，并返回每个维度对应的损失。"""
        v1 = self.t1(x, edge_index)
        v2 = self.t2(x, edge_index)
        h1 = encode_fn(v1["x"], v1["edge_index"])
        h2 = encode_fn(v2["x"], v2["edge_index"])
        if seed_size is not None:
            h1 = h1[:seed_size]
            h2 = h2[:seed_size]
        z1_full = self.projector(h1)
        z2_full = self.projector(h2)
        dim_losses: Dict[str, float] = {}
        loss_sum = torch.zeros((), device=h1.device)
        for dim in self.dims:
            z1 = z1_full[:, :dim]
            z2 = z2_full[:, :dim]
            dim_loss = self._nt_xent(z1, z2, self.tau)
            loss_sum = loss_sum + dim_loss
            dim_losses[f"dim_{dim}"] = float(dim_loss.detach().item())
        total = self.weight * loss_sum / len(self.dims)
        return total, dim_losses

    def mrl_loss(
        self,
        encode_fn,
        x: Tensor,
        edge_index: Tensor,
        seed_size: Optional[int] = None,
    ) -> Tensor:
        """兼容旧调用：仅返回总损失。"""
        total, _ = self.mrl_loss_with_details(encode_fn, x, edge_index, seed_size=seed_size)
        return total

    def fuse(self, h: Tensor) -> Tensor:
        """将编码器输出映射到最大维度表示（用于下游与前缀评估）。"""
        return self.projector(h)
