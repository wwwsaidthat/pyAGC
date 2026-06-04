"""互学习损失计算模块 v2 — z-level 短梯度路径。

与 ML_module.py (v1) 的核心区别 —— 梯度路径长度：

v1 (State-based):
    ML_loss → State → p(u,v) → softmax → sim → normalized_z → projector → encoder
    └────────────────────── 8 步，层层衰减 ──────────────────────┘

v2 (z-level):
    ML_loss → z → projector → encoder
    └────────── 3 步，和 GRACE NT-Xent 同级 ──────────┘

直接在 projector 输出 (z) 上约束相邻维度的正样本对相似度分布一致。
对每对相邻维度，将正样本对余弦相似度 Z-score 标准化后做 MSE，
消除不同维度间的系统性偏移，只约束相对分布。
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_z_level_ml_loss(
    z1_full: Tensor,
    z2_full: Tensor,
    mrl_dims: Sequence[int],
    ml_weight: float = 1.0,
) -> Tensor:
    """z-level 互学习损失。

    梯度路径: loss → z → projector → encoder（3 步）

    对每对相邻维度 (dim_{i-1}, dim_i):
      1. L2 归一化 z1, z2
      2. 提取正样本对余弦相似度 s_k = cos(z1[k], z2[k])
      3. Z-score 标准化: s_z = (s - mean(s)) / (std(s) + eps)
         目的: 消除高维嵌入天然更大的绝对相似度
      4. MSE: mean((s_z^{prev} - s_z^{curr})²)

    参数:
        z1_full: 视图1 的完整投影嵌入 (N, max_dim)
        z2_full: 视图2 的完整投影嵌入 (N, max_dim)
        mrl_dims: MRL 维度列表
        ml_weight: 互学习损失权重

    返回:
        ml_loss: 标量。相邻维度高度相关时 ~0.5，
                 ml_weight=1 时 ML/GRACE 占比约 5-15%
    """
    sorted_dims = sorted(mrl_dims)
    if len(sorted_dims) < 2:
        return torch.tensor(0.0, device=z1_full.device)

    device = z1_full.device
    ml_loss = torch.zeros((), device=device)
    count = 0

    for i in range(1, len(sorted_dims)):
        d_prev = sorted_dims[i - 1]
        d_curr = sorted_dims[i]

        # L2 归一化
        z1_p = F.normalize(z1_full[:, :d_prev], dim=-1)
        z2_p = F.normalize(z2_full[:, :d_prev], dim=-1)
        z1_c = F.normalize(z1_full[:, :d_curr], dim=-1)
        z2_c = F.normalize(z2_full[:, :d_curr], dim=-1)

        # 正样本对余弦相似度（对角线元素）
        sim_p = (z1_p * z2_p).sum(dim=-1)  # (N,)
        sim_c = (z1_c * z2_c).sum(dim=-1)  # (N,)

        # Z-score 标准化: 只比较跨样本的相对分布
        s_p = (sim_p - sim_p.mean()) / (sim_p.std() + 1e-8)
        s_c = (sim_c - sim_c.mean()) / (sim_c.std() + 1e-8)

        ml_loss = ml_loss + ((s_p - s_c) ** 2).mean()
        count += 1

    return ml_weight * ml_loss / max(count, 1)


class ZLevelMutualLearningLoss:
    """z-level 互学习损失计算器。

    梯度路径: loss → z → projector → encoder (3 步, 和 GRACE 同级)

    使用示例:
        ml_calc = ZLevelMutualLearningLoss(
            mrl_dims=[32, 64, 128, 256, 384, 512, 768],
            ml_weight=1.0,
        )

        # 在训练循环中
        ml_loss = ml_calc.compute(z1_full, z2_full)
        total = mrl_loss + ml_loss
    """

    def __init__(self, mrl_dims: Sequence[int], ml_weight: float = 1.0):
        self.mrl_dims = sorted({int(d) for d in mrl_dims})
        self.ml_weight = float(ml_weight)

        if len(self.mrl_dims) < 2:
            import warnings
            warnings.warn("mrl_dims 维度少于2个，互学习损失将恒为0")

    def compute(self, z1_full: Tensor, z2_full: Tensor) -> Tensor:
        """从 projector 输出直接计算 z-level 互学习损失。

        参数:
            z1_full: 视图1 的完整投影嵌入 (N, max_dim)
            z2_full: 视图2 的完整投影嵌入 (N, max_dim)

        返回:
            ml_loss: 标量（已包含 ml_weight）
        """
        return compute_z_level_ml_loss(
            z1_full, z2_full, self.mrl_dims, self.ml_weight
        )
