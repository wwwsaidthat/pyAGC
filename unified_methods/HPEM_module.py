"""HPEM (Hard-Pair Evolutionary Mining) 损失计算模块。

实现论文 CSNE 框架 Section 3.3 的 HPEM 机制：
大维度 prefix 的 InfoNCE loss 按 anchor 重新加权——在小维度 prefix 上
confusion score 低的"困难节点"获得更大权重，引导新增维度优先解决困难节点对。

核心公式：
    c_u^{(i-1)} = sim(h_u^{(i-1)}, h_{v^+}^{(i-1)}) - max_{v^-} sim(h_u^{(i-1)}, h_{v^-}^{(i-1)})
    w_u^{(i)} = softmax(-β · c_u^{(i-1)} / τ_i)
    L_HPEM^i = |B| · Σ_u w_u^{(i)} · ℓ_InfoNCE(u; H^{(i)})

模块职责：
    - 输入：当前维度 prefix 嵌入 (z1_curr, z2_curr) 和上一维度 prefix 嵌入 (z1_prev, z2_prev)
    - 用小维度计算 confusion score → softmax 得到 per-anchor 权重
    - 用大维度计算 per-anchor InfoNCE → 加权求和
    - 完全独立、可复用，不依赖任何特定方法类
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# 基础工具函数
# ============================================================================

def _rowwise_cosine_similarity(z1: Tensor, z2: Tensor, eps: float = 1e-8) -> Tensor:
    r"""计算两个视图之间的 B×B 余弦相似度矩阵。

    S[u, v] = cos_sim(z1_u, z2_v)

    参数:
        z1: (B, D) 第一视图嵌入
        z2: (B, D) 第二视图嵌入
        eps: 归一化稳定项

    返回:
        S: (B, B) 相似度矩阵
    """
    z1_norm = F.normalize(z1, p=2, dim=-1, eps=eps)
    z2_norm = F.normalize(z2, p=2, dim=-1, eps=eps)
    return z1_norm @ z2_norm.T


# ============================================================================
# Step 1: 计算 confusion score
# ============================================================================

def compute_confusion_scores(
    z1_prev: Tensor,
    z2_prev: Tensor,
) -> Tensor:
    r"""用小维度 prefix 计算每个 anchor 的 confusion score。

    c_u = sim(h_u, h_{v^+}) - max_{v^-} sim(h_u, h_{v^-})

    其中 v^+ 是对角元素（同一个节点的两个增强视图），v^- 是非对角元素。
    c_u 越小（甚至为负），说明小维度 prefix 对这个 anchor 区分能力越弱。

    参数:
        z1_prev: (B, D_prev) 小维度 prefix 的第一视图嵌入
        z2_prev: (B, D_prev) 小维度 prefix 的第二视图嵌入

    返回:
        confusion: (B,) 每个 anchor 的 confusion score
    """
    S = _rowwise_cosine_similarity(z1_prev, z2_prev)  # (B, B)
    B = S.size(0)

    # 正样本对相似度：对角元素
    pos_sim = S.diag()  # (B,)

    # 最难负样本相似度：每行去掉对角后取最大值
    # 将对角设为极小值，然后 max
    S_no_diag = S.clone()
    S_no_diag[range(B), range(B)] = -float("inf")
    hardest_neg_sim = S_no_diag.max(dim=1).values  # (B,)

    confusion = pos_sim - hardest_neg_sim  # (B,)
    return confusion


# ============================================================================
# Step 2: confusion → per-anchor 权重
# ============================================================================

def confusion_to_weights(
    confusion: Tensor,
    tau: float,
    beta: float,
) -> Tensor:
    r"""将 confusion score 转化为 per-anchor 损失权重。

    w_u = softmax(-β · c_u / τ)

    c_u 越小（越困难）→ -β·c_u/τ 越大 → softmax 后权重越大。
    detach confusion 保证梯度不流回小维度 prefix。

    参数:
        confusion: (B,) confusion scores
        tau: 当前维度的 temperature
        beta: 可学习的 scaling parameter

    返回:
        weights: (B,) 归一化后的 per-anchor 权重，sum = 1
    """
    logits = -beta * confusion.detach() / tau  # (B,)
    weights = F.softmax(logits, dim=0)  # (B,)
    return weights


# ============================================================================
# Step 3: per-anchor InfoNCE loss
# ============================================================================

def compute_per_anchor_infonce(
    z1: Tensor,
    z2: Tensor,
    tau: float,
) -> Tensor:
    r"""计算每个 anchor 的 InfoNCE loss（不做 mean）。

    ℓ_u = -log( exp(cos(z1_u, z2_u) / τ) / Σ_v exp(cos(z1_u, z2_v) / τ) )
        = -S[u,u]/τ + logsumexp_v(S[u,v]/τ)

    参数:
        z1: (B, D) 第一视图嵌入
        z2: (B, D) 第二视图嵌入
        tau: temperature

    返回:
        per_anchor_loss: (B,) 每个 anchor 的 InfoNCE loss
    """
    S = _rowwise_cosine_similarity(z1, z2)  # (B, B)
    pos_sim = S.diag()  # (B,)
    # log-sum-exp over all candidates (including positive)
    log_sum_exp = torch.logsumexp(S / tau, dim=1)  # (B,)
    per_anchor_loss = -pos_sim / tau + log_sum_exp  # (B,)
    return per_anchor_loss


# ============================================================================
# Step 4: 完整的 HPEM loss
# ============================================================================

def compute_hpem_loss(
    z1_curr: Tensor,
    z2_curr: Tensor,
    z1_prev: Tensor,
    z2_prev: Tensor,
    tau: float,
    beta: float,
) -> Tensor:
    r"""计算单个维度 prefix i 的 HPEM 损失。

    1. 用小维度 prefix (i-1) 计算 confusion score
    2. confusion → softmax 权重
    3. 用大维度 prefix (i) 计算 per-anchor InfoNCE
    4. 加权求和 × batch_size

    L_HPEM^i = |B| · Σ_u w_u^{(i)} · ℓ_InfoNCE(u; H^{(i)})

    参数:
        z1_curr: (B, D_i) 当前（大）维度 prefix 的第一视图嵌入
        z2_curr: (B, D_i) 当前（大）维度 prefix 的第二视图嵌入
        z1_prev: (B, D_{i-1}) 上一（小）维度 prefix 的第一视图嵌入
        z2_prev: (B, D_{i-1}) 上一（小）维度 prefix 的第二视图嵌入
        tau: 当前维度的 temperature
        beta: 可学习 scaling parameter

    返回:
        hpem_loss: 标量
    """
    B = z1_curr.size(0)

    # 1. 用小维度算 confusion score（detach 在 confusion_to_weights 内部）
    confusion = compute_confusion_scores(z1_prev, z2_prev)  # (B,)

    # 2. confusion → weights
    weights = confusion_to_weights(confusion, tau, beta)  # (B,)

    # 3. 用大维度算 per-anchor InfoNCE
    per_anchor_loss = compute_per_anchor_infonce(z1_curr, z2_curr, tau)  # (B,)

    # 4. 加权求和
    hpem_loss = B * (weights * per_anchor_loss).sum()
    return hpem_loss


# ============================================================================
# HPEMLoss 类（高级接口，类似 MutualLearningLoss）
# ============================================================================

class HPEMLoss(nn.Module):
    r"""HPEM 损失计算器（可学习 β）。

    对每个维度 > 1，用小维度计算 confusion → 加权大维度的 InfoNCE。
    β 是可学习参数，按论文通过训练自动调整。

    使用示例:
        hpem = HPEMLoss(beta_init=0.1)
        loss = hpem.compute_single(
            z1_curr=z1[:, :512], z2_curr=z2[:, :512],
            z1_prev=z1[:, :256], z2_prev=z2[:, :256],
            tau=0.5,
        )
    """

    def __init__(self, beta_init: float = 0.1):
        super().__init__()
        # 可学习的 β：正数，用 softplus 保证 > 0
        self._beta_raw = nn.Parameter(
            torch.tensor(self._inv_softplus(beta_init))
        )

    @staticmethod
    def _inv_softplus(x: float) -> float:
        """softplus 的近似逆，用于合理的初始化。"""
        import math
        if x > 20:
            return x
        return math.log(math.expm1(x))

    @property
    def beta(self) -> Tensor:
        """保证 β 始终为正，训练稳定。"""
        return F.softplus(self._beta_raw)

    def compute_single(
        self,
        z1_curr: Tensor,
        z2_curr: Tensor,
        z1_prev: Tensor,
        z2_prev: Tensor,
        tau: float,
    ) -> Tensor:
        r"""计算单个维度 prefix i 的 HPEM 损失。

        参数:
            z1_curr, z2_curr: 当前维度 prefix 嵌入
            z1_prev, z2_prev: 上一维度 prefix 嵌入
            tau: temperature

        返回:
            hpem_loss: 标量
        """
        return compute_hpem_loss(
            z1_curr=z1_curr,
            z2_curr=z2_curr,
            z1_prev=z1_prev,
            z2_prev=z2_prev,
            tau=tau,
            beta=self.beta,
        )

    def compute_all(
        self,
        z1_full: Tensor,
        z2_full: Tensor,
        mrl_dims: list,
        tau: float,
    ) -> Tensor:
        r"""对所有相邻维度对计算 HPEM 损失总和。

        对每个 i > 0：用 dim[i-1] 做 confusion → 加权 dim[i] 的 InfoNCE。
        dim[0]（最小维度）不计算 HPEM（使用均匀权重，等价于标准 InfoNCE）。

        参数:
            z1_full: (B, max_dim) 完整嵌入
            z2_full: (B, max_dim) 完整嵌入
            mrl_dims: 维度列表，已排序，如 [64, 128, 256, 512]
            tau: temperature

        返回:
            total_hpem: 标量
        """
        sorted_dims = sorted(mrl_dims)
        if len(sorted_dims) < 2:
            return torch.zeros((), device=z1_full.device)

        total = torch.zeros((), device=z1_full.device)
        for i in range(1, len(sorted_dims)):
            dim_prev = sorted_dims[i - 1]
            dim_curr = sorted_dims[i]
            z1_prev = z1_full[:, :dim_prev]
            z2_prev = z2_full[:, :dim_prev]
            z1_curr = z1_full[:, :dim_curr]
            z2_curr = z2_full[:, :dim_curr]
            total = total + self.compute_single(
                z1_curr=z1_curr,
                z2_curr=z2_curr,
                z1_prev=z1_prev,
                z2_prev=z2_prev,
                tau=tau,
            )
        return total
