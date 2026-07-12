"""HPEM (Hard-Pair Evolutionary Mining) 损失计算模块。

实现论文 CSNE 框架 Section 3.3 的 HPEM 机制，并内置维度自适应 temperature
（原 DAS 组件 A）——因为 τ_i 的唯一消费者就是 HPEM，放在这里消除不必要的参数传递。

核心公式：
    τ_i = τ_0 · exp(φ_1 · d_i/d_n + φ_2)                        ← 维度自适应温度
    c_u^{(i-1)} = sim(h_u^{(i-1)}, h_{v^+}^{(i-1)}) - max_{v^-} sim(...)
    w_u^{(i)} = softmax(-β · c_u^{(i-1)} / τ_i)
    L_HPEM^i = Σ_u w_u^{(i)} · ℓ_InfoNCE(u; H^{(i)})

可学习参数：
    β:      confusion → weight 缩放系数（softplus 保证 > 0）
    φ_1, φ_2: 维度自适应 temperature 参数
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# 基础工具函数
# ============================================================================

def _rowwise_cosine_similarity(z1: Tensor, z2: Tensor, eps: float = 1e-8) -> Tensor:
    r"""计算两个视图之间的 B×B 余弦相似度矩阵。"""
    z1_norm = F.normalize(z1, p=2, dim=-1, eps=eps)
    z2_norm = F.normalize(z2, p=2, dim=-1, eps=eps)
    return z1_norm @ z2_norm.T


def _inv_softplus(x: float) -> float:
    """softplus 的近似逆，用于合理的参数初始化。"""
    if x > 20:
        return x
    return math.log(math.expm1(x))


# ============================================================================
# Step 1: 计算 confusion score
# ============================================================================

def compute_confusion_scores(
    z1_prev: Tensor,
    z2_prev: Tensor,
) -> Tensor:
    r"""用小维度 prefix 计算每个 anchor 的 confusion score。

    c_u = sim(h_u, h_{v^+}) - max_{v^-} sim(h_u, h_{v^-})

    c_u 越小（甚至为负），说明小维度 prefix 对这个 anchor 区分能力越弱。

    参数:
        z1_prev: (B, D_prev) 小维度 prefix 的第一视图嵌入
        z2_prev: (B, D_prev) 小维度 prefix 的第二视图嵌入

    返回:
        confusion: (B,) 每个 anchor 的 confusion score
    """
    S = _rowwise_cosine_similarity(z1_prev, z2_prev)  # (B, B)
    B = S.size(0)

    pos_sim = S.diag()  # (B,)

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
    tau: Tensor,
    beta: Tensor,
) -> Tensor:
    r"""将 confusion score 转化为 per-anchor 损失权重。

    w_u = softmax(-β · c_u / τ)

    c_u 越小（越困难）→ -β·c_u/τ 越大 → softmax 后权重越大。
    detach confusion 保证梯度不流回小维度 prefix。

    参数:
        confusion: (B,) confusion scores
        tau: 标量 tensor，当前维度的 temperature
        beta: 标量 tensor，可学习 scaling parameter

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
    tau: Tensor,
) -> Tensor:
    r"""计算每个 anchor 的 InfoNCE loss（不做 mean）。

    ℓ_u = -S[u,u]/τ + logsumexp_v(S[u,v]/τ)

    参数:
        z1, z2: (B, D) 嵌入
        tau: 标量 tensor，temperature

    返回:
        per_anchor_loss: (B,)
    """
    S = _rowwise_cosine_similarity(z1, z2)  # (B, B)
    pos_sim = S.diag()  # (B,)
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
    tau: Tensor,
    beta: Tensor,
) -> Tensor:
    r"""计算单个维度 prefix i 的 HPEM 损失。

    L_HPEM^i = Σ_u w_u^{(i)} · ℓ_InfoNCE(u; H^{(i)})    （weighted mean 量级）

    参数:
        z1_curr, z2_curr: (B, D_i) 当前维度 prefix 嵌入
        z1_prev, z2_prev: (B, D_{i-1}) 上一维度 prefix 嵌入
        tau: 标量 tensor，当前维度的 temperature
        beta: 标量 tensor，可学习 scaling parameter

    返回:
        hpem_loss: 标量
    """

    confusion = compute_confusion_scores(z1_prev, z2_prev)  # (B,)
    weights = confusion_to_weights(confusion, tau, beta)  # (B,)
    per_anchor_loss = compute_per_anchor_infonce(z1_curr, z2_curr, tau)  # (B,)

    hpem_loss = (weights * per_anchor_loss).sum()
    return hpem_loss


# ============================================================================
# HPEMLoss 类（高级接口，内置维度自适应 temperature）
# ============================================================================

class HPEMLoss(nn.Module):
    r"""HPEM 损失计算器。

    可学习参数：
        β:       confusion → weight 缩放系数（softplus 保证 > 0）
        φ_1, φ_2: 维度自适应 temperature，τ_i = τ_0 · exp(φ_1 · d_i/d_n + φ_2)

    使用示例:
        hpem = HPEMLoss(tau_0=0.5, max_dim=768, beta_init=0.1)

        # 自动计算 τ_i，无需外部传入 tau
        loss = hpem.compute_single(
            z1_curr=z1[:, :512], z2_curr=z2[:, :512],
            z1_prev=z1[:, :256], z2_prev=z2[:, :256],
            dim_curr=512,
        )
    """

    def __init__(
        self,
        tau_0: float = 0.5,
        max_dim: int = 768,
        beta_init: float = 0.1,
    ):
        """
        参数:
            tau_0: 基础温度 τ_0
            max_dim: 最大维度 d_n，用于归一化 d_i / d_n
            beta_init: β 的初始值
        """
        super().__init__()
        self.tau_0 = float(tau_0)
        self.max_dim = int(max_dim)

        # β：confusion → weight 缩放
        self._beta_raw = nn.Parameter(
            torch.tensor(_inv_softplus(beta_init))
        )

        # φ_1, φ_2：维度自适应 temperature
        # 初始化 φ_1 = φ_2 = 0 → g_φ(1) = 1 → τ_max = τ_0
        self.phi_1 = nn.Parameter(torch.tensor(0.0))
        self.phi_2 = nn.Parameter(torch.tensor(0.0))

    @property
    def beta(self) -> Tensor:
        """保证 β 始终为正。"""
        return F.softplus(self._beta_raw)

    # ------------------------------------------------------------------
    # 维度自适应 temperature（原 DAS 组件 A）
    # ------------------------------------------------------------------

    def get_tau(self, dim: int) -> Tensor:
        r"""返回维度 dim 的 temperature。

        τ_i = τ_0 · exp(φ_1 · d_i/d_n + φ_2)

        参数:
            dim: 当前维度 d_i

        返回:
            tau_i: 标量 tensor
        """
        x = dim / self.max_dim
        g = torch.exp(self.phi_1 * x + self.phi_2)
        return torch.tensor(self.tau_0, device=g.device, dtype=g.dtype) * g

    # ------------------------------------------------------------------
    # 损失计算
    # ------------------------------------------------------------------

    def compute_single(
        self,
        z1_curr: Tensor,
        z2_curr: Tensor,
        z1_prev: Tensor,
        z2_prev: Tensor,
        dim_curr: int,
    ) -> Tensor:
        r"""计算单个维度 prefix i 的 HPEM 损失（自动使用该维度的 τ_i）。

        参数:
            z1_curr, z2_curr: 当前维度 prefix 嵌入
            z1_prev, z2_prev: 上一维度 prefix 嵌入
            dim_curr: 当前维度的数值，用于计算 τ_i

        返回:
            hpem_loss: 标量
        """
        tau = self.get_tau(dim_curr)
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
        mrl_dims: Sequence[int],
    ) -> Tensor:
        r"""对所有相邻维度对计算 HPEM 损失总和。

        对每个 i > 0：用 dim[i-1] 做 confusion → 加权 dim[i] 的 InfoNCE。
        dim[0]（最小维度）不计算 HPEM。

        参数:
            z1_full, z2_full: (B, max_dim) 完整嵌入
            mrl_dims: 维度列表，已排序，如 [64, 128, 256, 512]

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
            total = total + self.compute_single(
                z1_curr=z1_full[:, :dim_curr],
                z2_curr=z2_full[:, :dim_curr],
                z1_prev=z1_full[:, :dim_prev],
                z2_prev=z2_full[:, :dim_prev],
                dim_curr=dim_curr,
            )
        return total

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def log_info(self) -> dict:
        """返回当前可学习参数的快照，用于日志记录。"""
        with torch.no_grad():
            return {
                "hpem_beta": float(self.beta.item()),
                "hpem_phi_1": float(self.phi_1.item()),
                "hpem_phi_2": float(self.phi_2.item()),
            }
