"""CDMD：所有低维前缀向最高维前缀学习。

与 CDMD_module（相邻维度版本）的核心区别：
    - v4：相邻维度对 (i-1, i) 互相学习，鼓励相邻子空间产生一致的相似度结构
    - v2：每个低维度向最高维度学习，所有子空间表示都对齐到最丰富的高维表示

实现公式（对每个低维度 i，以最高维度 M 为目标）：

Step 1 — 计算最高维度的完整 B×B 相似度矩阵（作为目标分布）：
    S_max = z1_norm[:, :max_dim] @ z2_norm[:, :max_dim].T       →  (B, B)

Step 2 — 对每个低维度 i（i < max_dim）：
    S_i = z1_norm[:, :dim_i] @ z2_norm[:, :dim_i].T              →  (B, B)

Step 3 — 拉平 + ReLU：
    r_i = ReLU(flatten(S_i))                                      →  (B²,)
    r_max = ReLU(flatten(S_max))                                  →  (B²,)

Step 4 — 按样本数归一化的双向相对熵：
    L_i = (1/2B) [D(r_i || r_max) + D(r_max || r_i)]

Step 5 — 总互学习损失：
    L_CDMD = cdmd_weight × (1/(|M|-1)) × Σ_{dim_i < max_dim} L_i
"""

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# 与相邻维度版本保持相同的相似度、ReLU 证据和双向相对熵定义。
# ============================================================================

def compute_cross_view_similarity_matrix(z1: Tensor, z2: Tensor, eps: float = 1e-8) -> Tensor:
    r"""计算两个视图归一化嵌入的完整 B×B 相似度矩阵。

    S[j,k] = cosine_sim(z1_j, z2_k)
           = (z1_j · z2_k) / (||z1_j|| · ||z2_k||)

    参数:
        z1: 第一视图的 embedding，形状 (B, D)
        z2: 第二视图的 embedding，形状 (B, D)
        eps: 归一化数值稳定项

    返回:
        S: B×B 相似度矩阵
    """
    z1_norm = F.normalize(z1, p=2, dim=-1, eps=eps)
    z2_norm = F.normalize(z2, p=2, dim=-1, eps=eps)
    S = z1_norm @ z2_norm.T  # (B, B)
    return S


def positive_similarity_evidence(logits: Tensor, tau: float = 0.5) -> Tensor:
    r"""用 ReLU 保留正相似度并抑制负相似度。

    ``tau`` 仅为旧 checkpoint/命令兼容而保留，不参与计算。

    参数:
        logits: 任意形状的 Tensor

    返回:
        非负关系证据，与 logits 同形状
    """
    del tau
    return F.relu(logits)


# Backward-compatible alias for old imports/checkpoints.
temperature_softmax = positive_similarity_evidence


def symmetric_kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    r"""计算非负关系证据之间的双向相对熵（目标侧 detach）。

    L = (1/2B) [D(p || q) + D(q || p)]

    关键设计：每个 KL 方向中，作为"目标"的分布会被 detach，
    梯度只流经"学生"一侧，避免两边同时更新导致训练不稳定。

    参数:
        p: 第一组非负关系证据，形状 (B²,)
        q: 第二组非负关系证据，形状 (B²,)
        eps: 数值稳定项，对关系证据做 clamp 防止 log(0)

    返回:
        loss: 按样本数归一化的双向相对熵
    """
    p_clamp = p.clamp(min=eps)
    q_clamp = q.clamp(min=eps)

    # KL(p || q): q 作为目标 detach，梯度只流经 p
    kl_pq = (p * (p_clamp.log() - q_clamp.detach().log())).sum()
    # KL(q || p): p 作为目标 detach，梯度只流经 q
    kl_qp = (q * (q_clamp.log() - p_clamp.detach().log())).sum()

    numel = int(p.numel())
    batch_size = math.isqrt(numel)
    if batch_size * batch_size != numel:
        raise ValueError(f"CDMD 期望 B² 个关系元素，实际得到 {numel}")
    loss = 0.5 * (kl_pq + kl_qp) / batch_size
    return loss


# ============================================================================
# 核心：所有维度向最高维度学习
# ============================================================================

def compute_all_to_max_loss(
    z1_full: Tensor,
    z2_full: Tensor,
    mrl_dims: Sequence[int],
    cdmd_weight: float = 1.0,
    tau: float = 0.5,
) -> Tensor:
    r"""所有低维度向最高维度的 B×B 跨视图相似度矩阵学习。

    对每个低维度 i（dim_i < max_dim）：
    1. 切片 z[:, :dim_i]，计算 B×B 跨视图相似度矩阵
    2. 拉平 → ReLU → B² 维非负关系证据
    3. 与最大维度的证据做双向相对熵并除以样本数 B

    总损失 = cdmd_weight × 均值(所有低维度与最大维度的对称 KL)

    参数:
        z1_full: 第一视图的完整表示，形状 (B, hidden_dim)
        z2_full: 第二视图的完整表示，形状 (B, hidden_dim)
        mrl_dims: MRL 维度列表，例如 [64, 128, 256, 512]
        cdmd_weight: CDMD 损失权重

    返回:
        cdmd_loss: CDMD 损失标量（已包含 cdmd_weight）
    """
    if not mrl_dims or len(mrl_dims) < 2:
        return torch.zeros((), device=z1_full.device)

    sorted_dims = sorted(mrl_dims)
    max_dim = sorted_dims[-1]

    # 计算最高维度的参考关系证据
    z1_max = z1_full[:, :max_dim]
    z2_max = z2_full[:, :max_dim]
    S_max = compute_cross_view_similarity_matrix(z1_max, z2_max)  # (B, B)
    target_evidence = positive_similarity_evidence(S_max.flatten(), tau=tau)  # (B²,)

    # 每个低维度与最高维度做对称 KL
    cdmd_sum = torch.zeros((), device=z1_full.device)
    low_dims = sorted_dims[:-1]  # 除最大维度外的所有维度

    for dim in low_dims:
        z1_i = z1_full[:, :dim]
        z2_i = z2_full[:, :dim]
        S_i = compute_cross_view_similarity_matrix(z1_i, z2_i)  # (B, B)
        curr_evidence = positive_similarity_evidence(S_i.flatten(), tau=tau)  # (B²,)

        # 双向相对熵：低维关系证据 vs 最高维关系证据
        loss_pair = symmetric_kl_divergence(curr_evidence, target_evidence)
        cdmd_sum = cdmd_sum + loss_pair

    # 取均值 × 权重
    cdmd_loss = cdmd_weight * cdmd_sum / len(low_dims)
    return cdmd_loss


# ============================================================================
# CDMDLoss2 类（高级接口）
# ============================================================================

class CDMDLoss2:
    r"""互学习损失计算器 v2（所有维度向最高维度学习）。

    与 v4 的区别：
        - v4：相邻维度对 (i-1, i) 互学习
        - v2：所有低维度向最高维度学习

    提供面向对象的接口，便于在 GRACE+MRL+ML 等方法中复用。

    使用示例:
        cdmd_calculator = CDMDLoss2(
            mrl_dims=[64, 128, 256, 512],
            cdmd_weight=5.0,
        )

        # 直接传入完整投影器输出
        cdmd_loss = cdmd_calculator.compute(z1_full, z2_full)
    """

    def __init__(
        self,
        mrl_dims: Sequence[int],
        cdmd_weight: float = 1.0,
        tau: float = 0.5,
    ):
        """
        参数:
            mrl_dims: MRL 维度列表，例如 [64, 128, 256, 512]
            cdmd_weight: CDMD 损失权重
        """
        self.mrl_dims = sorted({int(d) for d in mrl_dims})
        self.cdmd_weight = cdmd_weight
        self.tau = float(tau)

        if len(self.mrl_dims) < 2:
            import warnings
            warnings.warn(
                f"mrl_dims 维度少于 2 个（当前 {len(self.mrl_dims)} 个），"
                f"互学习损失将恒为 0"
            )

    def compute(self, z1_full: Tensor, z2_full: Tensor) -> Tensor:
        r"""从投影器输出计算互学习损失（v2 — 所有维度向最高维学习）。

        参数:
            z1_full: 第一视图的编码器输出，形状 (B, hidden_dim)
            z2_full: 第二视图的编码器输出，形状 (B, hidden_dim)

        返回:
            cdmd_loss: CDMD 损失标量（已包含 cdmd_weight）
        """
        return compute_all_to_max_loss(
            z1_full=z1_full,
            z2_full=z2_full,
            mrl_dims=self.mrl_dims,
            cdmd_weight=self.cdmd_weight,
            tau=self.tau,
        )
