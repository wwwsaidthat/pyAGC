"""互学习损失计算模块 v2（全矩阵余弦相似度 + 对称 KL 散度 — 所有维度向最高维学习）。

与 ML_module (v4) 的核心区别：
    - v4：相邻维度对 (i-1, i) 互相学习，鼓励相邻子空间产生一致的相似度结构
    - v2：每个低维度向最高维度学习，所有子空间表示都对齐到最丰富的高维表示

实现公式（对每个低维度 i，以最高维度 M 为目标）：

Step 1 — 计算最高维度的完整 B×B 相似度矩阵（作为目标分布）：
    S_max = z1_norm[:, :max_dim] @ z2_norm[:, :max_dim].T       →  (B, B)

Step 2 — 对每个低维度 i（i < max_dim）：
    S_i = z1_norm[:, :dim_i] @ z2_norm[:, :dim_i].T              →  (B, B)

Step 3 — 拉平 + 温度缩放 + softmax：
    S̃_i = softmax(flatten(S_i) / τ_ml)                           →  (B²,)
    S̃_max = softmax(flatten(S_max) / τ_ml)                       →  (B²,)

Step 4 — 对称 KL 散度：
    L_i = ½ [KL(S̃_i || S̃_max) + KL(S̃_max || S̃_i)]

Step 5 — 总互学习损失：
    L_ML = ml_weight × (1/(|M|-1)) × Σ_{dim_i < max_dim} L_i
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# 复用 ML_module 的基础组件（compute_cross_view_similarity_matrix / temperature_softmax / symmetric_kl_divergence）
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


def temperature_softmax(logits: Tensor, tau_ml: float) -> Tensor:
    r"""温度缩放后做 softmax，将任意形状的 logits 转化为概率分布。

    prob = softmax(logits / tau_ml)

    参数:
        logits: 任意形状的 Tensor，会被自动视为一维参与 softmax
        tau_ml: 温度系数，控制分布的尖锐程度
            - tau_ml → 0+: 分布趋向 one-hot
            - tau_ml → ∞:  分布趋向均匀
            - tau_ml = 1:   标准 softmax

    返回:
        prob: softmax 概率分布，与 logits 同形状；sum(prob) = 1
    """
    prob = F.relu(logits)
    return prob


def symmetric_kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    r"""计算两个概率分布之间的对称 KL 散度（目标分布 detach）。

    L = ½ [KL(p || q) + KL(q || p)]

    关键设计：每个 KL 方向中，作为"目标"的分布会被 detach，
    梯度只流经"学生"一侧，避免两边同时更新导致训练不稳定。

    参数:
        p: 第一个概率分布，任意形状；要求 sum(p) ≈ 1
        q: 第二个概率分布，任意形状；要求 sum(q) ≈ 1
        eps: 数值稳定项，对概率做 clamp 防止 log(0)

    返回:
        loss: 对称 KL 散度标量
    """
    p_clamp = p.clamp(min=eps)
    q_clamp = q.clamp(min=eps)

    # KL(p || q): q 作为目标 detach，梯度只流经 p
    kl_pq = (p * (p_clamp.log() - q_clamp.detach().log())).sum()
    # KL(q || p): p 作为目标 detach，梯度只流经 q
    kl_qp = (q * (q_clamp.log() - p_clamp.detach().log())).sum()

    loss = 0.5 * (kl_pq + kl_qp)
    return loss


# ============================================================================
# 核心：所有维度向最高维度学习
# ============================================================================

def compute_all_to_max_loss(
    z1_full: Tensor,
    z2_full: Tensor,
    mrl_dims: Sequence[int],
    tau_ml: float,
    ml_weight: float = 1.0,
) -> Tensor:
    r"""所有低维度向最高维度的 B×B 跨视图相似度矩阵学习。

    对每个低维度 i（dim_i < max_dim）：
    1. 切片 z[:, :dim_i]，计算 B×B 跨视图相似度矩阵
    2. 拉平 → 温度缩放 softmax → B² 维概率分布
    3. 与最大维度的目标概率分布做对称 KL 散度

    总损失 = ml_weight × 均值(所有低维度与最大维度的对称 KL)

    参数:
        z1_full: 第一视图的完整 embedding，形状 (B, proj_dim)
        z2_full: 第二视图的完整 embedding，形状 (B, proj_dim)
        mrl_dims: MRL 维度列表，例如 [64, 128, 256, 512]
        tau_ml: 互学习温度系数
        ml_weight: 互学习损失权重

    返回:
        ml_loss: 互学习损失标量（已包含 ml_weight）
    """
    if not mrl_dims or len(mrl_dims) < 2:
        return torch.zeros((), device=z1_full.device)

    sorted_dims = sorted(mrl_dims)
    max_dim = sorted_dims[-1]

    # 计算最高维度的目标分布
    z1_max = z1_full[:, :max_dim]
    z2_max = z2_full[:, :max_dim]
    S_max = compute_cross_view_similarity_matrix(z1_max, z2_max)  # (B, B)
    target_prob = temperature_softmax(S_max.flatten(), tau_ml)    # (B²,)

    # 每个低维度与最高维度做对称 KL
    ml_sum = torch.zeros((), device=z1_full.device)
    low_dims = sorted_dims[:-1]  # 除最大维度外的所有维度

    for dim in low_dims:
        z1_i = z1_full[:, :dim]
        z2_i = z2_full[:, :dim]
        S_i = compute_cross_view_similarity_matrix(z1_i, z2_i)  # (B, B)
        curr_prob = temperature_softmax(S_i.flatten(), tau_ml)  # (B²,)

        # 对称 KL：低维度分布 vs 最高维度目标分布
        loss_pair = symmetric_kl_divergence(curr_prob, target_prob)
        ml_sum = ml_sum + loss_pair

    # 取均值 × 权重
    ml_loss = ml_weight * ml_sum / len(low_dims)
    return ml_loss


# ============================================================================
# MutualLearningLoss2 类（高级接口）
# ============================================================================

class MutualLearningLoss2:
    r"""互学习损失计算器 v2（所有维度向最高维度学习）。

    与 v4 的区别：
        - v4：相邻维度对 (i-1, i) 互学习
        - v2：所有低维度向最高维度学习

    提供面向对象的接口，便于在 GRACE+MRL+ML 等方法中复用。

    使用示例:
        ml_calculator = MutualLearningLoss2(
            mrl_dims=[64, 128, 256, 512],
            ml_weight=5.0,
            tau_ml=0.5,
        )

        # 直接传入完整投影器输出
        ml_loss = ml_calculator.compute(z1_full, z2_full)
    """

    def __init__(
        self,
        mrl_dims: Sequence[int],
        ml_weight: float = 1.0,
        tau_ml: float = 0.5,
    ):
        """
        参数:
            mrl_dims: MRL 维度列表，例如 [64, 128, 256, 512]
            ml_weight: 互学习损失权重
            tau_ml: 互学习温度系数，控制 softmax 的尖锐程度
                - 越小 → 分布越尖锐（只关注最相似/最不相似的节点）
                - 越大 → 分布越平滑（所有节点等权）
                - 建议范围: 0.1 ~ 1.0
        """
        self.mrl_dims = sorted({int(d) for d in mrl_dims})
        self.ml_weight = ml_weight
        self.tau_ml = tau_ml

        if len(self.mrl_dims) < 2:
            import warnings
            warnings.warn(
                f"mrl_dims 维度少于 2 个（当前 {len(self.mrl_dims)} 个），"
                f"互学习损失将恒为 0"
            )

    def compute(self, z1_full: Tensor, z2_full: Tensor) -> Tensor:
        r"""从投影器输出计算互学习损失（v2 — 所有维度向最高维学习）。

        参数:
            z1_full: 第一视图的投影 embedding，形状 (B, proj_dim)
            z2_full: 第二视图的投影 embedding，形状 (B, proj_dim)

        返回:
            ml_loss: 互学习损失标量（已包含 ml_weight）
        """
        return compute_all_to_max_loss(
            z1_full=z1_full,
            z2_full=z2_full,
            mrl_dims=self.mrl_dims,
            tau_ml=self.tau_ml,
            ml_weight=self.ml_weight,
        )
