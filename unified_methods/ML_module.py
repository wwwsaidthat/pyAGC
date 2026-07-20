"""互学习损失计算模块（v4 — 全矩阵余弦相似度 + 对称 KL 散度）。

实现公式（对每对相邻维度 i-1, i）：

Step 1 — 计算两个视图归一化嵌入的完整 B×B 相似度矩阵：
    S_i = z1_norm @ z2_norm.T       →  (B, B)
    对角元素 S_i[j,j] = 正样本对 (z1_j, z2_j) 的余弦相似度
    非对角元素 S_i[j,k] = 负样本对 (z1_j, z2_k) 的余弦相似度

Step 2 — 拉平 + 温度缩放 + softmax：
    S̃_i = softmax(flatten(S_i) / τ_ml)       →  (B²,)

Step 3 — 对称 KL 散度：
    L_ML^{i-1,i} = ½ [KL(S̃_{i-1} || S̃_i) + KL(S̃_i || S̃_{i-1})]

Step 4 — 总互学习损失：
    L_ML = ml_weight × (1/(|M|-1)) × Σ_{i=1}^{|M|-1} L_ML^{i-1,i}

与 v3（仅正样本对）的核心区别：
    - v3 只取矩阵对角（B 个正样本对）的相似度，softmax 后得到 B 维分布
    - v4 取完整 B×B 矩阵（含所有正样本 + 负样本对），softmax 后得到 B² 维分布，
      衡量的是"整个跨视图相似度矩阵结构"在相邻维度间的一致性
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# Step 1: 完整 B×B 跨视图相似度矩阵
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


# ============================================================================
# Step 2: 温度缩放 + softmax（在拉平后的 B² 向量上操作）
# ============================================================================

def temperature_softmax(logits: Tensor, tau: float = 0.5) -> Tensor:
    r"""用温度缩放 softmax 将 logits 转成合法概率分布。

    prob = softmax(logits / tau)

    参数:
        logits: 任意形状的 Tensor

    返回:
        prob: 和为 1 的概率分布，与 logits 同形状
    """
    if tau <= 0:
        raise ValueError(f"CDMD temperature 必须 > 0，当前为 {tau}")
    return F.softmax(logits / tau, dim=-1)


# ============================================================================
# Step 3: 对称 KL 散度
# ============================================================================

def symmetric_kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    r"""计算两个概率分布之间的对称 KL 散度（目标分布 detach）。

    L_ML^{i-1,i} = ½ [KL(p || q) + KL(q || p)]

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

    # p/q 已由 softmax 归一化，KL 本身不需要再除以 batch size。
    loss = 0.5 * (kl_pq + kl_qp)
    return loss


# ============================================================================
# Step 4: 完整互学习损失 pipeline（v4 — 全矩阵 B×B）
# ============================================================================

def compute_mutual_learning_loss_v4(
    z1_full: Tensor,
    z2_full: Tensor,
    mrl_dims: Sequence[int],
    ml_weight: float = 1.0,
    tau: float = 0.5,
) -> Tensor:
    r"""完整的互学习损失 pipeline（v4 — 全矩阵 B×B 相似度 + softmax）。

    对每对相邻维度 (i-1, i):
    1. 切片 z[:, :dim]，计算 B×B 跨视图相似度矩阵
    2. 拉平 → temperature-softmax → B² 维概率分布
    3. 对称 KL 散度

    总损失 = ml_weight × 均值(所有相邻对的对称 KL)

    内存优化：逐对处理相邻维度，始终只持有 2 个 B² 向量。

    参数:
        z1_full: 第一视图的完整 embedding，形状 (B, proj_dim)
        z2_full: 第二视图的完整 embedding，形状 (B, proj_dim)
        mrl_dims: MRL 维度列表，例如 [64, 128, 256, 512]
        ml_weight: 互学习损失权重

    返回:
        ml_loss: 互学习损失标量（已包含 ml_weight）
    """
    if not mrl_dims or len(mrl_dims) < 2:
        return torch.zeros((), device=z1_full.device)

    sorted_dims = sorted(mrl_dims)
    ml_sum = torch.zeros((), device=z1_full.device)

    # 先算第一个维度的 B×B 矩阵 → flatten → temperature-softmax
    prev_z1 = z1_full[:, :sorted_dims[0]]
    prev_z2 = z2_full[:, :sorted_dims[0]]
    prev_S = compute_cross_view_similarity_matrix(prev_z1, prev_z2)  # (B, B)
    prev_prob = temperature_softmax(prev_S.flatten(), tau=tau)        # (B²,)

    for i in range(1, len(sorted_dims)):
        dim = sorted_dims[i]
        curr_z1 = z1_full[:, :dim]
        curr_z2 = z2_full[:, :dim]
        curr_S = compute_cross_view_similarity_matrix(curr_z1, curr_z2)  # (B, B)
        curr_prob = temperature_softmax(curr_S.flatten(), tau=tau)       # (B²,)

        # 对称 KL：prev_prob vs curr_prob
        loss_pair = symmetric_kl_divergence(prev_prob, curr_prob)
        ml_sum = ml_sum + loss_pair

        # 释放 prev，复用为下一轮
        prev_prob = curr_prob

    # 取均值 × 权重
    ml_loss = ml_weight * ml_sum / (len(sorted_dims) - 1)
    return ml_loss


# ============================================================================
# MutualLearningLoss 类（高级接口）
# ============================================================================

class MutualLearningLoss:
    r"""互学习损失计算器（v4 — 全 B×B 矩阵 + temperature-softmax + 对称 KL）。

    提供面向对象的接口，便于在 GRACE+MRL+ML 等方法中复用。

    使用示例:
        ml_calculator = MutualLearningLoss(
            mrl_dims=[64, 128, 256, 512],
            ml_weight=5.0,
        )

        # 直接传入完整投影器输出
        ml_loss = ml_calculator.compute(z1_full, z2_full)
    """

    def __init__(
        self,
        mrl_dims: Sequence[int],
        ml_weight: float = 1.0,
        tau: float = 0.5,
    ):
        """
        参数:
            mrl_dims: MRL 维度列表，例如 [64, 128, 256, 512]
            ml_weight: 互学习损失权重
        """
        self.mrl_dims = sorted({int(d) for d in mrl_dims})
        self.ml_weight = ml_weight
        self.tau = float(tau)

        if len(self.mrl_dims) < 2:
            import warnings
            warnings.warn(
                f"mrl_dims 维度少于 2 个（当前 {len(self.mrl_dims)} 个），"
                f"互学习损失将恒为 0"
            )

    def compute(self, z1_full: Tensor, z2_full: Tensor) -> Tensor:
        r"""从投影器输出计算互学习损失（v4 — 全 B×B 矩阵）。

        参数:
            z1_full: 第一视图的投影 embedding，形状 (B, proj_dim)
            z2_full: 第二视图的投影 embedding，形状 (B, proj_dim)

        返回:
            ml_loss: 互学习损失标量（已包含 ml_weight）
        """
        return compute_mutual_learning_loss_v4(
            z1_full=z1_full,
            z2_full=z2_full,
            mrl_dims=self.mrl_dims,
            ml_weight=self.ml_weight,
            tau=self.tau,
        )
