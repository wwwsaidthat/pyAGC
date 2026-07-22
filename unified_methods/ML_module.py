"""互学习损失计算模块（相邻维度版本）。

实现公式（对每对相邻维度 i-1, i）：

Step 1 — 计算两个视图归一化嵌入的完整 B×B 相似度矩阵：
    S_i = z1_norm @ z2_norm.T       →  (B, B)
    对角元素 S_i[j,j] = 正样本对 (z1_j, z2_j) 的余弦相似度
    非对角元素 S_i[j,k] = 负样本对 (z1_j, z2_k) 的余弦相似度

Step 2 — 拉平 + ReLU：
    r_i = ReLU(flatten(S_i))                 →  (B²,)

Step 3 — 按样本数归一化的双向相对熵：
    L_ML^{i-1,i} = (1/2B) [D(r_{i-1} || r_i) + D(r_i || r_{i-1})]

Step 4 — 总互学习损失：
    L_ML = ml_weight × (1/(|M|-1)) × Σ_{i=1}^{|M|-1} L_ML^{i-1,i}

与仅对齐正样本的版本相比，当前实现取完整 B×B 矩阵（包含正、负
样本关系），经 ReLU 后衡量整个跨视图关系结构在相邻维度间的一致性。
"""

import math
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
# Step 2: ReLU 正关系证据（在拉平后的 B² 向量上操作）
# ============================================================================

def positive_similarity_evidence(logits: Tensor, tau: float = 0.5) -> Tensor:
    r"""保留正相似度并抑制负相似度。

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


# ============================================================================
# Step 3: 对称 KL 散度
# ============================================================================

def symmetric_kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    r"""计算非负关系证据之间的双向相对熵（目标侧 detach）。

    L_ML^{i-1,i} = (1/2B) [D(p || q) + D(q || p)]

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

    # p/q 来自展平的 B×B 关系矩阵。对所有关系求和后按 anchor
    # 样本数 B 取平均，而不是按 B² 个矩阵元素取平均。
    numel = int(p.numel())
    batch_size = math.isqrt(numel)
    if batch_size * batch_size != numel:
        raise ValueError(f"CDMD 期望 B² 个关系元素，实际得到 {numel}")
    loss = 0.5 * (kl_pq + kl_qp) / batch_size
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
    r"""完整的互学习损失 pipeline（全矩阵 B×B 相似度 + ReLU）。

    对每对相邻维度 (i-1, i):
    1. 切片 z[:, :dim]，计算 B×B 跨视图相似度矩阵
    2. 拉平 → ReLU → B² 维非负关系证据
    3. 双向相对熵求和后除以样本数 B

    总损失 = ml_weight × 均值(所有相邻对的对称 KL)

    内存优化：逐对处理相邻维度，始终只持有 2 个 B² 向量。

    参数:
        z1_full: 第一视图的完整表示，形状 (B, hidden_dim)
        z2_full: 第二视图的完整表示，形状 (B, hidden_dim)
        mrl_dims: MRL 维度列表，例如 [64, 128, 256, 512]
        ml_weight: 互学习损失权重

    返回:
        ml_loss: 互学习损失标量（已包含 ml_weight）
    """
    if not mrl_dims or len(mrl_dims) < 2:
        return torch.zeros((), device=z1_full.device)

    sorted_dims = sorted(mrl_dims)
    ml_sum = torch.zeros((), device=z1_full.device)

    # 先算第一个维度的 B×B 矩阵 → flatten → ReLU
    prev_z1 = z1_full[:, :sorted_dims[0]]
    prev_z2 = z2_full[:, :sorted_dims[0]]
    prev_S = compute_cross_view_similarity_matrix(prev_z1, prev_z2)  # (B, B)
    prev_evidence = positive_similarity_evidence(prev_S.flatten(), tau=tau)  # (B²,)

    for i in range(1, len(sorted_dims)):
        dim = sorted_dims[i]
        curr_z1 = z1_full[:, :dim]
        curr_z2 = z2_full[:, :dim]
        curr_S = compute_cross_view_similarity_matrix(curr_z1, curr_z2)  # (B, B)
        curr_evidence = positive_similarity_evidence(curr_S.flatten(), tau=tau)  # (B²,)

        loss_pair = symmetric_kl_divergence(prev_evidence, curr_evidence)
        ml_sum = ml_sum + loss_pair

        # 释放 prev，复用为下一轮
        prev_evidence = curr_evidence

    # 取均值 × 权重
    ml_loss = ml_weight * ml_sum / (len(sorted_dims) - 1)
    return ml_loss


# ============================================================================
# MutualLearningLoss 类（高级接口）
# ============================================================================

class MutualLearningLoss:
    r"""相邻维度 CDMD（全 B×B 矩阵 + ReLU + 双向相对熵）。

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
            z1_full: 第一视图的编码器输出，形状 (B, hidden_dim)
            z2_full: 第二视图的编码器输出，形状 (B, hidden_dim)

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
