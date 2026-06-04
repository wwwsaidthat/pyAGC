"""互学习损失计算模块 v3 — 节点难度排序一致性。

与 v1 / v2 的核心区别 —— 信号来源：

v1 (State-based):
    在 State（softmax 正样本概率的衍生物）上约束绝对值差异
    梯度路径: loss → State → p(u,v) → softmax → sim → z → encoder (8 步)

v2 (z-level):
    在 projector 输出 z 上约束正样本对余弦相似度的 Z-score MSE
    梯度路径: loss → z → projector → encoder (3 步)
    信号: 只用正样本对余弦相似度（对角线元素），信息量有限

v3 (difficulty-ranking):
    用每个节点的完整 NT-Xent 损失作为"对比难度"，约束相邻维度
    对节点难度的排序一致（Z-score MSE）
    梯度路径: loss → difficulty → cross_entropy → sim → z → encoder (5 步)
    信号: 完整对比难度（正样本 vs 所有负样本），信息量最大
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_per_sample_difficulty(sim: Tensor) -> Tensor:
    """从相似度矩阵提取每个节点的 NT-Xent 对比难度。

    节点 i 的难度 = CE(sim[i], label=i)
    - 低损失 → 正样本对 sim[i,i] 在所有候选 sim[i,:] 中突出 → 简单
    - 高损失 → 正样本对被负样本淹没 → 困难

    参数:
        sim: 相似度矩阵 (N, N)，已做 L2 归一化和温度缩放

    返回:
        difficulty: (N,) 每个节点的对比难度
    """
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels, reduction="none")  # (N,)


def compute_ranking_ml_loss(
    z1_full: Tensor,
    z2_full: Tensor,
    mrl_dims: Sequence[int],
    tau: float,
    ml_weight: float = 1.0,
) -> Tensor:
    """基于节点难度排序一致性的互学习损失。

    梯度路径: loss → difficulty → cross_entropy → sim → z → encoder (5 步)

    对每对相邻维度:
      1. L2 归一化 z1, z2 → 计算相似度矩阵 sim
      2. 从 sim 提取每节点的 NT-Xent 对比难度 difficulty
      3. Z-score 标准化（消除维度间的系统性偏移）
      4. MSE 约束两个维度的标准化难度分布一致

    参数:
        z1_full: 视图1 的完整投影嵌入 (N, max_dim)
        z2_full: 视图2 的完整投影嵌入 (N, max_dim)
        mrl_dims: MRL 维度列表
        tau: 温度系数（与 nt_xent 一致）
        ml_weight: 互学习损失权重

    返回:
        ml_loss: 标量。ml_weight=1 时 ML/GRACE 占比约 5-15%
    """
    sorted_dims = sorted(mrl_dims)
    if len(sorted_dims) < 2:
        return torch.tensor(0.0, device=z1_full.device)

    device = z1_full.device
    dim_difficulties = {}

    # 一次遍历: 对每个维度计算对比难度
    for dim in sorted_dims:
        z1 = F.normalize(z1_full[:, :dim], dim=-1)
        z2 = F.normalize(z2_full[:, :dim], dim=-1)
        sim = torch.mm(z1, z2.t()) / tau
        dim_difficulties[f"dim_{dim}"] = compute_per_sample_difficulty(sim)

    # 相邻维度的排序一致性约束
    ml_loss = torch.zeros((), device=device)
    count = 0

    for i in range(1, len(sorted_dims)):
        d_prev = sorted_dims[i - 1]
        d_curr = sorted_dims[i]

        diff_p = dim_difficulties[f"dim_{d_prev}"]
        diff_c = dim_difficulties[f"dim_{d_curr}"]

        # Z-score 标准化: 消除系统性偏移，只比较相对难度分布
        dp_z = (diff_p - diff_p.mean()) / (diff_p.std() + 1e-8)
        dc_z = (diff_c - diff_c.mean()) / (diff_c.std() + 1e-8)

        ml_loss = ml_loss + ((dp_z - dc_z) ** 2).mean()
        count += 1

    return ml_weight * ml_loss / max(count, 1)


class RankingMutualLearningLoss:
    """节点难度排序一致性损失计算器。

    梯度路径: loss → difficulty → cross_entropy → sim → z → encoder (5 步)

    使用示例:
        ml_calc = RankingMutualLearningLoss(
            mrl_dims=[32, 64, 128, 256, 384, 512, 768],
            tau=0.5,
            ml_weight=1.0,
        )

        ml_loss = ml_calc.compute(z1_full, z2_full)
        total = mrl_loss + ml_loss
    """

    def __init__(self, mrl_dims: Sequence[int], tau: float, ml_weight: float = 1.0):
        self.mrl_dims = sorted({int(d) for d in mrl_dims})
        self.tau = float(tau)
        self.ml_weight = float(ml_weight)

        if len(self.mrl_dims) < 2:
            import warnings
            warnings.warn("mrl_dims 维度少于2个，互学习损失将恒为0")

    def compute(self, z1_full: Tensor, z2_full: Tensor) -> Tensor:
        """从 projector 输出计算节点难度排序一致性损失。

        参数:
            z1_full: 视图1 的完整投影嵌入 (N, max_dim)
            z2_full: 视图2 的完整投影嵌入 (N, max_dim)

        返回:
            ml_loss: 标量（已包含 ml_weight）
        """
        return compute_ranking_ml_loss(
            z1_full, z2_full, self.mrl_dims, self.tau, self.ml_weight
        )
