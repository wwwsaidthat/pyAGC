"""互学习损失计算模块。

负责计算 Mutual Learning 相关量：
- 正样本对概率 p(u,v)、p(v,u)
- 基于概率构造的 State
- 各维度之间的互学习损失 L_ML

注意：GRACE (NT-Xent) 损失由 GRACECore.nt_xent 统一计算，
本模块不再重复实现，保证与 grace_mrl 严格等价。
"""

from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor


def compute_positive_pair_probs(z1: Tensor, z2: Tensor, tau: float) -> Tuple[Tensor, Tensor]:
    """计算双向正样本对概率 p(u,v) 与 p(v,u)。

    不负责 GRACE/NT-Xent 主损失，只从相似度矩阵中提取
    互学习所需的正样本概率。
    """
    import torch.nn.functional as F

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    batch_size = z1.size(0)
    sim = torch.matmul(z1, z2.T) / tau

    prob_u_v = F.softmax(sim, dim=1)
    prob_v_u = F.softmax(sim.t(), dim=1)
    idx = torch.arange(batch_size, device=z1.device)
    p_u_v = prob_u_v[idx, idx]
    p_v_u = prob_v_u[idx, idx]

    return p_u_v, p_v_u


def compute_state(p_u_v: Tensor, p_v_u: Tensor) -> Tensor:
    """计算 State 值。

    State 表示样本对的对比学习置信度，范围 [0, 1]。
    State 越接近 1 表示模型对该样本对的判别越自信。

    参数:
        p_u_v: p(u, v) 概率向量 (batch_size,)
        p_v_u: p(v, u) 概率向量 (batch_size,)

    返回:
        state: State 值向量 (batch_size,)
    """
    return 0.5 * (2 - p_u_v - p_v_u)


def compute_mutual_learning_loss(
    dim_states: Dict[str, Tensor],
    mrl_dims: Sequence[int],
    ml_weight: float = 1.0,
) -> Tensor:
    """计算维度间的互学习损失。

    鼓励高维模型学习低维模型的表示能力，通过约束 State 值实现。

    参数:
        dim_states: 各维度的 State 字典 {dim_str: state_tensor}
        mrl_dims: MRL 维度列表
        ml_weight: 互学习损失权重

    返回:
        ml_loss: 互学习损失标量
    """
    if not mrl_dims or len(mrl_dims) < 2:
        if dim_states:
            device = next(iter(dim_states.values())).device
            return torch.zeros((), device=device)
        return torch.tensor(0.0)

    sorted_dims = sorted(mrl_dims)
    first_dim = sorted_dims[0]
    device = dim_states[f"dim_{first_dim}"].device
    ml_loss = torch.zeros((), device=device)

    for i in range(1, len(sorted_dims)):
        dim_prev = sorted_dims[i - 1]
        dim_curr = sorted_dims[i]

        state_prev = dim_states[f"dim_{dim_prev}"]
        state_curr = dim_states[f"dim_{dim_curr}"]

        ml_loss = ml_loss + torch.abs(state_prev - state_curr).mean()

    return ml_weight * ml_loss / (len(sorted_dims) - 1)


class MutualLearningLoss:
    """互学习损失计算器类。

    提供更高级的接口，便于在不同模型中复用。

    使用示例:
        ml_calculator = MutualLearningLoss(mrl_dims=[128, 256, 512], ml_weight=5.0)

        # 已有各维度 State 时直接计算 ML 损失
        ml_loss = ml_calculator.compute(dim_states)
    """

    def __init__(self, mrl_dims: Sequence[int], ml_weight: float = 1.0):
        """
        参数:
            mrl_dims: MRL 维度列表
            ml_weight: 互学习损失权重
        """
        self.mrl_dims = sorted({int(d) for d in mrl_dims})
        self.ml_weight = ml_weight

        if len(self.mrl_dims) < 2:
            import warnings

            warnings.warn("mrl_dims 维度少于2个，互学习损失将恒为0")

    def compute(self, dim_states: Dict[str, Tensor]) -> Tensor:
        """从已有的各维度 State 计算互学习损失。

        参数:
            dim_states: 各维度的 State 字典，key 为 "dim_{dim}" 格式

        返回:
            ml_loss: 互学习损失标量（已包含 ml_weight）
        """
        return compute_mutual_learning_loss(dim_states, self.mrl_dims, self.ml_weight)
