"""DAS (Dimension-Adaptive Loss Scheduling) 模块。

实现论文 CSNE 框架 Section 3.4 的 DAS 组件 B——维度相关的 HPEM 损失权重。

组件 A（维度自适应 temperature τ_i）已迁移至 HPEM_module.py，
因为 τ_i 的唯一消费者就是 HPEM。

总损失公式：
    L_CSNE = Σ_{i>1} L_CDMD^{i-1,i} + Σ_i exp(λ · d_i/d_n) · L_HPEM^i

可学习参数：
    λ (lam): 控制 HPEM 损失权重随维度指数增长的系数

模块职责：
    - 输入维度 d_i 和最大维度 d_n
    - 返回该维度的 HPEM 损失权重 w_i = exp(λ · d_i/d_n)
    - 完全独立、可复用
"""

import torch
import torch.nn as nn
from torch import Tensor


class DASScheduler(nn.Module):
    r"""DAS 调度器：维度相关的 HPEM 损失权重。

    可学习参数：
        lam (λ): HPEM 损失权重的指数系数
                 初始化为 0 → 所有维度等权
                 λ > 0 → 高维获得更大 HPEM 权重

    使用示例:
        das = DASScheduler(max_dim=768)
        w_i = das.get_hpem_weight(256)  # exp(λ · 256/768)
    """

    def __init__(self, max_dim: int = 768):
        """
        参数:
            max_dim: 最大维度 d_n，用于归一化 d_i / d_n
        """
        super().__init__()
        self.max_dim = int(max_dim)

        # λ：初始化为 0 → 所有维度等权
        self.lam = nn.Parameter(torch.tensor(0.0))

    def _normalized_dim(self, dim: int) -> float:
        return dim / self.max_dim

    def get_hpem_weight(self, dim: int) -> Tensor:
        r"""返回维度 dim 的 HPEM 损失权重。

        w_i = exp(λ · d_i / d_n)

        参数:
            dim: 当前维度 d_i

        返回:
            weight: 标量 tensor
        """
        x = self._normalized_dim(dim)
        return torch.exp(self.lam * x)

    def get_all_hpem_weights(self, mrl_dims: list) -> Tensor:
        r"""批量获取所有维度的 HPEM 损失权重。"""
        return torch.stack([self.get_hpem_weight(d) for d in mrl_dims])

    def log_info(self) -> dict:
        r"""返回当前可学习参数的快照，用于日志记录。"""
        with torch.no_grad():
            return {"das_lam": float(self.lam.item())}
