"""DAS (Dimension-Adaptive Loss Scheduling) 模块。

实现论文 CSNE 框架 Section 3.4 的 DAS 机制，包含两个子组件：

组件 A — 每维度可学习 temperature：
    τ_i = τ_0 · g_φ(d_i / d_n)
    g_φ(x) = exp(φ_1 · x + φ_2)
    φ_1, φ_2 是可学习标量，初始化保证 g_φ(1) ≈ 1

组件 B — 维度相关的 HPEM 损失权重：
    L_CSNE = Σ_{i>1} L_CDMD^{i-1,i} + Σ_i exp(λ · d_i/d_n) · L_HPEM^i
    λ 是可学习标量，高维 prefix 获得指数级更大的 HPEM 权重

模块职责：
    - 输入维度 d_i 和最大维度 d_n
    - 返回该维度的 temperature τ_i 和 HPEM 损失权重
    - 完全独立、可复用，不依赖任何特定方法类
"""

import torch
import torch.nn as nn
from torch import Tensor


class DASScheduler(nn.Module):
    r"""DAS 调度器：维度自适应的 temperature 和 HPEM 损失权重。

    可学习参数：
        phi_1, phi_2: 控制 temperature 随维度的变化
        lam (λ):     控制 HPEM 损失权重随维度的变化

    使用示例:
        das = DASScheduler(tau_0=0.5, max_dim=768)

        # 对每个维度获取 temperature 和权重
        for dim in mrl_dims:
            tau_i = das.get_tau(dim)
            w_i = das.get_hpem_weight(dim)
            # w_i 用于加权 L_HPEM^i
            # tau_i 用于该维度的 InfoNCE 和 confusion → weights 计算
    """

    def __init__(self, tau_0: float = 0.5, max_dim: int = 768):
        """
        参数:
            tau_0: 基础 temperature
            max_dim: 最大维度 d_n，用于归一化 d_i / d_n
        """
        super().__init__()
        self.tau_0 = float(tau_0)
        self.max_dim = int(max_dim)

        # 可学习参数：初始化使得 g_φ(1) ≈ 1，即 φ_1 + φ_2 ≈ 0
        # φ_1 = 0, φ_2 = 0 → exp(0) = 1 → g_φ(1) = 1
        self.phi_1 = nn.Parameter(torch.tensor(0.0))
        self.phi_2 = nn.Parameter(torch.tensor(0.0))

        # λ：HPEM 损失权重的指数系数，初始化为 0 → 所有维度等权
        self.lam = nn.Parameter(torch.tensor(0.0))

    def _normalized_dim(self, dim: int) -> float:
        """将维度归一化到 [0, 1]。"""
        return dim / self.max_dim

    # ------------------------------------------------------------------
    # 组件 A：每维度可学习 temperature
    # ------------------------------------------------------------------

    def get_tau(self, dim: int) -> Tensor:
        r"""返回维度 dim 的 temperature。

        τ_i = τ_0 · exp(φ_1 · d_i/d_n + φ_2)

        参数:
            dim: 当前维度 d_i

        返回:
            tau_i: 标量 tensor
        """
        x = self._normalized_dim(dim)
        g = torch.exp(self.phi_1 * x + self.phi_2)
        return torch.tensor(self.tau_0, device=g.device, dtype=g.dtype) * g

    # ------------------------------------------------------------------
    # 组件 B：维度相关的 HPEM 损失权重
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 批量接口
    # ------------------------------------------------------------------

    def get_all_taus(self, mrl_dims: list) -> Tensor:
        r"""批量获取所有维度的 temperature。

        参数:
            mrl_dims: 维度列表

        返回:
            taus: (len(mrl_dims),) temperature 列表
        """
        return torch.stack([self.get_tau(d) for d in mrl_dims])

    def get_all_hpem_weights(self, mrl_dims: list) -> Tensor:
        r"""批量获取所有维度的 HPEM 损失权重。

        参数:
            mrl_dims: 维度列表

        返回:
            weights: (len(mrl_dims),) 权重列表
        """
        return torch.stack([self.get_hpem_weight(d) for d in mrl_dims])

    # ------------------------------------------------------------------
    # 参数日志（便于监控训练过程中参数的变化）
    # ------------------------------------------------------------------

    def log_info(self) -> dict:
        r"""返回当前可学习参数的快照，用于日志记录。"""
        with torch.no_grad():
            return {
                "das_phi_1": float(self.phi_1.item()),
                "das_phi_2": float(self.phi_2.item()),
                "das_lam": float(self.lam.item()),
            }
