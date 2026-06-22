"""CSNE (Croppable Self-Supervised Node Embedding) 方法类。

整合论文 CSNE 框架的全部三个机制：
    - CDMD (Cross-Dimensional Mutual Distillation)：相邻维度间的相似度结构互蒸馏
    - HPEM (Hard-Pair Evolutionary Mining)：困难节点对的进化挖掘 + 维度自适应温度
    - DAS (Dimension-Adaptive Loss Scheduling)：维度相关的 HPEM 损失权重调度

继承自 GRACEWithMRLMethod，复用 MRL 训练流程（数据增强、嵌入提取、多维度评估），
在此基础上叠加三个 CSNE 机制。

训练策略（两阶段）：
    第一阶段（1 ~ warmup_epochs）：仅训练 GRACE+MRL 损失（标准 InfoNCE per prefix）
    第二阶段（warmup_epochs+1 ~ total_epochs）：完整的 CSNE 损失
        L_CSNE = Σ_{i>1} L_CDMD^{i-1,i} + Σ_i exp(λ·d_i/d_n) · L_HPEM^i
"""

import itertools
from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor

from .grace_mrl_method import GRACEWithMRLMethod
from .ML_module import MutualLearningLoss
from .HPEM_module import HPEMLoss
from .DAS_module import DASScheduler


class CSNEMethod(GRACEWithMRLMethod):
    """CSNE 方法：整合 CDMD + HPEM + DAS。

    参数:
        mrl_dims: MRL 维度列表，如 [64, 128, 256, 512, 768]
        mrl_weight: MRL 损失权重（warmup 阶段使用）
        ml_weight: CDMD 互学习损失权重
        ml_module: CDMD 模块选择 ("ml" = 相邻互学习, "ml2" = 向最高维学习)
        hpem_beta_init: HPEM 的 β 参数初始值
        hpem_tau_0: HPEM 的基础温度 τ_0
        warmup_epochs: warmup 阶段 epoch 数
        full_epochs: 完整 CSNE 阶段 epoch 数
    """

    def __init__(
        self,
        *args,
        mrl_dims: Sequence[int],
        mrl_weight: float = 1.0,
        # CDMD
        ml_weight: float = 1.0,
        ml_module: str = "ml",
        # HPEM
        hpem_beta_init: float = 0.1,
        hpem_tau_0: float = 0.5,
        # DAS
        # （无额外参数，λ 自动初始化为 0）
        # 训练阶段
        warmup_epochs: int = 100,
        full_epochs: int = 100,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, mrl_dims=mrl_dims, mrl_weight=mrl_weight, **kwargs)
        self.method_name = "csne"

        # ---- 训练阶段配置 ----
        self.warmup_epochs = int(warmup_epochs)
        self.full_epochs = int(full_epochs)
        self.total_epochs = self.warmup_epochs + self.full_epochs
        self.verbose = verbose

        max_dim = max(self.mrl_dims)

        # ---- CDMD 模块（复用 ML_module）----
        self.ml_weight = float(ml_weight)
        if ml_module == "ml2":
            from .ML_module2 import MutualLearningLoss2
            self.cdmd = MutualLearningLoss2(
                mrl_dims=self.mrl_dims,
                ml_weight=ml_weight,
            )
        else:
            self.cdmd = MutualLearningLoss(
                mrl_dims=self.mrl_dims,
                ml_weight=ml_weight,
            )

        # ---- HPEM 模块（内置维度自适应 temperature τ_i）----
        self.hpem = HPEMLoss(
            tau_0=hpem_tau_0,
            max_dim=max_dim,
            beta_init=hpem_beta_init,
        )

        # ---- DAS 模块（仅 HPEM 损失权重 w_i = exp(λ·d_i/d_n)）----
        self.das = DASScheduler(max_dim=max_dim)

        # ---- epoch 计数器 ----
        self.epoch = 0

    # ------------------------------------------------------------------
    # CSNE 完整损失计算
    # ------------------------------------------------------------------

    def _grace_prefix_loss_with_details(
        self,
        z1_full: Tensor,
        z2_full: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        r"""计算 CSNE 总损失。

        warmup 阶段：标准 MRL（mean of InfoNCE per prefix）
        完整阶段：
            L_CSNE = Σ_{i>1} L_CDMD^{i-1,i} + Σ_i w_i^{DAS} · L_HPEM^i

        其中 w_i^{DAS} = exp(λ · d_i / d_n)，
        τ_i = τ_0 · exp(φ_1 · d_i/d_n + φ_2)（HPEM 内部计算），
        L_HPEM^1 使用均匀权重（等价于标准 InfoNCE），
        L_HPEM^{i>1} 使用 confusion-based 权重。
        """
        sorted_dims = self.mrl_dims
        dim_losses: Dict[str, float] = {}
        device = z1_full.device

        # ── warmup 阶段：仅标准 MRL ──
        if self.epoch <= self.warmup_epochs:
            loss_sum = torch.zeros((), device=device)
            for dim in sorted_dims:
                z1 = z1_full[:, :dim]
                z2 = z2_full[:, :dim]
                dim_loss = self.model.nt_xent(z1, z2, self.model.tau)
                loss_sum = loss_sum + dim_loss
                dim_losses[f"dim_{dim}"] = float(dim_loss.detach().item())

            grace_loss_avg = loss_sum / len(sorted_dims)
            total = self.mrl_weight * grace_loss_avg
            dim_losses["grace_loss"] = float(grace_loss_avg.detach().item())
            dim_losses["mrl_loss"] = float(total.detach().item())
            dim_losses["cdmd_loss"] = 0.0
            dim_losses["hpem_loss"] = 0.0
            return total, dim_losses

        # ── 完整 CSNE 阶段 ──
        # ---- 1. CDMD 损失：相邻维度间的互蒸馏 ----
        cdmd_loss = self.cdmd.compute(z1_full, z2_full)

        # ---- 2. HPEM 损失 + DAS 维度权重 ----
        hpem_total = torch.zeros((), device=device)
        hpem_detail: Dict[str, float] = {}

        # i = 0（最小维度）：均匀权重 InfoNCE
        dim_0 = sorted_dims[0]
        tau_0 = self.hpem.get_tau(dim_0)
        w_0 = self.das.get_hpem_weight(dim_0)
        z1_0 = z1_full[:, :dim_0]
        z2_0 = z2_full[:, :dim_0]
        base_loss_0 = self.model.nt_xent(z1_0, z2_0, float(tau_0.item()))
        hpem_total = hpem_total + w_0 * base_loss_0
        hpem_detail[f"dim_{dim_0}"] = float(base_loss_0.detach().item())

        # i > 0：confusion-based 加权的 HPEM（τ_i 由 HPEM 内部计算）
        for i in range(1, len(sorted_dims)):
            dim_prev = sorted_dims[i - 1]
            dim_curr = sorted_dims[i]
            w_i = self.das.get_hpem_weight(dim_curr)

            hpem_i = self.hpem.compute_single(
                z1_curr=z1_full[:, :dim_curr],
                z2_curr=z2_full[:, :dim_curr],
                z1_prev=z1_full[:, :dim_prev],
                z2_prev=z2_full[:, :dim_prev],
                dim_curr=dim_curr,  # HPEM 内部用此计算 τ_i
            )
            hpem_total = hpem_total + w_i * hpem_i
            hpem_detail[f"dim_{dim_curr}"] = float(hpem_i.detach().item())

        # ---- 3. 总损失 ----
        total = cdmd_loss + hpem_total

        # ---- 记录日志 ----
        dim_losses["cdmd_loss"] = float(cdmd_loss.detach().item())
        dim_losses["hpem_loss"] = float(hpem_total.detach().item())
        dim_losses["total_loss"] = float(total.detach().item())
        dim_losses.update(hpem_detail)
        dim_losses.update(self.hpem.log_info())
        dim_losses.update(self.das.log_info())

        return total, dim_losses

    # ------------------------------------------------------------------
    # 参数管理
    # ------------------------------------------------------------------

    def parameters(self, recurse: bool = True):
        """返回所有可训练参数的迭代器，包括 HPEM 和 DAS 的参数。"""
        return itertools.chain(
            super().parameters(recurse=recurse),
            self.hpem.parameters(recurse=recurse),
            self.das.parameters(recurse=recurse),
        )

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """返回所有可训练参数的名称，包括 HPEM 和 DAS 的参数。"""
        yield from super().named_parameters(prefix=prefix, recurse=recurse)
        yield from self.hpem.named_parameters(prefix=prefix + 'hpem.', recurse=recurse)
        yield from self.das.named_parameters(prefix=prefix + 'das.', recurse=recurse)
