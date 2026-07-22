"""MCNE 方法类。

整合论文 MCNE 框架的全部三个机制：
    - CDMD (Cross-Dimensional Mutual Distillation)：所有低维前缀与最高维前缀进行关系蒸馏
    - HPEM (Hard-Pair Evolutionary Mining)：困难节点对的进化挖掘 + 维度自适应温度
    - DALS (Dimension-Adaptive Loss Scheduling)：维度相关的 HPEM 损失权重调度

继承自 GRACEWithMRLMethod，复用 MRL 训练流程（数据增强、嵌入提取、多维度评估），
在此基础上叠加三个 MCNE 机制。

训练策略（两阶段）：
    第一阶段（1 ~ warmup_epochs）：仅训练 GRACE+MRL 损失（标准 InfoNCE per prefix）
    第二阶段（warmup_epochs+1 ~ total_epochs）：完整的 MCNE 损失
        L_MCNE = L_CDMD + Σ_i exp(λ·d_i/d_n) · L_HPEM^i
"""

from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor

from .grace_mrl_method import GRACEWithMRLMethod
from .ML_module import MutualLearningLoss
from .HPEM_module import HPEMLoss
from .DAS_module import DASScheduler


class MCNEMethod(GRACEWithMRLMethod):
    """MCNE 方法：整合 CDMD + HPEM + DALS。

    参数:
        mrl_dims: MRL 维度列表，如 [64, 128, 256, 512, 768]
        mrl_weight: MRL 损失权重（warmup 阶段使用）
        ml_weight: CDMD 互学习损失权重
        ml_module: CDMD 模块选择 ("ml" = 相邻互学习, "ml2" = 向最高维学习)
        hpem_beta_init: HPEM 的 β 参数初始值
        hpem_tau_0: HPEM 的基础温度 τ_0
        warmup_epochs: warmup 阶段 epoch 数
        full_epochs: 完整 MCNE 阶段 epoch 数
    """

    def __init__(
        self,
        *args,
        mrl_dims: Sequence[int],
        mrl_weight: float = 1.0,
        # CDMD
        ml_weight: float = 1.0,
        ml_module: str = "ml2",
        cdmd_tau: float = 0.5,
        # HPEM
        hpem_beta_init: float = 0.1,
        hpem_tau_0: float = 0.5,
        # 消融开关
        use_cdmd: bool = True,
        use_hpem: bool = True,
        use_das: bool = True,
        ablation_name: str = "mcne",
        # DALS
        # （无额外参数，λ 自动初始化为 0）
        # 训练阶段
        warmup_epochs: int = 100,
        full_epochs: int = 100,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, mrl_dims=mrl_dims, mrl_weight=mrl_weight, **kwargs)
        self.method_name = str(ablation_name)
        self.use_cdmd = bool(use_cdmd)
        self.use_hpem = bool(use_hpem)
        # DALS 只调度 HPEM；关闭 HPEM 时 DALS 在数学上没有消费者。
        self.use_das = bool(use_das) and self.use_hpem

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
                tau=cdmd_tau,
            )
        else:
            self.cdmd = MutualLearningLoss(
                mrl_dims=self.mrl_dims,
                ml_weight=ml_weight,
                tau=cdmd_tau,
            )

        # ---- HPEM 模块（内置维度自适应 temperature τ_i）----
        self.hpem = HPEMLoss(
            tau_0=hpem_tau_0,
            max_dim=max_dim,
            beta_init=hpem_beta_init,
        )

        # ---- DALS 模块（仅 HPEM 损失权重 w_i = exp(λ·d_i/d_n)）----
        self.das = DASScheduler(max_dim=max_dim)

        # ---- epoch 计数器 ----
        self.epoch = 0

    # ------------------------------------------------------------------
    # MCNE 完整损失计算
    # ------------------------------------------------------------------

    def _grace_prefix_loss_with_details(
        self,
        h1_full: Tensor,
        h2_full: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        r"""计算 MCNE 总损失。

        warmup 阶段：标准 MRL（mean of InfoNCE per prefix）
        完整阶段：
            L_MCNE = L_CDMD + Σ_i w_i^{DALS} · L_HPEM^i

        其中 w_i^{DAS} = exp(λ · d_i / d_n)，
        τ_i = τ_0 · exp(φ_1 · d_i/d_n + φ_2)（HPEM 内部计算），
        L_HPEM^1 使用均匀权重（等价于标准 InfoNCE），
        L_HPEM^{i>1} 使用 confusion-based 权重。
        """
        sorted_dims = self.mrl_dims
        dim_losses: Dict[str, float] = {}
        device = h1_full.device

        # ── warmup 阶段：仅标准 MRL ──
        if self.epoch <= self.warmup_epochs:
            loss_sum = torch.zeros((), device=device)
            for dim in sorted_dims:
                h1 = h1_full[:, :dim]
                h2 = h2_full[:, :dim]
                dim_loss = self.model.nt_xent(h1, h2, self.model.tau)
                loss_sum = loss_sum + dim_loss
                dim_losses[f"dim_{dim}"] = float(dim_loss.detach().item())

            grace_loss_avg = loss_sum / len(sorted_dims)
            total = self.mrl_weight * grace_loss_avg
            dim_losses["grace_loss"] = float(grace_loss_avg.detach().item())
            dim_losses["mrl_loss"] = float(total.detach().item())
            dim_losses["cdmd_loss"] = 0.0
            dim_losses["hpem_loss"] = 0.0
            dim_losses["grace_mrl_loss"] = float(total.detach().item())
            dim_losses["cdmd_enabled"] = float(self.use_cdmd)
            dim_losses["hpem_enabled"] = float(self.use_hpem)
            dim_losses["das_enabled"] = float(self.use_das)
            return total, dim_losses

        # ── 完整 MCNE 阶段 ──
        # ---- 1. CDMD 损失：所有低维前缀与最高维前缀互蒸馏 ----
        if self.use_cdmd:
            cdmd_loss = self.cdmd.compute(h1_full, h2_full)
        else:
            cdmd_loss = torch.zeros((), device=device)

        # ---- 2. HPEM 损失 + DALS 维度权重 ----
        hpem_total = torch.zeros((), device=device)
        hpem_detail: Dict[str, float] = {}

        if self.use_hpem:
            # i = 0（最小维度）：均匀权重 InfoNCE
            dim_0 = sorted_dims[0]
            tau_0 = self.hpem.get_tau(dim_0)
            w_0 = self.das.get_hpem_weight(dim_0) if self.use_das else torch.ones((), device=device)
            h1_0 = h1_full[:, :dim_0]
            h2_0 = h2_full[:, :dim_0]
            base_loss_0 = self.model.nt_xent(h1_0, h2_0, tau_0)
            hpem_total = hpem_total + w_0 * base_loss_0
            hpem_detail[f"dim_{dim_0}"] = float(base_loss_0.detach().item())

            # i > 0：confusion-based 加权的 HPEM（τ_i 由 HPEM 内部计算）
            for i in range(1, len(sorted_dims)):
                dim_prev = sorted_dims[i - 1]
                dim_curr = sorted_dims[i]
                w_i = self.das.get_hpem_weight(dim_curr) if self.use_das else torch.ones((), device=device)

                hpem_i = self.hpem.compute_single(
                    h1_curr=h1_full[:, :dim_curr],
                    h2_curr=h2_full[:, :dim_curr],
                    h1_prev=h1_full[:, :dim_prev],
                    h2_prev=h2_full[:, :dim_prev],
                    dim_curr=dim_curr,
                )
                hpem_total = hpem_total + w_i * hpem_i
                hpem_detail[f"dim_{dim_curr}"] = float(hpem_i.detach().item())
        else:
            # 去掉 HPEM 时必须保留标准 GRACE+MRL InfoNCE，否则只剩 CDMD
            # 一致性项，缺少防坍塌的实例对比目标，消融将不再公平。
            base_sum = torch.zeros((), device=device)
            for dim in sorted_dims:
                base_i = self.model.nt_xent(
                    h1_full[:, :dim], h2_full[:, :dim], self.model.tau
                )
                base_sum = base_sum + base_i
                hpem_detail[f"dim_{dim}"] = float(base_i.detach().item())
            hpem_total = self.mrl_weight * base_sum / len(sorted_dims)

        # ---- 3. 总损失 ----
        total = cdmd_loss + hpem_total

        # ---- 记录日志 ----
        dim_losses["cdmd_loss"] = float(cdmd_loss.detach().item())
        dim_losses["hpem_loss"] = float(hpem_total.detach().item()) if self.use_hpem else 0.0
        dim_losses["grace_mrl_loss"] = float(hpem_total.detach().item()) if not self.use_hpem else 0.0
        dim_losses["total_loss"] = float(total.detach().item())
        dim_losses["cdmd_enabled"] = float(self.use_cdmd)
        dim_losses["hpem_enabled"] = float(self.use_hpem)
        dim_losses["das_enabled"] = float(self.use_das)
        dim_losses.update(hpem_detail)
        if self.use_hpem:
            dim_losses.update(self.hpem.log_info())
        if self.use_das:
            dim_losses.update(self.das.log_info())

        return total, dim_losses
