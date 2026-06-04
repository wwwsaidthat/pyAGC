"""GRACE + MRL + Mutual Learning v2 — z-level 短梯度路径。

与 grace_ML.py (v1) 的区别 —— ML 损失的计算层面：

v1 (grace_ML):
    在 State（softmax 正样本概率的衍生物）上做互学习
    梯度路径: loss → State → p(u,v) → softmax → sim → z → encoder (8 步)

v2 (grace_ML2):
    在 projector 输出 z（L2 归一化后）上做互学习
    梯度路径: loss → z → projector → encoder (3 步, 和 GRACE 同级)

GRACE 损失部分与父类 GRACEWithMRLMethod 完全一致。
当 ml_weight=0 时与 grace_mrl 严格等价。
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data

from .grace_mrl_method import GRACEWithMRLMethod
from .ML_module2 import ZLevelMutualLearningLoss


class GRACEWithMRLMutualLearningMethodV2(GRACEWithMRLMethod):
    """GRACE + MRL + 互学习融合方法 v2（z-level, 短梯度路径）。

    继承自 GRACEWithMRLMethod，复用完整的 MRL 训练流程。
    GRACE 损失与父类调用同一个 self.model.nt_xent。
    ML 损失通过 ML_module2 在 projector 输出 z 上直接约束，
    梯度路径和 GRACE 同级，信号传导不受 softmax/State 衰减。
    """

    def __init__(
        self,
        *args,
        mrl_dims: Sequence[int],
        mrl_weight: float = 1.0,
        ml_weight: float = 1.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, mrl_dims=mrl_dims, mrl_weight=mrl_weight, **kwargs)
        self.method_name = "grace_mrl_ml_v2"
        self.ml_weight = float(ml_weight)
        self.verbose = verbose
        self.ml_calculator = ZLevelMutualLearningLoss(
            mrl_dims=self.mrl_dims, ml_weight=ml_weight
        )
        self.epoch = 0

    # ------------------------------------------------------------------
    # 覆盖父类的损失计算，叠加 z-level 互学习损失
    # ------------------------------------------------------------------

    def _grace_prefix_loss_with_details(
        self, z1_full: Tensor, z2_full: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """GRACE 损失（与父类一致）+ z-level ML 损失（短梯度路径）。

        GRACE: 通过 self.model.nt_xent 计算，与父类严格一致。
        ML:    在 projector 输出 z 上通过 ZLevelMutualLearningLoss 直接约束，
               梯度路径 loss → z → projector → encoder（3 步）。
        """
        dim_losses: Dict[str, float] = {}
        loss_sum = torch.zeros((), device=z1_full.device)

        for dim in self.mrl_dims:
            z1 = z1_full[:, :dim]
            z2 = z2_full[:, :dim]

            # GRACE 损失：与父类完全一致的调用方式
            dim_loss = self.model.nt_xent(z1, z2, self.model.tau)
            loss_sum = loss_sum + dim_loss
            dim_losses[f"dim_{dim}"] = float(dim_loss.detach().item())

        # MRL 损失：与父类公式一致
        grace_loss_avg = loss_sum / len(self.mrl_dims)
        mrl_loss = self.mrl_weight * grace_loss_avg

        # z-level 互学习损失（短梯度路径）
        # ml_weight=0 时跳过，与 grace_mrl 严格等价
        if self.ml_weight > 0.0:
            ml_loss = self.ml_calculator.compute(z1_full, z2_full)
        else:
            ml_loss = torch.zeros((), device=z1_full.device)

        # 总损失
        total = mrl_loss + ml_loss

        # 记录各分量损失
        dim_losses["ml_loss"] = float(ml_loss.detach().item())
        dim_losses["grace_loss"] = float(grace_loss_avg.detach().item())
        dim_losses["mrl_loss"] = float(mrl_loss.detach().item())

        return total, dim_losses

    # ------------------------------------------------------------------
    # 训练步骤：复用父类逻辑，仅附加 verbose 打印
    # ------------------------------------------------------------------

    def ssl_train_step_full(
        self, data: Data, device, optimizer
    ) -> float:
        """全量训练步骤。调用父类实现（自动使用覆盖后的损失计算）。"""
        loss = super().ssl_train_step_full(data, device, optimizer)
        if self.verbose:
            d = self.last_mrl_dim_losses
            print(
                f"[Epoch {self.epoch}] "
                f"GRACE Loss: {d.get('grace_loss', 0):.4f} | "
                f"ML Loss (z): {d.get('ml_loss', 0):.4f} | "
                f"Total Loss: {loss:.4f}"
            )
        return loss

    def ssl_train_step_neighbor(
        self,
        data: Data,
        input_nodes: Optional[Tensor],
        num_neighbors: Sequence[int],
        batch_size: int,
        device,
        optimizer,
    ) -> float:
        """邻居采样训练步骤。调用父类实现（自动使用覆盖后的损失计算）。"""
        loss = super().ssl_train_step_neighbor(
            data, input_nodes, num_neighbors, batch_size, device, optimizer
        )
        if self.verbose:
            d = self.last_mrl_dim_losses
            print(
                f"[Epoch {self.epoch}] "
                f"GRACE Loss: {d.get('grace_loss', 0):.4f} | "
                f"ML Loss (z): {d.get('ml_loss', 0):.4f} | "
                f"Total Loss: {loss:.4f}"
            )
        return loss
