"""GRACE + MRL + Mutual Learning v3 — 节点难度排序一致性。

与 grace_ML.py (v1) / grace_ML2.py (v2) 的区别：

v1 (grace_ML):
    在 State 上约束绝对值差异
    梯度路径: loss → State → p(u,v) → softmax → sim → z → encoder (8 步)

v2 (grace_ML2):
    在 projector 输出 z 上约束正样本对余弦相似度
    梯度路径: loss → z → projector → encoder (3 步)
    信号: 仅正样本对余弦相似度

v3 (grace_ML3):
    在每节点的 NT-Xent 对比难度上约束排序一致性
    梯度路径: loss → difficulty → cross_entropy → sim → z → encoder (5 步)
    信号: 完整对比难度（正样本 vs 所有负样本），信息量最大

GRACE 损失部分与父类 GRACEWithMRLMethod 完全一致。
当 ml_weight=0 时与 grace_mrl 严格等价。
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data

from .grace_mrl_method import GRACEWithMRLMethod
from .ML_module3 import RankingMutualLearningLoss


class GRACEWithMRLMutualLearningMethodV3(GRACEWithMRLMethod):
    """GRACE + MRL + 互学习融合方法 v3（节点难度排序一致性）。

    继承自 GRACEWithMRLMethod，复用完整的 MRL 训练流程。
    GRACE 损失与父类调用同一个 self.model.nt_xent。
    ML 损失通过 ML_module3 用每节点的完整对比难度做排序一致性约束，
    梯度路径 5 步，信号比 v2 更丰富。
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
        self.method_name = "grace_mrl_ml_v3"
        self.ml_weight = float(ml_weight)
        self.verbose = verbose
        self.ml_calculator = RankingMutualLearningLoss(
            mrl_dims=self.mrl_dims,
            tau=self.model.tau,
            ml_weight=ml_weight,
        )
        self.epoch = 0

    # ------------------------------------------------------------------
    # 覆盖父类的损失计算，叠加节点难度排序一致性损失
    # ------------------------------------------------------------------

    def _grace_prefix_loss_with_details(
        self, z1_full: Tensor, z2_full: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """GRACE 损失（与父类一致）+ 节点难度排序一致性 ML 损失。

        GRACE: 通过 self.model.nt_xent 计算，与父类严格一致。
        ML:    在每节点的 NT-Xent 对比难度上做排序一致性约束，
               梯度路径 loss → difficulty → cross_entropy → sim → z (5 步)。
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

        # 节点难度排序一致性损失（5 步梯度路径）
        if self.ml_weight > 0.0:
            ml_loss = self.ml_calculator.compute(z1_full, z2_full)
        else:
            ml_loss = torch.zeros((), device=z1_full.device)

        total = mrl_loss + ml_loss

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
        loss = super().ssl_train_step_full(data, device, optimizer)
        if self.verbose:
            d = self.last_mrl_dim_losses
            print(
                f"[Epoch {self.epoch}] "
                f"GRACE Loss: {d.get('grace_loss', 0):.4f} | "
                f"ML Loss (rank): {d.get('ml_loss', 0):.4f} | "
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
        loss = super().ssl_train_step_neighbor(
            data, input_nodes, num_neighbors, batch_size, device, optimizer
        )
        if self.verbose:
            d = self.last_mrl_dim_losses
            print(
                f"[Epoch {self.epoch}] "
                f"GRACE Loss: {d.get('grace_loss', 0):.4f} | "
                f"ML Loss (rank): {d.get('ml_loss', 0):.4f} | "
                f"Total Loss: {loss:.4f}"
            )
        return loss
