"""GRACE + MRL + Mutual Learning 方法类（v4 — 全 B×B 矩阵相似度 + 对称 KL）。

继承自 GRACEWithMRLMethod，复用其 MRL 训练逻辑（数据增强、嵌入提取、
多维度 GRACE 损失计算），在此基础上叠加互学习损失。

训练策略（两阶段）：
    - 第一阶段（1 ~ grace_only_epochs）：仅训练 GRACE+MRL 损失，
      让各维度子空间先学到有意义的表示。
    - 第二阶段（grace_only_epochs+1 ~ grace_only_epochs+grace_ml_epochs）：
      加入互学习损失，在已学到的表示基础上，
      鼓励相邻维度子空间产生一致的相似度结构。

损失公式（第二阶段）：
L_total = mrl_weight × (1/|M|) × Σ L_GRACE^i
        + ml_weight × (1/(|M|-1)) × Σ L_ML^{i-1,i}

互学习损失（v4 — 全 B×B 跨视图相似度矩阵）：
Step 1 — 计算两个视图归一化嵌入的 B×B 相似度矩阵：
    S_i = h1_norm[:, :i] @ h2_norm[:, :i].T       →  (B, B)

Step 2 — 拉平 + ReLU：
    r_i = ReLU(flatten(S_i))                       →  (B²,)

Step 3 — 双向相对熵按样本数归一化：
    L_ML^{i-1,i} = (1/2B) [D(r_{i-1} || r_i) + D(r_i || r_{i-1})]

与仅使用正样本对的版本不同，当前实现取完整 B×B 相似度矩阵，经
ReLU 得到非负关系证据，并对齐整个跨视图关系结构。

GRACE 损失计算与父类 GRACEWithMRLMethod 完全一致，
均通过 self.model.nt_xent 调用，保证 ml_weight=0 时严格等价。
"""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .grace_mrl_method import GRACEWithMRLMethod
from .ML_module import MutualLearningLoss


class GRACEWithMRLMutualLearningMethod(GRACEWithMRLMethod):
    """GRACE + MRL + 互学习融合方法（v4 — 全 B×B 矩阵相似度 + 对称 KL）。

    继承自 GRACEWithMRLMethod，复用完整的 MRL 训练流程。
    GRACE 损失部分与父类调用同一个 self.model.nt_xent，
    仅在此基础上额外计算跨视图相似度矩阵并叠加互学习损失。

    训练策略（两阶段）：
        - 第一阶段（1 ~ grace_only_epochs）：仅训练 GRACE+MRL 损失，
          让各维度子空间先学到有意义的表示，互学习损失置零。
        - 第二阶段（grace_only_epochs+1 ~ grace_only_epochs+grace_ml_epochs）：
          加入互学习损失，在已学到的表示基础上，
          鼓励相邻维度子空间产生一致的相似度结构。

    v4 与 v3 的唯一区别：ML 损失从仅正样本对 (B 维) 扩展为
    完整跨视图相似度矩阵 (B² 维)，包含所有正样本 + 负样本对的相似度信息。

    ml_weight=0 时与 GRACEWithMRLMethod 严格等价。
    """

    def __init__(
        self,
        *args,
        mrl_dims: Sequence[int],
        mrl_weight: float = 1.0,
        ml_weight: float = 1.0,
        grace_only_epochs: int = 100,
        grace_ml_epochs: int = 100,
        ml_module: str = "ml2",
        cdmd_tau: float = 0.5,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, mrl_dims=mrl_dims, mrl_weight=mrl_weight, **kwargs)
        self.method_name = "grace_mrl_ml"
        self.ml_weight = float(ml_weight)
        self.grace_only_epochs = int(grace_only_epochs)
        self.grace_ml_epochs = int(grace_ml_epochs)
        self.ml_module = ml_module
        self.verbose = verbose

        if ml_module == "ml2":
            from .ML_module2 import MutualLearningLoss2
            self.ml_calculator = MutualLearningLoss2(
                mrl_dims=self.mrl_dims,
                ml_weight=ml_weight,
                tau=cdmd_tau,
            )
        else:
            self.ml_calculator = MutualLearningLoss(
                mrl_dims=self.mrl_dims,
                ml_weight=ml_weight,
                tau=cdmd_tau,
            )
        self.epoch = 0

    # ------------------------------------------------------------------
    # 覆盖父类的损失计算方法，叠加互学习损失（v4：全 B×B 矩阵 + 对称 KL）
    # ------------------------------------------------------------------

    def _grace_prefix_loss_with_details(
        self,
        h1_full: Tensor,
        h2_full: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """计算各维度 GRACE 损失 + 互学习损失（v4：全 B×B 矩阵，两阶段训练）。

        GRACE 损失部分：与父类 GRACEWithMRLMethod 完全一致，
        通过 self.model.nt_xent 调用，保证计算结果严格等价。

        两阶段训练策略：
        - 第一阶段（epoch <= grace_only_epochs）：ml_loss = 0，仅训练 GRACE+MRL
        - 第二阶段（epoch > grace_only_epochs）：加入互学习损失

        互学习损失部分（v4）：
        直接传入完整编码器输出，由 MutualLearningLoss.compute()
        内部处理：切片 → B×B 矩阵 → flatten → ReLU → 双向相对熵 / B

        参数:
            h1_full: 第一视图的编码器输出，形状 (B, hidden_dim)
            h2_full: 第二视图的编码器输出，形状 (B, hidden_dim)

        返回:
            total: 总损失标量
            dim_losses: 各分量损失字典
        """
        dim_losses: Dict[str, float] = {}
        loss_sum = torch.zeros((), device=h1_full.device)

        for dim in self.mrl_dims:
            h1 = h1_full[:, :dim]
            h2 = h2_full[:, :dim]

            # GRACE 损失：与父类完全一致的调用方式
            dim_loss = self.model.nt_xent(h1, h2, self.model.tau)
            loss_sum = loss_sum + dim_loss
            dim_losses[f"dim_{dim}"] = float(dim_loss.detach().item())

        # MRL 损失：与父类公式一致
        grace_loss_avg = loss_sum / len(self.mrl_dims)
        mrl_loss = self.mrl_weight * grace_loss_avg

        # 第一阶段只训练 GRACE（无互学习损失），第二阶段再加入互学习损失
        if self.epoch > self.grace_only_epochs and self.ml_weight > 0.0:
            ml_loss = self.ml_calculator.compute(h1_full, h2_full)
        else:
            ml_loss = torch.zeros((), device=h1_full.device)

        # 总损失
        total = mrl_loss + ml_loss

        # 记录各分量损失，便于外部查询和日志输出
        dim_losses["ml_loss"] = float(ml_loss.detach().item())
        dim_losses["grace_loss"] = float(grace_loss_avg.detach().item())
        dim_losses["mrl_loss"] = float(mrl_loss.detach().item())

        return total, dim_losses

    # ------------------------------------------------------------------
    # 训练步骤：继承父类逻辑（GRACE 损失 + 互学习损失已在
    # _grace_prefix_loss_with_details 中计算），
    # 父类的 train_step 自动使用覆盖后的损失函数，无需额外修改。
    # ------------------------------------------------------------------
