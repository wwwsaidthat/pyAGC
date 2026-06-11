"""GRACE + MRL + Mutual Learning 方法类。

继承自 GRACEWithMRLMethod，复用其 MRL 训练逻辑（数据增强、嵌入提取、
多维度 GRACE 损失计算），在此基础上叠加互学习损失。

损失公式：
L_total = mrl_weight * (1/|M|) * sum(L_GRACE^i)
        + ml_weight * (1/(|M|-1)) * sum(L_ML^{i-1,i})

其中：
L_ML^{i-1,i} = mean(abs(State_{i-1} - State_i))
State = 0.5 * (2 - p(u,v) - p(v,u))

GRACE 损失计算与父类 GRACEWithMRLMethod 完全一致，
均通过 self.model.nt_xent 调用，保证 ml_weight=0 时严格等价。
"""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .grace_mrl_method import GRACEWithMRLMethod
from .ML_module import MutualLearningLoss, compute_state


class GRACEWithMRLMutualLearningMethod(GRACEWithMRLMethod):
    """GRACE + MRL + 互学习融合方法。

    继承自 GRACEWithMRLMethod，复用完整的 MRL 训练流程。
    GRACE 损失部分与父类调用同一个 self.model.nt_xent，
    仅在此基础上额外计算 State 并叠加互学习损失。
    ml_weight=0 时与 GRACEWithMRLMethod 严格等价。
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
        self.method_name = "grace_mrl_ml"
        self.ml_weight = float(ml_weight)
        self.verbose = verbose
        self.ml_calculator = MutualLearningLoss(
            mrl_dims=self.mrl_dims, ml_weight=ml_weight
        )
        self.epoch = 0

    # ------------------------------------------------------------------
    # 覆盖父类的损失计算方法，叠加互学习损失
    # ------------------------------------------------------------------

    def _grace_prefix_loss_with_details(
        self, z1_full: Tensor, z2_full: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """计算各维度 GRACE 损失 + 互学习损失。

        GRACE 损失部分：与父类 GRACEWithMRLMethod 完全一致，
        通过 self.model.nt_xent 调用，保证计算结果严格等价。

        State 计算：复用 nt_xent 中的 L2 归一化 + 相似度矩阵计算，
        然后提取正样本对概率 → State → 互学习损失。
        """
        dim_losses: Dict[str, float] = {}
        loss_sum = torch.zeros((), device=z1_full.device)

        # 仅当 ml_weight > 0 时才计算 State，避免 ml_weight=0 时
        # 额外的 autograd 节点影响 GRACE 损失的反向传播精度
        compute_ml = self.ml_weight > 0.0
        dim_states: Dict[str, Tensor] = {}

        for dim in self.mrl_dims:
            z1 = z1_full[:, :dim]
            z2 = z2_full[:, :dim]

            # GRACE 损失：与父类完全一致的调用方式
            dim_loss = self.model.nt_xent(z1, z2, self.model.tau)
            loss_sum = loss_sum + dim_loss
            dim_losses[f"dim_{dim}"] = float(dim_loss.detach().item())

            # State 计算（仅 ml_weight > 0 时执行）
            if compute_ml:
                z1_norm = F.normalize(z1, dim=-1)
                z2_norm = F.normalize(z2, dim=-1)
                sim = torch.matmul(z1_norm, z2_norm.T) / self.model.tau
                prob_u_v = F.softmax(sim, dim=1)
                prob_v_u = F.softmax(sim.t(), dim=1)
                idx = torch.arange(z1.size(0), device=z1.device)
                p_u_v = prob_u_v[idx, idx]
                p_v_u = prob_v_u[idx, idx]
                dim_states[f"dim_{dim}"] = compute_state(p_u_v, p_v_u)

        # MRL 损失：与父类公式一致
        grace_loss_avg = loss_sum / len(self.mrl_dims)
        mrl_loss = self.mrl_weight * grace_loss_avg

        # 互学习损失（ml_weight=0 时此项恒为 0，无额外 autograd 节点）
        ml_loss = self.ml_calculator.compute(dim_states) if compute_ml else torch.zeros((), device=z1_full.device)

        # 总损失
        total = mrl_loss + ml_loss

        # 记录各分量损失，便于外部查询和日志输出
        dim_losses["ml_loss"] = float(ml_loss.detach().item())
        dim_losses["grace_loss"] = float(grace_loss_avg.detach().item())
        dim_losses["mrl_loss"] = float(mrl_loss.detach().item())

        return total, dim_losses

    # ------------------------------------------------------------------
    # 训练步骤：继承父类逻辑（GRACE 损失 + 互学习损失已在 _grace_prefix_loss_with_details 中计算），
    # 父类的 train_step 自动使用覆盖后的损失函数，无需额外 print。
    # ------------------------------------------------------------------
