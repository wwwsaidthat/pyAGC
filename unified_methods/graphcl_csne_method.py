"""CDMD + HPEM + DAS on top of node-level GraphCL_MRL."""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .DAS_module import DASScheduler
from .HPEM_module import HPEMLoss, compute_confusion_scores, confusion_to_weights
from .ML_module import MutualLearningLoss
from .graphcl_mrl_method import GraphCLWithMRLMethod


class GraphCLCSNEMethod(GraphCLWithMRLMethod):
    def __init__(
        self,
        *args,
        mrl_dims: Sequence[int],
        mrl_weight: float = 1.0,
        ml_weight: float = 1.0,
        ml_module: str = "ml",
        cdmd_tau: float = 0.5,
        hpem_beta_init: float = 0.1,
        hpem_tau_0: float = 0.2,
        warmup_epochs: int = 100,
        full_epochs: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            mrl_dims=mrl_dims,
            mrl_weight=mrl_weight,
            **kwargs,
        )
        self.method_name = "graphcl_csne"
        self.warmup_epochs = int(warmup_epochs)
        self.full_epochs = int(full_epochs)
        self.epoch = 0
        if ml_module == "ml2":
            from .ML_module2 import MutualLearningLoss2
            self.cdmd = MutualLearningLoss2(self.mrl_dims, ml_weight=ml_weight, tau=cdmd_tau)
        else:
            self.cdmd = MutualLearningLoss(self.mrl_dims, ml_weight=ml_weight, tau=cdmd_tau)
        max_dim = max(self.mrl_dims)
        self.hpem = HPEMLoss(
            tau_0=hpem_tau_0,
            max_dim=max_dim,
            beta_init=hpem_beta_init,
        )
        self.das = DASScheduler(max_dim=max_dim)

    @staticmethod
    def _directional_per_anchor(z1: Tensor, z2: Tensor, tau: Tensor) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.T / tau
        diagonal = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        negative_lse = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
        return -logits.diag() + negative_lse

    def _hpem_graphcl_loss(
        self,
        z1_curr: Tensor,
        z2_curr: Tensor,
        z1_prev: Tensor,
        z2_prev: Tensor,
        dim_curr: int,
    ) -> Tensor:
        tau = self.hpem.get_tau(dim_curr)
        confusion = compute_confusion_scores(z1_prev, z2_prev)
        weights = confusion_to_weights(confusion, tau, self.hpem.beta)
        forward = self._directional_per_anchor(z1_curr, z2_curr, tau)
        if self.model.symmetric_loss:
            backward = self._directional_per_anchor(z2_curr, z1_curr, tau)
            per_anchor = 0.5 * (forward + backward)
        else:
            per_anchor = forward
        return (weights * per_anchor).sum()

    def _projected_loss(self, z1: Tensor, z2: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        if self.epoch <= self.warmup_epochs:
            loss, details = super()._projected_loss(z1, z2)
            details["cdmd_loss"] = 0.0
            details["hpem_loss"] = 0.0
            return loss, details

        cdmd_loss = self.cdmd.compute(z1, z2)
        hpem_total = torch.zeros((), device=z1.device)
        details: Dict[str, float] = {}

        first_dim = self.mrl_dims[0]
        first_tau = self.hpem.get_tau(first_dim)
        first_loss = self.model.contrastive_loss(
            z1[:, :first_dim], z2[:, :first_dim], tau=first_tau
        )
        hpem_total = hpem_total + self.das.get_hpem_weight(first_dim) * first_loss
        details[f"dim_{first_dim}"] = float(first_loss.detach().item())

        for previous, current in zip(self.mrl_dims[:-1], self.mrl_dims[1:]):
            current_loss = self._hpem_graphcl_loss(
                z1_curr=z1[:, :current],
                z2_curr=z2[:, :current],
                z1_prev=z1[:, :previous],
                z2_prev=z2[:, :previous],
                dim_curr=current,
            )
            hpem_total = hpem_total + self.das.get_hpem_weight(current) * current_loss
            details[f"dim_{current}"] = float(current_loss.detach().item())

        total = cdmd_loss + hpem_total
        details["cdmd_loss"] = float(cdmd_loss.detach().item())
        details["hpem_loss"] = float(hpem_total.detach().item())
        details["total_loss"] = float(total.detach().item())
        details.update(self.hpem.log_info())
        details.update(self.das.log_info())
        return total, details

