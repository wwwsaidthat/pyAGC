"""MCNE extension for PaGCL progressive views.

PaGCL itself uses Eq. (9)--(11), not InfoNCE.  This extension retains the
PaGCL objective and adds MCNE on every adjacent pair of progressive views.
Consequently it is an explicit PaGCL+MCNE extension, rather than a claim that
the original KDD paper contains CDMD/HPEM/DALS.
"""

from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor

from .DAS_module import DASScheduler
from .HPEM_module import HPEMLoss
from .ML_module2 import MutualLearningLoss2
from .pagcl_mrl_method import PaGCLWithMRLMethod


class PaGCLMCNEMethod(PaGCLWithMRLMethod):
    def __init__(
        self,
        *args,
        mrl_dims: Sequence[int],
        mrl_weight: float = 1.0,
        ml_weight: float = 1.0,
        cdmd_tau: float = 0.5,
        hpem_beta_init: float = 0.1,
        hpem_tau_0: float = 0.5,
        mcne_weight: float = 1.0,
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
        if mcne_weight < 0:
            raise ValueError("pagcl_mcne_weight must be >= 0")
        self.method_name = "pagcl_mcne"
        self.cdmd = MutualLearningLoss2(self.mrl_dims, ml_weight=ml_weight, tau=cdmd_tau)
        max_dim = max(self.mrl_dims)
        self.hpem = HPEMLoss(
            tau_0=hpem_tau_0,
            max_dim=max_dim,
            beta_init=hpem_beta_init,
        )
        self.das = DASScheduler(max_dim=max_dim)
        self.mcne_weight = float(mcne_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.full_epochs = int(full_epochs)
        self.epoch = 0

    def _mcne_pair(self, z1: Tensor, z2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        cdmd = self.cdmd.compute(z1, z2)
        hpem = torch.zeros((), device=z1.device)

        first_dim = self.mrl_dims[0]
        first_tau = self.hpem.get_tau(first_dim)
        first_loss = self.model_infonce(z1[:, :first_dim], z2[:, :first_dim], first_tau)
        hpem = hpem + self.das.get_hpem_weight(first_dim) * first_loss

        for previous, current in zip(self.mrl_dims[:-1], self.mrl_dims[1:]):
            current_loss = self.hpem.compute_single(
                z1[:, :current],
                z2[:, :current],
                z1[:, :previous],
                z2[:, :previous],
                dim_curr=current,
            )
            hpem = hpem + self.das.get_hpem_weight(current) * current_loss

        # MCNE overview objective averages dimensional HPEM terms by K.
        hpem = hpem / len(self.mrl_dims)
        return cdmd + hpem, cdmd, hpem

    @staticmethod
    def model_infonce(z1: Tensor, z2: Tensor, tau: Tensor) -> Tensor:
        """Standard bidirectional InfoNCE with the positive in the denominator."""
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        z2 = torch.nn.functional.normalize(z2, dim=-1)
        logits = z1 @ z2.T / tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (
            torch.nn.functional.cross_entropy(logits, labels)
            + torch.nn.functional.cross_entropy(logits.T, labels)
        )

    def _projected_sequence_loss(self, projected: Sequence[Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        pagcl_loss, details = super()._projected_sequence_loss(projected)
        if self.epoch <= self.warmup_epochs:
            details["cdmd_loss"] = 0.0
            details["hpem_loss"] = 0.0
            details["mcne_loss"] = 0.0
            return pagcl_loss, details

        pair_totals = []
        pair_cdmd = []
        pair_hpem = []
        for left, right in zip(projected[:-1], projected[1:]):
            pair_total, cdmd, hpem = self._mcne_pair(left, right)
            pair_totals.append(pair_total)
            pair_cdmd.append(cdmd)
            pair_hpem.append(hpem)
        mcne_loss = torch.stack(pair_totals).mean()
        cdmd_loss = torch.stack(pair_cdmd).mean()
        hpem_loss = torch.stack(pair_hpem).mean()
        total = pagcl_loss + self.mcne_weight * mcne_loss
        details["cdmd_loss"] = float(cdmd_loss.detach().item())
        details["hpem_loss"] = float(hpem_loss.detach().item())
        details["mcne_loss"] = float(mcne_loss.detach().item())
        details["total_loss"] = float(total.detach().item())
        details.update(self.hpem.log_info())
        details.update(self.das.log_info())
        return total, details
