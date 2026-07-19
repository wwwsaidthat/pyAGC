"""Experimental CSNE objective on top of paper-style BES_MRL.

This is a new combination, not a method claimed by either source paper.  The
two CSNE views are stochastic masks of the frozen multi-view BES features.
"""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .DAS_module import DASScheduler
from .HPEM_module import HPEMLoss
from .ML_module import MutualLearningLoss
from .bes_mrl_method import BESWithMRLMethod


class BESCSNEMethod(BESWithMRLMethod):
    def __init__(
        self,
        *args,
        mrl_dims: Sequence[int],
        mrl_weight: float = 1.0,
        csne_weight: float = 1.0,
        ml_weight: float = 1.0,
        ml_module: str = "ml",
        hpem_beta_init: float = 0.1,
        hpem_tau_0: float = 0.5,
        csne_warmup_epochs: int = 25,
        csne_batch_size: int = 1024,
        view_mask_1: float = 0.2,
        view_mask_2: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(*args, mrl_dims=mrl_dims, mrl_weight=mrl_weight, **kwargs)
        self.method_name = "bes_csne"
        self.csne_weight = float(csne_weight)
        self.csne_warmup_epochs = int(csne_warmup_epochs)
        self.csne_batch_size = int(csne_batch_size)
        self.view_mask_1 = float(view_mask_1)
        self.view_mask_2 = float(view_mask_2)
        if self.csne_batch_size < 2:
            raise ValueError("bes_csne_batch_size 必须 >= 2")
        if not (0 <= self.view_mask_1 < 1 and 0 <= self.view_mask_2 < 1):
            raise ValueError("BES_CSNE view mask 必须在 [0, 1) 内")
        max_dim = max(self.mrl_dims)
        if ml_module == "ml2":
            from .ML_module2 import MutualLearningLoss2
            self.cdmd = MutualLearningLoss2(self.mrl_dims, ml_weight=ml_weight)
        else:
            self.cdmd = MutualLearningLoss(self.mrl_dims, ml_weight=ml_weight)
        self.hpem = HPEMLoss(tau_0=hpem_tau_0, max_dim=max_dim, beta_init=hpem_beta_init)
        self.das = DASScheduler(max_dim=max_dim)

    @staticmethod
    def _nt_xent(z1: Tensor, z2: Tensor, tau: float) -> Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        similarity = z1 @ z2.T / float(tau)
        targets = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (
            F.cross_entropy(similarity, targets)
            + F.cross_entropy(similarity.T, targets)
        )

    def _csne_loss(self, z1: Tensor, z2: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        details: Dict[str, float] = {}
        shaping_epoch = max(self.epoch - self.backbone_epochs, 1)
        if shaping_epoch <= self.csne_warmup_epochs:
            losses = []
            for dim in self.mrl_dims:
                dim_loss = self._nt_xent(z1[:, :dim], z2[:, :dim], self.tau)
                losses.append(dim_loss)
                details[f"csne_dim_{dim}"] = float(dim_loss.detach().item())
            warmup = torch.stack(losses).mean()
            details["csne_warmup_loss"] = float(warmup.detach().item())
            return warmup, details

        cdmd = self.cdmd.compute(z1, z2)
        hpem_total = z1.sum() * 0.0
        first_dim = self.mrl_dims[0]
        first_tau = self.hpem.get_tau(first_dim)
        first_loss = self._nt_xent(z1[:, :first_dim], z2[:, :first_dim], float(first_tau.detach()))
        hpem_total = hpem_total + self.das.get_hpem_weight(first_dim) * first_loss
        for previous, current in zip(self.mrl_dims[:-1], self.mrl_dims[1:]):
            current_loss = self.hpem.compute_single(
                z1_curr=z1[:, :current],
                z2_curr=z2[:, :current],
                z1_prev=z1[:, :previous],
                z2_prev=z2[:, :previous],
                dim_curr=current,
            )
            hpem_total = hpem_total + self.das.get_hpem_weight(current) * current_loss
        total = cdmd + hpem_total
        details["cdmd_loss"] = float(cdmd.detach().item())
        details["hpem_loss"] = float(hpem_total.detach().item())
        details.update(self.hpem.log_info())
        details.update(self.das.log_info())
        return total, details

    def _extra_shaping_objective(
        self,
        input_views: Tensor,
        train_indices: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        if train_indices.numel() > self.csne_batch_size:
            selection = torch.randperm(train_indices.numel(), device=train_indices.device)[: self.csne_batch_size]
            indices = train_indices[selection]
        else:
            indices = train_indices
        selected = input_views[indices]
        layer = self.boundary_attention.layers[self._active_attention_layer()]
        model_device = next(layer.parameters()).device
        selected = selected.to(model_device)
        view_1 = F.dropout(selected, p=self.view_mask_1, training=True)
        view_2 = F.dropout(selected, p=self.view_mask_2, training=True)
        z1 = layer(view_1).mean(dim=1)
        z2 = layer(view_2).mean(dim=1)
        loss, details = self._csne_loss(z1, z2)
        weighted = self.csne_weight * loss
        details["csne_loss"] = float(weighted.detach().item())
        return weighted, details
