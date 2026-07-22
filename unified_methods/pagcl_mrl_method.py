"""Matryoshka extension of PaGCL's progressive invariance objective."""

from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor

from .pagcl_method import PaGCLMethod


class PaGCLWithMRLMethod(PaGCLMethod):
    def __init__(
        self,
        *args,
        mrl_dims: Sequence[int],
        mrl_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.method_name = "pagcl_mrl"
        self.mrl_dims = sorted({int(dim) for dim in mrl_dims})
        if not self.mrl_dims or any(dim <= 0 for dim in self.mrl_dims):
            raise ValueError("pagcl_mrl_dims must be non-empty positive integers")
        if self.hidden_dim < max(self.mrl_dims):
            raise ValueError("PaGCL_MRL requires hidden_dim >= max(mrl_dims)")
        # The paper objective acts directly on the fused representation.
        # Matryoshka prefixes therefore slice that same downstream embedding,
        # rather than an auxiliary contrastive projection head.
        self.mrl_weight = float(mrl_weight)

    def _projected_sequence_loss(self, projected: Sequence[Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        total = torch.zeros((), device=projected[0].device)
        details: Dict[str, float] = {}
        for dim in self.mrl_dims:
            dim_views = [embedding[:, :dim] for embedding in projected]
            dim_loss, invariant, variant = self.model.invariance_objective(dim_views)
            total = total + dim_loss
            details[f"dim_{dim}"] = float(dim_loss.detach().item())
            details[f"invariant_{dim}"] = float(invariant.detach().item())
            details[f"variant_{dim}"] = float(variant.detach().item())
        average = total / len(self.mrl_dims)
        weighted = self.mrl_weight * average
        details["pagcl_loss"] = float(average.detach().item())
        details["mrl_loss"] = float(weighted.detach().item())
        return weighted, details
