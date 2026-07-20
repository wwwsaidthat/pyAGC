"""Matryoshka representation learning extension for node-level GraphCL."""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graphcl_method import GraphCLMethod


class PrefixLinear(nn.Linear):
    """Linear layer whose first d outputs depend only on the first d inputs."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        mask = torch.tril(torch.ones(out_features, in_features))
        self.register_buffer("prefix_mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight * self.prefix_mask, self.bias)


class GraphCLPrefixProjector(nn.Sequential):
    """GraphCL's two-layer MLP with prefix-preserving connectivity for MRL."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(
            PrefixLinear(in_dim, out_dim),
            nn.ReLU(),
            PrefixLinear(out_dim, out_dim),
        )


class GraphCLWithMRLMethod(GraphCLMethod):
    def __init__(
        self,
        *args,
        mrl_dims: Sequence[int],
        mrl_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.method_name = "graphcl_mrl"
        self.mrl_dims = sorted({int(dim) for dim in mrl_dims})
        if not self.mrl_dims or any(dim <= 0 for dim in self.mrl_dims):
            raise ValueError("graphcl_mrl_dims 必须是非空正整数列表")
        if self.hidden_dim < max(self.mrl_dims):
            raise ValueError("GraphCL_MRL 要求 hidden_dim >= max(mrl_dims)")
        projector_out = int(self.model.projector[-1].out_features)
        if projector_out < max(self.mrl_dims):
            raise ValueError("GraphCL_MRL 要求 proj_dim >= max(mrl_dims)")
        # A dense projector would let z[:d] depend on every encoder coordinate,
        # while evaluation slices h[:d]. The masked projector aligns the MRL
        # training objective with the actual prefix embedding used downstream.
        self.model.projector = GraphCLPrefixProjector(self.hidden_dim, projector_out)
        self.mrl_weight = float(mrl_weight)

    def _projected_loss(self, z1: Tensor, z2: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        total = torch.zeros((), device=z1.device)
        details: Dict[str, float] = {}
        for dim in self.mrl_dims:
            dim_loss = self.model.contrastive_loss(z1[:, :dim], z2[:, :dim])
            total = total + dim_loss
            details[f"dim_{dim}"] = float(dim_loss.detach().item())
        average = total / len(self.mrl_dims)
        weighted = self.mrl_weight * average
        details["graphcl_loss"] = float(average.detach().item())
        details["mrl_loss"] = float(weighted.detach().item())
        return weighted, details
