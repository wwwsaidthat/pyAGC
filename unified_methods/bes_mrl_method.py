"""Matryoshka representation learning extension for BES."""

from typing import Dict, Sequence, Tuple

from torch import Tensor

from pyagc.models.bes import BESBoundaryState

from .bes_method import BESMethod


class BESWithMRLMethod(BESMethod):
    def __init__(
        self,
        *args,
        mrl_dims: Sequence[int],
        mrl_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        dims = sorted({int(dim) for dim in mrl_dims})
        if not dims or any(dim <= 0 for dim in dims):
            raise ValueError("mrl_dims 必须是非空的正整数列表")
        if max(dims) > self.hidden_dim:
            raise ValueError(
                f"BES_MRL 需要 hidden_dim >= max(mrl_dims)={max(dims)}，"
                f"当前 hidden_dim={self.hidden_dim}"
            )
        self.method_name = "bes_mrl"
        self.bes_dims = dims
        self.mrl_dims = dims
        self.mrl_weight = float(mrl_weight)

    def _build_boundary_states(
        self,
        input_views: Tensor,
        labels: Tensor,
        train_indices: Tensor,
    ) -> Dict[int, BESBoundaryState]:
        """Mine boundaries once at max dimension, then shape every prefix.

        Centroid prefixes are exact slices of the max-dimensional centroids.
        Sharing the hard-node set makes the nested objectives comparable and
        avoids seven global covariance inversions/k-NN searches per epoch.
        """
        max_dim = max(self.mrl_dims)
        original_dims = self.bes_dims
        self.bes_dims = [max_dim]
        try:
            max_state = super()._build_boundary_states(input_views, labels, train_indices)[max_dim]
        finally:
            self.bes_dims = original_dims
        return {dim: max_state for dim in self.mrl_dims}

    def _sampled_gravity_objective(
        self,
        input_views: Tensor,
        labels: Tensor,
        states: Dict[int, BESBoundaryState],
    ) -> Tuple[Tensor, Dict[str, float]]:
        loss, details = super()._sampled_gravity_objective(input_views, labels, states)
        weighted = self.mrl_weight * loss
        details["mrl_loss"] = float(weighted.detach().item())
        return weighted, details
