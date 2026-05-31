"""GRACE + MRL 方法类。"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .grace_method import GRACEMethod


class GRACEWithMRLMethod(GRACEMethod):
    """GRACE + MRL 融合方法。"""

    def __init__(self, *args, mrl_dims: Sequence[int], mrl_weight: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.method_name = "grace_mrl"
        dims = sorted({int(d) for d in mrl_dims})
        if len(dims) == 0:
            raise ValueError("mrl_dims 不能为空")
        if any(d <= 0 for d in dims):
            raise ValueError("mrl_dims 中所有维度必须 > 0")
        self.mrl_dims = dims
        self.mrl_weight = float(mrl_weight)
        max_dim = int(max(self.mrl_dims))
        proj_out = None
        if hasattr(self.model, "projector") and hasattr(self.model.projector, "__len__") and len(self.model.projector) > 0:
            last = self.model.projector[-1]
            if hasattr(last, "out_features"):
                proj_out = int(last.out_features)
        if proj_out is not None and proj_out < max_dim:
            raise ValueError(f"GRACE+MRL 需要 --proj-dim >= max(mrl_dims)={max_dim}, 但当前 proj-dim={proj_out}")
        self.last_mrl_dim_losses: Dict[str, float] = {}

    def output_dim(self) -> int:
        return int(max(self.mrl_dims))

    def _grace_prefix_loss_with_details(self, z1_full: Tensor, z2_full: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        dim_losses: Dict[str, float] = {}
        loss_sum = torch.zeros((), device=z1_full.device)
        for dim in self.mrl_dims:
            z1 = z1_full[:, :dim]
            z2 = z2_full[:, :dim]
            dim_loss = self.model.nt_xent(z1, z2, self.model.tau)
            loss_sum = loss_sum + dim_loss
            dim_losses[f"dim_{dim}"] = float(dim_loss.detach().item())
        total = self.mrl_weight * loss_sum / len(self.mrl_dims)
        return total, dim_losses

    def ssl_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        self.train()
        optimizer.zero_grad()
        x = data.x.to(device)
        e = data.edge_index.to(device)
        v1 = self.model.t1(x, e)
        v2 = self.model.t2(x, e)
        h1 = self.model.embed(v1["x"], v1["edge_index"])
        h2 = self.model.embed(v2["x"], v2["edge_index"])
        z1_full = self.model.projector(h1)
        z2_full = self.model.projector(h2)
        loss, dim_losses = self._grace_prefix_loss_with_details(z1_full, z2_full)
        loss.backward()
        optimizer.step()
        self.last_mrl_dim_losses = dim_losses
        return float(loss.item())

    def ssl_train_step_neighbor(
        self,
        data: Data,
        input_nodes: Optional[Tensor],
        num_neighbors: Sequence[int],
        batch_size: int,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.train()
        loader = NeighborLoader(
            data,
            input_nodes=input_nodes,
            num_neighbors=list(num_neighbors),
            batch_size=batch_size,
            shuffle=True,
        )
        total = 0.0
        count = 0
        dim_loss_sum: Dict[str, float] = {}
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            v1 = self.model.t1(batch.x, batch.edge_index)
            v2 = self.model.t2(batch.x, batch.edge_index)
            h1 = self.model.embed(v1["x"], v1["edge_index"])[: batch.batch_size]
            h2 = self.model.embed(v2["x"], v2["edge_index"])[: batch.batch_size]
            z1_full = self.model.projector(h1)
            z2_full = self.model.projector(h2)
            loss, dim_losses = self._grace_prefix_loss_with_details(z1_full, z2_full)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(batch.batch_size)
            count += int(batch.batch_size)
            for k, v in dim_losses.items():
                dim_loss_sum[k] = dim_loss_sum.get(k, 0.0) + v * int(batch.batch_size)
        if count > 0:
            self.last_mrl_dim_losses = {k: v / count for k, v in dim_loss_sum.items()}
        return total / max(count, 1)

    @torch.no_grad()
    def infer_embeddings(self, data: Data, mode: str, device: torch.device, eval_num_neighbors: Sequence[int], eval_batch_size: int) -> Tensor:
        self.eval()
        if mode == "full":
            h = self.model.embed(data.x.to(device), data.edge_index.to(device))
            return self.model.projector(h).cpu()
        loader = NeighborLoader(
            data,
            input_nodes=None,
            num_neighbors=list(eval_num_neighbors),
            batch_size=eval_batch_size,
            shuffle=False,
        )
        out: List[Tensor] = []
        for batch in loader:
            batch = batch.to(device)
            h = self.model.embed(batch.x, batch.edge_index)[: batch.batch_size]
            out.append(self.model.projector(h).cpu())
        return torch.cat(out, dim=0)

    def get_last_mrl_dim_losses(self) -> Dict[str, float]:
        """返回最近一个 epoch 的各维度 MRL 损失。"""
        return dict(self.last_mrl_dim_losses)
