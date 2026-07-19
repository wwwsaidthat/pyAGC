"""LRGAE 方法类。

LRGAE (Low-Rank Graph Auto-Encoder) 在标准 GAE 的基础上，
将重建目标从二值邻接矩阵替换为低秩去噪版本：

1. 对归一化邻接矩阵 Â = D^{-1/2} A D^{-1/2} 做截断 SVD
2. 得到目标嵌入 T = U Σ^{1/2}（形状 N×rank）
3. 训练 GCN 编码器使 z_i·z_j 匹配低秩目标 t_i·t_j
4. Loss = MSE(z_i·z_j/√d, t_i·t_j) + MSE(z_i·z_j/√d, 0)（负边）

低秩近似天然滤除高频噪声边，提供更干净的训练信号。
"""

from typing import List, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.encoders import GCN
from pyagc.models.lrgae import LRGAE, compute_low_rank_targets

from .base_method import BaseMethod


class LRGAEMethod(BaseMethod):
    """LRGAE 自监督方法。

    Args:
        in_dim: 输入特征维度。
        hidden_dim: 隐藏/输出维度。
        num_layers: GCN 层数。
        dropout: Dropout 比率。
        neg_ratio: 负边采样比例（相对正边数），默认 1.0。大图建议 0.25--0.5。
        lrgae_rank: SVD 截断秩，默认等于 hidden_dim。
        use_amp: 是否启用自动混合精度（AMP），可节省约 40% 显存。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        neg_ratio: float = 1.0,
        lrgae_rank: Optional[int] = None,
        use_amp: bool = False,
    ) -> None:
        super().__init__(method_name="lrgae", is_supervised=False)

        self.hidden_dim = hidden_dim
        self.neg_ratio = float(neg_ratio)
        self.lrgae_rank = lrgae_rank if lrgae_rank is not None else hidden_dim
        self.use_amp = bool(use_amp)

        # Build encoder and immediately wrap in LRGAE with a placeholder target.
        # The real low-rank target (needs edge_index for SVD) is computed lazily
        # on the first training/inference step and swapped in.
        # IMPORTANT: encoder ONLY lives inside self.model — no separate
        # self.encoder attribute, so state_dict keys are consistent.
        encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm="batch_norm",
        )
        rank = self.lrgae_rank
        placeholder = torch.zeros(1, rank)  # dummy target, replaced in _ensure_prepared
        self.model = LRGAE(
            encoder=encoder,
            target_embeddings=placeholder,
            neg_ratio=self.neg_ratio,
        )
        self._targets_prepared = False

    def output_dim(self) -> int:
        return self.hidden_dim

    def _ensure_prepared(self, data: Data) -> None:
        """Compute low-rank SVD targets and swap them into self.model (once).

        Guarded by both the _targets_prepared flag AND a shape check:
        if target_embeddings already has N rows (loaded from checkpoint),
        skip recomputation.
        """
        if self._targets_prepared:
            return
        # Checkpoint already contains real targets (shape N×rank, not 1×rank placeholder)
        if self.model.target_embeddings.size(0) > 1:
            self._targets_prepared = True
            return

        rank = min(self.lrgae_rank, self.hidden_dim, data.num_nodes - 2)
        edge_index_cpu = data.edge_index.cpu()

        targets = compute_low_rank_targets(
            edge_index=edge_index_cpu,
            num_nodes=data.num_nodes,
            rank=rank,
        )
        # Replace the placeholder buffer with real targets
        self.model.register_buffer("target_embeddings", targets)
        self._targets_prepared = True

    # ------------------------------------------------------------------
    # Full-graph training
    # ------------------------------------------------------------------

    def ssl_train_step_full(
        self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer
    ) -> float:
        self._ensure_prepared(data)
        self.train()

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp and device.type == "cuda"):
            z = self.model.forward(x, edge_index)
            loss_tensor = self.model.recon_loss(z, edge_index)
        loss_tensor.backward()
        optimizer.step()

        return float(loss_tensor.item())

    # ------------------------------------------------------------------
    # Neighbour-sampling training (large graphs)
    # ------------------------------------------------------------------

    def ssl_train_step_neighbor(
        self,
        data: Data,
        input_nodes: Optional[Tensor],
        num_neighbors: Sequence[int],
        batch_size: int,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self._ensure_prepared(data)
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
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp and device.type == "cuda"):
                z = self.model.forward(batch.x, batch.edge_index)
                node_map = getattr(batch, "n_id", None)
                loss_tensor = self.model.recon_loss(
                    z, batch.edge_index,
                    node_map=node_map,
                )
            loss_tensor.backward()
            optimizer.step()

            n_seed = int(batch.batch_size)
            total += float(loss_tensor.item()) * n_seed
            count += n_seed
            del batch

        if device.type == "cuda":
            torch.cuda.empty_cache()
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer_embeddings(
        self,
        data: Data,
        mode: str,
        device: torch.device,
        eval_num_neighbors: Sequence[int],
        eval_batch_size: int,
    ) -> Tensor:
        self._ensure_prepared(data)
        self.eval()

        if mode == "full":
            return self.model.forward(
                data.x.to(device), data.edge_index.to(device)
            ).cpu()

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
            h = self.model.forward(batch.x, batch.edge_index)[: batch.batch_size]
            out.append(h.cpu())
        return torch.cat(out, dim=0)
