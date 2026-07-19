"""LRGAE 方法类。

LRGAE (Low-Rank Graph Auto-Encoder) 通过重建归一化邻接矩阵的
PSD 低秩谱近似来学习节点嵌入。

1. Â = D^{-1/2} A D^{-1/2}
2. eigsh(which='LA') → (λ_+, Q_+)  正特征值
3. T = Q_+ √Λ_+    PSD 目标嵌入（纯 CPU 属性，不随模型迁移 GPU）
4. MSE(z_i·z_j/√d,  t_i·t_j)  对正负边都有软目标
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.encoders import GCN
from pyagc.models.lrgae import LRGAE, compute_low_rank_targets

from .base_method import BaseMethod


class LRGAEMethod(BaseMethod):
    """LRGAE — PSD 低秩图重建。

    Args:
        in_dim: 输入特征维度。
        hidden_dim: 隐藏/输出维度。
        num_layers: GCN 层数。
        dropout: Dropout 比率。
        neg_ratio: 负边采样比例，默认 1.0。
        lrgae_rank: 特征值保留数，默认 = hidden_dim。
        add_self_loops: 归一化邻接是否加自环。
        use_amp: 启用 AMP。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        neg_ratio: float = 1.0,
        lrgae_rank: Optional[int] = None,
        add_self_loops: bool = False,
        use_amp: bool = False,
    ) -> None:
        super().__init__(method_name="lrgae", is_supervised=False)

        self.hidden_dim = hidden_dim
        self.neg_ratio = float(neg_ratio)
        self.lrgae_rank = lrgae_rank if lrgae_rank is not None else hidden_dim
        self.add_self_loops = bool(add_self_loops)
        self.use_amp = bool(use_amp)

        encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm="batch_norm",
        )
        self.model = LRGAE(
            encoder=encoder,
            target_embeddings=None,  # lazy — computed or loaded later
            neg_ratio=self.neg_ratio,
        )
        self._targets_prepared = False

    # ------------------------------------------------------------------
    # 基础接口
    # ------------------------------------------------------------------

    def output_dim(self) -> int:
        return self.hidden_dim

    # ------------------------------------------------------------------
    # 加载 / 保存
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        """Override to include the CPU target attribute in the checkpoint."""
        sd = super().state_dict(*args, **kwargs)
        if self.model.has_targets:
            sd["target_embeddings_cpu"] = self.model.target_embeddings_cpu
        return sd

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
    ) -> None:
        """Remap legacy keys and restore the CPU target attribute.

        * ``_model.*`` / ``encoder.*`` → ``model.*`` (old code).
        * ``target_embeddings_cpu`` is extracted before the standard
          loader runs and restored via :meth:`LRGAE.set_targets`.
        """
        # ---- remap legacy keys ----
        _remap: Dict[str, Tensor] = {}
        for k, v in state_dict.items():
            if k == "target_embeddings_cpu":
                continue  # handled separately below
            if k.startswith("_model."):
                _remap[k[len("_model."):]] = v
            elif k.startswith("encoder."):
                _remap["model." + k] = v
            else:
                _remap[k] = v

        # ---- restore CPU target ----
        t_key = "target_embeddings_cpu"
        if t_key in state_dict:
            self.model.set_targets(state_dict[t_key])
            self._targets_prepared = True

        return super().load_state_dict(_remap, strict=strict)

    # ------------------------------------------------------------------
    # PSD 目标预计算（lazy）
    # ------------------------------------------------------------------

    def _ensure_prepared(self, data: Data) -> None:
        """Compute PSD spectral targets (once)."""
        if self._targets_prepared or self.model.has_targets:
            self._targets_prepared = True
            return

        rank = min(self.lrgae_rank, self.hidden_dim, data.num_nodes - 1)
        edge_index_cpu = data.edge_index.cpu()

        targets = compute_low_rank_targets(
            edge_index=edge_index_cpu,
            num_nodes=data.num_nodes,
            rank=rank,
            add_self_loops=self.add_self_loops,
        )
        self.model.set_targets(targets)
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
        with torch.amp.autocast("cuda", enabled=self.use_amp and device.type == "cuda"):
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
            n_seed = int(batch.batch_size)

            # Supervise edges whose source is a seed node.
            ei = batch.edge_index
            seed_mask = ei[0] < n_seed
            pos_edges = ei[:, seed_mask]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp and device.type == "cuda"):
                z = self.model.forward(batch.x, ei)
                node_map = getattr(batch, "n_id", None)
                loss_tensor = self.model.recon_loss(
                    z, pos_edges,
                    node_map=node_map,
                    negative_exclusion_edge_index=ei,
                )
            loss_tensor.backward()
            optimizer.step()

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
