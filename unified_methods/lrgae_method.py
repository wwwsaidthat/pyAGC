"""Unified training adapter for the paper's left-right GAE."""

from typing import List, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.models.lrgae import EdgeDecoder, LayerwiseGCNEncoder, LRGAE

from .base_method import BaseMethod


class LRGAEMethod(BaseMethod):
    """Structure-based lrGAE-vu for node representation learning."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        neg_ratio: float = 1.0,
        variant: int = 8,
        mask_ratio: float = 0.7,
        decoder_dim: int = 32,
        decoder_layers: int = 2,
        decoder_dropout: float = 0.2,
        decoder_batch_size: int = 131072,
        grad_norm: float = 1.0,
        use_amp: bool = False,
    ) -> None:
        super().__init__(method_name="lrgae", is_supervised=False)
        self.hidden_dim = int(hidden_dim)
        self.grad_norm = float(grad_norm)
        self.use_amp = bool(use_amp)

        encoder = LayerwiseGCNEncoder(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm="none",
        )
        decoder = EdgeDecoder(
            in_channels=hidden_dim,
            hidden_channels=decoder_dim,
            num_layers=decoder_layers,
            dropout=decoder_dropout,
        )
        self.model = LRGAE(
            encoder=encoder,
            decoder=decoder,
            variant=variant,
            mask_ratio=mask_ratio,
            neg_ratio=neg_ratio,
            decoder_batch_size=decoder_batch_size,
        )

    def output_dim(self) -> int:
        return self.hidden_dim

    def _optimise(
        self,
        x: Tensor,
        edge_index: Tensor,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        optimizer.zero_grad()
        with torch.amp.autocast(
            "cuda", enabled=self.use_amp and device.type == "cuda"
        ):
            loss = self.model.loss(x, edge_index)
        loss.backward()
        if self.grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
        optimizer.step()
        return float(loss.item())

    def ssl_train_step_full(
        self,
        data: Data,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.train()
        return self._optimise(
            data.x.to(device), data.edge_index.to(device), device, optimizer
        )

    def ssl_train_step_neighbor(
        self,
        data: Data,
        input_nodes: Optional[Tensor],
        num_neighbors: Sequence[int],
        batch_size: int,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Fallback for graphs that do not fit in memory.

        Each sampled graph is treated as a complete local training graph.  For
        faithful ogbn-arxiv runs prefer ``mode=full`` so negative sampling can
        exclude every true edge globally.
        """
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
            loss = self._optimise(
                batch.x, batch.edge_index, device, optimizer
            )
            seeds = int(batch.batch_size)
            total += loss * seeds
            count += seeds
            del batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return total / max(count, 1)

    @torch.no_grad()
    def infer_embeddings(
        self,
        data: Data,
        mode: str,
        device: torch.device,
        eval_num_neighbors: Sequence[int],
        eval_batch_size: int,
    ) -> Tensor:
        self.eval()
        if mode == "full":
            return self.model(
                data.x.to(device), data.edge_index.to(device)
            ).cpu()

        loader = NeighborLoader(
            data,
            input_nodes=None,
            num_neighbors=list(eval_num_neighbors),
            batch_size=eval_batch_size,
            shuffle=False,
        )
        outputs: List[Tensor] = []
        for batch in loader:
            batch = batch.to(device)
            embedding = self.model(batch.x, batch.edge_index)
            outputs.append(embedding[: batch.batch_size].cpu())
        return torch.cat(outputs, dim=0)
