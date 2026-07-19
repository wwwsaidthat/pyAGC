"""Boundary Embedding Shaping (BES) training method.

The implementation follows the paper protocol: supervised multi-view backbone
pretraining, frozen backbone, sequential boundary-attention shaping, global
training-node boundary statistics, gravity loss, and virtual adaptive updates.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.models.bes import (
    BESBoundaryState,
    BESMultiViewBackbone,
    BoundaryAttention,
    compute_gravity_loss,
    detect_boundary_nodes,
)

from .base_method import BaseMethod


class BESMethod(BaseMethod):
    """Paper BES plus an OGB-compatible Neighbor extension.

    Neighbor sampling is used only to train/infer the graph backbone.  Boundary
    statistics remain global, which avoids the invalid batch-local class centers
    from the previous implementation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.5,
        tau: float = 1.0,
        delta: float = 5.0,
        alpha: float = 1.0,
        num_classes: int = 1,
        total_epochs: int = 200,
        backbone_epochs: int = 100,
        num_heads: int = 2,
        attention_layers: int = 2,
        beta_size: int = 256,
        boundary_knn: int = 5,
        covariance_reg: float = 1e-4,
        max_boundary_candidates: int = 0,
        displacement_batch_size: int = 4096,
    ) -> None:
        # BES is label-informed but uses the representation-pretraining pathway.
        super().__init__(method_name="bes", is_supervised=False)
        if num_classes < 2:
            raise ValueError("BES 需要至少两个类别")
        if not 0 <= backbone_epochs < total_epochs:
            raise ValueError("bes_backbone_epochs 必须满足 0 <= value < pretrain_epochs")
        if attention_layers < 1 or beta_size < 1 or boundary_knn < 1:
            raise ValueError("BES attention_layers、beta_size、boundary_knn 必须 >= 1")
        if displacement_batch_size < 1:
            raise ValueError("bes_displacement_batch_size 必须 >= 1")
        self.backbone = BESMultiViewBackbone(in_dim, hidden_dim, num_layers, dropout)
        self.boundary_attention = BoundaryAttention(hidden_dim, num_heads, attention_layers)
        self.pretrain_classifier = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.tau = float(tau)
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.total_epochs = int(total_epochs)
        self.backbone_epochs = int(backbone_epochs)
        self.attention_layers = int(attention_layers)
        self.beta_size = int(beta_size)
        self.boundary_knn = int(boundary_knn)
        self.covariance_reg = float(covariance_reg)
        self.max_boundary_candidates = int(max_boundary_candidates)
        self.displacement_batch_size = int(displacement_batch_size)
        self.register_buffer("_bes_epoch", torch.tensor(0, dtype=torch.long))
        self.bes_dims = [self.hidden_dim]
        self.last_mrl_dim_losses: Dict[str, float] = {}
        self._shaping_cache_layer = -1
        self._shaping_cache_input: Optional[Tensor] = None
        self._shaping_cache_states: Optional[Dict[int, BESBoundaryState]] = None

    @property
    def epoch(self) -> int:
        return int(self._bes_epoch.item())

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._bes_epoch.fill_(int(value))

    def output_dim(self) -> int:
        return self.hidden_dim

    def _active_attention_layer(self) -> int:
        shaping_epochs = max(self.total_epochs - self.backbone_epochs, 1)
        progress = max(self.epoch - self.backbone_epochs - 1, 0)
        return min(progress * self.attention_layers // shaping_epochs, self.attention_layers - 1)

    def _trained_attention_layers(self) -> int:
        if self.epoch <= self.backbone_epochs:
            return 0
        return self._active_attention_layer() + 1

    def _configure_phase(self) -> None:
        backbone_phase = self.epoch <= self.backbone_epochs
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(backbone_phase)
        active = self._active_attention_layer()
        for index, layer in enumerate(self.boundary_attention.layers):
            for parameter in layer.parameters():
                parameter.requires_grad_(not backbone_phase and index == active)
        for parameter in self.pretrain_classifier.parameters():
            parameter.requires_grad_(True)

    def _backbone_pretrain_loss(self, views: Tensor, labels: Tensor, indices: Tensor) -> Tensor:
        embedding = views.mean(dim=1)
        return F.cross_entropy(self.pretrain_classifier(embedding[indices]), labels[indices].long())

    @torch.no_grad()
    def _boundary_inputs(self, views: Tensor) -> Tensor:
        active = self._active_attention_layer()
        if active == 0:
            return views.detach()
        model_device = next(self.boundary_attention.parameters()).device
        if views.device == model_device:
            return self.boundary_attention.forward_views(views, num_layers=active).detach()
        output = torch.empty_like(views, device="cpu")
        for start in range(0, views.size(0), self.displacement_batch_size):
            end = min(start + self.displacement_batch_size, views.size(0))
            shaped = self.boundary_attention.forward_views(
                views[start:end].to(model_device), num_layers=active
            )
            output[start:end].copy_(shaped.cpu())
        return output

    @torch.no_grad()
    def _build_boundary_states(
        self,
        input_views: Tensor,
        labels: Tensor,
        train_indices: Tensor,
    ) -> Dict[int, BESBoundaryState]:
        embedding = input_views.mean(dim=1)
        states: Dict[int, BESBoundaryState] = {}
        for dim in self.bes_dims:
            states[dim] = detect_boundary_nodes(
                h=embedding[:, :dim],
                labels=labels,
                train_indices=train_indices,
                num_classes=self.num_classes,
                delta=self.delta,
                knn_k=self.boundary_knn,
                covariance_reg=self.covariance_reg,
                max_candidates=self.max_boundary_candidates,
            )
        return states

    def _sampled_gravity_objective(
        self,
        input_views: Tensor,
        labels: Tensor,
        states: Dict[int, BESBoundaryState],
    ) -> Tuple[Tensor, Dict[str, float]]:
        layer = self.boundary_attention.layers[self._active_attention_layer()]
        model_device = next(layer.parameters()).device
        total: Optional[Tensor] = None
        valid = 0
        details: Dict[str, float] = {}
        for dim in self.bes_dims:
            state = states[dim]
            nodes = state.nodes
            details[f"boundary_{dim}"] = float(nodes.numel())
            if nodes.numel() == 0:
                details[f"dim_{dim}"] = 0.0
                continue
            if nodes.numel() > self.beta_size:
                selected = torch.randperm(nodes.numel(), device=nodes.device)[: self.beta_size]
                nodes = nodes[selected]
            local_views = input_views[nodes.to(input_views.device)].to(model_device)
            output = layer(local_views).mean(dim=1)[:, :dim]
            local_labels = labels[nodes.to(labels.device)].to(model_device)
            loss = compute_gravity_loss(output, local_labels, state.centroids[:, :dim], self.tau)
            details[f"dim_{dim}"] = float(loss.detach().item())
            total = loss if total is None else total + loss
            valid += 1
        if total is None:
            total = next(layer.parameters()).sum() * 0.0
        elif valid:
            total = total / valid
        details["gravity_loss"] = float(total.detach().item())
        return total, details

    def _extra_shaping_objective(
        self,
        input_views: Tensor,
        train_indices: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        return input_views.sum() * 0.0, {}

    def _global_displacement(self, input_views: Tensor, eta: float) -> float:
        layer = self.boundary_attention.layers[self._active_attention_layer()]
        parameters = [p for p in layer.parameters() if p.requires_grad]
        originals = [p.detach().clone() for p in parameters]
        model_device = next(layer.parameters()).device
        total = 0.0
        with torch.no_grad():
            for start in range(0, input_views.size(0), self.displacement_batch_size):
                end = min(start + self.displacement_batch_size, input_views.size(0))
                batch = input_views[start:end].to(model_device)
                baseline = layer(batch).mean(dim=1)
                for parameter, original in zip(parameters, originals):
                    parameter.copy_(original)
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-eta)
                virtual = layer(batch).mean(dim=1)
                total += torch.norm(virtual - baseline, dim=1).sum().item()
                for parameter, original in zip(parameters, originals):
                    parameter.copy_(original)
        return total

    def _adaptive_backward(
        self,
        loss: Tensor,
        input_views: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        loss.backward()
        layer = self.boundary_attention.layers[self._active_attention_layer()]
        parameters = [p for p in layer.parameters() if p.requires_grad]
        eta = float(optimizer.param_groups[0]["lr"])
        displacement = self._global_displacement(input_views, eta)
        scale = self.alpha / max(displacement, 1e-8)
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)

    def _shaping_step(
        self,
        views: Optional[Tensor],
        labels: Tensor,
        train_indices: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        optimizer.zero_grad()
        active_layer = self._active_attention_layer()
        if self._shaping_cache_layer != active_layer:
            if views is None:
                raise RuntimeError("BES attention 层切换时缺少全局 backbone embeddings")
            self._shaping_cache_input = self._boundary_inputs(views)
            self._shaping_cache_states = self._build_boundary_states(
                self._shaping_cache_input, labels, train_indices
            )
            self._shaping_cache_layer = active_layer
        if self._shaping_cache_input is None or self._shaping_cache_states is None:
            raise RuntimeError("BES shaping cache 未初始化")
        input_views = self._shaping_cache_input
        states = self._shaping_cache_states
        gravity, details = self._sampled_gravity_objective(input_views, labels, states)
        extra, extra_details = self._extra_shaping_objective(input_views, train_indices)
        objective = gravity + extra
        if any(state.nodes.numel() > 0 for state in states.values()) or extra.detach().abs().item() > 0:
            self._adaptive_backward(objective, input_views, optimizer)

        # The official classifier consumes stop-gradient shaped embeddings.
        layer = self.boundary_attention.layers[self._active_attention_layer()]
        model_device = next(layer.parameters()).device
        if train_indices.numel() > self.displacement_batch_size:
            choice = torch.randperm(train_indices.numel(), device=train_indices.device)[: self.displacement_batch_size]
            cls_indices = train_indices[choice]
        else:
            cls_indices = train_indices
        cls_views = input_views[cls_indices.to(input_views.device)].to(model_device)
        cls_output = layer(cls_views).mean(dim=1).detach()
        cls_loss = F.cross_entropy(
            self.pretrain_classifier(cls_output),
            labels[cls_indices.to(labels.device)].to(model_device).long(),
        )
        cls_loss.backward()
        optimizer.step()
        details.update(extra_details)
        details["classifier_loss"] = float(cls_loss.detach().item())
        details["attention_layer"] = float(self._active_attention_layer() + 1)
        self.last_mrl_dim_losses = details
        return float(objective.detach().item())

    def ssl_train_step_full(
        self,
        data: Data,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.train()
        self._configure_phase()
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        labels = data.y.to(device)
        train_indices = data.train_idx.to(device)
        if self.epoch <= self.backbone_epochs:
            optimizer.zero_grad()
            views = self.backbone(x, edge_index)
            loss = self._backbone_pretrain_loss(views, labels, train_indices)
            loss.backward()
            optimizer.step()
            self.last_mrl_dim_losses = {"backbone_loss": float(loss.detach().item())}
            return float(loss.detach().item())

        self.backbone.eval()
        views = None
        if self._shaping_cache_layer != self._active_attention_layer():
            with torch.no_grad():
                views = self.backbone(x, edge_index)
        return self._shaping_step(views, labels, train_indices, optimizer)

    @torch.no_grad()
    def _infer_backbone_views_neighbor(
        self,
        data: Data,
        device: torch.device,
        num_neighbors: Sequence[int],
        batch_size: int,
    ) -> Tensor:
        self.backbone.eval()
        loader = NeighborLoader(
            data,
            input_nodes=None,
            num_neighbors=list(num_neighbors),
            batch_size=batch_size,
            shuffle=False,
        )
        outputs = torch.empty(
            (data.num_nodes, 2, self.hidden_dim), dtype=data.x.dtype, device="cpu"
        )
        offset = 0
        for batch in loader:
            batch = batch.to(device)
            count = int(batch.batch_size)
            outputs[offset : offset + count].copy_(
                self.backbone(batch.x, batch.edge_index)[:count].cpu()
            )
            offset += count
        if offset != data.num_nodes:
            raise RuntimeError(f"BES 全局 embedding 推理节点数不匹配: {offset} != {data.num_nodes}")
        return outputs

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
        self._configure_phase()
        if self.epoch <= self.backbone_epochs:
            loader = NeighborLoader(
                data,
                input_nodes=input_nodes,
                num_neighbors=list(num_neighbors),
                batch_size=batch_size,
                shuffle=True,
            )
            total, count = 0.0, 0
            for batch in loader:
                batch = batch.to(device)
                seeds = torch.arange(int(batch.batch_size), device=device)
                optimizer.zero_grad()
                views = self.backbone(batch.x, batch.edge_index)
                loss = self._backbone_pretrain_loss(views, batch.y, seeds)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().item()) * int(batch.batch_size)
                count += int(batch.batch_size)
            average = total / max(count, 1)
            self.last_mrl_dim_losses = {"backbone_loss": average}
            return average

        # Global frozen embeddings/statistics; only the GNN forward is sampled.
        labels = data.y.cpu()
        train_indices = data.train_idx.cpu()
        views = None
        if self._shaping_cache_layer != self._active_attention_layer():
            views = self._infer_backbone_views_neighbor(data, device, num_neighbors, batch_size)
        return self._shaping_step(views, labels, train_indices, optimizer)

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
        layers = self._trained_attention_layers()
        if mode == "full":
            views = self.backbone(data.x.to(device), data.edge_index.to(device))
            if layers == 0:
                return views.mean(dim=1).cpu()
            return self.boundary_attention(views, num_layers=layers).cpu()

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
            views = self.backbone(batch.x, batch.edge_index)[: int(batch.batch_size)]
            embedding = views.mean(dim=1) if layers == 0 else self.boundary_attention(views, num_layers=layers)
            outputs.append(embedding.cpu())
        return torch.cat(outputs, dim=0)

    def get_last_mrl_dim_losses(self) -> Dict[str, float]:
        return dict(self.last_mrl_dim_losses)
