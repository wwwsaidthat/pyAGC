"""BES 方法类。

BES: Bootstrapped Embedding Selection.
ICML 2026.

方法主线:
  - Learnable GCN encoder 替代原始的双冻结编码器 + 自注意力
  - Boundary Detection: 用 Mahalanobis slab 距离检测决策边界附近的节点
  - Gravitational Repulsion: InfoNCE 风格的排斥损失将边界节点推离其他类中心
  - Virtual Step + Adaptive Gradient Scaling: 防止过冲的梯度缩放

大图适配:
  - NeighborLoader 进行 mini-batch 训练
  - 边界检测和排斥损失在每个 batch 的 seed nodes 上进行
  - 类别中心/协方差在 batch 内近似计算（与 SGRL 的 scattering center 类似）
"""

from typing import List, Optional, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.models.bes import BESEncoder, detect_boundary_nodes, compute_repulsion_loss

from .base_method import BaseMethod


class BESMethod(BaseMethod):
    """BES 自监督方法。

    参数:
        in_dim: 输入特征维度
        hidden_dim: 编码器隐藏/输出维度 (32, 64, 128, 256, 384, 512, 768)
        num_layers: GCN 层数
        dropout: Dropout 比率
        tau: 排斥损失温度 (默认 1.0)
        delta: Mahalanobis 边界检测阈值 (默认 5.0)
        alpha: Virtual step 缩放因子 (默认 1.0)
        num_classes: 数据集类别数
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        tau: float = 1.0,
        delta: float = 5.0,
        alpha: float = 1.0,
        num_classes: int = 1,
    ) -> None:
        super().__init__(method_name="bes", is_supervised=False)

        self.encoder = BESEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.delta = delta
        self.alpha = alpha
        self.num_classes = num_classes

    def output_dim(self) -> int:
        return self.hidden_dim

    # ------------------------------------------------------------------
    # 辅助: 从 index tensor 构造 boolean mask
    # ------------------------------------------------------------------

    @staticmethod
    def _index_to_mask(indices: Tensor, num_nodes: int, device: torch.device) -> Tensor:
        """将索引张量转换为 boolean mask。"""
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        mask[indices.to(device)] = True
        return mask

    # ------------------------------------------------------------------
    # 自监督训练 — 全图模式
    # ------------------------------------------------------------------

    def ssl_train_step_full(
        self,
        data: Data,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """全图训练步骤。

        流程:
          1. encoder forward → h
          2. 边界检测 (Mahalanobis slab)
          3. 排斥损失 + virtual step + 梯度缩放
          4. optimizer.step()

        Returns:
            Scalar loss value.
        """
        self.train()

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        labels = data.y.to(device)
        train_mask = self._index_to_mask(data.train_idx, data.num_nodes, device)

        optimizer.zero_grad()

        # 1. Forward
        h = self.encoder(x, edge_index)

        # 2. Boundary detection
        boundary_indices, class_centroids = detect_boundary_nodes(
            h=h,
            labels=labels,
            train_mask=train_mask,
            num_classes=self.num_classes,
            delta=self.delta,
        )

        if boundary_indices.numel() < 2:
            # No boundary nodes to repel — nothing to learn this epoch
            return 0.0

        # 3. Repulsion loss
        loss = compute_repulsion_loss(
            h=h,
            boundary_indices=boundary_indices,
            labels=labels,
            class_centroids=class_centroids,
            num_classes=self.num_classes,
            tau=self.tau,
        )

        if loss.item() == 0.0:
            return 0.0

        loss.backward()

        # 4. Virtual step + adaptive gradient scaling
        lr = optimizer.param_groups[0]["lr"]
        h_detached = h.detach()

        # Save original params and apply virtual step
        original_params = []
        for p in self.encoder.parameters():
            original_params.append(p.clone().detach())
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-lr * self.alpha)

        # Forward with perturbed params (eval mode for deterministic output)
        self.encoder.eval()
        with torch.no_grad():
            Z_virtual = self.encoder(x, edge_index)
        self.encoder.train()

        # Compute displacement magnitude
        delta_B = torch.norm(Z_virtual - h_detached, dim=1).sum().item()

        # Restore original params
        for p, orig_p in zip(self.encoder.parameters(), original_params):
            p.data.copy_(orig_p)

        # Scale gradients (clamp to prevent explosion)
        raw_scale = self.alpha / (delta_B + 1e-8)
        scale_factor = min(raw_scale, 10.0)

        for p in self.encoder.parameters():
            if p.grad is not None:
                p.grad.mul_(scale_factor)

        optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # 自监督训练 — Neighbor 采样模式
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
        """Mini-batch 训练（大图适配）。

        每个 batch:
          1. NeighborLoader 采样子图
          2. encoder forward 子图 → seed node embeddings
          3. 在 seed nodes 上做边界检测 (batch 内近似)
          4. 排斥损失 + virtual step (per-batch)
          5. optimizer.step()

        Returns:
            Weighted average loss across batches.
        """
        self.train()

        loader = NeighborLoader(
            data,
            input_nodes=input_nodes,
            num_neighbors=list(num_neighbors),
            batch_size=batch_size,
            shuffle=True,
        )

        total_loss = 0.0
        total_nodes = 0
        lr = optimizer.param_groups[0]["lr"]

        for batch in loader:
            batch = batch.to(device)
            n_seed = int(batch.batch_size)

            # Seed nodes are the first n_seed nodes in the batch
            seed_mask = torch.zeros(batch.x.size(0), dtype=torch.bool, device=device)
            seed_mask[:n_seed] = True

            optimizer.zero_grad()

            # 1. Forward on subgraph
            h = self.encoder(batch.x, batch.edge_index)

            # 2. Boundary detection on seed nodes only
            boundary_indices, class_centroids = detect_boundary_nodes(
                h=h,
                labels=batch.y,
                train_mask=seed_mask,
                num_classes=self.num_classes,
                delta=self.delta,
            )

            if boundary_indices.numel() < 2:
                # No active boundary nodes in this batch — skip
                total_nodes += n_seed
                continue

            # 3. Repulsion loss
            loss = compute_repulsion_loss(
                h=h,
                boundary_indices=boundary_indices,
                labels=batch.y,
                class_centroids=class_centroids,
                num_classes=self.num_classes,
                tau=self.tau,
            )

            if loss.item() == 0.0:
                total_nodes += n_seed
                continue

            loss.backward()

            # 4. Virtual step (per-batch)
            h_detached = h.detach()

            # Save original params
            original_params = []
            for p in self.encoder.parameters():
                original_params.append(p.clone().detach())
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-lr * self.alpha)

            # Forward with perturbed params
            self.encoder.eval()
            with torch.no_grad():
                Z_virtual = self.encoder(batch.x, batch.edge_index)
            self.encoder.train()

            # Displacement on seed nodes
            delta_B = torch.norm(
                Z_virtual[:n_seed] - h_detached[:n_seed], dim=1
            ).sum().item()

            # Restore original params
            for p, orig_p in zip(self.encoder.parameters(), original_params):
                p.data.copy_(orig_p)

            # Scale gradients
            raw_scale = self.alpha / (delta_B + 1e-8)
            scale_factor = min(raw_scale, 10.0)

            for p in self.encoder.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale_factor)

            optimizer.step()

            total_loss += float(loss.item()) * n_seed
            total_nodes += n_seed

        return total_loss / max(total_nodes, 1)

    # ------------------------------------------------------------------
    # 推理
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
        """提取节点嵌入用于下游线性评估。

        Args:
            data: 图数据。
            mode: "full" 或 "neighbor"。
            device: 推理设备。
            eval_num_neighbors: 评估时的邻居采样数。
            eval_batch_size: 评估时的 batch size。

        Returns:
            [N, hidden_dim] 节点嵌入。
        """
        self.eval()

        if mode == "full":
            return self.encoder(
                data.x.to(device), data.edge_index.to(device)
            ).cpu()

        # Neighbor mode: batched inference over all nodes
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
            h = self.encoder(batch.x, batch.edge_index)
            # Take only the seed node embeddings
            out.append(h[: batch.batch_size].cpu())

        return torch.cat(out, dim=0)
