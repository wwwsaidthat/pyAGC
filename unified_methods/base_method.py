"""统一方法基类定义。"""

from abc import ABC
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


class BaseMethod(ABC, nn.Module):
    """所有方法统一基类。"""

    method_name: str
    is_supervised: bool

    def __init__(self, method_name: str, is_supervised: bool) -> None:
        super().__init__()
        self.method_name = method_name
        self.is_supervised = is_supervised

    def output_dim(self) -> int:
        """返回用于下游分类的特征维度。"""
        raise NotImplementedError

    # --- 监督训练接口 ---
    def supervised_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        raise NotImplementedError

    def supervised_train_step_neighbor(
        self,
        data: Data,
        train_idx: Tensor,
        num_neighbors: Sequence[int],
        batch_size: int,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        raise NotImplementedError

    @torch.no_grad()
    def supervised_predict(
        self,
        data: Data,
        mode: str,
        device: torch.device,
        eval_num_neighbors: Sequence[int],
        eval_batch_size: int,
    ) -> Tensor:
        raise NotImplementedError

    # --- 自监督训练接口 ---
    def ssl_train_step_full(self, data: Data, device: torch.device, optimizer: torch.optim.Optimizer) -> float:
        raise NotImplementedError

    def ssl_train_step_neighbor(
        self,
        data: Data,
        input_nodes: Optional[Tensor],
        num_neighbors: Sequence[int],
        batch_size: int,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        raise NotImplementedError

    @torch.no_grad()
    def infer_embeddings(
        self,
        data: Data,
        mode: str,
        device: torch.device,
        eval_num_neighbors: Sequence[int],
        eval_batch_size: int,
    ) -> Tensor:
        raise NotImplementedError
