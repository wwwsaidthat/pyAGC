"""HPEM (Hard-Pair Evolutionary Mining) 损失计算模块。

实现论文 MCNE 框架的 HPEM 机制，并支持 DALS 的维度自适应温度：

核心公式：
    τ_i = τ_0 · exp(φ_1 · d_i/d_K + φ_2)
    c_u^{(i-1)} = sim(h_u^{(i-1)}, h_{v^+}^{(i-1)}) - max_{v^-} sim(...)
    w_u^{(i)} = softmax(-β · c_u^{(i-1)} / τ_i)
    L_HPEM^i = Σ_u w_u^{(i)} · ℓ_InfoNCE(u; H^{(i)})

可学习参数：
    β:      confusion → weight 缩放系数（softplus 保证 > 0）
    φ_1, φ_2: 维度自适应 temperature 参数
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# 基础工具函数
# ============================================================================

def _rowwise_cosine_similarity(h1: Tensor, h2: Tensor, eps: float = 1e-8) -> Tensor:
    r"""计算两个视图之间的 B×B 余弦相似度矩阵。"""
    h1_norm = F.normalize(h1, p=2, dim=-1, eps=eps)
    h2_norm = F.normalize(h2, p=2, dim=-1, eps=eps)
    return h1_norm @ h2_norm.T


def _inv_softplus(x: float) -> float:
    """softplus 的近似逆，用于合理的参数初始化。"""
    if x > 20:
        return x
    return math.log(math.expm1(x))


# ============================================================================
# Step 1: 计算 confusion score
# ============================================================================

def compute_confusion_scores(
    h1_prev: Tensor,
    h2_prev: Tensor,
) -> Tensor:
    r"""用小维度 prefix 计算每个 anchor 的 confusion score。

    c_u = sim(h_u, h_{v^+}) - max_{v^-} sim(h_u, h_{v^-})

    c_u 越小（甚至为负），说明小维度 prefix 对这个 anchor 区分能力越弱。

    参数:
        h1_prev: (B, D_prev) 小维度 prefix 的第一视图编码器输出
        h2_prev: (B, D_prev) 小维度 prefix 的第二视图编码器输出

    返回:
        confusion: (B,) 每个 anchor 的 confusion score
    """
    S = _rowwise_cosine_similarity(h1_prev, h2_prev)  # (B, B)
    B = S.size(0)

    pos_sim = S.diag()  # (B,)

    S_no_diag = S.clone()
    S_no_diag[range(B), range(B)] = -float("inf")
    hardest_neg_sim = S_no_diag.max(dim=1).values  # (B,)

    confusion = pos_sim - hardest_neg_sim  # (B,)
    return confusion


# ============================================================================
# Step 2: confusion → per-anchor 权重
# ============================================================================

def confusion_to_weights(
    confusion: Tensor,
    tau: Tensor,
    beta: Tensor,
) -> Tensor:
    r"""将 confusion score 转化为 per-anchor 损失权重。

    w_u = softmax(-β · c_u / τ)

    c_u 越小（越困难）→ -β·c_u/τ 越大 → softmax 后权重越大。
    detach confusion 保证梯度不流回小维度 prefix。

    参数:
        confusion: (B,) confusion scores
        tau: 标量 tensor，当前维度的 temperature
        beta: 标量 tensor，可学习 scaling parameter

    返回:
        weights: (B,) 归一化后的 per-anchor 权重，sum = 1
    """
    logits = -beta * confusion.detach() / tau  # (B,)
    weights = F.softmax(logits, dim=0)  # (B,)
    return weights


# ============================================================================
# Step 3: per-anchor InfoNCE loss
# ============================================================================

def compute_per_anchor_infonce(
    h1: Tensor,
    h2: Tensor,
    tau: Tensor,
) -> Tensor:
    r"""计算每个 anchor 的 InfoNCE loss（不做 mean）。

    ℓ_u = -S[u,u]/τ + logsumexp_v(S[u,v]/τ)

    参数:
        h1, h2: (B, D) 编码器输出
        tau: 标量 tensor，temperature

    返回:
        per_anchor_loss: (B,)
    """
    S = _rowwise_cosine_similarity(h1, h2)  # (B, B)
    pos_sim = S.diag()  # (B,)
    log_sum_exp = torch.logsumexp(S / tau, dim=1)  # (B,)
    per_anchor_loss = -pos_sim / tau + log_sum_exp  # (B,)
    return per_anchor_loss


# ============================================================================
# Step 4: 完整的 HPEM loss
# ============================================================================

def compute_hpem_loss(
    h1_curr: Tensor,
    h2_curr: Tensor,
    h1_prev: Tensor,
    h2_prev: Tensor,
    tau: Tensor,
    beta: Tensor,
) -> Tensor:
    r"""计算单个维度 prefix i 的 HPEM 损失。

    L_HPEM^i = Σ_u w_u^{(i)} · ℓ_InfoNCE(u; H^{(i)})    （weighted mean 量级）

    参数:
        h1_curr, h2_curr: (B, D_i) 当前维度 prefix 的编码器输出
        h1_prev, h2_prev: (B, D_{i-1}) 上一维度 prefix 的编码器输出
        tau: 标量 tensor，当前维度的 temperature
        beta: 标量 tensor，可学习 scaling parameter

    返回:
        hpem_loss: 标量
    """

    confusion = compute_confusion_scores(h1_prev, h2_prev)  # (B,)
    weights = confusion_to_weights(confusion, tau, beta)  # (B,)
    per_anchor_loss = compute_per_anchor_infonce(h1_curr, h2_curr, tau)  # (B,)

    hpem_loss = (weights * per_anchor_loss).sum()
    return hpem_loss


# ============================================================================
# HPEMLoss 类（高级接口，支持维度自适应 temperature）
# ============================================================================

class HPEMLoss(nn.Module):
    r"""HPEM 损失计算器。

    可学习参数：
        β: confusion → weight 缩放系数（softplus 保证 > 0）
        φ_1, φ_2: τ_i = τ_0 · exp(φ_1 · d_i/d_K + φ_2)

    使用示例:
        hpem = HPEMLoss(tau_0=0.1, max_dim=768, beta_init=0.1)

        loss = hpem.compute_single(
            h1_curr=h1[:, :512], h2_curr=h2[:, :512],
            h1_prev=h1[:, :256], h2_prev=h2[:, :256],
            dim_curr=512,
        )

    仅提供 ``tau`` 时使用固定温度；MCNE 同时提供 ``tau_0`` 和
    ``max_dim`` 以启用维度自适应温度。
    """

    def __init__(
        self,
        tau: Optional[float] = None,
        beta_init: float = 0.1,
        *,
        tau_0: Optional[float] = None,
        max_dim: Optional[int] = None,
        phi_1_init: float = 0.0,
        phi_2_init: float = 0.0,
    ):
        """
        参数:
            tau: 固定温度的向后兼容参数
            tau_0: 自适应温度的基础值
            max_dim: 最大维度 d_K
            beta_init: β 的初始值
            phi_1_init, phi_2_init: φ_1、φ_2 的初始值
        """
        super().__init__()
        if tau_0 is None:
            tau_0 = 0.5 if tau is None else tau
        if tau_0 <= 0:
            raise ValueError("tau_0 must be > 0")
        if max_dim is not None and max_dim <= 0:
            raise ValueError("max_dim must be > 0")

        self.tau_0 = float(tau_0)
        # 保留旧属性，避免外部代码读取 hpem.tau 时失效。
        self.tau = self.tau_0
        self.max_dim = int(max_dim) if max_dim is not None else None

        # β：confusion → weight 缩放
        self._beta_raw = nn.Parameter(
            torch.tensor(_inv_softplus(beta_init))
        )

        # 仅 MCNE（提供 max_dim）启用维度自适应温度；旧调用保持固定温度，
        # 也不会引入无效的可学习参数。
        if self.max_dim is not None:
            self.phi_1 = nn.Parameter(torch.tensor(float(phi_1_init)))
            self.phi_2 = nn.Parameter(torch.tensor(float(phi_2_init)))
        else:
            self.register_parameter("phi_1", None)
            self.register_parameter("phi_2", None)

    @property
    def beta(self) -> Tensor:
        """保证 β 始终为正。"""
        return F.softplus(self._beta_raw)

    def get_tau(
        self,
        dim: Optional[int] = None,
        use_adaptive_temperature: bool = True,
    ) -> Tensor:
        r"""返回当前维度的温度。

        当启用 DALS 时：
            τ_i = τ_0 · exp(φ_1 · d_i/d_K + φ_2)

        当关闭 DALS 或该实例未配置 ``max_dim`` 时，返回固定的 τ_0。
        """
        reference = self._beta_raw
        tau_0 = reference.new_tensor(self.tau_0)
        if not use_adaptive_temperature or self.max_dim is None:
            return tau_0
        if dim is None:
            raise ValueError("adaptive temperature requires dim")
        if dim <= 0 or dim > self.max_dim:
            raise ValueError(f"dim must be in [1, {self.max_dim}], got {dim}")
        x = float(dim) / float(self.max_dim)
        return tau_0 * torch.exp(self.phi_1 * x + self.phi_2)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """兼容恢复自适应温度之前保存的 MCNE checkpoint。"""
        for name in ("phi_1", "phi_2"):
            parameter = getattr(self, name)
            key = prefix + name
            if parameter is not None and key not in state_dict:
                state_dict[key] = parameter.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # ------------------------------------------------------------------
    # 损失计算
    # ------------------------------------------------------------------

    def compute_single(
        self,
        h1_curr: Tensor,
        h2_curr: Tensor,
        h1_prev: Tensor,
        h2_prev: Tensor,
        dim_curr: Optional[int] = None,
        use_adaptive_temperature: bool = True,
    ) -> Tensor:
        r"""计算单个维度 prefix i 的 HPEM 损失。

        参数:
            h1_curr, h2_curr: 当前维度 prefix 的编码器输出
            h1_prev, h2_prev: 上一维度 prefix 的编码器输出
            dim_curr: 当前维度 d_i
            use_adaptive_temperature: 是否使用 DALS 的 τ_i
        返回:
            hpem_loss: 标量
        """
        tau = self.get_tau(
            dim=dim_curr,
            use_adaptive_temperature=use_adaptive_temperature,
        ).to(device=h1_curr.device, dtype=h1_curr.dtype)
        return compute_hpem_loss(
            h1_curr=h1_curr,
            h2_curr=h2_curr,
            h1_prev=h1_prev,
            h2_prev=h2_prev,
            tau=tau,
            beta=self.beta,
        )

    def compute_all(
        self,
        h1_full: Tensor,
        h2_full: Tensor,
        mrl_dims: Sequence[int],
    ) -> Tensor:
        r"""对所有相邻维度对计算 HPEM 损失总和。

        对每个 i > 0：用 dim[i-1] 做 confusion → 加权 dim[i] 的 InfoNCE。
        dim[0]（最小维度）不计算 HPEM。

        参数:
            h1_full, h2_full: (B, max_dim) 完整编码器输出
            mrl_dims: 维度列表，已排序，如 [64, 128, 256, 512]

        返回:
            total_hpem: 标量
        """
        sorted_dims = sorted(mrl_dims)
        if len(sorted_dims) < 2:
            return torch.zeros((), device=h1_full.device)

        total = torch.zeros((), device=h1_full.device)
        for i in range(1, len(sorted_dims)):
            dim_prev = sorted_dims[i - 1]
            dim_curr = sorted_dims[i]
            total = total + self.compute_single(
                h1_curr=h1_full[:, :dim_curr],
                h2_curr=h2_full[:, :dim_curr],
                h1_prev=h1_full[:, :dim_prev],
                h2_prev=h2_full[:, :dim_prev],
                dim_curr=dim_curr,
            )
        return total

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def log_info(self) -> dict:
        """返回当前可学习参数的快照，用于日志记录。"""
        with torch.no_grad():
            info = {"hpem_beta": float(self.beta.item())}
            if self.phi_1 is not None and self.phi_2 is not None:
                info.update(
                    {
                        "hpem_phi_1": float(self.phi_1.item()),
                        "hpem_phi_2": float(self.phi_2.item()),
                    }
                )
            return info
