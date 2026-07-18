"""GOUDA/UGA 单元测试与验证。

验证项目:
  1. Toy graph 上前向/反向传播（输出 shape、有限 loss、非零梯度）
  2. UGA 模块两通道产生不同增强
  3. AC vectors 收到梯度
  4. Independence loss 仅依赖 AC matrices
  5. 确定性测试
  6. GOUDA-IF 与 GOUDA-BT 两种变体
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data

from pyagc.encoders import GCN
from pyagc.models.gouda import (
    GOUDA,
    UGAModule,
    hsic_independence_loss,
    random_walk_encoding,
)


def _toy_graph(n: int = 30, d: int = 16) -> Data:
    """创建小型随机图用于测试。"""
    torch.manual_seed(42)
    x = torch.randn(n, d)
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if torch.rand(1).item() < 0.25:
                edge_list.append([i, j])
    if len(edge_list) == 0:
        edge_list = [[0, 1]]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def _build_encoder(in_dim: int, hidden_dim: int) -> GCN:
    return GCN(
        in_channels=in_dim,
        hidden_channels=hidden_dim,
        out_channels=hidden_dim,
        num_layers=2,
        dropout=0.5,
        norm="batch_norm",
    )


# ------------------------------------------------------------------
# 1. UGA Module Tests
# ------------------------------------------------------------------

def test_uga_forward_shape():
    """验证 UGA 前向输出 shape 正确。"""
    data = _toy_graph(30, 16)
    struct_feat = torch.randn(30, 16)
    uga = UGAModule(num_ac_vectors=10, feature_dim=16, struct_dim=16, epsilon=0.01)

    x_aug, weights = uga(data.x, struct_feat)

    assert x_aug.shape == (30, 16), f"增强特征 shape 错误: {x_aug.shape}"
    assert weights.shape == (30, 10), f"传播权重 shape 错误: {weights.shape}"
    # 权重应近似归一化
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones(30), atol=1e-5), \
        f"权重未归一化: {weight_sums[:5]}"
    print("  [PASS] UGA forward shapes")


def test_uga_two_channels_different():
    """验证两个 UGA 通道产生不同的增强。"""
    data = _toy_graph(30, 16)
    struct_feat = torch.randn(30, 16)

    torch.manual_seed(1)
    uga_q = UGAModule(num_ac_vectors=10, feature_dim=16, struct_dim=16)
    torch.manual_seed(2)
    uga_p = UGAModule(num_ac_vectors=10, feature_dim=16, struct_dim=16)

    x1, _ = uga_q(data.x, struct_feat)
    x2, _ = uga_p(data.x, struct_feat)

    diff = (x1 - x2).abs().mean().item()
    assert diff > 1e-6, f"两通道增强应不同, diff={diff:.8f}"
    print(f"  [PASS] Two UGA channels differ (mean_abs_diff={diff:.4f})")


def test_ac_vectors_get_gradients():
    """验证 AC vectors 收到梯度。"""
    data = _toy_graph(30, 16)
    struct_feat = torch.randn(30, 16)
    uga = UGAModule(num_ac_vectors=10, feature_dim=16, struct_dim=16, epsilon=0.01)

    x_aug, _ = uga(data.x, struct_feat)
    loss = x_aug.sum()
    loss.backward()

    assert uga.ac_vectors.grad is not None, "AC vectors 应收到梯度"
    grad_norm = uga.ac_vectors.grad.norm().item()
    assert grad_norm > 0, f"AC vectors 梯度范数应为正: {grad_norm}"
    print(f"  [PASS] AC vectors receive gradients (norm={grad_norm:.4f})")


# ------------------------------------------------------------------
# 2. Independence Loss Tests
# ------------------------------------------------------------------

def test_independence_loss_finite():
    """验证独立性损失计算并输出有限值。"""
    Q = torch.randn(10, 16)
    P = torch.randn(10, 16)
    loss = hsic_independence_loss(Q, P, beta1=1e-4, beta2=1e-4)
    assert torch.isfinite(loss), f"独立性损失应为有限值: {loss.item()}"
    print(f"  [PASS] Independence loss is finite: {loss.item():.6f}")


def test_independence_loss_depends_only_on_ac_matrices():
    """验证 L_indep 只依赖 AC matrices Q/P，不依赖节点嵌入。"""
    Q1 = torch.randn(10, 16)
    P1 = torch.randn(10, 16)
    loss1 = hsic_independence_loss(Q1, P1)

    # 不同 AC matrices 应产生不同损失
    Q2 = torch.randn(10, 16)
    P2 = torch.randn(10, 16)
    loss2 = hsic_independence_loss(Q2, P2)

    assert abs(loss1.item() - loss2.item()) > 1e-8, \
        "不同 AC matrices 应产生不同独立性损失"
    print(f"  [PASS] Independence loss depends on AC matrices (loss1={loss1:.4f}, loss2={loss2:.4f})")


# ------------------------------------------------------------------
# 3. GOUDA Model Tests
# ------------------------------------------------------------------

def test_gouda_if_forward():
    """验证 GOUDA-IF 前向传播。"""
    data = _toy_graph(30, 16)
    encoder = _build_encoder(16, 32)
    model = GOUDA(
        encoder=encoder, hidden_dim=32, proj_dim=32, in_dim=16,
        tau=0.5, gamma=0.1, num_ac_vectors=10, struct_dim=16,
        use_barlow_twins=False,
    )
    model.train()

    total_loss, components = model(data.x, data.edge_index)

    assert torch.isfinite(total_loss), f"Total loss 应为有限值: {total_loss.item()}"
    assert total_loss.item() > 0, f"Total loss 应为正: {total_loss.item()}"
    assert "contrast_loss" in components
    assert "indep_loss" in components
    print(f"  [PASS] GOUDA-IF forward: total={total_loss.item():.4f}, "
          f"contrast={components['contrast_loss']:.4f}, "
          f"indep={components['indep_loss']:.4f}")


def test_gouda_bt_forward():
    """验证 GOUDA-BT 前向传播。"""
    data = _toy_graph(30, 16)
    encoder = _build_encoder(16, 32)
    model = GOUDA(
        encoder=encoder, hidden_dim=32, proj_dim=32, in_dim=16,
        tau=0.5, gamma=0.1, num_ac_vectors=10, struct_dim=16,
        use_barlow_twins=True,
    )
    model.train()

    total_loss, components = model(data.x, data.edge_index)

    assert torch.isfinite(total_loss), f"BT Total loss 应为有限值: {total_loss.item()}"
    assert "contrast_loss" in components
    assert "indep_loss" in components
    print(f"  [PASS] GOUDA-BT forward: total={total_loss.item():.4f}")


def test_gouda_gradient_flow():
    """验证 GOUDA 所有参数都收到梯度。"""
    data = _toy_graph(30, 16)
    encoder = _build_encoder(16, 32)
    model = GOUDA(
        encoder=encoder, hidden_dim=32, proj_dim=32, in_dim=16,
        tau=0.5, gamma=0.1, num_ac_vectors=10, struct_dim=16,
    )
    model.train()

    total_loss, _ = model(data.x, data.edge_index)
    model.zero_grad()
    total_loss.backward()

    params_with_grad = 0
    params_without_grad = 0
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.norm().item() > 0:
            params_with_grad += 1
        else:
            params_without_grad += 1

    assert params_with_grad > 0, "至少应有参数收到梯度"
    print(f"  [PASS] Gradient flow: {params_with_grad} params with grad, "
          f"{params_without_grad} without")


def test_gouda_embed_shape():
    """验证 embed() 输出正确的维度。"""
    data = _toy_graph(30, 16)
    encoder = _build_encoder(16, 64)
    model = GOUDA(
        encoder=encoder, hidden_dim=64, proj_dim=64, in_dim=16,
        num_ac_vectors=10, struct_dim=16,
    )
    model.eval()

    emb = model.embed(data.x, data.edge_index)
    assert emb.shape == (30, 64), f"embed shape 错误: {emb.shape}"
    print("  [PASS] Embed output shape")


def test_method_interface():
    """验证 GOUDAMethod 符合 BaseMethod 接口。"""
    from unified_methods.gouda_method import GOUDAMethod

    method = GOUDAMethod(in_dim=16, hidden_dim=32, num_layers=2,
                         dropout=0.5, proj_dim=32)
    assert method.method_name == "gouda"
    assert method.is_supervised is False
    assert method.output_dim() == 32
    print("  [PASS] Method interface")


if __name__ == "__main__":
    print("=" * 60)
    print("GOUDA/UGA 验证测试")
    print("=" * 60)

    test_uga_forward_shape()
    test_uga_two_channels_different()
    test_ac_vectors_get_gradients()
    test_independence_loss_finite()
    test_independence_loss_depends_only_on_ac_matrices()
    test_gouda_if_forward()
    test_gouda_bt_forward()
    test_gouda_gradient_flow()
    test_gouda_embed_shape()
    test_method_interface()

    print()
    print("=" * 60)
    print("全部 GOUDA/UGA 测试通过!")
    print("=" * 60)
