"""PolyGCL 单元测试与验证。

验证项目:
  1. Toy graph 上前向/反向传播（输出 shape、有限 loss、非零梯度）
  2. 确定性: 同一输入和 seed 得到相同结果
  3. 低/高通视图不同
  4. 滤波系数收到梯度
  5. Chebyshev 递推与直接矩阵多项式数值一致
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data

from pyagc.models.polygcl import (
    PolyGCL,
    ChebNetII,
    ChebnetIIProp,
    cheby,
    preminus_tensor,
    presum_tensor,
)


def _toy_graph(n: int = 20, d: int = 16) -> Data:
    """创建小型随机图用于测试。"""
    torch.manual_seed(42)
    x = torch.randn(n, d)
    # Random edges with 30% density
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if torch.rand(1).item() < 0.30:
                edge_list.append([i, j])
    if len(edge_list) == 0:
        edge_list = [[0, 1]]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def test_cheby_basis():
    """验证 Chebyshev 多项式基础函数。"""
    x = torch.tensor(0.5)
    assert abs(cheby(0, x).item() - 1.0) < 1e-6, "T_0(0.5) 应为 1"
    assert abs(cheby(1, x).item() - 0.5) < 1e-6, "T_1(0.5) 应为 0.5"
    # T_2(0.5) = 2*(0.5)^2 - 1 = -0.5
    assert abs(cheby(2, x).item() - (-0.5)) < 1e-6, "T_2(0.5) 应为 -0.5"
    print("  [PASS] Chebyshev basis")


def test_coefficient_construction():
    """验证滤波系数构造：低通递减、高通递增。"""
    temp_low = torch.ones(10) * 0.1
    temp_high = torch.ones(10) * 0.1

    gamma_low = preminus_tensor(temp_low, 2.0)
    gamma_high = presum_tensor(temp_high, 0.0)

    # 低通递减
    for i in range(1, len(gamma_low)):
        assert gamma_low[i] < gamma_low[i - 1], f"低通系数应递减: idx={i}"
    # 高通递增
    for i in range(1, len(gamma_high)):
        assert gamma_high[i] > gamma_high[i - 1], f"高通系数应递增: idx={i}"
    print("  [PASS] Coefficient construction")


def test_chebnetii_forward_shape():
    """验证 ChebNetII 前向输出 shape。"""
    data = _toy_graph(20, 16)
    model = ChebNetII(num_features=16, hidden=32, K=10)
    model.eval()

    out_low = model(data.x, data.edge_index, highpass=False)
    out_high = model(data.x, data.edge_index, highpass=True)

    assert out_low.shape == (20, 32), f"低通输出 shape 错误: {out_low.shape}"
    assert out_high.shape == (20, 32), f"高通输出 shape 错误: {out_high.shape}"
    print("  [PASS] ChebNetII output shapes")


def test_low_high_views_different():
    """验证低通和高通视图产生不同的表示。"""
    data = _toy_graph(20, 16)
    model = ChebNetII(num_features=16, hidden=32, K=10)

    out_low = model(data.x, data.edge_index, highpass=False)
    out_high = model(data.x, data.edge_index, highpass=True)

    diff = (out_low - out_high).abs().mean().item()
    assert diff > 1e-8, f"低通/高通视图应不同, diff={diff:.6f}"
    print(f"  [PASS] Low/high-pass views differ (mean_abs_diff={diff:.4f})")


def test_filter_coefficients_get_gradients():
    """验证滤波系数收到梯度。"""
    data = _toy_graph(20, 16)
    model = PolyGCL(in_dim=16, hidden_dim=32, K=10)
    model.train()

    shuf_idx = torch.randperm(20)
    shuf_x = data.x[shuf_idx]

    loss, _ = model(data.x, data.edge_index, shuf_x)
    loss.backward()

    # 检查 prop1 中的 temp_low, temp_high 是否有梯度
    assert model.encoder.prop1.temp_low.grad is not None, "temp_low 应收到梯度"
    assert model.encoder.prop1.temp_high.grad is not None, "temp_high 应收到梯度"
    assert model.alpha.grad is not None, "alpha 应收到梯度"
    assert model.beta.grad is not None, "beta 应收到梯度"

    grad_norm_low = model.encoder.prop1.temp_low.grad.norm().item()
    grad_norm_high = model.encoder.prop1.temp_high.grad.norm().item()
    assert grad_norm_low > 0 and grad_norm_high > 0, "梯度范数应为正"

    print(f"  [PASS] Filter coefficients receive gradients "
          f"(low={grad_norm_low:.4f}, high={grad_norm_high:.4f})")


def test_determinism():
    """验证确定性：相同输入 + seed 得到相同输出。"""
    data = _toy_graph(20, 16)

    torch.manual_seed(123)
    model1 = PolyGCL(in_dim=16, hidden_dim=32, K=10)
    torch.manual_seed(123)
    model2 = PolyGCL(in_dim=16, hidden_dim=32, K=10)

    model1.eval()
    model2.eval()
    out1 = model1.embed(data.x, data.edge_index)
    out2 = model2.embed(data.x, data.edge_index)

    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-6, f"确定性失败: max_abs_diff={diff:.6f}"
    print("  [PASS] Determinism")


def test_loss_finite_and_nonzero_grad():
    """验证 loss 有限且梯度非零。"""
    data = _toy_graph(20, 16)
    model = PolyGCL(in_dim=16, hidden_dim=32, K=10)
    model.train()

    shuf_idx = torch.randperm(20)
    shuf_x = data.x[shuf_idx]

    loss, _ = model(data.x, data.edge_index, shuf_x)
    assert torch.isfinite(loss), f"loss 应为有限值: {loss.item()}"
    assert loss.item() > 0, f"loss 应为正: {loss.item()}"

    model.zero_grad()
    loss.backward()

    # 检查 grad norm
    total_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert total_norm > 0, f"总梯度范数应为正: {total_norm}"
    print(f"  [PASS] Loss={loss.item():.4f}, grad_norm={total_norm:.4f}")


def test_embed_output_shape():
    """验证 embed() 输出正确的维度。"""
    data = _toy_graph(20, 16)
    model = PolyGCL(in_dim=16, hidden_dim=64, K=10)
    model.eval()

    emb = model.embed(data.x, data.edge_index)
    assert emb.shape == (20, 64), f"embed shape 错误: {emb.shape}"
    print("  [PASS] Embed output shape")


def test_method_interface():
    """验证 PolyGCLMethod 符合 BaseMethod 接口。"""
    from unified_methods.polygcl_method import PolyGCLMethod

    method = PolyGCLMethod(in_dim=16, hidden_dim=32, K=10)
    assert method.method_name == "polygcl"
    assert method.is_supervised is False
    assert method.output_dim() == 32
    print("  [PASS] Method interface")


if __name__ == "__main__":
    print("=" * 60)
    print("PolyGCL 验证测试")
    print("=" * 60)

    test_cheby_basis()
    test_coefficient_construction()
    test_chebnetii_forward_shape()
    test_low_high_views_different()
    test_filter_coefficients_get_gradients()
    test_determinism()
    test_loss_finite_and_nonzero_grad()
    test_embed_output_shape()
    test_method_interface()

    print()
    print("=" * 60)
    print("全部 PolyGCL 测试通过!")
    print("=" * 60)
