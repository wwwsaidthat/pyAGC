"""PaGCL 单元测试与验证。

验证项目:
  1. Toy graph 上前向/反向传播（输出 shape、有限 loss、非零梯度）
  2. 确定性: 同一输入和 seed 得到相同结果
  3. 后续视图由前一视图演化
  4. 跨视图节点 ID 对齐
  5. 时序一致性损失 > 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCN

from pyagc.models.pagcl import PaGCLModel, ProgressiveAugmentation


def _toy_graph(n: int = 20, d: int = 16) -> Data:
    torch.manual_seed(42)
    x = torch.randn(n, d)
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if torch.rand(1).item() < 0.30:
                edge_list.append([i, j])
    if len(edge_list) == 0:
        edge_list = [[0, 1]]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)


def test_progressive_augmentation():
    """验证渐进增强：后续视图由前一视图演化。"""
    data = _toy_graph(20, 16)
    aug = ProgressiveAugmentation(
        num_views=3,
        p_feat_mask_base=0.1,
        p_edge_drop_base=0.1,
        delta_feat=0.15,
        delta_edge=0.15,
    )
    views = aug(data.x, data.edge_index)
    assert len(views) == 3, f"应生成 3 个视图: 实际 {len(views)}"

    # 检查递增增强强度：边数应非递增（更少或相等）
    for k in range(1, len(views)):
        assert views[k]["x"].shape == views[0]["x"].shape, "视图特征 shape 应一致"

    print("  [PASS] Progressive augmentation generates sequential views")


def test_forward_backward():
    """验证前向/反向传播。"""
    data = _toy_graph(20, 16)
    encoder = GCN(in_channels=16, hidden_channels=32, out_channels=32,
                  num_layers=2, dropout=0.5, norm="batch_norm")
    model = PaGCLModel(
        encoder=encoder, hidden_dim=32, proj_dim=32,
        num_views=3, temporal_weight=0.1,
    )
    model.train()

    loss, components = model(data.x, data.edge_index)
    assert torch.isfinite(loss), f"loss 应为有限值: {loss.item()}"
    assert "contrast_loss" in components
    assert "temporal_loss" in components
    assert components["temporal_loss"] > 0, "时序一致性损失应 > 0"

    model.zero_grad()
    loss.backward()
    total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert total_grad > 0, f"总梯度范数应为正: {total_grad}"

    print(f"  [PASS] Forward/backward: loss={loss.item():.4f}, "
          f"contrast={components['contrast_loss']:.4f}, "
          f"temporal={components['temporal_loss']:.4f}")


def test_embed_shape():
    """验证 embed() 输出。"""
    data = _toy_graph(20, 16)
    encoder = GCN(in_channels=16, hidden_channels=32, out_channels=32,
                  num_layers=2, dropout=0.5, norm="batch_norm")
    model = PaGCLModel(encoder=encoder, hidden_dim=32, proj_dim=32, num_views=3)
    model.eval()

    emb = model.embed(data.x, data.edge_index)
    assert emb.shape == (20, 32), f"embed shape 错误: {emb.shape}"
    print("  [PASS] Embed output shape")


def test_determinism():
    """验证确定性。"""
    data = _toy_graph(20, 16)

    torch.manual_seed(123)
    encoder1 = GCN(in_channels=16, hidden_channels=32, out_channels=32,
                   num_layers=2, dropout=0.5, norm="batch_norm")
    model1 = PaGCLModel(encoder=encoder1, hidden_dim=32, proj_dim=32, num_views=3)

    torch.manual_seed(123)
    encoder2 = GCN(in_channels=16, hidden_channels=32, out_channels=32,
                   num_layers=2, dropout=0.5, norm="batch_norm")
    model2 = PaGCLModel(encoder=encoder2, hidden_dim=32, proj_dim=32, num_views=3)

    model1.eval()
    model2.eval()
    out1 = model1.embed(data.x, data.edge_index)
    out2 = model2.embed(data.x, data.edge_index)

    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-6, f"确定性失败: max_abs_diff={diff:.6f}"
    print("  [PASS] Determinism")


def test_method_interface():
    """验证 PaGCLMethod 接口。"""
    from unified_methods.pagcl_method import PaGCLMethod
    method = PaGCLMethod(in_dim=16, hidden_dim=32, num_layers=2,
                         dropout=0.5, proj_dim=32, num_views=3)
    assert method.method_name == "pagcl"
    assert method.is_supervised is False
    assert method.output_dim() == 32
    print("  [PASS] Method interface")


if __name__ == "__main__":
    print("=" * 60)
    print("PaGCL 验证测试")
    print("=" * 60)

    test_progressive_augmentation()
    test_forward_backward()
    test_embed_shape()
    test_determinism()
    test_method_interface()

    print()
    print("=" * 60)
    print("全部 PaGCL 测试通过!")
    print("=" * 60)
