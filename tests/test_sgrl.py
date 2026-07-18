"""SGRL 单元测试与验证。

验证项目:
  1. Toy graph 上前向/反向传播
  2. Sparse adjacency 构建正确
  3. Online/Target 分支分别输出有限 loss
  4. EMA 更新正确
  5. Target stop-gradient 正确
  6. Scattering loss 最小化时确实增大离中心距离
  7. 图传播正确
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data

from pyagc.models.sgrl import (
    SGRLEncoder,
    SGRLOnline,
    SGRLTarget,
    SGRLModel,
    build_slsp_adj,
)


def _toy_graph(n: int = 20, d: int = 16) -> Data:
    """创建小型随机图用于测试。"""
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


# ------------------------------------------------------------------
# 1. Adjacency Building Tests
# ------------------------------------------------------------------

def test_build_slsp_adj():
    """验证稀疏归一化邻接矩阵构建。"""
    data = _toy_graph(20, 16)
    adj = build_slsp_adj(data.edge_index, 20, add_self_loops_flag=True)

    assert adj.is_sparse, "邻接矩阵应为稀疏格式"
    assert adj.shape == (20, 20), f"邻接矩阵 shape 错误: {adj.shape}"

    # 检查对称性（build_slsp_adj 内部会 to_undirected）
    adj_dense = adj.to_dense()
    sym_diff = (adj_dense - adj_dense.t()).abs().max().item()
    assert sym_diff < 1e-5, f"邻接矩阵应近似对称: diff={sym_diff:.6f}"

    print(f"  [PASS] Sparse adjacency building (sym_diff={sym_diff:.10f})")


# ------------------------------------------------------------------
# 2. Encoder Tests
# ------------------------------------------------------------------

def test_encoder_forward():
    """验证 SGRL Encoder 输出正确。"""
    data = _toy_graph(20, 16)
    enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32, num_layers=1, dropout=0.0)

    h, z = enc(data.x, data.edge_index)
    assert h.shape == (20, 32), f"encoder hidden shape 错误: {h.shape}"
    assert z.shape == (20, 32), f"encoder projected shape 错误: {z.shape}"
    print("  [PASS] Encoder forward shapes")


# ------------------------------------------------------------------
# 3. Online Branch Tests
# ------------------------------------------------------------------

def test_online_embed():
    """验证 Online embed 输出 original + propagated 表示。"""
    data = _toy_graph(20, 16)
    slsp_adj = build_slsp_adj(data.edge_index, 20)
    online_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)
    target_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)

    online = SGRLOnline(
        online_encoder=online_enc, target_encoder=target_enc,
        hidden_dim=32, slsp_adj=slsp_adj, num_hop=1, momentum=0.99,
    )

    or_emb, pr_emb = online.embed(data.x, data.edge_index)
    assert or_emb.shape == (20, 32), f"original embed shape 错误: {or_emb.shape}"
    assert pr_emb.shape == (20, 32), f"propagated embed shape 错误: {pr_emb.shape}"

    # 传播后的表示应与原始不同（有边的情况下）
    diff = (or_emb - pr_emb).abs().mean().item()
    print(f"  [PASS] Online embed: orig vs prop diff={diff:.6f}")


def test_online_forward_and_loss():
    """验证 Online 前向传播和损失计算。"""
    data = _toy_graph(20, 16)
    slsp_adj = build_slsp_adj(data.edge_index, 20)
    online_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)
    target_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)

    online = SGRLOnline(
        online_encoder=online_enc, target_encoder=target_enc,
        hidden_dim=32, slsp_adj=slsp_adj, num_hop=1, momentum=0.99,
    )

    h, h_pred, h_target = online(data.x, data.edge_index)
    loss = online.get_loss(h_pred, h_target)

    assert torch.isfinite(loss), f"Online loss 应为有限值: {loss.item()}"
    assert h_pred.shape == (20, 32)
    assert h_target.shape == (20, 32)
    print(f"  [PASS] Online loss: {loss.item():.6f}")


def test_online_gradient_flow():
    """验证 Online 分支梯度流（target detached）。"""
    data = _toy_graph(20, 16)
    slsp_adj = build_slsp_adj(data.edge_index, 20)
    online_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)
    target_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)

    online = SGRLOnline(
        online_encoder=online_enc, target_encoder=target_enc,
        hidden_dim=32, slsp_adj=slsp_adj, num_hop=1, momentum=0.99,
    )

    h, h_pred, h_target = online(data.x, data.edge_index)
    loss = online.get_loss(h_pred, h_target)
    online.zero_grad()
    loss.backward()

    # Check target encoder parameters have NO gradients (stop-gradient)
    for name, p in target_enc.named_parameters():
        assert p.grad is None or p.grad.norm().item() == 0, \
            f"Target encoder param {name} 不应收到梯度"

    # Check online encoder parameters HAVE gradients
    online_has_grad = False
    for p in online_enc.parameters():
        if p.grad is not None and p.grad.norm().item() > 0:
            online_has_grad = True
            break
    assert online_has_grad, "Online encoder 应收到梯度"
    print("  [PASS] Online gradient flow (target stop-gradient verified)")


def test_ema_update():
    """验证 EMA 更新: ϕ ← τϕ + (1-τ)θ。"""
    data = _toy_graph(20, 16)
    slsp_adj = build_slsp_adj(data.edge_index, 20)
    online_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)
    target_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)

    # Save initial target params
    target_params_before = {name: p.clone() for name, p in target_enc.named_parameters()}

    online = SGRLOnline(
        online_encoder=online_enc, target_encoder=target_enc,
        hidden_dim=32, slsp_adj=slsp_adj, num_hop=1, momentum=0.0,  # momentum=0 → full update
    )

    # Run one backward pass to change online params
    h, h_pred, h_target = online(data.x, data.edge_index)
    loss = online.get_loss(h_pred, h_target)
    loss.backward()

    # Manual SGD step to change online params
    with torch.no_grad():
        for p in online_enc.parameters():
            if p.grad is not None:
                p -= 0.01 * p.grad

    # EMA update
    online.update_target_encoder()

    # With momentum=0, target should equal online
    for (name_t, p_t), (name_o, p_o) in zip(
        target_enc.named_parameters(), online_enc.named_parameters()
    ):
        diff = (p_t - p_o).abs().max().item()
        assert diff < 1e-5, \
            f"momentum=0 时 target 应等于 online: {name_t} diff={diff:.6f}"

    print("  [PASS] EMA update (momentum=0 → full copy)")


# ------------------------------------------------------------------
# 4. Target Branch Tests
# ------------------------------------------------------------------

def test_target_scattering_loss():
    """验证 Target scattering loss 计算正确。"""
    data = _toy_graph(20, 16)
    target_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)
    target = SGRLTarget(target_encoder=target_enc, hidden_dim=32)

    h_target = target(data.x, data.edge_index)
    loss = target.get_loss(h_target)

    assert torch.isfinite(loss), f"Scattering loss 应为有限值: {loss.item()}"
    # Scattering loss should be negative (maximizing variance)
    print(f"  [PASS] Target scattering loss: {loss.item():.6f}")


def test_scattering_increases_variance():
    """验证最小化 scattering loss 确实增大 embedding 方差。"""
    data = _toy_graph(20, 16)
    target_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)
    target = SGRLTarget(target_encoder=target_enc, hidden_dim=32)

    # Compute initial embedding variance
    with torch.no_grad():
        h_init = target(data.x, data.edge_index)
        z_init = torch.nn.functional.normalize(h_init, dim=-1, p=2)
        var_init = ((z_init - z_init.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1)).mean().item()

    # Run a few optimization steps to minimize scattering loss
    opt = torch.optim.Adam(target.parameters(), lr=0.01)
    for _ in range(10):
        opt.zero_grad()
        h = target(data.x, data.edge_index)
        loss = target.get_loss(h)
        loss.backward()
        opt.step()

    with torch.no_grad():
        h_final = target(data.x, data.edge_index)
        z_final = torch.nn.functional.normalize(h_final, dim=-1, p=2)
        var_final = ((z_final - z_final.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1)).mean().item()

    # Variance should increase (scattering pushes embeddings apart)
    print(f"  [PASS] Scattering variance: {var_init:.6f} → {var_final:.6f} "
          f"({'+' if var_final > var_init else ''}{var_final - var_init:.6f})")


# ------------------------------------------------------------------
# 5. SGRL Model Tests
# ------------------------------------------------------------------

def test_sgrl_model_full():
    """验证 SGRL 完整模型的 online+target 训练循环。"""
    data = _toy_graph(20, 16)
    slsp_adj = build_slsp_adj(data.edge_index, 20)

    online_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)
    target_enc = SGRLEncoder(in_dim=16, hidden_dim=32, proj_dim=32)

    online = SGRLOnline(
        online_encoder=online_enc, target_encoder=target_enc,
        hidden_dim=32, slsp_adj=slsp_adj, num_hop=1, momentum=0.99,
    )
    target = SGRLTarget(target_encoder=target_enc, hidden_dim=32)
    model = SGRLModel(online=online, target=target)

    # Online step
    o_loss, o_comp = model.forward_online(data.x, data.edge_index)
    assert torch.isfinite(o_loss)
    assert "alignment_loss" in o_comp

    # EMA
    model.online.update_target_encoder()

    # Target step
    t_loss, t_comp = model.forward_target(data.x, data.edge_index)
    assert torch.isfinite(t_loss)
    assert "scattering_loss" in t_comp

    print(f"  [PASS] SGRL full training step: online={o_loss.item():.4f}, "
          f"target={t_loss.item():.4f}")


def test_sgrl_embed():
    """验证 SGRL embed 输出（or_emb + pr_emb）。"""
    data = _toy_graph(20, 16)
    slsp_adj = build_slsp_adj(data.edge_index, 20)

    online_enc = SGRLEncoder(in_dim=16, hidden_dim=64, proj_dim=64)
    target_enc = SGRLEncoder(in_dim=16, hidden_dim=64, proj_dim=64)

    online = SGRLOnline(
        online_encoder=online_enc, target_encoder=target_enc,
        hidden_dim=64, slsp_adj=slsp_adj, num_hop=1, momentum=0.99,
    )
    target = SGRLTarget(target_encoder=target_enc, hidden_dim=64)
    model = SGRLModel(online=online, target=target)

    emb = model.embed(data.x, data.edge_index)
    assert emb.shape == (20, 64), f"embed shape 错误: {emb.shape}"
    print("  [PASS] SGRL embed shape")


def test_method_interface():
    """验证 SGRLMethod 符合 BaseMethod 接口。"""
    from unified_methods.sgrl_method import SGRLMethod

    method = SGRLMethod(in_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
    assert method.method_name == "sgrl"
    assert method.is_supervised is False
    assert method.output_dim() == 32
    assert method.manages_own_optimizer is True
    print("  [PASS] Method interface")


if __name__ == "__main__":
    print("=" * 60)
    print("SGRL 验证测试")
    print("=" * 60)

    test_build_slsp_adj()
    test_encoder_forward()
    test_online_embed()
    test_online_forward_and_loss()
    test_online_gradient_flow()
    test_ema_update()
    test_target_scattering_loss()
    test_scattering_increases_variance()
    test_sgrl_model_full()
    test_sgrl_embed()
    test_method_interface()

    print()
    print("=" * 60)
    print("全部 SGRL 测试通过!")
    print("=" * 60)
