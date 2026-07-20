import torch
from torch_geometric.data import Data

from unified_methods.graphcl_csne_method import GraphCLCSNEMethod
from unified_methods.graphcl_method import GRAPHCL_AUGMENTATIONS, GraphCLAugmentor, GraphCLMethod
from unified_methods.graphcl_mrl_method import GraphCLWithMRLMethod


def _data():
    generator = torch.Generator().manual_seed(11)
    nodes = 24
    source = torch.arange(nodes)
    target = (source + 1) % nodes
    edge_index = torch.stack((torch.cat((source, target)), torch.cat((target, source))))
    return Data(x=torch.randn(nodes, 6, generator=generator), edge_index=edge_index)


def _common():
    return dict(
        in_dim=6,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        proj_dim=16,
        tau=0.2,
        aug_1="edge_perturb",
        aug_2="attr_mask",
        aug_ratio_1=0.2,
        aug_ratio_2=0.2,
        symmetric_loss=False,
    )


def test_all_graphcl_augmentations_preserve_index_space():
    data = _data()
    for name in GRAPHCL_AUGMENTATIONS:
        torch.manual_seed(3)
        x, edge_index, anchors = GraphCLAugmentor(name, 0.2)(
            data.x, data.edge_index, anchor_size=12
        )
        assert x.shape == data.x.shape
        assert edge_index.dim() == 2 and edge_index.size(0) == 2
        assert anchors.shape == (12,) and anchors.dtype == torch.bool
        if edge_index.numel():
            assert int(edge_index.min()) >= 0
            assert int(edge_index.max()) < data.num_nodes


def test_graphcl_graphcl_mrl_and_graphcl_csne_train_one_step():
    data = _data()
    methods = [
        GraphCLMethod(**_common()),
        GraphCLWithMRLMethod(**_common(), mrl_dims=[4, 8, 16], mrl_weight=1.0),
        GraphCLCSNEMethod(
            **_common(),
            mrl_dims=[4, 8, 16],
            mrl_weight=1.0,
            ml_weight=1.0,
            warmup_epochs=0,
            full_epochs=1,
        ),
    ]
    for method in methods:
        method.epoch = 1
        optimizer = torch.optim.Adam(method.parameters(), lr=1e-3)
        loss = method.ssl_train_step_full(data, torch.device("cpu"), optimizer)
        assert torch.isfinite(torch.tensor(loss))


def test_graphcl_original_denominator_matches_reference_formula():
    torch.manual_seed(5)
    method = GraphCLMethod(**_common())
    z1 = torch.randn(7, 8)
    z2 = torch.randn(7, 8)
    stable = method.model.contrastive_loss(z1, z2)

    z1n = torch.nn.functional.normalize(z1, dim=-1)
    z2n = torch.nn.functional.normalize(z2, dim=-1)
    similarities = torch.exp((z1n @ z2n.T) / method.model.tau)
    positive = similarities.diag()
    reference = -torch.log(positive / (similarities.sum(dim=1) - positive)).mean()
    torch.testing.assert_close(stable, reference)


def test_graphcl_mrl_projector_preserves_prefix_dependencies():
    method = GraphCLWithMRLMethod(**_common(), mrl_dims=[4, 8, 16], mrl_weight=1.0)
    x = torch.randn(5, 16)
    changed = x.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])
    z = method.model.projector(x)
    z_changed = method.model.projector(changed)
    torch.testing.assert_close(z[:, :4], z_changed[:, :4])
