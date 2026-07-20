import torch

from unified_methods.csne_method import CSNEMethod


def _method(**overrides):
    kwargs = dict(
        in_dim=6,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        proj_dim=16,
        tau=0.5,
        p_feat_mask_1=0.0,
        p_edge_drop_1=0.0,
        p_feat_mask_2=0.0,
        p_edge_drop_2=0.0,
        mrl_dims=[4, 8, 16],
        mrl_weight=1.0,
        ml_weight=1.0,
        warmup_epochs=1,
        full_epochs=1,
    )
    kwargs.update(overrides)
    model = CSNEMethod(**kwargs)
    model.epoch = 2
    return model


def _views():
    generator = torch.Generator().manual_seed(7)
    z1 = torch.randn(8, 16, generator=generator, requires_grad=True)
    z2 = torch.randn(8, 16, generator=generator, requires_grad=True)
    return z1, z2


def test_all_ablation_combinations_are_finite_and_backwardable():
    for use_cdmd in (False, True):
        for use_hpem in (False, True):
            for use_das in (False, True):
                model = _method(
                    use_cdmd=use_cdmd,
                    use_hpem=use_hpem,
                    use_das=use_das,
                )
                z1, z2 = _views()
                loss, detail = model._grace_prefix_loss_with_details(z1, z2)
                assert torch.isfinite(loss)
                loss.backward()
                assert z1.grad is not None and torch.isfinite(z1.grad).all()
                assert detail["cdmd_enabled"] == float(use_cdmd)
                assert detail["hpem_enabled"] == float(use_hpem)
                assert detail["das_enabled"] == float(use_das and use_hpem)


def test_disabling_all_three_recovers_grace_mrl_after_warmup():
    model = _method(use_cdmd=False, use_hpem=False, use_das=False)
    z1, z2 = _views()
    actual, detail = model._grace_prefix_loss_with_details(z1, z2)
    expected = torch.stack(
        [model.model.nt_xent(z1[:, :d], z2[:, :d], model.model.tau) for d in model.mrl_dims]
    ).mean()
    torch.testing.assert_close(actual, expected)
    assert detail["hpem_loss"] == 0.0
    torch.testing.assert_close(torch.tensor(detail["grace_mrl_loss"]), expected.detach())


def test_registered_parameters_are_not_duplicated():
    model = _method()
    params = list(model.parameters())
    assert len(params) == len({id(parameter) for parameter in params})
