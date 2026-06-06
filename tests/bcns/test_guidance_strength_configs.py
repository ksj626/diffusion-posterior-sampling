from scripts.bcns.run_bcns_step4_ablation import method_configs


def _params_by_name(configs):
    return {name: params for name, _, params, _ in configs}


def test_guidance_strength_default_uses_paired_configs():
    configs = method_configs("guidance_strength", scale_default=0.3)
    names = [name for name, _, _, _ in configs]
    assert names == [
        "flow_struct_g0.02_w0.1",
        "flow_struct_g0.1_w1",
        "flow_struct_g0.5_w10",
        "flow_struct_g1.0_w50",
        "flow_struct_projected_g0.02_w0.1",
        "flow_struct_projected_g0.1_w1",
        "flow_struct_projected_g0.5_w10",
        "flow_struct_projected_g1.0_w50",
    ]
    params = _params_by_name(configs)
    assert params["flow_struct_g0.5_w10"]["gamma_max"] == 0.5
    assert params["flow_struct_g0.5_w10"]["structure_loss_weight"] == 10.0
    assert params["flow_struct_g0.5_w10"]["apply_noisy_known_projection"] is False
    assert params["flow_struct_projected_g1.0_w50"]["apply_noisy_known_projection"] is True
    assert params["flow_struct_projected_g1.0_w50"]["apply_every_n_steps"] == 5


def test_guidance_strength_full_grid_expands_all_pairs():
    configs = method_configs("guidance_strength", scale_default=0.3, full_strength_grid=True)
    assert len(configs) == 64
    names = {name for name, _, _, _ in configs}
    assert "flow_struct_g0.02_w0.1_every5" in names
    assert "flow_struct_projected_g1.0_w50_every10" in names


def test_luminance_lift_ablation_produces_lift_methods():
    configs = method_configs("luminance_lift", scale_default=0.3)
    params = _params_by_name(configs)
    assert "bcns_lift_flow_s0.5_projected" in params
    assert "bcns_lift_poisson_s1.0_projected" in params
    assert "bcns_lift_harmonic_s1.0_projected" in params
    assert params["bcns_lift_flow_s0.5_projected"]["target_builder"] == "luminance_lift_flow"
    assert params["bcns_lift_flow_s0.5_projected"]["target_builder_params"]["lift_scale"] == 0.5
    assert params["bcns_lift_poisson_s1.0_projected"]["target_builder"] == "luminance_lift_poisson"
    assert params["bcns_lift_harmonic_s1.0_projected"]["target_builder"] == "luminance_lift_harmonic"


def test_flow_strength_extended_uses_fixed_debug_strength():
    configs = method_configs("flow_strength_extended", scale_default=0.3)
    assert [name for name, _, _, _ in configs] == [
        "flow_weak",
        "flow_mid",
        "flow_strong",
        "flow_very_strong",
    ]
    for _, _, params, meta in configs:
        assert meta["sampling_steps"] == 100
        assert params["gamma_max"] == 0.5
        assert params["structure_loss_weight"] == 10.0
        assert params["apply_every_n_steps"] == 5
