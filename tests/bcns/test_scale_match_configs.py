from scripts.bcns.run_bcns_step4_ablation import method_configs


def _names(configs):
    return [name for name, _, _, _ in configs]


def _params(configs):
    return {name: params for name, _, params, _ in configs}


def _meta(configs):
    return {name: meta for name, _, _, meta in configs}


def test_scale_match_configs_include_explicit_scaled_mcg_baselines():
    configs = method_configs("mcg_bcns_scale_match", scale_default=1.0)
    names = _names(configs)
    assert "mcg_fixed_s0.3" in names
    assert "mcg_fixed_s1.0" in names
    params = _params(configs)
    assert params["mcg_fixed_s0.3"]["scale"] == 0.3
    assert params["mcg_fixed_s1.0"]["scale"] == 1.0


def test_scale_match_mcg_bcns_variants_use_matched_mcg_scale():
    configs = method_configs("mcg_bcns_scale_match", scale_default=1.0)
    params = _params(configs)
    assert params["mcg_bcns_lift_harmonic_m0.3"]["mcg_scale"] == 0.3
    assert params["mcg_bcns_lift_poisson_m0.3"]["mcg_scale"] == 0.3
    assert params["mcg_bcns_lift_harmonic_m1.0"]["mcg_scale"] == 1.0
    assert params["mcg_bcns_lift_poisson_m1.0"]["mcg_scale"] == 1.0


def test_scale_match_metadata_includes_mcg_scale():
    configs = method_configs("mcg_bcns_scale_match", scale_default=1.0)
    meta = _meta(configs)
    assert meta["mcg_fixed_s0.3"]["mcg_scale"] == 0.3
    assert meta["mcg_fixed_s1.0"]["mcg_scale"] == 1.0
    assert meta["mcg_bcns_lift_harmonic_m0.3"]["mcg_scale"] == 0.3
    assert meta["mcg_bcns_lift_poisson_m1.0"]["mcg_scale"] == 1.0


def test_ratio_sweep_configs_enable_ratio_control_and_expected_ratios():
    configs = method_configs("mcg_bcns_ratio_sweep", scale_default=1.0)
    names = set(_names(configs))
    assert "mcg_fixed_s1.0" in names
    assert "mcg_bcns_harmonic_r0.003" in names
    assert "mcg_bcns_poisson_r0.1" in names
    params = _params(configs)
    ratios = {
        params[name]["bcns_target_update_ratio"]
        for name in names
        if name.startswith("mcg_bcns_")
    }
    assert ratios == {0.003, 0.01, 0.03, 0.1}
    for name in names:
        if name.startswith("mcg_bcns_"):
            assert params[name]["mcg_scale"] == 1.0
            assert params[name]["bcns_ratio_control"] is True
            assert params[name]["bcns_ratio_clip_max"] == 100.0
