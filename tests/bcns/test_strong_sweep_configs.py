from scripts.bcns.run_bcns_step4_ablation import method_configs


def _names(configs):
    return [name for name, _, _, _ in configs]


def _params(configs):
    return {name: params for name, _, params, _ in configs}


def _meta(configs):
    return {name: meta for name, _, _, meta in configs}


def test_strong_ratio_sweep_contains_expected_ratios_and_sources():
    configs = method_configs("mcg_bcns_strong_ratio_sweep", scale_default=1.0)
    names = set(_names(configs))
    assert "mcg_fixed_s1.0" in names
    assert "harmonic_r0.1_clip1000_every2" in names
    assert "poisson_r3.0_clip1000_every2" in names
    params = _params(configs)
    ratios = {
        params[name]["bcns_target_update_ratio"]
        for name in names
        if name != "mcg_fixed_s1.0"
    }
    assert ratios == {0.1, 0.3, 1.0, 3.0}
    for name, values in params.items():
        if name == "mcg_fixed_s1.0":
            continue
        assert values["mcg_scale"] == 1.0
        assert values["bcns_ratio_control"] is True
        assert values["bcns_ratio_clip_max"] == 1000.0
        assert values["bcns_apply_every_n_steps"] == 2


def test_clip_sweep_contains_expected_clip_values():
    configs = method_configs("mcg_bcns_clip_sweep", scale_default=1.0)
    params = _params(configs)
    clips = {
        params[name]["bcns_ratio_clip_max"]
        for name in params
        if name != "mcg_fixed_s1.0"
    }
    assert clips == {100.0, 300.0, 1000.0, 3000.0, 10000.0}
    meta = _meta(configs)
    assert {row.get("sweep_family") for row in meta.values()} == {"clip", None}


def test_apply_every_sweep_contains_expected_frequencies():
    configs = method_configs("mcg_bcns_apply_every_sweep", scale_default=1.0)
    params = _params(configs)
    frequencies = {
        params[name]["bcns_apply_every_n_steps"]
        for name in params
        if name != "mcg_fixed_s1.0"
    }
    assert frequencies == {1, 2, 5, 10}
    assert "harmonic_r0.1_every10" in params
    assert "harmonic_r0.3_every1" in params
    assert "poisson_r0.3_every2" in params


def test_flow_evolution_sweep_contains_expected_pseudo_times():
    configs = method_configs("mcg_bcns_flow_evolution_sweep", scale_default=1.0)
    params = _params(configs)
    pseudo_times = {
        params[name]["bcns_target_builder_params"]["pseudo_time"]
        for name in params
        if name != "mcg_fixed_s1.0"
    }
    assert pseudo_times == {0.003, 0.01, 0.03, 0.1, 0.3}
    assert "flow_pt0p003_dt0p001_nu0p1_lift4_r0p1" in params
    assert "flow_pt0p3_dt0p001_nu0p2_lift16_r1" in params
    for name, values in params.items():
        if name == "mcg_fixed_s1.0":
            continue
        assert values["bcns_target_builder"] == "luminance_lift_flow"
        assert values["bcns_ratio_clip_max"] == 1000.0
        assert values["bcns_apply_every_n_steps"] == 2
