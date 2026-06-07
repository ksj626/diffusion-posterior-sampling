from scripts.bcns.run_bcns_step4_ablation import method_configs


def _names(configs):
    return [name for name, _, _, _ in configs]


def _params(configs):
    return {name: params for name, _, params, _ in configs}


def _meta(configs):
    return {name: meta for name, _, _, meta in configs}


def test_mcg_bcns_main_contains_expected_methods():
    configs = method_configs("mcg_bcns_main", scale_default=0.3)
    assert _names(configs) == [
        "ps",
        "projection_fixed",
        "mcg_fixed",
        "mcg_bcns_lift_harmonic",
        "mcg_bcns_lift_poisson",
        "mcg_bcns_lift_flow_strong",
    ]
    params = _params(configs)
    assert params["mcg_bcns_lift_harmonic"]["bcns_target_builder"] == "luminance_lift_harmonic"
    assert params["mcg_bcns_lift_poisson"]["bcns_target_builder"] == "luminance_lift_poisson"
    assert params["mcg_bcns_lift_flow_strong"]["bcns_target_builder"] == "luminance_lift_flow"
    assert params["mcg_bcns_lift_flow_strong"]["bcns_target_builder_params"]["pseudo_time"] == 0.03


def test_mcg_bcns_pde_strength_contains_lift_and_flow_variants():
    configs = method_configs("mcg_bcns_pde_strength", scale_default=0.3)
    names = set(_names(configs))
    assert "mcg_fixed" in names
    assert {"mcg_bcns_poisson_lift_s1", "mcg_bcns_poisson_lift_s2"}.issubset(names)
    assert {"mcg_bcns_poisson_lift_s4", "mcg_bcns_poisson_lift_s8"}.issubset(names)
    assert {"mcg_bcns_harmonic_lift_s1", "mcg_bcns_harmonic_lift_s2"}.issubset(names)
    assert {"mcg_bcns_harmonic_lift_s4", "mcg_bcns_flow_very_strong"}.issubset(names)
    meta = _meta(configs)
    assert meta["mcg_bcns_flow_weak"]["flow_pseudo_time"] == 0.003
    assert meta["mcg_bcns_flow_very_strong"]["flow_nu"] == 0.2
    assert meta["mcg_bcns_flow_very_strong"]["bcns_gamma_max"] == 1.0


def test_mcg_bcns_projection_effect_contains_projected_and_unprojected_variants():
    configs = method_configs("mcg_bcns_projection_effect", scale_default=0.3)
    names = set(_names(configs))
    assert names == {
        "mcg_fixed",
        "mcg_bcns_lift_poisson",
        "mcg_bcns_lift_poisson_unprojected",
        "mcg_bcns_lift_harmonic",
        "mcg_bcns_lift_harmonic_unprojected",
    }
    params = _params(configs)
    assert params["mcg_bcns_lift_poisson"]["apply_noisy_known_projection"] is True
    assert params["mcg_bcns_lift_poisson_unprojected"]["apply_noisy_known_projection"] is False
    assert params["mcg_bcns_lift_harmonic"]["apply_noisy_known_projection"] is True
    assert params["mcg_bcns_lift_harmonic_unprojected"]["apply_noisy_known_projection"] is False


def test_mcg_bcns_configs_expose_metric_metadata_fields():
    configs = method_configs("mcg_bcns_main", scale_default=0.3)
    meta = _meta(configs)
    row = meta["mcg_bcns_lift_poisson"]
    for key in (
        "mcg_scale",
        "bcns_gamma_max",
        "bcns_apply_every_n_steps",
        "bcns_target_builder",
        "lift_scale",
        "pde_source",
        "flow_pseudo_time",
        "flow_dt",
        "flow_nu",
    ):
        assert key in row
    assert row["method_family"] == "mcg_bcns"
    assert row["target_source"] == "poisson"
    assert row["transfer_mode"] == "luminance_lift"
