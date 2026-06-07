from scripts.bcns.run_bcns_step4_ablation import method_configs


def _names(configs):
    return [name for name, _, _, _ in configs]


def _params_by_name(configs):
    return {name: params for name, _, params, _ in configs}


def _meta_by_name(configs):
    return {name: meta for name, _, _, meta in configs}


def test_lift_candidates_produces_expected_method_names():
    configs = method_configs("lift_candidates", scale_default=0.3)
    assert _names(configs) == [
        "ps",
        "projection_fixed",
        "mcg_fixed",
        "bcns_lift_harmonic_s0.5",
        "bcns_lift_harmonic_s1.0",
        "bcns_lift_harmonic_s2.0",
        "bcns_lift_poisson_s0.5",
        "bcns_lift_poisson_s1.0",
        "bcns_lift_poisson_s2.0",
        "bcns_lift_harmonic_s1.0_projected",
        "bcns_lift_poisson_s1.0_projected",
    ]
    params = _params_by_name(configs)
    assert params["bcns_lift_harmonic_s1.0"]["target_builder"] == "luminance_lift_harmonic"
    assert params["bcns_lift_poisson_s1.0"]["target_builder"] == "luminance_lift_poisson"


def test_lift_scale_ablation_has_expected_scales_and_projection_variants():
    configs = method_configs("lift_scale", scale_default=0.3)
    meta = _meta_by_name(configs)
    scales = {entry["lift_scale"] for entry in meta.values()}
    projected = {entry["projected"] for entry in meta.values()}
    sources = {entry["target_source"] for entry in meta.values()}
    assert scales == {0.25, 0.5, 1.0, 2.0, 4.0}
    assert projected == {False, True}
    assert sources == {"harmonic", "poisson"}
    assert len(configs) == 20


def test_lift_fewstep_sweeps_sampling_steps_without_model():
    configs = method_configs("lift_fewstep", scale_default=0.3)
    steps = {meta["sampling_steps"] for _, _, _, meta in configs}
    names = set(_names(configs))
    assert steps == {5, 10, 25, 50, 100, 250, 1000}
    assert "bcns_lift_harmonic_s1.0" in names
    assert "bcns_lift_poisson_s1.0_projected" in names
    assert len(configs) == 49
