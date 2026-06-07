from scripts.bcns.run_bcns_step4_ablation import (
    ALLOWED_SAMPLING_STEPS,
    dataset_index_for_sample,
    method_configs,
    should_visualize_sample,
    validate_sampling_steps,
)


def _names(ablation_set):
    return [item[0] for item in method_configs(ablation_set, scale_default=1.0)]


def _by_name(ablation_set):
    return {item[0]: item for item in method_configs(ablation_set, scale_default=1.0)}


def test_step_20_is_accepted():
    assert 20 in ALLOWED_SAMPLING_STEPS
    assert validate_sampling_steps(20) == 20


def test_main_method_ablation_set_names_and_best_configs():
    names = _names("bcns_main_methods")

    assert names == [
        "ps",
        "projection_fixed",
        "mcg_fixed_s1.0",
        "mcg_bcns_harmonic_best",
        "mcg_bcns_poisson_best",
    ]

    by_name = _by_name("bcns_main_methods")
    harmonic_params = by_name["mcg_bcns_harmonic_best"][2]
    poisson_params = by_name["mcg_bcns_poisson_best"][2]

    assert harmonic_params["bcns_target_builder"] == "luminance_lift_harmonic"
    assert harmonic_params["mcg_scale"] == 1.0
    assert harmonic_params["bcns_ratio_control"] is True
    assert harmonic_params["bcns_target_update_ratio"] == 0.3
    assert harmonic_params["bcns_ratio_clip_max"] == 10000.0
    assert harmonic_params["bcns_apply_every_n_steps"] == 2
    assert harmonic_params["bcns_target_builder_params"]["lift_scale"] == 1.0
    assert harmonic_params["apply_noisy_known_projection"] is True

    assert poisson_params["bcns_target_builder"] == "luminance_lift_poisson"
    assert poisson_params["mcg_scale"] == 1.0
    assert poisson_params["bcns_ratio_control"] is True
    assert poisson_params["bcns_target_update_ratio"] == 1.0
    assert poisson_params["bcns_ratio_clip_max"] == 1000.0
    assert poisson_params["bcns_apply_every_n_steps"] == 2
    assert poisson_params["bcns_target_builder_params"]["lift_scale"] == 4.0
    assert poisson_params["bcns_target_builder_params"]["rhs_mode"] == "projected_mu_laplacian"
    assert poisson_params["apply_noisy_known_projection"] is True


def test_harmonic_solver_ablation_set_names():
    names = set(_names("bcns_main_solver_ablation_harmonic"))

    assert {
        "mcg_fixed_s1.0",
        "mcg_bcns_harmonic_best_sor",
        "mcg_bcns_harmonic_best_cg",
        "mcg_bcns_harmonic_best_ftcs",
        "mcg_bcns_harmonic_best_be",
        "mcg_bcns_harmonic_best_cn",
    }.issubset(names)


def test_poisson_solver_ablation_set_names():
    names = set(_names("bcns_main_solver_ablation_poisson"))

    assert {
        "mcg_fixed_s1.0",
        "mcg_bcns_poisson_best_sor",
        "mcg_bcns_poisson_best_cg",
        "mcg_bcns_poisson_best_ftcs",
        "mcg_bcns_poisson_best_be",
        "mcg_bcns_poisson_best_cn",
    }.issubset(names)


def test_frequency_ablation_set_names():
    names = _names("bcns_main_frequency_ablation")

    assert names == [
        "mcg_fixed_s1.0",
        "mcg_bcns_harmonic_best",
        "mcg_bcns_harmonic_freq_best",
    ]


def test_dataset_cycling_and_visualization_cap_helpers():
    assert [dataset_index_for_sample(i, 3) for i in range(8)] == [0, 1, 2, 0, 1, 2, 0, 1]
    assert should_visualize_sample(9, 10) is True
    assert should_visualize_sample(10, 10) is False
    assert should_visualize_sample(63, -1) is True
