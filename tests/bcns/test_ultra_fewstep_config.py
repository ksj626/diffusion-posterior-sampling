import pytest

from scripts.bcns.run_bcns_step4_ablation import (
    ALLOWED_SAMPLING_STEPS,
    method_configs,
    validate_sampling_steps,
)


def _names(configs):
    return [name for name, _, _, _ in configs]


def _params(configs):
    return {name: params for name, _, params, _ in configs}


def test_allowed_sampling_steps_include_ultra_fewstep_values():
    assert ALLOWED_SAMPLING_STEPS == (5, 10, 25, 50, 100, 250, 1000)
    assert validate_sampling_steps(5) == 5
    assert validate_sampling_steps(10) == 10
    assert validate_sampling_steps(25) == 25


def test_invalid_sampling_steps_still_fail():
    with pytest.raises(ValueError):
        validate_sampling_steps(7)


def test_ultra_fewstep_ablation_method_list_exists():
    configs = method_configs("mcg_bcns_ultra_fewstep", scale_default=1.0)
    assert _names(configs) == [
        "ps",
        "projection_fixed",
        "mcg_fixed_s1.0",
        "harmonic_r0.3_clip1000_every2",
        "poisson_r0.3_clip1000_every2",
        "flow_strong_r0.3_clip1000_every2",
    ]
    params = _params(configs)
    assert params["harmonic_r0.3_clip1000_every2"]["bcns_target_builder"] == "luminance_lift_harmonic"
    assert params["poisson_r0.3_clip1000_every2"]["bcns_target_builder"] == "luminance_lift_poisson"
    assert params["flow_strong_r0.3_clip1000_every2"]["bcns_target_builder"] == "luminance_lift_flow"
    assert params["flow_strong_r0.3_clip1000_every2"]["bcns_target_builder_params"]["pseudo_time"] == 0.03
