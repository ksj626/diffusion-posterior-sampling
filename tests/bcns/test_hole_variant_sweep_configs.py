from scripts.bcns.run_bcns_step4_ablation import DEFAULT_HOLE_VARIANTS, method_configs
from scripts.bcns.run_hole_variant_sweep import parse_hole_variants


def _names(configs):
    return [name for name, _, _, _ in configs]


def test_default_hole_variants_include_expected_structured_masks():
    assert set(DEFAULT_HOLE_VARIANTS) == {
        "thin_scratch",
        "thick_scratch_12",
        "thick_scratch_24",
        "text_mask",
        "freeform_medium",
        "center_box_96",
        "center_box_128",
    }


def test_hole_variant_sweep_method_list_is_compact():
    configs = method_configs("hole_variant_sweep", scale_default=1.0)
    assert _names(configs) == [
        "ps",
        "projection_fixed",
        "mcg_fixed_s1.0",
        "mcg_bcns_harmonic_best",
        "mcg_bcns_poisson_best",
    ]


def test_hole_variant_sweep_uses_ratio_control_best_configs():
    params = {name: params for name, _, params, _ in method_configs("hole_variant_sweep", scale_default=1.0)}
    for name in ("mcg_bcns_harmonic_best", "mcg_bcns_poisson_best"):
        assert params[name]["mcg_scale"] == 1.0
        assert params[name]["bcns_ratio_control"] is True
        assert params[name]["bcns_target_update_ratio"] == 0.03
        assert params[name]["apply_noisy_known_projection"] is True


def test_parse_hole_variants_defaults_and_custom_subset():
    assert parse_hole_variants("") == list(DEFAULT_HOLE_VARIANTS)
    assert parse_hole_variants("thin_scratch,text_mask") == ["thin_scratch", "text_mask"]
