import pytest

from bcns.schedules import BCNSSchedule
from scripts.bcns.run_bcns_step4_ablation import (
    create_sampler_for_sampling_steps,
    diffusion_config_for_sampling_steps,
    validate_sampling_steps,
)


def _diffusion_config():
    return {
        "sampler": "ddpm",
        "steps": 1000,
        "noise_schedule": "linear",
        "model_mean_type": "epsilon",
        "model_var_type": "learned_range",
        "dynamic_threshold": False,
        "clip_denoised": True,
        "rescale_timesteps": False,
        "timestep_respacing": "1000",
    }


def test_sampling_steps_override_builds_exact_respaced_sampler():
    sampler = create_sampler_for_sampling_steps(_diffusion_config(), 50)
    assert sampler.num_timesteps == 50
    assert len(sampler.timestep_map) == 50
    assert sampler.timestep_map[0] == 0
    assert sampler.timestep_map[-1] == 999


def test_sampling_steps_1000_preserves_full_reverse_update_count():
    sampler = create_sampler_for_sampling_steps(_diffusion_config(), 1000)
    assert sampler.num_timesteps == 1000
    assert len(sampler.timestep_map) == 1000


def test_sampling_steps_config_override_is_explicit():
    config = diffusion_config_for_sampling_steps(_diffusion_config(), 25)
    assert config["timestep_respacing"] == "25"


def test_sampling_steps_invalid_raises():
    with pytest.raises(ValueError):
        validate_sampling_steps(37)


def test_schedule_uses_actual_num_reverse_updates():
    schedule = BCNSSchedule(gamma_max=1.0, pde_start_frac=0.7, ramp_power=1.0)
    assert schedule.gamma(15, 50) == 0.0
    assert schedule.gamma(14, 50) > 0.0
