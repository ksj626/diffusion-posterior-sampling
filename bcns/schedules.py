"""Minimal schedules for BCNS target guidance."""

from dataclasses import dataclass


@dataclass
class BCNSSchedule:
    """Late-ramping scalar guidance schedule for future timestep-aware use."""

    gamma_max: float = 0.05
    tau2: float = 1.0
    pde_start_frac: float = 0.7
    ramp_power: float = 2.0

    def __post_init__(self) -> None:
        if self.gamma_max < 0:
            raise ValueError("gamma_max must be non-negative.")
        if self.tau2 <= 0:
            raise ValueError("tau2 must be positive.")
        if not 0.0 <= self.pde_start_frac <= 1.0:
            raise ValueError("pde_start_frac must be in [0, 1].")
        if self.ramp_power <= 0:
            raise ValueError("ramp_power must be positive.")

    def enabled(self, t_index: int, num_steps: int) -> bool:
        """Return whether late-trajectory guidance is enabled at reverse index."""

        self._validate_step(t_index, num_steps)
        progress = 1.0 - float(t_index) / float(max(num_steps - 1, 1))
        return progress >= self.pde_start_frac

    def gamma(self, t_index: int, num_steps: int) -> float:
        """Return late-ramped gamma from zero to ``gamma_max``."""

        self._validate_step(t_index, num_steps)
        if not self.enabled(t_index, num_steps):
            return 0.0
        if self.pde_start_frac >= 1.0:
            return self.gamma_max
        progress = 1.0 - float(t_index) / float(max(num_steps - 1, 1))
        local = (progress - self.pde_start_frac) / (1.0 - self.pde_start_frac)
        local = max(0.0, min(1.0, local))
        return self.gamma_max * (local ** self.ramp_power)

    def tau2_value(self, t_index: int, num_steps: int) -> float:
        """Return constant tau2 for this minimal skeleton."""

        self._validate_step(t_index, num_steps)
        return self.tau2

    @staticmethod
    def _validate_step(t_index: int, num_steps: int) -> None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if t_index < 0 or t_index >= num_steps:
            raise ValueError("t_index must satisfy 0 <= t_index < num_steps.")


def schedule_from_kwargs(**kwargs) -> BCNSSchedule:
    """Build a ``BCNSSchedule`` from keyword arguments."""

    allowed = {"gamma_max", "tau2", "pde_start_frac", "ramp_power"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(f"Unsupported schedule kwargs: {sorted(unknown)}.")
    return BCNSSchedule(**kwargs)
