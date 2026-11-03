"""Result objects shared by all methods."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class EffectResult:
    """Treatment vs control for one metric.

    effect      absolute difference treatment − control
    rel_effect  relative difference treatment / control − 1
    """

    metric: str
    method: str
    control_value: float
    treatment_value: float
    effect: float
    ci: tuple[float, float]
    rel_effect: float
    rel_ci: tuple[float, float]
    p_value: float
    n_control: int
    n_treatment: int
    alpha: float = 0.05
    std_error: float = math.nan
    p_value_adjusted: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        p = self.p_value if self.p_value_adjusted is None else self.p_value_adjusted
        return p < self.alpha

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return d

    def __str__(self) -> str:
        lo, hi = self.rel_ci
        p = self.p_value if self.p_value_adjusted is None else self.p_value_adjusted
        return (f"{self.metric} [{self.method}]: {self.control_value:.4g} → {self.treatment_value:.4g}, "
                f"{self.rel_effect:+.2%} (CI {lo:+.2%} … {hi:+.2%}), p = {p:.4g}")
