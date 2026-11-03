"""Experiment design: sample size, minimum detectable effect, duration.

The formulas are for a two-sided test of the difference in means with normal
approximation:  n_control = (z_{1-α/2} + z_{power})² · σ² · (1 + 1/k) / δ²,
where k = n_treatment / n_control. CUPED shrinks σ² by the factor (1 − ρ²).
"""
from __future__ import annotations

import math

from scipy import stats


def _z(alpha: float, power: float) -> float:
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    return stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)


def _resolve_mde(mde: float | None, mde_rel: float | None, baseline: float | None) -> float:
    if (mde is None) == (mde_rel is None):
        raise ValueError("give exactly one of mde (absolute) or mde_rel (relative)")
    if mde_rel is not None:
        if baseline is None:
            raise ValueError("mde_rel needs the baseline mean")
        mde = abs(baseline) * mde_rel
    if mde <= 0:
        raise ValueError("the effect to detect must be positive")
    return mde


def sample_size(std: float, mde: float | None = None, *, mde_rel: float | None = None,
                baseline: float | None = None, alpha: float = 0.05, power: float = 0.8,
                ratio: float = 1.0, variance_reduction: float = 0.0,
                std_treatment: float | None = None) -> dict:
    """Units needed per group to detect an effect of size `mde` (or `mde_rel` × baseline).

    variance_reduction: expected share of variance removed by CUPED (ρ² of the covariate).
    std_treatment: standard deviation expected in the treatment group. A multiplicative
        effect on a metric like revenue scales its spread too (std × (1 + lift)); assuming
        equal variances then overstates power. Default: equal to `std`.
    Returns {"control", "treatment", "total"}.
    """
    if std <= 0:
        raise ValueError("std must be positive")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if not 0 <= variance_reduction < 1:
        raise ValueError("variance_reduction must be in [0, 1)")
    delta = _resolve_mde(mde, mde_rel, baseline)
    std_t = std if std_treatment is None else std_treatment
    if std_t <= 0:
        raise ValueError("std_treatment must be positive")
    var = (std**2 + std_t**2 / ratio) * (1 - variance_reduction)
    n_c = math.ceil(_z(alpha, power) ** 2 * var / delta**2)
    n_t = math.ceil(n_c * ratio)
    return {"control": n_c, "treatment": n_t, "total": n_c + n_t}


def sample_size_proportion(baseline: float, mde: float | None = None, *,
                           mde_rel: float | None = None, alpha: float = 0.05,
                           power: float = 0.8, ratio: float = 1.0) -> dict:
    """Units per group to detect a change of a conversion rate from `baseline`."""
    if not 0 < baseline < 1:
        raise ValueError("baseline must be a proportion in (0, 1)")
    delta = _resolve_mde(mde, mde_rel, baseline)
    p2 = baseline + delta
    if not 0 < p2 < 1:
        raise ValueError("baseline + mde must stay within (0, 1)")
    var = baseline * (1 - baseline) + p2 * (1 - p2) / ratio
    n_c = math.ceil(_z(alpha, power) ** 2 * var / delta**2)
    n_t = math.ceil(n_c * ratio)
    return {"control": n_c, "treatment": n_t, "total": n_c + n_t}


def mde(std: float, n_control: int, *, n_treatment: int | None = None, alpha: float = 0.05,
        power: float = 0.8, variance_reduction: float = 0.0) -> float:
    """Smallest absolute effect detectable with the given group sizes."""
    if std <= 0 or n_control < 2:
        raise ValueError("need std > 0 and n_control >= 2")
    n_t = n_treatment or n_control
    se = std * math.sqrt((1 - variance_reduction) * (1 / n_control + 1 / n_t))
    return _z(alpha, power) * se


def mde_from_se(std_error: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """MDE of an experiment that has already run, from the standard error of its effect."""
    return _z(alpha, power) * std_error


def duration_days(total_units: int, daily_units: float, share_in_experiment: float = 1.0,
                  min_days: int = 7) -> int:
    """Days needed to collect `total_units`, at least `min_days` (a full weekly cycle)."""
    if daily_units <= 0 or not 0 < share_in_experiment <= 1:
        raise ValueError("daily_units must be positive and share in (0, 1]")
    return max(min_days, math.ceil(total_units / (daily_units * share_in_experiment)))
