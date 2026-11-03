"""Sequential testing: look at the results every day without inflating false positives.

Checking an ordinary p-value every day and stopping at the first p < 0.05 raises the
false positive rate from 5% to 20–30%. The mixture sequential probability ratio test
(mSPRT, Johari et al., 2017, "Peeking at A/B tests") gives *always-valid* p-values and
confidence intervals: they may be checked after every observation and the experiment
stopped at any moment, and the error rate still stays at α.

For a difference in means with estimate Δ and its variance V, and a normal mixing
distribution N(0, τ²) over the effect:

    Λ = sqrt(V / (V + τ²)) · exp(Δ² τ² / (2 V (V + τ²)))
    p_n = min(p_{n−1}, 1 / Λ_n)
    CI_n: Δ ± sqrt(V (V + τ²) / τ² · (ln((V + τ²) / V) − 2 ln α)), intersected over looks

τ is the scale of effects you expect; results are valid for any τ, a good choice only
makes the test faster. The variance is estimated from the data (an approximation that is
accurate for samples of a few hundred units and more).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Look:
    n_control: int
    n_treatment: int
    effect: float
    p_value: float
    ci: tuple[float, float]


@dataclass
class SequentialResult:
    looks: list[Look] = field(default_factory=list)
    alpha: float = 0.05
    tau: float = math.nan

    @property
    def stopped_at(self) -> int | None:
        """Index of the first look where the always-valid p-value fell below alpha."""
        for i, look in enumerate(self.looks):
            if look.p_value < self.alpha:
                return i
        return None

    @property
    def final(self) -> Look:
        return self.looks[-1]


def msprt_step(effect: float, var: float, tau: float, alpha: float) -> tuple[float, tuple[float, float]]:
    """p-value (1/Λ, not yet made monotone) and the CI at one look."""
    if var <= 0:
        return (0.0 if effect else 1.0), (effect, effect)
    t2 = tau * tau
    log_lr = 0.5 * math.log(var / (var + t2)) + effect**2 * t2 / (2 * var * (var + t2))
    p = 1.0 if log_lr <= 0 else math.exp(-log_lr)
    half = math.sqrt(var * (var + t2) / t2 * (math.log((var + t2) / var) - 2 * math.log(alpha)))
    return min(1.0, p), (effect - half, effect + half)


def msprt(control, treatment, looks=10, tau: float | None = None,
          alpha: float = 0.05) -> SequentialResult:
    """Always-valid analysis of a difference in means, evaluated at several looks.

    control, treatment  observations in arrival order
    looks               number of equally spaced looks, or a list of sample sizes per
                        group at which to look
    tau                 scale of plausible effects (default: 0.1 × standard deviation of
                        the data available at the first look)
    """
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    n_max = min(c.size, t.size)
    if isinstance(looks, int):
        if looks < 1:
            raise ValueError("looks must be >= 1")
        points = sorted({max(2, round(n_max * (k + 1) / looks)) for k in range(looks)})
    else:
        points = sorted(int(x) for x in looks)
        if not points or points[0] < 2 or points[-1] > n_max:
            raise ValueError(f"looks must be between 2 and {n_max}")

    cs, ts = np.cumsum(c), np.cumsum(t)
    cs2, ts2 = np.cumsum(c * c), np.cumsum(t * t)
    if tau is None:
        n0 = points[0]
        sd = np.concatenate([c[:n0], t[:n0]]).std(ddof=1)
        tau = 0.1 * sd if sd > 0 else 1.0
    if tau <= 0:
        raise ValueError("tau must be positive")

    res = SequentialResult(alpha=alpha, tau=float(tau))
    p_prev, lo_prev, hi_prev = 1.0, -math.inf, math.inf
    for n in points:
        mc, mt = cs[n - 1] / n, ts[n - 1] / n
        vc = max(cs2[n - 1] / n - mc * mc, 0.0) * n / (n - 1)
        vt = max(ts2[n - 1] / n - mt * mt, 0.0) * n / (n - 1)
        effect = mt - mc
        p, (lo, hi) = msprt_step(effect, (vc + vt) / n, tau, alpha)
        p_prev = min(p_prev, p)
        lo_prev, hi_prev = max(lo_prev, lo), min(hi_prev, hi)
        res.looks.append(Look(n, n, float(effect), float(p_prev), (float(lo_prev), float(hi_prev))))
    return res
