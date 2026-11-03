"""Statistical methods for comparing two groups.

Every function returns an :class:`EffectResult` with the absolute and the relative effect,
confidence intervals for both and a two-sided p-value.

Which one to use:

=====================  ==================================================================
welch                  mean of a per-user metric (revenue, time, orders) — the default
proportions            share of users who did something (conversion)
ratio                  metric = sum(numerator) / sum(denominator) when the unit of
                       randomization is the user but the metric is per session, per
                       click, per order (CTR, average check, revenue per session)
cuped                  welch with variance reduction by a pre-experiment covariate
bootstrap              medians, quantiles and other non-mean statistics
=====================  ==================================================================
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import stats

from .result import EffectResult


def _as_array(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if a.size < 2:
        raise ValueError(f"{name} needs at least 2 observations")
    if np.isnan(a).any():
        raise ValueError(f"{name} contains NaN; drop or fill missing values explicitly")
    return a


def _relative(mc: float, mt: float, var_mc: float, var_mt: float, q: float):
    """Relative effect mt / mc − 1 with Fieller's confidence interval.

    Fieller's interval is the set of ratios ρ for which (mt − ρ·mc)² ≤ q²(var_mt + ρ²·var_mc).
    It accounts for the uncertainty of the control mean, and it excludes "no change" exactly
    when the test of the difference is significant — the interval and the p-value never
    disagree. If the control mean itself is not distinguishable from zero the interval is
    unbounded and NaN is returned.
    """
    if mc == 0:
        return math.nan, (math.nan, math.nan)
    rel = mt / mc - 1
    a = mc * mc - q * q * var_mc
    b = mt * mc
    c = mt * mt - q * q * var_mt
    disc = b * b - a * c
    if a <= 0 or disc < 0:
        return rel, (math.nan, math.nan)
    root = math.sqrt(disc)
    return rel, ((b - root) / a - 1, (b + root) / a - 1)


def _from_moments(metric: str, method: str, mc: float, mt: float, var_mc: float, var_mt: float,
                  n_c: int, n_t: int, alpha: float, df: float | None = None,
                  notes: list[str] | None = None) -> EffectResult:
    """Build a result from the two group estimates and the variances of those estimates."""
    effect = mt - mc
    se = math.sqrt(var_mc + var_mt)
    dist = stats.t(df) if df is not None and math.isfinite(df) else stats.norm()
    q = dist.ppf(1 - alpha / 2)
    if se == 0:
        p = 1.0 if effect == 0 else 0.0
    else:
        p = float(2 * dist.sf(abs(effect) / se))
    rel, rel_ci = _relative(mc, mt, var_mc, var_mt, q)
    return EffectResult(metric=metric, method=method, control_value=mc, treatment_value=mt,
                        effect=effect, ci=(effect - q * se, effect + q * se), rel_effect=rel,
                        rel_ci=rel_ci, p_value=p, n_control=n_c, n_treatment=n_t, alpha=alpha,
                        std_error=se, notes=notes or [])


def welch(control, treatment, alpha: float = 0.05, metric: str = "metric") -> EffectResult:
    """Welch's t-test for the difference in means (unequal variances).

    Valid for skewed and heavy-tailed metrics (revenue) once groups are large, by the
    central limit theorem; it is the default for per-user metrics.
    """
    c, t = _as_array(control, "control"), _as_array(treatment, "treatment")
    vc, vt = c.var(ddof=1) / c.size, t.var(ddof=1) / t.size
    df = (vc + vt) ** 2 / (vc**2 / (c.size - 1) + vt**2 / (t.size - 1)) if vc + vt > 0 else None
    return _from_moments(metric, "welch", float(c.mean()), float(t.mean()), vc, vt,
                         c.size, t.size, alpha, df)


def proportions(successes_control: int, n_control: int, successes_treatment: int,
                n_treatment: int, alpha: float = 0.05, metric: str = "conversion") -> EffectResult:
    """Two-proportion z-test. The p-value uses the pooled variance (the null hypothesis
    says the proportions are equal); the confidence intervals use unpooled variances."""
    for s, n in ((successes_control, n_control), (successes_treatment, n_treatment)):
        if n < 1 or not 0 <= s <= n:
            raise ValueError("need 0 <= successes <= n and n >= 1")
    pc, pt = successes_control / n_control, successes_treatment / n_treatment
    res = _from_moments(metric, "proportions", pc, pt, pc * (1 - pc) / n_control,
                        pt * (1 - pt) / n_treatment, n_control, n_treatment, alpha)
    pooled = (successes_control + successes_treatment) / (n_control + n_treatment)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    res.p_value = 1.0 if se0 == 0 else float(2 * stats.norm.sf(abs(pt - pc) / se0))
    return res


def _ratio_moments(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Ratio sum(num)/sum(den) and the variance of that ratio by the delta method."""
    n = num.size
    mx, my = num.mean(), den.mean()
    if my == 0:
        raise ValueError("denominator sums to zero")
    vx, vy = num.var(ddof=1), den.var(ddof=1)
    cxy = np.cov(num, den, ddof=1)[0, 1]
    var = (vx / my**2 - 2 * mx * cxy / my**3 + mx**2 * vy / my**4) / n
    return float(mx / my), float(max(var, 0.0))


def ratio(num_control, den_control, num_treatment, den_treatment, alpha: float = 0.05,
          metric: str = "ratio") -> EffectResult:
    """Ratio metric sum(numerator) / sum(denominator), e.g. CTR = clicks / views.

    Inputs are **per randomization unit** (one row per user: that user's clicks and views).
    Treating every view as an independent observation underestimates the variance —
    views of the same user are correlated — and gives too many false positives.
    The delta method accounts for this correlation.
    """
    nc, dc = _as_array(num_control, "num_control"), _as_array(den_control, "den_control")
    nt, dt = _as_array(num_treatment, "num_treatment"), _as_array(den_treatment, "den_treatment")
    if nc.size != dc.size or nt.size != dt.size:
        raise ValueError("numerator and denominator must have one value per unit")
    rc, vc = _ratio_moments(nc, dc)
    rt, vt = _ratio_moments(nt, dt)
    return _from_moments(metric, "ratio (delta method)", rc, rt, vc, vt, nc.size, nt.size, alpha)


def cuped(control, treatment, covariate_control, covariate_treatment, alpha: float = 0.05,
          metric: str = "metric") -> EffectResult:
    """CUPED: subtract the part of the metric explained by a pre-experiment covariate.

    y' = y − θ (x − mean(x)),  θ = cov(y, x) / var(x), estimated on both groups together.
    The estimate stays unbiased (x is measured before randomization, so it is independent
    of the treatment) and the variance shrinks by a factor of 1 − ρ², ρ = corr(y, x).
    The covariate must not be affected by the treatment.
    """
    yc, yt = _as_array(control, "control"), _as_array(treatment, "treatment")
    xc = _as_array(covariate_control, "covariate_control")
    xt = _as_array(covariate_treatment, "covariate_treatment")
    if yc.size != xc.size or yt.size != xt.size:
        raise ValueError("covariate must have one value per unit")
    y, x = np.concatenate([yc, yt]), np.concatenate([xc, xt])
    vx = x.var(ddof=1)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
    mx = x.mean()
    res = welch(yc - theta * (xc - mx), yt - theta * (xt - mx), alpha, metric)
    rho = float(np.corrcoef(y, x)[0, 1]) if vx > 0 and y.var() > 0 else 0.0
    res.method = "cuped"
    res.notes.append(f"theta = {theta:.4g}; corr(y, x) = {rho:.3f}; "
                     f"variance reduced by ~{rho**2:.0%}")
    return res


def _resample_stat(x: np.ndarray, statistic, n_resamples: int, rng) -> np.ndarray:
    """Statistic over bootstrap resamples, in chunks to bound memory (~40 MB)."""
    chunk = max(1, int(5_000_000 // x.size))
    out = []
    for start in range(0, n_resamples, chunk):
        sample = rng.choice(x, size=(min(chunk, n_resamples - start), x.size))
        try:
            values = np.asarray(statistic(sample, axis=1), dtype=float)
        except TypeError:
            values = np.apply_along_axis(statistic, 1, sample)
        out.append(values)
    return np.concatenate(out)


def bootstrap(control, treatment, statistic: Callable[[np.ndarray], float] = np.median,
              alpha: float = 0.05, n_resamples: int = 5000, seed: int | None = 0,
              metric: str = "metric") -> EffectResult:
    """Percentile bootstrap for any statistic (median, 90th percentile, trimmed mean...).

    The p-value is obtained by inverting the bootstrap interval (the smallest alpha at
    which the interval excludes zero). It is approximate; for means prefer welch.
    """
    c, t = _as_array(control, "control"), _as_array(treatment, "treatment")
    rng = np.random.default_rng(seed)
    sc = _resample_stat(c, statistic, n_resamples, rng)
    st = _resample_stat(t, statistic, n_resamples, rng)
    diff = st - sc
    vc_hat, vt_hat = float(statistic(c)), float(statistic(t))
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    share_le = (np.sum(diff <= 0) + 1) / (n_resamples + 1)
    share_ge = (np.sum(diff >= 0) + 1) / (n_resamples + 1)
    p = float(min(1.0, 2 * min(share_le, share_ge)))
    if vc_hat != 0:
        rel = vt_hat / vc_hat - 1
        rel_boot = st / np.where(sc == 0, np.nan, sc) - 1
        rlo, rhi = np.nanquantile(rel_boot, [alpha / 2, 1 - alpha / 2])
    else:
        rel, rlo, rhi = math.nan, math.nan, math.nan
    return EffectResult(metric=metric, method=f"bootstrap ({getattr(statistic, '__name__', 'stat')})",
                        control_value=vc_hat, treatment_value=vt_hat, effect=vt_hat - vc_hat,
                        ci=(float(lo), float(hi)), rel_effect=rel, rel_ci=(float(rlo), float(rhi)),
                        p_value=p, n_control=c.size, n_treatment=t.size, alpha=alpha,
                        std_error=float(diff.std(ddof=1)))
