import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

import honestab as hb

rng = np.random.default_rng(0)
A = rng.lognormal(0, 1, 3000)
B = rng.lognormal(0.05, 1, 3500)


def test_welch_matches_scipy():
    r = hb.welch(A, B)
    ref = stats.ttest_ind(B, A, equal_var=False)
    assert r.p_value == pytest.approx(ref.pvalue, rel=1e-10)
    lo, hi = ref.confidence_interval(0.95)
    assert r.ci == pytest.approx((lo, hi), rel=1e-9)
    assert r.effect == pytest.approx(B.mean() - A.mean())
    assert (r.n_control, r.n_treatment) == (3000, 3500)


def test_relative_ci_is_fieller():
    r = hb.welch(A, B)
    mc, mt = A.mean(), B.mean()
    vc, vt = A.var(ddof=1) / A.size, B.var(ddof=1) / B.size
    q = stats.t(r_df(A, B)).ppf(0.975)
    for rho in (1 + r.rel_ci[0], 1 + r.rel_ci[1]):
        assert (mt - rho * mc) ** 2 == pytest.approx(q * q * (vt + rho * rho * vc), rel=1e-9)
    assert r.rel_effect == pytest.approx(mt / mc - 1)


def r_df(a, b):
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    return (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))


@pytest.mark.parametrize("seed", range(40))
def test_relative_ci_agrees_with_p_value(seed):
    g = np.random.default_rng(seed)
    a, b = g.lognormal(0, 1.5, 800), g.lognormal(0.1, 1.5, 800)
    r = hb.welch(a, b)
    excludes_zero = r.rel_ci[0] > 0 or r.rel_ci[1] < 0
    assert excludes_zero == (r.p_value < 0.05)


def test_relative_ci_unbounded_when_control_is_noise():
    r = hb.welch([-1.0, 1.0, -2.0, 2.0, 0.1], [1.0, 2.0, 3.0, 2.5, 1.5])
    assert math.isnan(r.rel_ci[0]) and math.isnan(r.rel_ci[1])


def test_proportions_matches_statsmodels():
    r = hb.proportions(500, 10000, 560, 10000)
    assert r.p_value == pytest.approx(proportions_ztest([560, 500], [10000, 10000])[1], rel=1e-9)
    assert (r.control_value, r.treatment_value) == (0.05, 0.056)
    se = math.sqrt(0.05 * 0.95 / 10000 + 0.056 * 0.944 / 10000)
    assert r.ci == pytest.approx((0.006 - 1.959964 * se, 0.006 + 1.959964 * se), rel=1e-6)
    assert r.rel_ci[0] < 0.12 < r.rel_ci[1]


@pytest.mark.parametrize("args", [(5, 4, 1, 2), (-1, 10, 1, 10), (1, 0, 1, 10)])
def test_proportions_invalid(args):
    with pytest.raises(ValueError):
        hb.proportions(*args)


def test_ratio_equals_linearization():
    """Delta-method variance of sum(x)/sum(y) equals the variance of the mean of the
    linearized metric (x − R·y) / mean(y)."""
    views = rng.integers(1, 30, 4000).astype(float)
    clicks = rng.binomial(views.astype(int), rng.beta(2, 20, 4000)).astype(float)
    r = hb.ratio(clicks[:2000], views[:2000], clicks[2000:], views[2000:])

    def lin_var(x, y):
        R = x.sum() / y.sum()
        return ((x - R * y) / y.mean()).var(ddof=1) / x.size

    se = math.sqrt(lin_var(clicks[:2000], views[:2000]) + lin_var(clicks[2000:], views[2000:]))
    assert r.std_error == pytest.approx(se, rel=1e-9)
    assert r.control_value == pytest.approx(clicks[:2000].sum() / views[:2000].sum())


def test_ratio_validation():
    with pytest.raises(ValueError, match="one value per unit"):
        hb.ratio([1, 2], [1, 2, 3], [1, 2], [1, 2])
    with pytest.raises(ValueError, match="zero"):
        hb.ratio([1, 2], [0, 0], [1, 2], [1, 2])


def test_cuped_reduces_variance_and_keeps_estimate():
    x = rng.normal(100, 20, 20000)
    y = x * 0.8 + rng.normal(0, 10, 20000)
    yc, yt, xc, xt = y[:10000], y[10000:] + 1.0, x[:10000], x[10000:]
    plain = hb.welch(yc, yt)
    adj = hb.cuped(yc, yt, xc, xt)
    rho2 = np.corrcoef(y, x)[0, 1] ** 2
    assert adj.std_error**2 == pytest.approx(plain.std_error**2 * (1 - rho2), rel=0.05)
    assert adj.effect == pytest.approx(1.0, abs=4 * adj.std_error)
    assert adj.method == "cuped" and "variance reduced" in adj.notes[0]


def test_cuped_with_constant_covariate_is_welch():
    c = hb.cuped(A[:100], B[:100], np.ones(100), np.ones(100))
    w = hb.welch(A[:100], B[:100])
    assert c.p_value == pytest.approx(w.p_value)


def test_bootstrap_deterministic_and_sensible():
    a = rng.lognormal(0, 1, 2000)
    b = rng.lognormal(0.3, 1, 2000)
    r1 = hb.bootstrap(a, b, np.median, n_resamples=2000, seed=1)
    r2 = hb.bootstrap(a, b, np.median, n_resamples=2000, seed=1)
    assert r1.ci == r2.ci and r1.p_value == r2.p_value
    assert r1.ci[0] < np.median(b) - np.median(a) < r1.ci[1]
    assert r1.p_value < 0.01 and r1.method == "bootstrap (median)"


def test_bootstrap_custom_statistic_without_axis():
    r = hb.bootstrap(A[:300], B[:300], lambda x: float(np.percentile(x, 90)), n_resamples=300)
    assert r.ci[0] < r.ci[1]


@pytest.mark.parametrize("bad", [[1.0], [[1.0, 2.0]], [1.0, float("nan")]])
def test_input_validation(bad):
    with pytest.raises(ValueError):
        hb.welch(bad, [1.0, 2.0, 3.0])


def test_result_str_and_dict():
    r = hb.welch(A, B)
    d = r.to_dict()
    assert d["significant"] == (r.p_value < 0.05)
    assert "welch" in str(r) and "%" in str(r)
