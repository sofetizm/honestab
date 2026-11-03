import math

import numpy as np
import pytest

import honestab as hb
from honestab.sequential import msprt_step


def test_step_formula():
    p, (lo, hi) = msprt_step(effect=0.3, var=0.01, tau=0.2, alpha=0.05)
    lam = math.sqrt(0.01 / 0.05) * math.exp(0.09 * 0.04 / (2 * 0.01 * 0.05))
    assert p == pytest.approx(1 / lam)
    half = math.sqrt(0.01 * 0.05 / 0.04 * (math.log(5) - 2 * math.log(0.05)))
    assert (lo, hi) == pytest.approx((0.3 - half, 0.3 + half))


def test_pvalues_monotone_and_cis_nested():
    rng = np.random.default_rng(0)
    res = hb.msprt(rng.normal(0, 1, 5000), rng.normal(0.05, 1, 5000), looks=25)
    ps = [lk.p_value for lk in res.looks]
    assert all(b <= a for a, b in zip(ps, ps[1:], strict=False))
    for prev, cur in zip(res.looks, res.looks[1:], strict=False):
        assert cur.ci[0] >= prev.ci[0] and cur.ci[1] <= prev.ci[1]
    assert res.looks[-1].n_control == 5000


def test_detects_large_effect_early():
    rng = np.random.default_rng(1)
    res = hb.msprt(rng.normal(0, 1, 4000), rng.normal(0.3, 1, 4000), looks=20)
    assert res.stopped_at is not None and res.stopped_at < 10
    assert res.final.ci[0] > 0


def test_explicit_looks_and_validation():
    x = np.arange(100.0)
    res = hb.msprt(x, x + 1, looks=[10, 50, 100], tau=1.0)
    assert [lk.n_control for lk in res.looks] == [10, 50, 100] and res.tau == 1.0
    for bad in ([1], [10, 200], []):
        with pytest.raises(ValueError):
            hb.msprt(x, x, looks=bad)
    with pytest.raises(ValueError):
        hb.msprt(x, x, looks=0)
    with pytest.raises(ValueError):
        hb.msprt(x, x, tau=-1)
