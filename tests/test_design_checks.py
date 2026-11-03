import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

import honestab as hb


@pytest.mark.parametrize("d", [0.05, 0.1, 0.2])
def test_sample_size_matches_statsmodels(d):
    n = hb.sample_size(std=1, mde=d)
    assert n["control"] == pytest.approx(NormalIndPower().solve_power(d, alpha=0.05, power=0.8), abs=1)


def test_sample_size_relative_ratio_and_cuped():
    base = hb.sample_size(std=200, mde_rel=0.05, baseline=1000)
    assert base == hb.sample_size(std=200, mde=50)
    cup = hb.sample_size(std=200, mde=50, variance_reduction=0.3)
    assert cup["control"] == pytest.approx(base["control"] * 0.7, abs=1)
    unequal = hb.sample_size(std=200, mde=50, ratio=2)
    assert unequal["treatment"] == pytest.approx(2 * unequal["control"], abs=1)


def test_sample_size_unequal_variance():
    eq = hb.sample_size(std=100, mde=10)["control"]
    lifted = hb.sample_size(std=100, mde=10, std_treatment=110)["control"]
    assert lifted == pytest.approx(eq * (1 + 1.21) / 2, abs=1)
    with pytest.raises(ValueError):
        hb.sample_size(std=100, mde=10, std_treatment=0)


def test_sample_size_proportion():
    n = hb.sample_size_proportion(0.05, mde_rel=0.05)
    h = proportion_effectsize(0.0525, 0.05)
    assert n["control"] == pytest.approx(NormalIndPower().solve_power(h, alpha=0.05, power=0.8), rel=0.01)


@pytest.mark.parametrize("kwargs", [
    {"std": 0, "mde": 1}, {"std": 1}, {"std": 1, "mde": 1, "mde_rel": 0.1},
    {"std": 1, "mde_rel": 0.1}, {"std": 1, "mde": 1, "variance_reduction": 1.0},
    {"std": 1, "mde": 1, "alpha": 1.5},
])
def test_sample_size_invalid(kwargs):
    with pytest.raises(ValueError):
        hb.sample_size(**kwargs)


def test_mde_is_inverse_of_sample_size():
    n = hb.sample_size(std=3, mde=0.25)["control"]
    assert hb.mde(3, n) == pytest.approx(0.25, rel=0.01)


def test_duration():
    assert hb.duration_days(100_000, 20_000) == 7
    assert hb.duration_days(300_000, 20_000, share_in_experiment=0.5) == 30
    with pytest.raises(ValueError):
        hb.duration_days(10, 0)


def test_srm():
    assert hb.srm([5000, 5000]).p_value == pytest.approx(1.0)
    bad = hb.srm([50000, 48500])
    assert not bad.ok and "SRM DETECTED" in str(bad)
    assert hb.srm([9000, 1000], [0.9, 0.1]).ok
    with pytest.raises(ValueError):
        hb.srm([10])
    with pytest.raises(ValueError):
        hb.srm([10, 10], [0.5])


def test_corrections_match_statsmodels():
    p = np.random.default_rng(3).random(15) * 0.2
    assert np.allclose(hb.holm(p), multipletests(p, method="holm")[1])
    assert np.allclose(hb.benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])
    with pytest.raises(ValueError):
        hb.multiple.adjust(p, "magic")
    with pytest.raises(ValueError):
        hb.holm([0.1, 1.5])


def test_beta_binomial():
    r = hb.bayes.beta_binomial(500, 10000, 600, 10000)
    assert r.prob_treatment_better > 0.99
    assert r.expected_loss_treatment < r.expected_loss_control
    lo, hi = r.rel_effect_ci
    assert lo < 0.2 < hi
