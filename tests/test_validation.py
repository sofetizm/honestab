"""Slow: a reduced version of validation/run.py with tolerant bounds."""
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
spec = importlib.util.spec_from_file_location(
    "validation_run", Path(__file__).resolve().parents[1] / "validation" / "run.py")
val = importlib.util.module_from_spec(spec)
spec.loader.exec_module(val)


def test_aa_false_positive_rates():
    res = val.aa_methods(400, 5000, 7)
    for name in ("Welch: выручка", "CUPED: выручка", "z-тест: конверсия", "дельта-метод: CTR"):
        assert res[name]["lo"] <= 0.05 <= res[name]["hi"] or res[name]["rate"] < 0.08, name


def test_peeking_inflates_and_msprt_controls():
    res = val.peeking(400, 2000, 20, 8)
    assert res["naive_by_looks"][20]["rate"] > 0.12
    assert res["msprt"]["rate"] < 0.07


def test_multiple_corrections():
    res = val.multiple_metrics(600, 10, 200, 9)
    assert res["без поправки"]["rate"] > 0.25
    assert res["Holm"]["rate"] < 0.08
