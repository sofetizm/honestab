"""Corrections for testing many metrics or many variants at once.

With 10 independent metrics and α = 0.05 the chance of at least one false positive is
1 − 0.95¹⁰ ≈ 40%.
"""
from __future__ import annotations

import numpy as np


def _check(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be a 1-D array of numbers in [0, 1]")
    return p


def holm(p_values) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values: controls the family-wise error rate
    (the chance of *any* false positive). Uniformly more powerful than Bonferroni."""
    p = _check(p_values)
    m = p.size
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values: controls the false discovery rate (the
    expected share of false positives among the significant results)."""
    p = _check(p_values)
    m = p.size
    order = np.argsort(p)[::-1]
    ranked = p[order] * m / np.arange(m, 0, -1)
    adjusted = np.minimum.accumulate(ranked)
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def adjust(p_values, method: str = "holm") -> np.ndarray:
    methods = {"holm": holm, "bh": benjamini_hochberg, "none": lambda p: _check(p).copy()}
    if method not in methods:
        raise ValueError(f"unknown correction {method!r}; use one of {sorted(methods)}")
    return methods[method](p_values)
