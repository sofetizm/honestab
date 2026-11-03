"""Checks that decide whether an experiment can be analysed at all."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SRMResult:
    observed: list[int]
    expected: list[float]
    p_value: float
    threshold: float

    @property
    def ok(self) -> bool:
        return self.p_value >= self.threshold

    def __str__(self) -> str:
        status = "OK" if self.ok else "SRM DETECTED"
        return (f"{status}: observed {self.observed}, expected {[round(e) for e in self.expected]}, "
                f"p = {self.p_value:.3g}")


def srm(counts, expected_shares=None, threshold: float = 0.001) -> SRMResult:
    """Sample ratio mismatch: are group sizes consistent with the planned split?

    A mismatch means that units were lost or added non-randomly (a bug in assignment,
    logging, filtering, a redirect that breaks for one variant). Any metric comparison is
    then biased, however significant it looks. The usual threshold is p < 0.001.
    """
    obs = np.asarray(counts, dtype=float)
    if obs.ndim != 1 or obs.size < 2 or (obs < 0).any():
        raise ValueError("counts must be at least two non-negative numbers")
    shares = (np.full(obs.size, 1 / obs.size) if expected_shares is None
              else np.asarray(expected_shares, dtype=float))
    if shares.size != obs.size or (shares <= 0).any():
        raise ValueError("expected_shares must be positive, one per group")
    exp = shares / shares.sum() * obs.sum()
    p = float(stats.chisquare(obs, exp).pvalue)
    return SRMResult([int(x) for x in obs], [float(e) for e in exp], p, threshold)
