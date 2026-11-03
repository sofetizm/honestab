"""Bayesian view of a conversion experiment (Beta-Binomial).

Gives "probability that B is better" and "expected loss if we choose B". Useful for
communicating results; it does **not** make peeking safe and does not replace the
pre-registered frequentist decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BayesResult:
    prob_treatment_better: float
    expected_loss_treatment: float
    expected_loss_control: float
    rel_effect_ci: tuple[float, float]


def beta_binomial(successes_control: int, n_control: int, successes_treatment: int,
                  n_treatment: int, prior: tuple[float, float] = (1.0, 1.0),
                  draws: int = 200_000, seed: int | None = 0, ci: float = 0.95) -> BayesResult:
    a, b = prior
    rng = np.random.default_rng(seed)
    pc = rng.beta(a + successes_control, b + n_control - successes_control, draws)
    pt = rng.beta(a + successes_treatment, b + n_treatment - successes_treatment, draws)
    rel = pt / pc - 1
    q = (1 - ci) / 2
    return BayesResult(
        prob_treatment_better=float(np.mean(pt > pc)),
        expected_loss_treatment=float(np.mean(np.maximum(pc - pt, 0))),
        expected_loss_control=float(np.mean(np.maximum(pt - pc, 0))),
        rel_effect_ci=(float(np.quantile(rel, q)), float(np.quantile(rel, 1 - q))),
    )
