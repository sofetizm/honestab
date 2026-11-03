"""Synthetic experiments with a known true effect — for demos, tests and validation.

Each user has a latent activity level. It drives sessions, page views, clicks,
conversion and revenue, and also the revenue *before* the experiment (the CUPED
covariate). Users differ in their click-through rate, so views of one user are
correlated — the situation where a naive per-view test of CTR goes wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_experiment(n_users: int = 20_000, *, revenue_lift: float = 0.0,
                    conversion_lift: float = 0.0, ctr_lift: float = 0.0, split: float = 0.5,
                    srm_loss: float = 0.0, seed: int | None = 0) -> pd.DataFrame:
    """One row per user.

    Columns: user_id, group ("A" control, "B" treatment), sessions, views, clicks,
    time_spent (minutes), orders, converted (0/1), revenue, pre_revenue (revenue in the four weeks before).

    revenue_lift, conversion_lift, ctr_lift  true relative effects in group B
    srm_loss  share of light users (one session) of group B lost by a "logging bug";
              creates a sample ratio mismatch and a biased comparison
    """
    rng = np.random.default_rng(seed)
    activity = rng.gamma(shape=2.0, scale=0.5, size=n_users)
    is_b = rng.random(n_users) < split

    sessions = 1 + rng.poisson(2.5 * activity)
    views = sessions + rng.poisson(1.5 * sessions)
    user_ctr = rng.beta(2, 28, n_users)
    user_ctr = np.where(is_b, np.clip(user_ctr * (1 + ctr_lift), 0, 1), user_ctr)
    clicks = rng.binomial(views, user_ctr)
    time_spent = rng.lognormal(mean=np.log(4 * (1 + activity)), sigma=0.8)

    # purchases: a persistent per-user order rate and average check, the same before and
    # during the experiment — that is what makes the pre-period a useful CUPED covariate
    order_rate = 0.15 * activity * rng.gamma(shape=0.3, scale=1 / 0.3, size=n_users)
    check = rng.lognormal(mean=np.log(1500), sigma=0.6, size=n_users)
    rate_now = np.where(is_b, order_rate * (1 + conversion_lift), order_rate)
    orders = rng.poisson(rate_now)
    converted = (orders > 0).astype(int)
    revenue = orders * check * rng.lognormal(0, 0.3, n_users) * np.where(is_b, 1 + revenue_lift, 1)
    pre_orders = rng.poisson(order_rate * 4)
    pre_revenue = pre_orders * check * rng.lognormal(0, 0.3, n_users)

    df = pd.DataFrame({
        "user_id": np.arange(n_users),
        "group": np.where(is_b, "B", "A"),
        "sessions": sessions, "views": views, "clicks": clicks,
        "time_spent": time_spent.round(2),
        "orders": orders, "converted": converted, "revenue": revenue.round(2),
        "pre_revenue": pre_revenue.round(2),
    })
    if srm_loss > 0:
        lost = is_b & (sessions == 1) & (rng.random(n_users) < srm_loss)
        df = df[~lost].reset_index(drop=True)
    return df


def to_views(df: pd.DataFrame) -> pd.DataFrame:
    """Expand per-user data to one row per page view (clicked 0/1) — the shape that tempts
    analysts to run a per-view test of CTR."""
    rows = np.repeat(df.index.to_numpy(), df["views"].to_numpy())
    out = df.loc[rows, ["user_id", "group"]].reset_index(drop=True)
    clicked = np.zeros(len(out), dtype=int)
    pos = 0
    for v, c in zip(df["views"].to_numpy(), df["clicks"].to_numpy(), strict=True):
        clicked[pos:pos + c] = 1
        pos += v
    out["clicked"] = clicked
    return out
