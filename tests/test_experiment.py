import pandas as pd
import pytest

import honestab as hb
from honestab.datasets import make_experiment, to_views

PRIMARY = hb.Metric("revenue", column="revenue", covariate="pre_revenue", role="primary")


def metrics():
    return [PRIMARY,
            hb.Metric("conversion", column="converted", kind="proportion"),
            hb.Metric("ctr", numerator="clicks", denominator="views"),
            hb.Metric("median time", column="time_spent", kind="median"),
            hb.Metric("sessions", column="sessions", role="guardrail")]


def test_dataset_shape_and_truth():
    df = make_experiment(5000, seed=1)
    assert list(df.columns) == ["user_id", "group", "sessions", "views", "clicks", "time_spent",
                                "orders", "converted", "revenue", "pre_revenue"]
    assert set(df.group) == {"A", "B"} and (df.clicks <= df.views).all()
    assert ((df.converted == 1) == (df.orders > 0)).all()
    views = to_views(df.head(50))
    assert len(views) == df.head(50).views.sum()
    assert views.clicked.sum() == df.head(50).clicks.sum()


def test_ship_when_primary_improves():
    df = make_experiment(40000, revenue_lift=0.25, seed=2)
    rep = hb.analyze(df, metrics())
    assert rep.decision == "КАТИТЬ" and rep.srm.ok
    assert rep.results[0].method == "cuped"
    assert rep.results[1].p_value_adjusted is not None and rep.results[0].p_value_adjusted is None
    md = rep.to_markdown()
    assert "## Решение: КАТИТЬ" in md and "revenue ✱" in md
    assert list(rep.table().columns)[:3] == ["metric", "role", "method"]


def test_no_effect_reports_mde():
    rep = hb.analyze(make_experiment(10000, seed=5), metrics())
    assert rep.decision == "НЕТ ЗНАЧИМОГО ЭФФЕКТА"
    assert any("MDE" in r for r in rep.reasons) and 0 < rep.mde_primary < 1


def test_srm_blocks_decision():
    df = make_experiment(20000, revenue_lift=0.25, srm_loss=0.5, seed=4)
    rep = hb.analyze(df, metrics())
    assert rep.decision.startswith("НЕ ДОВЕРЯТЬ") and not rep.srm.ok


def test_guardrail_blocks_shipping():
    df = make_experiment(40000, revenue_lift=0.25, seed=5)
    df.loc[df.group == "B", "sessions"] -= 1
    rep = hb.analyze(df, metrics())
    assert rep.decision == "НЕ КАТИТЬ" and "sessions" in rep.reasons[0]


def test_primary_got_worse():
    rep = hb.analyze(make_experiment(40000, revenue_lift=-0.3, seed=6), [PRIMARY])
    assert rep.decision == "НЕ КАТИТЬ"


def test_lower_is_better_metric():
    df = make_experiment(40000, revenue_lift=-0.3, seed=6)
    m = hb.Metric("cost", column="revenue", role="primary", higher_is_better=False)
    assert hb.analyze(df, [m]).decision == "КАТИТЬ"


@pytest.mark.parametrize("bad", [
    {"name": "x"},
    {"name": "x", "column": "a", "kind": "magic"},
    {"name": "x", "column": "a", "role": "boss"},
    {"name": "x", "numerator": "a"},
    {"name": "x", "column": "a", "kind": "proportion", "covariate": "b"},
])
def test_metric_validation(bad):
    with pytest.raises(ValueError):
        hb.Metric(**bad)


def test_analyze_validation():
    df = make_experiment(2000, seed=7)
    with pytest.raises(ValueError, match="primary"):
        hb.analyze(df, [hb.Metric("a", column="revenue")])
    with pytest.raises(ValueError, match="not found"):
        hb.analyze(df, [hb.Metric("a", column="nope", role="primary")])
    with pytest.raises(ValueError, match="unexpected groups"):
        hb.analyze(pd.concat([df, df.assign(group="C")]), [PRIMARY])
    with pytest.raises(ValueError, match="must be present"):
        hb.analyze(df, [PRIMARY], treatment="Z")
    with pytest.raises(ValueError, match="only 0 and 1"):
        hb.analyze(df, [hb.Metric("a", column="orders", kind="proportion", role="primary")])
