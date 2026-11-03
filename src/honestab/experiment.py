"""High-level analysis: one call from a table of users to a decision.

    from honestab import Metric, analyze
    report = analyze(df, [
        Metric("revenue", column="revenue", covariate="pre_revenue", role="primary"),
        Metric("conversion", column="converted", kind="proportion"),
        Metric("ctr", numerator="clicks", denominator="views"),
        Metric("sessions", column="sessions", role="guardrail"),
    ])
    print(report.to_markdown())
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import methods
from .checks import SRMResult, srm
from .design import mde_from_se
from .multiple import adjust
from .result import EffectResult

KINDS = ("mean", "proportion", "ratio", "median")
ROLES = ("primary", "secondary", "guardrail")


@dataclass
class Metric:
    """What to measure.

    kind="mean"        column: per-unit value (Welch; CUPED if `covariate` is given)
    kind="proportion"  column: 0/1 per unit
    kind="ratio"       numerator / denominator columns, per unit (delta method)
    kind="median"      column: per-unit value (bootstrap)
    role               primary — the decision metric, exactly one;
                       secondary — reported with a multiple-testing correction;
                       guardrail — must not get significantly worse
    higher_is_better   direction used for the decision and for guardrails
    """

    name: str
    column: str | None = None
    kind: str = "mean"
    numerator: str | None = None
    denominator: str | None = None
    covariate: str | None = None
    role: str = "secondary"
    higher_is_better: bool = True

    def __post_init__(self):
        if self.numerator or self.denominator:
            self.kind = "ratio"
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind must be one of {KINDS}")
        if self.role not in ROLES:
            raise ValueError(f"{self.name}: role must be one of {ROLES}")
        if self.kind == "ratio" and not (self.numerator and self.denominator):
            raise ValueError(f"{self.name}: a ratio metric needs numerator and denominator")
        if self.kind != "ratio" and not self.column:
            raise ValueError(f"{self.name}: column is required")
        if self.covariate and self.kind != "mean":
            raise ValueError(f"{self.name}: CUPED covariate is supported for kind='mean' only")

    def columns(self) -> list[str]:
        cols = [self.numerator, self.denominator] if self.kind == "ratio" else [self.column]
        return cols + ([self.covariate] if self.covariate else [])


@dataclass
class Report:
    srm: SRMResult
    results: list[EffectResult]
    metrics: list[Metric]
    decision: str
    reasons: list[str] = field(default_factory=list)
    mde_primary: float | None = None
    alpha: float = 0.05
    correction: str = "holm"

    def table(self) -> pd.DataFrame:
        rows = []
        for m, r in zip(self.metrics, self.results, strict=True):
            rows.append({
                "metric": r.metric, "role": m.role, "method": r.method,
                "control": r.control_value, "treatment": r.treatment_value,
                "rel_effect": r.rel_effect, "rel_ci_low": r.rel_ci[0], "rel_ci_high": r.rel_ci[1],
                "p_value": r.p_value, "p_adjusted": r.p_value_adjusted,
                "significant": r.significant,
            })
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        lines = [f"## Решение: {self.decision}", ""]
        lines += [f"- {r}" for r in self.reasons] + [""]
        lines += [f"**Проверка SRM:** {'пройдена' if self.srm.ok else 'НЕ ПРОЙДЕНА'} — "
                  f"размеры групп {self.srm.observed}, ожидалось "
                  f"{[round(e) for e in self.srm.expected]}, p = {self.srm.p_value:.3g}", ""]
        lines += ["| Метрика | Роль | Метод | Контроль | Тест | Эффект | 95% ДИ | p | p (с поправкой) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for m, r in zip(self.metrics, self.results, strict=True):
            adj = "—" if r.p_value_adjusted is None else f"{r.p_value_adjusted:.3g}"
            mark = " ✱" if r.significant else ""
            lines.append(
                f"| {r.metric}{mark} | {m.role} | {r.method} | {r.control_value:.4g} | "
                f"{r.treatment_value:.4g} | {_pct(r.rel_effect)} | "
                f"[{_pct(r.rel_ci[0])}; {_pct(r.rel_ci[1])}] | {r.p_value:.3g} | {adj} |")
        lines += ["", f"✱ — статистически значимо при α = {self.alpha} "
                      f"(вторичные метрики — с поправкой {self.correction})."]
        notes = [f"- {r.metric}: {n}" for r in self.results for n in r.notes]
        if notes:
            lines += ["", "Примечания:"] + notes
        return "\n".join(lines)


def _pct(x: float) -> str:
    return "—" if x != x else f"{x:+.2%}"


def _run_metric(m: Metric, a: pd.DataFrame, b: pd.DataFrame, alpha: float) -> EffectResult:
    if m.kind == "ratio":
        return methods.ratio(a[m.numerator], a[m.denominator], b[m.numerator], b[m.denominator],
                             alpha, m.name)
    if m.kind == "proportion":
        ya, yb = a[m.column].to_numpy(), b[m.column].to_numpy()
        if not np.isin(np.concatenate([ya, yb]), (0, 1)).all():
            raise ValueError(f"{m.name}: a proportion metric must contain only 0 and 1")
        return methods.proportions(int(ya.sum()), ya.size, int(yb.sum()), yb.size, alpha, m.name)
    if m.kind == "median":
        return methods.bootstrap(a[m.column], b[m.column], np.median, alpha, metric=m.name)
    if m.covariate:
        return methods.cuped(a[m.column], b[m.column], a[m.covariate], b[m.covariate],
                             alpha, m.name)
    return methods.welch(a[m.column], b[m.column], alpha, m.name)


def analyze(df: pd.DataFrame, metrics: list[Metric], *, group_col: str = "group",
            control: str = "A", treatment: str = "B", expected_split: float = 0.5,
            alpha: float = 0.05, correction: str = "holm",
            srm_threshold: float = 0.001) -> Report:
    """Analyse a two-group experiment with one row per randomization unit."""
    primaries = [m for m in metrics if m.role == "primary"]
    if len(primaries) != 1:
        raise ValueError("exactly one metric must have role='primary'")
    if group_col not in df:
        raise ValueError(f"column {group_col!r} not found")
    missing = sorted({c for m in metrics for c in m.columns()} - set(df.columns))
    if missing:
        raise ValueError(f"columns not found: {missing}")
    groups = set(df[group_col].unique())
    if not {control, treatment} <= groups:
        raise ValueError(f"groups {control!r} and {treatment!r} must be present; found {groups}")
    if groups - {control, treatment}:
        raise ValueError(f"unexpected groups {groups - {control, treatment}}; filter them first")

    a, b = df[df[group_col] == control], df[df[group_col] == treatment]
    check = srm([len(a), len(b)], [1 - expected_split, expected_split], srm_threshold)
    results = [_run_metric(m, a, b, alpha) for m in metrics]

    secondary = [i for i, m in enumerate(metrics) if m.role == "secondary"]
    if secondary:
        adjusted = adjust([results[i].p_value for i in secondary], correction)
        for i, p in zip(secondary, adjusted, strict=True):
            results[i].p_value_adjusted = float(p)

    decision, reasons = _decide(metrics, results, check, alpha)
    i_primary = metrics.index(primaries[0])
    mde_abs = mde_from_se(results[i_primary].std_error, alpha)
    base = results[i_primary].control_value
    mde_rel = mde_abs / abs(base) if base else None
    if decision == "НЕТ ЗНАЧИМОГО ЭФФЕКТА" and mde_rel is not None:
        reasons.append(f"При таком объёме эксперимент замечает эффект от ~{mde_rel:.1%} "
                       "(MDE при мощности 80%); эффект меньше этого мог остаться незамеченным.")
    return Report(check, results, metrics, decision, reasons, mde_rel, alpha, correction)


def _decide(metrics, results, check: SRMResult, alpha: float):
    if not check.ok:
        return ("НЕ ДОВЕРЯТЬ РЕЗУЛЬТАТАМ (SRM)", [
            f"Размеры групп не соответствуют плану (p = {check.p_value:.2g}). Часть пользователей "
            "потеряна или добавлена неслучайно, поэтому сравнение метрик смещено. "
            "Найдите причину (назначение в группы, логирование, фильтры) до любых выводов."])
    reasons = []
    bad = []
    for m, r in zip(metrics, results, strict=True):
        worse = (r.effect < 0) == m.higher_is_better
        if m.role == "guardrail" and r.p_value < alpha and worse and r.effect != 0:
            bad.append(m.name)
            reasons.append(f"Guardrail «{m.name}» значимо ухудшилась: {r.rel_effect:+.2%} "
                           f"(p = {r.p_value:.3g}).")
    m, r = next((m, r) for m, r in zip(metrics, results, strict=True) if m.role == "primary")
    good = (r.effect > 0) == m.higher_is_better
    if bad:
        return "НЕ КАТИТЬ", reasons
    if r.p_value < alpha and good:
        reasons.insert(0, f"Основная метрика «{m.name}» значимо улучшилась: {r.rel_effect:+.2%} "
                          f"(95% ДИ {r.rel_ci[0]:+.2%} … {r.rel_ci[1]:+.2%}).")
        return "КАТИТЬ", reasons
    if r.p_value < alpha:
        reasons.insert(0, f"Основная метрика «{m.name}» значимо ухудшилась: {r.rel_effect:+.2%}.")
        return "НЕ КАТИТЬ", reasons
    reasons.insert(0, f"Изменение основной метрики «{m.name}» ({r.rel_effect:+.2%}) "
                      f"статистически не значимо (p = {r.p_value:.3g}).")
    return "НЕТ ЗНАЧИМОГО ЭФФЕКТА", reasons
