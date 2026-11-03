"""Validation by simulation: do the methods keep their promises?

A statistical method makes two promises:
  1. With no real effect (an A/A test) it reports a significant result in α = 5% of cases.
  2. With a real effect it finds it as often as the design calculation says (power).
This script checks both on thousands of simulated experiments and also reproduces common
mistakes (peeking, per-view tests of CTR, many metrics without correction) to show how
much they break promise 1.

  python validation/run.py            # full run, ~10–15 minutes, writes validation/
  python validation/run.py --quick    # small run for a smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats

import honestab as hb
from honestab.datasets import make_experiment

OUT = Path(__file__).resolve().parent
ALPHA = 0.05


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rate(hits: list[bool]) -> dict:
    k, n = int(np.sum(hits)), len(hits)
    lo, hi = wilson(k, n)
    return {"rate": k / n, "lo": lo, "hi": hi, "n": n}


def split(df):
    return df[df.group == "A"], df[df.group == "B"]


# ------------------------------------------------------------------- experiments
def aa_methods(sims: int, n_users: int, seed: int) -> dict:
    """A/A: every method should be significant in ~5% of runs."""
    hits = {k: [] for k in ("Welch: выручка", "CUPED: выручка", "z-тест: конверсия",
                            "дельта-метод: CTR", "z-тест по просмотрам: CTR", "бутстреп: медиана времени")}
    for i in range(sims):
        a, b = split(make_experiment(n_users, seed=seed + i))
        hits["Welch: выручка"].append(hb.welch(a.revenue, b.revenue).p_value < ALPHA)
        hits["CUPED: выручка"].append(
            hb.cuped(a.revenue, b.revenue, a.pre_revenue, b.pre_revenue).p_value < ALPHA)
        hits["z-тест: конверсия"].append(
            hb.proportions(a.converted.sum(), len(a), b.converted.sum(), len(b)).p_value < ALPHA)
        hits["дельта-метод: CTR"].append(
            hb.ratio(a.clicks, a.views, b.clicks, b.views).p_value < ALPHA)
        hits["z-тест по просмотрам: CTR"].append(
            hb.proportions(a.clicks.sum(), a.views.sum(), b.clicks.sum(), b.views.sum()).p_value < ALPHA)
        if i < max(100, sims // 5):
            hits["бутстреп: медиана времени"].append(
                hb.bootstrap(a.time_spent, b.time_spent, n_resamples=1000, seed=i).p_value < ALPHA)
    return {k: rate(v) for k, v in hits.items()}


def peeking(sims: int, n_per_group: int, looks: int, seed: int) -> dict:
    """A/A with a look after every 1/looks of the data; stop at the first p < α."""
    rng = np.random.default_rng(seed)
    naive_by_looks = {k: [] for k in (1, 2, 5, 10, 20, looks)}
    seq = []
    points = [round(n_per_group * (k + 1) / looks) for k in range(looks)]
    for _ in range(sims):
        c = rng.lognormal(0, 1.2, n_per_group)
        t = rng.lognormal(0, 1.2, n_per_group)
        cs, ts, cs2, ts2 = np.cumsum(c), np.cumsum(t), np.cumsum(c * c), np.cumsum(t * t)
        pvals = []
        for n in points:
            mc, mt = cs[n - 1] / n, ts[n - 1] / n
            vc = (cs2[n - 1] / n - mc * mc) * n / (n - 1)
            vt = (ts2[n - 1] / n - mt * mt) * n / (n - 1)
            z = (mt - mc) / math.sqrt((vc + vt) / n)
            pvals.append(2 * stats.norm.sf(abs(z)))
        for k in naive_by_looks:
            idx = [round(looks * (j + 1) / k) - 1 for j in range(k)]
            naive_by_looks[k].append(min(pvals[j] for j in idx) < ALPHA)
        seq.append(hb.msprt(c, t, looks=points).stopped_at is not None)
    return {"naive_by_looks": {k: rate(v) for k, v in naive_by_looks.items()},
            "msprt": rate(seq), "looks": looks}


def multiple_metrics(sims: int, m: int, n: int, seed: int) -> dict:
    """A/A with m independent metrics: chance of at least one false positive."""
    rng = np.random.default_rng(seed)
    raw, holm, bh = [], [], []
    for _ in range(sims):
        a, b = rng.normal(size=(m, n)), rng.normal(size=(m, n))
        p = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
        raw.append((p < ALPHA).any())
        holm.append((hb.holm(p) < ALPHA).any())
        bh.append((hb.benjamini_hochberg(p) < ALPHA).any())
    return {"metrics": m, "без поправки": rate(raw), "Holm": rate(holm),
            "Benjamini–Hochberg": rate(bh)}


def power(sims: int, n_users: int, lift: float, seed: int) -> dict:
    """Real effect: observed power vs the design formula, with and without CUPED.

    A multiplicative effect also inflates the variance of the treatment group
    (std × (1 + lift)); the textbook formula assumes equal variances and overstates power.
    Both versions of the formula are reported."""
    ref = make_experiment(400_000, seed=seed - 1)
    ref_b = make_experiment(400_000, revenue_lift=lift, split=1.0, seed=seed - 2)
    va, vb = ref.revenue.var(), ref_b.revenue.var()
    rho2 = np.corrcoef(ref.revenue, ref.pre_revenue)[0, 1] ** 2
    delta = ref.revenue.mean() * lift
    n_per_group = n_users / 2
    z = stats.norm.ppf(1 - ALPHA / 2)

    def theory(vr: float, var_sum: float) -> float:
        se = math.sqrt((1 - vr) * var_sum / n_per_group)
        return float(stats.norm.sf(z - delta / se) + stats.norm.cdf(-z - delta / se))

    w, c = [], []
    for i in range(sims):
        a, b = split(make_experiment(n_users, revenue_lift=lift, seed=seed + i))
        w.append(hb.welch(a.revenue, b.revenue).p_value < ALPHA)
        c.append(hb.cuped(a.revenue, b.revenue, a.pre_revenue, b.pre_revenue).p_value < ALPHA)
    std = math.sqrt(va)
    return {"lift": lift, "n_users": n_users, "rho2": float(rho2),
            "welch": {**rate(w), "theory_equal_var": theory(0.0, 2 * va),
                      "theory": theory(0.0, va + vb)},
            "cuped": {**rate(c), "theory_equal_var": theory(rho2, 2 * va),
                      "theory": theory(rho2, va + vb)},
            "sample_size_for_80%": {
                "welch": hb.sample_size(std, delta)["total"],
                "cuped": hb.sample_size(std, delta, variance_reduction=rho2)["total"]}}


def sequential_power(sims: int, n_per_group: int, effect: float, seed: int) -> dict:
    """The price of peeking safely: power of one analysis at the end vs mSPRT with 20 looks."""
    rng = np.random.default_rng(seed)
    fixed, seq, stop_share = [], [], []
    points = [round(n_per_group * (k + 1) / 20) for k in range(20)]
    for _ in range(sims):
        c = rng.normal(0, 1, n_per_group)
        t = rng.normal(effect, 1, n_per_group)
        fixed.append(hb.welch(c, t).p_value < ALPHA)
        res = hb.msprt(c, t, looks=points, tau=effect)
        seq.append(res.stopped_at is not None)
        if res.stopped_at is not None:
            stop_share.append((res.stopped_at + 1) / 20)
    return {"fixed": rate(fixed), "msprt": rate(seq),
            "msprt_mean_stop": float(np.mean(stop_share)) if stop_share else math.nan}


def srm_check(sims: int, n_users: int, seed: int) -> dict:
    """A/A where a "logging bug" loses part of the light users of group B.
    The metric comparison then shows a fake effect; the SRM check is what catches it."""
    alarm_ok, alarm_bug, fake_ok, fake_bug = [], [], [], []
    for i in range(sims):
        for loss, alarm, fake in ((0.0, alarm_ok, fake_ok), (0.3, alarm_bug, fake_bug)):
            df = make_experiment(n_users, srm_loss=loss, seed=seed + i)
            alarm.append(not hb.srm(df.group.value_counts().reindex(["A", "B"])).ok)
            a, b = split(df)
            fake.append(hb.welch(a.sessions, b.sessions).p_value < ALPHA)
    return {"группы разделены правильно": {"srm_alarm": rate(alarm_ok), "fake_effect": rate(fake_ok)},
            "в B потеряно 30% пользователей с одной сессией": {"srm_alarm": rate(alarm_bug),
                                                               "fake_effect": rate(fake_bug)}}


def ci_coverage(sims: int, seed: int) -> dict:
    """Share of 95% CIs that contain the true relative effect (should be ~95%)."""
    rng = np.random.default_rng(seed)
    true_rel = 0.05
    cover_w, cover_r = [], []
    for _ in range(sims):
        a = rng.lognormal(0, 1, 5000)
        b = rng.lognormal(0, 1, 5000) * (1 + true_rel)
        lo, hi = hb.welch(a, b).rel_ci
        cover_w.append(lo <= true_rel <= hi)
        pa, pb = 0.10, 0.10 * (1 + true_rel)
        ca, cb = rng.binomial(20000, pa), rng.binomial(20000, pb)
        lo, hi = hb.proportions(ca, 20000, cb, 20000).rel_ci
        cover_r.append(lo <= true_rel <= hi)
    return {"Welch, относительный эффект": rate(cover_w),
            "z-тест долей, относительный эффект": rate(cover_r)}


# ------------------------------------------------------------------------ output
def pct(r: dict) -> str:
    return f"{r['rate']:.1%} [{r['lo']:.1%}; {r['hi']:.1%}]"


def markdown(res: dict) -> str:
    L = ["# Валидация методов симуляциями", "",
         f"Сгенерировано: {res['meta']['date']}, {res['meta']['minutes']:.1f} мин. "
         "В скобках — 95% доверительный интервал (Вильсон) для доли.", "",
         "## 1. A/A-тесты: доля ложных срабатываний должна быть ≈ 5%", "",
         "| Метод | Ложные срабатывания | Симуляций |", "|---|---|---|"]
    for k, v in res["aa"].items():
        flag = " ⚠️" if v["lo"] > 0.05 else ""
        L.append(f"| {k}{flag} | {pct(v)} | {v['n']} |")
    pk = res["peeking"]
    L += ["", f"## 2. Подглядывание: проверка после каждой из {pk['looks']} порций данных", "",
          "| Сколько раз смотрим | Обычный t-тест, остановка при p < 0.05 | Симуляций |",
          "|---|---|---|"]
    for k, v in pk["naive_by_looks"].items():
        L.append(f"| {k} | {pct(v)} | {v['n']} |")
    L.append(f"| {pk['looks']}, mSPRT (always-valid p-value) | **{pct(pk['msprt'])}** | "
             f"{pk['msprt']['n']} |")
    mm = res["multiple"]
    L += ["", f"## 3. {mm['metrics']} метрик без эффекта: хотя бы одно ложное срабатывание", "",
          "| Поправка | Вероятность | Симуляций |", "|---|---|---|"]
    for k in ("без поправки", "Holm", "Benjamini–Hochberg"):
        L.append(f"| {k} | {pct(mm[k])} | {mm[k]['n']} |")
    pw = res["power"]
    L += ["", f"## 4. Мощность: настоящий эффект +{pw['lift']:.0%} к выручке, "
              f"{pw['n_users']:,} пользователей", "",
          f"Доля дисперсии, которую убирает CUPED (ρ²): {pw['rho2']:.2f}.", "",
          "| Метод | Мощность в симуляции | Формула с ростом дисперсии в B | "
          "Учебная формула (дисперсии равны) |",
          "|---|---|---|---|"]
    for k in ("welch", "cuped"):
        L.append(f"| {k} | {pct(pw[k])} | {pw[k]['theory']:.1%} | {pw[k]['theory_equal_var']:.1%} |")
    L += ["", "Мультипликативный эффект увеличивает и разброс метрики в тестовой группе; учебная "
              "формула этого не учитывает и завышает мощность t-теста (в `hb.sample_size` для "
              "этого есть параметр `std_treatment`). На метриках с тяжёлыми хвостами, как выручка, "
              "и стандартное отклонение, и ρ² оцениваются неточно, поэтому формула мощности — "
              "ориентир с точностью до нескольких процентных пунктов. Вывод для дизайна: "
              "закладывайте запас по размеру выборки."]
    sp = res["sequential_power"]
    L += ["", "## 4б. Цена безопасного подглядывания", "",
          "Настоящий эффект, обычный тест рассчитан на мощность ~80%.", "",
          "| Анализ | Нашёл эффект | Симуляций |", "|---|---|---|",
          f"| t-тест один раз в конце | {pct(sp['fixed'])} | {sp['fixed']['n']} |",
          f"| mSPRT, 20 проверок | {pct(sp['msprt'])} | {sp['msprt']['n']} |", "",
          f"Когда mSPRT находит эффект, он останавливается в среднем на "
          f"{sp['msprt_mean_stop']:.0%} запланированной длительности."]
    L += ["", "## 5. SRM: потеря части пользователей создаёт фальшивый эффект", "",
          "A/A-тест (эффекта нет). «Баг логирования» теряет часть пользователей с одной сессией "
          "в группе B.", "",
          "| Ситуация | Метрика «сессий на пользователя» значимо выросла | Сработала проверка SRM |",
          "|---|---|---|"]
    for k, v in res["srm"].items():
        L.append(f"| {k} | {pct(v['fake_effect'])} | {pct(v['srm_alarm'])} |")
    L += ["", "## 6. Покрытие доверительных интервалов (должно быть ≈ 95%)", "",
          "| Интервал | Содержит истинный эффект | Симуляций |", "|---|---|---|"]
    for k, v in res["coverage"].items():
        L.append(f"| {k} | {pct(v)} | {v['n']} |")
    L += ["", "![peeking](peeking.png)", "", "![false positives](aa.png)"]
    return "\n".join(L) + "\n"


def plots(res: dict, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pk = res["peeking"]
    ks = sorted(int(k) for k in pk["naive_by_looks"])
    ys = [pk["naive_by_looks"][k]["rate"] * 100 for k in ks]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ks, ys, "o-", color="#cf222e", label="обычный t-тест + остановка при p < 0.05")
    ax.axhline(pk["msprt"]["rate"] * 100, color="#1f6feb", lw=2,
               label=f"mSPRT, {pk['looks']} проверок")
    ax.axhline(5, color="#57606a", ls="--", lw=1, label="обещанные 5%")
    ax.set_xticks(ks)
    ax.set_xlabel("Сколько раз смотрим на результаты")
    ax.set_ylabel("Ложные срабатывания, %")
    ax.set_ylim(0, max(ys) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "peeking.png", dpi=150)
    plt.close(fig)

    aa = res["aa"]
    names = list(aa)
    vals = [aa[k]["rate"] * 100 for k in names]
    err = [[(aa[k]["rate"] - aa[k]["lo"]) * 100 for k in names],
           [(aa[k]["hi"] - aa[k]["rate"]) * 100 for k in names]]
    colors = ["#cf222e" if aa[k]["lo"] > 0.05 else "#1f6feb" for k in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(names, vals, xerr=err, color=colors, capsize=3)
    ax.axvline(5, color="#57606a", ls="--", lw=1)
    ax.set_xlabel("Ложные срабатывания в A/A-тестах, % (должно быть 5%)")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "aa.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = 0.1 if args.quick else 1.0
    started = time.time()
    res = {}
    steps = [
        ("aa", lambda: aa_methods(int(2000 * s), 10_000, 1000)),
        ("peeking", lambda: peeking(int(2000 * s), 5000, 20, 2)),
        ("multiple", lambda: multiple_metrics(int(4000 * s), 10, 500, 3)),
        ("power", lambda: power(int(1000 * s), 40_000, 0.10, 50_000)),
        ("sequential_power", lambda: sequential_power(int(1000 * s), 5000, 0.056, 70_000)),
        ("srm", lambda: srm_check(int(500 * s), 20_000, 90_000)),
        ("coverage", lambda: ci_coverage(int(2000 * s), 4)),
    ]
    for name, fn in steps:
        t0 = time.time()
        res[name] = fn()
        print(f"{name:<9} done in {time.time() - t0:5.1f}s", flush=True)
    res["meta"] = {"date": time.strftime("%Y-%m-%d"), "minutes": (time.time() - started) / 60,
                   "quick": args.quick}
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out / "VALIDATION.md").write_text(markdown(res), encoding="utf-8")
    plots(res, out)
    print(markdown(res))


if __name__ == "__main__":
    main()
