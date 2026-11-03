---
name: ab-test-analysis
description: Analyse A/B test results or design an experiment with statistical rigour (SRM check, CUPED, delta method for ratio metrics, multiple testing, peeking, sample size) using the honestab library. Use when asked to analyse an experiment, compare a control and a treatment group, decide whether to ship a variant, or compute sample size, MDE or test duration.
---

# A/B test analysis

Most wrong experiment conclusions do not come from the wrong formula but from skipped
checks: lost users, a metric computed per event instead of per user, ten metrics tested
without correction, results checked every day. Follow the steps in order.

## 0. Tools
```bash
python -c "import honestab" 2>/dev/null || pip install honestab
```
If installation from PyPI fails, use `pip install git+https://github.com/sofetizm/honestab`.
The CLI is `honestab`; the Python API is `import honestab as hb`.

## 1. Clarify the experiment before computing anything
Find out (from the user, the data or the docs) and write down:
- the **hypothesis** and the **one primary metric** that decides;
- **guardrail** metrics that must not get worse (errors, latency, cancellations...);
- the **randomization unit** (user, device, session) and the planned **split**;
- whether results were **checked during the test** and the test stopped early.

If something is unknown, state your assumption explicitly in the report. Do not pick the
primary metric after looking at the results.

## 2. Get the data into one row per randomization unit
- If the data is per event (views, sessions, orders), **aggregate to one row per unit**
  first. Never run a test where rows are events: events of one user are correlated and
  the p-values come out far too small.
- A metric that is a ratio of event counts (CTR = clicks / views, average check = revenue /
  orders) becomes two columns per unit — numerator and denominator — and is analysed with
  the delta method (`kind: ratio`).
- Check that no unit appears in both groups; report how many if any (and exclude them).
- Missing values: decide explicitly (0 for "did nothing", or exclude) and say which.

## 3. Check sample ratio mismatch first
```bash
honestab srm <n_control> <n_treatment> [--shares 0.5 0.5]
```
If p < 0.001, **stop**: users were lost or added non-randomly and every metric comparison
is biased. Report the mismatch and possible causes (assignment, logging, filters,
redirects, bots) instead of interpreting metrics.

## 4. Choose the method for each metric
| Metric | `kind` | Method |
|---|---|---|
| per-user value (revenue, orders, time) | `mean` | Welch t-test |
| same, with the value of the metric **before** the test | `mean` + `covariate` | CUPED |
| share of users who did something | `proportion` | z-test |
| ratio of sums (CTR, average check, revenue per session) | `ratio` | delta method |
| median, percentile | `median` | bootstrap |

Details and formulas: `references/methods.md`.

## 5. Run the analysis
Write a config and run it:
```yaml
group_col: group        # column with the variant
control: A
treatment: B
expected_split: 0.5     # planned share of the treatment group
metrics:
  - {name: revenue per user, column: revenue, covariate: pre_revenue, role: primary}
  - {name: conversion, column: converted, kind: proportion}
  - {name: ctr, numerator: clicks, denominator: views}
  - {name: errors per user, column: errors, role: guardrail, higher_is_better: false}
```
```bash
honestab analyze data.csv --config experiment.yaml --json
```
Exactly one metric is `primary`; `secondary` metrics get a Holm correction automatically;
`guardrail` metrics block shipping if they get significantly worse.

## 6. Interpret honestly
- Report the **effect with its confidence interval**, not just the p-value:
  "+3.1% (95% CI +0.4% … +5.8%)".
- Not significant means "no effect detected", not "no effect". Always state the
  **MDE** of the experiment (it is in the report): effects smaller than that could be missed.
- Never say "trending towards significance" or re-test with another method until p < 0.05.
- If the test was **stopped early** because the result looked significant, the p-value is
  not valid. Say so; with data in arrival order use `hb.msprt` (always-valid p-values).
- Segment breakdowns (country, platform) are **exploratory**: many segments → false
  positives. Report them as hypotheses for a new test, never as the conclusion.
- Outliers: only a rule fixed in advance and applied to both groups equally (e.g.
  winsorize at the pooled 99.9th percentile). Never remove "strange" users from one group.

Common traps with numbers: `references/pitfalls.md`.

## 7. Report
Use `references/report-template.md`: decision first, then the primary metric with its CI,
guardrails, secondary metrics, checks performed (SRM, data issues), MDE, assumptions and
limitations.

## Designing an experiment
```bash
honestab design --kind proportion --baseline 0.05 --mde-rel 0.05 --daily 20000
honestab design --kind mean --baseline 280 --std 1100 --mde-rel 0.05 --variance-reduction 0.25 --daily 20000
```
- Use the **standard deviation of the metric per unit** from historical data.
- With a pre-period covariate, pass `--variance-reduction` = ρ² (correlation of the
  metric with its pre-period value, squared) — CUPED needs fewer users.
- For a multiplicative effect on a heavy-tailed metric (revenue) the spread of the
  treatment group grows too: pass `std_treatment=std*(1+lift)` to `hb.sample_size` and
  keep a margin of 10–20%.
- Run at least **one full week** to cover weekly seasonality, even if the sample size is
  reached sooner.
- Fix the primary metric, guardrails, sample size and duration **before** the start.
