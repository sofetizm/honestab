"""Command line interface.

  honestab analyze data.csv --config experiment.yaml [--json] [--out report.md]
  honestab design --kind proportion --baseline 0.05 --mde-rel 0.05 --daily 20000
  honestab design --kind mean --baseline 250 --std 900 --mde-rel 0.03 --variance-reduction 0.3
  honestab srm 50120 49710 [--shares 0.5 0.5]
  honestab example data.csv --revenue-lift 0.05
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from .checks import srm
from .datasets import make_experiment
from .design import duration_days, sample_size, sample_size_proportion
from .experiment import Metric, analyze

CONFIG_KEYS = {"group_col", "control", "treatment", "alpha", "expected_split", "correction",
               "srm_threshold", "metrics"}

EXAMPLE_CONFIG = """\
group_col: group
control: A
treatment: B
alpha: 0.05
metrics:
  - name: revenue per user
    column: revenue
    covariate: pre_revenue
    role: primary
  - name: conversion
    column: converted
    kind: proportion
  - name: ctr
    numerator: clicks
    denominator: views
  - name: sessions per user
    column: sessions
    role: guardrail
"""


def load_config(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        cfg = json.loads(text)
    else:
        import yaml

        cfg = yaml.safe_load(text)
    unknown = set(cfg) - CONFIG_KEYS
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}; allowed: {sorted(CONFIG_KEYS)}")
    if not cfg.get("metrics"):
        raise SystemExit("config must list metrics")
    return cfg


def _json_safe(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_json_safe(v) for v in x]
    return x


def cmd_analyze(args) -> int:
    cfg = load_config(args.config)
    df = pd.read_csv(args.data)
    metrics = [Metric(**m) for m in cfg.pop("metrics")]
    report = analyze(df, metrics, **cfg)
    if args.json:
        out = json.dumps(_json_safe({
            "decision": report.decision, "reasons": report.reasons,
            "srm": {"ok": report.srm.ok, "p_value": report.srm.p_value,
                    "observed": report.srm.observed},
            "mde_primary_rel": report.mde_primary,
            "metrics": [dict(r.to_dict(), role=m.role) for m, r in
                        zip(report.metrics, report.results, strict=True)],
        }), ensure_ascii=False, indent=2)
    else:
        out = report.to_markdown()
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
    print(out)
    return 0


def cmd_design(args) -> int:
    if args.kind == "proportion":
        n = sample_size_proportion(args.baseline, args.mde, mde_rel=args.mde_rel,
                                   alpha=args.alpha, power=args.power, ratio=args.ratio)
    else:
        if args.std is None:
            raise SystemExit("--std is required for --kind mean")
        n = sample_size(args.std, args.mde, mde_rel=args.mde_rel, baseline=args.baseline,
                        alpha=args.alpha, power=args.power, ratio=args.ratio,
                        variance_reduction=args.variance_reduction,
                        std_treatment=args.std_treatment)
    print(f"Units per group: control {n['control']:,}, treatment {n['treatment']:,}; "
          f"total {n['total']:,}")
    if args.daily:
        print(f"Duration: {duration_days(n['total'], args.daily, args.share)} days "
              f"at {args.daily:,.0f} units/day, {args.share:.0%} of traffic in the experiment "
              "(at least one full week)")
    return 0


def cmd_srm(args) -> int:
    res = srm(args.counts, args.shares, args.threshold)
    print(res)
    return 0 if res.ok else 1


def cmd_example(args) -> int:
    df = make_experiment(args.users, revenue_lift=args.revenue_lift,
                         conversion_lift=args.conversion_lift, ctr_lift=args.ctr_lift,
                         srm_loss=args.srm_loss, seed=args.seed)
    df.to_csv(args.path, index=False)
    cfg = Path(args.path).with_suffix(".yaml")
    cfg.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    print(f"wrote {args.path} ({len(df):,} users) and {cfg}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="honestab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyse an experiment: one CSV row per unit")
    a.add_argument("data")
    a.add_argument("--config", required=True, help="YAML or JSON with metrics (see README)")
    a.add_argument("--json", action="store_true")
    a.add_argument("--out")
    a.set_defaults(func=cmd_analyze)

    d = sub.add_parser("design", help="sample size and duration")
    d.add_argument("--kind", choices=["mean", "proportion"], default="mean")
    d.add_argument("--baseline", type=float, help="current mean or conversion rate")
    d.add_argument("--std", type=float, help="standard deviation of the metric (kind=mean)")
    d.add_argument("--std-treatment", type=float,
                   help="expected std in the treatment group (default: same as --std)")
    d.add_argument("--mde", type=float, help="absolute effect to detect")
    d.add_argument("--mde-rel", type=float, help="relative effect to detect, e.g. 0.03")
    d.add_argument("--alpha", type=float, default=0.05)
    d.add_argument("--power", type=float, default=0.8)
    d.add_argument("--ratio", type=float, default=1.0, help="treatment / control size")
    d.add_argument("--variance-reduction", type=float, default=0.0, help="CUPED: rho^2")
    d.add_argument("--daily", type=float, help="eligible units per day")
    d.add_argument("--share", type=float, default=1.0, help="share of traffic in the test")
    d.set_defaults(func=cmd_design)

    s = sub.add_parser("srm", help="sample ratio mismatch check")
    s.add_argument("counts", type=int, nargs="+")
    s.add_argument("--shares", type=float, nargs="+")
    s.add_argument("--threshold", type=float, default=0.001)
    s.set_defaults(func=cmd_srm)

    e = sub.add_parser("example", help="write a synthetic experiment CSV and its config")
    e.add_argument("path")
    e.add_argument("--users", type=int, default=20_000)
    e.add_argument("--revenue-lift", type=float, default=0.0)
    e.add_argument("--conversion-lift", type=float, default=0.0)
    e.add_argument("--ctr-lift", type=float, default=0.0)
    e.add_argument("--srm-loss", type=float, default=0.0)
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_example)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
