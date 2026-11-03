# Methods

All methods return the absolute effect (treatment − control), the relative effect
(treatment / control − 1), confidence intervals for both and a two-sided p-value.

## Welch t-test — `hb.welch(control, treatment)`
Difference in means with unequal variances. Valid for skewed metrics such as revenue
once each group has a few thousand units (central limit theorem); the simulations in
`validation/` confirm a 5% false positive rate on zero-inflated revenue.

The relative interval uses Fieller's method: it accounts for the uncertainty of the
control mean and excludes 0 exactly when the p-value is below α.

## Proportions — `hb.proportions(successes_c, n_c, successes_t, n_t)`
z-test for conversion. The p-value uses the pooled variance (correct under the null
hypothesis), the intervals use unpooled variances.

## Delta method for ratio metrics — `hb.ratio(num_c, den_c, num_t, den_t)`
For a metric R = Σx / Σy where the randomization unit is the user but x and y are
counted per event (clicks and views, revenue and orders). Inputs are per user.

    Var(R) ≈ (1/n) · [ Var(x)/μy² − 2·μx·Cov(x, y)/μy³ + μx²·Var(y)/μy⁴ ]

A "per-view" test that treats every view as independent ignores that one user's views
are correlated; in the validation it gives about twice the promised false positive rate.

## CUPED — `hb.cuped(y_c, y_t, x_c, x_t)`
Variance reduction with a covariate x measured **before** the experiment (usually the
same metric in the previous weeks):

    y' = y − θ·(x − mean(x)),   θ = Cov(y, x) / Var(x)   (pooled over both groups)

The estimate stays unbiased; the variance drops by the factor 1 − ρ², ρ = corr(y, x).
Equivalent to "the same power with (1 − ρ²) of the users". The covariate must not be
affected by the treatment — never use something measured after the start.

## Bootstrap — `hb.bootstrap(control, treatment, statistic=np.median)`
Percentile bootstrap for statistics without a simple formula (median, 90th percentile).
The p-value is approximate (interval inversion). For means prefer Welch: it is exact
enough and much faster.

## Multiple testing — `hb.holm(p_values)`, `hb.benjamini_hochberg(p_values)`
With m metrics and no correction, P(at least one false positive) = 1 − (1 − α)^m:
40% for 10 metrics. Holm controls that probability (family-wise error); Benjamini–Hochberg
controls the share of false discoveries and is less strict. `analyze` applies the chosen
correction to secondary metrics.

## Sequential testing — `hb.msprt(control, treatment, looks=20)`
Mixture SPRT (Johari et al., 2017): always-valid p-values and confidence intervals. The
data may be checked after every batch and the test stopped at any moment without
inflating the error rate. Costs some power compared with a single analysis at the end.
`tau` is the scale of plausible effects; the result is valid for any value.

## SRM — `hb.srm([n_c, n_t], [share_c, share_t])`
Chi-square goodness-of-fit test of group sizes against the planned split. Threshold
p < 0.001. A failed SRM check invalidates the comparison; find the cause first.

## Design — `hb.sample_size`, `hb.sample_size_proportion`, `hb.mde`, `hb.duration_days`

    n per group = (z_{1−α/2} + z_{power})² · σ² · (1 − ρ²) · 2 / δ²

σ is the per-unit standard deviation of the metric, δ the absolute effect to detect,
ρ² the variance reduction from CUPED (0 without it).
