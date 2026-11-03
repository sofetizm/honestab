# Pitfalls

Numbers are from `validation/VALIDATION.md` (thousands of simulated A/A tests, where the
correct answer is always "no effect" and a correct method is wrong in 5% of cases).

| Mistake | What happens | Do instead |
|---|---|---|
| Checking the p-value every day and stopping at the first p < 0.05 | 20 looks → 24.5% false positives instead of 5% (mSPRT: 1.2%) | Fix the duration in advance, or use `msprt` |
| Testing CTR with one row per view | 9.8% false positives — twice the promised 5% (delta method: 4.8%) | One row per user, `kind: ratio` (delta method) |
| Reporting the best of 10 metrics | 39.6% chance that at least one is "significant" by luck (Holm: 4.8%) | One primary metric; Holm correction for the rest |
| Ignoring a sample ratio mismatch | Losing 30% of one-session users in one group made a fake "effect" in 99.8% of A/A tests | Run `srm` first; stop if p < 0.001 |
| Not using pre-experiment data | Power 57.5% instead of 75.3% with the same 40,000 users (ρ² = 0.29) | CUPED with the pre-period value of the metric |
| A CUPED covariate measured after the start | The treatment leaks into the covariate; biased effect | Only data from before randomization |
| "Not significant, so there is no effect" | The test may simply be too small | Report the MDE: "effects below X% would likely be missed" |
| Removing outliers from one group, or after seeing results | Biased, arbitrary results | A rule fixed in advance, applied to the pooled data |
| Declaring a winner in one segment | 10 segments ≈ 40% chance of a lucky one | Segment results are hypotheses for a new test |
| Units in both groups (e.g. users who changed devices) | Contaminated comparison, diluted effect | Count them, exclude, report |
| Running less than a full week | Weekday effects and novelty bias | At least 7 days, whole weeks |
| Stopping a test as soon as the sample size is reached on day 2 | Only heavy users are in the sample | Plan duration, not just sample size |
| Sample size from the control-group std only, for a multiplicative effect | The treatment group's spread grows too; real power is lower than planned | `std_treatment = std × (1 + lift)` in `hb.sample_size`, plus a safety margin |
