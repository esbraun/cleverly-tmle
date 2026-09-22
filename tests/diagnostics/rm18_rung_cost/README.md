# RM18 multi-arm rung design cost

This directory holds the computation behind "What the multi-arm rung design would cost" in RM18
of `docs/roadmap.md`. `RM18-slopes` sanctions a rung design for the two multi-arm DR-TMLE
contraction slopes, and this row defers its run. The code sizes that design under the rule that
`CONTRACTION_REPLICATES` in `tests/studies/drtmle_properties.py` declares for the binary ladder.
It fits no estimator, and it changes no study, verdict or committed row.

| file | what it does |
| --- | --- |
| `cost.py` | reads `tests/canonical/multi_arm_drtmle/properties.csv`, computes the delta-method and surrogate costs, and writes one row per input, check and cost |
| `cost.csv` | the committed output |

`tests/unit/test_rm18_rung_cost_diagnostic.py` runs in the fast tier. It recomputes each input
and each delta-method figure longhand from `properties.csv`. It checks that each rounded budget
reaches a half-width of one, and that one replication fewer does not. It evaluates the surrogate
at the recorded budget and one grid step below it. It also checks the binary rows against the
projections that the binary rule records.

## Run

`--output` is required, so a bare run cannot overwrite the committed record. The seeds are fixed,
so a rerun gives the same bytes.

```bash
python -m tests.diagnostics.rm18_rung_cost.cost --output <scratch>/cost.csv
cmp <scratch>/cost.csv tests/diagnostics/rm18_rung_cost/cost.csv
```

## The rule

The rule reads the first-rung bias `b1` of each positive arm, and the spread `c` of the
`both_wrong` control. `c` is the mean of `empirical_se * sqrt(n)` over the three control rungs.
The rule carries the bias down the ladder as `1/n` and the spread as `1/sqrt(n)`. It does not
read the slope, the interval or the verdict of either positive rate cell. The ladder runs
n = 2,000, 4,000 and 8,000. The middle rung keeps 600 replications, because its centred weight
is zero.

| method | how it finds the smallest `R` at each outer rung |
| --- | --- |
| delta method | `R = (z * c / (log 4 * b1 * n1))^2 * (n1 + n3)`, rounded up. `z` is the 99.5% normal quantile, `n1` is 2,000 and `n3` is 8,000 |
| surrogate | draws each rung's bias from a normal law with the rule's mean and spread, 40,000 times. It fits the slope of `log |bias|` on `log n`, and reads half the 99% range of that slope. It searches `R` in steps of 1,000, at three fixed seeds |

## The recorded result

| arm | first-rung bias | smallest `R`, delta method | smallest `R`, surrogate at three seeds |
| --- | --- | --- | --- |
| `outcome_correct` | 0.001127 | 9,240, from 9,239.22 | 19,000, 20,000 and 19,000 |
| `treatment_correct` | 0.000585 | 34,267, from 34,266.75 | 71,000, 73,000 and 70,000 |

The control spread `c` is 1.1660. On the binary inputs, the surrogate gives half-widths of 1.93,
1.00 and 0.84 at 800, 2,000 and 2,400 replications. The binary rule records 2.022, 1.049 and
0.876, and each value here is within 5% of those. The delta method gives 0.64 at 2,400, so it
understates the surrogate width when a rung's bias is near its Monte Carlo error.

Both first-rung bias intervals cover zero. The needed `R` grows as the inverse square of the
true bias, so these figures are not an upper bound on the cost.
