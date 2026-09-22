# RM18 one-sided robustness reading

This directory holds the reading that RM18 of `docs/roadmap.md` declares in "The one-sided
robustness reading, declared before it is computed". That subsection fixes the four statistics,
the binary rule and the multi-arm rule. The code here applies them to committed rows. It fits no
estimator and changes no verdict.

| file | what it does |
| --- | --- |
| `read.py` | reads `tests/canonical/drtmle/replicates.csv.gz`, `tests/canonical/multi_arm_drtmle/replicates.csv.gz` and `tests/canonical/multi_arm_drtmle/property-replicates.csv.gz`. It writes one row per statistic |
| `readings.csv` | the committed output. Each declared row carries the reading of its configuration. The binary `both_correct` row enters both binary readings and carries no reading of its own. The `scope` column marks each row `declared` or `supplementary` |

`tests/unit/test_rm18_one_sided_bias_diagnostic.py` rebuilds `readings.csv` from the committed
artifacts in the fast tier. It also checks each rule case and the Welch interval against SciPy.
It checks each statistic against the interval its study publishes. Six in-memory mutations shift
the R rows or the property cell, and each one moves the reading. It also checks each
supplementary Welch interval against SciPy, each Bonferroni width against the Student quantiles,
and the rung bias against the interval its study publishes.

## Run

`--output` is required, so a bare run cannot overwrite the committed record.

```bash
python -m tests.diagnostics.rm18_one_sided_bias.read --output <scratch>/readings.csv
cmp <scratch>/readings.csv tests/diagnostics/rm18_one_sided_bias/readings.csv
```

## Statistics

Each interval is a 99% interval. The bias of one row is `estimate - truth`.

| statistic | rows | interval |
| --- | --- | --- |
| (i) | `cleverly` on the primary configuration: binary `ate` at n = 3,000, multi-arm `ate[medium vs high]` at n = 2,000 | Student, df = replications - 1 |
| (ii) | `drtmle-r` on the same binary rows | Student |
| (iii) | `cleverly` minus `drtmle-r`, paired on `replicate` | Student |
| `property-cell bias` | multi-arm `double_robustness/treatment_correct`, n = 2,000, 600 replications | Student. The declaration names no such statistic. It is the first term of (iv), and it equals the committed `bias_ci_lower` and `bias_ci_upper` of that cell |
| (iv) | `property-cell bias` minus multi-arm (i) | Welch, with the Welch-Satterthwaite df |

Each of (i) and (ii) equals the `bias_ci_lower` and `bias_ci_upper` of its row in the study's
committed `performance-tests.csv`. Each binary (iii) equals the `paired_ci_lower` and
`paired_ci_upper` of its `ate` row in `tests/canonical/drtmle/equivalence.csv`.

## How the code resolves the rule

| point | resolution |
| --- | --- |
| "on the side of (i)" | the interval excludes zero, and its sign equals the sign of the (i) point estimate |
| order of the tests | an (i) interval that covers zero, or a `both_correct` (iii) interval that excludes zero, reads `unresolved` first. The code then tries `shared`, `estimator-specific` and `mixed` |
| (ii) or (iii) on the side opposite to (i) | `unresolved`, because it matches no named pattern |

## A difference from the planning read

The declaration discloses a planning-read Welch interval of about -0.0003 to 0.0077 for the
multi-arm cell. The declared statistic (iv) gives 0.001160 to 0.008650, which excludes zero. The
two intervals use different second samples. The planning read used the
`double_robust_contraction/treatment_correct_n2000` property cell, which gives -0.00030 to
0.00777. That cell has the same configuration: a misspecified outcome regression and a correct
treatment mechanism. Statistic (iv) uses the primary rows, where both nuisances are correct, as
the declaration requires. The declaration states that the declared statistics govern, so the
reading uses (iv).

## Supplementary rows

A review asked for these rows after the declared reading ran. The declaration names none of them.
Each one has `scope` set to `supplementary` and an empty `reading`, and no rule reads it. The
declared verdict `not consistent` stands as the declaration defines it.

| statistic | rows | 99% interval | why it is here |
| --- | --- | --- | --- |
| `rung bias` | `double_robust_contraction/treatment_correct_n2000`, 600 replications | -0.002274 to 0.003444 | the second sample of the multi-arm configuration at n = 2,000 |
| `rung bias minus (i)` | the rung against the primary rows, by Welch | -0.002581 to 0.004917 | the declared comparison, on the other sample of the same configuration |
| `property-cell bias minus rung bias` | the property cell against the rung, by Welch | -0.000296 to 0.007771 | whether the two samples of one configuration differ |
| `pooled bias minus (i)` | both samples, 1,200 replications, against the primary rows, by Welch | -0.000123 to 0.006196 | the declared comparison on all rows of the configuration |
| `(iii) at the Bonferroni level over three` | each binary paired difference, at the level 1 - 0.01/3 | `treatment_correct`: -0.000063 to 0.002073 | the declaration adjusts for no multiplicity over its three paired intervals |
| `(iii) minus outcome_correct (iii)` | the binary `treatment_correct` paired differences against the `outcome_correct` ones, by Welch | 0.000138 to 0.002055 | whether the increment differs between the two one-sided configurations |
| `(iii) minus both_correct (iii)` | the same, against `both_correct`, by Welch | 0.000067 to 0.001947 | whether the increment differs from the configuration where both nuisances are correct |

The Bonferroni intervals of `outcome_correct` and `both_correct` also cover zero. The binary
scenarios draw independent samples, so the direct contrasts use Welch. The multi-arm excess that
statistic (iv) reports is not reproduced by the rung, the other sample of the same configuration.
The rows do not establish that the configuration adds bias, and they do not establish that it
adds none.
