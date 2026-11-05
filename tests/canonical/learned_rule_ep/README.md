# Learned-rule value, EP rule learner

This directory holds the declared repeated-sampling study of P1 part (c). The subject is
`TMLE(learned_rule=LearnedRule(loss="ep", modifiers=Modifiers(["W1", "W2"], domains={"W1": (-1, 1)}, levels={"W2": (0, 1)})), n_folds=10, cv_evaluation=True, targeting_scheme="pooled", g_bounds="auto")`
with `repeats=1`. Each fold's rule is the sign of the EP curve of the effect given `(W1, W2)`,
fitted on the fold's training rows from the shipped 5-fold inner cross-fit. The default sieve's
size is chosen by cross-validation over the inner folds, with the nuisances refitted inside each.
The target is the fold-average value of the rules the fit learned. The study is declared, not
registered: no support row cites it until its run lands.

| part | declaration |
| --- | --- |
| scenarios | RM30 `non_exceptional` and `misspecified_limit` with the main-terms outcome learner and the logistic treatment learner (Q12: the RM30 GLMs), rows only. The primary phase fits n = 2,000 for 2,000 replications per law; the calibration cells fit n = 4,000 |
| limit | with the inner treatment learner correct, the EP risk is doubly robust and each fold curve converges to the additive projection `E[tau \| W1] + E[tau \| W2] - E[tau]` of the effect. Its sign changes once in `W1` for each `W2` (thresholds -0.0797 and -0.1203 at `non_exceptional`, 0.2900 and 0.2662 at `misspecified_limit`), with a share near zero linear in the distance, so C3 holds. Against `d0` it differs on 1.0% of the population and gives up 2.5e-5 of value, 0.002 SE at n = 4,000 |
| truth | rule L4: each replication refits each fold's rule through `fit_fold_rule` with the fold's recorded inner seed and the fit's resolved `g_bounds` and `alpha`, requires the fit's rule on every validation row, integrates each rule's value on the RM30 grid, and averages over the folds. The construction rests on the unit witnesses W-c1 to W-c3 of `tests/unit/test_learned_rule_ep.py` |
| properties | `interval_calibration`: `<law>__declared` for both laws and `optimal_known_g__declared` (the value of the optimal rule with `g0` supplied at `non_exceptional`, truth `Psi_d0 = 0.577115`), 7,200 draws each at n = 4,000, with the shrunken-SE (0.70) and noise controls of each, and the fold-locality control `non_exceptional__validation_rule_control` (each fold's EP curve fitted on its own validation rows, 500 draws at n = 2,000), whose coverage interval must lie below 0.92. `small_sample_diagnostic` fits each law at n = 1,000 for 2,000 draws and `exceptional_law_report` fits the RM30 `exceptional` law at n = 2,000 for 2,400 draws; both are role `diagnostic` and state no verdict. Every gated and diagnostic row also carries the paired plug-in fit on the same sample and folds |
| verdicts | the framework defaults, read on the error `estimate - truth` of each row: coverage floor 0.90 and SE screen 0.80 to 1.20 on the primary phase, SE-ratio band 0.93 to 1.07 and coverage band 0.92 to 0.98 on the calibration cells, 99% intervals |
| comparator | none |

The declarations are in `tests/studies/learned_rule_ep.py`,
`tests/studies/learned_rule_ep_properties.py` and `tests/studies/_learned_rule_ep_law.py`. The
red-cell readings are in the docstring of `regenerate.py`, declared before any run and before the
pilot's summary was read.

## Sizing

The sizing pilot (`tests/studies/learned_rule_ep_pilot.py`, summary
`tests/studies/learned_rule_ep_pilot.csv`) ran on its own record seed 20261131 and streams
`se_ratio_pilot` and `se_ratio_pilot_folds`, which no declared draw reads.
`tests/unit/test_learned_rule_ep_design.py` checks the seeds and recomputes every reading below
from the committed summary.

| pilot cell | n | draws | SE ratio | oracle-SE ratio | coverage | standardized bias | CPU-s per fit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `non_exceptional__declared` | 4,000 | 2,000 | 0.970 | 0.969 | 0.9365 | -0.021 | 5.3 |
| `misspecified_limit__declared` | 4,000 | 600 | 0.990 | 0.987 | 0.963 | -0.015 | 5.0 |
| `optimal_known_g__declared` | 4,000 | 600 | 0.999 | 0.998 | 0.957 | -0.097 | 4.3 |
| `non_exceptional__primary` | 2,000 | 300 | 0.943 | 0.940 | 0.930 | -0.024 | 4.1 |
| `non_exceptional__validation_rule_control` | 2,000 | 60 | 0.915 | 0.926 | 0.017 | 3.74 | 0.2 |

The plan's rule reads the `non_exceptional` ratio minus one pilot SE: 0.970 - 0.017 = 0.953,
below 0.96. So R = 7,200 per calibration cell, and the `T49` rows stay experimental whatever the
run shows, with the reason "the calibration cells are not powered at the measured SE deficit"
(`CALIBRATION_REPLICATES` and `T49_ADMITTED` in `learned_rule_ep_properties.py`). The oracle-SE
ratio sits at the SE ratio, so the reported SE is the fixed-rule SE of the realized rules and the
deficit is the error variance that rule learning adds, as the learned-rule-learners pilot found.

At R = 7,200 each gated cell passes with probability at least 0.97 at its pilot SE ratio and
coverage. Averaged over the pilot's own error (the SE ratio normal about the pilot ratio, the
coverage under its Jeffreys posterior), the assurance is 0.74 for `non_exceptional`, 0.81 for
`misspecified_limit` and 0.85 for `optimal_known_g`, below the 0.9 target: the pilot error is as
large as the margin it is read against. The primary floor passes with probability 0.99 at the
pilot coverage and 0.77 over its 300-draw pilot. The control's interval lies below 0.92 with
assurance above 0.999. The cells run as declared and are never downgraded.

## Run

A declared run takes `--output` and `--jobs` only, on the Python and package versions that rule L7
of `tests/canonical/learned_rule_run.py` declares, with one thread per numerical library:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u -m tests.canonical.learned_rule_ep.regenerate --jobs 4 --output <empty scratch directory>
```

Cost from the pilot's CPU seconds, with the harness refit of each fold rule and the paired
plug-in fit: the calibration cells about 60 CPU-hours, the primary phase 9, the diagnostic cells
7, the exceptional cell 5.5 and the control 1, about 83 CPU-hours, 21 hours on 4 cores. A
replicate at n = 2,000 costs about 8 CPU-s against the plan's 7, inside the 1.5 times that would
split the 4,000-row cells into a run of their own.

A smoke run passes another `--replicates` and writes outside the repository.
`--property-harness-check 2` also fits two draws of every property cell and prints the rule
agreement and solver warnings:

```bash
python -u -m tests.canonical.learned_rule_ep.regenerate --replicates 4 --jobs 4 --allow-failures --output <scratch> --property-harness-check 2
```
