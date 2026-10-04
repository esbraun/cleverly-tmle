# Review merged PRs 253 through 264

The review starts from merged `main` at `12fd6820d7745b59216f6add32feebb088e2c17e`.
Three agents inspected the twelve PR descriptions, merge changes, current implementation,
scientific contracts, regression instruments, and committed study artifacts. The orchestrator
planned the corrections in commit `42eab43c` before assigning implementation. A second agent
independently reviewed the combined fixes and required stronger nonzero witnesses.

| PRs | review record |
| --- | --- |
| 253–258 | [Notebook, scope, band, and result composition review](253-258.md) |
| 259–263 | [Natural extensions, missing data, attributable effects, and strata](259-263.md) |
| 264 | [Cluster variance, reference rules, and F28 determinations](264.md) |

## Confirmed implementation findings

| defect | correction | regression instrument |
| --- | --- | --- |
| A contrast reads a ratio or transformed curve on the wrong scale | Both composition helpers recover each input's reported-scale curve before applying the derivative | `test_derived_scale_and_bootstrap.py`: nonunit ratios, mixed scales, transformed RR/OR inputs, complemented probabilities, and singular-scale refusals |
| Boundary bootstrap draws crash a derived ratio | The shared ratio function marks out-of-domain draws missing. The derived summary counts each requested unusable draw once | The same module: independent arithmetic for boundary, nonfinite, and absent failed draws; arbitrary callback errors remain visible |
| A valid transformed estimate raises during display | Representation and frame reporting share the missing default-test rule | The same module: log/logit intervals and explicit interior tests remain available while the default null refuses |
| Composite diagnostics hide initial factor clipping | The diagnostic unions raw-factor masks and reuses `Propensity.truncate` | `test_drtmle_review_diagnostics.py`: clipping with interior exit states, overlapping factors, and binary complement roundoff |
| A correction diagnostic compares one stratum's score with a marginal mean | A shared helper restores weighted stratum masses and handles both score layouts | The same module: independent weighted arithmetic and live or deliberately perturbed nonzero diagnostic states |
| Zero-mass clusters pass a validation-fold support check | Both fold-evaluation settings require two positive-mass clusters before learning | `test_weighted_cluster_preflight.py`: supplied and generated splits, zero/one active cluster, and two-active-cluster acceptance |

The corrections reuse shared result composition, clipping masks, stratum masses, and cluster
counts. They preserve the existing covariance and weighted-sampling contracts.

## Scientific and scope determinations

The reports assign a determination to every reviewed PR and every red group introduced by
these changes. A comparator's agreement supports implementation parity; it does not prove
finite-sample coverage. Reported coverage, calibration, nonlinear bias, and diagnostic controls
retain their original verdicts and margins. The review identifies no justified universal SE
inflation, alternate seed selection, or post-hoc margin change that repairs these failures.

The informative-size fold bias has an independently reconstructed ratio-bias witness. The
shipped stacked report is the supported practical remedy. A differently weighted fold estimator
needs a coherent point, fluctuation, influence curve, variance, and registered study contract.

The cluster documentation now describes the pooled fold count and minimum input reference as
reporting rules measured by the studies. It removes an unsupported general coverage guarantee.
Independent variance-component approximations do not establish that guarantee for correlated
contrasts; see the [NIST Welch–Satterthwaite definition](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/welchsat.htm).
The F28 floor remains an evidence boundary. An undefined mean of a `t_1` reference does not
prevent finite confidence quantiles or prove that every possible interval is unavailable.
The natural-extension record now states the separate point-treatment and longitudinal floors.

Unimplemented qualifying extensions keep their named owners and explicit refusals. The review
does not substitute an approximation to a different estimand for a missing derivation.

## Registered evidence and verification

No registered study changes its inputs, fitted arrays, curves, intervals, exits, or verdicts.
The detailed reports name the relevant drivers and the inspected hooks. Current registered
contrast cells use untransformed level inputs; their bootstrap ratio cells are not exercised.
The diagnostic readers are outside the registered computation. The cluster studies use only
positive weights. These facts make study regeneration unnecessary for the confirmed fixes.
All canonical artifacts match the PR base.

PR 265 merged while the full review suite ran. The integration preserves its shipped
clustered cross-fitted LTMLE and its X27 refusal. The corrected verdict table retains the
10-cluster point-treatment floor and 20-cluster longitudinal floor. Its two-positive-mass-fold
requirement explicitly describes point-treatment fold reports; longitudinal targeting is pooled.
The new `clustered-cross-fitted-ltmle` and `few-cluster-cross-fitted-ltmle` drivers do not
exercise the changed input-scale composition or correction readers. The independent integration
review finds no required numerical source correction or study regeneration.

Validation runs use the dedicated worktree. The local launcher binds the process and inherited
workers to two logical CPUs, sets BLAS and OpenMP thread limits to one, and limits xdist to two workers.
The original checkout and its editable installation remain untouched.

| gate | result |
| --- | --- |
| full fast suite before PR 265 integration | 15,675 passed, 235 skipped |
| focused integration suite | 235 passed, including longitudinal cluster-status propagation and all three new regression modules |
| new regression instruments | 55 passed, including nine old-index mutation controls |
| Ruff lint and whole-tree formatting after integration | pass; 900 files formatted |
| Mypy after integration | pass; 106 source files |
| whole prose ledger refresh | 3 judged findings, 0 undecided; no ledger changes |
| documentation build after integration | pass with warnings treated as errors |

The independent reviewer closes every implementation finding and approves the integration.
