# R `ltmle` parity for the sequential baselines (X41 cell P)

This directory holds the R runner of plan cell P. The cell is not registered. A follow-up pull
request registers it as its own study, so the Python study `sequential-baselines` does not
depend on it.

| file | content |
| --- | --- |
| `run_study.R` | fits R `ltmle` 1.3-0 twice per regime and replicate on laws `D2B` and `G`, and writes one point per regime and estimator |

The two `ltmle` calls are the calls of the W9 fixture (`tests/canonical/sequential_baselines`):

| point | call |
| --- | --- |
| g-computation | `gcomp=TRUE`, `stratify=TRUE` |
| IPTW | `gcomp=FALSE`, `stratify=FALSE`. `CalcIPTW` divides by the weight sum, which is the Hajek form |

Each `Qform` reads the covariates and no arm column. `cleverly`'s `outcome_design` omits the arm
columns at a label or rule node, so an arm column changes a main-terms fit under a rule.
Each `gform` reads the full history. Both calls use `SL.library="glm"`, `gbounds=c(0.01, 1)`
and `estimate.time=FALSE`. On `G` the call sets `Yrange` to the declared `q_bounds`.

## Open items before registration

| item | reason |
| --- | --- |
| the Python phase | write the samples and truths, call `run_study.R` through the shared driver, and gate each point at `1e-6` |
| the `G` outcome model | `ltmle` fits a logistic regression to the scaled outcome. `cleverly`'s `GLM(family="binomial")` now refuses a target with three or more labels, so the matching `cleverly` fit needs a declared fractional-outcome route |
| the `Q` clip | `ltmle` clips the initial `Q` to `[1e-4, 0.9999]`. The design test must show that the clip does not bind at the declared `n` |
