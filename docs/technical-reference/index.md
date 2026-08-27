# Technical reference

This reference accounts for every statistical method `cleverly` ships, and for the evidence that
each one is implemented correctly.

The project validates a method in two independent ways. It compares `cleverly` against a canonical
implementation, where a maintained one exists. It also measures `cleverly` against the
repeated-sampling properties the method's own source theory predicts. Neither one replaces the
scientific instruments that check the derivation itself: the exact-law, Gateaux, remainder,
identity, and deliberate-mutation checks recorded in the [evidence manifest](evidence.md).

## Implementation matrix

The [implementation validation grid](method-evidence/validation-grid.md) covers the methods with a
registered repeated-sampling study. This matrix covers every implementation family `cleverly`
ships, and names the correctness evidence for each.

| implementation family | theory and citation | `cleverly` implementation | external provenance | correctness evidence |
| --- | --- | --- | --- | --- |
| Counterfactual means; ATE; risk and odds ratios; multi-arm contrasts | [Point-treatment TMLE](point-treatment-tmle.md#counterfactual-means-and-arm-contrasts); van der Laan & Rubin (2006) | [`estimators/tmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/tmle.py), [`targets/builtin.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/targets/builtin.py) | R `tmle`; pinned `tmle3` parameter/update source | [`ate`, `ey1`, `ey0`, `rr`, `or`](evidence.md#the-table) oracle, Gateaux, remainder, and identity rows |
| ATT and ATC | [Conditional-population effects](point-treatment-tmle.md#conditional-population-effects); van der Laan (2010) | [`inference/influence.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/influence.py) | no numeric external oracle claimed | [`att`, `atc`](evidence.md#the-table), including conditioning-event influence terms |
| Natural-course mean, PAR, and PAF | [Population interventions](point-treatment-tmle.md#population-interventions); Díaz Muñoz & van der Laan (2012) | [`targets/builtin.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/targets/builtin.py) | stochastic-intervention literature; no bounded parity witness | [`ey_obs`, `par`, `paf`](evidence.md#the-table) exact-law and transformation checks |
| Missing outcomes and controlled direct effects | [Observed-data extensions](point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects); Díaz & van der Laan (2017) for the randomized missing-outcome construction | [`estimators/direct_effect.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/direct_effect.py), [`estimators/tmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/tmle.py) | R `tmle` / `drtmle` are provenance only within their documented scope | [`tests/discrete_law_mar.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/discrete_law_mar.py), [`tests/discrete_law_cde.py`](https://github.com/esbraun/cleverly-tmle/blob/main/tests/discrete_law_cde.py), associated Gateaux and remainder tests |
| Deterministic, dynamic, and stochastic regimes | [Known regimes](point-treatment-tmle.md#known-regimes); Robins (2004), Díaz Muñoz & van der Laan (2012), Díaz & van der Laan (2013) | [`interventions/base.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/base.py), [`estimators/tmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/tmle.py) | pinned `tmle3` `LF_static` / `Param_TSM` source | [`ey_regime`, `ate_regime`](evidence.md#the-table), dynamic-rule and support mutations |
| Continuous modified treatment policies | [Modified treatment policies](point-treatment-tmle.md#modified-treatment-policies); Díaz Muñoz & van der Laan (2012), Haneuse & Rotnitzky (2013), Díaz et al. (2023) | [`interventions/shift.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/shift.py), [`estimators/tmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/tmle.py) | literature construction; no moving package specification | [`ey_shift`, `ate_shift`](evidence.md#the-table), continuous-law Gateaux and remainder checks |
| Incremental propensity-score interventions | [Incremental interventions](point-treatment-tmle.md#incremental-propensity-score-interventions); Kennedy (2019) | [`interventions/incremental.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/incremental.py), [`estimators/tmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/tmle.py) | no external implementation used as acceptance evidence | [`ey_ipsi`, `ate_ipsi`](evidence.md#the-table), including nonzero treatment-score and one-sided remainder witnesses |
| Point and longitudinal MSM projections | [Marginal structural models](msm-projections.md); Neugebauer & van der Laan (2007), Rosenblum & van der Laan (2010), Petersen et al. (2014) | [`msm.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/msm.py), [`longitudinal/msm.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/msm.py) | pinned `tmle3` `Param_MSM`; projected ordinary R `ltmle` regimen fits; raw R `ltmleMSM` differs in projection scale | gated [point](method-evidence/point-treatment-msm-projection.md) and [ordinary longitudinal](method-evidence/ordinary-longitudinal-msm-projection.md) studies, plus the point [`msm`](evidence.md#the-table) and [longitudinal MSM](evidence.md#longitudinal-estimands-outside-the-target-registry) oracle rows |
| Longitudinal end-of-study TMLE | [Sequential regression](longitudinal-tmle.md#end-of-study-regimen-means); Bang & Robins (2005), van der Laan & Gruber (2012) | [`longitudinal/estimator.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/estimator.py), [`longitudinal/sequential.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/sequential.py) | R `ltmle` 1.3-0 for ordinary fitting; pinned `lmtp` 1.5.4 for the cross-fitted row | [longitudinal evidence row](evidence.md#longitudinal-estimands-outside-the-target-registry), gated [ordinary](method-evidence/ordinary-end-of-study-longitudinal-tmle.md) and [cross-fitted](method-evidence/cross-fitted-end-of-study-longitudinal-tmle.md) studies, exact law, Gateaux and mutations |
| Longitudinal survival and competing risks | [Event-process TMLE](longitudinal-tmle.md#survival-and-competing-risks); Stitelman et al. (2012) | [`longitudinal/estimator.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/longitudinal/estimator.py) | R `ltmle` 1.3-0 for ordinary survival; pinned `lmtp` 1.5.4 for cross-fitted survival and both competing-risk rows | gated ordinary and cross-fitted [survival](method-evidence/ordinary-survival-curve-longitudinal-tmle.md) and [competing-risk](method-evidence/ordinary-competing-risk-longitudinal-tmle.md) studies, with a separate cross-fitted page for each; exact laws, Gateaux checks, and deliberate mutations remain in [longitudinal evidence](evidence.md#longitudinal-estimands-outside-the-target-registry) |
| Collaborative TMLE | [Collaborative selection](collaborative-tmle.md); van der Laan & Gruber (2010), Ju et al. (2019) | [`estimators/ctmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/ctmle.py) | pinned `ctmle3` outcome-adaptive source; pinned `tmle3` / `sl3` fold semantics | [C-TMLE estimator-variant evidence](evidence.md#estimator-variants-over-registered-targets), the gated [selector](method-evidence/selector-based-point-treatment-c-tmle.md) and [outcome-adaptive](method-evidence/outcome-adaptive-point-treatment-c-tmle.md) studies, selector mutations and support path tests |
| DR-TMLE | [Doubly-robust inference](dr-tmle/index.md); van der Laan (2014), Benkeser et al. (2016/2017), Benkeser & Hejazi (2023) | [`estimators/drtmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/drtmle.py), [`estimators/reduced.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/reduced.py) | pinned R `drtmle` 1.1.2 source and numerical comparator for the binary complete-data construction | registered [canonical DR-TMLE study](method-evidence/canonical-dr-tmle.md), [DR-TMLE contract](dr-tmle/index.md), theorem identity, Gateaux, remainder, score, cross-fit, and missing-data tests |
| Cross-fitting and CV-TMLE | [CV-TMLE and cross-fitting](cv-tmle.md); Zheng & van der Laan (2011), Levy (2018) | [`learners/crossfit.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/learners/crossfit.py) | pinned `tmle3` stacked validation likelihood and `sl3` fold/full prediction source | gated [stacked CV-TMLE](method-evidence/stacked-point-treatment-cv-tmle.md) and [fold-evaluated CV-TMLE](method-evidence/fold-evaluated-point-treatment-cv-tmle.md) repeated-sampling rows, plus leakage and repeated-fold mutations |
| Inference, diagnostics, and sensitivity | [How every method reports uncertainty](inference.md) and [sensitivity and validation methods](validation-methods.md); [sensitivity references](../references.md#sensitivity-analysis) | [`inference/`](https://github.com/esbraun/cleverly-tmle/tree/main/src/cleverly/inference), [`sensitivity/`](https://github.com/esbraun/cleverly-tmle/tree/main/src/cleverly/sensitivity), [`validation/`](https://github.com/esbraun/cleverly-tmle/tree/main/src/cleverly/validation), [`assessment.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/assessment.py) | no external implementation used as acceptance evidence | inference, assessment, sensitivity, and serialization mutations |

"Existing implementation" has two meanings in this matrix. One is the local module that executes
the method. The other is a pinned external implementation, where the source was audited. External
parity is never the derivation or the acceptance gate.

## How to read the evidence column

The [evidence manifest](evidence.md) distinguishes independent scientific checks from bounded
implementation witnesses. An exact finite-support law can verify a complete functional but miss a
term that vanishes at the truth. A Gateaux comparison can verify the influence curve but share a
mistake with the law. Remainder-rate and deliberate-mutation tests make those blind spots visible.
External implementations are therefore localized provenance, not a substitute for these checks.

```{toctree}
:maxdepth: 2

method-evidence/index
evidence
validation-methods
inference
point-treatment-tmle
cv-tmle
collaborative-tmle
dr-tmle/index
longitudinal-tmle
msm-projections
scope-and-refusals
../references
```
