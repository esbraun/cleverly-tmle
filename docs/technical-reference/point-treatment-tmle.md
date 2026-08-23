# Point-treatment TMLE

## What this solves

You have one exposure, measured once, and an outcome measured after it. You want the effect of that
exposure on that outcome, and the exposure was not randomized. Adjusting for confounders by putting
them in a regression gives you a coefficient whose meaning depends on the model. Weighting by the
inverse propensity gives you an estimate whose variance depends on the worst-behaved row.

Point-treatment TMLE gives you a plug-in estimate of a parameter you named before you fitted
anything, with a valid interval, while letting a machine-learning model fit both nuisances.

| your situation | what this method buys | what it costs |
| --- | --- | --- |
| observational data, confounders measured | a doubly-robust estimate: consistent if the outcome regression **or** the treatment mechanism is consistent | you must name the estimand first. The method will not tell you which contrast you meant |
| you want flexible learners for the nuisances | the estimate stays a plug-in, so it respects the outcome's bounds and the parameter's range | a valid interval needs a *product* rate on the two nuisances, which flexible learners do not guarantee. See [CV-TMLE](cv-tmle.md#what-this-solves) |
| you want more than a difference in means | arm means, ATE, ATT, ATC, natural-course mean, PAR, PAF, risk ratio, and odds ratio from one fit | each target is a separate registered parameter with its own influence curve. Two axes cannot share one fit |
| some outcomes are missing, or you want an effect at a fixed intermediate | missingness and a controlled direct effect compose into the same clever covariate | positivity is then needed for the product of the mechanisms, not for the treatment alone |
| the exposure is continuous, or the policy is a rule rather than a level | regimes, modified treatment policies, and incremental tilts run through the same engine | each axis is a different estimand with a different influence curve. They are not interchangeable |

Reach for a different entry when the exposure repeats over time
([longitudinal TMLE](longitudinal-tmle.md)), when you want the effect summarised by a working model
([MSM projections](msm-projections.md)), when the adjustment set is large and mostly irrelevant
([collaborative TMLE](collaborative-tmle.md)), or when you expect one nuisance to be inconsistent
and still want an interval ([DR-TMLE](dr-tmle.md)).

Three worked applied analyses cover this entry. The
[point-treatment tutorial](../examples/point-treatment-tmle.md) is the main one, and it also
demonstrates the conditional-population effects below. The intervention axes have their own
tutorial in [intervention axes](../examples/interventions.md). Missing outcomes have theirs in
[survey non-response](../examples/survey-nonresponse.md).

## The algorithm as implemented

The ordinary fit does five things in order. It fits the outcome regression $Q$ and the treatment
mechanism $g$. It bounds the predictions where the estimand requires it. It fluctuates $Q$ along a
least-favorable logistic submodel. It evaluates the targeted counterfactual means. It builds the
interval from the targeted influence curve.

Implementation:
[`estimators/tmle.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/tmle.py),
[`estimators/targeting.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/targeting.py),
and
[`targets/builtin.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/targets/builtin.py).
The theory is van der Laan and Rubin (2006) and Gruber and van der Laan (2010); see the
[targeted-learning references](../references.md#targeted-learning-in-general). R `tmle` and the
pinned `tmle3` update and parameter source are implementation references and not oracles.

### Counterfactual means and arm contrasts

For observed data $O=(W,A,Y)$ and treatment level $a$, the counterfactual mean is

$$
\psi_a(P) = E_P\{Q_P(a,W)\}, \qquad Q_P(a,w)=E_P(Y\mid A=a,W=w).
$$

Under consistency, conditional exchangeability, and treatment positivity, this observed-data
functional identifies $E(Y^a)$. Its efficient influence function is

$$
D_a(P)(O)=\frac{\mathbb{1}(A=a)}{g_P(a\mid W)}\{Y-Q_P(A,W)\}
            + Q_P(a,W)-\psi_a(P).
$$

`CounterfactualMean` reports one or all $\psi_a$. `ATE` reports $\psi_a-\psi_{a_0}$ for each
non-reference arm. `RiskRatio` and `OddsRatio` apply smooth transformations and use delta-method
influence curves. Binary and multi-valued discrete treatments run through the same code. The arm
label stays a structured parameter-key field rather than a string to parse.

### Conditional-population effects

`ATT` and `ATC` condition the effect on the population that received one observed arm. Their
influence curves include the randomness of that conditioning event. They are not ATE curves with a
different summary label. Positivity is needed at the counterfactual reference arm *within* the
conditioning population.

The estimators are `att_estimate` and `atc_estimate` in
[`inference/influence.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/inference/influence.py).

### Population interventions

`NaturalCourseMean` reports $E(Y)$. With a declared reference intervention $a_0$,

$$
\operatorname{PAR}=E(Y)-E(Y^{a_0}), \qquad
\operatorname{PAF}=1-\frac{E(Y^{a_0})}{E(Y)}.
$$

The intervention mean uses the targeted arm mean. The complete-data natural-course mean uses the
empirical distribution. PAF is undefined when the observed outcome risk is zero. The construction
follows Diaz Munoz and van der Laan (2012).

### Missing outcomes and controlled direct effects

With an observation indicator $\Delta$, the outcome residual carries an extra inverse weight from
the observation mechanism $\pi(A,W)=P(\Delta=1\mid A,W)$. The missing-at-random composition needs
positivity of $\pi$ wherever the intervention places mass. Missingness is a design role. Missing
adjustment values and missing treatment values are not implicitly covered.

For a declared intermediate $Z$, `ControlledDirectEffect(intermediate=z)` targets the treatment
contrast with $Z$ fixed at $z$. Its clever covariate composes the treatment, intermediate, and
observation mechanisms. This is a controlled direct effect at a specified level. It is not a
mediation decomposition.

**Double robustness means something different here, and the difference is worth stating.** Without
missingness the fit is consistent if $Q$ is right or if $g$ is right. With missingness it is
consistent if $Q$ is right, or if the *product* $g\pi$ is right. A correct propensity buys nothing
on its own when the missingness model is wrong, and errors in the two mechanisms can cancel
exactly.

Implementation:
[`estimators/direct_effect.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/direct_effect.py).
Diaz and van der Laan (2017) supplies the randomized-trial missing-outcome construction.

### Weights, strata, and clusters

Fixed observation weights define the tilted target law $dP_w=w\,dP/E_P(w)$. They are used in the
nuisance losses, in targeting, in the plug-in average, and in the influence-curve covariance.
`strata=` produces stratum-specific parameters. `cluster=` changes the independent unit for
covariance and fold construction, and it does not change the estimand. See
[how every method reports uncertainty](inference.md#clusters).

## Variations

### Estimator options

| option | what it changes | is it a different estimator? |
| --- | --- | --- |
| `fluctuation="logistic"` or `"linear"` | the submodel the outcome regression is fluctuated along | no. Both solve the same score |
| `algorithm="iterative"` or `"one_step"` | Newton iteration, or the universal least-favorable submodel of van der Laan and Gruber (2016) | no. The two are pinned to agree by an exact identity |
| `target_weights=` | whether the fluctuation carries the covariate as a weight or as a regressor | no. The weighted and clever-covariate forms solve the same equation |
| `g_bounds=` | the treatment-mechanism truncation. `"auto"` is target-aware | it changes the finite-sample procedure. Report it with the support diagnostics |
| `q_bounds=`, `submodel_alpha=` | the outcome scaling, and the logistic submodel bound | no |
| `screen_treatment=`, `screen_threshold=`, `min_retain=` | covariate screening for the treatment mechanism | no |
| `cross_fit=`, `n_folds=`, `repeats=` | sample splitting for the nuisances | **yes.** See [CV-TMLE](cv-tmle.md) |

`submodel_alpha` bounds the logistic submodel. `alpha` is the interval's significance level. The
two are separate keywords because they once shared a name and a fit read one as the other.

### Known regimes

For a known stochastic intervention $g^*(a\mid w)$,

$$
\psi_{g^*}(P)=E_P\left\{\sum_a Q_P(a,W)g^*(a\mid W)\right\}.
$$

Its efficient influence function has residual weight $g^*(A\mid W)/g_P(A\mid W)$ and a centered
plug-in term. Deterministic static and dynamic rules are degenerate cases of $g^*$. Identification
needs positivity only where the regime assigns mass. A known rule does not depend on $P$, so its
influence function carries no term for estimating the rule.

`Static`, `Rule`, and `Stochastic` implement the intervention protocol. `RegimeMean` and
`RegimeContrast` define the levels and the contrasts. See Robins (2004), Diaz Munoz and van der
Laan (2012), and Diaz and van der Laan (2013). Implementation:
[`interventions/base.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/base.py)
and
[`interventions/support.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/support.py).

### Modified treatment policies

For a continuous exposure and an invertible shift $d(a,w)$, a modified treatment policy targets

$$
\psi_d(P)=E_P\{Q_P(d(A,W),W)\}.
$$

The residual clever covariate is a conditional-density ratio. For a simple additive shift by
$\delta$ away from a boundary it takes the form

$$
H_d(A,W)=\frac{g_P(A-\delta\mid W)}{g_P(A\mid W)},
$$

with the inverse-map and Jacobian terms the declared policy needs. Identification requires the
shifted dose to stay inside the observed conditional support. A fixed `cap=` is part of the policy.
Estimating the cap from the same data would define a different, pathwise-dependent intervention.

The implementation fits a conditional density, targets the outcome regression as a function of
dose, and evaluates it at $d(A,W)$. Missingness and intermediate mechanisms multiply the density
ratio when those roles are declared. The estimator is doubly robust in the outcome regression and
the complete density-and-mechanism product.

**A modified treatment policy is not the stochastic regime it induces.** The shift $d$ induces
$g^d(b \mid W) = \sum_{a: d(a,W)=b} g(a \mid W)$, and a `Stochastic` regime at that density has the
same mean *and* the same clever covariate, entry for entry. The influence curves differ anyway. A
regime's plug-in term averages $Q$ over the doses and is a function of $W$ alone. A shift's reads
the dose the unit actually received. The gap is exactly

$$
\operatorname{Var}(D_{\text{mtp}}) = \operatorname{Var}(D_{\text{regime}})
  + \operatorname{Var}\{Q(d(A,W),W) - E[Q(d(A,W),W) \mid W]\},
$$

so a modified treatment policy is strictly *harder* to estimate than the regime inducing the same
mean. Delegating one to the other reports a standard error that is too small.

Theory: Diaz Munoz and van der Laan (2012), Haneuse and Rotnitzky (2013), and Diaz, Williams,
Hoffman and Schenck (2023). Implementation:
[`interventions/shift.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/shift.py),
[`learners/density.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/learners/density.py),
and
[`fluctuation/submodel.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/fluctuation/submodel.py).

### Incremental propensity-score interventions

For binary treatment, an incremental intervention multiplies the observed odds by $\delta$:

$$
q_\delta(1\mid W)=\frac{\delta g_P(1\mid W)}
                       {\delta g_P(1\mid W)+1-g_P(1\mid W)}.
$$

The denominator keeps the clever covariate bounded even when the observed propensity approaches
zero, so this parameter does not need conventional treatment positivity.

**This is the one axis that targets the treatment mechanism.** The intervention is a functional of
$P$, so the efficient influence function carries a term for the pathwise derivative through $g$:

$$
D = \frac{\delta A + 1 - A}{D_\delta}\{Y-Q(A,W)\}
  + \frac{\delta\{Q(1,W)-Q(0,W)\}}{D_\delta^2}(A-g)
  + m(W) - \psi(\delta).
$$

The middle term lives in the tangent space of the treatment mechanism, so no fluctuation of $Q$
reaches it. The mechanism therefore gets a logistic submodel of its own whose score is exactly that
term. Each covariate reads the other's fitted value, so the two alternate. The alternation is
coordinate ascent on one joint likelihood, because the outcome and treatment quasi-likelihoods are
separate factors, so the joint value never decreases. `score_check()` reports two rows for such a
fit rather than one.

**It is the only estimand here that is not doubly robust.** Because $g$ appears in the estimand,
every term of the second-order remainder carries $(\hat g - g_0)$ as a factor. A consistent
mechanism kills the remainder whatever $Q$ does. A consistent $Q$ does not, and no accuracy in it
can. Read the interval as conditional on $g$ being right, which is why `diagnostics.support()`
matters more here than elsewhere. There is no doubly-robust fallback.

With `delta=` the guarantee *tightens* rather than weakening: $\hat g$ right **and** one of
$\hat\pi$, $Q$ right, because the squared mechanism-error term is free of $\pi$ and survives
everything else.

Kennedy (2019) is the primary theory reference. Implementation:
[`interventions/incremental.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/incremental.py)
and
[`fluctuation/mechanism.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/fluctuation/mechanism.py).

### Support reports

Arm positivity, regime support, shift support, and incremental support answer four different
questions. The intervention classes therefore expose distinct reports rather than reducing every
overlap question to one propensity histogram.

## Validation issues special to this method

The generic instruments are described in
[sensitivity and validation methods](validation-methods.md#how-the-library-certifies-itself). Five
things about *this* method are not generic.

**Two arms cannot distinguish arm-keyed code from two-column code.** A law with three arms can, and
its labels sort into a different order than they were written in, so a helper that equates an arm
code with an arm position fails rather than passes. Every multi-arm claim is made on that law.

**Two static regimes cannot distinguish mixing over the arms from picking a column.** The regime
oracle therefore carries three kinds of regime: a static one, a rule that depends on $W$, and a
stochastic one that is degenerate nowhere.

**A cap above the largest dose never exercises the shift's boundary indicator.** A unit can only
have been shifted *to* dose $a$ if the shift from $a-\delta$ was not itself held back, so the
covariate carries a further indicator. The shift law therefore has two caps, and the tight one is
what caught that indicator missing.

**A Gateaux check on an exact law cannot see a mechanism read at the wrong dose.** At the truth
$\epsilon$ is zero, so the reported curve reads the observed block and the untargeted $Q$, and no
counterfactual block is read at all. That mutation is pinned structurally in
`tests/unit/test_shift_submodel.py` and behaviourally at nonzero $\epsilon$ in
`tests/unit/test_shift_fit.py`. It was applied and seen to pass the Gateaux module first.

**The two halves of double robustness are not interchangeable when positivity is strained.** With
$Q$ right, the estimand is recovered by integrating a regression over the covariate distribution,
which needs no overlap at all. With only $g$ right, everything rests on inverse-propensity weights.
On a process with 11% of the population below $g=0.05$, that half stops delivering, at a measured
bias of $-0.13$ against $-0.01$ for the outcome half. That is the positivity premise failing rather
than a truncation artefact. `tests/e2e/test_double_robustness.py` runs both overlap regimes and
pins the asymmetry.

| where to read the evidence | what is there |
| --- | --- |
| [implementation validation grid](index.md#implementation-validation-grid) | the registered study row for ordinary point-treatment TMLE |
| [canonical point-treatment TMLE](method-evidence.md#canonical-point-treatment-tmle) | 34 accuracy tests, 17 paired comparisons against R `tmle3`, and 12 theory-property cells, test by test |
| [the evidence manifest](evidence.md#the-table) | which oracle, Gateaux, remainder, and identity instruments each registered target has, and what none of them would see |
