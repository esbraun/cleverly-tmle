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
| some outcomes are missing, or you want an effect at a fixed intermediate | missingness and a controlled direct effect compose into the same clever covariate | arm-indexed targets need positivity for the product of the mechanisms; the natural-course mean needs response positivity only |
| the exposure is continuous, or the policy is a rule rather than a level | regimes, modified treatment policies, and incremental tilts run through the same engine | each axis is a different estimand with a different influence curve. They are not interchangeable |

Reach for a different entry in four cases.

| when | read |
| --- | --- |
| the exposure repeats over time | [longitudinal TMLE](longitudinal-tmle.md) |
| you want the effect summarised by a working model | [MSM projections](msm-projections.md) |
| the adjustment set is large and mostly irrelevant | [collaborative TMLE](collaborative-tmle.md) |
| you expect one nuisance to be inconsistent and still want an interval | [DR-TMLE](dr-tmle/index.md) |

Three worked applied analyses cover this entry. The
[point-treatment tutorial](../examples/point-treatment-tmle.ipynb) is the main one, and it also
demonstrates the conditional-population effects below. The intervention axes have their own
tutorial in [intervention axes](../examples/interventions.ipynb). Missing outcomes have theirs in
[survey non-response](../examples/survey-nonresponse.ipynb).

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

Under consistency, no interference, conditional exchangeability, and treatment positivity, this
observed-data functional identifies $E(Y^a)$. Its efficient influence function is

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

The intervention mean uses the targeted arm mean. With complete outcomes, the natural-course mean
uses the empirical distribution. PAF is undefined when the observed outcome risk is zero. The
complete-data construction follows Díaz Muñoz and van der Laan (2012); the missing-outcome
natural-course construction is stated below.

### Missing outcomes and controlled direct effects

With an observation indicator $\Delta$, let $X=(A,W)$,
$m(X)=E(Y\mid\Delta=1,X)$, and $\pi(X)=P(\Delta=1\mid X)$. Under missingness at random given
$(A,W)$ and response positivity, `NaturalCourseMean` identifies

$$
\psi=E\{m(X)\}, \qquad
D(O)=\frac{\Delta}{\pi(X)}\{Y-m(X)\}+m(X)-\psi.
$$

The estimator fits the logistic fluctuation among respondents with the clever covariate
$1/\pi(X)$. It then averages the targeted outcome regression over every input row. This target
has two nuisance functions: the outcome regression and the response mechanism. It is consistent
when either is correct, and its exact remainder is

$$
R_2(P,P_0)=E_0\left[\left\{1-\frac{\pi_0(X)}{\pi(X)}\right\}
  \{m(X)-m_0(X)\}\right].
$$

For the ordinary first-order interval, the fitted response probability is bounded away from zero.
The estimated influence curve converges in $L_2(P_0)$ inside one Donsker class. The remainder is
$o_P(n^{-1/2})$. A sufficient remainder rate is
$\|\hat\pi-\pi_0\|_{P_0}\|\hat m^\star-m_0\|_{P_0}=o_P(n^{-1/2})$.

The efficiency claim also requires the bounded response fit to converge to $\pi_0$. Clipping cannot
supply that condition below the true response probability on a set with positive probability.

#### Stacked CV-TMLE for this target

The cross-fitted implementation uses one package-generated, near-balanced V-fold partition. For
fold $v$, it fits $m_v$ and $\pi_v$ on the training complement. The outcome regression uses only
respondents in that complement. Both nuisances predict only the held-out rows.

One common fluctuation coefficient targets the stacked predictions:

$$
\operatorname{logit}m_{v,\epsilon}(X)
=\operatorname{logit}m_v(X)+\frac{\epsilon}{\pi_v(X)},
\qquad
P_n\!\left[\frac{\Delta}{\pi_v(X)}\{Y-m_{v,\epsilon}(X)\}\right]=0.
$$

Here $v$ is the fold that holds each row. The estimator evaluates the point and influence curve on
the same stacked rows:

$$
\widehat\psi=P_n m_{v,\widehat\epsilon}(X),
\qquad
D_i=\frac{\Delta_i}{\pi_v(X_i)}\{Y_i-m_{v,\widehat\epsilon}(X_i)\}
    +m_{v,\widehat\epsilon}(X_i)-\widehat\psi.
$$

The variance is $P_nD_i^2/n$, the raw second moment of the curve. The score equation and the
plug-in point above give $P_nD_i=0$ to targeting tolerance. This variance therefore equals the
centered-rule variance $\{n(n-1)\}^{-1}\sum_i(D_i-\bar D)^2$ times $(n-1)/n$. The estimate
declares the `"second_moment"` covariance rule, so `covariance()` and contrasts use the same moment. See
[covariance rules](inference.md#covariance-rules).

The estimator needs at least one respondent and one nonrespondent in each training complement. It
checks the sample and then every complement, before either learner receives data. Each failure
raises `DataError`.

| failure | what the message tells you to do |
| --- | --- |
| the sample holds fewer than two respondents or fewer than two nonrespondents | use the in-sample estimator. No fold count or `random_state` can succeed |
| one training complement holds no respondent or no nonrespondent | increase `n_folds`, use a different `random_state`, or use the in-sample estimator |

Both messages name the in-sample estimator as `CrossFitting(enabled=False)`. The engine form is
`cross_fit=False`. Change that one setting. `stratify_by="none"` is required by two cross-fitted
contracts, this one and the [arm-indexed contract](#stacked-cv-tmle-for-arm-indexed-targets), and
it is accepted everywhere else. One fold balances nothing, so an in-sample fit keeps whichever
policy the declaration carries.

Cross-fitting removes the Donsker condition on the initial nuisance classes. It does not remove
response positivity, $L_2(P_0)$ convergence of the estimated curve, or the second-order condition.
The common coefficient must converge. Its finite-dimensional fluctuation class must satisfy the
source's entropy condition.

The fold-specific remainder is

$$
R_{2,v}=P_0\!\left[
  \left\{1-\frac{\pi_0(X)}{\pi_v(X)}\right\}
  \{m_{v,\widehat\epsilon}(X)-m_0(X)\}
\right].
$$

The stacked remainder $\sum_v(n_v/n)R_{2,v}$ must be $o_P(n^{-1/2})$. The corresponding product
rate for the response and targeted outcome errors is sufficient.

After relabeling the response indicator as its treatment, Levy (2018) supplies the stacked update
and whole-sample plug-in in its abstract. Levy's Section 3.1 supplies the asymptotic expansion
after the pooled score is solved. Zheng and van der Laan (2011), Sections 2 and 2.1, supply the
training-complement nuisance construction and the untargeted empirical-distribution component.
Their Theorem 2 gives the partial-targeting expansion and its conditions. The
[CV-TMLE reference](cv-tmle.md) gives the fold-weighting boundary.

No treatment mechanism enters either implementation. Both fits require one scalar `ey_obs`,
binary treatment, unweighted iid rows, and iterative unweighted logistic targeting.

| fit | outcome | cross-fitting declaration |
| --- | --- | --- |
| ordinary | binary, or continuous with fixed `q_bounds` | `CrossFitting(enabled=False)` |
| stacked | binary | `CrossFitting(enabled=True, n_folds=10, repeats=1, stratify_by="none", targeting_scheme="pooled", fold_evaluation=False, split_plan=None)`. `n_folds` must be 2 or more. The registered study uses 10 |

The stacked fit refuses one fold with `CapabilityError` before any learner is fitted. One fold is
the in-sample estimator. The refusal stops that estimate from being reported under the stacked
contract and its second-moment covariance rule.

[Missing-outcome natural-course contracts](scope-and-refusals.md#missing-outcome-natural-course-contracts)
lists every refusal for both fits. The source for the ordinary fit is
Díaz, Carone and van der Laan (2016), Section 2 and Equations (1)–(5). The registered
[ordinary missing-outcome natural-course TMLE study](method-evidence/ordinary-missing-outcome-natural-course-tmle.md)
checks both robustness halves, targeting, complete-case controls, root-$n$ behavior, efficiency,
and interval calibration. It also compares both laws with the R `tmle` 2.1.1 population-mean path.
The separate
[stacked missing-outcome natural-course CV-TMLE study](method-evidence/stacked-missing-outcome-natural-course-cvtmle.md)
checks the cross-fitted estimator under its own contract.

For arm-indexed counterfactual means, the outcome residual instead carries the inverse product of
the treatment mechanism $g(A\mid W)$ and the observation mechanism $\pi(A,W)$. That
missing-at-random composition needs positivity of both mechanisms wherever the intervention places
mass. Missingness is a design role. Missing adjustment values and missing treatment values are not
implicitly covered.

For a declared intermediate $Z$, `ControlledDirectEffect(intermediate=z)` targets the treatment
contrast with $Z$ fixed at $z$. Its clever covariate composes the treatment, intermediate, and
observation mechanisms. This is a controlled direct effect at a specified level. It is not a
mediation decomposition.

**Double robustness differs between these targets.** The natural-course mean is consistent if
$m$ is right or if $\pi$ is right. An arm-indexed target with missingness is consistent if $Q$ is
right, or if the *product* $g\pi$ is right. A correct treatment mechanism buys nothing on its own
when the response mechanism is wrong, and errors in the two mechanisms can cancel exactly.

Arm-indexed and controlled-direct-effect implementation:
[`estimators/direct_effect.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/direct_effect.py).
Diaz and van der Laan (2017) supplies the randomized-trial missing-outcome construction.
[Stacked CV-TMLE for arm-indexed targets](#stacked-cv-tmle-for-arm-indexed-targets) states the
cross-fitted contract and its evidence.

### Stacked CV-TMLE for arm-indexed targets

Ordinary TMLE cross-fits arm-indexed means and contrasts with missing outcomes under one stacked
CV-TMLE contract. The contract applies to a fit when all of the conditions below hold
(`_is_arm_indexed_missing_crossfit` in `estimators/tmle.py`):

- the outcome is missing for at least one row;
- `CrossFitting(enabled=True)`;
- the parameters are indexed by the arms of a discrete treatment;
- the design declares no intermediate; and
- the request contains no `NaturalCourseMean`, PAR, or PAF.

The shift, incremental, regime, MSM, and controlled-direct-effect fits are outside this contract.
The natural-course mean has its own contract above. PAR and PAF keep their own refusal.

The table gives the admitted settings. Before fold generation, the fit refuses a departure from
each row except the band settings in the inference row. The table leaves other settings open, for
example the learners and `g_bounds`.
[Missing-outcome arm-indexed contract](scope-and-refusals.md#missing-outcome-arm-indexed-contract)
lists each refusal and its order.

| setting | admitted |
| --- | --- |
| estimator | `TMLE` or `TMLEMethod` |
| targets | `ey`, `ey0`, `ey1`, and `ate`. `rr` and `or` for a binary outcome. With K arms, `ey` reports `ey[a]` for each arm, and `ate`, `rr`, and `or` compare each non-reference arm with the reference arm |
| treatment | two or more arms |
| outcome | binary, or continuous with a fixed `q_bounds` equal to the known outcome support |
| folds | package-generated, `stratify_by="none"`, `n_folds` of at least 2, `repeats=1`, and `split_plan=None` |
| targeting | `Targeting(fluctuation="logistic", algorithm="iterative", target_weights=False)`, with `targeting_scheme="pooled"` and `fold_evaluation=False` |
| rows | unweighted iid rows, with no clusters and no baseline strata |
| inference | pointwise influence-curve Wald intervals, and `n_bootstrap=0`. The simultaneous band is on by default. It admits every `multiplier_kind` and a custom `n_multiplier`. The registered study measures only the Rademacher default with 1,000 draws |

`CrossFitting` defaults to `stratify_by="treatment"`, and the default estimand list of a two-arm fit
includes `att` and `atc`. Every fit must therefore set `stratify_by="none"`, and a two-arm fit
must name its estimands.
The [RM17 audit](../roadmap.md#rm17-data-dependent-fold-strata-and-outcome-scales-under-cross-fitting)
found no result for treatment-stratified folds or for an outcome scale from held-out rows. For
that reason, the contract uses unstratified folds and a prespecified `q_bounds`.

**The joint vector.** For fold $v$, the fit trains $Q_v$ on the respondents of the training
complement, and it trains $g_v$ and $\pi_v$ on the whole complement. Each nuisance predicts only
the held-out rows. One logistic fluctuation has one coefficient for each arm. The clever
covariate for arm $a$ is

$$
H_a(O)=\frac{\mathbb 1\{A=a\}\,\Delta}{g_v(a\mid W)\,\pi_v(a,W)}.
$$

The fluctuation fits on the respondents only. Each row has at most one nonzero column
(`src/cleverly/fluctuation/submodel.py`). The arm-$a$ counterfactual prediction moves by
$\epsilon_a/\{g_v(a\mid W)\pi_v(a,W)\}$ on the logit scale. The point and the curve for arm $a$ are

$$
\widehat\psi_a=P_n Q^\star_v(a,W),
\qquad
D_{a,i}=\frac{\mathbb 1\{A_i=a\}\Delta_i}{g_v(a\mid W_i)\pi_v(a,W_i)}
  \{Y_i-Q^\star_v(a,W_i)\}+Q^\star_v(a,W_i)-\widehat\psi_a.
$$

For a continuous outcome, $Y$ enters on the scale that `q_bounds` fixes, and the fit reports the
original units. Each `ate`, `rr`, and `or` is a function of two coordinates of the joint vector.
Its curve is the delta-method combination of the two same-row arm curves, with ratios on the log
scale. The gradient and the Hessian separate by arm, but the line search and the stopping rule act
on the whole vector. The joint fit therefore equals separate per-arm fits to solver tolerance only.

**The centered rule and the band.** Each estimate declares the `"centered"` covariance rule. The
variance is $\operatorname{var}(D_a)/n$ with the $n-1$ denominator. `covariance()` and
`contrast()` use the sample covariance of the same-row curves. See
[covariance rules](inference.md#covariance-rules). The stacked natural-course mean keeps the
`"second_moment"` rule. The two rules agree to first order.

`Inference(simultaneous=True)` is the default. The band draws multipliers on the matrix of centered
curves. The default multipliers are Rademacher, and `multiplier_kind="mammen"` or `"normal"` also
fits. The band takes the max-t quantile of those draws, scaled by the pointwise standard errors.

The band rests on two results. The first is the joint expansion of Zheng and van der Laan
(2011), Theorem 2. The second is a conditional multiplier central limit theorem for a fixed number
of estimands. No reviewed paper states this band. The
[audit record](../references.md#point-treatment-and-stochastic-interventions) gives the support
chain for each step.

**The preflight.** After fold generation and before any learner call, the fit checks the minimum
content below. A failure raises `DataError`, and the message names the remedy in the last column.
In that column, "in sample" is `CrossFitting(enabled=False)`. The engine form is
`cross_fit=False`.

| scope | minimum content | remedy the message names |
| --- | --- | --- |
| sample | two respondents, two nonrespondents, and two respondents in each arm | fit in sample |
| sample, binary outcome | two respondents with each outcome | fit in sample. With no respondent at one outcome, no remedy: an in-sample fit sees the same single class |
| sample, each role whose learner is a package `SuperLearner` with a classification task | three rows in each class of the role target | replace that learner. With exactly two rows in the class, the message also names fit in sample |
| each training complement | one respondent, one nonrespondent, one row in each arm, and one respondent in each arm | increase `n_folds`, use a different `random_state`, or fit in sample |
| each training complement, binary outcome | both outcome classes among the respondents | the same as the row above |
| each training complement, each role whose learner is a package `SuperLearner` with a classification task | two rows in each class of the role target | the same as the row above |

The role targets are the outcome among respondents, the response indicator, and the treatment. The
outcome target is the scaled outcome that the fit trains on. A complement failure names the repeat
and the fold.

Each sample row is the least count that some partition can satisfy. A class with $c$ rows puts at
least one row in some validation fold, so that fold's complement holds at most $c - 1$. With
`n_folds` equal to $n$, every complement drops exactly one row and holds $c - 1$. A complement
minimum of $k$ therefore needs $c \ge k + 1$ in the sample, and no fold count or `random_state`
rescues a smaller sample. The comment above the check in `src/cleverly/estimators/tmle.py` gives
the same argument.

A `SuperLearner` fitted in sample runs its own stratified split over all $c$ rows. That split
needs two rows in each class, so the in-sample remedy holds for a Super Learner class only at
$c = 2$.

The last row follows from the package `SuperLearner`. For a classification task, its inner split
stratifies on the learner target (`src/cleverly/learners/super_learner.py`), and the fold resolver
raises when a class has one member (`src/cleverly/learners/crossfit.py`). With two rows in each
class, every inner training set holds both classes. A `SuperLearner` with `task=None` counts when
it infers a classification task from its target. The default learner for a binary role is a
classification `SuperLearner`, so the rule applies when you pass no learner.

The preflight reads the resolved learner of each role and fits nothing. It cannot see a
`SuperLearner` nested inside a user pipeline or another wrapper. That learner can still fail
inside the fit with too few rows in a class.

**Sources and evidence.** Díaz and van der Laan (2017), Section 2.1 and Equation (1), give the
per-arm curve. Gruber and van der Laan (2012), Section 2.3, give the clever covariate for a
generic arm, and their Appendix A gives each contrast. Zheng and van der Laan (2011), Section 2
and Theorem 2, give the vector parameter and the training-complement construction. Levy (2018)
gives the stacked update and the whole-sample plug-in. The
[audit record](../references.md#point-treatment-and-stochastic-interventions) gives each locator.

| evidence | what it checks |
| --- | --- |
| [stacked arm-indexed missing-outcome CV-TMLE study](method-evidence/stacked-arm-indexed-missing-outcome-cvtmle.md) | truth tests, paired R `tmle` 2.1.1 comparisons, calibration, band joint coverage, union-model cells, and overfitting controls on two-arm and three-arm laws |
| `tests/unit/test_arm_indexed_stacked_mar.py` | each refusal and preflight row with zero learner calls, and nonzero witnesses with deliberate mutations for the fluctuation, the mask, both inverse factors, the contrasts, the covariance, the band, the held-out scale, and training-set leakage |

### Weights, strata, and clusters

Fixed observation weights define the tilted target law $dP_w=w\,dP/E_P(w)$. They are used in the
nuisance losses, in targeting, in the plug-in average, and in the influence-curve covariance.
The registered [weighted point-treatment study](method-evidence/weighted-point-treatment-tmle.md)
checks that construction against R `tmle` and includes an omitted-weight control that recovers the
exact selected-population target.
The [learned-nuisance weighted study](method-evidence/learned-weighted-point-treatment-tmle.md)
also fits both nuisance regressions with those weights. Its learner-only control separates the
target and selected plug-ins, while weighted targeting repairs the control under a correct
treatment mechanism.
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

so a modified treatment policy has variance at least as large as the regime inducing the same mean.
The added term is a conditional variance, and it is positive wherever the shift moves the dose.
Delegating one to the other omits that term and reports a standard error that is too small.

Theory: Díaz Muñoz and van der Laan (2012), Haneuse and Rotnitzky (2013), and Díaz, Williams,
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
| [implementation validation grid](method-evidence/validation-grid.md) | the registered study row for ordinary point-treatment TMLE |
| [canonical point-treatment TMLE](method-evidence/canonical-point-treatment-tmle.md) | 34 accuracy tests, 17 paired comparisons against R `tmle3`, and 12 theory-property cells, test by test |
| [ordinary missing-outcome TMLE](method-evidence/ordinary-missing-outcome-tmle.md) | observational MAR truth tests, paired R `tmle` comparisons, and the three-nuisance robustness contract |
| [stacked arm-indexed missing-outcome CV-TMLE](method-evidence/stacked-arm-indexed-missing-outcome-cvtmle.md) | cross-fitted two-arm and three-arm MAR means and contrasts, paired R `tmle` comparisons, calibration, a simultaneous band, union-model cells, and overfitting controls |
| [controlled direct-effect TMLE](method-evidence/controlled-direct-effect-tmle.md) | both intermediate levels, paired recoded R `tmle` comparisons, four-nuisance robustness, exact efficiency, and a frozen native-result defect |
| [deterministic regimes](method-evidence/deterministic-point-treatment-regimes.md) | a dynamic rule and static reduction against pinned R `lmtp`, with rule and targeting mutations |
| [known stochastic regimes](method-evidence/stochastic-point-treatment-regimes.md) | a comparator-free known-density study with stochastic-density and targeting controls |
| [continuous modified treatment policies](method-evidence/continuous-modified-treatment-policies.md) | natural, uncapped, and actively capped policies against pinned R `lmtp`, with density-ratio and cap controls |
| [incremental interventions](method-evidence/incremental-propensity-interventions.md) | a treatment-mechanism-dependent curve, including its treatment-score control and a gated R `npcausal` comparison |
| [the evidence manifest](evidence.md#the-table) | which oracle, Gateaux, remainder, and identity instruments each registered target has, and what none of them would see |
