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
complete-data construction follows Díaz Muñoz and van der Laan (2012). Hubbard and van der Laan
(2008), Equation (2), give the two transforms. The missing-outcome natural-course construction is
stated below, and [PAR and PAF with missing outcomes](#par-and-paf-with-missing-outcomes) gives the
attributable stack.

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
| one training complement holds no respondent or no nonrespondent | use the in-sample estimator, or collect more observations at the rare level |

Both messages name the in-sample estimator as `CrossFitting(enabled=False)`. The engine form is
`cross_fit=False`. Change that one setting. The complement message names no redraw. The split is
drawn from the seed alone, so a fold count or a seed that happens to fit was chosen by reading the
values the split must not read.

`stratify_by="none"` is the default, and it is the only policy any fit that draws a split accepts.
A non-collaborative in-sample fit draws no split, so it keeps whichever policy the declaration
carries and applies none of it. The
[fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules) give the refusals in full.

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

No treatment mechanism enters either implementation of the scalar `ey_obs`. Both fits require
iterative unweighted logistic targeting. The treatment may have two or more arms. The covariate of
the natural course is $(A,W)$, so the number of arms changes nothing in the derivation.

The in-sample fit admits fixed analysis weights (`weights=`) and clusters (`id=`), with the rules of
[weights, strata, and clusters](#weights-strata-and-clusters). The stacked fit covers unweighted
iid rows. With three or more arms, the stacked fit also needs two respondents in each arm in the
sample and one in each training complement. It checks this before any learner and raises
`DataError` otherwise.

| fit | outcome | cross-fitting declaration |
| --- | --- | --- |
| ordinary | binary, or continuous with fixed `q_bounds` | `CrossFitting(enabled=False)` |
| stacked | binary | `CrossFitting(enabled=True, n_folds=10, repeats=1, stratify_by="none", targeting_scheme="pooled", fold_evaluation=False, split_plan=None)`. `n_folds` must be 2 or more. The registered study uses 10 |

One fold is the in-sample estimator. `CrossFitting(enabled=True, n_folds=1)` is therefore refused
when the declaration is constructed, with `MethodConfigurationError`. The engine raises
`ValueError` for the same input. The refusal stops that estimate from being reported under the
stacked contract and its second-moment covariance rule.

[Missing-outcome natural-course contracts](scope-and-refusals.md#missing-outcome-natural-course-contracts)
lists every refusal for both fits. The same refusals apply to a fit that reports `ey_obs`, PAR
or PAF beside arm targets. The source for the ordinary fit is
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
mass. Missingness is a design role. Missing adjustment values are not covered. A missing treatment
is covered only when it is declared, as the next paragraph says.

**A missing treatment.** Declare the treatment observation indicator with `treatment_delta=<column>`
on `fit()` or `CausalData`, `treatment_missingness=<column>` on `PointTreatment`, or `DeltaA=` on
`tmle()`. The indicator is 1 where the treatment is recorded. The package does not infer a missing
treatment from a missing value. The fit then runs the composite TMLE for the arm means and their
contrasts. For arm $a$ its clever covariate is $C_a / g_{c,a}$, with the composite indicator
$C_a = \Delta_A \Delta 1\{A = a\}$ and the composite mechanism
$g_{c,a}(W) = P(\Delta_A = 1 \mid W)\,g(a \mid \Delta_A = 1, W)\,\pi(a, W)$. The table gives
the conditions, and the [DR-TMLE contract](dr-tmle/theorem.md#observational-missing-data-the-composite-indicator)
gives the argument.

| condition | statement |
| --- | --- |
| treatment missing at random | $Y(a)$ is independent of $\Delta_A$ given $(A, W)$. The recording may depend on $A$ |
| outcome missing at random | $Y$ is independent of $\Delta$ given $(A, \Delta_A = 1, W)$ |
| composite positivity | $g_{c,a}(W) > 0$ for every arm |
| fit | in sample, `cross_fit=False`; `W` complete on every row |

The recording may depend on the treatment, so $P(A = a \mid W)$ is not identified. The ATT, the
ATC, `ey_obs`, the PAR and the PAF each read the treatment law of every row, and the fit refuses
each one by name. The default and `"all"` target lists drop them. Fixed weights, clusters, baseline
strata and the bootstrap are admitted, with the rules of the next section.

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
Diaz and van der Laan (2017) supplies the randomized-trial missing-outcome construction. The
composite construction is in
[`estimators/composite.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/estimators/composite.py).
[Stacked CV-TMLE for arm-indexed targets](#stacked-cv-tmle-for-arm-indexed-targets) states the
cross-fitted contract and its evidence.

### PAR and PAF with missing outcomes

With missing outcomes, PAR and PAF compare two parameters of one observed law. The reference arm is
$a_0$, and $g(a\mid W)=P(A=a\mid W)$:

$$
\psi_{\mathrm{obs}}=E\{m(A,W)\},\qquad \psi_{a_0}=E\{m(a_0,W)\},\qquad
\operatorname{PAR}=\psi_{\mathrm{obs}}-\psi_{a_0},\qquad
\operatorname{PAF}=1-\psi_{a_0}/\psi_{\mathrm{obs}}.
$$

Each parent is a shipped estimator. The natural-course mean is the estimator above. The arm mean
is the missing-outcome arm mean of the previous section, with the clever covariate
$1\{A=a_0\}/\{g(a_0\mid W)\pi(a_0,W)\}$. The fit solves the two fluctuations separately, from
one initial fit. It then reports every estimate from one stack on the same rows. The table gives
each step.

| step | statement |
| --- | --- |
| stack | both parents are asymptotically linear on the same rows, with curves $D_{\mathrm{obs}}$ and $D_{a_0}$, so the pair is jointly normal with the covariance of the same-row curves |
| PAR | linearity: $D_{\mathrm{PAR}}=D_{\mathrm{obs}}-D_{a_0}$ |
| PAF | the delta method: $D_{\mathrm{PAF}}=-D_{a_0}/\psi_{\mathrm{obs}}+\psi_{a_0}D_{\mathrm{obs}}/\psi_{\mathrm{obs}}^2$ |
| targeting | each parent keeps its own fluctuation. The stack needs only that each parent is asymptotically linear, so separate targeting is valid |
| weights | a fixed weight defines a tilted law, and both curves carry the weight |
| clusters | the cluster is the unit, and both curves are summed within each cluster |

The two points are plug-ins of two different targeted regressions. The report is therefore not one
substitution estimator, and $\hat\psi_{\mathrm{obs}}$ need not equal
$\sum_aP_n[1\{A=a\}\hat m^\star_{\mathrm{mean}}(a,W)]$. Validity does not depend on that
equality. Each coordinate is the shipped estimator: the joint `ey_obs` equals the scalar fit, and
each arm mean equals the arm-only fit (`tests/unit/test_attributable_mar_stack.py`). The
covariance, a contrast and the default band read the cross-covariance of the two curves. On the
registered law their correlation is 0.775.

| condition | from |
| --- | --- |
| iid rows, or independent clusters as the units, with optional fixed analysis weights; one outcome regression $m$ fitted on respondents; $\pi(A,W)$ and $g$ fitted on all rows | both parents |
| missing at random, $Y\perp\Delta\mid A,W$; consistency and $Y(a_0)\perp A\mid W$ for the reference arm | both parents. The natural course needs no treatment exchangeability |
| response positivity $\pi(A,W)>0$, and product positivity $g(a_0\mid W)\pi(a_0,W)>0$ | the natural course; the reference arm |
| each parent's remainder is $o_P(n^{-1/2})$: $E_0[(1-\pi_0/\hat\pi)(\hat m-m_0)]$ and $E_0[(1-g_0\pi_0/(\hat g\hat\pi))(\hat Q-Q_0)(a_0,\cdot)]$ | Díaz, Carone and van der Laan (2016), Equation (3); the arm-mean remainder |
| in sample, the Donsker conditions of each parent. Stacked, each parent's stacked remainder | the parents |
| PAF: a binary outcome, and $\psi_{\mathrm{obs}}$ bounded away from zero | the delta method |

PAR and PAF are consistent if $m$ is consistent, or if both $\hat\pi$ and the product
$\hat g\hat\pi$ at $a_0$ are consistent. A correct product with a wrong $\hat\pi$ rescues the
reference arm and not the natural course. `tests/unit/test_remainder_attributable_mar.py` checks
this case on an exact law. The PAF uses the shipped denominator rule. It raises when
$\hat\psi_{\mathrm{obs}}\le0$, and no other guard is added. For a log-scale interval of
$1-\operatorname{PAF}=\psi_{a_0}/\psi_{\mathrm{obs}}$, take the ratio of the two reported means
with the same-row curves.

The table gives the admitted fits. Each fit is `TMLE` or `TMLEMethod`. A request that reads the
natural course beside arm targets takes this route: `par`, `paf`, or `ey_obs` beside any arm mean,
contrast, or, in sample, the ATT and the ATC.

| fit | admitted |
| --- | --- |
| in sample | two or more arms; binary outcome, or PAR with a continuous outcome and fixed `q_bounds`; fixed `weights=` and `id=` |
| stacked | the binary outcome, two or more arms, unweighted iid rows, and both stacked contracts: this one and the [arm-indexed contract](#stacked-cv-tmle-for-arm-indexed-targets) |

`estimands="all"` keeps its arm-only target list with missing outcomes. A joint fit adds a second
fluctuation, so it is a named request. The refusals keep their reasons:

| composition | reason |
| --- | --- |
| `DRTMLE`, `CTMLE` | no DR-TMLE or collaborative natural-course mean ships. `DRTMLE(guard=())` is the ordinary TMLE, and the message says to use `TMLE`. [F5](../roadmap.md#f5-other-refused-c-tmle-and-dr-tmle-compositions) tracks the refused DR-TMLE compositions |
| stacked baseline strata | the second-moment variance term of the stacked estimator has no stratum form ([F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants)). The in-sample fit admits strata |
| `n_bootstrap > 0` | no audited bootstrap result covers this fit ([F2](../roadmap.md#f2-targeted-bootstrap-inference)) |
| stacked `weights=` or `id=`, stacked continuous outcome, fold targeting, fold evaluation, repeats | the stacked contracts cover unweighted iid rows and a binary outcome ([F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants)) |

The registered
[missing-outcome attributable-effect TMLE study](method-evidence/missing-outcome-attributable-effects-tmle.md)
checks the stack on two laws, both robustness halves, the product-only case, calibration of PAR and
PAF, the default band, and fixed weights and clusters. It pairs every scenario with R `tmle` 2.1.1.

### Stacked CV-TMLE for arm-indexed targets

Ordinary TMLE cross-fits arm-indexed means and contrasts with missing outcomes under one stacked
CV-TMLE contract. The contract applies to a fit when all of the conditions below hold
(`_is_arm_indexed_missing_crossfit` in `estimators/tmle.py`):

- the outcome is missing for at least one row;
- `CrossFitting(enabled=True)`;
- the parameters are indexed by the arms of a discrete treatment;
- the design declares no intermediate; and
- the request is not the scalar `NaturalCourseMean`.

The shift, incremental, regime, MSM, and controlled-direct-effect fits are outside this contract.
The scalar natural-course mean has its own contract above. A fit that reports `ey_obs`, PAR or PAF
beside arm targets meets both contracts, and it may also request `ey_obs`, `par` and `paf`.

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

`CrossFitting` defaults to `stratify_by="none"`, so this contract's fold policy needs no setting.
The default estimand list of a two-arm fit includes `att` and `atc`, so a two-arm fit must name its
estimands. Unstratified folds and a prespecified `q_bounds` are not special to this contract. The
[fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules) apply them to every
cross-fitted fit, because the audit found no result for treatment-stratified folds and none for an
outcome scale taken from held-out rows.

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
| each training complement | one respondent, one nonrespondent, one row in each arm, and one respondent in each arm | fit in sample, or collect more observations at the rare level |
| each training complement, binary outcome | both outcome classes among the respondents | the same as the row above |
| each training complement, each role whose learner is a package `SuperLearner` with a classification task | two rows in each class of the role target | the same as the row above |

The role targets are the outcome among respondents, the response indicator, and the treatment. The
outcome target is the scaled outcome that the fit trains on. A complement failure names the repeat
and the fold. It names no redraw, because the split reads no treatment and no outcome, and a fold
count or a seed that happens to fit was chosen by reading them.

A cross-fitted fit outside this contract runs the same three questions under
`_preflight_training_support`. It asks that each arm appear in at least two independent units,
that each training complement hold every arm and both observed outcome classes of a binary
outcome, and that a package classification `SuperLearner` hold two rows of each class in each
complement. The independent unit is the row, and it is the *cluster* when `id=` declares one. The
[fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules) give every message.

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

`strata=` produces stratum-specific parameters. A stratum parameter is the marginal parameter of
the law given $S=s$, for a fixed partition $S$ of the adjustment columns with $P(S=s)>0$. Every
target group solves one score block $I(S=s) H_s / P_n(S=s)$ for each equation of its targeting.
Each stratum curve is $I(S=s) D_s / P_n(S=s)$. The table gives the construction of each group.

| target group | stratum construction |
| --- | --- |
| arm means and contrasts, ATT, ATC, PAR, regimes, shifts, identity-link MSMs, the in-sample natural-course mean | the blocks in one pooled fluctuation |
| incremental interventions (`incremental=`) | outcome blocks and treatment-mechanism blocks, alternating until both settle |
| MSMs with a log or logit link | the marginal coefficients from the unstratified solve, and the stratum coefficients from a nested fluctuation of the blocks at each stratum's coefficients, recorded on `Fluctuation.stratified` |
| continuous-dose MSMs | as the identity or linked MSM above; a dose has no arm share |
| `DRTMLE` at a non-empty `guard` | every equation and tilt in blocks, with every reduced regression fitted inside each stratum |

The marginal estimate of an alternating group is the $P_n(S=s)$-weighted mixture of the stratum
estimates. A linked MSM is the exception: its marginal coefficients come from their own
fluctuation, because each stratum block reads that stratum's coefficients. The stratum
coefficients are the coefficients of one MSM with the expanded design $(I(S=s)\varphi)_s$, whose
projection loss separates by stratum. The marginal estimate of a stratified `DRTMLE` fit reduces
on $(g_n(W), S)$, so it differs from the unstratified fit. The
[DR-TMLE estimands page](dr-tmle/supported-estimands.md) states why it stays valid.

The extension is the finite-partition step of each base result: Kennedy (2019), Theorem 2, for
the incremental curve; the package's least-squares projection for MSMs; Díaz, Carone and van der
Laan (2016) for the natural course; Benkeser et al. (2017), Theorem 1, for DR-TMLE. Each stratum
inherits the base result's conditions inside the stratum. The
[baseline-strata study](method-evidence/stratified-point-treatment-tmle.md) measures the arm
targets against an exact law and pairs them with R `tmle3` `tmle_stratified`. Exact-law tests pin
each new construction: `tests/unit/test_stratified_incremental_exact.py`,
`tests/unit/test_stratified_msm_exact.py`, `tests/unit/test_stratified_continuous_msm_exact.py`,
`tests/unit/test_stratified_natural_course_exact.py` and
`tests/unit/test_stratified_drtmle_exact.py`.

These requests with `strata=` refuse before any learner:

| request | reason |
| --- | --- |
| `targeting_scheme="fold"` | each stratum would need a fold-local update, and no published result covers one |
| `cv_evaluation=True` | the fold-evaluated estimate needs stratum shares and a stratum-indexed fold average in each validation fold ([X28](../roadmap.md#x28-fold-evaluated-cv-tmle-with-baseline-strata)) |
| an incremental target with a stratum that lacks a treatment arm | the stratum's treatment-mechanism equation has no finite root |
| an MSM whose design is singular inside a stratum | the stratum projection is not one coefficient vector; the shipped rank rule decides |
| a `DRTMLE` stratum with no trainable rows of an arm in some training complement | its reduced regressions cannot be fitted inside the stratum |
| the stacked natural-course mean | its second-moment variance term has no stratum form ([F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants)) |

`cluster=` changes the independent unit for covariance and fold construction, and it does not
change the estimand.

A fit with a declared missing treatment admits fixed weights, clusters and baseline strata, as the
missing-outcome TMLE does. The weights tilt the law, and the composite mechanism is the tilted
law's mechanism, because every factor is fitted with weighted loss. A cluster stays the unit of
the covariance. Each stratum's score block uses the composite covariate.

A grouped fold draw permutes the distinct cluster labels and cuts them into near-equal parts, so
every row of a cluster lands in one fold. The cluster is then the independent unit the preflight
counts, so each arm must appear in two distinct clusters. See
[how every method reports uncertainty](inference.md#clusters) and the
[fold and outcome-scale rules](cv-tmle.md#fold-and-outcome-scale-rules).

## Variations

### Estimator options

| option | what it changes | is it a different estimator? |
| --- | --- | --- |
| `fluctuation="logistic"` or `"linear"` | the submodel the outcome regression is fluctuated along | no. Both solve the same score |
| `algorithm="iterative"` or `"one_step"` | Newton iteration, or the universal least-favorable submodel of van der Laan and Gruber (2016) | no. The two are pinned to agree by an exact identity |
| `target_weights=` | whether the fluctuation carries the covariate as a weight or as a regressor | no. The weighted and clever-covariate forms solve the same equation |
| `g_bounds=` | the treatment-mechanism truncation. `"auto"` is target-aware | it changes the finite-sample procedure. Report it with the support diagnostics |
| `q_bounds=`, `submodel_alpha=` | the outcome scaling, and the logistic submodel bound | no. A cross-fitted fit of a continuous outcome refuses `q_bounds=None` |
| `screen_treatment=`, `screen_threshold=`, `min_retain=` | covariate screening for the treatment mechanism | no |
| `cross_fit=`, `n_folds=`, `repeats=` | sample splitting for the nuisances | **yes.** See [CV-TMLE](cv-tmle.md). `cross_fit=True` with fewer than two folds is refused at construction |

`submodel_alpha` bounds the logistic submodel. `alpha` is the interval's significance level. The
two are separate keywords because they once shared a name and a fit read one as the other.

### Known regimes

For a known stochastic intervention $g^*(a\mid w)$,

$$
\psi_{g^*}(P)=E_P\left\{\sum_a Q_P(a,W)g^*(a\mid W)\right\}.
$$

Its efficient influence function has residual weight $g^*(A\mid W)/g_P(A\mid W)$ and a centered
plug-in term. Deterministic static and dynamic rules are degenerate cases of $g^*$. Identification
needs positivity only where the regime assigns mass. A fixed rule does not depend on $P$, so its
influence function carries no term for learning the rule.

`Rule` asks for this condition as a declaration: pass `rule_kind="known"`. The package refuses an
undeclared rule, and a rule declared `"estimated"`, when the rule is built. `TMLE` refuses both
again before any learner. For a threshold at a sample mean, the
[scope page](scope-and-refusals.md#wrong-by-construction) gives the ratio of the reported to the
exact standard error on one exact law.

A user-written `Intervention` class declares `density_kind = "known"` as an attribute. `TMLE`
refuses the class before any learner when that attribute is missing, `None`, or `"estimated"`.
Code cannot inspect a closure, so a false `"known"` declaration on a rule or a class still fits.
[RM28](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm28-declared-densities-of-user-written-interventions) records the decision.

`Stochastic` asks for this condition as a declaration: pass `density_kind="known"`. The package
refuses an undeclared density, and a density declared `"estimated"`, when the regime is built.
`TMLE` refuses both again before any learner. For a population-law odds tilt of the mechanism,
the [scope page](scope-and-refusals.md#wrong-by-construction) gives the omitted variance term on
one exact law. A realized learned density defines a different target. Its inference needs
conditions that the `Stochastic` API does not check.

`Static`, `Rule`, and `Stochastic` implement the intervention protocol. `RegimeMean` and
`RegimeContrast` define the levels and the contrasts. See Robins (2004), Diaz Munoz and van der
Laan (2012), and Diaz and van der Laan (2013). Implementation:
[`interventions/base.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/base.py)
and
[`interventions/support.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/support.py).

### Learned rules

`LearnedRuleValue` estimates the value of a treatment rule that the fit learns. The engine
spelling is `TMLE(learned_rule=LearnedRule())`. Van der Laan and Luedtke (2015), Section 7, define
the target, and their Appendix B gives its CV-TMLE. The target is a data-adaptive parameter. It
depends on the realized split and on the rules that the fit learns:

$$
\tilde\psi_{0n} = \frac1V \sum_{v=1}^V \Psi_{d_{nv}}(P_0),\qquad
\Psi_d(P) = E_P\,\bar Q_P(d(W), W),\qquad
d_{nv}(w) = \mathbb 1\{\bar Q_{nv}(1, w) - \bar Q_{nv}(0, w) > 0\}.
$$

Here $\bar Q_{nv}$ is the outcome regression that the fit trains on the training rows of fold
$v$. A tie assigns control. The target is not the value of one rule fitted on all rows, and it is
not the value of the optimal rule.
[X11](../roadmap.md#x11-learned-policy-follow-ups) parts (a) and (f) hold those two targets.

The fit reuses the regime fluctuation and the fold-evaluated CV-TMLE. The table gives each step.

| step | what the fit computes | source |
| --- | --- | --- |
| rule | $d_i = d_{n,v(i)}(W_i)$ from the out-of-fold outcome regression, so no fit that saw row $i$ sets its rule | JCI, Section 7.1 |
| clever covariate | $H_i = \mathbb 1\{A_i = d_i\} / g_{n,v(i)}(A_i \mid W_i)$, with $g$ truncated at `g_bounds` | JCI, Appendix B |
| fluctuation | one $\varepsilon$ on the pooled validation rows, with the weights $n / (V n_v)$ | JCI, Appendix B, Equation (21) |
| estimate | $\psi^*_n = (1/V) \sum_v \psi^*_{nv}$, where $\psi^*_{nv}$ is the mean of $\bar Q^*_{nv}(d_i, W_i)$ over the validation rows of fold $v$ | Montoya (2023b), Section 3.2 |
| influence curve | $D_i = H_i\,(Y_i - \bar Q^*_{nv}(A_i, W_i)) + \bar Q^*_{nv}(d_i, W_i) - \psi^*_{nv}$, centred at each fold's own estimate | Montoya (2023b), Section 4.2 |
| variance | $V^{-2} \sum_v n_v^{-2} \sum_{i \in v} D_i^2$ | JCI, Section 7.2, at equal folds |
| interval | the Wald interval at the 0.975 normal quantile | Montoya (2023b), Section 4.2 |

"JCI" is van der Laan and Luedtke (2015). "Montoya (2023b)" is Montoya, van der Laan, Skeem and
Petersen (2023), *International Journal of Biostatistics* 19(1):239–259. The
[references](../references.md#point-treatment-and-stochastic-interventions) give every locator
and the version of each source that this project read.

The conditions come from JCI, Theorem 6 and Corollary 3. The data cannot confirm C2 to C4.

| id | condition |
| --- | --- |
| C1 | a bounded outcome: a binary outcome, or a continuous outcome with a declared `q_bounds` |
| C2 | strong positivity. `g_bounds` regularizes the fitted denominator only |
| C3 | a limiting rule: the fold curves converge to the curve of one fixed rule |
| C4 | the fold-average remainder is $o_P(n^{-1/2})$. Alternatively, Corollary 3 requires Theorem 6 with $g=g_0$, negligible remainder replacement, and the linearization in Equation (14). Under these conditions, each fold's maximum-likelihood mechanism fit in a correct model gives a conservative interval |
| C5 | no Donsker condition, so the learners can be flexible |
| C6 | iid rows, with no weights, clusters, missing outcomes or intermediate variable |

C3 fails at an exceptional law, where the effect is zero for a share of units and the learner is
consistent. The fold rules then have no fixed limit, and the interval can under-cover. Two
registered studies measure the interval. The table gives their readings.

| study | laws | reading |
| --- | --- | --- |
| [gated study](method-evidence/learned-rule-cvtmle.md) | `non_exceptional` and `misspecified_limit`, which meet C3 | every primary test and every property cell passes. Coverage is 0.9465 and 0.9413 over 6,000 replications at n = 2,000 |
| [boundary study](method-evidence/learned-rule-cvtmle-boundary.md) | `exceptional`, where C3 fails, and `weak_blip`, within C3 | the interval under-covers at each law. Coverage is 0.8938 and 0.8948, and the SE ratio is 0.8211 and 0.8314 |

The fit needs `CrossFitting(enabled=True, fold_evaluation=True)`, one repeat, pooled targeting and
no full-refit bootstrap. `refuse_learned_rule_composition` in
[`interventions/learned.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/learned.py)
refuses every other composition before any learner, in the order that the table gives. A refusal
that no setting repairs comes before a refusal whose remedy is a setting, as the
[RM24](https://github.com/esbraun/cleverly-tmle/blob/4ce96cda2bda93ba9233026977e3ff63ea3e0003/docs/roadmap.md#rm24-refusals-after-the-nuisance-fit) order rule requires. For
`LearnedRuleValue`, `CausalStudy.identify` runs the rows that the data decide, and
`IdentifiedEffect.estimate` runs the rows that the settings decide.

| order | request | item |
| ---: | --- | --- |
| 1 | `CTMLE` or `DRTMLE` with `learned_rule=` | [F27](../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) |
| 2 | `learned_rule=` beside `interventions=`, `shifts=`, `incremental=`, `msm=` or `reference=`, or beside an arm estimand | [X11](../roadmap.md#x11-learned-policy-follow-ups) (c) for `interventions=`, `reference=` and an arm estimand. [F17](../roadmap.md#f17-joint-point-treatment-parameter-axes) for `shifts=`, `incremental=` and `msm=` |
| 3 | a continuous treatment | F27 |
| 4 | a treatment with more than two arms | X11 (d) |
| 5 | missing outcomes, `delta=` | [F21](../roadmap.md#f21-other-missing-outcome-cv-tmle-variants), with a learned-rule text that runs before the F21 refusal |
| 6 | `intermediate=` | F27 |
| 7 | `weights=` | F27 |
| 8 | `id=` | F27 |
| 9 | `strata=` | F27 |
| 10 | `cross_fit=False`, or one fold | X11 (a) |
| 11 | `cv_evaluation=False` | X11 (e) |
| 12 | `targeting_scheme="fold"` | X11 (h) |
| 13 | `repeats` above 1 | F27 |
| 14 | `n_bootstrap` above 0, the full-refit bootstrap | F27 |

Rows 1 and 2 run when the estimator is built. `CrossFitting`, `TMLEMethod` and the `TMLE`
constructor also refuse a fold policy that the shared rules forbid, before the data exist. On a
learned-rule `TMLE`, rows 10 to 14 replace such a refusal where one of them applies.
[The refusals a caller can meet](cv-tmle.md#the-refusals-a-caller-can-meet) quotes each remedy.

The result records the fit under `result.extra["learned_rule"]`, a `LearnedRuleRecord`. The
table gives its fields.

| field | content |
| --- | --- |
| `fold_sizes`, `fold_weights` | the validation rows of each fold, and the weight $1/V$ of each fold |
| `fold_estimates` | $\psi^*_{nv}$ for each fold |
| `treated_shares` | the share of each fold's validation rows that its rule assigns to the higher arm code |
| `blip_quantiles`, `quantile_levels` | quantiles of the estimated blip on each fold's validation rows |
| `rule`, `target` | the rule class and the target kind, in words |

`result.summary()` prints the target line of `LearnedRuleRecord.describe()`. The rule of each row
is `result.nuisance.regimes.values[:, 1, 0]`. The fit keeps no fitted fold model, and the saved
folds, learner templates and seeds replay it.

A refit relearns the rules, so `refute` reads `unavailable`, and the full-refit bootstrap and
`repeats` above 1 refuse. Every sensitivity analysis reads `unavailable` for another reason: no
derivation for this target was reviewed.
[F27](../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) holds the
sensitivity analyses. The support report and `truncation_curve` run, because they read the same
rules.

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

**The standard error reads the estimated density ratio.** When that ratio is far from the true
ratio, the standard error does not describe the spread of the TMLE. In the probe below, a flexible
density fitted out of fold overstates the spread. An overfit density fitted in sample understates
it. The table gives
the mean standard error over the empirical SD for the `+1.0` uncapped shift. The data are
`make_shift_dose(n=3000)` on 60 seeds, with three folds when cross-fitted. The outcome learner is
`HistGradientBoostingRegressor` in every row.

| density learner | bins | fit | mean SE / empirical SD |
| --- | --- | --- | --- |
| `HistGradientBoostingClassifier`, default settings | 40 | cross-fitted | 3.91 |
| `HistGradientBoostingClassifier`, default settings | 40 | cross-fitted, ratio trimmed at its 0.999 quantile | 2.46 |
| `HistGradientBoostingClassifier`, default settings | 40 | in sample | 0.76 |
| exact law | 320 | cross-fitted | 0.98 |

The probe build trimmed the ratio at its 0.999 quantile in the last three rows. The shipped
estimator applies no bound to the ratio, and the first row is its behavior. The trim bound 3 of
3000 rows on seed 9000. The influence curve matched an independent recomputation to 5e-12 in
every fit, so the cause is the ratio and not the formula.

The table gives the expected ratio at the observed dose under the true density.
An upper cap alone does not preserve conditional support for negative shifts or support gaps.

| policy condition | expected ratio under the true density |
| --- | --- |
| any capped or uncapped shift | $P(d(A,W) \in \operatorname{supp} g(\cdot \mid W))$ |
| policy preserves conditional support | 1 |

`diagnostics.support()` reports that mean as `mean_ratio` and its value on the rows each fold
holds out as `fold_mean_ratio`. The `+1.0` probe shift is uncapped. The `make_shift_dose` dose is
normal given the covariates, so the reference mean is 1. On seed 9000 the cross-fitted booster
reads fold means 1.46, 1.20 and 1.87. The exact law reads 1.03, 0.98 and 1.07.

A fold mean far from its reference value can signal density estimation error, but sampling variation also affects it.
Estimated zero densities at assigned doses flag model support failures.
These diagnostics do not establish true conditional support or identification.

No source here calibrates a threshold, so the report gives the number and no warning. No
registered study covers an estimated or cross-fitted shift density.

Evidence: the probe scripts and logs in `reviews/notebook-review/probes/iv-n3/` (`probe_ic.py`,
`summary_ic.log`, `fold_check.log`), the analysis in
`reviews/notebook-review/investigations/iv-n3.md`, and `tests/unit/test_shift_ratio_mean.py`.

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

With `strata=`, the outcome and mechanism equations each take one block per stratum, so each
stratum's mechanism is tilted along its own covariate. A stratum that holds no row of some arm
refuses before any learner, because its mechanism equation then has no finite root. Kennedy (2019),
Section 6, lists "how mean outcomes under different interventions vary with covariates" as future
work. That is a conditional curve in the covariates. A fixed finite partition is the marginal
result applied inside each stratum, so the passage records no objection to it.
`tests/unit/test_stratified_incremental_exact.py` checks each stratum curve against Theorem 2.

Kennedy (2019) is the primary theory reference. Implementation:
[`interventions/incremental.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/incremental.py)
and
[`fluctuation/mechanism.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/fluctuation/mechanism.py).

### Support reports

Arm positivity, regime support, shift support, and incremental support answer four different
questions. The intervention classes therefore expose distinct reports rather than reducing every
overlap question to one propensity histogram.

The regime support table reads two mechanisms. `min g`, `max ratio` and `ratio effective n` read
the treatment mechanism before truncation, which is what the data support. `score load` reads the
clever covariate of the targeting step, which divides by the truncated mechanism. The table prints
both facts under its rows. A bound that binds makes the two counts differ.
`TestTheSupportTableStatesEachColumnsBasis` in `tests/unit/test_summary_and_message_accuracy.py`
checks each count against its own mechanism at such a bound.

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
