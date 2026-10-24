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
| you expect one nuisance to be inconsistent and still want an interval | a declared [known treatment mechanism](#known-treatment-mechanism), or [DR-TMLE](dr-tmle/index.md), whose one-wrong interval is established only in [the listed cases](dr-tmle/theorem.md#the-conditioning-terms-are-linear-in-one-error) |

Three worked applied analyses cover this entry. The
[point-treatment tutorial](../examples/point-treatment-tmle.ipynb) is the main one, and it also
demonstrates the conditional-population effects below. The intervention axes have their own
tutorial, [choosing a navigation policy](../examples/interventions.ipynb). Missing outcomes have theirs in
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
$\hat\psi_{\mathrm{obs}}\le0$, and no other guard is added. `res.ratio("ey0", "ey_obs")` gives the log-scale interval of
$1-\operatorname{PAF}=\psi_{a_0}/\psi_{\mathrm{obs}}$ from the same-row curves.

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
| `DRTMLE`, `CTMLE` other than the per-arm `oat` design | no DR-TMLE natural-course mean ships, and no selector-based or shared-design collaborative one. The per-arm `oat` design reports a point estimate. Its observation factor is `P(Delta = 1 \| A, Qbar(A, W))`, read at the realised arm ([collaborative TMLE](collaborative-tmle.md#baseline-strata-and-missing-data-in-the-per-arm-design)). `DRTMLE(guard=())` is the ordinary TMLE, and the message says to use `TMLE`. [X50](../roadmap.md#x50-further-dr-tmle-compositions) part (c) tracks the DR-TMLE natural-course mean, and [X51](../roadmap.md#x51-c-tmle-point-estimates-for-further-targets) part (a) the C-TMLE point estimates |
| stacked PAR, PAF, or `ey_obs` beside arm targets, with baseline strata | the arm-indexed stacked contract runs baseline strata under the experimental label. The finite-partition step qualifies, and it needs a witness and a registered cell ([X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants) part (e)) |
| `n_bootstrap > 0` | the fit runs and carries the experimental label. [X55](../roadmap.md#x55-bootstrap-extensions) part (a) holds the in-sample fit, and [F2](../roadmap.md#f2-targeted-bootstrap-inference) the stacked fit |
| stacked `weights=` or `id=`, stacked continuous outcome, fold evaluation, repeats | the fit runs and carries the experimental label. The stacked contracts cover unweighted iid rows and a binary outcome ([X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants) parts (a), (b) and (e)). Fold targeting stays refused |

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
| arm means and contrasts, ATT, ATC, PAR, regimes, shifts, identity-link MSMs, the natural-course mean in sample and stacked | the blocks in one pooled fluctuation |
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
[DR-TMLE estimands page](dr-tmle/supported-estimands.md) states the conditions under which it is
valid.

The extension is the finite-partition step of each base result: Kennedy (2019), Theorem 2 of
arXiv v3, for the incremental curve; the package's least-squares projection for MSMs; Díaz,
Carone and van der Laan (2016) for the natural course; Benkeser et al. (2017), Theorem 1, for
DR-TMLE. Each stratum inherits the base result's conditions inside the stratum. The
[baseline-strata study](method-evidence/stratified-point-treatment-tmle.md) measures the arm
targets against an exact law and pairs them with R `tmle3` `tmle_stratified`. Exact-law tests pin
each new construction: `tests/unit/test_stratified_incremental_exact.py`,
`tests/unit/test_stratified_msm_exact.py`, `tests/unit/test_stratified_continuous_msm_exact.py`,
`tests/unit/test_stratified_natural_course_exact.py` and
`tests/unit/test_stratified_drtmle_exact.py`. Three registered studies measure them: the
[stratified incremental MSM study](method-evidence/stratified-incremental-msm.md), the
[stratified identity-link MSM study](method-evidence/stratified-msm-identity.md), and the
[stratified DR-TMLE study](method-evidence/stratified-dr-tmle.md).

These requests with `strata=` refuse before any learner:

| request | reason |
| --- | --- |
| `targeting_scheme="fold"` | fold targeting fits the fluctuation on the validation rows. Montoya et al. (2023), Section 3.2, Step 2(c), publish that update, and the package does not implement it with strata. [X61](../roadmap.md#x61-fold-targeting-and-fold-evaluation) part (b) holds this composition |
| `cv_evaluation=True` | the fold-evaluated estimate needs stratum shares and a stratum-indexed fold average in each validation fold ([X28](../roadmap.md#x28-fold-evaluated-cv-tmle-with-baseline-strata)) |
| an incremental target with a stratum that lacks a treatment arm | the stratum's treatment-mechanism equation may have no finite root. It has none only if some direction of the covariate separates, which depends on the fitted outcome regression, so the guard reads the data alone |
| an MSM whose design is singular inside a stratum | the stratum projection is not one coefficient vector; the shipped rank rule decides |
| a `DRTMLE` stratum with no trainable rows of an arm in some training complement | its reduced regressions cannot be fitted inside the stratum |
| a stacked PAR, PAF, or `ey_obs` beside arm targets | the arm-indexed stacked contract refuses strata. The step qualifies, and it needs a witness and a registered cell ([X53](../roadmap.md#x53-missing-outcome-cv-tmle-variants) part (e)) |

`cluster=` changes the independent unit for covariance and fold construction, and it does not
change the estimand. The estimand is the row-weighted mean $\mu_I$ of Wang, Park, Small and Li
(2024), Section 2, at equal or unequal cluster sizes. Pass `weights` equal to one over the
cluster size to estimate the cluster-average mean $\mu_C$. Below 40 clusters with positive weight
mass the interval uses a Student $t$ reference, and below 10 the fit withholds it
([clusters](inference.md#clusters)).

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
| `stopping=` | the rule that decides the score equation is solved. `"relative"` is the shipped test | it changes when the targeting stops and how the result judges the residual. See [stopping rules](#stopping-rules-step-bound-and-best-iterate) |
| `step_bound=`, `step_search=` | the longest step of one iteration, and whether a line search sets its length | it changes the path of the iteration. The score equation is the same |
| `iterate=` | which iterate of the walk the fit reports | it changes the reported fit only when the walk oscillates |
| `outcome_by_arm=` | one outcome regression for each treatment arm | it changes the nuisance estimator. See [per-arm outcome regression](#per-arm-outcome-regression) |
| `treatment_adjustment=` | the adjustment columns the treatment mechanism reads | **yes.** It restricts the statistical model. See [treatment-mechanism covariates](#treatment-mechanism-covariates) |
| `cross_fit=`, `n_folds=`, `repeats=` | sample splitting for the nuisances | **yes.** See [CV-TMLE](cv-tmle.md). `cross_fit=True` with fewer than two folds is refused at construction |

`submodel_alpha` bounds the logistic submodel. `alpha` is the interval's significance level. The
two are separate keywords because they once shared a name and a fit read one as the other.

### Stopping rules, step bound and best iterate

`Targeting` carries four options of the `tmle3` updater. Each default runs the shipped code path
unchanged.

| option | values | `tmle3` option | what it decides |
| --- | --- | --- | --- |
| `stopping=` | `"relative"` (default), `"scaled_var"`, `"sample_size"` | `convergence_type` | when the score equation counts as solved |
| `step_bound=` | `None` (default) or a positive number | `constrain_step`, `delta_epsilon` | the longest step of one outer iteration |
| `step_search=` | `True` (default) or `False` | `optim_delta_epsilon` | whether a line search sets the length of a bounded step |
| `iterate=` | `"last"` (default), `"best"` | `use_best` | which iterate the fit reports |

The tests in `tests/unit/test_stopping_rule.py`, `test_step_bound.py` and `test_best_iterate.py`
assert that a written-out default gives the same fit as an omitted option. The
[architecture invariants](../architecture-invariants.md#option-defaults-keep-the-shipped-code-path)
state the rule that holds the defaults.

#### The three rules

A rule compares the score of each fluctuation column with a threshold. Let $n_u$ be the number of
distinct units among the rows the solver receives. A unit is a cluster when `id=` is set, and a
row otherwise. Let "span" be the range of the outcome.

| rule | statistic of column $j$ | threshold |
| --- | --- | --- |
| `relative` | $\lvert P_n S_j\rvert / \operatorname{mean}\lvert w h_j\rvert$ | `tol`. This is the shipped test |
| `sample_size` | $\lvert P_n S_j\rvert \cdot \text{span}$ | $1/n_u$ |
| `scaled_var` | $\lvert P_n S_j\rvert \cdot \text{span}$ | $\mathrm{sd}_j / (\sqrt{n_u}\,\min(\log n_u, 10))$ |

The fit meets a declared rule when the largest ratio of statistic to threshold is at most one.
`Fluctuation.rule_ratio` holds that ratio, and `Fluctuation.thresholds` holds the thresholds.
Both are `None` under `relative`.

The standard deviation $\mathrm{sd}_j$ is the sample standard deviation, with `ddof=1`, of the
per-unit curve $IC_{uj} = (n_u / n)\sum_{i \in u} c_{ij}$. The contribution $c_{ij}$ depends on the
column.

| column | contribution $c_{ij}$ |
| --- | --- |
| one arm mean of the `mean` group, with unit weights | the full arm curve $\text{span}\,\{w h (Y - Q^*) + Q^*_a(W) - P_n Q^*_a\}$, as `tmle3` uses |
| every other column, and every weighted fit | the score column $\text{span}\, w h (Y - Q^*)$ |

The second row is a declared difference from `tmle3` (D4). The score column has no
$Q^*_a(W)$ term, so the threshold of such a column is about the one an arm column would have.
`tests/unit/test_stopping_rule.py` computes each threshold by hand with `pandas` group sums for a
clustered continuous fixture.

The rule decides the outer test of the solver. The inner Newton step keeps `tol=min(tol, 1e-12)`.
On an unbounded iterative fit the rule therefore changes `converged`, `failure`, `n_iter` and the
warning. It moves the estimate when the shrink binds after the first step, under `step_bound=`,
and under `algorithm="one_step"`. The witnesses are in `tests/unit/test_stopping_rule.py`.

| fit | what the witness shows |
| --- | --- |
| one-step walk, `sample_size` | the estimate differs from the full walk by more than $10^{-6}$. The walk stops at the index the hand count gives |
| iterative fit with separation and `submodel_alpha=0.99` | `sample_size` converges at iteration 1. The shipped test stays unconverged after 20 iterations |

A fit under `fluctuation="linear"` solves the score equation in one closed-form step. It has no
iterate, rule or step bound, so `Targeting` refuses all three beside it.

When a fit misses its rule, it warns with a pinned text:

> targeting step did not meet the {rule} stopping rule after {k} iteration(s) (largest score over
> its threshold = {x:.3g}): ...

#### What a declared rule leaves in the estimate

A declared rule leaves a score residual by design. The residual is a score residual and not a
nuisance remainder, so it is not a linear-remainder term. Write $c = 1/\min(\log n_u, 10)$.

| column | bound on the residual of a fit that met `scaled_var` |
| --- | --- |
| one arm-mean column with unit weights | $\lvert r_j\rvert \le c\, \mathrm{se}_j$, so the residual is at most $c$ standard errors |
| a score-column arm row (D4) | about $c$. The score part and $Q^*_a(W)$ are uncorrelated in the population and not exactly in the sample |
| a contrast $g$ | $\lvert g\cdot r\rvert \le c \sum_j \lvert g_j\rvert\, \mathrm{se}_j$, so $b \le c \sum_j \lvert g_j\rvert\, \mathrm{se}_j / \mathrm{se}_{\text{contrast}}$ |

A contrast has no small a priori bound. The threshold of each column uses the full arm curve. The
two arm curves share the $Q^*_a(W)$ terms, and the contrast cancels them. So the ratio
$\sum_j \lvert g_j\rvert\, \mathrm{se}_j / \mathrm{se}_{\text{contrast}}$ can be large.

The witness is the law $W \sim N(0,1)$, $A \sim \text{Bern}(0.5)$, $Y = W + A + 0.1\varepsilon$.
`TestTheRuleBound` in `tests/unit/test_stopping_rule_report.py` fits it with seed 3 and
$n = 2{,}000$. The ratio is 9.95, so the `ate` row has a `rule_bound_se` of 1.31 against
$\sqrt{2}\,c = 0.19$. The same file pins `wald_coverage_bound(1.009, 0.05)` at 0.8277. At the
ratio 9.95 and $c = 0.1$, the residual is $b = 0.995$ standard errors and the bound is about 0.831.
Marginal structural model columns are contrasts in this sense.

The coverage bound is $\Phi(z - b) - \Phi(-z - b)$ at a residual of $b$ standard errors.
`wald_coverage_bound` computes it. The table gives the 95% bound for one arm-mean column at the
residual $c$.

| $n_u$ | $c$ | 95% interval, one arm-mean column |
| ---: | ---: | ---: |
| 50 | 0.256 | 0.9425 |
| 100 | 0.217 | 0.9446 |
| 500 | 0.161 | 0.9470 |
| 1,000 | 0.145 | 0.9476 |
| 22,026 and above | 0.100 | 0.9489 |

`tests/unit/test_stopping_rule_report.py` checks the table against `scipy.stats.norm`. The bound
is conditional on the residual being the only error. It does not replace the interval's own
conditions.

`sample_size` leaves a residual of at most $1/(n_u\, \mathrm{se})$ standard errors. That is
$o(n^{-1/2})$, and it depends on the outcome's units. For a rare binary outcome or a narrow
`span` it can reach about one standard error at $n = 100$. The realised line reports the exact
value on every fit.

#### What the result reports

| output | content |
| --- | --- |
| `res.summary()` | one pinned line: "stopping rule {rule}: the largest score residual is {b:.3g} standard errors, on {name}. At that residual a {level:.0%} Wald interval keeps at least {bound:.4f} coverage." |
| `res.summary()`, rule missed | the pinned line "the targeting did not meet the {rule} stopping rule; {detail}" |
| `score_check()`, fluctuation rows | the bar is `Fluctuation.thresholds`. A row passes when the rule ratio is at most one. The rule also judges the `(strata)` row of a stratified working model and the `(reduced)` row of a `DRTMLE` fit, because the rule stops those outcome solves too. A `(mechanism)` row keeps the default bar |
| `score_check()`, `DRTMLE` correction rows | under a declared rule, a `D*_Q` or `D*_Y` row is a `"diagnostic"`, because it restates the outcome columns that the `(reduced)` row judges. A `D*_g`, `D*_A` or `D*_M` row and every identity row keep their bars |
| `score_check()`, estimand rows | kind `"diagnostic"`. They cannot fail. They carry `residual_se`, `coverage_bound` and, under `scaled_var`, `rule_bound_se` |
| `score_check().to_frame()` | the columns `stopping`, `residual_se`, `coverage_bound` and `rule_bound_se`, only under a declared rule |

The residual in standard errors is $\lvert P_n IC\rvert / \mathrm{se}$ on the scale of the
estimate. The bound before the fit is read is
$\texttt{rule\_bound\_se} = c \sum_j \lvert g_j\rvert\, \mathrm{se}_j / \mathrm{se}$. Here $g$ holds
the column weights of the estimand. The term $\mathrm{se}_j = \mathrm{sd}_j / \sqrt{n_u}$ is the
standard error of the threshold's own column. It equals the package variance for the same curve
(`inference/cluster.py`).

| estimand row | `rule_bound_se` |
| --- | --- |
| arm mean, unit weights | $c$, to $10^{-12}$ |
| `att`, `atc` | about $c$ |
| `ate` and other contrasts | $c \sum_j \lvert g_j\rvert\, \mathrm{se}_j / \mathrm{se}$, with no a priori value |
| weighted arm row | computed by hand in the test, and not asserted at most $c$ |
| stratified fit, marginal row | $\sum_s w_s\, \mathrm{threshold}_s / \mathrm{se}$ over the stratum blocks. $w_s$ is $P_n(S=s)$ for an arm mean or a difference, times $P_n(A=a \mid S=s) / P_n(A=a)$ for `att` or `atc` |
| stratified fit, stratum row | the thresholds of that stratum's block over the stratum standard error |
| `DRTMLE`, and a fit with several draws | `nan`. The corrected curve carries terms that the outcome columns do not |
| incremental tilt | `nan`. The curve also carries the treatment-mechanism equation, which the rule does not govern |
| working-model coefficient | `nan`. The curve is the score columns times the inverse normalising matrix, which the result does not carry |
| shift, policy, regime, ratio | `nan`. The result does not map these estimates to score columns |

A stratified block uses the score column as its contribution (declared difference D4), so its
marginal row is not $c$. `TestTheRuleBound` computes the stratified rows by hand. It also checks
that the weights reproduce the mean of each reported curve to $10^{-10}$ on a one-step walk.

For a fit that met its rule, `residual_se` is at most `rule_bound_se`. An unconverged fit can
exceed it. `TestTheRuleBound` asserts the inequality only on fits that met the rule. It also holds two
mutation controls. One computes `rule_bound_se` as $\sqrt{2}\,c$, and the other drops the weight of
the second arm. Each fails the contrast test.

#### Alternating engines

Three engines alternate between the outcome equation and another equation. Their outer exits
judge the outcome equation by the rule ratio under a declared rule. Each other equation keeps
`tol`.

| engine | outer exit under a declared rule |
| --- | --- |
| point IPSI | the outcome entry is `tol` times the solver's ratio, and the mechanism score keeps `tol`. The solver's ratio is current, because the outcome solve ends each round |
| `DRTMLE` | the ratio restated at the pair after the last move, in both update orders. Equations (9) and (10) keep `tol` |
| `DRTMLE` with a missing outcome | the outcome entry of the worst score is `tol` times the ratio restated from the stacked scores. The drift blocks keep `tol` |

A ratio is valid only at the pair of fit and score where it was computed. The `"benkeser"` order
and the missing-outcome reduction both move the regression or the mechanism after the outcome
solve. So `_restated_outcome_score` recomputes thresholds and ratio at the restated pair. No exit
reads a stale ratio. The scaled quantity $\texttt{tol} \cdot \text{ratio}$ is one ruler on every
round, so the stall rule still compares like with like.

`tests/unit/test_stopping_rule_engines.py` audits each exit. It recomputes the ratio by hand at
every round of the trace with `pandas`, and asserts that the ratio the exit read equals it. In the
stale-ratio witness (`order="benkeser"`, seed 5, step 0.01) the restated ratio reaches 1.75 while
the solver's own ratio is 0.966. Six mutation controls fail the audit, among them an exit that
reads the solver's ratio and a restatement that keeps the old thresholds.

The projection exits of an MSM test the shift in $\beta$. They are not outcome-score tests, and
the rule does not change them.

#### Where the rule applies

| engine | (a) to (c) | $n_u$ counts |
| --- | --- | --- |
| `TMLE`, every point group | yes | the clusters among the solver's rows, else the rows |
| point IPSI, `DRTMLE`, `CTMLE` | yes | as `TMLE`. A C-TMLE candidate counts the units of its training rows |
| CV-TMLE, pooled, fold-targeted or repeated | yes | the units of the rows each solve receives. A fold-targeted fit counts its fold's units, so the bound is per fold |
| `LTMLE`, per node | yes | the clusters of the panel, else the rows of the panel. The node solver receives every row. See [longitudinal targeting options](longitudinal-tmle.md#targeting-options-at-each-node) |
| `SDR`, longitudinal `IPSI`, the sequential baselines | refused | none. They fit no fluctuation |

The estimator builds the context once, in `_retarget_detailed`. So `TMLE.retarget`, the truncation
curve and the bootstrap use the same context. `tests/unit/test_stopping_rule_engines.py` fits every
point group of the table under both rules. It also covers a fold solve, a candidate solve, a
truncation curve and a bootstrap.

#### The step bound

`step_bound=delta` needs `algorithm="iterative"` and `fluctuation="logistic"`. Each outer
iteration takes the `tmle3` constrained step.

1. Read the score $s$. Stop when $\lVert s\rVert_2 = 0$.
2. Set the direction $d = s/\lVert s\rVert_2$. The collapsed covariate is $Hd$. A single column has $d = \operatorname{sign}(s)$.
3. With `step_search=True`, find $t^*$ in $(0, \delta]$ that minimises the weighted mean quasi-loss along $d$. The search is bounded Brent search with `xatol` $1.49 \times 10^{-8}$.
4. With `step_search=False`, take $t^* = \delta$.
5. Add $t^* d$ to $\epsilon$. Carried fits move by the same step.

At unit weights the loss is the unweighted loss of `tmle3`. With weights, it is the weighted
extension. Under `target_weights` the loss is the fit's own weighted form. The direction $d$ comes
from the unweighted score of the same equation.

`tests/unit/test_step_bound.py` checks the whole path against a SciPy walk written from the
`tmle3` description, at $10^{-7}$. The first step binds at 0.15 and is below the Newton step of
0.2157. The second step is interior at 0.0657. Six mutation controls fail the reference: an
un-normalised $Hs$, a reversed direction, `xatol=1e-3`, an upper bound of $2\delta$, the Newton
direction and an unweighted loss.

A bounded walk whose initial fit already meets its rule takes no step. Under a loose declared
rule this can mean no targeting at all. In `tests/unit/test_step_bound.py`, `scaled_var` on a
point TMLE with seed 2 gives `n_iter=0`. A step that moves nothing also ends the walk. The Newton
path under a declared rule takes one step before it tests the rule, as `tmle3` does.

**Is `relative` reachable under a step bound?** With `stopping="relative"` the walk reaches
`tol=1e-10` only under conditions. `TestRelativeReachability` measured it on arm-mean fits with
`step_bound` in $\{0.01, 0.05, 0.25\}$.

| setting | result in the recorded run |
| --- | --- |
| `step_search=True`, default `max_iter=20` | never reached. 0 of 6 point fits and 0 of 32 solver cells. The unconverged walk ends at $1.3\times10^{-9}$ to $8.2\times10^{-9}$ |
| `step_search=True`, `max_iter=100` | 5 of 6 point fits (35 to 92 iterations), 20 of 32 solver cells |
| `step_search=True`, `max_iter=1000` | 6 of 6 point fits (35 to 145 iterations, median 65), 28 of 32 solver cells (21 to 444 iterations, median 52). The 4 misses plateau at $1.1\times10^{-10}$ to $4\times10^{-10}$ |
| `step_search=False`, any `max_iter` up to 1,000 | never reached. 0 of 38 cells. The walk lands within one bound of the root and oscillates, with a relative score of $5\times10^{-4}$ to $10^{-2}$ on the point fits |
| `LTMLE` nodes (one column), `step_bound=0.05`, search on | 3 of 4 nodes reach `relative` within `max_iter=20`, and 4 of 4 within 200. With search off, 0 of 4 |
| `stopping="sample_size"` or `"scaled_var"`, `step_bound=0.05`, search on | met in 0 to 4 iterations. The point fits take 1 and 0. With search off, 3 of 4 `LTMLE` nodes meet it |

Each outer iteration is one line search along the steepest-ascent direction, so convergence is
linear and takes tens of iterations. The search stops within $1.5\times10^{-8}$ of the minimiser.
That leaves a relative score of about $10^{-10}$ to $10^{-9}$ at the root, at the edge of `tol`.
Near that edge, floating-point rounding sets the iteration count. On a CI runner with Python 3.13,
two of the six point fits took 25 and 190 iterations. A Windows run reaches the tolerance in nine
iterations for the continuous point fit with `step_bound=0.25`.

The `tmle3` pairing is therefore a declared rule with a larger `max_iter`. A bounded walk that stops
short of its rule warns, reports `converged=False` and `failure="max_iter_reached"`, and fails
`score_check()`. On a binary outcome with a linear learner the label is `bounds_pinned` when more
than 1% of the predictions sit at the shrink bounds. The Newton path reports the same label.

#### The best iterate

`iterate="best"` keeps a snapshot of the best iterate $k \ge 1$. It ranks the iterates by the
2-norm of the raw score vector, and it keeps the first of tied iterates, as `tmle3` does. The
initial fit is never a candidate while a step exists. With no step, `best_iterate` is 0.

The loop still stops by the rule. `n_iter` and `trace` describe the whole walk. `converged` is
judged at the reported iterate. `Fluctuation.best_iterate` holds the index.

| path | what `"best"` does |
| --- | --- |
| the iterative walk with Newton steps | the norm falls at each step, so best equals last in the usual case |
| a walk that oscillates, such as `step_search=False` | best differs from last, and the estimates differ by more than $10^{-6}$ in `tests/unit/test_best_iterate.py` |
| `algorithm="one_step"` | the walk rejects every norm increase, so best equals last. A tie takes the last iterate |

`tests/unit/test_best_iterate.py` replays $k^*$ steps on an oscillating fixed-step walk and
compares the result. Four mutation controls fail it. They report $k^*+1$, rank by the relative
statistic, include iterate 0, or take the first of ties on the one-step path.

#### Declared differences from `tmle3`

| difference | what `cleverly` does |
| --- | --- |
| D4 | a column that is not one unit-weight arm mean uses the score column as its contribution, not the full curve |
| first step | `tmle3_Update$update` takes a step before it first tests the rule. The Newton path of `solve_fluctuation` under a declared rule also steps first. The bounded walk (`step_bound=`) and the one-step walk test first, so a fit whose initial score meets the rule takes no step there |
| unconstrained update | `tmle3` takes one maximum-likelihood step. `cleverly` solves the Newton equation, not a walk |
| ranking | `use_best` ranks by the raw score vector. The relative statistic of the default test is not used |
| weights | a weighted fit uses the weighted loss in the line search, which `tmle3` does not define |

The R fixture `tmle3_updater.json` compares the updater steps, epsilons and estimates at the
default `alpha`. All ten updater runs agree with R at the tolerances of the fixture README. See the
[evidence manifest](evidence.md#targeting-and-fitting-options).

### Per-arm outcome regression

`ModelSpec(outcome_by_arm=True)` fits one outcome regression for each treatment arm. The R
comparator is `drtmle(stratify=TRUE)`. The default fits one regression on the covariates and the
treatment.

| item | rule |
| --- | --- |
| training rows | in each fold, the rows with an observed outcome, a recorded treatment, and $A = a$ |
| design | the covariates alone, plus the intermediate under `intermediate=`. The treatment column is absent |
| prediction at the observed arm | the model of the row's own arm |
| prediction at a counterfactual $a$ | the model of arm $a$, on every row |
| arm-coded modified treatment policy | each row reads the model of its shifted arm |
| companion predictions | the same rule, for the evaluation companion of a doubly robust fit |

`_fit_outcome_by_arm` serves four sites: the primary nuisances, the nested inner designs of
`DRTMLE`, the C-TMLE selection base, and the honest base of `CTMLE(strategy="oat")`. Each site
runs the shipped call when the option is off.

The option composes with missing outcomes and missing treatment, an intermediate, strata,
clusters, weights, arm MSMs, point IPSI, C-TMLE base fits, OAT and `DRTMLE`. It adds a diagnostic
for each arm and fold under `diagnostics["outcome"]`. A Super Learner asks its split rule of each
arm's rows.

| refusal | trigger |
| --- | --- |
| R-f1 | a continuous dose. It has no arms. The error is a `CapabilityError` with code `invalid_declaration` |
| R-f2 | an arm with no training row in a fold. The error is a `DataError` that names the fold |
| R-f3 | a longitudinal design. The point-only message of the method applies |

[Scope and refusals](scope-and-refusals.md#targeting-and-fitting-option-refusals) gives each text.

`tests/unit/test_outcome_by_arm.py` compares the predictions with a hand fit of
`LogisticRegression` for each arm and fold at $10^{-12}$. It covers 12 compositions, among them
three arms, weights, clusters, strata, an arm MSM, point IPSI and a controlled direct effect. The
witness is an $A \times W$ interaction law. There the per-arm and pooled predictions differ by more
than $10^{-3}$ at the primary, nested, selection and OAT sites.

Mutation controls swap the arm models, drop the arm from the mask, and read the observed arm for a
shifted arm. Each fails the hand comparison. On the R fixture `drtmle_stratify.json`, the arm
regressions agree with `drtmle(stratify = TRUE)` to $10^{-8}$ and the estimates to $10^{-6}$.

### Treatment-mechanism covariates

`PointTreatment(treatment_adjustment=[...])` lets the treatment mechanism read a subset $W_g$ of the
adjustment set $W$. `TMLE.fit(treatment_covariates=)` is the low-level entry. The R comparator is
`ctmle` with `Wg=`. The subset may be empty. The outcome regression and every other nuisance keep
the full set.

| nuisance | covariates it reads |
| --- | --- |
| treatment mechanism, including the dose density and the classifier ratio | $W_g$ |
| covariate screening and the retained set | $W_g$ |
| outcome regression | $W$ |
| outcome and treatment observation | $W$ |
| policy rule evaluation | $W$ |

The declaration restricts the observed-data model, and it does not change the estimand. The
design carries a pinned text in `Identification.model_restrictions`. Every estimator on the design
prints it after the assumptions, and a low-level fit prints the same text from the data. The text
reads, with $W_g$ written as `{name, name}`:

> statistical-model restriction, treatment-mechanism covariates {Wg}: the treatment mechanism
> depends on the adjustment covariates only through {Wg}. It restricts the observed-data model and
> does not change the estimand. The fit is consistent when the treatment mechanism fitted on {Wg}
> is consistent, which needs the restriction, or when the outcome regression is consistent. The
> interval needs the restriction and both fitted models consistent at the usual product rate. When the
> restriction fails, the remainder is first order in the outcome-regression error. To exclude
> instruments, omit them from adjustment= instead: the outcome regression and the treatment
> mechanism then both use {Wg}.

The inference status stays `influence_curve`, claimed under the declared restriction.

#### The remainder when the restriction fails

Write $Z = W \setminus W_g$ and $\bar g(W_g) = E[g_0(W) \mid W_g]$. A fit on $W_g$ has
$g_n \to \bar g$. For the arm-1 mean the remainder splits in two:

$$
R_2 = P_0\big[(1 - g_0/\bar g)(Q^* - Q_0)\big] + P_0\big[(g_0/\bar g - g_0/g_n)(Q^* - Q_0)\big].
$$

Because $E[1 - g_0/\bar g \mid W_g] = 0$, the first part equals
$P_0[(1 - g_0/\bar g)\{(Q^* - Q_0) - E[Q^* - Q_0 \mid W_g]\}]$. So it is linear in the part of the
outcome error that varies with $Z$ given $W_g$, and its factor does not shrink. The second part is
a product, and it is first order in $g_n - \bar g$ when $Q$ is wrong, which is the ordinary
one-wrong case.

| case | first part | interval |
| --- | --- | --- |
| A: $g_0$ depends on $W$ only through $W_g$ | 0 (the H1 analogue) | covered at the product rate |
| B: instruments in $Z$, $Q_0$ free of $Z$ given $(A, W_g)$, $Q_n$ on the full $W$ | linear in the $Z$-varying outcome error | **not covered** |
| B, with $Z$ also removed from `adjustment=` | 0 (the H2 analogue: a TMLE of $(W_g, A, Y)$) | covered |
| $Z$ holds confounders | linear | not covered |

For $Q = Q_0 + \delta k$ the identity $\text{error} = P_0[(1 - g_0/\bar g)\,\delta k]$ is exact only for
`fluctuation="linear"`. There targeting moves $Q$ by $\epsilon/\bar g$, a function of $W_g$, which
the factor annihilates. Under the default logistic fluctuation the update is not a function of
$W_g$. The exact identity is $\text{error} = P_0[(1 - g_0/\bar g)(Q^* - Q_0)]$ with the targeted
$Q^*$.

The exact-law check draws $W_1$ and $Z$ as independent Bernoulli(1/2) variables.

| quantity | value |
| --- | --- |
| true mechanism $g_0$ | $0.25 + 0.25 W_1 + 0.25 Z$ |
| projected mechanism $\bar g$ | $0.375 + 0.25 W_1$ |
| true outcome regression $Q_0(a, W_1)$ | $0.2 + 0.3a + 0.2W_1$ |
| fitted outcome regression $Q$ | $Q_0 + 0.1Z$ |

The cell probabilities are multiples of 1/16, so $16m$ rows give $P_n = P_0$ exactly.

| case | linear fluctuation, arm-1 error | logistic fluctuation, arm-1 error |
| --- | --- | --- |
| $k = Z$ | $-0.0133333$ ($-1/75$, equal to the $\delta$-formula) | $-0.0138815$ (equal to the $Q^*$ formula) |
| $k = W_1$ | 0 | 0 |
| $g_0 = \bar g$ | 0 | 0 |

`tests/unit/test_treatment_covariates.py` reproduces the table with $m = 64$ (1,024 rows),
`cross_fit=False`, `q_bounds=(0, 1)` and `g_bounds=(0.01, 0.99)`. The default `q_bounds` widen the
outcome range by 10%, which moves the errors to $-0.0080000$ and $-0.0069586$. The test uses a
cell-mean classifier, which returns $\bar g$ on $\{W_1\}$ and $g_0$ on $\{W_1, Z\}$ exactly, and a
raw-scale outcome stub.

| test | assertion |
| --- | --- |
| witness, linear | the error equals $P_0[(1 - g_0/\bar g)\,\delta k]$ at $10^{-10}$, and $\lvert\text{error}\rvert > 10^{-3}$ |
| witness, logistic | the error equals the $Q^*$ formula at $10^{-10}$, and $\lvert\text{error}\rvert > 10^{-3}$ |
| vanishing controls | $k = W_1$ and $g_0 = \bar g$ give 0 at $10^{-10}$ |
| case B with $Z$ out of `adjustment=` | the error is 0 at $10^{-10}$ |
| mutation: fit $g$ on the full set | the classifier then returns $g_0$, the error becomes 0, and the witness fails |
| spy learner | the mechanism learner sees exactly $W_g$ at every site. The other learners see the full set |

**Advice.** To exclude instruments, omit them from `adjustment=`. Then the outcome regression and
the treatment mechanism both use $W_g$, and the interval is covered. Use `treatment_adjustment=`
when the outcome regression needs the columns and the treatment mechanism does not.

#### Compositions

| composition | result |
| --- | --- |
| `DRTMLE` | the design prints the restriction. The full summary adds a line under Method details: when treatment depends on other covariates too, the treatment model is wrong, and no result shows that the interval keeps its coverage. The mechanism-wrong rows of the [case table](dr-tmle/theorem.md#the-conditioning-terms-are-linear-in-one-error) decide the interval |
| `CTMLE` | the pool, the ordering, the candidates and the lasso design read $W_g$. See [collaborative TMLE](collaborative-tmle.md#treatment-mechanism-covariates-and-the-candidate-pool) |
| the omitted-variable bound | admitted on a plain `TMLE` fit, under the restriction. The fitted Riesz representer uses $g_n(W_g)$. See [the bound](validation-methods.md#omitted-variable-bounds-robustness-value-benchmark-and-contours) |
| `with_extra_covariate`, used by the common-cause refutation | the new column joins $W_g$, so the refit keeps the restriction on the old columns |
| `without_covariates` | the field is intersected with the remaining columns |

A refusal covers each composition that has no meaning.

| refusal | trigger |
| --- | --- |
| R-h1 | a column outside `adjustment=` |
| R-h2 | `CTMLE(strategy="oat")` |
| R-h3 | point `incremental=`, because the shift is a function of the fitted propensity score |
| R-h4 | a known `treatment_probabilities` |
| R-h5 | a C-TMLE `candidates=` or `ordering=` outside the subset |
| R-h6 | a repeated name |

[Scope and refusals](scope-and-refusals.md#targeting-and-fitting-option-refusals) gives each text.
The R fixture `ctmle_wg.json` checks the candidate pool of the restriction. See
[treatment-mechanism covariates and the candidate pool](collaborative-tmle.md#treatment-mechanism-covariates-and-the-candidate-pool).

### Known treatment mechanism

A randomized trial fixes its treatment mechanism $g_0(a\mid W)$ by design. Declare it with
`treatment_probabilities=` in `TMLE.fit` or in `CausalData.from_frame`. In the study API, declare
it with `PointTreatment(treatment_probabilities=...)`. The fit then divides by $g_0$, and it fits no
treatment learner.

| form | reading |
| --- | --- |
| `{level: "column"}` | each level's probability is the named frame column. The columns are not adjustment covariates |
| `{level: array}` | one `(n,)` array per arm. Every arm must be named |
| `(n, K)` array | columns in the sorted level order |
| `(n,)` array | two arms only, read as $P(A = \text{levels}[1]\mid W)$ |

The declaration is a data field (`CausalData.known_treatment`), not an estimator setting.
`TMLE.fit` accepts it beside a dataframe, and the array forms beside a prepared `CausalData`. A
container that already declares a mechanism refuses a second declaration. A known mechanism also
refuses `treatment_adjustment=`, because the declaration restricts a fitted mechanism and a known
one is not fitted.

The claims below hold when every mechanism the fit divides by is declared. For a
point treatment, that is complete data with no `intermediate=`.

With $g=g_0$ the remainder
$P_0\{(g-g_0)/g\,(\bar Q-\bar Q_0)\}$ is zero. The estimate is then consistent for any limit
$\bar Q_\infty$ of the outcome regression. Its influence curve is exactly
$D^*(\bar Q_\infty,g_0)$. Moore and van der Laan (2009), Section 2, state the consistency.
Their Section 5 states that inference with the true mechanism is not conservative. The table gives
the curve each fit reports.

| fit with a declared mechanism | reported curve | source |
| --- | --- | --- |
| arm means, ATE, RR, OR, PAR, PAF, regimes and MSMs, on complete data | the ordinary curve at $g_0$, which is the exact influence curve | Moore and van der Laan (2009), Sections 2 and 5 |
| missing outcomes (`delta=`) and a controlled direct effect (`intermediate=`) | the ordinary curve at $g_0$ and at the estimated second mechanism. The declaration replaces the treatment mechanism only. The observation mechanism $\hat\pi$, or the intermediate mechanism $\hat q_z$, is still estimated. The remainder is the product of that mechanism's error and the outcome error, so the ordinary TMLE conditions apply. The interval is established when both are consistent at a product rate. It is not established for a wrong outcome regression | the missing-outcome and controlled-direct-effect contracts above |
| ATT and ATC | the shipped ATT and ATC curve at $g_0$. It is exact, and it is not the known-score efficient curve | the algebra below; Hahn (1998) |
| `incremental=` | the curve of the known stochastic regime $q_\delta(g_0)$, with no mechanism term, for any number of levels | Kennedy (2019), the remark after Corollary 2 |
| `learned_rule=` | the shipped learned-rule curve. The rule reads the outcome regression only | the learned-rule contract above |
| `DRTMLE` | see the [DR-TMLE contract](dr-tmle/theorem.md#a-known-treatment-mechanism) | Benkeser et al. (2017), Theorem 1 |
| `LTMLE` | see [known node mechanisms](longitudinal-tmle.md#known-node-mechanisms) | van der Laan and Gruber (2012), Theorem 2 and Section 4 |

A known mechanism is not the only choice. The table compares the three mechanisms a trial analyst
can use.

| mechanism | what the interval means |
| --- | --- |
| the declared design mechanism, on complete data with no `intermediate=` | an interval that is not conservative, for any outcome learner |
| maximum likelihood in a parametric model that contains $g_0$, such as a logistic regression on the randomization strata | a smaller true variance when $\bar Q$ is wrong. The reported standard error is then conservative (Moore and van der Laan 2009, Section 4.2 and Section 7.3, Table IV; Petersen et al. 2014, Section 3.7) |
| a flexible treatment learner, with an outcome regression that can be wrong | the TMLE need not be asymptotically linear (Benkeser et al. 2017). Use the declared mechanism |

The registered [known-mechanism TMLE study](method-evidence/known-treatment-mechanism.md) reports
the spread of the first two side by side, as a row with no verdict.

**The ATT and the ATC.** Write $p=P(A=1)$ and $\bar Q_{a,0}$ for the true arm regression. At the
declared mechanism,

$$
P_0\{A(Y-\bar Q_1)\} = P_0\{g_0(\bar Q_{1,0}-\bar Q_1)\}, \qquad
P_0\Big\{(1-A)\frac{g_0}{1-g_0}(Y-\bar Q_0)\Big\} = P_0\{g_0(\bar Q_{0,0}-\bar Q_0)\}.
$$

The shipped plug-in is the treated mean of $\bar Q^*_1-\bar Q^*_0$. With the ATT score
equation, it gives $\hat\psi-\psi_0=(P_n-P_0)D^*_{\text{ATT}}(\bar Q_\infty,g_0)$ up to
the empirical-process term. The remainder is zero for any $\bar Q$, and the ATC is the mirror
image.

The curve is not the efficient one when the score is known. Hahn (1998) gives a smaller ATT
bound for a known propensity score than for an unknown one, as read from the abstract. By the
remainder algebra above, the g-weighted plug-in $P_n\{g_0(\bar Q^*_1-\bar Q^*_0)\}/P_n g_0$ has
the smaller curve. The package does not ship it.

**Incremental interventions.** With $g_0$ known, $q_\delta(g_0)=\delta g_0/(\delta g_0+1-g_0)$ is
a known stochastic regime. With $K$ levels it is $q_\delta(a\mid W)=\delta_a g_0(a\mid W)/D_\delta(W)$. The model with a known mechanism has no mechanism scores, so the
efficient curve is Kennedy's curve projected off that tangent space. The projection removes the
middle term in $(A-g)$ and leaves the regime curve. Kennedy (2019), arXiv v3, states it after
Corollary 2: "If the propensity scores were known, the efficient influence function would just be
the first weighted average term." The fit therefore solves no mechanism equation.
`tests/unit/test_known_treatment_mechanism.py` checks that the fit equals the
`Stochastic(density_kind="known")` fit at $q_\delta(g_0)$.

**Modified treatment policies.** A discrete `policies=` fit reads $g$ only in the ratio
$g^d/g$, and the plug-in reads the observed treatment. A declared $g_0$ is the degenerate estimate
$g_n=g_0$. The remainder is a product of the ratio error and the outcome error, so it is zero.
The shipped curve at $(\bar Q_\infty,g_0)$ is therefore exact. The fit evaluates the discrete formula at the
declaration. `tests/unit/test_known_treatment_mechanism.py` checks that the fit equals the fit
whose learner returns $g_0$, at two and three arms. `ratio="classifier"` learns the ratio, so it
is refused beside a declaration.

**Weights.** With `weights=`, the declaration is the mechanism of the weight-tilted law. That law's
mechanism equals the design mechanism when the weights depend on $W$ only, as survey and transport
weights do. The declaration is the analyst's assertion, as the array itself is.

**Bounds.** Truncating a known mechanism moves the estimate with no variance reason. Before any
learner, the fit checks each declared value against every bound pair it uses. These are the
ATE-family pair, and the ATT and ATC pair when `att` or `atc` is requested. If a pair would move
a value, the fit refuses:

> treatment_probabilities has {k} values outside the truncation bounds [{lo}, {hi}] (smallest
> {min}, largest {max}). Truncating a known design mechanism moves the estimate with no variance
> reason. Pass g_bounds=(lower, upper) that contains every known probability.

`truncation_curve()` evaluates only the pairs that the fit admits. An explicit bound whose
symmetric pair moves a declared value is refused with a sentence that quotes the
fit's condition, and with the `invalid_declaration` code. The default grid omits each canned pair that moves a declared value.
An admitted pair moves no declared value. Each row carries a `treatment_mechanism`
column that reads `"known"`. `tests/unit/test_known_treatment_mechanism.py` pins the refusal and
the default grid.

**Readers.** Each reader that refits uses the data it is handed, so the declaration follows the
rows.

| reader | declared mechanism |
| --- | --- |
| `n_bootstrap=` and the subset refutation | carried. Each replicate reads the declared values of the rows it drew |
| the placebo refutation | dropped, because a permuted treatment has a mechanism of its own. The refit estimates it, and the report says so. The permuted treatment is independent of $W$, so the re-estimated null stays centred |
| simulated confounding | carried as the flipped mechanism. The flip of strength $s$ is independent of $W$, so the flipped treatment's mechanism is $g_s(a\mid W)=(1-s)\,g_0(a\mid W)+s\,g_0(1-a\mid W)$. Each cell declares $g_s$, and its refit fits no treatment learner |
| `random_common_cause`, measurement error and the negative-control outcome | carried |
| replay and `refit` | carried. A refit of a known-mechanism fit reproduces it bit for bit |
| `retarget`, the truncation curve, the MNAR tilt, and the omitted-variable bound | read the cached declared mechanism |
| the omitted-variable `benchmark` | refused. A declared mechanism does not change when a covariate is dropped |
| `variable_importance` | not carried. Each candidate is a different treatment with no declaration |

**Refusals.** Each of these refuses before any learner.

| composition | error | reason |
| --- | --- | --- |
| `CTMLE`, or `CollaborativeTMLEMethod` on a declaring design | `CapabilityError` | C-TMLE selects or fits the mechanism, and the declaration leaves nothing to select or fit |
| a continuous treatment | `CapabilityError` | the declaration gives arm probabilities, and a dose has a conditional density |
| `policies=` with `ratio="classifier"` | `CapabilityError` | the classifier learns the ratio that the declaration fixes. Pass `ratio="density"` |
| `treatment_delta=` | `CapabilityError` | a missing treatment needs $P(A=a\mid\Delta_A=1,W)$, which equals the design mechanism only when recording is independent of treatment given $W$ |
| `screen_treatment=True` | `ValueError` | the screen selects covariates for the treatment learner, which the declaration replaces |
| `DRTMLE(evaluation=...)` whose companion declares no mechanism | `CapabilityError` | the companion's mechanism must be its own declaration. Declare the same columns on an evaluation frame, or pass a `CausalData` companion that declares its own |
| a second declaration of the mechanism | `ValueError` | declare it once, on the data or at fit |

The result records the choice. `TMLEConfig.treatment_mechanism` reads `"known"`, the summary
prints `treatment mechanism: known (declared)`, and `to_frame()` adds a `treatment_mechanism`
column. A saved result keeps the record.

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
| 2 | `learned_rule=` beside `interventions=`, `policies=`, `incremental=`, `msm=` or `reference=`, or beside an arm estimand | [X11](../roadmap.md#x11-learned-policy-follow-ups) (c) |
| 3 | a continuous treatment | F27 |
| 4 | a treatment with more than two arms | X11 (d) |
| 5 | missing outcomes, `delta=` | F27, with a learned-rule text that runs before the missing-outcome refusal |
| 6 | `intermediate=` | X11 (j) |
| 7 | `weights=` | X11 (k) |
| 8 | `id=` | X11 (l) |
| 9 | `strata=` | X11 (m) |
| 10 | `cross_fit=False`, or one fold | X11 (a) |
| 11 | `cv_evaluation=False` | X11 (e) |
| 12 | `targeting_scheme="fold"` | X11 (h) |
| 13 | `repeats` above 1 | X11 (i) |
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

A refit relearns the rules, so the full-refit bootstrap and `repeats` above 1 refuse.
`simulated_confounding` and the `refute` tests that refit hold each row's realized rule fixed.
The [simulated common-cause surface](validation-methods.md#simulated-common-cause-stress-surface) states that reading.

The other sensitivity analyses read `unavailable`, because the package does not implement them for this target.
[X11](../roadmap.md#x11-learned-policy-follow-ups) part (n) holds the omitted-variable bound.
[F27](../roadmap.md#f27-learned-policy-value-outside-the-published-conditions) holds the
E-value and the missingness tilt. The support report and `truncation_curve` run, because they read the same
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

`TMLE(policies=...)` accepts the policy classes in the table. On a continuous dose, each class
gives the ratio by Equation (3) of Díaz, Williams, Hoffman and Schenck (2023):
$g^d(b \mid w) = \sum_j 1\{b \in I_j\}\, g(b_j(b) \mid w)\, |b_j'(b)|$. Here piece $j$ maps the
interval $I_j$, and $b_j$ is the inverse of that piece.

| class | map $d(a, w)$ | ratio source |
| --- | --- | --- |
| `Shift(delta, cap)` | $a + \delta$, held at $a$ where $a + \delta$ exceeds `cap` | closed form |
| `Scale(factor, cap)` | $a \cdot$ `factor`, held at $a$ where the product exceeds `cap` | closed form, Jacobian `1 / factor` |
| `Piecewise(pieces, closed)` | a `Shift` or a `Scale` on each interval | closed form per piece |
| `ModifiedPolicy(name, pieces=...)` | a declared `map`, `inverse` and `derivative` per `Piece` | Equation (3) |
| `ModifiedPolicy(name, apply=...)` | any function that returns a level | categorical treatment only |
| `ModifiedPolicy(..., randomizer=Randomizer(...))` | one branch per randomizer value, with known probabilities | the weighted mean of the branch ratios |
| `RiskRatioTilt(delta)` | the `ipsi` rule of `lmtp` on a 0/1 treatment | discrete formula |

Every observed dose must lie in exactly one piece of a `ModifiedPolicy`. A `Piecewise` policy
leaves a dose in no interval unchanged, and its ratio counts that dose as an identity piece.
Each moving piece must be strictly monotone, and its declared inverse must invert its map. The
declared derivative of the inverse must equal $1 / d'(a)$ from a central difference of the map.
`policy_kind="known"` declares that the map does not
depend on the observed-data law, and a fit accepts no other value. A policy that fails these
checks is refused before any nuisance fit.

On a categorical treatment the ratio is the discrete formula
$g^d(b \mid w) = \sum_a 1\{d(a, w) = b\}\, g(a \mid w)$, and a map must return a level of the
treatment. `RiskRatioTilt` reports on its own axis, `ey_rr_tilt` and `ate_rr_tilt`, and needs the
levels 0 and 1. A randomized policy reports one mean. That mean and its influence curve are the
mixtures of the branch means and curves, with the known probabilities as weights.

`ratio="classifier"` replaces the density with the stacked classification of Section 5.4 of
Díaz et al. (2023). The treatment learner classifies an observed row against its policy copy,
and the ratio is the odds $u / (1 - u)$. This route fits no density, so the support report gives
`min_density` and `unsupported` as `None` and says that they are not measured.

Evidence: `tests/unit/test_policy_point_exact.py` checks the ratio and the influence curve of each
class on the exact laws of `tests/discrete_law_policy_point.py`. Its mutation controls drop the
inverse and the Jacobian, and each control fails. `tests/unit/test_influence_gateaux_rr_tilt.py`
checks the tilt on `tests/discrete_law.py`. The registered study is
[`policy-point-mtp`](method-evidence/point-modified-treatment-policies.md).

The bin count limits the density route at a point treatment as it does at a longitudinal node. The
default `density_bins=None` grows the count with the sample, `max(20, ceil(2 n^(1/3)))`. A
growing count is necessary for a consistent density, and the hazard learner must also be
consistent.

An explicit `density_bins=` is used as given. The memory paragraph of the
[longitudinal section](longitudinal-tmle.md#modified-treatment-policies-at-a-node) applies here
too. A coarse count
inflates the influence curve where the tail bins are wide, so the intervals are conservative.
When efficiency matters, use `ratio="classifier"` with a flexible classifier, or set
`density_bins=` higher. The
[longitudinal section](longitudinal-tmle.md#modified-treatment-policies-at-a-node) gives the
measurements.

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
[`interventions/policy.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/interventions/policy.py),
[`learners/density.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/learners/density.py),
and
[`fluctuation/submodel.py`](https://github.com/esbraun/cleverly-tmle/blob/main/src/cleverly/fluctuation/submodel.py).

### Vector treatments

A vector treatment holds several treatment columns. Díaz, Williams, Hoffman and Schenck (2023),
Section 2, let the treatment be a vector $A = (C, D)$. The components $C$ are categorical and the
components $D$ are continuous. Theorem 1 identifies the policy mean for such a vector. Part (ii)
reads the mean as the modified treatment policy effect, and part (i) reads it as a stochastic
intervention. At one decision both parts need the same exchangeability assumption.
`Identification.interpretations` records the two readings (`tests/unit/test_mtp_interpretations.py`).

Hoffman et al. (2024), Introduction, name interventions on several exposures at once. This
section covers the point treatment. The
[longitudinal section](longitudinal-tmle.md#vector-nodes-with-a-continuous-component) covers a
node of `LTMLE` and `SDR`.

| declaration | what `CausalData` stores | what the fit reports |
| --- | --- | --- |
| `treatment=["A1", "A2"]`, all columns categorical | one categorical treatment over the observed tuples. Each tuple is a level, and there are at most 20 | every estimand of a categorical treatment, with tuple levels. `policies=` takes a `JointPolicy` or a `ModifiedPolicy` that maps tuples |
| `treatment=["flag", "dose"]` with `continuous_treatment=["dose"]` | the components in an $(n, d)$ array, with `treatment_kind == "vector"` | `ey_policy` and `ate_policy` of `TMLE(policies=[JointPolicy(...)])`, and nothing else |
| the same, with `treatment_support={"dose": Support(lower, upper)}` | the declared support of the dose, given the covariates, the categorical components and the earlier doses | the same estimands. Before any learner, the fit refuses an observed dose outside the support and a branch that assigns a dose outside the support at the components the branch assigns (`tests/unit/test_mtp_vector_support.py`) |

`from_arrays` and `PointTreatment` take the same two keywords. A vector names two or more
distinct columns. `treatment_kind=` describes a treatment of one column, so a vector refuses it.
The treatment name joins the columns with `+`.

A vector with a dose has no arms. Every reader of an arm raises `CapabilityError`, and the
message names `TMLE(policies=[JointPolicy(...)])`. An all-categorical vector is one categorical
treatment, so it has arms and every arm-indexed estimand runs. A level is a tuple, and
`ey[(1, 2.0)]` is the mean when the intervention sets every component. The fit of every default
estimand equals the fit of the same data with the tuples pre-combined into one column
(`tests/unit/test_point_vector_treatment.py`, W12).

**The policy.** `JointPolicy(name, components={column: policy}, policy_kind="known")` declares
one policy for each component. A component that `components` leaves out keeps its natural value.
The table gives the policy classes and what each map reads. A read of any other component raises
`DataError`, and `h.get(...)` counts as a read.

| component | policy | what the map reads |
| --- | --- | --- |
| dose | `Shift`, `Scale`, `Piecewise`, or a `ModifiedPolicy` with `pieces` | the covariates, the natural values of the categorical components, and the natural values of the doses before it in `order` |
| categorical | a `ModifiedPolicy` with `apply`, an arithmetic map on numeric labels, or a `RiskRatioTilt` on a component coded 0 and 1 | the covariates and the natural values of every other component, doses included |

`order=` lists the doses, and it defaults to the column order. A dose map reads only the doses
before it in `order`. A map that reads a later dose raises `DataError`, and the message names an
`order` that resolves the read. Maps that read each other in a cycle are refused.

A dose map reads the natural flag, as the vector shift of `lmtp` does. It does not read the flag
that the policy assigns. `tests/unit/test_point_vector_treatment.py` (W4) sets the flag and gates
the dose on the natural flag. The law makes that estimand differ by more than $10^{-2}$ from the
estimand of a map that reads the assigned flag. A mutation that reads the assigned flag fails the
witness (M3, `tests/unit/test_point_vector_mutations.py`).

A `Randomizer` may sit on the `JointPolicy` or on one component. The component randomizers are
independent. The fit reads them as one joint randomizer over the product of their values, with the
product of their probabilities. A `RiskRatioTilt` of a binary component desugars to the two-point
randomizer that `lmtp::ipsi` draws, and it joins that product
(`tests/unit/test_vector_component_randomizer.py`).

**The ratio.** `ratio=` selects the route, and the default is `"density"`. The ratio is the joint
ratio $g^d(a \mid w) / g(a \mid w)$ at the observed vector.

| route | how the fit finds the ratio | what it needs |
| --- | --- | --- |
| `ratio="density"` | the chain rule. The fit estimates the law of the categorical components, then a pooled-hazard density for each dose in `order`. The induced density sums over the categorical preimages and applies Equation (3) of Díaz et al. (2023) to each dose in turn | the dose maps are strictly monotone piecewise maps with a declared inverse and derivative. Each map reads the doses before it in `order` and never a later one |
| `ratio="classifier"` | the stacked classification of Section 5.4 of Díaz et al. (2023). The design holds the covariates, an indicator for each categorical component, and the raw doses | the ratio exists and is bounded. The fit never evaluates a map's inverse |

The density of the induced law at $(c', b)$ is
$g^d(c', b \mid w) = \sum_c \sum_j 1\{d^C(c, \beta, w) = c'\}\, 1\{\beta_k \in I_{j_k}\}\,
g(c, \beta \mid w) \prod_k |\partial \beta_k / \partial b_k|$. Here $\beta = \beta(c, b)$ is the
preimage of $b$ along the pieces $j$, computed one dose at a time in `order`. The sum counts every
categorical cell $c$ that the policy sends to $c'$. The module docstring of
`cleverly/interventions/joint_density.py` gives the derivation. The identity piece of a capped or
uncovered dose is an explicit piece with inverse and derivative one. A shortcut that gives it the
ratio one is wrong when the cell or an earlier dose moved (W17, M14).

The fit checks the declarations before any learner. The checks run on each frame that the ratio
evaluates, which includes each candidate cell and each preimage of an earlier dose. A check
covers the partition of the pieces, strict monotonicity, the inverse, the derivative, and the
level of each categorical map. A `PositivityWarning` reports a dose that a policy assigns outside
the doses observed in the cell it sends a unit to. Positivity needs this: when the categorical
map moves a unit from cell $c$ to $c'$, the assigned dose must lie in the support of the dose in
$c'$.

**Bounds, cells and memory.** The density route bounds the mechanism of the categorical
components with `g_bounds`. The classifier route cannot split the ratio, so `g_bounds` does not
reach it. With a dose independent of the flag, a flag-only policy at the vector equals the
categorical fit of the flag with the bound binding
(`test_m1_a_flag_only_policy_equals_the_categorical_fit_with_the_bound_binding` in
`tests/unit/test_vector_density_route.py`).
The density route refuses a categorical cell that a training fold lacks, before any learner, and
its message names `ratio='classifier'`. The route also counts the counterfactual designs before it
allocates them. It raises `MethodConfigurationError` when they do not fit in the memory.

**What composes.** The table lists each composition and its evidence. The fit equals the fit of
the same data read as a scalar dose, where that reading exists
(`test_h1_every_reader_of_a_vector_result_equals_its_scalar_reading`).

| composition | status | evidence |
| --- | --- | --- |
| weights, clusters with `id=`, cross-fitting | supported. Integer weights equal the expanded sample, a cluster stays whole, and the classifier folds split units, not stacked rows | `tests/unit/test_point_vector_treatment.py`, on the classifier route |
| `delta=`, `strata=`, `intermediate=` | supported. The component columns enter the missingness, intermediate and outcome designs | `tests/unit/test_point_vector_coarsened.py`, on both routes |
| `dumps`, `loads`, `subset`, `retarget` | supported | `tests/unit/test_point_vector_treatment.py` |
| assessment and sensitivity rows | the capability table of a vector fit equals that of a scalar dose fit. A truncation row equals the refit at its bound. The simulated common-cause surface takes one strength axis for each component. The placebo permutes whole component rows | `tests/unit/test_point_vector_assessment.py`, `tests/unit/test_vector_simulated_confounding.py` |
| `CausalStudy` | `PointTreatment(treatment=[...], continuous_treatment=[...])` with `ModifiedTreatmentPolicy`. `ModelSpec(ratio=...)` selects the route, and `density_bins` reaches the density route | `tests/unit/test_vector_study_spine.py` |

`ModelSpec(ratio="classifier")` and the shortcut `ratio="classifier"` on a method reach both the
point and the longitudinal engine. A study refuses `ratio="classifier"` where it estimates no
policy ratio, such as an `ATE`, a declared mechanism, `DRTMLEMethod`, `CollaborativeTMLEMethod`
and `IPSIMethod`. Each refusal runs before any learner.

[Scope and refusals](scope-and-refusals.md#vector-treatments) lists what a vector refuses and the
theoretical reason for each refusal.

**Evidence.** `tests/unit/test_point_vector_treatment.py` checks five plans on the mixed law of
`tests/discrete_law_policy_point.py` (W1). It also checks the all-categorical reading (W2).
`tests/unit/test_vector_density_route.py` checks every plan on the density route. It adds a
many-to-one flag map (W16) and an ordered plan on a law with two doses (W15). Each check compares
the estimate and the influence curve with the g-formula and its complex-step Gateaux derivative.
The agreement is within $10^{-10}$.

The mutation controls are in `tests/unit/test_point_vector_mutations.py` and
`tests/unit/test_vector_density_mutations.py`. Each control moves a witness by more than
$10^{-4}$. The registered study is
[`vector-mtp`](method-evidence/vector-modified-treatment-policies.md).

### Incremental propensity-score interventions

For binary treatment, an incremental intervention multiplies the observed odds by $\delta$:

$$
q_\delta(1\mid W)=\frac{\delta g_P(1\mid W)}
                       {\delta g_P(1\mid W)+1-g_P(1\mid W)}.
$$

The denominator keeps the clever covariate bounded even when the observed propensity approaches
zero, so this parameter does not need conventional treatment positivity.

**Categorical treatment.** A treatment with $K$ levels takes one factor $\delta_a>0$ for each level. The intervention
draws the level from the exponential tilt of the observed mechanism:

$$
q_\delta(a\mid W)=\frac{\delta_a\, g_P(a\mid W)}{D_\delta(W)},
\qquad D_\delta(W)=\sum_b \delta_b\, g_P(b\mid W).
$$

Multiplying every factor by one constant changes nothing. The family therefore has $K-1$ free
parameters: the odds of level $a$ against level $b$ are multiplied by $\delta_a/\delta_b$. With
$K=2$ and $(\delta_0,\delta_1)=(1,\delta)$ the formula is the binary one above.

The package declares the factors as `Incremental({label: factor})`. A level that the mapping does
not name keeps the factor 1. A scalar `Incremental(delta)` names two arms, so a treatment with
other than two levels refuses it before any learner. The table lists the declarations.

| declaration | two levels | three or more levels |
| --- | --- | --- |
| `Incremental(2.0)` | the odds of the second sorted level times 2 | refused with `invalid_declaration` |
| `Incremental({"high": 2.0, "low": 0.5})` | the factors of the named levels, the others keep 1 | the same |
| `Incremental({"nope": 2.0})` | refused when `"nope"` is not a level | refused when `"nope"` is not a level |

Keys are the treatment labels that `CausalData.treatment_levels` reports. A vector treatment
whose components are all categorical is one categorical treatment over its observed tuples. Its
keys are the tuples. A vector with a continuous component is refused.

Identification is Kennedy (2019), Theorem 1, applied to a finite partition of the treatment. It
needs consistency and exchangeability. It needs no positivity, because
$q_\delta(a\mid W)=0$ wherever $g_P(a\mid W)=0$.

The derivative of $m(W)=\sum_a q_\delta(a\mid W)Q(a,W)$ in the mechanism is
$\partial m/\partial g_b=\delta_b\{Q(b,W)-m\}/D_\delta$. The mechanism score of level $b$ is
$1\{A=b\}-g_b$. Because $\sum_b \delta_b g_b\{Q(b,W)-m\}/D_\delta=0$, the sum collapses and
the efficient influence function is

$$
D=\frac{\delta_A}{D_\delta}\{Y-Q(A,W)\}+\frac{\delta_A}{D_\delta}\{Q(A,W)-m(W)\}+m(W)-\psi(\delta)
 =\frac{\delta_A}{D_\delta}\{Y-m(W)\}+m(W)-\psi(\delta).
$$

At $K=2$ the middle term is Kennedy's $\delta\{Q(1,W)-Q(0,W)\}(A-g)/D_\delta^2$. The clever
covariate $\delta_A/D_\delta$ lies in $[\min\delta/\max\delta,\ \max\delta/\min\delta]$, so it never
divides by $g$. `IncrementalSupport.guaranteed` reports this interval.

**The remainder is exact.** Let $\hat m$ and $\hat D$ be the tilted mean and normaliser at the
fitted nuisances, and $m_0$ and $D_0$ the values at the truth. Then

$$
R=E_0\!\left[(\hat m-m_0)\,\frac{\hat D-D_0}{\hat D}\right],
\qquad \hat D-D_0=\sum_b \delta_b(\hat g_b-g_{0b}).
$$

Every term of $R$ carries $(\hat g-g_0)$. Hence $R$ is zero when $\hat g=g_0$. No term is linear
in the error of one nuisance alone, so the construction exposes no linear remainder. The interval
holds under $\|\hat g-g_0\|(\|\hat g-g_0\|+\|\hat Q-Q_0\|)=o_P(n^{-1/2})$ with the
$L_2(P_0)$ norm over the levels. `tests/unit/test_categorical_incremental_exact.py` checks $R$
against the closed form to $10^{-12}$.

**The mechanism fluctuation is a softmax.** With three or more levels the treatment-mechanism
submodel is $g_\epsilon(a)\propto \hat g(a)\exp\{\epsilon^\top H(a)\}$. The covariate of
tilt $r$ is $H_r(a,W)=\{\delta_{r,a}/D_{\delta_r}\}\{Q^*(a,W)-m_r(W)\}$, the coefficient of level
$a$ in the mechanism score. The weighted log-likelihood is concave in $\epsilon$. A damped Newton
step with backtracking solves it and keeps every row on the simplex. Two levels keep the logistic
submodel and Kennedy's arithmetic bit for bit, with $\delta=\delta_1/\delta_0$. This is the binary
surface rule in [the architecture invariants](../architecture-invariants.md).

**Evidence class.** The estimator is validated by Python property evidence: exact-law checks of
the estimate, the curve and the remainder, deliberate-mutation controls, and the registered study
[categorical incremental policies](method-evidence/categorical-incremental.md). The rows `T35` match a three-level treatment, in sample or cross-fitted. The study fitted saturated learners, in sample and at five folds, and the rows match any learner and fold count. A categorical fit outside
those rows has the support label `experimental`. No external comparator exists for the categorical
tilt.

`tests/unit/test_categorical_incremental_compositions.py` checks four compositions against
identities. Integer weights equal row duplication. Singleton clusters equal the iid fit, and
clusters of four give the cluster standard error written longhand. Five exact copies of the sample
as five folds give the exact cross-fitted fit. A bootstrap replicate equals the fit on the
resampled rows. The mutation controls clear a margin of $10^{-4}$, except one that rescales the
rows of $g^*$, which clears $10^{-5}$ because it leaves the estimate unchanged.

**This is the one axis that targets the treatment mechanism.** The intervention is a functional of
$P$, so the efficient influence function carries a term for the pathwise derivative through $g$.
For binary treatment it is:

$$
D = \frac{\delta A + 1 - A}{D_\delta}\{Y-Q(A,W)\}
  + \frac{\delta\{Q(1,W)-Q(0,W)\}}{D_\delta^2}(A-g)
  + m(W) - \psi(\delta).
$$

The middle term lives in the tangent space of the treatment mechanism, so no fluctuation of $Q$
reaches it. The mechanism therefore gets a submodel of its own whose score is exactly that
term. The submodel is logistic for two levels and a softmax for more. Each covariate reads the other's fitted value, so the two alternate. The alternation is
coordinate ascent on one joint likelihood, because the outcome and treatment quasi-likelihoods are
separate factors, so the joint value never decreases. `score_check()` reports two rows for such a
fit rather than one.

Under a declared `stopping=` rule the outer exit judges the outcome equation by the rule ratio.
The mechanism equation keeps `tol`. [Stopping rules](#stopping-rules-step-bound-and-best-iterate)
gives the exit. A `treatment_adjustment=` declaration is refused beside `incremental=`. The shift
is a function of the fitted propensity score. A mechanism on a subset would define the shift on
that subset, and that changes the estimand.

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
refuses before any learner, because its mechanism equation may then have no finite root. The root is
missing only if some direction separates: every observed level maximises the covariate in its row.
That condition depends on the fitted outcome regression, so the guard reads the data alone and is
conservative, as the binary guard was.

Kennedy (2019),
arXiv v3, Section 6, lists "how mean outcomes under different interventions vary with covariates" as future
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

The regime support table reads two mechanisms. `min probability`, `max ratio` and `effective n`
read the treatment mechanism before truncation, which is what the data support. `score load` reads
the clever covariate of the targeting step, which divides by the truncated mechanism. A note under
the table states both facts. A bound that binds makes the two counts differ.
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
| [canonical point-treatment TMLE](method-evidence/canonical-point-treatment-tmle.md) | 34 accuracy tests, 17 comparator-backed paired comparisons against R `tmle3`, and 12 theory-property cells, test by test |
| [ordinary missing-outcome TMLE](method-evidence/ordinary-missing-outcome-tmle.md) | observational MAR truth tests, comparator-backed paired R `tmle` comparisons, and the three-nuisance robustness contract |
| [stacked arm-indexed missing-outcome CV-TMLE](method-evidence/stacked-arm-indexed-missing-outcome-cvtmle.md) | cross-fitted two-arm and three-arm MAR means and contrasts, comparator-backed paired R `tmle` comparisons, calibration, a simultaneous band, union-model cells, and overfitting controls |
| [controlled direct-effect TMLE](method-evidence/controlled-direct-effect-tmle.md) | both intermediate levels, comparator-backed paired recoded R `tmle` comparisons, four-nuisance robustness, exact efficiency, and a frozen native-result defect |
| [deterministic regimes](method-evidence/deterministic-point-treatment-regimes.md) | a dynamic rule and static reduction, comparator-backed against pinned R `lmtp`, with rule and targeting mutations |
| [known stochastic regimes](method-evidence/stochastic-point-treatment-regimes.md) | a comparator-free known-density study with stochastic-density and targeting controls |
| [continuous modified treatment policies](method-evidence/continuous-modified-treatment-policies.md) | natural, uncapped, and actively capped policies, comparator-backed against pinned R `lmtp`, with density-ratio and cap controls |
| [incremental interventions](method-evidence/incremental-propensity-interventions.md) | a treatment-mechanism-dependent curve, including its treatment-score control and a gated, comparator-backed R `npcausal` comparison |
| [the evidence manifest](evidence.md#the-table) | which oracle, Gateaux, remainder, and identity instruments each registered target has, and what none of them would see |
