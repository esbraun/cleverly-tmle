# Parametric g-formula

## What this solves

`ParametricGFormula` estimates the regimen levels and contrasts that [`LTMLE`](longitudinal-tmle.md)
estimates. It uses the same identification assumptions and the same target. It differs in how it
reaches the target. `SequentialGComputation` regresses backward through the nodes. This estimator
models the law of each time-varying covariate and simulates each unit forward under the regimen
(Robins 1986; McGrath et al. 2020).

| estimator | what the estimate reads | interval |
| --- | --- | --- |
| `ParametricGFormula` | one model per time-varying covariate at nodes $2, \ldots, T$, and the outcome model (or one hazard model per node) | the stacked estimating equation of the `GLM` models, with the simulation error |
| `SequentialGComputation` | the outcome and pseudo-outcome regressions | the stacked estimating equation of the `GLM` regressions |

Both intervals hold for the target only when every model that the estimate reads is correctly
specified. The parametric g-formula needs more models than the backward form, and it needs each of
them to be correct. A fit with the backward form tolerates a misspecified covariate law that this
fit does not. [When to choose it](../user-guide/longitudinal.md#simulate-the-parametric-g-formula)
lists the reasons to pay that price.

```python
from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import ParametricGFormula

frame, _ = make_longitudinal(n=300, seed=0)
result = ParametricGFormula(
    {"always": 1, "never": 0},
    reference="never",
    covariate_types={"L2": "normal"},
    random_state=0,
).fit(
    frame,
    outcome="Y",
    treatment=["A1", "A2"],
    baseline=["W1", "W2"],
    time_varying=[[], ["L2"]],
    censoring=["C1", "C2"],
)
print(result.summary())
```

`gformula()` is the one-call form. `ParametricGFormulaMethod` (name `"parametric_gformula"`) is the
study method.

## Status

| item | state | evidence |
| --- | --- | --- |
| point estimates and stacked interval, `GLM` models | shipped, status `influence_curve` | the unit witnesses [below](#validation) |
| repeated-sampling evidence | registered study `parametric-gformula`. The method-evidence page `docs/technical-reference/method-evidence/parametric-gformula.md` lists the cells | Python property evidence |
| the `pygformula` comparator run | pending. The packages fit different models ([below](#how-it-differs-from-gformula-and-pygformula)) | none |
| a published analytic variance for the simulation form | none exists | [Inference](#inference) |
| pooled-over-time models | not shipped. [X73](../roadmap.md#x73-pooled-over-time-parametric-g-formula-models) tracks it | none |
| a repeated outcome | refused. [X71](../roadmap.md#x71-repeated-outcomes-for-the-sequential-baselines) tracks it | none |

## The estimator

### The models

Write $H_t$ for the history before node $t$: the baseline, the earlier covariates and the earlier
treatments. The fit estimates the following models. Each model has its own node and its own rows.

| model | response | fit rows |
| --- | --- | --- |
| covariate $L_{t,j}$, $t \ge 2$ | binary, categorical or Gaussian, given $H_t$ and the earlier columns of node $t$ | uncensored and event-free through node $t - 1$ |
| outcome, end of study | the outcome at node $T$ given the full history | uncensored through node $T$ |
| hazard at node $t$, survival | the baseline-category multinomial over {none, cause 1, ...} | uncensored through node $t$ and event-free before $t$ |
| natural treatment at a tilt node, at a policy node $t \ge 2$, or (as a diagnostic) at a fixed-dose or rule-set dose node | binary, categorical or Gaussian, given $H_t$ | uncensored and event-free through node $t - 1$ |

Every model is node-specific and uses main terms by default. `GLM(interactions=k)` adds products.
The baseline and the node-1 covariates are each unit's observed values. A categorical covariate
enters every later design as drop-first indicators. The plan intervenes on censoring, so the
simulation draws no censoring.

### The simulation

The fit replicates each unit $m$ times (`n_draws`, default 10). Each copy draws its covariates
forward in the declared order within a node. The regimen then sets the treatment at the node.

| regimen at node $t$ | the simulated treatment |
| --- | --- |
| a static arm | the arm |
| a known rule at a categorical node | the rule, read on the simulated history |
| a known stochastic policy | a draw from the policy density on the simulated history |
| an odds tilt $\delta$ | a draw from $\delta g / (\delta g + 1 - g)$, with $g$ the natural law. This includes node 1 |
| a modified treatment policy | the policy applied to a natural draw from the fitted model. Node 1 takes the observed $A_1$ as the natural value |
| a fixed dose at a continuous node | the dose |
| a known rule at a continuous node | the dose that the rule sets on the simulated history. The rule draws nothing |

Robins (1986), equation (4.7) in Section 4E, p. 1423 (Corollary 4.1), gives the formula that this
simulation evaluates. Section 4I, pp. 1426 to 1427, gives the Monte Carlo evaluation. That
evaluation averages predicted survival products and draws no deaths. The outcome functional of a
copy follows it.

| outcome | the functional of a copy |
| --- | --- |
| end of study | the predicted outcome mean at node $T$. The fit never draws an outcome |
| survival or competing risk | the discrete cumulative incidence $F_j(k) = \sum_{t \le k} S(t - 1)\, h_{jt}$ along the copy's hazards |

The estimate is $\hat\psi = \sum_i w_i \hat\Phi_i / \sum_i w_i$, with $\hat\Phi_i$ the mean of the
functional over the $m$ copies of unit $i$.

### The random numbers

Each draw is an inverse-CDF transform of a uniform. A Philox4x32-10 generator makes the uniform
(Salmon et al. 2011). The uniform is a pure function of five indices: stream, unit, copy, node and
column. The key is the fit's simulation seed, which comes from `random_state`.

| property | consequence |
| --- | --- |
| the draws use the inverse CDF only | no draw consumes a number of uniforms that depends on the parameters |
| every regimen reads the same uniforms | the regimens share their simulation error (common random numbers). A contrast carries the smaller error of the difference |
| stream 0 gives the $m$ copies behind $\hat\Phi_i$ | the estimate |
| stream 1 gives `n_derivative_draws` copies | the derivatives. Their error is independent of the estimate. The default is $\max(m, 10)$, and the minimum is 2 |

`tests/unit/test_parametric_gformula.py` (`TestPhilox`) checks the block function against the
three Random123 known answers.

## Inference

### The stacked curve

Each model solves its own score equation. No model's target depends on another model's parameter,
so the Jacobian of the stacked system is block diagonal. A Gaussian model adds the variance
equation $P_n[w R (r^2 - \sigma^2)] = 0$ with the maximum-likelihood divisor. The cross block
is then exactly zero. The curve is

$$
\mathrm{IF}_i = \frac{w_i}{\bar w}(\hat\Phi_i - \hat\psi) + \sum_k D_k^\top M_k^{-1} S_{k,i}.
$$

$S_{k,i}$ is the score of model $k$ at unit $i$. $M_k$ is its information over $n$. $D_k$ is the
derivative of $\hat\psi$ in the parameters of model $k$. The curve uses $\hat\Phi_i$, so its
variance is the statistical variance plus the simulation variance. A cluster is the unit of the
curve.

| model | $D_k$ |
| --- | --- |
| outcome mean at node $T$; a hazard | pathwise. The derivative runs through the link, and for a hazard through the incidence sum |
| a drawn covariate; a drawn natural treatment; a tilted draw | the score-function identity $\partial_\theta E_\theta[f] = E_\theta[(f - b)\, \partial_\theta \log p_\theta]$. The baseline $b$ is the mean of the unit's other copies in stream 1 |

The baseline does not depend on the copy's own draw, so the identity holds exactly. The identity
needs no smoothness of $f$ in the drawn value. A downstream binary draw breaks smoothness, and the
identity is unaffected. The tilted draw uses the score of the tilted law.

### The simulation error

Conditional on the data, the copies are independent across units. Unit $i$ has
$\hat\Phi_i = \Phi_i + e_i$, with $E[e_i \mid \text{data}] = 0$ and $\mathrm{Var}(e_i \mid
\text{data}) = \sigma_i^2 / m$. Stream 1 estimates $\sigma_i^2$ from the within-unit variance of
the copies. This estimate is valid at `n_draws = 1`.

The result reports the share of each curve's variance that the simulation contributes:
$\sum_i (w_i / \bar w)^2 \sigma_i^2 / m$ divided by $\sum_i \mathrm{IF}_i^2$. A cluster sum
replaces the unit in the denominator. A contrast reads the difference of the two regimens' copies.
`result.simulation.monte_carlo_share` holds the shares, and `summary()` prints the largest.
The fit warns with `MonteCarloWarning` above 10%. Raise `n_draws` to lower the share.

### Why the argument holds

Augment each observation $O_i$ with its uniforms $U_i$. The pairs $(O_i, U_i)$ are independent and
identically distributed. Then $\hat\psi = P_n[w\, \bar f(O, U; \hat\theta)] / P_n w$ is a two-step
estimator. It has the influence function
$w(\bar f(O, U; \theta_0) - \psi)/\bar w + D^\top M^{-1} S$, because
$E[\bar f - \Phi \mid O] = 0$. The variance is then the statistical variance plus
$E[w^2 \sigma^2] / (m \bar w^2)$. This holds for a contrast under common random numbers and for
cluster sums.

The simulated function is not smooth in $\theta$. A binary draw is $1\{U < p_\theta\}$. The
step from the augmented function to its derivative therefore needs stochastic equicontinuity.

| step | source |
| --- | --- |
| the copy-averaged functional of indicator draws at fixed uniforms is a Euclidean class, continuous in $L_2$ at $\theta_0$ | Pakes & Pollard (1989), Lemma 2.17, p. 1037. Lemma 2.16, p. 1036, states the equicontinuity |
| the estimator is asymptotically normal, and the derivative is that of the expectation | Pakes & Pollard (1989), Theorem 3.3, p. 1040 |
| the same step for a simulated, nonsmooth estimating equation | Newey & McFadden (1994), Theorem 7.2, p. 2186, condition (v) |

Theorem 7.3, p. 2188, of Newey & McFadden does not cover a discontinuous simulator. The argument
uses Pakes & Pollard for the primitive conditions. The project applies these results to this
estimator. The sources do not state the result for it.

**No published analytic variance exists for the simulation form.** McGrath et al. (2020) and
`gfoRmula` 1.1.1 report the nonparametric bootstrap. Neither gives an analytic variance or an
account of the Monte Carlo error. Zivich et al. (arXiv 2306.10976v3, Section 3.1) give a sandwich
variance for the iterated conditional expectation form only.

The interval of this page is therefore a natural extension. It combines a stacked estimating
equation, the chain rule, and the simulated-estimator step above. Its repeated-sampling evidence is
the registered study `parametric-gformula`, which includes cells at `n_draws = 1`. See the
method-evidence page `docs/technical-reference/method-evidence/parametric-gformula.md`.

### Status

| fit | status | what the result reports |
| --- | --- | --- |
| every model that the estimate reads is a `GLM`, or a node with one observed class | `influence_curve` | the stacked curve, with an interval |
| a model that the estimate reads uses another learner | `adaptive_baseline_plugin` | the point estimate, and a diagnostic curve under `plugin_std_error`. The curve drops each non-`GLM` block |
| `weights_estimated=True` | `estimated_weight_plugin` | the stacked curve as a diagnostic |

The models that the estimate reads are every covariate model, the outcome model, every hazard
model, and the natural-treatment model of each tilt node and each policy node $t \ge 2$. A node
whose treatment mechanism the data declares has no model (see
[Declared treatment mechanisms](#declared-treatment-mechanisms)). A learner that the estimate does
not read leaves the status unchanged. No result gives the asymptotic linearity of the simulation
form with data-adaptive learners.

The full-refit bootstrap runs and reports `bootstrap sd`, as for the other baselines.

## Declared treatment mechanisms

A tilt node, and a modified treatment policy at a node $t \ge 2$, draw the natural treatment. When
the data declares the mechanism of that node (`treatment_probabilities=`), the draw reads the
declared probabilities of the unit. The fit fits no natural-treatment model at that node and adds
no score term. The summary line "known factors (declared, no learner)" then names the node.

| condition | consequence |
| --- | --- |
| the declared law does not depend on a covariate that the simulation redraws | the draw is the declared law |
| the declared law depends on the observed past | the draw uses each unit's row at every simulated history. It is the declared law only if the simulation leaves that past unchanged |
| a unit with positive weight has no declared value at the node | `CapabilityError` before any learner, with the count of units. The simulation draws every unit at the node, including a unit that left the study earlier |

`TestDeclaredNaturalTreatment` checks the draw against the exact law of a step-function mechanism
that no main-terms logistic model reproduces.

## Fixed doses, rule-set doses and model reliance

`LTMLE` refuses a fixed number, and a rule that sets a dose, at a continuous node, because the
nonparametric target is not pathwise differentiable. Under a parametric model the target is a
smooth function of a finite parameter, and the delta method applies. `ParametricGFormula` accepts
both kinds of node, and the result states the reliance.

A rule at a continuous node must be a `DynamicRegimen` with `rule_kind="known"`. The fit calls the
rule on the history frame of each simulated path and draws nothing. The dose is a function of
the simulated covariates, so it has no score term of its own. The score term of each covariate
model multiplies a functional that already includes the dose. A rule that returns a value that is not finite
raises `DataError`.

| item | content |
| --- | --- |
| `result.simulation.model_reliance` | always set when a plan holds a fixed or rule-set dose. It says that the estimate and interval hold inside the fitted laws, which the fit extrapolates to the dose at every node |
| support rows | for each fixed or rule-set dose and node, the weighted share of simulated histories whose fitted mean dose lies more than 2 fitted standard deviations from the dose, and the weighted share of observed doses within 2 fitted standard deviations of the dose |
| support row of a rule | `dose` is `None`. The dose of each simulated history is the rule on that history. The window around each observed dose is centred on the rule on the observed history |
| the dose models | fitted for the diagnostic only. They are not in the stack |
| mechanism diagnostics | not available for any regimen of a fit with a fixed or rule-set dose. A dose has no density ratio |

A large off-support share means that the model carries the estimate with little help from the
data. The share is not a test. It reports where the extrapolation happens.

## Positivity

The estimate exists without positivity. Where no unit follows the regimen, the fitted laws
extrapolate. The interval holds only if the laws are correct there. The mechanism diagnostics of
the fit stay the check on overlap for a static, rule or policy regimen.

One check runs for every model. A `GLM` drops a design column that is constant, or a combination
of other columns, on its fit rows. When a simulated row departs from that relation, no unit with
that history follows the regimen, and the prediction would assume that the column has no effect.
The fit refuses with `DataError` on either simulation stream.
`test_an_aliased_column_the_regimen_leaves_is_refused` checks it.

## The order within a node

Several covariates can share one node. The model of the second reads the first, and the model of
the first would have read the second under the opposite order. The two orders define non-nested
models. At most one order is correctly specified in general. The draw order is the order of the
columns in `time_varying`. `result.simulation.orders` records it, and the summary repeats the
caveat. `TestOrderAndNodeOne` checks that the simulation uses the order of the fit.

## Settings

| setting | meaning |
| --- | --- |
| `n_draws` | the copies per unit behind the estimate. The default is 10 |
| `n_derivative_draws` | the copies per unit behind the derivatives. The default is $\max(m, 10)$, with a minimum of 2 |
| `covariate_types=` | maps a column, or a tuple of one-hot columns, to `"binary"`, `"categorical"` or `"normal"`. Only a column coded 0/1 is inferred as binary |
| `covariate_learner=` | the template of every covariate model. A Gaussian covariate needs a `GLM` |
| `outcome_learner=` | the outcome model of an end-of-study outcome. The default is a main-terms `GLM` of the outcome's family |
| `hazard_learner=` | the template of every hazard model of a survival outcome |
| `treatment_learner=` | the natural-treatment models and the diagnostic mechanism |
| `random_state=` | seeds the simulation. A fit repeats exactly |

The default learner library is a `GLM`. A `GLM` family other than `"auto"` is refused, because the
fit chooses the family of each model from its response.

## Refusals

| request | error |
| --- | --- |
| `n_folds > 1` | `CapabilityError`, `implementation_gap`. The fit is in sample only |
| `msm=` | `CapabilityError`, `api_redirection`. `LTMLE(msm=...)` fits a working model |
| `alpha=`, `max_iter=`, `tol=` | `MethodConfigurationError`. No fluctuation exists |
| `pseudo_learner=` | `CapabilityError`. The fit has no pseudo-outcome regression |
| a risk-ratio tilt | `CapabilityError`, `api_redirection`. The fit simulates an odds tilt only |
| an undeclared time-varying column that is not coded 0/1 | `CapabilityError`, before any learner. The message names `covariate_types=` |
| a Gaussian covariate, or a stacked continuous natural treatment, with a learner that is not a `GLM` | `CapabilityError`. The in-sample residual variance of another learner is biased down |
| a learner that nothing reads, such as `hazard_learner=` on an end-of-study outcome | `CapabilityError` |
| a continuous outcome, a binomial `GLM`, and `q_bounds=None` | `CapabilityError`, as for `SequentialGComputation` |
| a declared treatment mechanism with a missing value at a tilt or policy node | `CapabilityError` |
| a repeated outcome | `CapabilityError`, `implementation_gap`. [X71](../roadmap.md#x71-repeated-outcomes-for-the-sequential-baselines) tracks it |
| a truncation curve | `CapabilityError`, `api_redirection`. The estimate divides by no mechanism |
| a regimen that leaves the span of an aliased `GLM` column | `DataError`, from the simulation |

## How it differs from `gfoRmula` and `pygformula`

McGrath et al. (2020) describe `gfoRmula`. `pygformula` 1.2.0 implements the same algorithm in
Python. This estimator follows the same logic, and the table lists each difference. File and line
references are to `gfoRmula` 1.1.1 and `pygformula` 1.2.0.

| item | `gfoRmula` and `pygformula` | `ParametricGFormula` |
| --- | --- | --- |
| time-0 row | the observed row is used as is, and only the time-0 treatment is overwritten (McGrath et al. 2020, Supplement 1, p. 6, step 2a; `simulate.R:195-222`) | the same. A modified treatment policy takes the observed $A_1$ as the natural value |
| Monte Carlo sample | `nsimul` defaults to $n$ and draws each unit once. A different `nsimul` draws ids with replacement (`gformula.R:962-964, 1022-1042`; `pygformula` `parametric_gformula.py:548-551`) | a fixed number of copies per unit. The fit never resamples units |
| outcome | the predicted mean at the last time (McGrath et al. 2020, Supplement 1, p. 8, eq. 10; `simulate.R:580-600`) | the same |
| models over time | one pooled model per covariate, fitted on all rows with time greater than 0 (`pred.R:298`) | one model per node and covariate |
| normal covariates | one residual standard deviation per covariate, divisor $n$ (`helper.R:425-427`). The draw is clamped to the observed range by default (`gformula.R:977-987`) | a node-specific standard deviation with the same divisor. The draw is not truncated |
| categorical covariates | the code takes the most likely class (`simulate.R:350`), although the paper states a multinomial draw | a multinomial draw |
| inference | the nonparametric bootstrap of ids, with refit and re-simulation (`bootstrap.R:130-200`) | the stacked curve with the simulation term. The bootstrap stays available |
| Monte Carlo error | enters through the bootstrap only | a reported share of each variance |

The packages differ from this estimator in the model class. A comparison of estimates is a
comparison of two model classes, unless the data come from a law where both are correct.
`pygformula` cannot express node-specific covariate models, so a matching comparison needs a
two-node panel for the covariate laws. The `pygformula` comparator run is pending.

Pooled-over-time models with lag and cumulative-average terms are the default of both packages.
[X73](../roadmap.md#x73-pooled-over-time-parametric-g-formula-models) tracks them.

## Validation

All witnesses are in `tests/unit/test_parametric_gformula.py`. Each exact-law witness enumerates
the law with complex-step arithmetic, so it does not share code with the simulation.

| witness | what it checks | test class |
| --- | --- | --- |
| W1 | the estimate equals the exact g-formula on saturated binary, survival and competing-risk laws, within 4 Monte Carlo standard errors | `TestExactLaws` |
| W2 | the estimate equals the enumerated functional at the fitted parameters, off a saturated law | `TestProductionLawB`, `TestProductionLawR` |
| W3a | with the exact terms in place of the simulation, the curve equals the derivative of the functional in each observation weight, with non-uniform weights and clusters | `TestExactTermsLawB`, `TestGenericGateauxCurves`, `TestExactTermsClusterLawB` |
| W3b | the production derivative equals the exact one for each draw kind (binary, categorical, Gaussian mean and variance, natural treatment, tilt) and each pathwise kind (outcome, hazard) | `TestProductionLawB`, `TestProductionLawR`, `TestOddsTilt`, `TestModifiedTreatmentPolicy`, `TestSurvivalAndCompetingRisks` |
| W4 | the curve variance minus the exact statistical variance equals the simulation variance, at `n_draws` 1 and 200 | `TestProductionLawB` |
| W5 | the curve has mean zero | `TestProductionLawB` |
| W6 | the contrast curve is the difference of the level curves, and common random numbers shrink the simulation part of a contrast | `TestProductionLawB` |
| W9 | on a saturated law the estimate equals the nonparametric plug-in, and the statistical curve equals the efficient influence curve | `TestExactLaws` |
| W10 | a fixed dose, and a rule that sets a dose from the simulated history, on a linear-Gaussian law: the estimate and delta-method standard error equal the closed form. The rule also meets the weight-derivative check and the exact production derivative | `TestFixedDose`, `TestRuleSetDose` |
| share | the reported simulation variance equals the variance of the estimate across 300 seeds, for two levels and a contrast | `TestShareAgainstTheSeeds` |
| cluster share | the share divides by the clustered variance of the curve | `TestClusterMonteCarloShare` |
| support | the window share and the off-support share equal an independent least-squares computation, with and without weights, and for a rule | `TestFixedDose`, `TestRuleSetDose` |
| declared mechanism | a declared step-function mechanism gives the exact value, and a fit that ignores the declaration misses it | `TestDeclaredNaturalTreatment` |
| order, node 1 | the simulation uses the fit order and the observed node-1 covariates | `TestOrderAndNodeOne` |
| refusals | each refusal arrives before any learner | `TestRefusals` |

Each mutation control changes one source line of `gformula.py` in a copy of the tree. The named
witness fails.

| mutation | witness that fails |
| --- | --- |
| a binary draw score dropped, or its sign flipped | W3b, `TestProductionLawB` |
| the Gaussian variance equation dropped from the curve | W3a, `TestGenericGateauxCurves` |
| the Gaussian score without the factor $1/\sigma^2$ | W3b, `TestProductionLawR` |
| the Monte Carlo term halved by averaging two streams | W4, `TestProductionLawB` |
| a baseline that includes the copy's own value | W3b, `TestProductionLawB` |
| a weight dropped from one score, or from the outer mean | W3a, `TestExactTermsLawB` |
| unit sums in place of cluster sums in the share | `TestClusterMonteCarloShare` |
| the survival factor that drops the competing cause | W1, `TestExactLaws` |
| a tilt score that ignores the tilt | W3b, `TestOddsTilt` |
| per-regimen streams | W6, `TestProductionLawB` |
| the draw order reversed in the simulation only | `TestOrderAndNodeOne` |
| a declared mechanism ignored | `TestDeclaredNaturalTreatment` |
| a support window read in variance units | W10, `TestFixedDose` |
| a rule-set dose read on the observed history in place of the simulated history | W3b and W10, `TestRuleSetDose` |
| a rule-set support window centred at zero, or no support row for a rule | W10, `TestRuleSetDose` |

No mutation simulates censoring. The estimator has no code path that does, because the plan
intervenes on censoring.

## References

- Robins (1986), *Mathematical Modelling* 7:1393 to 1512.
- McGrath et al. (2020), *Patterns* 1(3):100008.
- Pakes & Pollard (1989), *Econometrica* 57(5):1027 to 1057.
- Newey & McFadden (1994), *Handbook of Econometrics* IV, chapter 36.
- Stefanski & Boos (2002), *The American Statistician* 56(1):29 to 38.

[References](../references.md) lists the pages read.
