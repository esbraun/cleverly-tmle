# Scope and refusals

`cleverly` refuses a composition it has not derived, rather than returning a convenient
approximation to a different estimand. This page says how to read such a refusal, and it tabulates
the one axis where the answer is least obvious: which surfaces take more than two treatment arms.

## How to read a refusal

A refusal is always *by name*. The keyword is accepted and rejected with a stated reason. It does
not arrive as an `unexpected keyword argument` that names none.

Refusals are not all the same kind of thing. What you should do about one depends on where the
problem is, and there are three places it can be.

| section | where the problem is | what to do about it |
| --- | --- | --- |
| [Not written yet](#not-written-yet) | in this package | the parameter is well defined and nobody has written it here. Ask for it, compute it elsewhere, or contribute it. Proposed work is on the [roadmap](../roadmap.md) |
| [A different question](#a-different-question) | in the question | what was asked for is a different estimand, usually with its own identification assumptions. Decide which one you meant. No flag here produces the other, and one that quietly did would answer something nobody asked |
| [Wrong by construction](#wrong-by-construction) | in the method | the naive version *runs* and returns a plausible number that is wrong, usually with a known direction of error. Read these as warnings about the analysis, not about this package's coverage |

A fourth group needs no taxonomy. A fit whose *data* cannot support what you declared is refused
where the problem arises. Examples are a horizon at which no event was observed among a regimen's
followers, a cause with no events, a regimen nobody followed, and two absorbing causes firing at
one node. Those are statements about the sample.

### Not written yet

Nothing is wrong with wanting any of these. They are gaps in coverage, and the message says so
rather than implying the request was ill-posed.

| refused | where |
| --- | --- |
| `DRTMLE` with observational missing outcomes, missing treatment, `intermediate=`, fold-wise targeting, `treatment_probabilities=` under `n_bootstrap=`, composition with `CTMLE`, or `reduction="bivariate"` composed with `delta=` | [method presets](../user-guide/methods-learners.md#method-presets) |
| the MNAR tilt on a `shifts=` fit | [modified treatment policies](../user-guide/estimands.md#modified-treatment-policies) |
| `intermediate=` and a multi-valued treatment with `incremental=` | [incremental interventions](../user-guide/estimands.md#incremental-propensity-score-interventions) |
| the targeted bootstrap and `res.sensitivity` for `LTMLE` | [longitudinal diagnostics](../user-guide/longitudinal.md#diagnostics) |
| longitudinal `msm=` with `n_folds > 1` | [MSM projections](msm-projections.md#the-longitudinal-projection). It needs an unsaturated projection property and a repeated-sampling study for coefficient inference |
| blocked-temporal and rolling-origin splits | [two fold layers](../user-guide/methods-learners.md#two-fold-layers) |
| replicate weights (BRR, jackknife) | [observation weights](../user-guide/data-design.md#observation-weights-are-not-estimand-weights). These are a set of designs rather than one weight vector, so the shape they want is a refit per replicate outside the estimator |

Which multi-arm surfaces are covered, and which four are not, is tabulated in one place:
[where a multi-valued treatment is supported](#where-a-multi-valued-treatment-is-supported).

Several former gaps have landed. Multi-valued longitudinal treatment nodes, multi-valued selector
and outcome-adaptive C-TMLE, multi-valued DR-TMLE, `ATT` and `ATC` on a multi-valued treatment,
observation weights and a working model over regimens for `LTMLE`, shift fits with `delta=`,
`intermediate=` and weights, and multi-arm omitted-variable and MNAR sensitivity analyses are all
supported now.

The remaining shift gap is narrower than it was. The tilt itself is written. The missing derivation
must establish whether the tilted parameter is still the shift parameter.

### A different question

These are well-posed parameters, but the selected estimator does not target them. No setting turns
one into the other.

| refused | the question it would answer instead |
| --- | --- |
| `eliminate=` on a competing-risks fit | the incidence of a cause if the competing events were *removed*. That intervenes on them rather than conditioning on the history. It needs a further factor per node in the denominator, and its own no-unmeasured-confounding and positivity assumptions for the competing event. What is reported instead is the incidence with the competing causes left alone |
| `intermediate=` on `LTMLE` | a controlled direct effect fixes a mediator at one time point. Over a sequence, with mediators that are themselves time-varying, that is a different identification rather than a further column |
| `ey1` and `ey_regime` from one fit; `msm=` with `interventions=` or `shifts=` | each keyword declares what "counterfactual" means for the fit, or how the counterfactuals are summarised. One fluctuation solves one set of score equations, so a fit reporting parameters from two axes would put two of them under one heading |
| the per-arm propensity table on a continuous fit; `stratify_folds="treatment+outcome"` on a continuous outcome or dose | a per-arm table has no rows when there are no arms. `diagnostics.support()` is not itself refused. On a fit that declared `shifts=` it dispatches to the question that does apply, which is whether the density *ratio* stays bounded |
| `res.sensitivity`, `res.diagnostics`, `res.validate()` and `res.save()` on an `LTMLE` result | each is part of the shared result contract. Stagewise support, scores, and nuisance loss are supported. Sensitivity operations without a longitudinal derivation report `unavailable` |

### Wrong by construction

These are worth reading even if you never reach for the keyword. Most are mistakes that are easy to
make by hand, in any framework, and none of them announces itself. Each produces a number, and the
number is wrong.

| refused | what goes wrong if it is done anyway |
| --- | --- |
| a `Stochastic` regime whose density came from the estimated mechanism | `g*` becomes a functional of `P`, so the influence curve carries a term for the pathwise derivative through `ghat` that a regime's curve does not have. The reported standard error is too **small** |
| an incremental intervention built by hand as a `Stochastic` regime | the same omission. Its size is exactly `Var(delta * (Qbar(1,W) - Qbar(0,W)) / D^2 * (A - g))`. Too small, always |
| a shift's inference taken from the regime inducing the same density | the means and the clever covariates agree entry for entry. The curves do not. The gap is `Var(Qbar(d(A,W),W) - E[Qbar(d(A,W),W) | W])`. Too small, always |
| a shift fit run on the complete cases when outcomes are missing | it is an ordinary shift fit on a *different* joint law of `(A, W)`, so it converges to a different number, and nothing in its own output says so. Measured on the dose fixture at 0.17, four standard errors, with a mechanism whose slopes are mild. `delta=` is what corrects it |
| a missingness or intermediate mechanism read at the observed dose rather than the assigned one | the fluctuation updates `Qbar` as a function of the dose, so `Qbar*(d(A,W),W)` is the update evaluated where the policy sends the unit. Silent wherever the mechanism does not depend on the dose, and invisible to a Gateaux check on an exact law |
| a "stabilised" MSM weighting `h` by the estimated mechanism | the same argument once more. `h` becomes a functional of `P`, and a term goes missing from the influence curve |
| `g_bounds=` or `truncation_curve()` on an `incremental=` fit | `g` is *inside* the estimand, so truncating it moves the parameter rather than regularising a denominator. The result is a number for a parameter nobody declared |
| a `cap=` fitted from the data on a shift | the estimand becomes data-dependent. The interval conditions on an estimated boundary, and every bootstrap replicate targets a slightly different policy |
| `CTMLE` on an `incremental=` fit | each candidate `ghat` defines a different estimand, so the cross-validated search selects between *estimands* rather than between estimators |
| splitting a cluster across folds to buy more of them | the out-of-fold predictions stop being independent of the rows they are used on, and the standard error shrinks in exactly the direction `id=` was passed to prevent |
| median-of-estimates aggregation over `repeats=` | the median of the estimates is not the estimator whose curve is the median of the curves, so the point estimate and its interval describe different functionals |
| a cross-validated variance of the across-draw average curve | at equal fold sizes it collapses to the pooled uncentred second moment for *every* partition. That is vacuous rather than merely arbitrary |
| a one-shot non-identity-link MSM | the derivative of the inverse link depends on the coefficient, so a single pass reports a standard error for an equation it did not solve. The link is supported. What is refused is skipping the alternation it needs |
| frequency (count) weights | they assert a sample size the variance does not use. Expand the rows instead, which says the same thing where every part of the fit can see it |
| an `LTMLE` outcome missing for a reason other than censoring | its probability of being observed is silently taken to be one. Encode it as a final censoring column so that it is estimated and enters the cumulative product |
| a binary-only target on a multi-arm fit | it would report a contrast of arms `0` and `1` out of five, under the name of a parameter about all of them. Targets declare `requires_binary_treatment` for this |
| `MSM.linear` on non-numeric arm labels | a model linear in the arm reads it as a dose to interpolate between, and the fallback coding is the sort order. That is a dose scale nobody chose |

Three of these share one mechanism, and it generalises past this package. **If an intervention's
density is a functional of `P`, the influence function carries a term for the pathwise derivative
through the estimated mechanism.** Omit that term and the standard error is too small. Under
positivity the missing variance is positive, and the table above writes it down exactly.

Two more share another. **A nuisance that sits inside the estimand is not a knob.** Truncating `g`
on an incremental fit, or fitting a shift's `cap` from the data, moves the target rather than
regularising the estimator.

Two method entries carry their own detailed refusal tables, each with a `kind` column naming which
of the three sections above the row belongs to:
[marginal structural models](msm-projections.md#variations) and
[longitudinal TMLE](longitudinal-tmle.md#variations).

## Where a multi-valued treatment is supported

A multinomial treatment mechanism is the default construction here rather than a variant of a
binary one. `CausalData` carries `treatment_levels`, `Propensity` holds an `(n, K)` simplex, and
`learners._fitting.predict_probabilities` produces it. Two arms are the `K = 2` case of that
construction and not a separate path, which is what the
[bit-for-bit invariant](../architecture-invariants.md#dataframes-and-labels) is about.

So "does this estimator take more than two arms" usually has the answer "yes, through the same code
as two". The informative entries are the four that do not. The `status` column uses the vocabulary
of [How to read a refusal](#how-to-read-a-refusal) above, plus `waiting on published theory` from
the [roadmap's eligibility rules](../roadmap.md#eligibility).

| surface | status | why |
| --- | --- | --- |
| `TMLE`: `ey` per arm, `ate` / `att` / `atc` per non-reference arm, regimes, MSMs over arms | supported | one counterfactual mean per arm and one contrast per non-reference arm. The [oracle-law gate](validation-methods.md#the-oracle-law-gate) states that a target meant for more than two arms needs a branch on the three-armed law |
| `DRTMLE`: univariate and bivariate reductions | supported | [armwise one-vs-rest](dr-tmle/index.md#variations), with each reduction and correction indexed by a free level. The cited theorem is binary, so this is an implementation-backed armwise extension rather than a claim about that theorem's literal scope |
| `CTMLE`: selectors and `strategy="oat"` | supported | one shared `n x K` categorical mechanism, selected against one nonredundant vector. See the [standing decision](../architecture-invariants.md#targets-interventions-and-variants) |
| `LTMLE`: categorical nodes, static and dynamic regimens | supported | [treatment over time](longitudinal-tmle.md#the-algorithm-as-implemented). Each node owns its level set, and the clever covariate selects the assigned label's probability |
| positivity, omitted-variable, E-value and MNAR sensitivity | supported | each is one parameter per contrast, and each reads its arms from the parameter's structured index rather than assuming two |
| `ey1` / `ey0` and the incremental estimands, on a multi-arm fit | wrong by construction | they *name* one of exactly two arms, so on five arms they would report a contrast of arms `0` and `1` under the name of a parameter about all of them. Declared by `requires_binary_treatment`. The multi-arm path reports per-arm `ey` instead |
| `incremental=` itself, above two arms | a different question | an odds multiplier names two arms. One odds per contrast is well posed, and it is a *different intervention* with a different influence function rather than a generalisation of this one |
| stochastic categorical policies and continuous doses at a longitudinal node | a different question | both change the intervention *density* rather than which label is assigned, so neither is the parameter the sequential regression identifies |
| `DRTMLE` with `delta=` at more than two arms | waiting on published theory | Diaz and van der Laan's missing-outcome theorem is stated for a binary randomized treatment, and the per-arm assembly of its observation, treatment and outcome correction blocks is not in it. See the [roadmap](../roadmap.md#d1-multi-arm-missing-outcome-dr-tmle) |

The last row is the only one a source could close as it stands. Neither `a different question` row
would be closed by a source. Each would be answered by a different estimand, with its own
derivation, oracle law, and evidence.
