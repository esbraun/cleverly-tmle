# DR-TMLE: doubly-robust inference

`DRTMLE` is a TMLE whose *interval* stays valid when only one primary nuisance is consistently
estimated. Each guarantee needs the rate conditions in
[the remainder terms](#the-remainder-terms-and-the-rate-conditions). A plain TMLE's exact remainder
is a signed integral. Under positivity its absolute value is bounded by a product of the two
nuisance errors.

That bound is why the *point estimate* is doubly robust for consistency. The interval needs the
stronger condition `√n R_2 → 0`, and with one error not shrinking the remainder is first order in
the other. So **`TMLE` is doubly robust for consistency and singly robust for inference**, and
`DRTMLE` closes that second gap by solving two additional reduced-regression score equations.

This document is the reference for what is supported, what the theorem covers, what the
implementation chooses where the theorem is silent, and what a caller has to check. The runnable
recipe is in [the user guide](user-guide.md#doubly-robust-inference); the derivation of what the
extra equations remove is in
[the DR-TMLE reference entry](technical-reference/dr-tmle.md#what-this-solves).

## The release claim, in one paragraph

**Conditional validity.** The default univariate algorithm computes what Benkeser, Carone, van der
Laan & Gilbert's Theorem 1 derives; `reduction="bivariate"` computes van der Laan (2014), Theorem
3's earlier binary construction. Checks compare both constructions with their remainder
derivations and with the parameter's Gateaux derivative. The exact finite-support laws and the
remainder identities run at three arms as well as two, in the union-model cells where exactly one
correction survives. The reported interval is valid **conditional on** the practitioner obtaining
adequate primary and reduced-regression fits. Those are rate conditions on estimated functions.
They are not verifiable from a fit's own output, and in particular **numerical score convergence
does not verify them**; see [section 6](#6-solved-scores-do-not-establish-nuisance-consistency),
which is the single most important thing on this page.

What this is *not*: a better point estimate. The three empirical means are all driven to zero, so
the extra terms cannot move `Ψ̂` and only move its variance. Read a `DRTMLE` fit as the same
estimate with an interval entitled to be believed under weaker conditions.

And it is *not* the efficient estimator. Under misspecification the canonical gradient at `P_0` is
still `D*`. What the three equations leave is `D = D* − D*_Q − D*_g`, the estimator's asymptotic
influence function at the nuisance limits, and it is generally not efficient there. When both
nuisances are consistent, the corrections converge to zero and the curve approaches the ordinary
efficient curve. At the true nuisance functions, the corrections vanish row by row.

---

## 1. Supported estimands


A **discrete point treatment** and the `mean` group: every treatment-specific mean and the
reference-arm contrasts requested through `ey` and `ate`. For multiple levels, the
univariate implementation follows R `drtmle`'s armwise construction (`R/fluctuate.R`, `fluctuateG`):
reduced regressions and the two extra equations are fitted once per arm, and equation (9)
independently fluctuates each one-vs-rest mechanism margin with response `1(A = a)`, offset
`logit(g_a)`, covariate `Qr_a / g_a`, one scalar per arm.

The targeted margins are **not renormalised**, and the reason is that what this estimator
owes is a set of solved score equations rather than a likelihood: projecting the `K` tilted
margins back onto the simplex would move every one of them off the root just found. They
are not inert, and the estimate does see them: the next round's equation (8) divides by `g_a*`,
so the targeted `Qbar*` and hence `psi` do depend on margins that sum to something other than
one (measured: `0.9975` on the three-armed fixture). That is what `drtmle` does too, and it is
licensed by the score equations rather than by the mechanism still being a conditional
distribution. The initial categorical mechanism *is* compatible and sums to one, using
cleverly's existing multiclass learner path; `diagnostics.support()` reports how far the
targeted rows depart from it.

**Two arms keep their own route, which is not the armwise one.** `drtmle` fluctuates both
margins independently even at `K = 2`; cleverly instead tilts `g_1` alone along a
two-column covariate, so `g_0* = 1 - g_1*` holds exactly. Both solve the *same two* score
equations. Column 0's is `P_n[Qr_0/g_0* {1(A=a_0) - g_0*}] = 0` once the sign convention in
`reduced_mechanism_covariate` is unwound. They differ only in the submodel, hence at second
order. So the estimator is not continuous in `K`, and a reader comparing a two-arm
cleverly fit against a two-arm `drtmle` fit should expect agreement in the equations solved
rather than in the iterates.

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import ATE, CausalStudy, DRTMLEMethod, PointTreatment
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=1000, seed=0)
study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3", "W4"),
    ),
)
res = study.estimate(
    ATE(),
    method=DRTMLEMethod(guard=("Q", "g")),
    outcome_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    random_state=0,
)
```

`guard=` says which extra equations to solve, in `drtmle`'s vocabulary and crossed the way that
package crosses it. Both apply by default. It also names the corrections the reported curve
subtracts: one per equation solved, so `guard=("g",)` reports `D = D* − D*_Q` and the score
check's verdict names *that* curve. The other equation's correction is still recomputed and
reported, held to no threshold, because it is what says what the guard did not buy. An empty
guard is a plain `TMLE`, bit for bit.

### Supported, with conditions

| keyword | status |
| --- | --- |
| `delta=` | **randomized binary trials only**, using Díaz & van der Laan (2017)'s missing-outcome construction. Set `randomized=True` to estimate the treatment probabilities. Alternatively, pass row-aligned known probabilities as `treatment_probabilities=` to `fit`. Use a mapping keyed by treatment level, `{"placebo": p0, "active": p1}`, or `(n, 2)` in encoded arm order, or `(n,)` read as the probability of the arm whose code is `1`. This surface requires `cross_fit=False`, `repeats=1`, pooled reductions, and no analysis weights or evaluation companion. With `guard=()`, the same array configures a **plain TMLE** at the design mechanism. The result is the ordinary estimator bit for bit, which is what a pure randomization-probability analysis wants. None of those conditions then holds, because no extra equation is being solved and no theorem is claimed. |
| `weights=` | **fixed analysis weights only.** The estimand is the parameter of the tilted law `dP_w = w dP / E[w]`. The derivation was read at an unweighted law; transporting it needs the reduced regressions to be `P_w`-conditional expectations, which weighted loss gives, *and* the mechanism they condition on and divide by to be the `P_w` mechanism, which holds because they are built from `nuisance.propensity`. `tests/unit/test_remainder_drtmle.py` runs the whole expansion at two tilted laws and keeps the wrong transport as a control that fails. |
| `repeats=` | supported; varies exactly one thing, the **primary split**. Each draw fits its own reductions and runs its own alternation; the report is the mean of the draws with the curves averaged elementwise. `result.extra["drtmle"]` describes **draw 0 only**. |
| `reduction="bivariate"` | supported for complete outcomes and discrete treatment. It fits one reduced probability on the two-column `(Qbar-hat(a,W), g-hat(a|W))` design and uses van der Laan's distinct `D_Y`, once per arm as the pinned R implementation does; univariate remains the default because its reduced regressions can converge faster. The cited theorem is binary, so the multi-arm case is an implementation-backed armwise extension rather than a claim about that theorem's literal scope. |
| a random-forest reduction learner | computes, but steps **outside** the cross-fitting argument of [section 3](#reduced-regression-cross-fitting), because its fitted class grows with `n`. Not refused; scoped. |
| `g_bounds=` a fixed value | permitted, and it puts the fit outside the asymptotic half of the [bound-inactive scope](#the-bound-inactive-scope): the argument needs a bound *sequence* going to zero, which `"auto"` supplies and a fixed bound above `ess inf g_0` does not. |

### Refused by name

Each because the derivation read here does not cover it, not because the loop would not run. All
raise at construction or at `fit`, with a message naming what a derivation would need.

| refused | why |
| --- | --- |
| continuous treatment | the reductions and corrections are indexed by treatment mass at a discrete arm; a continuous dose requires density-based equations |
| `reduction="bivariate"` with `delta=` | the supported missing-outcome estimator is Díaz & van der Laan's distinct five-reduction construction, not the complete-outcome one- versus two-dimensional choice. `reduction="univariate"` selects that published missing-data cycle; a bivariate analogue of its five reductions and three correction blocks has not been derived here. |
| `att` / `atc` | a different score equation with no reduced-dimension derivation |
| `interventions=`, `shifts=`, `incremental=`, `msm=` | as above |
| observational treatment with `delta=` | Díaz & van der Laan (2017) derives the construction for randomized trials; the canonical package accepting observational treatment is implementation provenance, not a theorem for that composition |
| missing treatment (`treatment_delta=`) | reserved for a future published construction. Canonical missing-`A` smoke tests do not supply this package's required identification, corrected curve, remainder, and rate conditions |
| `treatment_probabilities=` with `n_bootstrap=`, **whatever `guard=` is** | the array is row-aligned to the data as passed, and a replicate refits on resampled rows it cannot be reindexed to. An n-out-of-n resample passes the length check, so the misalignment would be silent; `randomized=True` estimates the mechanism inside each replicate instead. Unconditional on the guard, because the array is row-aligned however few equations are being solved |
| `treatment_probabilities=` without `delta=` | it replaces the treatment learner outright, and nothing read here states a complete-data construction that reads a known design mechanism differently from a fitted one |
| `intermediate=` | the reduced equations carry no controlled-intermediate factor |
| `targeting_scheme="fold"` | each fold would need its own reduced regressions and alternation |
| `cv_evaluation=True` | the common-update construction would need the corrected parameter and influence curve derived under fold-wise evaluation |
| composition with `CTMLE` | a reduced regression conditions on `ĝ` *as a covariate*, and C-TMLE's `ĝ` is deliberately not an estimate of `g_0`. C-TMLE also scores its path by the loss of the targeted `Q̄`, so the criterion choosing `ĝ` presupposes that `Q̄` is informative. That is precisely the case this variant insures against. |
| estimated weights (`weights_estimated=`) | the ordinary answer is that the interval conditions on the weights, and that answer is an argument about `D*` rather than about `Q_r`, `g_{r,1}` and `g_{r,2}` |
| `evaluation=` with `repeats>1`, `targeting="one_step"`, or `target_weights=True` | each by name; the middle one on cost, up to 20,000 adaptive steps |
| `reduced_crossfit="nested"` with `cross_fit=False` or `n_folds < 3` | there is no complement to leave a fold out of; nested leaves two folds out at a time |

---

## 2. The theorem-backed contract

### The sources

The papers are not kept in the repository, so the table gives section, equation, theorem, or page
locators that resolve independently of a local copy.

| document | supplies |
| --- | --- |
| Díaz & van der Laan (2017), *Doubly robust inference for targeted minimum loss-based estimation in randomized trials with missing outcome data*, Statistics in Medicine 36:3807–3819 | §2.1's observed-data model and EIF; equation (6)'s reduced regressions; Theorems 1–2 and equations (11)–(13)'s correction terms and recursive targeting algorithm. The article explicitly leaves a cross-validated extension to future work. |
| Benkeser & Hejazi (2023), *Doubly-Robust Inference in R using `drtmle`*, Observational Studies 9(2):43–78 | equations (5)–(10) and both reduced-regression constructions; the package workflow; multi-level treatments (§4.6, pp. 66–67); cross-validation (§4.7, p. 69) |
| Benkeser, Carone, van der Laan & Gilbert (2016), U.C. Berkeley Division of Biostatistics Working Paper Series, paper 356 | §3.1's bivariate construction and its `D_A`/`D_Y` displays (p. 9); equation (2) (p. 9); §3.2's univariate `D_Y`, **Theorem 1**, `D^{*,#}`, the variance and the recursive algorithm (pp. 10–11); appendix A's bivariate remainder and rate conditions (pp. 19–20); appendix B's univariate remainder (p. 21); appendix C on unnecessary correction terms (pp. 21–22) |
| Benkeser, Carone, van der Laan & Gilbert (2017), Biometrika 104(4):863–880 | the *published* Theorem 1, authoritative wherever the working paper and it differ; §3.1 also states the earlier bivariate construction and the simulations compare both corrected TMLEs. |
| van der Laan (2014), IJB 10(1):29–57, Theorem 3 and its proof | the bivariate construction's binary parameter, targeted recursion, corrected influence function `D* - D_A - D_Y`, Donsker/mean-square assumptions, product remainders, and union-model limits. |

`benkeser/drtmle` 1.1.2's source is where several formulae here were transcribed from. That is
**provenance, not a target**: comparing against that package's numbers is refused by decision, and
no R enters this repository or its CI. Two names are inverted between the paper and that source.
This package's `ReducedSet.gr1` is R's `grn2`, and `gr2` is R's `grn1`. This is the single
easiest thing here to transcribe backwards.

### The objects

For treatment level `a`, with `Q̄_0(a, w) = E_0(Y | A = a, W = w)`, the target is
`ψ_0(a) = E_0{Q̄_0(a, W)}` and `ATE = ψ_0(1) − ψ_0(0)`. The ordinary efficient influence function
is `D*(Q, g)(O) = A/g(W)·{Y − Q̄(W)} + Q̄(W) − Ψ(Q)`.

The default construction uses three univariate reduced regressions however many covariates the fit
adjusted for:

```text
Q_r    := Q̄_{0,r}(Q̄, g)(w)      = E_0[ Y − Q̄(W) | A = 1, g(W) = g(w) ]
g_{r,1} := g_{1,0,r}(Q̄)(w)       = E_0[ A | Q̄(W) = Q̄(w) ]
g_{r,2} := g_{2,0,r}(Q̄, g)(w)    = E_0[ (A − g(W))/g(W) | Q̄(W) = Q̄(w) ]
```

The two corrections, in the orientation this package computes and the appendices derive:

```text
D_A = D*_g = (Q_r/g)·{A − g}
D_Y = D*_Q = 1_a·(g_{r,2}/g_{r,1})·{Y − Q̄}
```

The bivariate alternative retains `Q_r` and replaces both reduced mechanisms by one probability:

```text
g_r(a|w) = P_0{A=a | Qbar(a,W)=Qbar(a,w), g(a|W)=g(a|w)}
D_Y      = 1_a·{g_r(a|W)-g(a|W)}/[g(a|W)g_r(a|W)]·{Y-Qbar(a,W)}
```

This is exactly the pinned R source's two-column `estimategrn` branch, `fluctuateQ2` clever
covariate, and `eval_Dstar_Q` correction. Those branches loop over every requested level of a
discrete treatment, so cleverly uses the same one-versus-rest probability and correction for each
arm. In the stored state `gr1` is that probability and
`gr2` is `NaN`, because the latter regression does not exist on this path; using zero would make
an accidental univariate formula look valid.

and the limiting influence function

```text
D^{*,#}(Q, g) = D*(Q, g) − I(g = g_0)·D_A − I(Q̄ = Q̄_0)·D_Y
```

The indicators contain the doubly-robust claim. At the true primary nuisance functions, both
corrections vanish and the ordinary efficient influence function is recovered.

The two extra score equations, as the software article states them:

```text
(9)   P_n[ Q_r(a,W)/g*(a|W) · {1_a − g*(a|W)} ]                    = 0
(10)  P_n[ 1_a · g_{r,2}(a|W)/g_{r,1}(a|W) · {Y − Q̄*(a,W)} ]        = 0
```

### Theorem 1

**Suppose** either `Q̄ = Q̄_0` or `g = g_0`, and let the targeted collection
`(Q̄*_n, Q̄*_{n,r}, g*_n, g*_{1,n,r}, g*_{2,n,r})` satisfy the three empirical score conditions

```text
B_n     = P_n D*(Q*_n, g*_n)                            = o_p(n^(−1/2))
B_{A,n} = P_n D_A(Q̄*_{n,r}, g*_n)                       = o_p(n^(−1/2))
B_{Y,n} = P_n D_Y(Q̄*_n, g*_{1,n,r}, g*_{2,n,r})         = o_p(n^(−1/2))
```

**and** suppose the appendix second-order terms satisfy `R_{Q,n} = o_p(n^(−1/2))` and
`R_{g,n} = o_p(n^(−1/2))`. **Then** the targeted plug-in is asymptotically linear with influence
function `D^{*,#}`, and `√n(ψ̂ − ψ_0) ⇝ N(0, σ²)` with

```text
σ̂²_n = P_n [ D*(Q*_n, g*_n) − D_A − D_Y ]²
```

Three things to read off it:

- the score conditions are `o_p(n^(−1/2))`, **not** exact zeros. This implementation's exact-zero
  ambition is stricter than the theorem, and its numerical stopping rule is not obviously either;
- the conditions are on the **targeted** collection, including the starred *reduced* nuisances,
  which is why `retarget` on a `DRTMLE` result costs a fit rather than arithmetic on cached
  arrays. That cost follows the source, and is not a design slip;
- the variance is `P_n[·]²` of the **rowwise** corrected curve, so nothing that summarises the
  curve before squaring is computing this. The ATE curve is the **rowwise difference** of the two
  arm curves, which is what makes an ATE-only diagnostic insufficient: arm-specific errors cancel
  in a difference.

`tests/unit/test_theorem_drtmle.py::TestTheReportedVarianceIsTheorem1s` pins the last property.
The interval built from the package's own corrections is the one Theorem 1's terms give. The
uncentred `P_n{D}²` differs from the reported variance by exactly `(P_n D)²`, and the contrast
reads the covariance rather than the sum of the arms.

### Randomized trials with missing outcomes

For observed data `O=(W,A,Delta,Delta Y)`, write `g_A(a|W)=P(A=a|W)`,
`g_Delta(a,W)=P(Delta=1|A=a,W)`, and `g=g_A g_Delta`. Díaz & van der Laan's
efficient influence function is

```text
D* = 1(A=a, Delta=1)/g · {Y − Qbar(a,W)} + Qbar(a,W) − psi(a).
```

The implementation keeps the five one-dimensional regressions and the three correction
blocks in the paper separate:

```text
gamma_A = P(A=a | Qbar_a)
gamma_Delta = P(Delta=1 | A=a,Qbar_a)
r_A = E[{1(A=a)-g_A}/g_A | Qbar_a]
r_Delta = E[{Delta-g_Delta}/(g_A g_Delta) | A=a,Qbar_a]
e = E[Y-Qbar_a | A=a,Delta=1,g_A g_Delta]

D_A = e/g_A · {1(A=a)-g_A}
D_Delta = 1(A=a)e/(g_A g_Delta) · {Delta-g_Delta}
D_Y = 1(A=a,Delta=1) · {r_A/(gamma_A gamma_Delta)+r_Delta/gamma_Delta}
      · {Y-Qbar_a}
```

The targeting cycle jointly updates the ordinary outcome and `D_Y` covariates, updates
`g_Delta` within each arm, updates the shared binary `g_A` path, refits all five reductions,
and repeats until all four score blocks settle. `correction_check` reports `D_A`, `D_Delta`
and `D_Y` separately; checking only `D_A + D_Delta` would be blind to equal and opposite score
errors. Missing-outcome fits therefore require `guard=("Q", "g")`.

Treatment probabilities and observation probabilities retain their own bounds: `g_bounds`
applies to `g_A` and `gamma_A`, while `nuisance_bound` applies to `g_Delta` and
`gamma_Delta`. Their product is derived for the ordinary outcome clever covariate and the
positivity report, never stored as a third nuisance. The treatment truncation curve moves only
the treatment bound; `mechanism=True` moves only the observation bound.

Because it is derived, the positivity report's `P(A=a,Delta=1|W)` row **has no bound of its
own**, and reading it as though it did is a mistake worth naming: each factor is truncated and
the two are then multiplied, so the product is never compared against `g_bounds[0] ×
nuisance_bound`. That row's `clipped` therefore counts the cells where either factor's
truncation moved the product. Its `ess_ratio` weights by `clip(g)·clip(π)`, so the denominator
the equation forms. Counting against the product of the floors instead reports a strict subset,
since a small factor beside a large one leaves the product above it: measured at **1.1%** against
a true **20.1%** on the pinched fixture in `tests/unit/test_drtmle_missing.py`.

The shipped scope follows the paper rather than the broader canonical package: binary randomized
treatment, MAR and positivity, no cross-fitting, and no weights, repeats, fold targeting, or
evaluation companion. `randomized=True` estimates `g_A`; `treatment_probabilities=` supplies known
row-aligned probabilities and bypasses the treatment learner. Prefer the mapping form
`{"placebo": p0, "active": p1}`: the positional forms bind to arm *codes*, which are indices into
the sorted levels, so a `(n,)` vector is the probability of the second sorted level and not of
"the treated arm". Observational treatment and missing treatment remain refused because this paper
does not derive those compositions. Known probabilities are row-aligned fit data and are retained
with the complete fitted result in the trusted joblib artifact, together with the estimator needed
for supported later refits.

### The sign of the mechanism correction

**Resolved in favour of this implementation, on the working paper's own appendices.** Nothing
that reports only a point estimate could have caught the discrepancy, since all three
empirical means are driven to zero and what a flipped sign moves is the variance.

*The charge.* The §3.1 display defines `D_A := −(Q_r/g)(A − g)`, with a leading minus, and
Theorem 1 reports `D^{*,#} = D* − D_A − D_Y`. Read off those two, the theorem's mechanism
contribution is `+u` where the code's is `−u`.

*What settles it.* First, the same paper prints the object twice with two signs: §3.2 redefines
`D_Y` for the univariate construction, the default here, with **no** leading minus, twelve lines
after printing the bivariate one with one. Two displays of one object with opposite signs is
already a reason not to settle the question from a display.

Second, and decisively, **appendices A and B derive both terms, and each derivation fixes the
orientation.** Each reads `P_0[term] = −(P_n − P_0)·D + B_n + (second order)` with `B_n := P_n·D`,
and `P_0[u] = P_n[u] − (P_n − P_0)[u]` is an identity for any `u` whatever. The decomposition is
therefore satisfiable **only** with `D` equal to the positive term. Appendix A's opening step
says which quantity is being decomposed, and is checkable rather than interpretable:

```text
−P_0{ (Q_r/g_0)·(g_n − g_0) } = P_0{ (Q_r/g_0)·(A − g_n) }      since E_0[A | W] = g_0
```

The right-hand side is positive. `tests/unit/test_theorem_drtmle.py` checks that identity on the
exact law, and checks that the correction's mean is materially nonzero **there**. Were that mean
zero, both readings would agree and the question would be unanswerable. The test then checks the
consequence: the asymptotic-linearity representation closes to `1e-12` with the corrections
**subtracted**, and fails by **exactly twice the correction** when they are added. Watched to fail
against three mutations, one of them the flipped sign in the library itself.

So the leading minus in the §3.1 display is not a rival convention to be matched. It is
inconsistent with the derivation in the same document, with Theorem 1's own variance formula, and
with the exact-law arithmetic here. The published 2017 article confirms the corrected
constructions but does not replace that algebraic sign witness. The code follows the identity,
not a display in either edition.

Two further sign slips in the same document, recorded so a later reader does not re-derive them.
Equation (2) is printed with `+(P_n − P_0)D − B_n` while its appendices derive
`−(P_n − P_0)D + B_n`, which is the same slip under `D → −D`. Equation (2) also **crosses its
second-order labels**: appendix A's block, the `D_A` one, is collected into `R_{Q,n}`, while (2)
pairs `D_A` with `R_{g,n}`. Harmless for the implementation, which reads neither label, and worth
knowing before quoting (2).

### Appendix C: a correction that is not needed costs nothing

The paper's own account of why solving an equation you did not need is asymptotically free. Both
answers rest on a vanishing:

```text
D_A = 0  for every g,   because Q_r    = 0 when Q̄ = Q̄_0
D_Y = 0  for every Q̄,   because g_{r,2} = 0 when g  = g_0
```

Each `B` then decomposes into an empirical-process term plus a second-order one, both
`o_p(n^(−1/2))` under the appendices' rate conditions.

**This is also the source of the blindness that shapes every test on this estimator.** `Q_r` and
`g_{r,2}` are zero *row by row* at correct nuisances, so any check taken at the truth is blind to
a flipped sign, to an update order, and to a reduction vintage alike. It is why
`tests/unit/test_theorem_drtmle.py` and `tests/unit/test_influence_gateaux_drtmle.py` are taken at
values where the corrections do **not** vanish, and why the fixtures they use are misspecified on
purpose.

### The remainder terms, and the rate conditions

The mechanism-misspecification branch, appendix B:

```text
R̃_{5,n} = P_0[ { (A/g_{1,0n,r})·g_{2,0n,r} − (A/g_{1,0,r})·g_{2,0,r} } · (Y − Q̄_n) ]
R̃_{6,n} = P_0[ { (A/g_{1,0,r})·g_{2,0,r}   − (A/g_{1,n,r})·g_{2,n,r} } · (Y − Q̄_n) ]
M̃_{2,n} = (P_n − P_0)[ D_Y(Q̄_n, g_{1,n,r}, g_{2,n,r}) − D_Y(Q̄_0, g_{1,0,r}, g_{2,0,r}) ]
R_{g,n} = R̃_{5,n} + R̃_{6,n} + M̃_{2,n}
```

and the outcome-misspecification branch, appendix A, `R_{Q,n} = R_{3,n} + R_{4,n} + M_{1,n}`:

```text
R_{3,n} = P_0[ { (Q̄_{0n,r} − Q̄_{0,r}) / g_0 } · (g_0 − g_n) ]
R_{4,n} = P_0[ { Q̄_{0,r}/g_0 − Q̄_{n,r}/g_n }  · (g_0 − g_n) ]
M_{1,n} = (P_n − P_0)[ D_A(Q̄_{n,r}, g_n) − D_A(Q̄_{0,r}, g_0) ]
```

with `Q̄_{0n,r}(w) := E_0{Y − Q̄(W) | g_n(W) = g_n(w), g_0(W) = g_0(w)}` evaluated at the
*estimated* propensity as well as the true one, which is what makes `R_{3,n}` an approximation
error rather than a fitted one. Appendix A also carries `R*_n = R_{1,n} + R_{2,n}`, the part that
is second order whichever nuisance is right.

**The paper's rate conditions are illustrative, not necessary.** It states that it *generally
suffices* that, for `R_{g,n}`:

```text
‖Q̄_n − Q̄_0‖_2 = o_p(n^(−1/4))
‖g_{2,0n,r} − g_{2,0,r}‖_2 = o_p(n^(−1/4))
‖g_{2,n,r}  − g_{2,0,r}‖_2 = o_p(n^(−1/4))
```

and for `R_{Q,n}`, in the same "if, for example" form, `o_p(n^(−1/4))` on the reduced regression's
approximation error `‖Q̄_{0n,r} − Q̄_{n,r}‖_2`, on its fitted error `‖Q̄_{n,r} − Q̄_{0,r}‖_2`, and on
the primary propensity's error `‖g_n − g_0‖_2`. Both appendices add an empirical-process pair in
one shape: a `P_0`-Donsker class containing the *estimated* curve, plus `L_2(P_0)` convergence of
the estimated curve to its limit.

**These are the conditions the release claim is conditional on**, and they are conditions on
estimated functions that no fit can check for itself.

### The bound-inactive scope

**Truncation is not in the theorem's algorithm.** The theorem's `D_g` is evaluated at the same
`g*` its score is solved at, and that `g*` is not truncated anywhere: there is one mechanism,
produced by an unconstrained `expit` fluctuation, appearing identically in equation (9)'s
covariate, in equation (9)'s residual and in `D_A`. Boundedness is an *assumption about `g_0`*,
not an operation on `ĝ`. So the theorem supports neither this package's original hybrid (bounded
denominator, raw residual) nor R's post-fit clip.

The package instead uses a **constrained estimating equation**. It puts `clip` inside `F` and
solves that for a root, so the final score *is* the declared estimator's score
(`fluctuation/mechanism.py::solve_bounded_mechanism`). Where nothing clips, that solver returns
the unconstrained solve untouched, so such a fit **is** the estimator Theorem 1 is stated for, not
an approximation to it.

The guarantee is therefore scoped rather than assumed through:

- a fit whose truncations are **inactive** is inside the theorem-backed contract;
- a **bound-active** fit is empirically supported and reported as outside it. It is not a failing
  fit: on `weak_overlap_dgp` every identity holds at `1e-17` and every score is negligible while a
  third of the `(row, arm)` pairs clip at the initial mechanism.

**Three truncations must be inactive, not one**. A randomized missing-outcome fit has five
truncations because it divides by two separately bounded mechanisms. The last truncation has no
corresponding theorem assumption:

| truncation | witness on the fit | in Theorem 1's assumptions |
| --- | --- | --- |
| `ĝ` at the fit | `CorrectionRow.initial_clipped` | an assumption on `g_0`, not an operation on `ĝ` |
| `π̂` at the fit for missing-outcome fits | `CorrectionRow.observation_clipped` | an assumption on `π_0`, on the same footing |
| `g*` at the exit | `CorrectionRow.margin` | the same one |
| `π*` at the exit for missing-outcome fits | `CorrectionRow.observation_margin` | the same one |
| `g_{r,1}` in equation (10)'s covariate, or `γ_a`/`γ_Δ` on a missing-outcome fit | `CorrectionRow.gr1_margin` | **none**. It is a regression of an arm indicator on `Q̄̂`, and `g_0 > δ` does not imply a positive lower bound |

**The two observation rows are not implied by the treatment ones**, which is why they are
separate columns rather than a wider reading of the existing ones. A randomized trial's
treatment mechanism is flat by design and cannot clip, while its observation mechanism is a
fitted probability that can sit at its floor on a large share of rows. A check reading only the
treatment witnesses therefore reports `contract = "theorem"` on a fit that is squarely bound-active.
`tests/unit/test_drtmle_missing.py::TestTheContractSeesTheObservationTruncations` is the pair of
fits that says so: a well-behaved trial reading `"theorem"`, and one whose `π̂` is pinned on a
fifth of its rows reading `"bound-active"` with **only** the two `π` entries named.

The **asymptotic** half of the inactive-bound claim needs three conditions and two are not
Theorem 1's: `g_0 ∈ [δ, 1−δ]`, which is the theorem's; a bound sequence eventually below `δ`,
which `g_bounds="auto"` supplies as `5/(√n·log n) → 0` and a fixed bound above `ess inf g_0` does
not supply at all; and `ĝ` consistent in **sup** norm, which is stronger than the `L_2` conditions
the theorem assumes and is **unverified**.

`CorrectionCheck.contract` reports `"theorem"`, `"bound-active"` or `"none"`. **It is a scope
label and not a verdict**. `CorrectionCheck.passed` deliberately does not read it, because
folding it in would report a perfectly well-solved fit as broken.

---

## 3. Targeting and cross-fitting, as implemented

### The alternation

The paper's recursive algorithm, with `H_1(g) = a/g`, `H_2(g_1,g_2) = a·g_2/g_1` and
`H_3(Q_r,g) = Q_r/g`: initialise; fluctuate `Q̄` along `H_1`; estimate `g_{r,1}` and `g_{r,2}` at
the once-updated outcome regression; fluctuate along `H_2`; estimate `Q_r` at the twice-updated
one; fluctuate `g` along `H_3`; iterate until the three empirical means are approximately zero.
The outcome fluctuations are fitted **using only rows with `A = a`**, which is this package's
per-arm indicator design.

Two things the algorithm has no counterpart for and this implementation adds: a **closing pass** at
frozen final reductions, so the reported scores are the scores of the state returned; and the
truncation of [the previous section](#the-bound-inactive-scope).

The alternation is **not guaranteed to converge.** Equation (10)'s covariate becomes small on
exactly the fits anybody wants, so its inner solve can be singular or stop at working precision.
The archived 96-fit sweep had 87 tolerance exits, 8 stalls, and 1 cap. It had cross-fitting disabled
and therefore did not cover the shipped 10-fold default.

A fixed-seed `glm` check of the current source produced the following default-path evidence. The
counter's historical name is `ill_conditioned`, but it counts every equation-(10) inner failure,
including a tolerance-limited full-rank solve; `res.validate()` therefore describes these as
numerically difficult rather than asserting that every Hessian was singular.

| n | primary folds | exit | numerically difficult rounds | final score check |
| ---: | ---: | --- | ---: | --- |
| 200 | off | cap | 0 of 50 | passed |
| 200 | 10 | stall | 3 of 46 | passed |
| 1000 | off | tolerance | 0 of 44 | passed |
| 1000 | 10 | tolerance | 2 of 16 | passed |
| 3000 | off | cap | 0 of 50 | passed |
| 3000 | 10 | tolerance | 0 of 8 | passed |

The affected 10-fold `n=200` solves had full-rank two-column designs and absolute scores between
`4e-13` and `9e-13`; the statistical score check passed because those residuals were negligible.
No argument here *proves* the iterates approach a common zero of the three equations, which is why
[the diagnostics](#5-diagnostics-to-inspect) surface both convergence and numerical difficulty.

### The update order

`update_order="drtmle"` (default) or `"benkeser"`. **A diagnostic keyword rather than a tuning
one.** `"drtmle"` follows the canonical R package: equation (9), refit only `g_{r,1}` and
`g_{r,2}`, equations (10) and (8), then refit only `Q_r`. `"benkeser"` follows the published
six-step recursion. The working paper's step 7 states its termination as the three empirical means being
approximately zero, so its six-step order is one route to a fixed point rather than something
Theorem 1 assumes about the returned collection. The theorem's hypotheses are conditions on the
returned collection, not on the route. `"benkeser"` implements that order beside R `drtmle`'s,
sharing the stopping rule, stall test and closing pass, deliberately: what is in question is the
route, and a comparison in which two things differ answers nothing.

Two cautions apply. Compare the **scores and the estimates**, not the fluctuation coefficients. The
submodels a round passes through differ, so an `epsilon` from one is not an `epsilon` from the
other. And compare at the **same nuisances**: same data, same `random_state`.

On the compared draws, the routes agree on `ψ` but not on `σ²_n`. At `n = 600`, the `ate`
estimates differ by `9e-03` of a standard error while the standard errors differ by 2.3%, with
both fits solving all three equations. That is not a contradiction of step 7 and is what step 7
does not say: the exit condition constrains the three empirical *means*, while the reported
variance is the second moment of a curve built from the reductions, which the two routes leave at
different vintages by construction.

### The canonical `cvFolds` mapping

The 2023 article calls `cvFolds` cross-validated DR-TMLE, but the name does not select either of
the additional CV estimators exposed by this package. Reading the pinned R source settles the
implementation map:

1. `estimateQ_loop`, `estimateG_loop`, `estimateQrn_loop`, and `estimategrn_loop` fit on each
   training-fold complement and reorder their validation predictions into row order;
2. `fluctuateG`, `fluctuateQ1`, and `fluctuateQ2` then receive those assembled arrays once and fit
   one global alternation, not one alternation per validation fold;
3. the reported arm mean is `mean(QnStar[[a]])`, and covariance is `cov(DnoStarMat) / n` from the
   rowwise corrected curve.

In cleverly's vocabulary that is `cross_fit=True`, `reduced_crossfit="pooled"`,
`targeting_scheme="pooled"`, and `cv_evaluation=False`. The first two settings produce the
out-of-fold arrays; the last two keep the single alternation, whole-sample plug-in, and ordinary
corrected-curve covariance. `tests/unit/test_drtmle_crossfit.py` pins all four pieces, including an
unequal-fold witness on which replacing the plug-in or variance by an equal-fold construction is
nonzero.

This is **implementation provenance, not the missing theorem**. Benkeser & Hejazi §4.7 states
that cross-validated nuisance estimates weaken entropy conditions and the canonical source shows
the intended computation, but the published Theorem 1 does not derive a cross-fitted expansion.
The package's argument for the generated reduced regressions is the next section. Nothing in the
source derives a corrected parameter or influence curve for `targeting_scheme="fold"` or
`cv_evaluation=True`, so both remain refused rather than inheriting ordinary TMLE's modes by name.

### Reduced-regression cross-fitting

`reduced_crossfit="pooled"` (default) reuses the primary split as it stands; `"nested"` takes
fold `k`'s training designs *and targets* from primary models fitted with fold `k` left out as
well. **A diagnostic keyword rather than a tuning one**, for a specific reason: the argument for
the cheap construction needs one quantity to vanish, and that quantity *is* the difference between
these two.

**Cross-fitting is not in the theorem.** The canonical mapping above is source evidence for the
implemented computation, and the 2023 article's statement that cross-validated nuisance estimates
weaken the entropy conditions is supporting evidence that cross-validated DR-TMLE is intended;
neither is the missing proof.
What is specifically unaddressed is the **pooled** construction, in which an observation influences
other rows' generated regressors and then returns to its own reduced-regression prediction through
those rows. Generic cross-fitting results do not cover that because the conditional independence
they turn on is exactly what the generated design breaks.

The argument supplied here. Writing `ĥ_k` for what fold `k`'s reduced regression contributes and
`h̃_k` for the nested version:

```text
(P_{n,k} − P_0) ĥ_k = (P_{n,k} − P_0) h̃_k             [A]
                    + (P_{n,k} − P_0) Δ_k ∘ ĝ^(−k)     [B],   Δ_k = Q̂_r^(−k) − Q̃_r^(−k)
```

**[A]** is the ordinary cross-fitting argument: conditional on the fold-`k` complement both maps
are fixed, so the term has conditional mean zero and is `o_p(n^(−1/2))` under the `L_2` convergence
Theorem 1 already assumes. **[B]** is the whole of the open question, because `Δ_k` is precisely
the object that depends on fold `k`; no cross-fitting lemma reaches it. What does is asymptotic
equicontinuity, needing two conditions:

> **(E)** the reduction learner's fitted functions of one or two scalars lie, with probability tending to
> one, in a class whose bracketing entropy is bounded **uniformly in the underlying measure**.

The structural fact that makes (E) available is that **the reductions have fixed dimension**:
one scalar for the default construction and two for the bivariate reduced probability, however
many covariates the fit adjusted for. Composition with a fixed map transports brackets exactly,
so the entropy requirement falls entirely on a class of functions of at most two variables and
not at all on the primary nuisances' complexity. The measure-free phrasing is load-bearing, since
the pushforward is a *random* measure and a bound holding at `P_0` says nothing about it. A
fixed-basis linear or logistic model is a bounded fixed-dimensional sieve; histogram
boosting is a fixed bounded-variation ball when its iteration and leaf limits are fixed. A
CV-selected round count would take boosting out of the class. A random forest is outside.

> **(S)** `‖Δ_k‖_{L_2} = o_p(1)`: the reduction fit is `L_2`-continuous in the design and target
> columns it is handed.

**(S) is the open condition.** It is free for a fixed-basis linear smoother and not free for
anything that *selects* structure from the data: a split point, a bandwidth, a CV-chosen
candidate. An arbitrarily small design perturbation can move any of those selections discretely.
Boosting is entropy-safe and design-continuity-unsafe. `reduced_crossfit="nested"` is what computes
`Δ_k`; a measured dispatch put its consequence on `ψ` at or below what a redrawn fold split moves
in every cell, which is **supported, not shown**, since a consequence can hold by cancellation.

Two things not to read into "nested". It costs `K` times the primary nuisance fitting, and was
measured at **1.3x to 17x** a pooled fit's wall clock over four draws, reaching the outer cap on
two. What dominates that cost is not the extra fitting: nested reductions are noisier, so equation
(10)'s near-singular solve takes more rounds. And **neither construction makes the targeted
collection fold-independent**: `epsilon` is solved on all `n` rows, since
`targeting_scheme="fold"` is refused, so a nested fit is *nested in the nuisance models and pooled
in the tilt*.

### Where the truncations are

On the univariate construction, `g_{r,2}`'s bound is fixed at **fit** time, because the array *is*
a regression of a quotient by the mechanism. It is the only bound in the package chosen at fit
time rather than at targeting time. Two consequences a reader will otherwise trip on.
`DiagnosticsFacade.truncation_curve` moves the clever covariate's denominator and does **not** move
these arrays, so that part of the curve is flat *by construction*; `ReducedSet.g_bounds` is on
record so a reader of such a curve can find out the sweep never reached them. And `gr1` is stored
**untruncated** and bounded at read time through `ReducedSet.bounded_gr1`, column by column and
not complemented across arms.

One further condition sits beside (E) and is a rate rather than an entropy bound. `g_{r,2}`'s
target is `(1_a − ĝ)/ĝ` at the bounded mechanism, so its envelope is `1/lo − 1`, and equation
(10)'s covariate divides by `g_{r,1}` truncated at the same bound, so that envelope is `O(1/lo²)`.
Under `g_bounds="auto"`, `lo → 0` and the envelope **grows with `n`**. That pulls against the
bound-sequence row of [the scope section](#the-bound-inactive-scope), where exactly that shrinkage
is what makes the truncation asymptotically inactive. Both are open.

---

## 4. What the nuisances must satisfy

The union condition, `Q̄ = Q̄_0` **or** `g = g_0`, is assumed and not checked. Beyond it, the
release claim rests on the rate conditions of
[the remainder section](#the-remainder-terms-and-the-rate-conditions), which fall on **five**
estimated functions rather than two for the univariate construction:

1. the primary outcome regression `Q̄_n`;
2. the primary propensity `g_n`;
3. the reduced outcome regression `Q_r`;
4. the reduced propensity `g_{r,1}`;
5. the reduced propensity `g_{r,2}`.

The bivariate construction instead has four: the same two primary nuisances, `Q_r`, and its
single two-column reduced probability `g_r`.

The reduced regressions are the part it is easy to forget, and they are where the guarantee is
bought. Their consistency is **estimated, unmeasured**: a saturated learner recovers them exactly
on the exact law, which is consistency at one learner on one law and not a rate. A study over
6,000 fits found the interval demonstrably better than a plain TMLE's where one nuisance is badly
estimated: `0.844`/`0.848` against `0.532`/`0.472` in the cell built for it. It was **nominal
nowhere**, the best reading being `0.880`, with the three reductions fitted by `glm`. That is a
measurement of a configuration, not of the theorem's condition.

Practical consequences:

- **Choose the reduction learners deliberately.** They default to the primary specification.
  `reduced_outcome_learner=` and `reduced_treatment_learner=` are two keywords rather than one
  because the tasks differ: `g_{r,1}` (or bivariate `g_r`) is a conditional probability and
  `Q_r` plus univariate `g_{r,2}` are conditional means of signed quantities. A learner
  *instance* built for classification cannot serve `Q_r`, whose target is an outcome residual.
- **A flexible primary nuisance does not buy a flexible reduction.** The reductions are on one
  scalar by default and two for bivariate `g_r`, so their bias is a low-dimensional smoothing
  question independent of how well the primary fit adjusted for `W`.
- **A random-forest reduction is outside the entropy condition** of
  [section 3](#reduced-regression-cross-fitting), and is not warned about at runtime. The automatic
  Super Learner includes a forest, so choose explicit reduced learners when claiming that condition.

---

## 5. Diagnostics to inspect

In cost order. The first two are free.

| call | what it answers |
| --- | --- |
| `res.score_verdict` | the score check's verdict, carried whether it passed or not. `summary()` prints it whenever it **fails**; a passing fit says nothing extra. Derived from the fluctuations rather than stored, so a reloaded fit recomputes it. |
| `res.diagnostics.score_equations()` | the same score object, asked for directly |
| `res.validate()` | the default assessment; reports `warning` when equation (10) had numerically difficult inner solves even if the returned score equations pass, with the affected round count and fraction |
| `res.diagnostics.corrections()` | the low-level doubly-robust rows: per arm, per equation. Empty unless the fit is a guarded `DRTMLE`. |
| `res.diagnostics.nuisance_models()` | the primary fits' held-out risk and diagnostics |
| `res.diagnostics.refute()` | negative controls; costs refits |

`correction_check()` recomputes each arm's `P_n[w D*_g]` and `P_n[w D*_Q]` **from the exact
returned state** and reports the residual against the score the targeting step recorded. Five
conditions govern how it does so, each ruling out a way of passing for the wrong reason: per arm
and never only on the ATE, since arm-specific errors cancel in a difference; **before** the
contrast is constructed; with the row weights included; on **one outcome scale**; and, in the
tests, on a fixture where the truncation binds. That one scale is the outcome's own, so that a
correction score and `se/√n` are comparable numbers rather than two quantities a factor of
`range` apart.

**Two failures, and they are not the same failure.**

- *An identity residual*, meaning `Δ_g` or `Δ_Q` above `IDENTITY_TOLERANCE = 1e-12`, is a
  **software defect**. The fit solved one expression and reported another, and no amount of
  further iteration would fix it, because the loop is not posing the equation the curve needs.
  The tolerance sits seven orders above the arithmetic and four below the smallest observed real
  failure.
- *A correction score* above the inferential tolerance is a **fit that did not solve its
  equations**. That is the ordinary failure, reported per arm and per equation so a reader can
  see which.

*And a row that is neither.* A fit guarding one nuisance solves one of the two extra equations, so
its curve subtracts one correction; the other term is still reported, marked `solved=False`, as the
diagnostic saying what is **not** in this curve. Such a row is not a failure and cannot be one:
nothing subtracts it.

Read `CorrectionCheck.contract` alongside `passed`, never folded into it. `passed` answers *did
this fit solve what it reports*; `contract` answers *which estimator the numbers are evidence
about*.

---

## 6. Solved scores do not establish nuisance consistency

This is the one thing to take away from the page.

The score equations being solved is a statement about **numerical targeting**. It is not a
statement about whether the method-specific functions of [section 4](#4-what-the-nuisances-must-satisfy) are
adequately estimated, and the two are independent in a way that is easy to get backwards, because
a fit with badly wrong reductions returns a `psi`, an `se` and a confidence interval formatted
exactly like a good one, with every score green.

**The evidence is `tests/unit/test_oracle_reductions.py`**, and it is worth stating as a result
rather than a caveat. On an exact law, with **exact** reduced regressions handed to a real
alternation, the estimator recovers the truth *despite misspecified primary nuisances*, which is
the whole point of the variant. With **wrong** reductions, the estimate moves, and **every score
equation still passes**. Nothing on the face of such a fit distinguishes it from the good one.

Three consequences for practice:

1. **The score check is necessary, not sufficient.** Treat a failing score check as
   disqualifying and a passing one as saying nothing about the nuisances.
2. **Inspect the reduced-regression fits themselves**, not just the equations built from them.
   Their diagnostics are on `result.extra["drtmle"].diagnostics`, keyed `"qr"`, `"gr1"`, `"gr2"`
   on the univariate reduction, `"qr"`, `"gr1"` on the bivariate reduction, and `"gamma_a"`,
   `"gamma_m"`, `"r_a"`, `"r_m"`, `"e"` on the missing-outcome one. The constructions do not
   fit the same regressions, so they cannot report under the same names.
   `result.extra["drtmle"].reduction` says which ran, read off the
   set that was fitted rather than off the `reduction=` setting, and `.missingness_bound` records
   the bound the two observation reductions were formed at.
3. **Where you cannot argue the rate conditions, do not treat the interval as settled.** Use this
   estimator where you have a reason to think one primary nuisance is badly estimated; that is the
   regime it was derived for and the regime the evidence covers.

The same distinction, once more, in the theorem's own terms: Theorem 1 licenses an interval
*conditional on* the three empirical scores being `o_p(n^(−1/2))` **and** the two second-order
remainder terms being `o_p(n^(−1/2))`. A fit can report on the first. Nothing reports on the
second.

---

## What the validation programme established

The active registered [canonical DR-TMLE study](technical-reference/method-evidence.md#canonical-dr-tmle)
now adds a theory-first comparison with pinned R `drtmle` on the paper's binary complete-data law.
Most paired cells establish bounded equivalence, but none establishes the prespecified
coverage-superiority route; one paired cell is inconclusive. The both-correct calibration and
root-n cells pass, while both one-correct robustness cells fail their finite-sample bias rule.
Those red results are committed evidence, not exceptions to the theorem: the theorem remains
conditional on rate and remainder premises that a fitted dataset cannot certify.

The older drift-law programme below asks a different, harder question. It is retained as
historical evidence rather than silently generalized to the registered paper law.

This page's claims rest on a closed programme of six pieces: a theoretical audit against the
sources, a targeting-and-exit study, a controlled coverage demonstration, a reference study for
the reduced regressions, a construction ablation, and a terminal experiment. The
[evidence index](technical-reference/evidence.md) records the acceptance instruments. The
programme itself, including its study harnesses, replicate records, differential diagnostics,
dispatch workflows, and working notes, is archived at the `drtmle-validation-archive-2026-08` tag
rather than on `main`. In summary:

**Established.** The implementation is faithful to Theorem 1: the corrected curve is the Gateaux
derivative of the parameter, the sign of the mechanism correction is the appendices' orientation,
the reported variance is Theorem 1's, the three score equations are solved at the state returned,
and the interval is materially better than a plain TMLE's where one nuisance is badly estimated.

**Not established in that archived drift-law programme, and recorded as such.** Three things.
Nominal coverage anywhere in that study,
the best reading being `0.880`. A localized cause for that shortfall: a six-contrast construction
ablation over 2,496 fits returned a **null** on its primary column, and a terminal experiment over
both a selection and an independent audit cohort nominated **nothing**. And any `src/` change
justified against the theorem.

Two measured quantities account for the shortfall, and they are one premise measured twice: the
second-order remainder Theorem 1 assumes negligible does not vanish at these sizes, and the
reported `se` runs about 10% short of the spread it covers in one drift cell and about 16% *long*
in the other. The second is therefore not a separate defect in the variance estimator. `σ̂²_n` is
Theorem 1's own quantity, valid to first order exactly when the condition the first quantity fails
holds.

That is why the release claim is conditional validity, and why the conditions are stated on this
page rather than assumed away.

## References

- Benkeser, D. & Hejazi, N. (2023). *Doubly-Robust Inference in R using `drtmle`*. Observational
  Studies 9(2):43–78.
- Benkeser, D., Carone, M., van der Laan, M. J. & Gilbert, P. B. (2016). *Doubly-robust
  Nonparametric Inference on the Average Treatment Effect*. U.C. Berkeley Division of Biostatistics
  Working Paper Series, paper 356.
- Benkeser, D., Carone, M., van der Laan, M. J. & Gilbert, P. B. (2017). *Doubly robust
  nonparametric inference on the average treatment effect*. Biometrika 104(4):863–880.
- van der Laan, M. J. (2014). *Targeted estimation of nuisance parameters to obtain valid
  statistical inference*. International Journal of Biostatistics 10(1):29–57.

The full citation list is in [the references page](references.md).
