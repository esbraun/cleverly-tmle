# DR-TMLE: doubly-robust inference

`DRTMLE` is a TMLE whose *interval* stays valid when only one of the two primary nuisances is
consistently estimated. Every interval this package reports is valid when the second-order
remainder is negligible, and for a plain TMLE that remainder is the product
`‖ĝ − g_0‖ · ‖Q̄̂ − Q̄_0‖`. A product goes to zero when one factor does, which is why the *point
estimate* is doubly robust; the interval needs `√n R_2 → 0`, and with one factor not shrinking
`R_2` is first order in the other. So **`TMLE` is doubly robust for consistency and singly robust
for inference**, and this estimator closes the second gap by solving two further score equations
built from reduced-dimension regressions of each nuisance's residual on the other.

This document is the reference for what is supported, what the theorem covers, what the
implementation chooses where the theorem is silent, and what a caller has to check. The runnable
recipe is in [the user guide](user-guide.md#doubly-robust-inference); the derivation of what the
extra equations remove is in
[the methodology page](methodology.md#doubly-robust-inference-what-the-extra-equations-remove).

## The release claim, in one paragraph

**Conditional validity.** The algorithm computes what Benkeser, Carone, van der Laan & Gilbert's
Theorem 1 derives — checked against the theorem's own appendices, against the Gateaux derivative
of the parameter, against exact finite-support laws, and against the remainder identities — and
the interval it reports is valid **conditional on** the practitioner obtaining adequate primary
and reduced-regression fits. Those are rate conditions on estimated functions. They are not
verifiable from a fit's own output, and in particular **numerical score convergence does not
verify them**; see [section 6](#6-solved-scores-do-not-establish-nuisance-consistency), which is
the single most important thing on this page.

What this is *not*: a better point estimate. The three empirical means are all driven to zero, so
the extra terms cannot move `Ψ̂` and only move its variance. Read a `DRTMLE` fit as the same
estimate with an interval entitled to be believed under weaker conditions.

And it is *not* the efficient estimator. Under misspecification the canonical gradient at `P_0` is
still `D*`. What the three equations leave is `D = D* − D*_Q − D*_g`, the estimator's asymptotic
influence function at the nuisance limits, and it is generally not efficient there. When both
nuisances are consistent the corrections vanish row by row and this is the ordinary efficient
estimator — which is exactly the case the variant is not for.

---

## 1. Supported estimands

A **binary point treatment** and the `mean` group: `ey1`, `ey0` and `ate`, reported under those
names. Scope is what the sources *derive*, which is narrower than what the R package `drtmle`
accepts.

```python
from cleverly import DRTMLE
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=1000, seed=0)

res = (
    DRTMLE(
        estimands=("ate",),
        guard=("Q", "g"),
        outcome_learner="glm",
        treatment_learner="glm",
        random_state=0,
    )
    .fit(frame, outcome="Y", treatment="A")
    .single()
)
```

`guard=` says which extra equations to solve, in `drtmle`'s vocabulary and crossed the way that
package crosses it. Both by default. It also says which corrections the reported curve subtracts —
one per equation solved, so `guard=("g",)` reports `D = D* − D*_Q` and the score check's verdict
names *that* curve. The other equation's correction is still recomputed and reported, held to no
threshold, because it is what says what the guard did not buy. An empty guard is a plain `TMLE`,
bit for bit.

### Supported, with conditions

| keyword | status |
| --- | --- |
| `weights=` | **fixed analysis weights only.** The estimand is the parameter of the tilted law `dP_w = w dP / E[w]`. The derivation was read at an unweighted law; transporting it needs the reduced regressions to be `P_w`-conditional expectations, which weighted loss gives, *and* the mechanism they condition on and divide by to be the `P_w` mechanism, which holds because they are built from `nuisance.propensity`. `tests/unit/test_remainder_drtmle.py` runs the whole expansion at two tilted laws and keeps the wrong transport as a control that fails. |
| `repeats=` | supported; varies exactly one thing, the **primary split**. Each draw fits its own reductions and runs its own alternation; the report is the mean of the draws with the curves averaged elementwise. `result.extra["drtmle"]` describes **draw 0 only**. |
| `library="rich"` | computes, but steps **outside** the cross-fitting argument of [section 3](#reduced-regression-cross-fitting) via `forest`, whose fitted class grows with `n`. Not refused; scoped. |
| `g_bounds=` a fixed value | permitted, and it puts the fit outside the asymptotic half of the [bound-inactive scope](#the-bound-inactive-scope): the argument needs a bound *sequence* going to zero, which `"auto"` supplies and a fixed bound above `ess inf g_0` does not. |

### Refused by name

Each because the derivation read here does not cover it, not because the loop would not run. All
raise at construction or at `fit`, with a message naming what a derivation would need.

| refused | why |
| --- | --- |
| multi-valued treatment (`n_arms != 2`) | no multi-arm theorem is reproduced anywhere in hand, and the *targeted* mechanism's simplex compatibility is an open question rather than a known defect. A new multi-arm implementation should prefer a simplex-preserving multinomial fluctuation unless a theorem licenses independent armwise updates. |
| continuous treatment | as above |
| `reduction="bivariate"` | van der Laan (2014)'s single bivariate reduced mechanism. The equations are reproduced in Benkeser & Hejazi (2023); the **theorem** requires van der Laan (2014) Theorem 3, which is not in hand. Missing with it: the formal statement, its assumptions, the asymptotic expansion, its influence function, its remainder decomposition, and any cross-fitted version. |
| `att` / `atc` | a different score equation with no reduced-dimension derivation |
| `interventions=`, `shifts=`, `incremental=`, `msm=` | as above |
| `delta=`, `intermediate=` | the equations carry no missingness or intermediate factor. There is also an unsettled derivation question behind `delta=`: R's `eval_Dstar_g` applies the missingness indicator to `D*_g` and `reduced_corrections` does not. It is not live — no fit either package accepts has a missing outcome — and it must be settled **from the derivation** before the refusal is lifted. |
| `targeting_scheme="fold"` | each fold would need its own reduced regressions and alternation |
| `cv_evaluation=True` | the common-update construction would need the corrected parameter and influence curve derived under fold-wise evaluation |
| composition with `CTMLE` | a reduced regression conditions on `ĝ` *as a covariate*, and C-TMLE's `ĝ` is deliberately not an estimate of `g_0`; and C-TMLE scores its path by the loss of the targeted `Q̄`, so the criterion choosing `ĝ` presupposes `Q̄` is informative — precisely the case this variant insures against. |
| estimated weights (`weights_estimated=`) | the ordinary answer — that the interval conditions on the weights — is an argument about `D*` rather than about `Q_r`, `g_{r,1}` and `g_{r,2}` |
| `evaluation=` with `repeats>1`, `targeting="one_step"`, or `target_weights=True` | each by name; the middle one on cost, up to 20,000 adaptive steps |
| `reduced_crossfit="nested"` with `cross_fit=False` or `n_folds < 3` | there is no complement to leave a fold out of; nested leaves two folds out at a time |

---

## 2. The theorem-backed contract

### The sources

Neither paper is kept in the repository, so every locator below carries a page number: a path
resolves for a reader who already has the file, and a page number resolves for one who does not.

| document | supplies |
| --- | --- |
| Benkeser & Hejazi (2023), *Doubly-Robust Inference in R using `drtmle`*, Observational Studies 9(2):43–78 | equations (5)–(10) and both reduced-regression constructions; the package workflow; multi-level treatments (§4.6, pp. 66–67); cross-validation (§4.7, p. 69) |
| Benkeser, Carone, van der Laan & Gilbert (2016), U.C. Berkeley Division of Biostatistics Working Paper Series, paper 356 | §3.1's bivariate construction and its `D_A`/`D_Y` displays (p. 9); equation (2) (p. 9); §3.2's univariate `D_Y`, **Theorem 1**, `D^{*,#}`, the variance and the recursive algorithm (pp. 10–11); appendix A's bivariate remainder and rate conditions (pp. 19–20); appendix B's univariate remainder (p. 21); appendix C on unnecessary correction terms (pp. 21–22) |
| Benkeser, Carone, van der Laan & Gilbert (2017), Biometrika 104(4):863–880 | the *published* Theorem 1, authoritative wherever the working paper and it differ. **Unread**, and it gates nothing — see [the sign section](#the-sign-of-the-mechanism-correction). |
| van der Laan (2014), IJB 10(1):29–57, Theorem 3 | the bivariate construction's regularity conditions. **Not in hand**, which is why `reduction="bivariate"` is refused. |

`benkeser/drtmle` 1.1.2's source is where several formulae here were transcribed from. That is
**provenance, not a target**: comparing against that package's numbers is refused by decision, and
no R enters this repository or its CI. Two names are inverted between the paper and that source —
this package's `ReducedSet.gr1` is R's `grn2` and `gr2` is R's `grn1` — which is the single
easiest thing here to transcribe backwards.

### The objects

For treatment level `a`, with `Q̄_0(a, w) = E_0(Y | A = a, W = w)`, the target is
`ψ_0(a) = E_0{Q̄_0(a, W)}` and `ATE = ψ_0(1) − ψ_0(0)`. The ordinary efficient influence function
is `D*(Q, g)(O) = A/g(W)·{Y − Q̄(W)} + Q̄(W) − Ψ(Q)`.

The three reduced regressions — univariate however many covariates the fit adjusted for, which is
what lets them be estimated fast enough whether or not the primary nuisances can:

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

and the limiting influence function

```text
D^{*,#}(Q, g) = D*(Q, g) − I(g = g_0)·D_A − I(Q̄ = Q̄_0)·D_Y
```

The indicators are the doubly-robust content in one line: when **both** primary nuisances are
correct, both corrections vanish and the ordinary efficient influence function is recovered.

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
  arrays — a cost of following the source rather than a design slip;
- the variance is `P_n[·]²` of the **rowwise** corrected curve, so nothing that summarises the
  curve before squaring is computing this. The ATE curve is the **rowwise difference** of the two
  arm curves, which is what makes an ATE-only diagnostic insufficient: arm-specific errors cancel
  in a difference.

`tests/unit/test_theorem_drtmle.py::TestTheReportedVarianceIsTheorem1s` pins the last of those —
the interval built from the package's own corrections is the one Theorem 1's terms give, the
uncentred `P_n{D}²` differs from the reported variance by exactly `(P_n D)²`, and the contrast
reads the covariance rather than the sum of the arms.

### The sign of the mechanism correction

**Resolved in favour of this implementation, on the working paper's own appendices.** Nothing
that reports only a point estimate could have caught the discrepancy, since all three
empirical means are driven to zero and what a flipped sign moves is the variance.

*The charge.* The §3.1 display defines `D_A := −(Q_r/g)(A − g)`, with a leading minus, and
Theorem 1 reports `D^{*,#} = D* − D_A − D_Y`. Read off those two, the theorem's mechanism
contribution is `+u` where the code's is `−u`.

*What settles it.* First, the same paper prints the object twice with two signs: §3.2 redefines
`D_Y` for the univariate construction — the one implemented here — with **no** leading minus,
twelve lines after printing the bivariate one with one. Two displays of one object with opposite
signs is already a reason not to settle the question from a display.

Second, and decisively, **appendices A and B derive both terms, and each derivation fixes the
orientation.** Each reads `P_0[term] = −(P_n − P_0)·D + B_n + (second order)` with `B_n := P_n·D`,
and `P_0[u] = P_n[u] − (P_n − P_0)[u]` is an identity for any `u` whatever — so the decomposition
is satisfiable **only** with `D` equal to the positive term. Appendix A's opening step says which
quantity is being decomposed, and is checkable rather than interpretable:

```text
−P_0{ (Q_r/g_0)·(g_n − g_0) } = P_0{ (Q_r/g_0)·(A − g_n) }      since E_0[A | W] = g_0
```

The right-hand side is positive. `tests/unit/test_theorem_drtmle.py` checks that identity on the
exact law, checks that the correction's mean is materially non-zero **there** — without which both
readings agree and the question is unanswerable — and checks the consequence: the
asymptotic-linearity representation closes to `1e-12` with the corrections **subtracted** and fails
by **exactly twice the correction** when they are added. Watched to fail against three mutations,
one of them the flipped sign in the library itself.

So the leading minus in the §3.1 display is not a rival convention to be matched. It is
inconsistent with the derivation in the same document, with Theorem 1's own variance formula, and
with the exact-law arithmetic here. The published 2017 article is unread and gates nothing: if it
prints the same display, the display is wrong there too; if it prints the positive form, the
working paper's display was a typo. Either way the code does not move.

Two further sign slips in the same document, recorded so a later reader does not re-derive them.
Equation (2) is printed with `+(P_n − P_0)D − B_n` while its appendices derive
`−(P_n − P_0)D + B_n` — the same slip under `D → −D`. And equation (2) **crosses its second-order
labels**: appendix A's block, the `D_A` one, is collected into `R_{Q,n}`, while (2) pairs `D_A`
with `R_{g,n}`. Harmless for the implementation, which reads neither label, and worth knowing
before quoting (2).

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

with `Q̄_{0n,r}(w) := E_0{Y − Q̄(W) | g_n(W) = g_n(w), g_0(W) = g_0(w)}` — evaluated at the
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

What this package does instead is a **constrained estimating equation** — `clip` inside `F`,
solved for a root — so the final score *is* the declared estimator's score
(`fluctuation/mechanism.py::solve_bounded_mechanism`). Where nothing clips, that solver returns
the unconstrained solve untouched, so such a fit **is** the estimator Theorem 1 is stated for, not
an approximation to it.

The guarantee is therefore scoped rather than assumed through:

- a fit whose truncations are **inactive** is inside the theorem-backed contract;
- a **bound-active** fit is empirically supported and reported as outside it. It is not a failing
  fit: on `weak_overlap_dgp` every identity holds at `1e-17` and every score is negligible while a
  third of the `(row, arm)` pairs clip at the initial mechanism.

**Three truncations have to be inactive, not one**, and the third has no assumption in the theorem
to lean on:

| truncation | witness on the fit | in Theorem 1's assumptions |
| --- | --- | --- |
| `ĝ` at the fit | `CorrectionRow.initial_clipped` | an assumption on `g_0`, not an operation on `ĝ` |
| `g*` at the exit | `CorrectionRow.margin` | the same one |
| `g_{r,1}` in equation (10)'s covariate | `CorrectionRow.gr1_margin` | **none** — it is a regression of an arm indicator on `Q̄̂`, and `g_0 > δ` does not imply it is bounded away from zero |

The **asymptotic** half of the inactive-bound claim needs three conditions and two are not
Theorem 1's: `g_0 ∈ [δ, 1−δ]`, which is the theorem's; a bound sequence eventually below `δ`,
which `g_bounds="auto"` supplies as `5/(√n·log n) → 0` and a fixed bound above `ess inf g_0` does
not supply at all; and `ĝ` consistent in **sup** norm, which is stronger than the `L_2` conditions
the theorem assumes and is **unverified**.

`CorrectionCheck.contract` reports `"theorem"`, `"bound-active"` or `"none"`. **It is a scope
label and not a verdict** — `CorrectionCheck.passed` deliberately does not read it, because
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

The alternation is **not guaranteed to converge.** Equation (10)'s covariate is near-singular on
exactly the fits anybody wants, so a draw can exit at the outer cap and report
`failure = "max_iter_reached"`. Over a 96-fit sweep, 87 reach the tolerance, 8 stall and 1 runs out
of rounds. No argument here *proves* the iterates approach a common zero of the three equations,
which is why [the diagnostics](#5-diagnostics-to-inspect) decide rather than the argument.

### The update order

`update_order="cleverly"` (default) or `"paper"`. **A diagnostic keyword rather than a tuning
one.** The working paper's step 7 states its own termination as the three empirical means being
approximately zero, so its six-step order is one route to a fixed point rather than something
Theorem 1 assumes about the collection returned — the theorem's hypotheses are conditions on the
returned collection, not on the route. `"paper"` implements that order beside this package's,
sharing the stopping rule, stall test and closing pass, deliberately: what is in question is the
route, and a comparison in which two things differ answers nothing.

Two cautions. Compare the **scores and the estimates**, never the fluctuation coefficients — the
submodels a round passes through differ, so an `epsilon` from one is not an `epsilon` from the
other. And compare at the **same nuisances**: same data, same `random_state`.

On the draws compared, the routes agree on `ψ` and not quite on `σ²_n` — at `n = 600` the `ate`
estimates differ by `9e-03` of a standard error while the standard errors differ by 2.3%, with
both fits solving all three equations. That is not a contradiction of step 7 and is what step 7
does not say: the exit condition constrains the three empirical *means*, while the reported
variance is the second moment of a curve built from the reductions, which the two routes leave at
different vintages by construction.

### Reduced-regression cross-fitting

`reduced_crossfit="pooled"` (default) reuses the primary split as it stands; `"nested"` takes
fold `k`'s training designs *and targets* from primary models fitted with fold `k` left out as
well. **A diagnostic keyword rather than a tuning one**, for a specific reason: the argument for
the cheap construction needs one quantity to vanish, and that quantity *is* the difference between
these two.

**Cross-fitting is not in the sources.** The theorem presents no cross-fitted version. The 2023
article's statement that cross-validated nuisance estimates weaken the entropy conditions is
supporting evidence that cross-validated DR-TMLE is intended, and it is not the missing proof.
What is specifically unaddressed is the **pooled** construction, in which an observation influences
other rows' generated regressors and then returns to its own reduced-regression prediction through
those rows — generic cross-fitting results do not cover that, because the conditional independence
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

> **(E)** the reduction learner's fitted functions of one scalar lie, with probability tending to
> one, in a class whose bracketing entropy is bounded **uniformly in the underlying measure**.

The structural fact that makes (E) available is that **the reductions are univariate**: the
regression is on one scalar however many covariates the fit adjusted for, and composition with a
fixed map transports brackets exactly, so the entropy requirement falls entirely on a class of
functions of one variable and not at all on the primary nuisances' complexity. The measure-free
phrasing is not pedantry — the pushforward is a *random* measure, so a bound holding at `P_0` says
nothing. `mean`, `glm`, `glmnet` and `gam` are bounded fixed-dimension sieves; `boost` is a fixed
bounded-variation ball, because `max_iter=200`, `learning_rate=0.05`, `max_leaf_nodes=15` and
`early_stopping=False` are **hard-coded constants** in `learners/library.py` — a CV-selected round
count would take boosting out of the class. `forest` is outside, so `library="rich"` is outside.

> **(S)** `‖Δ_k‖_{L_2} = o_p(1)` — the reduction fit is `L_2`-continuous in the design and target
> columns it is handed.

**(S) is the open condition.** It is free for a fixed-basis linear smoother and not free for
anything that *selects* structure from the data — a split point, a bandwidth, a CV-chosen
candidate — where an arbitrarily small design perturbation can move the selection discretely.
Boosting is entropy-safe and design-continuity-unsafe, and it is the default reduction learner
whenever the primary one is boosting. `reduced_crossfit="nested"` is what computes `Δ_k`; a
measured dispatch put its consequence on `ψ` at or below what a redrawn fold split moves in every
cell, which is **supported, not shown**, since a consequence can hold by cancellation.

Two things not to read into "nested". It costs `K` times the primary nuisance fitting, and was
measured at **1.3x to 17x** a pooled fit's wall clock over four draws, reaching the outer cap on
two — what dominates is that nested reductions are noisier, so equation (10)'s near-singular solve
takes more rounds. And **neither construction makes the targeted collection fold-independent**:
`epsilon` is solved on all `n` rows, since `targeting_scheme="fold"` is refused, so a nested fit is
*nested in the nuisance models and pooled in the tilt*.

### Where the truncations are

`g_{r,2}`'s bound is fixed at **fit** time — the only bound in the package chosen at fit time
rather than at targeting time — because the array *is* a regression of a quotient by the mechanism.
Two consequences a reader will otherwise trip on. `SensitivityAnalysis.truncation_curve` moves the
clever covariate's denominator and does **not** move these arrays, so that part of the curve is
flat *by construction*; `ReducedSet.g_bounds` is on record so a reader of such a curve can find out
the sweep never reached them. And `gr1` is stored **untruncated** and bounded at read time through
`ReducedSet.bounded_gr1`, column by column and not complemented across arms.

One further condition sits beside (E) and is a rate rather than an entropy bound. `g_{r,2}`'s
target is `(1_a − ĝ)/ĝ` at the bounded mechanism, so its envelope is `1/lo − 1`, and equation
(10)'s covariate divides by `g_{r,1}` truncated at the same bound, so that envelope is `O(1/lo²)`.
Under `g_bounds="auto"`, `lo → 0` and the envelope **grows with `n`** — which pulls against the
bound-sequence row of [the scope section](#the-bound-inactive-scope), where exactly that shrinkage
is what makes the truncation asymptotically inactive. Both are open.

---

## 4. What the nuisances must satisfy

The union condition — `Q̄ = Q̄_0` **or** `g = g_0` — is assumed, not checked. Beyond it, the
release claim rests on the rate conditions of
[the remainder section](#the-remainder-terms-and-the-rate-conditions), which fall on **five**
estimated functions rather than two:

1. the primary outcome regression `Q̄_n`;
2. the primary propensity `g_n`;
3. the reduced outcome regression `Q_r`;
4. the reduced propensity `g_{r,1}`;
5. the reduced propensity `g_{r,2}`.

The reduced regressions are the part it is easy to forget, and they are where the guarantee is
bought. Their consistency is **estimated, unmeasured**: a saturated learner recovers them exactly
on the exact law, which is consistency at one learner on one law and not a rate. A study over
6,000 fits found the interval demonstrably better than a plain TMLE's where one nuisance is badly
estimated — `0.844`/`0.848` against `0.532`/`0.472` in the cell built for it — and **nominal
nowhere**, the best reading being `0.880`, with the three reductions fitted by `glm`. That is a
measurement of a configuration, not of the theorem's condition.

Practical consequences:

- **Choose the reduction learners deliberately.** They default to the primary specification.
  `reduced_outcome_learner=` and `reduced_treatment_learner=` are two keywords rather than one
  because the tasks differ: `g_{r,1}` is a conditional probability and the other two are
  conditional means of a signed quantity. A learner *instance* built for classification cannot
  serve `Q_r`, whose target is an outcome residual.
- **A flexible primary nuisance does not buy a flexible reduction.** The reductions are on one
  scalar, so their bias is a one-dimensional smoothing question and independent of how well the
  primary fit adjusted for `W`.
- **`library="rich"` is outside the entropy condition** of
  [section 3](#reduced-regression-cross-fitting) via `forest`, and is not warned about at runtime.

---

## 5. Diagnostics to inspect

In cost order. The first two are free.

| call | what it answers |
| --- | --- |
| `res.score_verdict` | the score check's verdict, carried whether it passed or not. `summary()` prints it whenever it **fails**; a passing fit says nothing extra. Derived from the fluctuations rather than stored, so a reloaded fit recomputes it. |
| `res.validation.score_check()` | the same object, asked for directly |
| `res.validation.correction_check()` | the doubly-robust rows: per arm, per equation. Empty unless the fit is a guarded `DRTMLE`. |
| `res.validation.nuisance()` | the primary fits' held-out risk and diagnostics |
| `res.validation.refute()` | negative controls; costs refits |

`correction_check()` recomputes each arm's `P_n[w D*_g]` and `P_n[w D*_Q]` **from the exact
returned state** and reports the residual against the score the targeting step recorded. Five
conditions on how it does so, each ruling out a way of passing for the wrong reason: per arm and
never only on the ATE, since arm-specific errors cancel in a difference; **before** the contrast is
constructed; with the row weights included; on **one outcome scale** — everything is reported on
the outcome's own scale, so that a correction score and `se/√n` are comparable numbers rather than
two quantities a factor of `range` apart; and, in the tests, on a fixture where the truncation
binds.

**Two failures, and they are not the same failure.**

- *An identity residual* — `Δ_g` or `Δ_Q` above `IDENTITY_TOLERANCE = 1e-12` — is a **software
  defect**. The fit solved one expression and reported another, and no amount of further iteration
  would fix it, because the loop is not posing the equation the curve needs. The tolerance sits
  seven orders above the arithmetic and four below the smallest observed real failure.
- *A correction score* above the inferential tolerance is a **fit that did not solve its
  equations** — the ordinary thing, reported per arm and per equation so a reader can see which.

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
statement about whether the five functions of [section 4](#4-what-the-nuisances-must-satisfy) are
adequately estimated, and the two are independent in a way that is easy to get backwards, because
a fit with badly wrong reductions returns a `psi`, an `se` and a confidence interval formatted
exactly like a good one, with every score green.

**The evidence is `tests/unit/test_oracle_reductions.py`**, and it is worth stating as a result
rather than a caveat. On an exact law, with **exact** reduced regressions handed to a real
alternation, the estimator recovers the truth *despite misspecified primary nuisances* — which is
the whole point of the variant. With **wrong** reductions, the estimate moves, and **every score
equation still passes**. Nothing on the face of such a fit distinguishes it from the good one.

Three consequences for practice:

1. **The score check is necessary, not sufficient.** Treat a failing score check as
   disqualifying and a passing one as saying nothing about the nuisances.
2. **Inspect the reduced-regression fits themselves**, not just the equations built from them.
   Their diagnostics are on `result.extra["drtmle"].diagnostics`, keyed `"qr"`, `"gr1"`, `"gr2"`.
3. **Where you cannot argue the rate conditions, do not treat the interval as settled.** Use this
   estimator where you have a reason to think one primary nuisance is badly estimated; that is the
   regime it was derived for and the regime the evidence covers.

The same distinction, once more, in the theorem's own terms: Theorem 1 licenses an interval
*conditional on* the three empirical scores being `o_p(n^(−1/2))` **and** the two second-order
remainder terms being `o_p(n^(−1/2))`. A fit can report on the first. Nothing reports on the
second.

---

## What the validation programme established

A closed programme of six pieces — a theoretical audit against the sources, a targeting-and-exit
study, a controlled coverage demonstration, a reference study for the reduced regressions, a
construction ablation, and a terminal experiment — is what this page's claims rest on. The
[roadmap's standing decisions](roadmap.md#standing-decisions) carry what it decided. In summary:

**Established.** The implementation is faithful to Theorem 1: the corrected curve is the Gateaux
derivative of the parameter, the sign of the mechanism correction is the appendices' orientation,
the reported variance is Theorem 1's, the three score equations are solved at the state returned,
and the interval is materially better than a plain TMLE's where one nuisance is badly estimated.

**Not established, and recorded as such.** Nominal coverage anywhere in the study, the best reading
being `0.880`; a localized cause for that shortfall — a six-contrast construction ablation over
2,496 fits returned a **null** on its primary column, and a terminal experiment over both a
selection and an independent audit cohort nominated **nothing**; and any `src/` change justified
against the theorem. Two measured quantities account for the shortfall and are one premise measured
twice: the second-order remainder Theorem 1 assumes negligible does not vanish at these sizes, and
the reported `se` runs about 10% short of the spread it covers in one drift cell and about 16%
*long* in the other — so the second is not a separate defect in the variance estimator. `σ̂²_n` is
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
