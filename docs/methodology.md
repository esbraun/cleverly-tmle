# Technical appendix

What each estimand *is*, the influence curve that makes it efficient, the second-order
remainder that double robustness consists of, and — for every one of them — the test that
fails when it is built wrong.

The order is the [user guide](user-guide.md)'s: one section per algorithm, then [how this is
validated](#how-this-is-validated) across all of them, then the taxonomy behind every refusal
the library states, then how to add an estimand of your own, then the references.

## Regimes: the density-ratio covariate

The influence curve is checked on the same footing as the ATE's: against the complex-step
Gateaux derivative of an independently written functional at `1e-12`
(`tests/unit/test_influence_gateaux_regime.py`), and the second-order remainder against its
closed form (`tests/unit/test_remainder_regime.py`), over a static regime, a rule that
depends on `W`, and a stochastic one that is degenerate nowhere.

## Shifting a continuous dose: why an MTP is not the regime it induces

A shift is *not* the stochastic regime that induces the same density, though the temptation
to reuse `regime_means` is real: `d` induces `g^d(b | W) = Σ_{a: d(a,W)=b} g(a | W)`, and the
`Stochastic` regime at that density has the same mean *and* the same clever covariate, entry
for entry. The influence curves differ anyway. A regime's plug-in term averages `Qbar` over
the doses, `Σ_b g^d(b | W) Qbar(b, W)`, a function of `W` alone; a shift's reads the dose the
unit actually received, `Qbar(d(A, W), W)`. The two agree only in conditional expectation
given `W`, and the gap is exactly

```
Var(D_mtp) = Var(D_regime) + Var( Qbar(d(A,W),W) − E[Qbar(d(A,W),W) | W] )
```

so a modified treatment policy is strictly *harder* to estimate than the regime inducing the
same mean. `tests/unit/test_influence_gateaux_shift.py` keeps that as a negative control: it
asserts the two means agree, the two influence curves do not, and the identity above holds —
so a later "simplification" that delegates one to the other fails loudly.

The influence curve is checked on the same footing as the rest: against the complex-step
Gateaux derivative of an independently written functional at `1e-12`, on a law with four
ordered doses and **two** caps (`tests/discrete_law_shift.py`). The second cap is not
redundant. A unit can only have been shifted *to* dose `a` if the shift from `a - δ` was not
itself held back, so `h` carries the further indicator `1{a ≤ u}` — which is invisible
whenever the cap sits at or above the largest dose, the common case. The tight cap is what
caught that term missing.

**Coarsening the outcome.** With `delta=` the covariate becomes `h(a, W) / π(a, W)`, and the
argument that makes it the existing derivation with one further factor is the same one that
settled `incremental=` with `delta=`: only the residual term needs the inverse weight, since
`Qbar(d(A,W),W) − Ψ` is a function of `(A, W)` and both are recorded whatever happens to `Y`.
What is *not* the same is where `π` is read. The fluctuation updates `Qbar` as a function of
the dose, so `Qbar*(d(A,W), W)` needs the covariate — and hence the mechanism — at the dose
the policy assigns. The arm path does this too; its `1{A = a}` indicator hides it.

Two things about checking that are worth recording, because both were measured rather than
assumed. The oracle law had to be a **new** one crossing `tests/discrete_law_shift.py` with
`tests/discrete_law_cde.py`, and its `π` had to vary with the **dose** — a mechanism depending
on `W` alone makes `π(d(a,w), w) = π(a, w)` identically, collapses the whole `(n, S+1)` array
to identical columns, and leaves a law that proves nothing while passing. And a Gateaux check
on an exact law **cannot see** a mechanism evaluated at the wrong dose in a counterfactual
block: there `epsilon` is zero, so the reported curve reads the observed block and the
untargeted `Qbar`, and no counterfactual block is read at all. That mutation is pinned
structurally in `tests/unit/test_shift_submodel.py` and behaviourally in
`tests/unit/test_shift_fit.py`; it was applied and seen to pass the Gateaux module first.

## Tilting the odds of treatment: two score equations

**Two score equations, and the mechanism is targeted.** Because `q_δ` is built out of `g`,
the efficient influence function carries a term for the pathwise derivative through it:

```
φ = (δA + 1 - A)/D · {Y - Q̄(A,W)}            ← the Q̄ score
  + δ{Q̄(1,W) - Q̄(0,W)}/D² · (A - g)          ← ∂m/∂g
  + m(W) - Ψ(δ)
```

The second term lives in the tangent space of the *treatment* mechanism, so no
fluctuation of `Q̄` can reach it. This is the first estimator here that targets the
mechanism: `g` gets a logistic submodel of its own whose score is exactly that term, and
because each covariate reads the other's fitted value the two alternate. The alternation
is coordinate ascent on one joint likelihood — the outcome and treatment quasi-likelihoods
are separate factors — so the joint value never decreases and the loop has an actual
convergence argument. `score_check()` therefore reports two rows per fit rather than one:

```
target             kind          |score|    threshold  ok
-----------------  ------------  ---------  ---------  ---
ipsi               fluctuation   6.9e-17    2.4e-06    yes
ipsi (mechanism)   fluctuation   8.7e-10    2.4e-06    yes
```

The per-estimand rows below them already check the two equations *jointly* — the influence
curve holds both terms, so its mean cannot be zero unless both are solved — but they
cannot say which one stalled. These can.

**It is not the stochastic regime at the same density**, and the temptation is real: a
`Stochastic` regime evaluated at `q_δ` has the same mean and, entry for entry, the same
clever covariate. Its influence curve is the one above without the middle term. The gap is
not a wash — the extra term is mean zero given `W` and orthogonal to both halves of the
regime curve, so

```
Var(D_ipsi) = Var(D_regime) + Var( δ{Q̄(1,W) - Q̄(0,W)}/D² · (A - g) )
```

exactly. Treating an incremental intervention as the regime that induces it does not
merely report a different quantity: it reports a standard error that is too **small**,
always. `tests/unit/test_influence_gateaux_ipsi.py` keeps that identity as a negative
control, on the terms the shift axis already set.

**It is not doubly robust, and it is the only estimand here that is not.** `g` appears in
the estimand itself, so every term of the second-order remainder carries `(ĝ - g₀)` as a
factor:

```
R₂ = (δ-1)δ · E[(g₀-ĝ)² (Q̄₀(1,W) - Q̄₀(0,W)) / (D₀ D̂²)]     ← survives a perfect Q̄
   + (δ-1)  · E[(g₀-ĝ)/D̂ · {q̂(Q̄₀(1,·) - Q̄(1,·)) + (1-q̂)(Q̄₀(0,·) - Q̄(0,·))}]
```

A consistent mechanism kills the remainder whatever `Q̄` does; a consistent `Q̄` does not,
and no accuracy in it can. Read the interval as conditional on `g` being right — which is
why `sensitivity.positivity()` still reports on this axis and matters *more* here than
elsewhere, there being no doubly-robust fallback. `tests/unit/test_remainder_ipsi.py`
asserts both directions as equalities rather than as an absence.

`delta=` **is** supported here, and the way it came to be is worth stating: it was once
refused on the grounds that a further mechanism inside the covariate would be a different
derivation with no oracle law to check it against, and both halves of that were wrong.
`tests/discrete_law_mar.py` is such a law, and taken to it the derivation is the same one
with an extra factor — `π(A, W)` divides the *outcome* half of the covariate and Kennedy's
`∂m/∂g` term is untouched, because `q_δ` is a functional of `P(A | W)` and both `A` and `W`
are recorded whatever happens to `Y`. What does change is the **guarantee**, which
*tightens* rather than weakening: `ĝ` right **and** one of `π̂`, `Q̄` right, since the
`(ĝ − g₀)²` term is π-free and survives everything else, and the two mechanisms cannot
trade off the way they do on the arm path.

What is refused rather than approximated: `intermediate=` (a controlled direct effect
under a tilt of the mechanism is a parameter this package has not written down, and
reporting one would mean guessing at its influence function), a multi-valued treatment (an
odds multiplier names two arms, and Kennedy's tilt has no single-parameter generalisation
to a multinomial mechanism), and `CTMLE` — which is **wrong by construction** rather than a
gap: it cross-validates the *choice* of `ĝ`, and each candidate `ĝ` defines a different
`Ψ(δ)`, so the search would be selecting between estimands rather than between estimators.

The influence curve is checked on the same footing as the rest: against the complex-step
Gateaux derivative of an independently written functional at `1e-12`, for a tilt above
one, a tilt below, and the natural course, with deliberate-mutation controls that fail if
the `∂m/∂g` term is dropped, mis-scaled or mis-signed.

## The MSM projection: its matrix and its remainder

With the identity link the clever covariate is `h(a, V) φ(a, V) / g(a | W)`, one column per term, so
the score equation is one per coefficient rather than one per arm — which is why this is a
fourth parameter axis and why `msm=` cannot be combined with `interventions=` or `shifts=`. The
counterfactuals are still the arms; what changed is the summary. A **saturated** working
model — one indicator per arm — reproduces the per-arm report exactly, point estimate and
influence curve alike, which `tests/e2e/test_msm.py` asserts against a plain fit.

### Under a link

Three things change inside, and each is a place the obvious generalisation is wrong:

- **The clever covariate reads `β`.** It is `h(a,V)·(dm/dη)·φ(a,V) / g(a|W)`, and `dm/dη`
  is a function of `m(a,V;β)`. So the fluctuation and the projection are solved *together*,
  alternating until both settle. It converges fast — the shift in `β` falls by a factor of
  10⁻³ to 10⁻⁴ per round, because `β` reaches the covariate only through that smooth factor
  — and `res.fluctuations["msm"].projection` carries the per-round trace.
- **The matrix the influence curve is premultiplied by is no longer the Gram matrix.** It is
  `M = E[Σ_a h ((dm/dη)² − (Q̄ − m)·d²m/dη²) φφᵀ]`, and the second term vanishes only where
  the working model *fits* — which is exactly what a projection does not promise. Dropping
  it is wrong in a way no saturated-model check can see, so the oracle's working model is
  deliberately unsaturated and carries that mutation as a control.
- **The remainder is second-order without being zero.** With the identity link a correct
  mechanism drives `R₂` to exactly zero, and that exactness is the *linearity* of the
  estimating equation in `β`, not a stronger form of double robustness. Under a link what is
  left is quadratic in `β̂ − β₀`; `tests/unit/test_remainder_msm.py` measures the rate rather
  than asserting the equality. The other half is untouched: a correct outcome regression
  still gives exactly zero, under every link.

Under `targeting_scheme="fold"` each fold solves its own `β`, since `β` is a coefficient the
covariate reads and the point of fold-wise targeting is that no row contributes to any
coefficient that fluctuates it. The pooled score is still exactly zero — each fold's is zero
at the `β` its own rows were fluctuated at.

A saturated working model still reproduces the per-arm report, now through the link:
`expit(β_a)` is `E[Y(a)]` to machine precision, with influence curves related by the delta
method, which is what says a link is a reparameterisation of the same counterfactual means
rather than a second estimator.

One thing is still **refused rather than approximated**, because of the derivation, in the
sense [How to read a refusal](#how-to-read-a-refusal) sets out:

| refused | kind | what it would need |
| --- | --- | --- |
| weights derived from the estimated mechanism (a "stabilised" MSM) | wrong by construction | `h` would be a functional of `P`, so the EIF carries a further term for the pathwise derivative through `ĝ` — the same argument that gives an incremental intervention its own axis. Supplying such weights anyway does not fail; it reports a standard error that is too small |

The influence curve is checked on the same footing as the rest: against the complex-step
Gateaux derivative of an independently written functional
(`tests/unit/test_influence_gateaux_msm.py`), under every link, and the second-order
remainder against its closed form (`tests/unit/test_remainder_msm.py`). The oracle's
working model is deliberately **not** saturated — three coefficients against six `(w, a)`
cells — because a saturated one agrees with the means whatever the projection code does, and
its weights are deliberately **not** uniform: with `h ≡ 1` the design is orthogonal and
`β_a` collapses to the marginal ATE *identically*, so code that reported the ATE under the
name `msm[a]` would pass every check. Under a link the oracle solves its own normal
equations by a *fixed* number of Newton steps and no convergence test — a comparison is not
analytic, and a functional that branched on one could not be differentiated by a complex
step at all — with a check that doubling the count moves neither the value nor the curve.

## Treatment given over time: the sequential regression

The estimator is the **sequential regression** of Bang & Robins (2005) as targeted by van
der Laan & Gruber (2012). The g-formula here is an iterated conditional expectation, so it
is `T` ordinary regressions run backwards, each one's prediction the next one's outcome:

```
Q̄_{T+1} = Y
Q̄_t(H_t) = E[ Q̄_{t+1} | H_t, A_t = a_t, C_t = 1 ]        for t = T, …, 1
Ψ        = E[ Q̄_1(H_1) ]
```

Each regression is fitted on the units that followed `ā` and stayed under observation
through `t`, and predicts for those that did so through `t − 1` — which are *exactly* the
units the previous step is fitted on. That is what makes the recursion close, and
`tests/unit/test_longitudinal_data.py` asserts the two masks are the same set rather than
leaving it to be read off this paragraph.

That substitution estimator is not efficient and has no influence curve. Each node
therefore gets its own logistic submodel, with clever covariate

```
h_t = 1{Ā_t = ā_t, C̄_t = 1} / ∏_{s ≤ t} g_s(a_s | H_s) c_s(H_s, a_s)
```

whose score is the `t`-th term of

```
D*(O) = Σ_t h_t ( Q̄*_{t+1} − Q̄*_t ) + Q̄*_1(H_1) − Ψ
```

so solving all `T` of them makes the fit solve `P_n D* = 0`. The recursion carries the
**targeted** prediction forward, not the initial one, so a residual left by one node is
regressed away by the next instead of accumulating.

The influence curve is checked on the same footing as every other estimand in this library:
against the complex-step Gateaux derivative of an independently written g-formula, on a
two-time-point law whose every cell probability is a multiple of `1/N` so that a sample of
`N` rows realises it *exactly* (`tests/discrete_law_longitudinal.py`). Handed the saturated
learner there, the point estimate is the truth to the last bit and the reported curve
matches the derivative to `1e-14` **absolute** — the comparison is made with `rtol=0`, as
every other Gateaux check in this repository is, because these curves reach order 20 and
a relative tolerance would quietly loosen the claim by six orders of magnitude. A negative
control in `tests/unit/test_influence_gateaux_longitudinal.py` fails if the censoring
probabilities are dropped from the cumulative product, and a gate in the same file fails
if the estimator reports a parameter the law has no longhand functional for.

What is **refused rather than approximated**, and why. Each is refused *by name*: the
keyword is accepted and rejected with the row below, rather than arriving as an
`unexpected keyword argument` that names no reason. The `kind` column says which sort of
refusal it is, in the sense [How to read a refusal](#how-to-read-a-refusal) sets out — the
rows are not all the same sort of thing, and only one of them is a warning about your
analysis rather than about this package's coverage.

| refused | kind | what it would need |
| --- | --- | --- |
| eliminating the competing events | a different question | what is reported is the cause-specific cumulative incidence with the competing causes *left alone*, so a competing event is part of the history. Removing it makes it an intervened node: a further factor per node in the denominator, and its own no-unmeasured-confounding and positivity assumptions. **Competing risks themselves are supported** — see [Competing risks](user-guide.md#competing-risks) |
| `intermediate=` | a different question | a controlled direct effect fixes a mediator at one time point; over a sequence of nodes, with mediators that are themselves time-varying, that is a different identification rather than a further column |
| a multi-valued treatment at a node | not written yet | the cumulative product needs one factor per arm per node, and the report one parameter per *sequence* of arms — which is readable through [a working model over the regimens](user-guide.md#summarising-the-regimens-a-marginal-structural-model), so that is the machinery it would report through |
| an outcome missing for a reason other than censoring | wrong by construction | left as it is, the probability of observing it is silently taken to be one. Encode it as a final censoring column, so it is estimated and enters the cumulative product |
| the targeted bootstrap, and `res.sensitivity` | not written yet | both refit against resampled or re-truncated nuisances. `g_bounds` enters the *pseudo-outcome* of every earlier node through the recursion, so changing it changes what the earlier regressions were fitted to: there is no `retarget` here that re-solves the fluctuation alone, and the whole backward pass has to run again. For positivity — the assumption that bites hardest here — `res.diagnostics()` already answers the question |

## Dynamic rules: what the oracle law checks

The influence curve is checked on the same footing as the static case, on the same law
(`tests/discrete_law_longitudinal.py`): `W` and `L₂` are binary there, so a rule is a
lookup over four cells and the oracle can state one longhand while the estimator is handed
the same plan as a callable. Three earn their place — one that ignores the history, which
must reproduce the constant plan it equals *bit for bit*; one reading `L₂`, which no static
plan can express; and one dynamic at the **first** node, the only case where the follower
mask compares against a per-unit value at `t = 1`. Two deliberate mutations confirm the
controls bite, and one of them is worth knowing about: evaluating the mechanism at a
constant arm turns six influence-curve comparisons red and leaves *every point estimate
green*, because with an exact initial fit `epsilon` is zero, `psi` is the plug-in, and no
error in `g` can move it.

`make_longitudinal` ships a quadrature truth for a rule too, so the nightly coverage tier
covers one. Getting that right needed a different integration rule rather than a wider one:
an indicator puts a step function into the integrand, where a Gauss–Hermite rule converges
algebraically rather than spectrally, and the naive version moved by `1.7e-3` between 48
and 64 nodes — worse than the Monte Carlo it exists to avoid. The `L₂` axis is therefore
integrated as two Gauss–Legendre panels meeting at the jump, which makes the arm constant
*within* a panel and the answer stable to `1e-13` under refinement.

## A working model over regimens: pooling and rank

`link="log"` and `link="logit"` mean what they mean at one node, and
`res.coefficients(scale="ratio")` exponentiates them. Three things differ inside, and each
is a place the obvious generalisation is wrong:

- **The node fluctuation is pooled across the regimens.** At one node the covariate's `p`
  columns get their rank by summing over the arms *within a row*: a unit contributes
  `φ(a, V)` at the arm it received. A regimen is a plan and not a value some unit took, so
  there is nothing to sum over within a row — a per-regimen covariate is `φ(ā, V)` times
  the scalar `h_t`, and whenever the working model has no effect modifier `φ(ā, V)` is
  *constant down the rows*, making that covariate rank one and collapsing its `p` score
  equations into one. So each node solves a **single** fluctuation over the regimens
  stacked, with one shared `epsilon`. The backward recursion is therefore *lockstep*:
  outer over the nodes, inner over the regimens, one update, all carried forward together.
- **A saturated working model reproduces the per-regimen report** — one indicator per
  regimen makes the stacked covariate exactly block-diagonal, and each block is the array
  the plain recursion would have used, entry for entry. Not *bit for bit* on the estimate,
  though, and that is worth knowing: the pooled Newton's convergence test and line search
  are taken over all the stacked rows, so the two can stop on different iterates. On a law
  the sample realises exactly no step is taken at all and the agreement is exact; elsewhere
  it is `1e-11`.
- **Under a link, one round of the alternation is a whole backward pass.** `β` enters the
  covariate through `dm/dη`, so `Q̄*_t` moves with it — and `Q̄*_t` is node `t−1`'s
  *regression target*, so every earlier node's learner is refit. There is no fixed `Q̄⁰` to
  restart from; the fixed point is stated over the whole pass. It costs four or five passes
  in practice, since `β` reaches the covariate only through that smooth factor. The
  mechanism is free of `β` and is fitted once.

`h(ā, V)` and `weights=` are different objects and must stay so: the first says how the
regimens are traded off inside the projection and sits in the covariate, the second tilts
the *population* the projection is taken over. Merging them would divide the estimating
equation by the very tilt it applies. The projection is solved on the **raw** outcome
scale, as it is at one node and for the same reason — a coefficient vector has no single
scale to map back with.

The influence curve is checked against the complex-step Gateaux derivative of an
independently written projection on the same exact law
(`tests/unit/test_influence_gateaux_longitudinal_msm.py`), under every link. The oracle's
working model is deliberately **not** saturated — three coefficients against twelve
`(W, regimen)` cells — and its weights deliberately **not** uniform, for the reasons [the
point-treatment oracle](#the-msm-projection-its-matrix-and-its-remainder) gives; both
choices are asserted on the law itself, so they are shown to be load-bearing rather than
claimed to be. Seven mutations were applied and the tests watched; three of them passed on
the first try and each was a real gap, which is recorded in the roadmap item below.

## Survival: which population each node is fitted on

**Which population each node's regression is fitted on is the whole of what changes**, and
it is the one thing here easy to get backwards. The recursion is the same one, seeded at
the horizon with `Q̄_{k+1} = 0` and carrying back

```
Z_t = Y_t + (1 − Y_t) Q̄*_{t+1}
```

fitted on the units at risk *entering* `t` — event-free through `t − 1`, which is one node
earlier than the censoring factor runs to. A unit that has the event at `t` **is** in node
`t`'s regression: it is the observation that the event happened. It is not in node
`t + 1`'s. So the identity the recursion closed on generalises rather than holds:

```
at_risk(t + 1) == following(t) & event-free at t
```

Tidying that `t − 1` to a `t` — which reads like a correction, since the censoring index
really is `t` — silently drops every failure from its own node's regression, biases the
risk downwards, and leaves every score at `1e-16` and every convergence flag green. It is
a deliberate mutation in `tests/unit/test_influence_gateaux_survival.py`, and it turns 26
of that module's 30 tests red.

What does **not** change is the positivity story. Being event-free is part of the history,
not an intervened node, so it enters the *indicator* of the clever covariate and never its
denominator: the cumulative product is still over the `2T` treatment and censoring factors,
truncated per factor, and `res.diagnostics()` reports the same weights — now with a
`horizon` column beside the `time` one, since the leverage is shared across horizons and
the `epsilon` is not.

And the pin that says this is a generalisation rather than a second estimator beside the
first: **a fit whose event can only happen at the last node reproduces the end-of-study fit
bit for bit** — `psi`, the whole influence curve, and every `epsilon`. It is pinned as such
in `tests/e2e/test_ltmle.py`.

## Competing risks: the cause-specific recursion

**Which competing-risks question this answers.** The competing causes are **left alone**.
`cif_regimen[always, relapse @ t=2]` is the incidence of relapse in a world where everyone
took the regimen and death went on happening as it does — a total effect. The other
estimand, the incidence of relapse if death were *eliminated*, intervenes on the competing
event: it needs a further factor per node in the clever covariate's denominator, and its
own no-unmeasured-confounding and positivity assumptions for the competing event itself.
That is a different question rather than a setting — see [A different
question](#a-different-question) — and `eliminate=` is refused by name saying so. Neither is a cause-specific hazard ratio, and neither is a Fine–Gray
subdistribution hazard; nothing here reports either.

**What changes in the recursion is one factor.** The pseudo-outcome carried back is

```
Z_t = 1{cause j at t} + 1{no event at t} · Q̄*_{t+1}
```

— a **cause-specific numerator** against an **all-cause survival factor**. A unit that left
through the competing cause contributes a zero and carries nothing forward, because it is
no more available to have this cause's event than one that already had it. Writing
`1 − 1{cause j at t}` there is the cause's own survival, is wrong by exactly the mass that
left through the other causes, and is the mutation
`tests/unit/test_influence_gateaux_competing.py` exists to catch — it takes 21 of that
module's 130 tests, every one of them at `t = 2`, since at the first horizon there is no
survival factor to get wrong.

Everything else is what it was. A competing event is part of the *history* rather than an
intervened node, so it enters the clever covariate's **indicator** and never its
denominator: positivity is still a statement about the same `2T` treatment and censoring
factors, `g_bounds` means what it meant, and `at_risk`, `following` and the mechanism's fit
mask are all-cause — leaving the risk set is leaving it, whichever cause did it. The causes
therefore share every nuisance fit and differ only in what is regressed, so `J` causes cost
`J` backward passes and one mechanism. A fit declaring a single cause reproduces a
single-event survival fit **bit for bit**, which is what makes this a generalisation rather
than a second estimator: end-of-study, then survival, then this.

## C-TMLE: how the selection is evidenced

Two more things about how this is evidenced, because they change how the numbers in
[the C-TMLE example](user-guide.md#collaborative-tmle) should be read. When `Qbar` is
correctly specified — as it is in the example process — the
*empty* propensity model is a legitimate MSE-minimising choice, and C-TMLE usually makes
it: 10 seeds out of 10 for the greedy search at `n = 700`. That is right, not a defect, but
it means a favourable comparison against plain TMLE on such a process would also be won by
a selector hard-wired to select nothing, so it is not evidence that the search
discriminates between covariates. The claim that it does is tested where selecting nothing
is *wrong* — with the outcome model reduced to a constant, the search includes the
confounder in every seed and still leaves the instrument out, while a do-nothing selector
is biased by 0.81 against 0.037. Second, there is no cross-language check: R's `ctmle` is
not compared against here or in CI. `cleverly.estimators.ctmle` sets both out in full.

## Doubly-robust inference: what the extra equations remove

`TMLE` is **doubly robust for consistency and singly robust for inference**, and the
distinction is the whole of what `DRTMLE` is for. The second-order remainder is a product,

```text
R_2 = || g-hat - g_0 || * || Qbar-hat - Qbar_0 ||
```

so one inconsistent nuisance still leaves `R_2 -> 0` and `psi-hat` consistent — which is
what [the double-robustness grid](https://github.com/esbraun/cleverly-tmle/blob/main/tests/e2e/test_double_robustness.py)
checks. The *interval* needs the strictly stronger `sqrt(n) * R_2 -> 0`. With both nuisances
converging at `n^(-1/4)` the product delivers it; with only one, the bad factor stops
shrinking, `R_2` becomes first order in the good one's error, and no nonparametric estimator
drives that below `n^(-1/2)`. So the estimator stops being asymptotically linear: its bias
does not grow but its coverage decays as `n` does. That is the sentence in [what the folds do
and do not buy](#cross-fitting-what-the-folds-do-and-do-not-buy) about a *product rate*,
read as a warning rather than as a condition.

van der Laan (2014) and Benkeser, Carone, van der Laan & Gilbert (2017) close it by solving
two further score equations, built from **reduced-dimension** regressions of each nuisance's
residual on the *other* nuisance. Writing `1_a` for `1{A = a}`:

```text
Qr(a, w)    = E[ Y - Qbar-hat(a, W) | A = a, g-hat(a|W) = g-hat(a|w) ]
gr1(a | w)  = P( A = a | Qbar-hat(a, W) = Qbar-hat(a, w) )
gr2(a | w)  = E[ {1_a - g-hat(a|W)} / g-hat(a|W) | Qbar-hat(a, W) = Qbar-hat(a, w) ]
```

Each is **univariate however many covariates the fit adjusted for**, which is the point:
they can be estimated fast enough whether or not the primary nuisances can. The equations,
in the software paper's numbering, are

| | equation | fluctuates |
| --- | --- | --- |
| (8) | `Pn[ 1_a / g*(a\|W) * (Y - Qbar*(a, W)) ] = 0` | `Qbar`, the ordinary covariate |
| (9) | `Pn[ Qr(a, W) / g*(a\|W) * (1_a - g*(a\|W)) ] = 0` | **`g`** |
| (10) | `Pn[ 1_a * gr2(a\|W) / gr1(a\|W) * (Y - Qbar*(a, W)) ] = 0` | `Qbar`, a second covariate |

and the reported influence curve is `D = D* - D*_Q - D*_g` with `D*_g` and `D*_Q` the
left-hand sides above, row by row. All three empirical means are zero after targeting, so
the subtraction **cannot move the point estimate**; it moves only the variance.

Four things about this are easy to get wrong, and each has an instrument.

**`guard=` is crossed.** `guard="Q"` guards against a misspecified *outcome regression* and
adds equation (9), which fluctuates `g`; `guard="g"` guards against a misspecified
*mechanism* and adds equation (10), which fluctuates `Qbar`. The keyword names the nuisance
you are worried about, not the one the equation it adds moves. An empty guard fits no
reduced regressions at all and is bit-for-bit a plain TMLE.

**The reduced regressions are refitted inside the alternation.** Equations (9) and (10) are
stated at *starred* `Qr*`, `gr*`, and the source's algorithm maps initial estimates of the
outcome regression, the mechanism **and the reduced regressions** into estimates satisfying
them. Holding them at their initial fit would solve a different equation. The cost is that
`retarget` is no longer arithmetic on cached arrays: a truncation curve on a `DRTMLE` fit
costs about a fit per point, and a plain `TMLE` handed these nuisances refuses rather than
re-solving against arrays it cannot refresh.

**The exact-law instrument is blind here, and derivably so.** Under a law the sample
realises exactly with a saturated learner — the setting of every `test_influence_gateaux*`
module — both nuisances are exact, so `Qr` and `gr2` have identically zero targets and
vanish *row by row*. Both extra coefficients are then zero and the estimator reproduces
`TMLE`. Those modules therefore supply a degeneracy check and would pass against a wrong
sign, an omitted term or a wrong `gr1` — which is a probability, does not vanish, and sits
in a denominator whose numerator does. What can see it is the remainder idiom, at nuisances
that are wrong on purpose: `tests/unit/test_remainder_drtmle.py` states what the guards
remove and `tests/unit/test_influence_drtmle.py` carries the difference-not-a-sum as an
explicit negative control.

**One guard removes the whole first-order remainder; two over-correct.** Each extra equation
subtracts a *projection* of `R_2` — equation (9) onto the sigma-algebra of `g-hat`, the other
onto that of `Qbar-hat` — and where both are all of `sigma(W)` either projection recovers the
whole of it, so the pair leaves exactly `-R_2`. That is arithmetic on a finite-support law
rather than a defect: asymptotically at most one of the two errors fails to vanish, so at
most one projection is non-negligible, which is why `drtmle` solves both by default.

**The influence curve is a fidelity claim about `drtmle`, not a theoretical result.** The
form above is what that package computes, read off its implementation. Theorem 1 of Benkeser
et al. (2017) is where the influence function is derived, and **it has not been read here**;
if the two disagree the theorem wins and `cleverly.inference.influence.reduced_corrections`
is wrong. Scope is likewise set by what has been *derived* rather than by what `drtmle`
accepts: a binary treatment, the `mean` group, and the univariate reduction. A multi-valued
treatment is a candidate rather than a refusal on principle — the equations are written with
a free `a` and nothing in them has a two-arm step — but van der Laan (2014) states its
problem for a binary treatment, the per-arm mechanism tilts do not renormalise, and an
implementation that accepts an argument is not a proof that the argument is licensed.

## Cross-fitting: what the folds do and do not buy

Cross-fitting the nuisances is what removes the Donsker condition on the nuisance
*estimators*. Pooled targeting on top of that adds an empirical-process term of its own,
because `epsilon` is fit on the rows it fluctuates — controlled, but by a different
argument: *conditional on the training-fold fits* `Qbar` is fixed, and `{Qbar(epsilon)}`
is then indexed by a fixed finite-dimensional coefficient over a compact set (two entries
for the default estimand, one per arm), Lipschitz in it, and so Donsker however complex
`Qbar` is.

That controls the empirical-process term and nothing else. Efficiency still needs the rest:
positivity bounding the clever covariate (the `g_bounds` truncation), the estimated
influence curve converging in `L_2`, the score solved to `o_P(n^-1/2)`, and a second-order
remainder that is `o_P(n^-1/2)` by a *product rate* on `ghat` and `Qbarhat` — a condition
on the learners, which the finite-dimensional fluctuation does not supply. That last one is
the condition [doubly-robust inference](#doubly-robust-inference-what-the-extra-equations-remove)
weakens, and the only one of the four that a *variant* of the estimator can do anything
about. Note too that a
single pooled `epsilon_hat` couples the folds: each row's nuisance prediction is out of
fold, but its *targeted* prediction is not. The two schemes share a first-order limit under
those conditions — but they are not the same estimator, and Zheng & van der Laan prove
their result for the fold-targeted construction specifically. See `targeting_scheme` in the
API docs for the full statement.

## What the score check proves, and what it does not

`score_check()` is necessary, not sufficient: it verifies that the fluctuation reached the
root of the equation the library posed, which a *consistently* wrong clever covariate would
also do. That the equation itself is the right one is a claim about the library rather than
about your fit, and is checked in the test suite — against the numerical Gateaux derivative
of the target parameter (`tests/unit/test_influence_gateaux.py`) and against the
second-order product remainder that double robustness consists of
(`tests/unit/test_remainder.py`), both exactly, on a law with finite support.

Missing outcomes get the same treatment rather than being taken on trust, because the
`1/P(Δ=1 | A, W)` factor is the kind of thing that solves its own score equation whether
or not it is right. `tests/discrete_law_mar.py` carries a finite-support law whose support
*is* the observed-data support — `(W, A, Δ)` always, `Y` only when `Δ=1` — and the two
modules above have `_mar` counterparts checking the influence curve against the numerical
Gateaux derivative at all eighteen points, including the six where nothing was observed and
the residual term must vanish exactly. What double robustness means there is not what it
means without missingness, and the remainder module states it: **consistent if `Q̄` is
right, or if the _product_ `g·π` is right** — a correct propensity buys nothing on its own
when the missingness model is wrong, and errors in the two mechanisms can cancel exactly.

The controlled direct effect now has the same class of proof rather than an argument for one.
`tests/discrete_law_cde.py` carries a law on `(W, A, Z, Δ, ΔY)` whose CDE changes sign between
the two levels of `Z` — so confusing them inverts the answer rather than nudging it — and
whose negative controls are the mistakes this parameter invites: dropping the `q_z` factor
(which quietly estimates a total effect), using the other level's density, substituting the
marginal `P(Z=z)` for the conditional `q_z(a, W)`, and averaging the plug-in over the `Z=z`
stratum instead of over everybody.

One more thing worth knowing about double robustness, because the reassuring form of the
slogan is the one that sticks: **the two halves are not interchangeable when positivity is
strained.** With `Q̄` right the estimand is recovered by integrating a regression over the
covariate distribution, which needs no overlap at all. With only `g` right, everything rests
on `1/g` weights — and on a process with 11% of the population below `g = 0.05` that half
stops delivering, at a measured bias of −0.13 against −0.01 for the outcome half. It is not a
truncation artefact and not a bug; it is the positivity premise failing.
`tests/e2e/test_double_robustness.py` runs both overlap regimes and pins the asymmetry.

## How this is validated

Three tiers, and they fail on different mistakes. Worth knowing which claims rest on which,
because a simulation that comes out well is the weakest of the three and the easiest to read
too much into.

**Exact proofs, on laws a sample realises exactly.** `tests/discrete_law*.py` build
finite-support distributions whose every cell probability is a multiple of `1/N`, so an
`N`-row frame *is* the law rather than a draw from it. Handed oracle nuisances the initial
fit is exactly right, `epsilon` is zero, and the influence curve the estimator reports is
the EIF at `P₀` rather than an estimate of it. Against that: the reported curve equals the
complex-step Gateaux derivative of the identification formula — written longhand, sharing no
code with the library — to `1e-12`, for every estimand: the seven binary ones without
missingness (`test_influence_gateaux.py`), under MAR (`..._mar.py`) and for the controlled
direct effect (`..._cde.py`), and the per-arm means and contrasts on a **three-armed** law
(`..._multi.py`, against `tests/discrete_law_multi.py`). The third arm is not decoration:
two arms cannot distinguish code that keys everything by arm from code that has two columns
and calls them 0 and 1, and that law's labels sort into a different order than they were
written in so a helper equating arm code with arm position fails rather than passes. The
regime estimands get the same treatment (`..._regime.py`), over a static regime, a rule
that depends on `W` and a stochastic one that is degenerate nowhere — three kinds because
two static regimes could not distinguish code that mixes over the arms from code that
picks a column. The incremental estimands get it on the *same* law as the arm-indexed ones
(`..._ipsi.py`): `q_δ` is built from `g`, and `g` is a ratio of linear forms in the cell
probabilities, so the functional stays analytic and the complex step differentiates
through the mechanism as well as through `Q̄` — which is exactly the term at issue, and
one no regime can exercise. Three tilts, one above one and one below, because a sign
error in `∂m/∂g` survives on one side; and `δ = 1`, where the curve is `Y - Ψ` row by row
whatever the nuisances are. The shift estimands get it too (`..._shift.py`, against
`tests/discrete_law_shift.py`), on a law with four ordered doses and two caps — the tight
one because a cap above the largest dose never exercises the `1{a ≤ u}` factor, and that
law found the factor missing. Coarsening that outcome needs a further law again
(`..._shift_cde.py`, against `tests/discrete_law_shift_cde.py`), which crosses those doses
with the `(Δ, Z)` axes of `tests/discrete_law_cde.py` — a fourth dimension rather than a
wider third, because both parents have to go on proving their own derivations unchanged, and
because `π` and `q_z` have to vary with the *dose* for any of it to be checkable at all.
The longitudinal estimator answers to a law of its own
(`..._longitudinal.py`, against `tests/discrete_law_longitudinal.py`), on a two-time-point
process where `L₂` is caused by `A₁` and confounds `A₂` and where censoring depends on the
history at both nodes — so a fit that dropped either the intermediate covariate or the
censoring factors misses the truth rather than merely losing efficiency. Its oracle is a
*saturated learner* rather than a hand-written nuisance, because the oracle at the earlier
node is an expectation of the later node's regression: nothing anybody would want to
transcribe, and an unpenalised cell-mean fit on a law the sample realises exactly is
identical to it. A **survival** outcome gets a law of its own beside it (`..._survival.py`,
against `tests/discrete_law_survival.py`) rather than a wider version of that one, which
has to go on proving the end-of-study derivation unchanged: an absorbing `Y₁` puts a fourth
structural-missingness pattern into the support — a unit that had the event has no `L₂`,
`A₂`, `C₂` or `Y₂`, which is a different exit from being censored — and the parameter is a
curve, so the two horizons are checked as two parameters of one distribution. **Competing
risks** get a third law on the same footing (`..._competing.py`, against
`tests/discrete_law_competing.py`), for the same reason again: the event node is
three-valued, the support carries a missingness pattern per cause, and the cause-specific
incidence is a numerator over one cause against a survival factor over *all* of them —
which is the one thing neither sibling can check, since with a single cause the two
readings coincide. That law also answers for itself before any estimator reads it: every
influence curve it produces has mean zero under the law it is taken at, the causes and the
event-free probability exhaust the mass, and collapsing the causes reproduces the
single-event recursion — none of which can be arranged by agreeing with a buggy fit,
because no fit is involved. The second-order remainder is checked against its closed form in the nine
matching `test_remainder*.py` modules, which is what double robustness actually consists of.
Every one of these modules carries deliberate-mutation controls: each plausible way of
building the thing wrong is shown to move the answer by more than `1e-2`, four orders past
the window the real assertions use.

**Deterministic invariants and algebraic identities.** Relabelling `A ← 1 - A` has to swap
`EY1`/`EY0`, negate the ATE and turn ATT into `-ATC`; an arm-independent outcome mean has to
give exactly zero however hard the treatment was confounded; weights scale out; the iterative
and one-step solvers agree; the weighted and clever-covariate fluctuations solve the same
equation. These are cheap and they fail on the mistakes statistics is worst at catching — a
swapped sign, a swapped conditioning population.

**Simulation, for the claims that are about repeated sampling and nothing else.** Coverage,
root-n consistency, type I error and the estimator variants live in
`tests/e2e/test_coverage_slow.py` and run nightly. The double-robustness grid runs on every
push, in both a comfortable-overlap and a weak-overlap version — and the two disagree, which
is the point of having both.

Two things this does **not** include, stated plainly because their absence is easy to miss.
There is no comparison against another implementation: not R's `tmle`, not `tmle3`, not
`ctmle`. And `score_check()` passing is not evidence that the equation was the right one —
see below.

## How to read a refusal

The table below, every section above it, and the [user guide](user-guide.md) throughout, say
of certain things that they are "refused
rather than approximated". A refusal is always *by name* — the keyword is accepted and
rejected with a stated reason, rather than arriving as an `unexpected keyword argument`
that names none.

They are not all the same kind of thing. What a reader should do about one depends entirely
on **where the problem is**, and there are three places it can be: in this package, in the
question, or in the method itself.

| section | where the problem is | what to do about it |
| --- | --- | --- |
| [Not written yet](#not-written-yet) | in this package | the parameter is well defined and its derivation is settled; nobody has written it here. Ask for it, or compute it elsewhere. The ones judged worth building are on the [roadmap](roadmap.md#refusals-worth-lifting) |
| [A different question](#a-different-question) | in the question | what was asked for is a different estimand, usually with its own identification assumptions. Decide which one was meant; no flag here produces the other, and one that quietly did would be answering something nobody asked |
| [Wrong by construction](#wrong-by-construction) | in the method | the naive version *runs* and returns a plausible number that is wrong, usually with a known direction of error. Read these as warnings about the analysis, not about this package's coverage |

A fourth group needs no taxonomy: a fit whose *data* cannot support what was declared — a
horizon at which no event was observed among a regimen's followers, a cause with no events,
a regimen nobody followed, two absorbing causes firing at one node. Those are refused where
they arise, and they are statements about the sample rather than about anything here.

### Not written yet

Nothing is wrong with wanting any of these; they are gaps in coverage, and the message says
so rather than implying the request was ill-posed.

| refused | where |
| --- | --- |
| `CTMLE` on a multi-valued treatment | [multi-valued treatment](user-guide.md#multi-valued-treatment) |
| `DRTMLE` on a multi-valued treatment, with `delta=`/`intermediate=`, fold-wise, or composed with `CTMLE`; and `reduction="bivariate"` | [doubly-robust inference](user-guide.md#doubly-robust-inference) |
| the MNAR tilt on a `shifts=` fit | [shifting a continuous dose](user-guide.md#missing-outcomes-an-intermediate-and-weights-on-a-dose) |
| `intermediate=` and a multi-valued treatment with `incremental=` | [tilting the odds of treatment](user-guide.md#tilting-the-odds-of-treatment) |
| a multi-valued treatment at a node, the targeted bootstrap and `res.sensitivity` for `LTMLE` | [treatment over time](user-guide.md#treatment-given-over-time) |
| blocked-temporal and rolling-origin splits | [cross-fitting](user-guide.md#cross-fitting-and-cv-tmle) |
| replicate weights (BRR, jackknife) — a set of designs rather than one weight vector, so the shape it wants is a refit per replicate outside the estimator | [observation weights](user-guide.md#observation-weights-and-which-population-they-define) |

Four rows have left this list entirely: `ATT` / `ATC` on a multi-valued treatment is
roadmap item 1, observation weights for `LTMLE` is item 3, a working model over regimens is
item 4, and `delta=`, `intermediate=` and weights with `shifts=` is item 5. A fifth row was
*shortened* rather than removed — item 6 lifted the omitted-variable bound and the MNAR tilt
on a multi-valued treatment and left `CTMLE` behind. All of them have landed, and the
[roadmap](roadmap.md#refusals-worth-lifting)'s list is now empty. The row
item 5 left behind is a narrower gap than the one it replaced: the tilt itself is written, and
what is missing is the derivation saying whether the tilted parameter is still the shift
parameter.
The rest are there because nobody
has asked, not because anything stands in the way — with one exception worth naming:
`CTMLE` on a multi-valued treatment is the only row here whose *derivation* is unsettled,
since both searches order candidates by one propensity margin and with `K` arms there is no
canonical single ordering. That is now the whole of its row, where it used to share one with
two refusals that turned out to be transcription: an omitted-variable bound is one linear
functional at a time and an MNAR tilt is one arm's regression at a time, so both are a wider
loop over the contrasts rather than a derivation that stops at two arms.

### A different question

These are well-posed parameters. They are simply not the one being estimated, and no
setting turns one into the other.

| refused | the question it would answer instead |
| --- | --- |
| `eliminate=` on a competing-risks fit | the incidence of a cause if the competing events were *removed*, which intervenes on them rather than conditioning on the history — a further factor per node in the denominator, and its own no-unmeasured-confounding and positivity assumptions for the competing event. What is reported instead is the incidence with the competing causes left alone |
| `intermediate=` on `LTMLE` | a controlled direct effect fixes a mediator at one time point; over a sequence, with mediators that are themselves time-varying, that is a different identification rather than a further column |
| `ey1` and `ey_regime` from one fit; `msm=` with `interventions=` or `shifts=` | each keyword declares what "counterfactual" means for the fit, or how the counterfactuals are summarised. One fluctuation solves one set of score equations, so a fit reporting parameters from two axes would be putting two of them under one heading |
| `sensitivity.positivity()` on a continuous fit; `stratify_folds="treatment+outcome"` on a continuous outcome or dose | a per-arm propensity table has no rows when there are no arms. `sensitivity.shift_support()` asks the question that does apply — whether the density *ratio* stays bounded |
| `res.sensitivity`, `res.validation` and `res.save()` on an `LTMLE` result | each says what it would need rather than raising `AttributeError`. For positivity — the assumption that bites hardest there — `res.diagnostics()` is the answer, reporting the cumulative weight and effective `n` per regimen per node |

### Wrong by construction

The ones worth reading even by someone who never reaches for the keyword: most are mistakes
that are easy to make by hand, in any framework, and none of them announces itself. Each
produces a number, and the number is wrong.

| refused | what goes wrong if it is done anyway |
| --- | --- |
| a `Stochastic` regime whose density came from the estimated mechanism | `g*` becomes a functional of `P`, so the EIF carries a term for the pathwise derivative through `ĝ` that a regime's curve does not have. The reported standard error is too **small** |
| an incremental intervention built by hand as a `Stochastic` regime | the same omission, and its size is exactly `Var(δ{Q̄(1,W) − Q̄(0,W)}/D² · (A − g))`. Too small, always |
| a shift's inference taken from the regime inducing the same density | means and clever covariates agree entry for entry; the curves do not. The gap is `Var(Q̄(d(A,W),W) − E[Q̄(d(A,W),W) | W])`. Too small, always |
| a shift fit run on the complete cases when outcomes are missing | it is an ordinary shift fit on a *different* joint law of `(A, W)`, so it converges to a different number — and nothing in its own output says so. Measured on the dose fixture at 0.17, four standard errors, with a mechanism whose slopes are mild. `delta=` is what corrects it |
| a missingness or intermediate mechanism read at the observed dose rather than the assigned one | the fluctuation updates `Q̄` as a function of the dose, so `Q̄*(d(A,W),W)` is the update evaluated where the policy sends the unit. Silent wherever the mechanism does not happen to depend on the dose — and invisible to a Gateaux check on an exact law, where `epsilon` is zero and no counterfactual block is read |
| a "stabilised" MSM weighting `h` by the estimated mechanism | the same argument once more — `h` a functional of `P`, a term missing from the EIF |
| `g_bounds=` or `truncation_curve()` on an `incremental=` fit | `g` is *inside* `Ψ(δ)`, so truncating it moves the estimand rather than regularising a denominator. The result is a number for a parameter nobody declared |
| a `cap=` fitted from the data on a shift | the estimand becomes data-dependent: the interval conditions on an estimated boundary, and every bootstrap replicate targets a slightly different policy |
| `CTMLE` on an `incremental=` fit | each candidate `ĝ` defines a different `Ψ(δ)`, so the cross-validated search selects between *estimands* rather than between estimators |
| splitting a cluster across folds to buy more of them | the out-of-fold predictions stop being independent of the rows they are used on, and the standard error shrinks in exactly the direction `id=` was passed to prevent |
| median-of-estimates aggregation over `repeats=` | the median of the `psi_r` is not the estimator whose curve is the median of the `IC_r`, so the point estimate and its interval describe different functionals |
| a cross-validated variance of the across-draw average curve | at equal fold sizes it collapses to the pooled uncentred second moment for *every* partition — vacuous rather than merely arbitrary |
| a one-shot non-identity-link MSM | `∂m/∂β` depends on `β`, so a single pass reports a standard error for an equation it did not solve. The link is supported; what is refused is skipping the `(β, ε)` alternation it needs |
| frequency (count) weights | they assert a sample size the variance does not use. Expand the rows instead, which says the same thing where every part of the fit can see it |
| an LTMLE outcome missing for a reason other than censoring | its probability of being observed is silently taken to be one. Encode it as a final censoring column so that it is estimated and enters the cumulative product |
| a binary-only target on a multi-arm fit | it would report a contrast of arms `0` and `1` out of five under the name of a parameter about all of them. Targets declare `requires_binary_treatment` for this: `EY1`/`EY0`, which name one of exactly two arms, and the incremental estimands, whose tilt multiplies an *odds*. Not `ATT`/`ATC`, which are one parameter per non-reference arm and say so in the name |
| `MSM.linear` on non-numeric arm labels | a model linear in the arm reads it as a dose to interpolate between, and the fallback coding is the sort order — alphabetical, for strings — which is a dose scale nobody chose |

Three of these share one mechanism, and it is worth naming because it generalises past this
package: **if an intervention's density `g*` is a functional of `P`, the influence function
carries a term for the pathwise derivative through the estimated mechanism.** Omit it — by
building the intervention out of `ĝ`, or by borrowing a curve from the regime that induces
the same density — and the standard error is too small, every time, by a variance that can
be written down exactly. Two more share another: **a nuisance that sits inside the estimand
is not a knob**, so truncating `g` on an incremental fit or fitting a shift's `cap` from the
data moves the target rather than regularising the estimator.

The two detailed refusal tables — [marginal structural
model](#the-msm-projection-its-matrix-and-its-remainder) and [treatment over
time](#treatment-given-over-time-the-sequential-regression) — carry a `kind` column
naming which of the three
sections above each row belongs to.

## The oracle-law gate

Registering a target whose reported parameters have no branch in the `functional` of an
oracle law — `tests/discrete_law.py` for the arm- and regime-indexed estimands,
`tests/discrete_law_shift.py` for the shift-indexed ones — is a **test failure**, not an
oversight caught in review. The evidence this package offers that
an influence curve is correct is that it agrees, to ~1e-12, with one obtained by complex-step
differentiation of an independently written functional on an exactly representable law. An
estimand without that has no such evidence, and the registry is deliberately not allowed to
make skipping it easy. The gate walks the *parameter* names a target reports rather than the
target name, so a per-arm target needs an oracle for each — and a target intended for more
than two arms needs one on `tests/discrete_law_multi.py`, the three-armed law, since two arms
cannot distinguish code that keys by arm from code that has two columns and calls them 0 and 1.
The gate runs in both directions: an oracle branch no target reports is dead code, so a law
and the registry must cover each other exactly.

## References

- van der Laan & Rubin (2006), *Targeted Maximum Likelihood Learning*.
- Gruber & van der Laan (2010), *A targeted maximum likelihood estimator of a causal effect on a
  bounded continuous outcome*.
- Gruber & van der Laan (2012), *tmle: An R Package for Targeted Maximum Likelihood Estimation*.
- Zheng & van der Laan (2011), *Cross-validated targeted minimum-loss-based estimation*.
- van der Laan & Gruber (2010), *Collaborative double robust targeted maximum likelihood
  estimation*.
- Gruber & van der Laan (2010), *An application of collaborative targeted maximum likelihood
  estimation in causal inference and genomics*.
- Ju, Gruber, Lendle, Chambaz, Franklin, Wyss, Schneeweiss & van der Laan (2019), *Scalable
  collaborative targeted learning for high-dimensional data*.
- van der Laan & Gruber (2016), *One-step targeted minimum loss-based estimation*.
- Bang & Robins (2005), *Doubly robust estimation in missing data and causal inference models*.
- van der Laan & Gruber (2012), *Targeted minimum loss based estimation of causal effects of
  multiple time point interventions*.
- Neugebauer & van der Laan (2007), *Nonparametric causal effects based on marginal structural
  models*.
- Orellana, Rotnitzky & Robins (2010), *Dynamic regime marginal structural mean models for
  estimation of optimal dynamic treatment regimes*.
- Kennedy (2019), *Nonparametric causal effects based on incremental propensity score
  interventions*.
- Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), *Long story short: omitted variable bias
  in causal machine learning*.
- VanderWeele & Ding (2017), *Sensitivity analysis in observational research: introducing the
  E-value*.
- Scharfstein, Rotnitzky & Robins (1999), *Adjusting for nonignorable drop-out using semiparametric
  nonresponse models*.

The last three are for `DRTMLE`, and
[doubly-robust inference](#doubly-robust-inference-what-the-extra-equations-remove) rests on
them — on the first two for the estimating equations and on the third's implementation for
the influence curve, which is a fidelity claim rather than a transcription of a theorem.
**Theorem 1 of Benkeser et al. (2017) has not been read here**, and that section says what
turns on it.

- van der Laan (2014), *Targeted estimation of nuisance parameters to obtain valid statistical
  inference*.
- Benkeser, Carone, van der Laan & Gilbert (2017), *Doubly robust nonparametric inference on the
  average treatment effect*.
- Benkeser & Hejazi (2023), *Doubly-robust inference in R using `drtmle`*.
