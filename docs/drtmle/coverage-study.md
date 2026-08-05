# The coverage study: the design, both tiers, and what each has shown

`DRTMLE`'s definition of done is one sentence — *a demonstration that the interval attains its
nominal coverage where a plain `TMLE`'s does not* — and [piece C](../roadmap.md#c-the-demonstration)
is that demonstration. [The validation plan's §5](validation-plan.md#5-the-controlled-study-piece-c)
specifies it: two tiers, both off-diagonal cells, a prescribed rate with a committed drift
coefficient, three sizes, 250 replicates minimum, and rules frozen before the dispatch. **That
document is the specification and this one is the design**: what the cells actually are, what
constants were committed, and what the instrument has measured so far.

**C is three pull requests and two have landed.** The split follows the rule the roadmap is
lettered under — grouped where the *evidence* is shared — and the three have different evidence:

| PR | what it lands | evidence |
| --- | --- | --- |
| **C1** — *landed* | the harness, Tier 1 complete, the workflow, item 25's per-fit witness | the exact remainder of a prescribed sequence, and that the instrument works |
| **C2** — *landed* | Tier 2's prescribed-rate learners, the evaluation companion `P₀D̂` needs, the remainder columns | item 13's rate |
| **C3** | the pilot, the freeze, the final study, the second seed batch | item 3, and gates 1 and 2 |

**Tier 1 is not the demonstration and this page will not be read as one.** Its nuisance sequence
is handed to the estimator rather than learned, which is the only construction in which *"the
intended asymptotic regime was entered"* is true by definition — so it is where a remainder can be
read off exactly, and it is not an applied claim. **Tier 2 is the demonstration**, it landed with
C2, and its section is [below](#tier-2-a-prescribed-rate-rather-than-a-prescribed-sequence).

## Tier 1's two cells

Both off the diagonal of the misspecification grid, because which nuisance is wrong is the whole
axis and one cell is an anecdote. `benchmarks/drtmle_injection.py` is the code and carries the
same reasoning at each constant; [Tier 2](#tier-2-a-prescribed-rate-rather-than-a-prescribed-sequence)
is the same two cells with both nuisances fitted.

| cell | `Q̂` | `ĝ` |
| --- | --- | --- |
| **`q-drift`** | `Q̄₀ + n^(−α)·h_a` — consistent, slowly | `g₁ ≠ g₀`, fixed: `g₀`'s log odds shifted by `0.8` |
| **`g-drift`** | `Q̄₀ + d_a`, fixed, with `d ∈ [0.5, 1.5]` and arm 0 at half of arm 1 | `g₀ + n^(−α)·λ·d·g₀(1−g₀)` — consistent, slowly |

`α = 0.25`. It is the familiar bar for the *both-consistent* product condition and is **sufficient
rather than necessary** here: in an off-diagonal cell the misspecified nuisance's error is `O(1)`,
so the root-`n` drift is `n^(1/2−α)` and any `α < 1/2` grows. It is reported as a knob rather than
defended as a threshold, and what argues against pushing it smaller is the other side of the
ledger — the appendix-B terms `DRTMLE` needs negligible are built out of the *same* primary
nuisances, so a badly enough estimated one degrades the corrected estimator too.

### Three design decisions, each of which could look right and be wrong

**The base law is `linear_dgp()`, and it is chosen for overlap rather than for difficulty.** Every
misspecification here is prescribed, so nothing is asked of the process except that its own
mechanism stay interior — and a law whose propensity sits near `0.5` is the one that puts the cells
**inside** [the supported contract](../roadmap.md#the-supported-contract-and-item-25), so that a
coverage number is evidence about Theorem 1's estimator rather than about the constrained rendering
beside it. Whether that worked is measured rather than assumed, and [§ what Tier 1 already
showed](#what-tier-1-already-showed-and-it-is-not-what-the-design-expected) is where it did not
entirely.

**The outcome scaler is declared, not recovered.** The estimator maps `Y` onto `[0, 1]` before
fitting `Q̄`, so an injected outcome regression has to apply the same affine map.
`tests/conftest.py`'s `OracleOutcomeContinuous` recovers it by regressing the scaled outcome on the
raw structural mean, which is exact for a *binary* outcome and carries an `O(n^(−1/2))` error from
the noise for a continuous one — **the same order as the drift being injected**. So both estimators
are passed `q_bounds=(-8.0, 14.0)`: the map is then known in advance and identical across draws, and
a draw whose outcome leaves that support raises from `OutcomeScaler.from_outcome` rather than being
silently rescaled. The support is wide — `linear_dgp`'s outcome reaches roughly `[-3.5, 9.1]` at
`n = 2,400` — which costs a little of the logistic fluctuation's leverage and costs it identically
to both estimators.

**The drift coefficient is the deliverable, not the rate.** `α < 1/2` is not sufficient, because
the remainder is an **inner product** rather than a norm:

```text
R_2,a  = P_0[ (ĝ_a − g_0,a)/ĝ_a · (Q̂_a − Q̄_0,a) ]  =  n^(−α)·c_a + o(n^(−α))
c_a    = P_0[ (g_1,a − g_0,a)/g_1,a · h_a ]
```

`c_a` can vanish with `‖h_a‖ > 0` because `h_a` is orthogonal to the misspecification weight, and
`c_1 − c_0` can vanish in the ATE with both arm coefficients nonzero. So `h_a` is chosen **aligned
with that weight** and normalised so the coefficients come out at the design's declared values,
with the **arms given opposite signs** — which makes `c_ATE = c_1 − c_0` a sum of magnitudes and
cancellation impossible rather than merely unlikely.

### The committed calculation

Computed by `DGP.expectation`, which is the same Sobol rule `DGP.truth()` integrates with — a
second quadrature would put a Monte Carlo error of its own between a coefficient and the coverage
it explains. **Committed here before any fit was run**, which is what §5 asks:

| cell | `α` | `c₁` | `c₀` | `c_ATE` | `b₁` | `b₀` | `b_ATE` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `q-drift` | 0.25 | +0.2000 | −0.2000 | +0.4000 | +0.2000 | −0.2000 | +0.4000 |
| `g-drift` | 0.25 | +0.2440 | −0.1560 | +0.4000 | +0.0841 | −0.0159 | +0.1000 |

**The `b` columns are C3b's and they are the ones a shortfall is sized from.** `c` is the
*plug-in* remainder's coefficient and `b` the **estimator's bias**; the pair is what §5's
targeted-coefficient clause asks a design to declare, and the pilot's failure was reading the
first as the second. `b_ATE` differs between the cells and the reason is
[positivity](#tier-1-the-drift-has-to-survive-targeting-not-merely-exist), not preference.

`q-drift` declares both arm coefficients; `g-drift` declares only the ATE's, and that is structural
rather than a shortcut — a binary treatment's mechanism has **one** free function, since the
estimator reads `ĝ(1|W)` off a classifier and takes the complement, so one perturbation determines
both arms and only their combination can be set. What the arm coefficients then come out at is a
finding, and `tests/unit/test_drtmle_coverage.py` holds `c`'s to a floor of `0.02` and `b`'s to a
tenth of their own contrast — a share rather than an absolute, since the two cells declare
different `b_ATE`.

`c_ATE = 0.40` is sized from the drift a coverage number can resolve, and the sizing is **a
prediction the pilot checks**. With `σ_ATE` measured at `2.6` (a `se` of `0.106` at `n = 600` on one
injected fit), `bias/se ≈ n^(1/2−α)·c/σ` puts the plain interval's shift at `0.76`, `0.91` and
`1.08` standard errors at `600 / 1,200 / 2,400` — a `TMLE` coverage near `0.87 / 0.86 / 0.81`, so a
shortfall of `0.08` to `0.14`. That is comfortably clear of gate 2's predeclared `0.05` at every
size, and it is resolvable against 250 replicates' Monte Carlo error of `0.014`. **Provisional until
the pilot**, which is the one point at which §5 permits it to move.

> **The pilot checked it and it is wrong**, and not in its arithmetic: `bias/se ≈ n^(1/2−α)·c/σ`
> reads `c` as the coefficient of the **estimator's bias**, and `c` is the coefficient of the
> **plug-in** remainder. Measured, the two differ by a factor of about twenty and the predicted
> shortfall does not appear at all — Tier 1's `TMLE` covers at `0.90` to `1.00`. [What the pilot
> measured](#what-the-pilot-measured) has the numbers and
> `benchmarks/drtmle_tier1_bias.py` is the measurement; the paragraph above is kept as written
> because it is the prediction the pilot was run to test, and rewriting it would hide that it
> was tested.
>
> **The arithmetic is now restored to it rather than discarded**, which is C3b: the sizing was
> right and was applied to the wrong column, so the repair declares `b_ATE = 0.40` in `q-drift`
> and the shift is `0.76 / 0.91 / 1.08` standard errors again — measured on 24 draws at
> `+1.93 / +2.17 / +3.31` root-`n` bias, against `−0.22 / −0.56 / +0.11` before. The pilot's
> "factor of about twenty" is also corrected below: it was a noise-floor artefact, and the exact
> ratio is **436**.

The realised remainder, by quadrature — both columns, at the repaired design:

| cell | `n` | `n^α R₂(Q̂)` | declared `c_ATE` | `n^α R₂(Q̄*)` | declared `b_ATE` |
| --- | --- | --- | --- | --- | --- |
| `q-drift` | 600 / 1,200 / 2,400 | +0.40000 at all three | +0.4000 | +0.40000 at all three | +0.4000 |
| `g-drift` | 600 / 1,200 / 2,400 | +0.39918 / +0.39895 / +0.39887 | +0.4000 | +0.09963 / +0.09936 / +0.09911 | +0.1000 |

`q-drift`'s are exact at every size, and the two columns coincide for a reason worth knowing:
declaring `b = c` forces `P₀[w_a h_a] = 0`, so the injection is exactly **orthogonal to the
fluctuation's own score**, `ε` is zero in the limit and nothing is absorbed. Measured on real
fits, a fitted `ε` of `+0.00026 ± 0.00183` and an absorbed share of `0.0000`. `g-drift`'s carry
the `o(n^(−α))` term its drifting mechanism contributes.

The nuisance-error slopes are `−0.250` for the drifting nuisance and `+0.000` for the misspecified
one in each cell, which is the pair §5 asks for: a study reporting only the shrinking norm could
not tell a product going to zero because one factor does from one going to zero because both do —
and the second is the regime a plain interval is already valid in.

### What is integrated exactly, and what is not

`R₂` above is the **plug-in remainder at the injected sequence**, and it is exact because both
primary nuisances are prescribed functions of `W`. Two qualifications, both of which belong on the
face of the design rather than in a footnote.

> **This paragraph said the targeting step moves `Q̂` to `Q̄*` by `O_p(n^(−1/2))`, *"smaller than
> the injected `n^(−α)` at every `α < 1/2`, so it leaves the drift's leading term where it is"*.
> That is false, it is the single sentence the whole mis-sizing came out of, and it is struck
> rather than deleted because the prediction it licensed is what the pilot tested.** `ε` is not
> driven by sampling noise here. It is driven by the injected bias — the score equation
> `P₀[w_a(Q̄*_a − Q̄_{0,a})] = 0` has to remove it — so the step is `O(n^(−α))`, *exactly* the
> order of the injection, and at the shape this design used to inject it removed 98.6% of it.

So the plug-in remainder above is **not** the estimator's bias, and the design needs a second
declared coefficient rather than a second reading of the same one. The two are one expression at
two regressions:

```text
R_2(Q-hat)  = P_0[ (ĝ − g_0)/ĝ · (Q̂ − Q̄_0) ]        the plug-in remainder,  coefficient c
R_2(Qbar*)  = P_0[ (ĝ − g_0)/ĝ · (Q̄* − Q̄_0) ]       the estimator's bias,   coefficient b
```

and [the repair](#the-repair-and-what-would-say-each-half-of-it-is-wrong) is what makes `b` a
declared number. `exact_remainder` integrates the first and `exact_targeted_remainder` the second,
the latter by solving the population score exactly rather than linearising it.

`R_remaining` — the doubly-robust curve's own remainder — needs `P₀D̂` at the **fitted** reduced
regressions, so it needs their values on covariates no fold trained at. That is the fold-retained
nuisance object §5 puts in Tier 2, and **C2 landed it as `DRTMLE(evaluation=…)`**: an independent
draw carried through the fit, one copy of every nuisance per outer fold, moved by the same
targeting steps the fitted arrays take. It is available at *either* tier — `--evaluation-n` is the
knob — and [the remainder section](#the-remainder-item-13) says what it computes and what it
approximates.

## Sizes, replicates and cost

`600 / 1,200 / 2,400`, three because two are suggestive and three carry a rate. Fifty replicates per
cell per size for the **pilot**; the frozen study wants 250 at minimum and 500 if the budget
reaches, at which a coverage estimate's Monte Carlo standard error is `0.014` and `0.010`. Changing
sizes or counts *after* seeing coverage is permitted only as a new experiment, documented as one.

**The cost was re-timed rather than inherited**, which the roadmap asks for before C is re-scoped:
it had been costed from a 43s `DRTMLE` fit, measured before piece B1b and before the exit criterion
item 7 replaced.

| measurement | on a four-core sandbox container |
| --- | --- |
| `bench_drtmle.py --processes nonlinear --sizes 1200 --seeds 2 --jobs 1` | 2 fits in 11s, **5.6s median per fit** — against the **43s** on record, at the same size |
| the same at `--sizes 400` | 2 fits in 33s, **16.4s median** — *slower* at fewer rows, as `test_drtmle_fit.py` already records: noisier nuisances make the loop longer |
| this harness (Tier 1), `--sizes 300 --replicates 4 --jobs 2` | 8 draws / 16 fits in 47s, **1.2s median per fit** |
| this harness, `--sizes 600 1200 --replicates 6 --jobs 2` | 24 draws / 48 fits in 117s, 1.2s median |

The `43s → 5.6s` row is the one that re-scopes C, and it is a factor of 7.7 at the size the stale
figure was measured at — which is [what B2b's dispatch predicted](investigation-log.md#the-exit-distribution-under-the-rule-that-is-actually-in-force)
when it put the corrected exit criterion at *"a seventh of the wall clock"*.

A Tier-1 fit is cheap because the primary nuisances are function evaluations rather than learner
fits; what it pays for is the alternation. So the pilot — 2 cells × 3 sizes × 50 replicates, 300
draws and 600 fits — is well under an hour on a runner and is still too much for the sandbox. It
runs from `.github/workflows/drtmle-coverage.yml`, dispatch-only, a matrix over the cells; the
per-replicate rows travel as an artefact, since `benchmarks/results/` is generated output and a file
from a two-core runner reads as a fact about the package rather than about that box.

**Tier 2 was expected not to be this cheap, and it is.** Its nuisances are fitted, which is what
the 43s was measuring — so C2 re-timed before re-scoping, exactly as this did, and the answer is
that the additive smoother is cheap:

| measurement | on a four-core sandbox container |
| --- | --- |
| a **tier-2** `DRTMLE` fit, `q-drift` at `n = 600`, with a 2,000-row companion | **5.4s** |
| the same at `g-drift` | **7.4s** |
| the tier-2 harness, `--sizes 300 --replicates 2 --jobs 1 --evaluation-n 800` | 2 draws / 4 fits in 9s, **1.7s median per fit** |

So the frozen study is affordable at either tier and `drtmle-coverage.yml`'s 300-minute cap is
generous rather than tight. The companion is what the remainder columns cost, and it is a
prediction per fold per nuisance per round with no further learner fit — a few seconds at the
pilot's evaluation size, and it scales with `--evaluation-n` rather than with `n`.

**E1 measured that scaling rather than reasoning about it, and it is flatter than the sentence
above implies.** Two to four times the companion rows costs two to six per cent of the fit at
either tier — `20.6s` against `19.8s` in the worst cell measured, Tier 2 `g-drift` at `n = 2,400`
with 4,096 rows against 2,000. The kernel smoother's cost is dominated by the rows it *trains* on
rather than by the rows it predicts at, so a finer evaluation rule is close to free and the grid
could be taken several rungs beyond what any cell has needed. [The
table](investigation-log.md#what-it-cost-which-is-nearly-nothing).

## The rules

Frozen in [the validation plan's §5](validation-plan.md#the-decision-rules-frozen-before-the-dispatch)
and **not restated here**, deliberately: a rule written down twice is a rule that can differ from
itself. What the harness implements of them, and where:

- *compatible with 0.95* is the plan's own Wald form, `|coverage − 0.95| ≤ 1.96√(p(1−p)/M)`, with a
  **Wilson** interval reported beside it — at 0.98 over 50 replicates a Wald interval reaches above
  1, and an upper limit that cannot be attained makes the verdict satisfied by construction on the
  high side;
- the shortfall is **paired on the draw**. Both estimators fit the same draw at the same injected
  nuisances, so the per-replicate difference of coverage indicators has a standard error of order
  `1/M` rather than `1/√M` where the two agree — which is what makes gate 2's `0.05` on the
  *difference* resolvable at 250 replicates instead of at an order of magnitude more;
- **the primary coverage number counts an algorithmically invalid fit as a failure of the
  procedure.** Coverage over the surviving fits is conditional on a non-random subset selected on a
  diagnostic correlated with the fit having gone wrong; reporting that as *the* coverage is the same
  class of error as reporting a per-protocol analysis as intention-to-treat. The other two
  accountings — excluded with the exclusion rate beside it, and the rate as its own outcome — are
  printed next to it;
- a fit that **raised** is in the denominator, recorded with what it raised;
- the invalid share is split into `identity` and `score` **columns**, because gate 1 asks for
  the two apart — clause 2 is *zero state-identity failures* and clause 3 is *every required
  score negligible* — and the tolerance clause 3 is read at is printed in the run banner rather
  than left to a default nobody wrote down;
- a **mixed** cell is reported pooled, with the two contract populations beside it as
  description and never as a verdict. That is §5's fourth rule, it is C3's decision taken
  before the dispatch, and [what forced it](#what-tier-1-already-showed-and-it-is-not-what-the-design-expected)
  is below.

## Tier 2: a prescribed *rate* rather than a prescribed sequence

`benchmarks/drtmle_tier2.py`. The same two cells and the same base law, with **both nuisances
fitted** — which is what makes this the demonstration and Tier 1 not. The trap the roadmap
records applies here and only here: `tests/e2e/test_double_robustness.py`'s "correct" cell is an
*oracle*, which makes `R₂` exactly zero and a plain `TMLE`'s interval already valid, so the gap
this study is about opens only where the good nuisance is estimated.

| cell | `Q̂` | `ĝ` |
| --- | --- | --- |
| **`q-drift`** | an oversmoothed additive kernel regression, consistent at `O(h_n²)` | a logistic GLM on `{W2, W3}`, whose limit is not `g₀` anywhere |
| **`g-drift`** | per-arm GLMs on subsets, whose limits are not `Q̄₀` | an oversmoothed additive kernel smoother of the arm indicator |

### The committed smoothing sequence

```text
h_n = c_h · n^(−β),    β = α/2 = 0.125,    c_h = 1.15
```

applied **one covariate at a time** — an additive backfit of one-dimensional Nadaraya–Watson
smoothers. A local-constant bias is `O(h²)`, so halving `α` is what makes `R₂` drift at `α`: the
two tiers then share the rate of the **remainder**, which is what they have to share to be about
one regime. The nuisance's own `L₂` error falls at `0.125`, which is a genuinely slow learner and
is the point.

**Two design decisions here are findings rather than preferences**, and both are §5's own
inner-product trap arriving through a new door.

**Not a regressogram, which is what §5 names.** A regressogram's bias oscillates in sign within
every bin, so its `L₂` norm is `O(B⁻¹)` while its *inner product with a smooth weight* is `O(B⁻²)`
— and the remainder is an inner product. Matching a declared remainder rate with one therefore
needs a bin count large enough that the fit is variance-dominated at the sizes this study reaches,
and its remainder is then sampling noise rather than a drift. A local-constant bias is
`h²[½∇²m + ∇m·∇log p]`, smooth and single-signed against a monotone weight, and no cancellation is
available to it. §5's list is illustrative; what it asks for is a sequence chosen in advance.

**Additive rather than a product kernel**, and that is the curse of dimensionality rather than a
modelling assumption. A four-dimensional product kernel wide enough to be bias-dominated at
`n = 600` smooths over essentially the whole covariate space — measured at an `L₂` error of `1.81`
against an outcome standard deviation of `1.75`, which is not a slow learner but a broken one —
while one narrow enough to be a regression has a variance of the same order as its bias. One
dimension at a time has variance `O(1/(nh))` and is bias-dominated at a bandwidth that still
resolves the function. It leaves the bias formula the coefficient is committed from unchanged.

### The committed calculation, and what it is a *prediction* of

`c_h` is the one number chosen to hit a target rather than derived: it is sized so `q-drift`'s
predicted `c_ATE` lands at Tier 1's committed `0.40`, since `c_a ∝ c_h²` and the two tiers are only
comparable if their drifts are.

| cell | `c₁` | `c₀` | `c_ATE` | `‖drifting‖` at 600 → 2,400 | `‖wrong‖` |
| --- | --- | --- | --- | --- | --- |
| `q-drift` | `+0.195` | `−0.193` | `+0.389` | `0.383 → 0.271` | `0.073`, fixed |
| `g-drift` | `+0.205` | `−0.205` | `+0.410` | `0.023 → 0.017` | `1.28`, fixed |

The arms' coefficients have **opposite signs in both cells**, so `c_ATE` is a sum of magnitudes and
cancellation in the contrast is impossible rather than merely unlikely — Tier 1 gets that by giving
its arms opposite signs by hand, and here it falls out of `b₀ = −b₁`.

**And it is a prediction rather than an identity**, which is the honest difference between the
tiers. Tier 1 *normalises* its injected shape so the coefficient comes out at a declared number;
here the estimator's bias is what it is, and the design's number is what the run is read against.
On one draw at `n = 600` the measured `n^α R₂` came out at `0.407` against the predicted `0.389`
(`q-drift`) and `0.370` against `0.410` (`g-drift`) — which is the check §5 asks for in place of
inferring the regime from an `L₂` rate, and the pilot is what turns one draw into a number.

**On this law a subset model's error has mean zero at every arm**, because the covariates are
independent standard normals. So the untargeted plug-in contrast is unbiased and the whole of what
a coverage shortfall can come from is `R₂`. That is the cleanest separation this design could have,
and it is worth saying because it is the *opposite* of Tier 1's situation, where
`G_DRIFT_ARM0_RATIO` exists to stop an error identical at both arms making the contrast
accidentally right.

## The remainder, item 13

`benchmarks/drtmle_remainder.py`, and `DRTMLE(evaluation=…)` beneath it. Three columns, and they
are not the same kind of number.

**`R_remaining` is exact given the companion.** `P₀D̂` is the population mean of the *fitted*
doubly-robust curve, which needs the curve as a function of `(W, A, Y)` — and an array of
out-of-fold predictions defines one nowhere. §5 refuses `P_nD̂` in its place by name: that is what
targeting drove to zero. The companion supplies the function by evaluating every fold's nuisances
at an independent draw and moving them by the **same targeting steps** the fitted arrays take.

**The fold convention, which §5 requires be documented rather than discovered:**

```text
P₀D̂  =  Σ_k (n_k / n) · E₀[ D̂^(k)(O) ]
```

with `n_k` the rows fold `k` holds out — the estimator's own fold weighting, not a uniform one.

**`R₂` at the fitted nuisances is exact too**, and is the regime-entry column Tier 2 gets in place
of Tier 1's quadrature over a prescribed sequence. It is checked against that quadrature at Tier 1,
where both are computable: two routes to one population integral sharing no code, which is the
strongest check available here.

**`R_Q` and `R_g` are approximated, and what is approximated in them is on the face of the module.**
The branch *sums* need fewer limits than the terms do — writing out `R₃ + R₄` and `R̃₅ + R̃₆`, the
univariate limits `Q̄_{0,r}`, `g_{1,0,r}` and `g_{2,0,r}` **cancel** — and what is left is the
fitted reductions, which the companion has exactly, plus the two `0n` limits, which are population
conditional means of computable quantities given computable scalars and so are quadratures rather
than fits. Each is estimated by a binned average over the evaluation draw at **two bin counts**,
and the difference between them travels beside the column as its own error; a branch smaller than
that error is reported as `-` rather than as a number.

**The empirical-process terms `M₁` and `M̃₂` are refused by name.** They are `(Pₙ − P₀)` of a
difference of estimated curves, and under the fold convention above `Pₙ` and `P₀` are taken at
different renderings of the nuisances — out of fold on the fitting sample, fold-conditional on the
evaluation draw. There is no single-sample expression that is both. What is reported is the
second-order half of each branch, which is the half gate 1's clause 4 is about: an empirical-process
term is `o_p(n^(−1/2))` under the Donsker and `L₂` conditions §5 lists and carries no product of
nuisance errors to cancel against.

**One property of the columns matters for how the study reads them.** `P₀D̂` is a quadrature, so its
error is `sd(D)/√m` and it lands *directly* in a replicate's `R_remaining`: at `m = 1,500` that
error is `0.026` against a remainder of order `0.007`, so a single draw's column is mostly noise.
The harness draws an **independent** evaluation sample per replicate, so it averages down across
them, and every entry in the remainder table carries its Monte Carlo standard error.

### The rule that quadrature is taken by, which E1 changed

**Averaging down across replicates is not the same as being small**, and C3c is where the
difference cost something: the draw's error is inside the spread every Monte Carlo error in the
remainder table is computed from, so `1.427 ± 0.091` is a number about the estimator *and* the
instrument together. [E1](../roadmap.md#what-e1-landed-and-what-e1b-withdrew) separates them, and
[the specification's rule](validation-plan.md#evaluating-pd-which-is-not-automatic-for-a-cross-fitted-fit)
is where the change and its written reason live. Three things belong here, because they are
properties of *this design* rather than of the specification.

**`--quadrature-points` is the lever and the i.i.d. draw is still the default.** The dispatch table
below is not edited: `evaluation_n = 2000` is what C3c ran and its four artefacts reproduce under
it. Which rule produced a row is now a field on the row (`companion_rule`, `companion_rows`), so a
later run and C3c's are distinguishable without reading an invocation.

**On this law the deterministic rule is exact in two of three coordinates.** Both cells are drawn
from `linear_dgp`, whose outcome is gaussian with additive error and whose treatment is binary —
so `E₀[Y | A, W] = Q̄₀(A, W)` closes the `Y` integral and a two-term sum closes the `A` one, leaving
a Sobol quadrature in `W` on the same grid `truth()` uses. A cell drawn from a law with a nonlinear
outcome link would need that paragraph rewritten rather than reused, and
`benchmarks/drtmle_remainder.quadrature_frame` refuses such a law by name rather than integrating
it.

**What the ladder measured, and what it did not.**
`benchmarks/drtmle_companion_grid.py` reads a ladder off one fit per draw and compares it against
the i.i.d. rule at C3c's own `m = 2,000`. The tables are in
[the investigation log](investigation-log.md#what-the-e1-ladder-measured), with the retraction
attached to each: what they show is that the across-draw spread of `√n R_rem` is several-fold
smaller under the grid than under the draw, which is two measured standard deviations. What they do
**not** show is how that difference apportions between the rule and the estimator — E1 read
`1 − s²_grid/s²_draw` as that share and it is not one — nor how large the grid's own error is,
which a successive difference between rungs does not bound. Both are
[E1b's](../roadmap.md#what-e1b-measures).
**None of it makes C3c's flat column false, and none of it says whether the decline resolves** —
which is a rate, and rates are read at E5. The remaining spread is the estimator's own second-order
sampling variation; how it compares against a decline is exactly the measurement E1 declines to
substitute for. What it does is make E5's reading of the same column a reading about the estimator.

## What Tier 1 already showed, and it is not what the design expected

Item 25's witness landed with this piece — `CorrectionCheck.contract` and its three columns — and
its first run found something the record's medians could not show. This is the finding, with its
numbers, and it is a **scope** measurement rather than a defect: every fit below passes its score
check and every state identity holds.

**A sixth to a third of well-overlapped draws exit outside the theorem-backed contract**, and the
initial mechanism has nothing to do with it. Over six draws per cell:

| cell | `n` | bound-active | worst `clip share` | min `margin` | min `gr1 margin` |
| --- | --- | --- | --- | --- | --- |
| `q-drift` | 600 | 1/6 | 0.0000 | **0.0e+00** | 0.216 |
| `q-drift` | 1,200 | 2/6 | 0.0000 | **0.0e+00** | 0.406 |
| `g-drift` | 600 | 2/6 | 0.0000 | **0.0e+00** | 0.263 |
| `g-drift` | 1,200 | 0/6 | 0.0000 | 1.5e-01 | 0.344 |

The initial mechanism never clips and `g_{r,1}` stays well interior. What is active is the **exit
margin**, at exactly zero — the targeted `g*` sitting against a truncation bound.

**And the cause is not positivity.** Two draws of `q-drift` at `n = 600`, same settings, same law:

| | pinned (data `368974633`, fold `403478673`) | interior (data `2002320325`, fold `4034082052`) |
| --- | --- | --- |
| initial `ĝ(1\|W)` | [0.3464, 0.8631] | [0.4195, 0.8688] |
| `g_bounds="auto"` | (0.03191, 0.96809) | same |
| targeted `g*` | **[0.031910, 0.968090]** — both bounds attained | [0.155000, 0.688179] |
| mechanism `epsilon` | **24.47** | **0** |
| q99 \|`Q_r/g*`\| | 0.00734 | 0.0346 |
| `margin` | 0.0 exactly | 0.131 |

Equation (9)'s clever covariate is `Q_r/g*`, and `Q_r = Q̄₀ − Q̄*` **vanishes where the outcome
regression is right** — which in `q-drift` it asymptotically is. So the score `Pₙ[H₉(1_a − g*)] = 0`
is being solved against a covariate whose 99th percentile is `7e-03`, and its root is an `epsilon`
of `24` on the logit scale, which drives rows to *both* bounds on a draw whose initial mechanism
never leaves `[0.35, 0.86]`.

**That is the mirror of [item 4](../roadmap.md#limitations-recorded-rather-than-fixed).** Item 4:
`g_{r,2}` vanishes where the mechanism is right, so equation (10)'s covariate is worst conditioned
on the fits anybody wants. Here: `Q_r` vanishes where the outcome regression is right, so equation
(9)'s covariate is worst conditioned on exactly the off-diagonal cell in which `Q̄` is the
consistent nuisance — the cell the variant exists for. The two are one observation seen from either
end, and the second half of it had not been written down.

Three consequences, and none of them is that a fit is wrong:

1. **item 25's second bullet is right about the initial mechanism and not established about the
   exit.** It reads a `clip share` of `0.000` and a `margin` of `0.11` to `0.20` off the B2b
   dispatch and concludes that an inactive truncation is *"the ordinary case at the sizes this
   variant runs at"*. Those margins are **medians over twelve draws**, and a minority at exactly
   zero is invisible to a median. The share is what has to be reported, and it is now a column.
2. **gate 1's clause 0 is a share per cell, not a label per cell.** A study's cells will be *mixed*,
   so the honest reading is a proportion of each cell's coverage number that is evidence about the
   constrained rendering — which is a reporting decision C3 has to take before its dispatch, and
   which this page flags rather than takes.

   **C3 has taken it**, before its dispatch as clause 0 requires, and it is [§5's fourth
   operational rule](validation-plan.md#four-rules-that-make-the-gates-operational): the pooled
   number stays primary and is what clauses 5 and 6 read, the share is what clause 0 reads, and
   the two contract populations are reported *beside* them as description — because the label is
   a post-fit property of the draw, so selecting on it conditions on a non-random subset exactly
   as excluding invalid fits would. `stratum_rows` is the table and the rule is in §5 rather than
   in two places.
3. **the contract label must stay out of the verdict**, which is why `CorrectionCheck.passed` does
   not read it. On this evidence a fit that is *sound in every way the package can check* is
   routinely outside the theorem's scope, and folding the label into a pass/fail would report a
   sixth of well-behaved draws as broken and send a reader round a problem that is not there.

Six draws per cell is a share and not a rate; the pilot is what turns it into one. What is *not*
statistical is the pinned draw above — one fit, exhibited with its arithmetic, which is the right
instrument for a mechanism rather than for a frequency.

## What the pilot measured

*Four dispatches of [`drtmle-coverage.yml`](../../.github/workflows/drtmle-coverage.yml), both
tiers × both cells, 50 replicates per cell per size at `600 / 1,200 / 2,400`, seed `20250801`,
`--evaluation-n 2000`, commit `5a474e8`, runs
[30942738792](https://github.com/esbraun/cleverly-tmle/actions/runs/30942738792) (tier 1) and
[30942745702](https://github.com/esbraun/cleverly-tmle/actions/runs/30942745702) (tier 2). 600
fits. Wall clock 696s to 1,243s a job at `jobs=2`; median **2.1s** per fit at tier 2, so the
re-timing above stands and a 250-replicate dispatch is affordable.*

**A pilot is not evidence for any gate.** Fifty replicates carry a Monte Carlo standard error near
`0.03` on a coverage estimate, which is enough to check that a design entered the regime it
committed to and not enough to resolve `0.95` against `0.88`. What is below is read as a check on
the *design*, which is what §5 runs a pilot for.

### Tier 1 has no gap for the variant to close, and the reason is the design's

| cell | `TMLE` ate coverage | `DRTMLE` | `TMLE` `√n` bias | `DRTMLE` `√n R_rem` |
| --- | --- | --- | --- | --- |
| `q-drift` | 1.000 / 1.000 / 0.980 | 0.920 / 0.920 / 0.940 | −0.22 / −0.56 / +0.11 | +0.28 / +0.51 / +0.39 |
| `g-drift` | 0.920 / 0.940 / 0.900 | 0.940 / 0.940 / 0.860 | −0.31 / −0.40 / +0.30 | +0.04 / +0.14 / +0.27 |

The injection is exactly what it committed to — `n^α R₂` reads `+0.4000` at every size — and the
plain interval does not under-cover anywhere. It **over**-covers, at `se` ratios up to `1.52`.

**The prediction and the column are about two different quantities**, and
`benchmarks/drtmle_tier1_bias.py` is what settled it rather than an argument. Evaluating one
expression at two regressions on the same rows of the same fits, `q-drift`, 24 draws:

| `n` | mean bias | `R₂(Q̂)` | `R₂(Q̄*)` | declared |
| --- | --- | --- | --- | --- |
| 600 | −0.00362 ± 0.02032 | +0.08082 | −0.00391 | +0.08082 |
| 1,200 | −0.00543 ± 0.00996 | +0.06797 | +0.01052 | +0.06796 |
| 2,400 | +0.00833 ± 0.00905 | +0.05716 | −0.00154 | +0.05715 |

`ψ̂ − ψ₀ = (Pₙ − P₀)D* + R₂(Q̄*)` and the first term is mean-zero across draws, so the bias has to
track the **targeted** remainder — and it does, at all three sizes and within Monte Carlo error.
`exact_remainder` integrates the **initial** one, which its docstring says and which is the honest
name for it; what was wrong is the sizing paragraph's reading of it. The `R₂(Q̂)` column reproduces
that quadrature to five decimals, which is what says both arms of the comparison are computed
correctly rather than merely differently.

So the consequence is structural rather than a constant to re-tune: **Tier 1 injects its drift into
`Q̂`, and the fluctuation's own free parameter absorbs it.** No choice of `c` makes that tier
produce a coverage gap, because the perturbation never reaches the estimate.

### Tier 2 has a gap, in one cell, under a regime it did not commit to

| cell, `n = 2,400` | `TMLE` | `DRTMLE` | paired difference | `√n R_rem`, 600 → 2,400 |
| --- | --- | --- | --- | --- |
| `q-drift` | 0.540 | 0.760 | **+0.220 ± 0.072**, resolved | 1.54 → 1.42 → 1.19 |
| `g-drift` | 0.700 | 0.740 | +0.040 ± 0.040 | 4.17 → 3.91 → **5.07** |

`q-drift` is the shape the variant exists for and `g-drift` is not: at `n = 600` there `DRTMLE` is
**worse** than `TMLE`, by `−0.120 ± 0.055`, resolved. And the realised `n^α R₂` at the fitted
nuisances is `0.59`–`0.68` against the committed `0.389`/`0.410`, drifting upward rather than
settling, so neither cell is in the regime that was frozen.

**Read against the gates, this pilot fails gate 1** — clause 4 (`g-drift`'s corrected remainder
*rises* with `n`), clause 5 (`q-drift`'s `DRTMLE` `se` ratio is `0.817`) and clause 6 (`0.760` and
`0.740` at the largest size, both incompatible with `0.95`). Gate 2's shortfall is met in `q-drift`
alone.

### Three things worked, and they are worth recording as such

- **`identity` is `0` across all 600 fits**, so gate 1's clause 2 holds and the column C3 added to
  read it earned itself on its first run. Every `score` failure is a fit that did not converge,
  which is a different finding and now says so.
- **The bound-active share, now a rate rather than a share of six draws.** C1's witness read
  one-to-two of six and this page carried that as *"a sixth to a third"*; over 50 draws a cell it is
  lower everywhere and it **falls with `n`** in three of the four arms, which is what item 25's
  asymptotic argument says it should do:

  | tier, cell | `n = 600` | `n = 1,200` | `n = 2,400` |
  | --- | --- | --- | --- |
  | 1, `q-drift` | 10/50 = **20%** | 5/50 = 10% | 5/50 = 10% |
  | 1, `g-drift` | 5/50 = 10% | 1/50 = 2% | 0/50 = **0%** |
  | 2, `q-drift` | 0/50 = **0%** | 0/50 = 0% | 1/50 = 2% |
  | 2, `g-drift` | 3/50 = 6% | 3/50 = 6% | 3/50 = 6% |

  The initial mechanism clips essentially nowhere (`worst clip share` `0.0000` in eleven of twelve
  cells, `0.0017` in the twelfth) and `g_{r,1}` stays interior throughout (`min gr1 margin` `0.117`
  to `0.343`), so this is the **exit** margin alone — C1's diagnosis, at a hundred times the draws.
  Cells are mixed, which is why §5's fourth rule exists. The strata never separated by more than
  Monte Carlo error at these counts: the largest gap is `g-drift` Tier 2 at `n = 1,200`, `0.894`
  theorem-side against `0.667` bound-active on **three** draws, whose Wilson interval is
  `[0.208, 0.939]`. That is a stratum too small to read, reported rather than interpreted.
- **The invalid share is `0.000` to `0.060`**, so the proposed 2% threshold is exceeded once, at
  `g-drift` tier 2 `n = 600`.

**What this leaves open is a design decision and not a defect.** No score-check or state-identity
failure appeared anywhere, and nothing here is evidence against the estimator. What the pilot
falsified is the *instrument's* premise: Tier 1 cannot produce the gap it was built to produce, and
Tier 2's regime is not the one that was committed. Both bear on whether the 250-replicate dispatch
would measure the thing it is for, and §5 permits the design to move **before** that run and not
after it.

> **This whole section is kept as the pilot read it, and [the repair](#the-repair-and-what-would-say-each-half-of-it-is-wrong)
> below is what the numbers became.** Three of its readings have since moved — the factor was
> `436` and not twenty, `g-drift`'s corrected remainder is not rising, and Tier 2's realised
> coefficient is stable rather than drifting — and each is corrected where it is used rather than
> edited away here, because a pilot's value is that it was run before anyone knew the answer.

## The repair, and what would say each half of it is wrong

**The decision is to fix the design rather than to run it or to abandon it**, and [§5's
rules](validation-plan.md#the-decision-rules-frozen-before-the-dispatch) permit exactly that:
they may be changed *before* the final run, with a written reason. This section is the written
reason. It is deliberately two halves, because the two tiers failed differently and one repair
does not cover both.

### Why the drift vanished, in algebra rather than in prose

This is the part a future reader needs first, because it says what any candidate injection has to
satisfy. Write the fluctuation's score equation in population form — it is what targeting solves:

```text
P_0[ 1{A=a}/ĝ_a · (Y − Q̄*_a) ] = 0   ⟹   P_0[ (g_0/ĝ)(Q̄* − Q̄_0) ] = 0
```

and put it beside the remainder the estimator's bias actually is:

```text
R_2(Q̄*) = P_0[ (1 − g_0/ĝ)(Q̄* − Q̄_0) ] = P_0[ Q̄* − Q̄_0 ] − 0
```

So **`R₂(Q̄*)` is the plain mean offset of the targeted regression**, and the score equation is
precisely a constraint that drives a `g₀/ĝ`-*weighted* offset to zero. That is the same statement
as `ψ̂ − ψ₀ = (Pₙ − P₀)Q̄* + P₀[Q̄* − Q̄₀]`, and it is why the measured bias tracked `R₂(Q̄*)`.

Now carry the injection through. With `Q̄* − Q̄₀ = n^(−α)h + ε·s`, where `s` is the fluctuation's
own direction, the score fixes `ε` and leaves

```text
R_2(Q̄*) = n^(−α) · ( P_0[h] − P_0[(g_0/ĝ)h] · P_0[s] / P_0[(g_0/ĝ)s] )
```

**which vanishes exactly when `h` and `s` carry the same weighted-to-unweighted ratio.** The design
chose `h_a ∝ (g₁ − g₀)/g₁` to make `c_a = P₀[(g₁−g₀)/g₁ · h_a]` large — the right condition for the
*plug-in* remainder and **no condition at all on the bracket above**.

> **This paragraph used to end *"the display above is derived from the measurement rather than
> verified end to end"*, and said to treat it as a hypothesis. It is now verified**, which is what
> C3b ran before touching any injection. The bracket is a **linear functional of `h`** against a
> computable weight — write `w_a = g₀/ĝ`, `κ_a = P₀[s_a]/P₀[w_a s_a]` and `v_a = 1 − κ_a w_a`, and
> the display above is exactly `b_a = P₀[v_a h_a]`. That is what makes the design repairable
> rather than only diagnosable: `b_a` is a coefficient a design can be *built* to hit, the same
> way `c_a` always was.
>
> Measured over 24 draws at three sizes in both cells, the fitted `ε` agrees with the population
> one within a standard error everywhere; in `g-drift`, where the measurement resolves, the
> measured `R₂(Q̄*)` reads `+0.00380 ± 0.00084` against a predicted `+0.00471`. So the population
> arithmetic describes the fits. `benchmarks/drtmle_tier1_bias.py`'s second table is the
> decomposition, and the accounting closes as an identity: `b_a = c_a + ε̃_a P₀[u_a S_a]`.

**And the ratio was not twenty.** That reading was a noise-floor artefact — at the old shape the
measured `R₂(Q̄*)` was consistent with zero at 24 draws, so the pilot could bound it and not
measure it. The exact coefficients are `b_ATE = 0.00092` against `c_ATE = 0.40` in `q-drift`, a
factor of **436**, and `0.0259` in `g-drift`, a factor of 15. The absorbed share is `98.6%` and
`100.9%` at `q-drift`'s two arms, `92.5%` and `95.4%` at `g-drift`'s.

**A second thing the pilot had no column for, and it is the sharper one.** The design gives its
arms opposite signs so that `c_ATE` is a *sum* of magnitudes and cancellation in the contrast is
impossible. That property **does not survive targeting**: at the old shape `b₁` and `b₀` came out
*both positive*, so `b_ATE` was a difference of magnitudes — `0.00056 − 0.00038` — and smaller than
either arm's. The no-cancellation guarantee was being enforced on the column that was not the
estimand's, which is the same mistake as the sizing, one level down.

### Tier 1: the drift has to survive targeting, not merely exist

**The repair, and it landed.** A second linear condition rather than a larger constant. Since
`c_a = P₀[u_a h_a]` and `b_a = P₀[v_a h_a]` are both linear functionals of the free shape, `h_a`
goes in the span of their two representers and a 2×2 **Gram** solve puts both at declared values at
once. The old design is the one-condition special case, so this is a generalisation rather than a
replacement — and the span is the minimum-norm choice, which matters because the injection has to
stay inside `Q_BOUNDS` after scaling.

**It cleared its acceptance test.** `benchmarks/drtmle_tier1_bias.py`, 24 draws at three sizes:
`R₂(Q̄*)` reads `+0.08429 ± 0.01129`, `+0.07897 ± 0.00833`, `+0.05547 ± 0.00471` against a declared
`+0.08082 / +0.06796 / +0.05715`, and the root-`n` bias reads **`+1.93 / +2.17 / +3.31`** where the
pilot read `−0.22 / −0.56 / +0.11`. That is the growing drift the design was built to produce, and
it is pre-flight condition 1 in both cells.

**The possibility this page refused to talk itself out of is closed, and by a measurement.** The
live alternative was that *no* injection into a single nuisance produces a first-order shortfall,
in which case Tier 1 is a remainder anchor and the repair is a scope correction. It is decided by
whether `v_a` is degenerate: `v_a` vanishes identically only if `w_a` is constant, i.e. only if
`ĝ_a ∝ g_{0,a}`, and if it did the fluctuation would reach every direction the design can inject.
Measured, `‖v_a‖ = 0.070` at both arms and the Gram's condition number is 16 and 80.
`tests/unit/test_drtmle_coverage.py` asserts both, and it is the test that would have failed if the
alternative were true.

**Where the alternative *does* bite is one cell, and the constraint is positivity.** `g-drift`
cannot hold `b_ATE = 0.40`: it perturbs a **probability** rather than a regression with a declared
support, the fluctuation absorbs 92–95% of what is injected there, and buying a surviving `0.40`
needs `ĝ` to reach `−0.16` at `n = 200` — at which `InjectedMechanism` raises rather than clipping.
Scanned against both pre-flight conditions, `0.10` is the largest value whose regime-entry column
stays put (`+0.0996 / +0.0994 / +0.0991`, within `0.5%`) with `ĝ` keeping a margin of `0.099`.

That is a **scope statement and it belongs on the face of the design**: a drift of `0.10` puts the
plain interval's shift at `0.19` to `0.27` standard errors, so a `TMLE` shortfall in `g-drift` is
`0.005` to `0.008` — real, and far below gate 2's predeclared `0.05`. So **`q-drift` is the cell a
shortfall is claimed in**, and `g-drift` is where `DRTMLE` is checked to hold nominal under a drift
and where the remainder is read off. The design note's "Tier 1 may be a remainder anchor" arrives
in one cell of two rather than in the tier, and it is a property of the estimand's setting rather
than of this instrument.

**One consequence worth knowing about.** Declaring `b = c`, which is what `q-drift` does, forces
`P₀[w_a h_a] = 0` — the injection is exactly orthogonal to the fluctuation's own score, `ε` is zero
in the limit, and the two remainder columns coincide. Measured, a fitted `ε` of
`+0.00026 ± 0.00183` and an absorbed share of `0.0000`.

### Tier 2: the constant, not the exponent

**The problem was never the same problem, and C3b's first finding here is that Tier 2 does not
suffer Tier 1's at all.** Its two coefficients agree to five figures — `b_ATE = 0.3895` against
`c_ATE = 0.3886` in `q-drift`, and equal in `g-drift` — because both of its error shapes are
**linear** in independent standard normals, so each has population mean zero, and the fluctuation's
step is driven by exactly that mean. In `g-drift` the score weight is identically one at the limit,
so `ε` is the mean of the outcome error and is zero exactly. That is why Tier 2 produced a coverage
gap in the pilot while Tier 1 could not: absorption is a property of whether the nuisance's error
is aligned with the fluctuation's direction, and a smoother's bias on a symmetric law is not.

**What was wrong was a reading, and correcting it changes which knob moves.** The pilot saw
`0.59`–`0.68` against `0.389`/`0.410` and read it as *drifting upward*, so the section this
replaces put the **exponent** in question. Re-measured at the targeted column over 12 draws at
three sizes, the realised coefficient is `+0.6242 / +0.5863 / +0.6173` in `q-drift` — a **spread of
0.06**, which is stable, at a ratio of `1.58×`. That is the design note's own *"a constant that is
1.5× can be re-normalised, a drifting coefficient cannot"*, landing on the first branch.

So `β` stays at `α/2`, which is what keeps the two tiers about one regime. **The obvious next move
is that `c_h` is the knob — the committed calculation is the `h²` leading term alone, `h(600)` is
`0.517`, so `h⁴` is not negligible and shrinking `h` should close the gap. That was run and it is
wrong**, which is the second thing C3b measured here:

| `c_h` | `h(600)` | predicted `b_ATE` | realised at `n = 600` |
| --- | --- | --- | --- |
| 1.15 | 0.517 | 0.3895 | `+0.6265` — **1.61×** |
| 1.00 | 0.450 | 0.2946 | `+0.5234` — 1.78× |
| 0.90 | 0.405 | 0.2386 | `+0.4549` — 1.91× |
| 0.80 | 0.360 | 0.1885 | `+0.3861` — 2.05× |
| 0.70 | 0.315 | 0.1443 | `+0.3195` — **2.21×** |

The ratio **rises** as the bandwidth falls, which is the opposite of an `h⁴` truncation error and
identifies the omitted term as **variance-side rather than bias-side**: both nuisances are fitted
on the same rows, so their estimation errors covary, and that covariance enters the remainder's
inner product without shrinking with `h`. So no bandwidth makes the leading-order prediction
correct, shrinking it makes the agreement worse, and `c_h` stays at `1.15`.

**What moves instead is the number the pre-flight reads against**, which §5 permits at the pilot
and only there. Tier 2 gains a `COMMITTED_B_ATE`, measured at a stated protocol, with the analytic
prediction reported beside it as the leading-order term it is — the two tiers already differ in
exactly this way, since Tier 1 *solves* its shape to hit a declared number and here the estimator's
bias is what it is. **This is not the shortfall being tuned for**: the drift is *stronger* than
predicted, not weaker, and `q-drift`'s `TMLE` covers `0.750 / 0.583 / 0.500` against `DRTMLE`'s
`0.833 / 0.917 / 0.917` — a gap far past gate 2's predeclared `0.05` either way.

**And condition 3 appeared not to fail.** The pilot's `g-drift` corrected remainder *rose*
(`4.17 → 3.91 → 5.07`); re-measured it read `+2.85 / +3.30 / +2.62`, and `q-drift`'s
`+1.48 / +1.23 / +1.26`. Both appeared to fall from the first size to the last, though at 12 draws
the Monte Carlo errors are `±0.42` to `±0.83` and the final study is what resolves them.

> **This reading did not survive the dispatch, and it is left standing with the correction beside
> it rather than rewritten.** [C3c](investigation-log.md#what-the-c3c-dispatch-measured) read the
> same column at 250 draws and got `4.13 / 4.12 / 4.83` in `g-drift` — back at the pilot's level,
> not falling. The `+2.85 / +3.30 / +2.62` was inside its own error of the pilot's numbers all
> along, which is exactly what the verdict table said by reading condition 3 `unresolved` rather
> than `pass`. The sentence to have written is *the rise is not resolvable at this count*; the one
> written was *the rise was not there to repair*. **`unresolved` is not a weak `pass`**, and the
> table's whole purpose is to keep those apart — the summary prose is where they got merged.

### What both halves have to clear before any 250-replicate dispatch

1. `R₂(Q̄*)` at the declared `n^(−α)b`, not `R₂(Q̂)` at `n^(−α)c` — the check the old design would
   have failed;
2. the realised `n^α R₂` at the fitted nuisances stable across the three sizes and near its
   committed value;
3. `√n R_rem` falling rather than rising in **both** cells.

None of these needs a coverage study, all three are minutes, and the reason to state them here is
that a study dispatched without them measures a design nobody has checked — which is what
happened, and what the pilot cost was small enough to catch.

**They are a table now rather than a paragraph.** `benchmarks/drtmle_coverage.py` prints them last,
one row per condition per cell with a verdict, and `.github/workflows/drtmle-coverage.yml`'s header
says to read it first. Conditions 1 and 2 are read on the plain `TMLE`, since that is the estimator
whose regime the design commits and whose interval a shortfall is claimed against; condition 3 on
`DRTMLE`, since it is item 13's. A run with no evaluation draw reports condition 3 as `-` rather
than as a failure — *not measurable* and *failed* must not read alike.

The one tolerance in it is stated as a rule and not taken from a result: §5 names no number, so a
quarter is written down once in the module and the verdict column says which was applied. Condition
3 failing stays a **finding rather than a fault in the design** — it is a condition of Theorem 1,
so the estimator would then be outside the assumptions its own guarantee needs at these sizes.

### What the pre-flight read, at 12 draws and three sizes in both tiers

*Both tiers, both cells, `600 / 1,200 / 2,400`, seed `20250801`. Tier 1 at `--evaluation-n 1200`
and Tier 2 at `1500`; 72 draws each, 355s wall clock at `jobs=2` for Tier 2. This is minutes
rather than a dispatch, which is the whole point of the conditions being what they are.*

| condition | Tier 1 `q-drift` | Tier 1 `g-drift` | Tier 2 `q-drift` | Tier 2 `g-drift` |
| --- | --- | --- | --- | --- |
| **1** bias at the committed `n^(−α)b` | `+0.4003 ± 0.0362` vs `+0.4000` | `+0.0979 ± 0.0029` vs `+0.1000` | `+0.6173 ± 0.0200` vs `+0.6100` | `+0.6773 ± 0.0637` vs `+0.6200` |
| **2** `n^α R₂` stable | `+0.430 / +0.453 / +0.400` | `+0.098 / +0.104 / +0.098` | `+0.624 / +0.586 / +0.617` | `+0.520 / +0.672 / +0.677` |
| **3** `√n R_rem` falling | `+0.51 / +0.52 / +0.93` | `+1.05 / +1.37 / +1.80` | `+1.48 / +1.23 / +1.26` | `+2.85 / +3.30 / +2.62` |

**Conditions 1 and 2 pass in all four cells**, and Tier 1's are the tight ones — `+0.4003` against
a declared `+0.4000`, on real fits at the largest size, which is the repair confirmed rather than
argued.

**Condition 3 is `unresolved` everywhere, and that is a reading rather than a verdict.** `P₀D̂` is
a quadrature whose error lands directly in each replicate's remainder and `√n` multiplies it, so
the Monte Carlo errors here are `±0.42` to `±0.83` — every reading is inside its own error of
every other. *Not resolvable at this draw count* and *failed* are different things and the table
says which it is; separating them is what the 250-replicate dispatch exists for, and it is
[item 13](../roadmap.md#what-is-still-open)'s number rather than this design's.

### What the study was dispatched as, written down before its first fit

*This subsection was committed **before** dispatch A and is not edited afterwards.* §5's rule is
that the study's inputs may be changed before the final run and not after it, and a seed chosen
once coverage is visible is a seed chosen for its answer — so both batches' inputs are here, in
advance, including the second batch's.

| input | value | why this value |
| --- | --- | --- |
| `tier` | `2` | the demonstration. Tier 1 hands the estimator a prescribed sequence, so its regime entry is true by construction rather than measured; the roadmap's C3c row is two dispatches and these are they |
| `cells` | `q-drift g-drift` | both. `q-drift` is the cell a gate-2 shortfall is claimed in and `g-drift` is where `DRTMLE` is checked to hold nominal under a drift, which is the scope statement the repair put on the design |
| `sizes` | `600 1200 2400` | three, because two are suggestive and three carry a rate |
| `replicates` | `250` | §5's frozen minimum, at which a coverage estimate's Monte Carlo standard error is `0.014`. Not 500: the pilot's paired shortfall was `+0.220 ± 0.072`, which 250 resolves several times over, and a job that reaches the 300-minute cap prints **no** table rather than a partial one |
| `seed` | `20250801`, then `20250802` | batch A is the harness default and the pilot's. The two `SeedSequence` streams are prefix-stable, so batch A shares the pilot's *data* seeds and not its fold splits — the split is part of the procedure whose coverage is being measured. Batch B is the independent second batch gate 1's clause 7 reads, dispatched **after** A completes |
| `evaluation_n` | `2000` | the workflow default, above the pre-flight's `1500`. It is item 13's quadrature and scales with itself rather than with `n`, and its error is what left condition 3 unresolved at 12 draws |
| `jobs` | `2` | each fit is single-threaded and two fit the runner |
| `rows` | `false` | the per-replicate record §5 asks for travels as the JSONL artefact the workflow uploads. Three thousand printed rows would push the ten summary tables out of a readable log |

**No code changes land before or between the two batches.** Gate 1's clause 7 asks whether the
qualitative conclusion reproduces, and an instrument built between A and B would make the two
batches runs of different code — which is the one thing a reproduction check cannot survive. The
comparison is read across the two dispatches' tables.

## What the study measured

*Both dispatches, exactly as the subsection above committed them.
[The numbers](investigation-log.md#what-the-c3c-dispatch-measured) are in the investigation log;
what belongs here is the reading and the gate readout.*

**The design worked and the estimator did not clear.** Those are two findings and this section
keeps them apart, because the first is what three revisions of this note were spent on and the
second is what the study was for.

**The design worked**: conditions 1 and 2 pass in all four cell-runs, `n^α R₂(Q̄*)` landing within
`0.95x`–`1.01x` of its committed value at the largest size in both cells and both batches. This is
the first of three attempts to enter the regime it named. And the gap the whole construction exists
to produce is there, in `q-drift`, at `+0.312 ± 0.031` and `+0.376 ± 0.033` paired on the draw at
`n = 2,400` — `TMLE` at `0.532` and `0.472` against `DRTMLE` at `0.844` and `0.848`.

**The estimator did not clear**, and **two** measured quantities account for it rather than one.
The first is `√n R_remaining`, flat in `q-drift` (`1.43 / 1.26 / 1.25`, and `1.28 / 1.19 / 1.17` in
the second batch) and not falling in `g-drift` (`4.13 / 4.12 / 4.83`, and `4.04 / 3.93 / 4.31`).
Theorem 1 assumes it negligible; at these sizes and at these reductions it is not. The second is
the interval's own width: `σ²ₙ` is the empirical variance of the estimated curve and **treats the
reduced regressions as known**, so their estimation error is in `ψ̂`'s spread and not in the
variance estimate — the `se ratio` of `0.903`, reproduced to the digit. The split of `q-drift`'s
eleven points at the largest size is about six to the first and five to the second.

**An earlier revision of this paragraph said "everything else follows from that", and the section
two below is where it stops being true.** The `se` shortfall has its own mechanism, the 99 invalid
fits have a third, and `cancel` at `1.99x` is the branch decomposition failing to separate the two
remainder terms rather than either failing to vanish. Four failing clauses, three mechanisms — and
attributing all four to the remainder is what would send the repair to the wrong one.

> **`cancel` is withdrawn as evidence for clause 1.4, and the clause still fails.** The ratio is
> `(|R_Q| + |R_g|) / |R_Q + R_g|` of two **binned** estimates, and what was reported beside them as
> their "error" is a movement between two bin counts — a successive-refinement difference, which is
> the statistic [E1b withdrew](../roadmap.md#what-e1-landed-and-what-e1b-withdrew) for the
> quadrature ladder and which [§5](validation-plan.md#reporting-r_q-and-r_g-separately) now
> withdraws here. Nothing bounds those two estimates' error, so a ratio built from them is
> **unread** rather than failed, and `192/250` is a stability count rather than a resolution one.
>
> **The verdict does not move**, and saying so is the point of writing this here rather than
> quietly deleting the number. Clause 1.4's first half — `√n R_rem` flat at `1.43 / 1.26 / 1.25`
> in `q-drift` and not falling in `g-drift` — reads a column with a *measured* replicate spread
> beside it, which is exactly what E1b landed. It carries the clause alone. A retraction that
> moved a verdict would be a retraction doing work it is not entitled to do, and `unresolved` is
> not a weak `pass`. What would make the second half readable is [E2](../roadmap.md#e-what-c3c-handed-back).

**That is a finding rather than a fault in the design, and this note said so before any number
existed**: *"Condition 3 failing stays a finding rather than a fault in the design — it is a
condition of Theorem 1, so the estimator would then be outside the assumptions its own guarantee
needs at these sizes."* It is now measured rather than anticipated, at 250 replicates and two
seeds.

**And the measurement is of a configuration, which is the sentence this note most needs to keep
straight.** The three reduced regressions were fitted by `glm` —
`benchmarks/drtmle_coverage.REDUCED_LEARNER`, named explicitly because `DRTMLE` would otherwise
hand the *injected* primary specification to them — and
[the concordance](theorem-concordance.md)'s `reduced regressions consistent` row reads *estimated,
unmeasured rates* and `unverified`, before this dispatch and after it.

> **A sentence here said Tier 2's two primaries are injected analytic sequences, and that is Tier
> 1's description rather than this one's.** `benchmarks/drtmle_tier2.py` **fits** both — an
> oversmoothed additive kernel regression and a subset GLM — which is what makes it the tier the
> demonstration is read from. The roadmap withdrew the same claim in its own copy; this one had
> stood. What it was load-bearing for is the sharper form of the point, and that does not survive
> it: the three reductions are not *the only* thing a learner fits at Tier 2. The weaker form is
> what the paragraph needs and is true — they are fitted, by `glm`, and their rates are unmeasured.

So the study establishes *this estimator, at this reduction, does not meet the condition at these
sizes*. It does not establish that the condition is unmeetable, and
[piece E](../roadmap.md#e-what-c3c-handed-back) is that distinction turned into work — starting
with an **oracle** reduction, which decides in one quadrature whether the learner is where the
remainder lives.

**`g-drift`'s reading is narrower than it looks and the entry column is why.** `DRTMLE` is
resolvably *worse* than `TMLE` at `n = 600` in both batches. But conditions 1 and 2 are read on the
plain `TMLE` by construction, and the same table's `DRTMLE` row reads `0.97x` and `0.92x` — the
correction removes 3–8% of the targeted remainder there against 70% in `q-drift`. So `g-drift` is a
cell the *plain* estimator's regime was entered in and the *corrected* estimator's was not, which
is a statement about how far this design reaches and not about the variant in its own regime.
[Lesson 17](investigation-log.md#what-the-sizings-got-wrong) is that distinction as a rule.

### The gates, read out clause by clause

*Read against [the rules frozen in §5](validation-plan.md#the-decision-rules-frozen-before-the-dispatch),
which were not changed after the numbers existed. Each row names the column it is read from, so a
reader can check the verdict against the log rather than take it.*

| clause | read from | reading | verdict |
| --- | --- | --- | --- |
| **1.0** contract frozen, cells inside it | contract table | every cell-run `BOUND-ACTIVE`, `1.2%`–`8.8%` of draws; initial clip share `0.0000`–`0.0017` | **mixed** — reported pooled, strata beside as description |
| **1.1** theorem concordance closes | [A1a](theorem-concordance.md) | closed, item 21 with it | pass |
| **1.2** zero state-identity failures | validity table, `identity` | `0` across all 6,000 fits | **pass** |
| **1.3** every required score negligible | validity table, `score`, at tolerance `1e-3` | `1–7%` of `DRTMLE` fits invalid in every cell | **fail** |
| **1.4** `√n R_rem → 0` in both cells, no cancellation | remainder table, `sqrt(n) R_rem`, `cancel` | flat in `q-drift`, not falling in `g-drift`. `cancel` reaching `1.99x` is **withdrawn as evidence** — see below — and the verdict does not move, because the first half carries it alone | **fail**, on `√n R_rem` |
| **1.5** `se ratio` in `[0.90, 1.10]`, largest size, both cells | coverage table | `q-drift` `0.903` / `0.903`; `g-drift` `1.157` / `1.156` | **fail** in `g-drift` |
| **1.6** coverage compatible with `0.95`, largest size, both cells | coverage table, `compatible` | `0.844` / `0.848` and `0.780` / `0.784`; `NO` in all four | **fail** |
| **1.7** reproduces in the second seed batch | batch B against batch A | every qualitative claim reproduces; `se ratio` and entry column to the digit | **pass**, and see the note below on which batch is the fresh one |
| **2.1** a `≥ 0.05` shortfall, difference excluding zero, `√n R₂` not vanishing | shortfall and remainder tables | `q-drift`: shortfall `+0.418` / `+0.478`, difference `+0.312 ± 0.031` / `+0.376 ± 0.033`, `√n R₂` `4.31` / `4.26` and rising | **pass** |
| **2.2** invalid-fit rate below its threshold | validity table, `invalid share` | over `2%` in ten of twelve cell-runs | **fail** |
| **2.3** computational cost acceptable | wall clock | `2.7s`–`4.7s` median per fit; a 250-replicate cell is 77–112 minutes on one runner | pass |
| **2.4** advantage persists in an applied stress setting | — | **not read here** | not read |

**So gate 1 fails at clauses 3, 4, 5 and 6, and gate 2 passes its clause 1 and fails its clause 2.**
`DRTMLE` is not cleared, and the honest one-line summary is that *the variant removes most of the
plain estimator's bias in the regime it was built for, and is left with a second-order remainder
that Theorem 1 assumes away and `n = 2,400` does not deliver*.

**One clause is worth reading twice, because the temptation is to soften it.** Clause 6 asks
whether `DRTMLE` attains nominal coverage, not whether it beats `TMLE`. It does not, in either
cell, at any size, in either batch — the best reading anywhere is `0.880` — and the `+0.376` in
the row above it does not change that. [Lesson 18](investigation-log.md#what-the-sizings-got-wrong)
is that as a rule.

**And clause 1.0 is the one whose reading is neither pass nor fail.** Every cell-run is mixed, at
shares that make the pooled number the estimator as shipped and nothing else the primary. Neither
stratum is quoted here as the theorem-backed estimator's coverage — doing so is
[stop-ship 15](../roadmap.md#stop-ship) with a number attached — and the two are printed in the
harness's stratum table for the one question the share cannot answer, which is whether the
populations behave differently. They do not visibly: `0.850` against `0.667` at `q-drift`'s largest
size in batch B, on 247 draws against 3.

**And the reading that clause earns is that the *design* did not meet its own intent.** The
harness's header says of these cells that *"the cells here are designed to be inside it; the column
is what checks that rather than assuming it"* — and the column came back mixed in all four
cell-runs. Doing the right thing with the result (pooled primary, strata as description, neither
quoted as the theorem-backed estimator's coverage) does not turn a mixed population into a clean
theorem reading. **One study cannot be both**, and
[E4](../roadmap.md#e-what-c3c-handed-back) is that split: a *theorem* design whose bounds are
inactive under a predeclared rule, and a *stress* design that keeps them active and says so.

**Two batches, and only one of them is a fresh confirmation.** The `SeedSequence` streams are
prefix-stable, so **batch A shares the pilot's data seeds** — drawing its own fold splits, which is
part of the procedure being measured, but not its own draws — and the pilot is what forced the
design change [C3b](#the-repair-and-what-would-say-each-half-of-it-is-wrong) landed. Batch B's
stream is fresh. The two agree on every qualitative claim, so this does not weaken the result; it
names which batch carries the weight. The design table said this before the dispatch and clause
1.7's verdict reads the two symmetrically, which is the gap this note closes. E5's dispatch uses
streams no pilot has touched, on either batch.

### Twice a coverage study here found no gap, and the pair is the thing to carry forward

The first is on the roadmap already: a pilot over the off-diagonal grid put `TMLE` and `DRTMLE` at
`0.958` apiece in one cell and `1.000` in the other, and the diagnosis was that a correctly
specified *parametric* nuisance converges at `n^(−1/2)`, so `R₂` is `O(n^(−1))` and the product
condition never binds. The whole Tier-1/Tier-2 construction exists to answer that. It returned "no
gap" a second time in C3a's pilot, for a different reason: the targeting step removing what was
injected.

**Both were the study failing to *enter* the regime rather than the estimator failing in it, and
both are now closed** — the first by this construction, the second by
[the repair](#the-repair-and-what-would-say-each-half-of-it-is-wrong). What a future agent should
carry forward is not the tally but the **shape** the two share, because a third instance would look
like neither: in each case a column that was exactly right sat beside a prediction that was wrong,
and in each case the resolution was that the two were about different quantities. A nuisance norm
is not a remainder; a plug-in remainder is not a bias. Where a design's own column agrees with its
own arithmetic and the fits disagree with both, the question to ask first is **which quantity each
of them is**, not what constant to change.

The instrument that now enforces this is the pre-flight, and its whole content is that a design's
number has to be checked against a *fit* before the expensive run and not after it.

**The third study found a gap, and the shape above is what it found *inside* the gap.** C3c's
`q-drift` produced `+0.312 ± 0.031` and `+0.376 ± 0.033`, so the tally stops at two. But `g-drift`
is the same shape one level down: the entry column was exactly right about the estimator it was
read on and said nothing about the other, and the resolution was again that two columns in one
table are about different quantities. Whoever reads this next should expect the shape to recur in
whatever the *next* expensive run is, rather than expect the tally to.
