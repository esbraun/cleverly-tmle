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

| cell | `α` | `c₁` | `c₀` | `c_ATE` | min \|c\| |
| --- | --- | --- | --- | --- | --- |
| `q-drift` | 0.25 | +0.2000 | −0.2000 | +0.4000 | 0.2000 |
| `g-drift` | 0.25 | +0.2520 | −0.1480 | +0.4000 | 0.1480 |

`q-drift` declares both arm coefficients; `g-drift` declares only the ATE's, and that is structural
rather than a shortcut — a binary treatment's mechanism has **one** free function, since the
estimator reads `ĝ(1|W)` off a classifier and takes the complement, so one perturbation determines
both arms and only their combination can be set. What the arm coefficients then come out at is a
finding, and `tests/unit/test_drtmle_coverage.py` holds both to a floor of `0.02`.

`c_ATE = 0.40` is sized from the drift a coverage number can resolve, and the sizing is **a
prediction the pilot checks**. With `σ_ATE` measured at `2.6` (a `se` of `0.106` at `n = 600` on one
injected fit), `bias/se ≈ n^(1/2−α)·c/σ` puts the plain interval's shift at `0.76`, `0.91` and
`1.08` standard errors at `600 / 1,200 / 2,400` — a `TMLE` coverage near `0.87 / 0.86 / 0.81`, so a
shortfall of `0.08` to `0.14`. That is comfortably clear of gate 2's predeclared `0.05` at every
size, and it is resolvable against 250 replicates' Monte Carlo error of `0.014`. **Provisional until
the pilot**, which is the one point at which §5 permits it to move.

The realised remainder, by quadrature:

| cell | `n` | `R₂` (ATE) | `n^α R₂` | declared `c_ATE` |
| --- | --- | --- | --- | --- |
| `q-drift` | 600 / 1,200 / 2,400 | +0.08082 / +0.06796 / +0.05715 | +0.40000 at all three | +0.4000 |
| `g-drift` | 600 / 1,200 / 2,400 | +0.08049 / +0.06769 / +0.05694 | +0.39837 / +0.39842 / +0.39853 | +0.4000 |

`q-drift`'s is exact at every size because its mechanism is fixed, so `R₂` **is** `n^(−α)c`;
`g-drift`'s carries the `o(n^(−α))` term its drifting mechanism contributes and closes on the
coefficient from below. The nuisance-error slopes are `−0.250` for the drifting nuisance and
`+0.000` for the misspecified one in each cell, which is the pair §5 asks for: a study reporting
only the shrinking norm could not tell a product going to zero because one factor does from one
going to zero because both do — and the second is the regime a plain interval is already valid in.

### What is integrated exactly, and what is not

`R₂` above is the **plug-in remainder at the injected sequence**, and it is exact because both
primary nuisances are prescribed functions of `W`. Two qualifications, both of which belong on the
face of the design rather than in a footnote.

The targeting step moves `Q̂` to `Q̄*` by `O_p(n^(−1/2))`, which is smaller than the injected
`n^(−α)` at every `α < 1/2`, so it leaves the drift's leading term where it is and changes the
remainder at the next order. The number above is therefore the *regime's* remainder and not the
realised fit's.

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
