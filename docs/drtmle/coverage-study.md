# The coverage study: the design, and what Tier 1 already showed

`DRTMLE`'s definition of done is one sentence — *a demonstration that the interval attains its
nominal coverage where a plain `TMLE`'s does not* — and [piece C](../roadmap.md#c-the-demonstration)
is that demonstration. [The validation plan's §5](validation-plan.md#5-the-controlled-study-piece-c)
specifies it: two tiers, both off-diagonal cells, a prescribed rate with a committed drift
coefficient, three sizes, 250 replicates minimum, and rules frozen before the dispatch. **That
document is the specification and this one is the design**: what the cells actually are, what
constants were committed, and what the instrument has measured so far.

**C is three pull requests and this is the first.** The split follows the rule the roadmap is
lettered under — grouped where the *evidence* is shared — and the three have different evidence:

| PR | what it lands | evidence |
| --- | --- | --- |
| **C1** — *this one* | the harness, Tier 1 complete, the workflow, item 25's per-fit witness | the exact remainder of a prescribed sequence, and that the instrument works |
| **C2** | Tier 2's prescribed-rate learners, the fold-retained nuisances `P₀D̂` needs, both appendix-B remainder branches | item 13's rate |
| **C3** | the pilot, the freeze, the final study, the second seed batch | item 3, and gates 1 and 2 |

**Tier 1 is not the demonstration and this page will not be read as one.** Its nuisance sequence
is handed to the estimator rather than learned, which is the only construction in which *"the
intended asymptotic regime was entered"* is true by definition — so it is where a remainder can be
read off exactly, and it is not an applied claim. Tier 2 is the demonstration.

## The two cells

Both off the diagonal of the misspecification grid, because which nuisance is wrong is the whole
axis and one cell is an anecdote. `benchmarks/drtmle_injection.py` is the code and carries the
same reasoning at each constant.

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

`R_remaining` — the doubly-robust curve's own remainder, and the two appendix-B branches — needs
`P₀D̂` at the **fitted** reduced regressions, so it needs their values on covariates no fold trained
at. That is the fold-retained nuisance object §5 puts in Tier 2, it is **piece C2's**, and item 13
goes with it. Nothing here reports a corrected remainder, and the harness does not print a column
for one.

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

**Tier 2 will not be this cheap.** Its nuisances are fitted, which is what the 43s figure was
measuring, so C2 re-times before it re-scopes exactly as this did.

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
- a fit that **raised** is in the denominator, recorded with what it raised.

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
3. **the contract label must stay out of the verdict**, which is why `CorrectionCheck.passed` does
   not read it. On this evidence a fit that is *sound in every way the package can check* is
   routinely outside the theorem's scope, and folding the label into a pass/fail would report a
   sixth of well-behaved draws as broken and send a reader round a problem that is not there.

Six draws per cell is a share and not a rate; the pilot is what turns it into one. What is *not*
statistical is the pinned draw above — one fit, exhibited with its arithmetic, which is the right
instrument for a mechanism rather than for a frequency.
