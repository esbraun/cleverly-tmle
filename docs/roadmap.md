# Roadmap

Two lists, and they are different kinds of thing. **Variants** are estimators that plug into
the shared base classes (`estimators/base.py`, `inference/`, `learners/`, `fluctuation/`).
[Refusals worth lifting](#refusals-worth-lifting) are parameters this package already has the
machinery for and has simply not written down — drawn from [Not written
yet](methodology.md#not-written-yet), which is the full list of candidates rather than the
chosen ones. **All five have landed**, so that list is now a record of what was done and what
the sizing got wrong rather than a plan; the remaining work is the second variant below, sized
against this codebase under [What `drtmle` would touch](#what-drtmle-would-touch).

## Variants

- **longitudinal TMLE (`cleverly.longitudinal.LTMLE`) — landed**, for static regimens and
  for **dynamic rules** `d_t(H_t)`, with time-varying confounding and monotone censoring,
  for a **survival outcome** — one absorbing event indicator per node, reporting the
  cumulative risk curve with joint bands over it — and for **competing risks**, where more
  than one absorbing state per node makes the report a cause-specific cumulative incidence
  per cause; see [Treatment given over time](user-guide.md#treatment-given-over-time),
  [A survival outcome](user-guide.md#a-survival-outcome) and [Competing
  risks](user-guide.md#competing-risks). What
  it still refuses is listed there under a `kind` column. The largest thing it will go on
  refusing is the *other* competing-risks estimand — the incidence under **elimination** of
  the competing events, which intervenes on them rather than conditioning on the history,
  and so is [a different question](methodology.md#a-different-question) rather than a gap: a
  further factor per node in the denominator, and its own identification. A working model over
  regimens was the largest thing it was *missing* as against refusing; that is item 4 below
  and has landed
- **doubly-robust nonparametric inference (`drtmle`)** — Benkeser, Carone, van der Laan &
  Gilbert (2017). Every interval reported here is valid when the second-order remainder is
  negligible, which needs *both* nuisances converging fast enough; `drtmle` buys an interval
  that stays valid when only one of them is consistent, by estimating additional
  reduced-dimension regressions of each nuisance's residual on the other and solving their
  score equations too. That is a genuine variant rather than a further estimand, so it plugs
  in at `TMLE._nuisances` and the targeting step rather than at the target registry — which
  is right as far as it goes, and is two of the four seams it turns out to touch. It predates
  the five below and was sized from the paper rather than from a read of what would have to
  change here; [What `drtmle` would touch](#what-drtmle-would-touch) is that read

## What `drtmle` would touch

The read the bullet above was missing, taken against `estimators/`, `fluctuation/`,
`inference/` and `tests/` rather than against the paper. It is written **before** any of the
work, which is the point: each of items 1 to 5 below records what its sizing got wrong, and
the misses here are ones that can be named in advance rather than found by mutation
afterwards. Nothing in it is a decision about the derivation — those are still open, and the
[three things to pin](#three-things-to-pin-before-any-code) says which.

### Four seams, where the sizing names two

1. **`TMLE._nuisances`**, as `CTMLE` does — but *adding* fields to `NuisanceEstimates` rather
   than replacing `propensity`, which is the difference between the two variants and the
   reason this one cannot be "override `_nuisances` and let the inherited `retarget` do the
   rest". The two reduced-dimension regressions go through `cross_fit_predictions` untouched:
   a one-column design, a residual target, `fit_mask` for the arm's rows.
2. **The targeting dispatch** in `TMLE._retarget_detailed`, which today has exactly two
   special branches — `needs_mechanism(group)` and `needs_projection(nuisance, group)` — and
   a default. This is a third, and a `solve_with_reduction` beside the two solvers in
   `estimators/targeting.py`. If one of the extra equations fluctuates **`g`**, as the
   paper's shape suggests, then `solve_with_mechanism` is what it resembles, down to
   returning a re-derived `NuisanceEstimates` the way that one returns `retilted`.
3. **`inference/influence.py`**, where the reported curve gains terms the plain one has no
   analogue of. `ipsi_means` is the precedent for that and for saying in its docstring
   exactly what reporting the plain curve instead would cost.
4. **`estimators/serialize.py`**: `FORMAT_VERSION` 8 → 9 for the extra arrays, on the terms
   versions 4 and 5 were bumped. A reloaded fit that had lost them would report a plain
   TMLE's interval under the variant's name, which is the shape of mistake that bump exists
   to prevent.

### The exact-law instrument cannot see what this estimator buys

This is the finding that matters most, and unlike its two predecessors it is derivable rather
than discovered. Under a law the sample realises exactly with a saturated learner — the
setting of every `tests/unit/test_influence_gateaux*.py` module — both nuisances are exact, so
both reduced-dimension regressions have identically zero targets: `E[Y − Q̄₀ | A = a, ĝ] = 0`
and `E[1{A=a}/g₀ − 1 | Q̄₀] = 0`. Both extra fluctuation coefficients are then zero and the
estimator reproduces `TMLE` exactly. So the package's primary evidence that a curve is right
supplies only a **degeneracy check** here, and would pass against an implementation whose
extra terms are wrong in any way that vanishes at the truth. That is items 4 and 5's lesson
arriving a third time, and the first time it has been seen coming.

What *can* see it is the remainder idiom. `tests/unit/test_remainder.py` evaluates the von
Mises expansion at nuisances that are **wrong on purpose** on the finite-support law,
deterministically and to machine precision, and this estimator's claim is precisely a
statement about that remainder — that a product of the two nuisance errors is replaced by
products of *reduced-dimension* ones. It is statable there as an equality with `TMLE`'s
product form as the negative control, which makes that module the thing to write **first**,
before the estimator rather than after it.

End to end the claim is about **coverage, not bias**, and that distinction is the whole
variant. `TMLE`'s double robustness is a statement about the *point estimate*: `R₂` is the
product `‖ĝ − g₀‖·‖Q̄ − Q̄₀‖`, so one inconsistent nuisance still leaves `R₂ → 0` and `ψ̂`
consistent — which `tests/unit/test_remainder.py` already checks exactly. The interval needs
the strictly stronger `√n · R₂ → 0`. With both nuisances consistent at `n^(−1/4)` the product
delivers that; with only one, the bad factor stops shrinking and `R₂` becomes *first-order* in
the good one's error, which no nonparametric estimator drives below `n^(−1/2)`. So the
estimator stops being asymptotically linear and its coverage decays as `n` grows while its
bias does not. **`TMLE` is doubly robust for consistency and singly robust for inference**,
and closing that second gap is what this variant is for.

Which makes the nightly instrument a `CoverageStudy` over the off-diagonal of
`tests/e2e/test_double_robustness.py`'s grid, with `TMLE`'s own coverage as the control this
has to beat — and one trap in building it. That grid's "correct" cell is an **oracle**
(`OracleOutcomeContinuous`, `OracleTreatment`), which makes the good nuisance exactly right,
`R₂` exactly zero, and `TMLE`'s interval already valid; a study built that way would show
nothing to buy. The gap opens only where the good nuisance is *estimated*, so the study needs
a correctly-specified learner in that slot rather than the truth. Nightly tier; never run it
in the sandbox.

### Three things to pin before any code

Each is invisible to the check that would otherwise catch it.

1. **The two extra estimating equations, and which nuisance each fluctuates.** If one of them
   fluctuates `g` the targeting is an alternation, and whether an alternation terminates is
   not automatic: `solve_with_mechanism` terminates because its two steps are coordinate
   ascent on one joint likelihood, and `solve_with_projection` is a separate function
   precisely because that argument did not carry over to a least-squares half.
2. **Which influence curve is reported** — `D*` at the targeted nuisances alone, or `D*` plus
   the extra components. The difference is exactly zero at the truth, so no exact-law test can
   see it, and it surfaces only as mis-coverage in the off-diagonal cells, which is the one
   thing the estimator exists to fix.
3. **How the reduced regressions are cross-fitted.** Their *design* is itself an out-of-fold
   prediction — `ĝ(W)` or `Q̄(a, W)` — so fitting them on the same `Folds` trains fold `k`'s
   regression on design values produced by models that saw fold `k`. That is the dependence
   `tests/unit/test_crossfit_leakage.py` exists to prevent, arriving through the design matrix
   rather than through the target.

### Scope to declare rather than discover

Binary, `mean`-group only — `ey1`, `ey0`, `ate` — is what the derivation covers. Every other
axis this package has (`att`/`atc`, `regime`, `shift`, `ipsi`, `msm`, and a multi-valued
treatment) must be **refused by name** rather than silently handed a plain fluctuation, on the
rule `LTMLE` established: a subsystem that was never taught about a variant raising
`AttributeError` is not a refusal. Two of them need deciding rather than inheriting.
`sensitivity/omitted_variable.py`'s Riesz representer reads
`submodel.observed[:, 1] - submodel.observed[:, 0]`, and the extra covariates change what a
column means. And the truncation curve and the MNAR tilt reach the targeting through
`retarget`, so they would re-solve the extra equations — probably right, and right by
inheritance rather than by decision, which is how the wrong version of it would also arrive.

### Sizing

Comparable to `CTMLE` or larger, and **not** transcription in the way the five items below
were. `CTMLE` swaps one array and inherits every influence curve, sensitivity analysis and
diagnostic untouched; this adds arrays, a targeting branch, curve terms and a serializer
version, and each of those is a place a reader could be told a plain TMLE's number. Four to
six commits in `src/`, plus a remainder module, a nightly coverage study and a section of the
guide — and the remainder module first.

## Refusals worth lifting

Everything under [Not written yet](methodology.md#not-written-yet) is a candidate; these are
the ones that answer a question applied causal inference actually asks *and* rest on a derivation
that is already settled, so the work is transcription and checking rather than research.
Nothing here is blocked on a modelling question.

**The order is a dependency order, not a preference order.** Each item is independently
shippable, but taken in sequence some of them hand work to the next: the first was
self-contained and unblocks two sensitivity analyses; the second builds the projection
machinery the fourth copies; the third and fourth both change `fit_regimen` and
`fit_mechanism`, so doing them adjacent is one round of churn in those signatures rather than
two — and taking them adjacent paid: the third left the recursion carrying the data's
weights, which the fourth inherited rather than adding.
The fifth was last because its cost is dominated by test infrastructure rather than by
derivation — it is the only one needing a *new* oracle law rather than a branch on an
existing one, and that held: the `src/` change was four small commits and the law, its
Gateaux module, its remainder module and the mutation hunting were the rest of it.

**All five have landed.** What remains here is the second variant above — now sized against
this codebase rather than against the paper, under [What `drtmle` would
touch](#what-drtmle-would-touch) — and a handful of refusals under [Not written
yet](methodology.md#not-written-yet) that are there because nobody has asked rather than
because anything stands in the way.

1. **`ATT` / `ATC` for a multi-valued treatment — landed.** "The effect among those who
   actually received arm `a`" is now `att[a vs ref]`, one per non-reference arm, with
   `atc[a vs ref]` the same contrasts among the reference arm's units; see [multi-valued
   treatment](user-guide.md#multi-valued-treatment). The derivation was the binary one with
   `1{A=1}` and `1{A=0}` replaced by `1{A=a}` and `1{A=r}` and the odds by `g_a / g_r`, so the
   fluctuation gained a column per contrast rather than a group, and
   `tests/discrete_law_multi.py` gained `att[a vs r]` branches rather than a new law. Two
   things worth recording. The reference arm loads *every* column of that fluctuation —
   it is the arm each contrast is taken against — so the Hessian is no longer diagonal as
   the `mean` group's is, and `Submodel.contrast_columns` exists because a column is now
   keyed by the contrast it carries rather than by an arm it updates. And they are **not**
   in a multi-arm default report: `2(K-1)` further parameters would have moved the
   simultaneous bands of every multi-arm fit that already ran, so `default_arms="binary"`
   keeps them opt-in. What still follows from the same contrast machinery, and is now the
   next thing this unblocks rather than part of it: the omitted-variable bound and the MNAR
   tilt on a multi-valued treatment
2. **A non-identity link for `msm=` — landed.** `link="log"` and `link="logit"` make a
   coefficient a log risk ratio or a log odds ratio, and `res.coefficients(scale="ratio")`
   exponentiates them; see
   [the MSM section](user-guide.md#summarising-the-arms-a-marginal-structural-model).
   The identity path is bit for bit unchanged — `dm/dη` is one there, so the covariate is
   the same array and the projection the same `np.linalg.solve`. Three things worth
   recording, because the sizing above got two of them wrong. The alternation is **not**
   `solve_with_mechanism`: that loop terminates because its two steps are coordinate ascent
   on one joint likelihood, and a projection is a least-squares solve rather than a
   likelihood, so `solve_with_projection` is a sibling that restarts from `Q̄⁰` each round
   for a clean fixed point instead of continuing for a monotone one. It converges much
   faster than the mechanism loop besides — `1e-3` to `1e-4` per round against 0.15 to 0.52
   — because `β` reaches the covariate only through a smooth factor. The matrix the curve
   is premultiplied by the inverse of gains a **curvature term** that vanishes only where
   the working model fits, so no saturated check can catch its absence; that mutation is
   now a control in the oracle. And the remainder stops being *exactly* zero when the
   mechanism is right — that exactness was the linearity of the estimating equation in `β`,
   not a stronger double robustness — so the test measures a rate where it asserted an
   equality. What this unblocks is the fourth item, which copies the projection's shape
3. **Observation weights for `LTMLE` — landed.** `LTMLE(...).fit(frame, weights="w")`
   estimates the declared regimen parameters in the tilted population `dP_w = w dP / E[w]`,
   with every node's mechanism, censoring factor and sequential regression fitted by
   weighted loss, every node's score equation weighted, and the reported curve
   `(w / E[w]) · D*(P_w)`; see [treatment over
   time](user-guide.md#treatment-given-over-time). It was
   the transcription the sizing said it was — the statement was already derived in
   `data/weighting.py` and the plumbing already carried a weight vector — but three things
   are worth recording. The **refusal's stated reason was wrong**, in the same way item 5's
   was: it said observation weights "put a further per-unit factor in the clever covariate's
   denominator at every node", and they do not. A weight tilts the *population*; `h_t`
   divides by the `2T` mechanism factors and by nothing else, and putting `w` there would
   divide the estimating equation by the very tilt it applies. What *does* move is
   `g_bounds="auto"`, resolved at Kish's effective `n` as it is at one node — and over `T`
   nodes that compounds rather than cancels, since the bound reaches every factor. And the
   leverage `res.diagnostics()` reports is now `w / ∏g` rather than `1/∏g`, on
   `sensitivity/positivity.py`'s reasoning that the two reweightings multiply. The oracle
   was a branch on the existing law rather than a new one: `tests/discrete_law_longitudinal.py`
   gained the tilt and its Gateaux derivative, and its saturated learner had to start
   *honouring* `sample_weight` — accepting and discarding it would have left the estimator
   holding `P_0`'s conditionals while its estimand was at `P_w`, which is the one mistake
   here that leaves `epsilon` non-zero rather than silent. The nightly tier gained a
   coverage study on a *biased sample*, where selection is a known `π(W_1)` and `w = 1/π`,
   so the truth is the unweighted one unchanged; ignoring the weights there costs about
   fourteen Monte Carlo standard errors of bias on each counterfactual mean — and almost
   nothing on their contrast, which is why that control is taken on a level
4. **A marginal structural model over regimens, for `LTMLE` — landed.**
   `LTMLE(regimens, msm=MSM(...))` reports `msm_regimen[<term>]` in place of a mean per
   plan: `β` is the `h`-weighted projection of `E[Y^ā | V]` onto `m(ā, V; β)`, under every
   link, with `V` a subset of the baseline covariates and the horizon *inside* the design
   on a survival fit — see [summarising the
   regimens](user-guide.md#summarising-the-regimens-a-marginal-structural-model). Four things
   are worth recording, and the sizing above got two of them wrong.
   The structural difference from the point-treatment working model is that **the node
   fluctuation must be pooled across the regimens**, and the reason is a rank argument
   nobody had made: there the `p` columns are separated by summing over the arms *within a
   row*, and a regimen is a plan rather than a value some unit took, so a per-regimen
   covariate is `φ(ā, V)` scaled by the scalar `h_t` — rank one whenever the model has no
   effect modifier, collapsing its `p` score equations into one. Each node therefore solves
   one fluctuation over the regimens stacked, with a single shared `epsilon`, and the real
   churn was control flow rather than mathematics: `fit_regimen`'s one-plan-at-a-time
   backward pass had to become lockstep.
   **A link costs a whole backward pass per round**, not a re-solved fluctuation, which the
   sizing had not seen: `β` enters every *earlier* node's regression target through the
   recursion, so there is no fixed `Q̄⁰` to restart from and the fixed point is stated over
   the whole pass. Measured at four or five rounds, contracting by `1e-4` each — the
   point-treatment rate, and for the same reason.
   `solve_projection` was reused **verbatim** with its arm axis read as the regimen axis,
   which is what item 2 bought; `MSMSet` deliberately was not, since its second axis is
   arms in its field name, its docstring and its accessors, and its constructor reads a
   `CausalData` throughout. Only the rank rule is shared. `h(ā, V)` and the observation
   weights stayed apart exactly as item 3 predicted.
   Two claims turned out weaker than expected and both are now stated as they are. The
   saturated reduction is **not** bit-for-bit and cannot be — the pooled Newton's
   convergence test and line search are taken over all the stacked rows — so it is exact on
   the exact law, where no step is taken, and `1e-11` elsewhere. And of seven deliberate
   mutations, **three passed on the first try**: the baseline-frame pin was blind because
   every longitudinal fixture here has an empty `time_varying[0]`, making
   `history_frame(1)` the same object; dropping the observation weight from the curve was
   invisible because nothing exercised a working model on a weighted fit; and the source's
   claim that an `at_risk` mask "leaves `epsilon` non-zero" was simply false, since the
   covariate is already zeroed off `trained_on` and the substitution moves no reported
   number at all. The first two are now covered and the third is now stated correctly
5. **`shifts=` with `delta=` — landed, and with `intermediate=` and `weights=` besides.**
   `_refuse_continuous_combinations` refused all three on one reason and the reason was
   wrong for all three, so lifting them was one change rather than three; see [missing
   outcomes, an intermediate, and weights on a
   dose](user-guide.md#missing-outcomes-an-intermediate-and-weights-on-a-dose). The
   derivation was the existing one with a further factor, exactly as it had been for
   `incremental=` with `delta=`: `H(a, W) = h(a, W) / {π(a, W) q_z(a, W)}`, and only the
   residual term is inverse-weighted because `Q̄(d(A,W),W) − Ψ` is a function of `(A, W)` and
   both are recorded whatever happens to `Y`. Five things are worth recording, and the sizing
   above got three of them wrong.
   The `(n, S + 1)` array belongs on **`NuisanceEstimates`, not on `ShiftSet`**, which the
   sizing had backwards. `bounded_missingness(nuisance_bound)` truncates at *targeting* time,
   and `retarget`, the MNAR override and `truncation_curve(mechanism=True)` all depend on
   that; folding `1/π` into `ShiftSet.design` at fit time would freeze the bound and make the
   mechanism truncation curve **flat by construction** — which reads as "the estimate does not
   hinge on the truncation choice", a wrong conclusion reported silently. Keeping it where the
   arm path keeps it also meant no `ShiftSet` field, no `subset` branch and no serializer
   change, and `clever_covariate_inputs` worked untouched.
   **A Gateaux check on an exact law cannot see the mistake this item is about.** At `epsilon
   = 0` the reported curve reads the *observed* block of the covariate and the untargeted
   `Q̄`, so dividing every block by the mechanism at the observed dose — the whole error the
   `(n, S + 1)` array exists to prevent — passes all 39 tests of the new Gateaux module. So
   does applying the selection indicator to the counterfactual blocks. Both were applied and
   watched to pass before the two instruments that *do* catch them were written: a structural
   pin on the covariate's blocks, and a plug-in with `epsilon != 0`. This is item 4's lesson
   again and it did not announce itself the second time either.
   `intermediate=` came free and turned up a live bug on the way. `mtp_submodel` applied the
   `1{Z = z}` indicator to its counterfactual blocks, where `mean_submodel` deliberately does
   not — the blocks are already at `Z = z` by construction — so every row whose intermediate
   took the other level would have carried an **un-updated** prediction into the plug-in. Dead
   code while `intermediate=` was refused here; a silent bug the moment it was not.
   `weights=` needed no `src/` change at all beyond deleting the refusal: `fit_conditional_density`
   already routed the weights through the long expansion, `shift_means` already averaged and
   scaled by them, and `_bounds_n` already resolved `auto` at the effective `n`. Item 3's
   compounding-bound story does **not** carry over, though, and saying so matters: `g_bounds`
   does not bite on this axis at all, since there is no per-arm propensity and the ratio is
   untruncated, so `nuisance_bound` is the only truncation a shift fit has.
   The oracle law was the cost the sizing said it was, and one law rather than the predicted
   one-per-lift: `tests/discrete_law_shift_cde.py` crosses the doses with `(Δ, Z)` and takes
   `level=None` for the parameter a `delta=`-only fit reports, so the two cannot disagree by
   construction. Its one indispensable property is that `π` and `q_z` vary with the **dose** —
   a mechanism depending on `W` alone makes `π(d(a,w), w) = π(a, w)` identically and the whole
   feature untestable. With the law's own nuisances the fit returns the truth *exactly* at all
   three levels; the mechanism's quantile binning is why that, and not a coverage study, is
   the strongest end-to-end statement available here.
   What is left refused is a narrower gap than the one it replaced: the **MNAR tilt** on a
   shift fit. The tilt re-mixes `Q̄` under a moved mechanism, a shift's plug-in is `Q̄` at the
   assigned dose, and whether the tilted parameter is still the shift parameter has not been
   derived — so this is a missing derivation rather than missing transcription, which is why
   it is not being carried forward as a sixth item

## On native acceleration

A Rust extension for the numerical kernels was planned. `benchmarks/bench_tmle.py` says it
is not worth building. Profiling a full fit by module (`cProfile`, total time):

| fit | cleverly-authored code | scikit-learn + LightGBM |
| --- | --- | --- |
| n=5,000, `library="default"` | **0.5%** | 44% |
| n=20,000, `library="glm"` | 22% | 17% |

The targeting step is 1.5–1.7% of a `glm` fit and does not appear at all in a `default`
one — it is a 2×2 Newton solve with a closed-form Hessian. Nuisance estimation dominates,
and it already runs in compiled code. Note how much the preset matters: `glm` is the
cheapest library available, so it makes every other line's share look several times larger
than it is. Benchmark with `--library default` before drawing a conclusion.

The 22% figure above is almost entirely *one* function, and profiling it turned up waste
rather than arithmetic — waste that was cheaper to fix than to rewrite:

- **The multiplier bootstrap was 92–95% multiplier *generation* and 2–3% matrix product.**
  It drew a full float64 uniform to produce one Rademacher sign. Generating bits instead
  is ~2.4× faster. Better: for `multiplier_kind="normal"` the max-t law has a closed form
  — `xi @ IC` is a linear map of a Gaussian — so the whole resampling loop collapses to
  one covariance and a draw from an *m*-dimensional normal, which is **80–360× faster**
  and never allocates a `(n_replicates, n)` array.

  That speed is not free, and `multiplier_kind` still defaults to `"rademacher"`. The
  closed form exists *because* the Gaussian max-t law depends on the influence curves only
  through their covariance — so `"normal"` is a plug-in normal approximation rather than a
  resampling scheme, and it cannot see the leverage a `1/g(W)` clever covariate produces
  under weak overlap. Simulated against a brute-force max-t distribution, it is biased
  conservative there (+0.14 on a true 2.16 at n=200, +0.07 at n=2,000), while `"rademacher"`
  stays within 0.02. On well-behaved influence curves all three kinds agree. Use `"normal"`
  when *n* is large, the curves are well behaved, and resampling actually shows up in a
  profile.
- **The cluster bootstrap rebuilt its membership index inside every replicate**, an
  `O(n_clusters × n)` scan per draw. Building it once is **24–160× cheaper** per replicate,
  which a 1000-replicate cluster bootstrap pays back a thousand times over.
- `cluster_sums` used `np.add.at`, which is unbuffered; `np.bincount` is ~2× faster.

None of that needed Rust, and the package stays pure-Python. The other place that mattered
turned out to be thread scheduling rather than arithmetic: nuisance fits run
single-threaded by default so parallelism happens across folds and candidates instead of
inside each fit (see `cleverly.learners.set_thread_limit`).

**When to revisit this.** Native code pays where the nuisance estimator is *not* an
scikit-learn model, and today none of them is. The trigger is **HAL** (highly adaptive
lasso) and its undersmoothed variant: a zero-order spline basis of `n × O(n·d)` binary
indicators that scikit-learn's lasso cannot take, where basis enumeration, sparse assembly
and coordinate descent are a natural fit for a native extension — R's `hal9001` ships a C++
backend for exactly this. The EP-learner benefits *through* HAL rather than on its own; its
other cost is targeting a *k*-dimensional score with *k* = basis size, which is BLAS-bound
and already fine. Longitudinal and survival TMLE are weaker cases: the loop over timepoints
is Python, but each body is a nuisance fit, so they stay scikit-learn-bound. That remains a
prediction rather than a measurement — `benchmarks/bench_tmle.py` has no `LTMLE` case, so
profile one before acting on it.

The measurement is reproducible — rerun the benchmark before revisiting this.
