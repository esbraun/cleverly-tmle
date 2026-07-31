# Roadmap

Two lists, and they are different kinds of thing. **Variants** are estimators that plug into
the shared base classes (`estimators/base.py`, `inference/`, `learners/`, `fluctuation/`).
[Refusals worth lifting](#refusals-worth-lifting) are parameters this package already has the
machinery for and has simply not written down — drawn from [Not written
yet](methodology.md#not-written-yet), which is the full list of candidates rather than the
chosen ones.

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
  further factor per node in the denominator, and its own identification. The largest thing it is
  missing, as against refusing, is a working model over regimens, which is fourth below
- **doubly-robust nonparametric inference (`drtmle`)** — Benkeser, Carone, van der Laan &
  Gilbert (2017). Every interval reported here is valid when the second-order remainder is
  negligible, which needs *both* nuisances converging fast enough; `drtmle` buys an interval
  that stays valid when only one of them is consistent, by estimating additional
  reduced-dimension regressions of each nuisance's residual on the other and solving their
  score equations too. That is a genuine variant rather than a further estimand, so it plugs
  in at `TMLE._nuisances` and the targeting step rather than at the target registry. **Its
  scope has not been audited against this codebase**, unlike the five below — it predates
  them, and the sizing here is from the paper rather than from a read of what would have to
  change

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
The fifth is last because its cost is dominated by test infrastructure rather than by
derivation — it is the only one needing a *new* oracle law rather than a branch on an
existing one.

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
5. **`shifts=` with `delta=`.** Last for what it costs rather than for what it is worth: it
   is the only one of the five needing a *new* oracle law, so most of the work is test
   infrastructure. The audit also changed what this item is. The refusal's stated reason
   was **wrong** — `data/causal_data.py` said each of `delta=`, `intermediate=` and
   `weights=` becomes "a conditional density" on a continuum, but `P(Δ = 1 | A, W)` is a
   conditional *probability* of a binary event, an ordinary classifier with the dose as a
   numeric feature, and does not become a density because `A` is continuous. That is the
   same mistake made and then overturned for `incremental=` with `delta=`, where the
   derivation turned out to be the existing one with an extra factor. What is genuinely
   required is smaller than the old message claimed and larger than nothing: `π` evaluated
   at the *shifted* dose as well as the observed one, since the arm path evaluates `π_a(W)`
   at each counterfactual arm rather than at the observed treatment — an `(n, S + 1)` array
   threaded through `ShiftSet` and `mtp_submodel`, which currently discards `missingness`
   outright — and an oracle law crossing the two that exist,
   `tests/discrete_law_shift.py` having no `Δ` and `tests/discrete_law_mar.py` no doses.
   Missing outcomes with a continuous exposure are routine, so this is worth doing.
   Correcting the stated reason has already been done and is not part of it; re-auditing
   `intermediate=` on the same grounds is, since `P(Z = z | A, W)` is a probability too

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
