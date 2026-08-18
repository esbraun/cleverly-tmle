# The longitudinal masks: the algorithm was quadratic, and the fit does not notice

Two changes, in the order the plan put them: measure the phases first, then fix the masks,
because the ratio `findings.md` reports for the mask fix is of the *cached-nuisance
recursion* and the question is what it is worth in a fit.

The answer is: almost nothing, and the fix is worth making anyway. Both halves of that are
the point.

> Measured on the four-core Intel Xeon @ 2.80 GHz container this repository's cloud
> sessions run in, `/proc/loadavg` under 0.6, Python 3.11, linear/logistic sklearn nuisance
> estimators, `n_folds=2`. The original run predates the estimator-object API.

## 1. The phases now say what they are

`benchmarks/numba/scenarios/pipelines.py` estimated a longitudinal fit's package-owned
share by running it under `cProfile` and bucketing lines by **filename** — anything under
`sklearn`, `lightgbm`, `joblib`, `scipy` or `threadpoolctl` was "the learner", and the rest
was "the package". `findings.md` §5 read 57.6% post-nuisance off that, with a footnote
saying why the figure is large.

`cleverly.utils.phases` replaces it: named regions, wall clock, off by default, entered
through `LTMLE.profile_phases()`. A survival fit, `n = 4,000`, `T = 2`, two regimens, two
horizons:

```
phase                          calls  exclusive s   share  inclusive s
----------------------------------------------------------------------
mechanism_fit                      4       0.3050  38.7%       0.3050
outcome_learner_fit                6       0.2921  37.1%       0.2921
inference                          1       0.1576  20.0%       0.1576
fluctuation                        6       0.0102   1.3%       0.0102
clever_covariate                   6       0.0009   0.1%       0.0009
influence_curve                    1       0.0004   0.1%       0.0004
pseudo_outcome                     6       0.0003   0.0%       0.0003
mask_construction                  7       0.0003   0.0%       0.0003
----------------------------------------------------------------------
accounted for                              0.7668  97.3%       0.7878
```

Three things in that table are worth more than the 57.6% it replaces.

**The learners are 76%**, directly measured rather than inferred from a filename. So the
package-owned half of a `glm` longitudinal fit is about a fifth, not three fifths — and the
old figure's excess was the profiler's per-call overhead charged to the code that makes the
most calls, which is precisely the code under investigation.

**`inference` is 20%**, and it is the multiplier bootstrap: `LTMLE` defaults to
`n_multiplier=2000`. The original profile found the same thing from the other side
("the largest *arithmetic* line in a longitudinal fit is the Rademacher multiplier draw at
14%"). That is the phase `bootstrap_numpy.md`'s change acts on, and it is the largest
package-owned line here by a factor of fifteen.

**The recursion's own arithmetic is 1.5%** — fluctuation, clever covariate, pseudo-outcome
and masks together. Which is the context for everything below.

## 2. The masks were `O(T² n)`

`LongitudinalData.at_risk(t)` and `.following(t)` each rebuilt a prefix from scratch:
`uncensored[:, :t].all(axis=1)`, and a Python loop of `t` boolean `&` passes in
`followed_through` with an `assignment_matrix` call inside it. Called at every node, that is
`Σ_t O(t·n) = O(T² n)` per regimen — and a survival fit pays it once per horizon on top.
(`_event_free` was already a column read; `event` is stored cumulatively for exactly this
reason, which is what the fix generalises to the other two.)

`LongitudinalData.regimen_masks(assignment)` scans once — `np.logical_and.accumulate` over
the `(n, T)` conjunction — and hands back three `(n, T+1)` matrices whose column `t` is the
prefix. `at_risk(t)` and `following(t)` are column reads. `fit_regimen` builds one per
regimen, `longitudinal.msm` one per regimen for the whole alternation, and `fit_mechanism`
one for the two regimen-independent factors.

The `at_risk`/`following` asymmetry survives intact, because it is the thing most likely to
be tidied away: `following(t)` reads the censoring and follow factors at `t` and the event
factor at `t - 1`, since a unit that had the event *at* `t` **is** the observation that it
happened and belongs in that node's regression.
`tests/unit/test_longitudinal_masks.py` asserts every mask at every node against the methods
themselves, on end-of-study, survival and competing-risks fits, under static plans and a
dynamic rule — and carries a negative control that fails if the event factor is read at `t`.

The mask term, isolated, behaves as the complexity says:

| T | rebuilt per node | scanned once | |
| ---: | ---: | ---: | ---: |
| 2 | 0.5 ms | 0.3 ms | 1.7× |
| 5 | 1.5 ms | 0.4 ms | 3.8× |
| 10 | 3.6 ms | 0.6 ms | 6.0× |
| 20 | 10.4 ms | 1.2 ms | **8.7×** |
| 40 | 34.8 ms | 3.0 ms | **11.6×** |

The rebuild quadruples when `T` doubles and the scan roughly doubles, which is the `O(T²n)`
against `O(Tn)` written out.

## 3. And it is 0.06% of the fit

The same runs, whole-fit:

| T | fit, scanned | mask share | fit, rebuilt | mask share |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.191 s | 0.13% | 0.198 s | 0.26% |
| 5 | 0.470 s | 0.08% | 0.468 s | 0.32% |
| 10 | 0.953 s | 0.07% | 1.010 s | 0.36% |
| 20 | 2.032 s | 0.06% | 2.408 s | 0.43% |
| 40 | 4.740 s | 0.06% | 4.654 s | 0.75% |

**The masks were never more than three quarters of a percent of a fit**, at forty nodes, on
the cheapest learner preset the package offers. The whole-fit column is noise either way
(0.98× to 1.19× across the five rows, in both directions), because `T` nodes of masks scale
against `T` nodes of *regressions* and the regressions win at every `T`.

So `findings.md` §2.3's "**2.4× of the recursion at `T = 20`**" is correct and is a statement
about the benchmark's cached-nuisance region, where the learner fits are excluded by
construction. Carried into a fit it is a fifth of a percent. This is the same denominator
error §5 of that document warns about, applied to its own recommendation — and it is why the
plan put the phase timing *before* the mask fix rather than after it.

**The fix stays.** It removes an `O(T²)` term for an `O(T)` one at no cost in memory that
matters (`3n(T+1)` bytes — 78 MB at `n = 10⁶, T = 25`, against the `(n, T)` float64 arrays
the container already holds at eight times that), it is exactly equivalent node by node, and
it is a precondition for the compiled recursion `findings.md` §2.3 recommends: a kernel that
compiled the rebuild would be compiling redundant work. What it is not is a speed-up anyone
fitting a model will see.

## 4. What the phase table says to do instead

Read down the first table. In a `glm` longitudinal fit the package owns roughly a fifth of
the runtime, and **`inference` is essentially all of it** — the multiplier bootstrap at 20%,
against 1.5% for the entire backward recursion including its masks, its fluctuations and its
clever covariates.

That inverts the ordering `findings.md` §7 gives, which puts the LTMLE and survival
recursions above the bootstrap on the grounds that they are "the largest absolute savings in
the package". Measured through the API rather than through a cached-nuisance region, the
recursion's package-owned arithmetic is 1.5% of a fit and the bootstrap is 20% of it.
