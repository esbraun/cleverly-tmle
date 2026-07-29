# cleverly

Targeted maximum likelihood estimation (TMLE) for Python — with sensitivity analysis and
validation diagnostics treated as first-class parts of the estimator, not afterthoughts.

`cleverly` is a Python counterpart to the R targeted-learning ecosystem (`tmle`, `tlverse`/`tmle3`).
It takes **pandas or polars** dataframes interchangeably (via [narwhals](https://narwhals-dev.github.io/narwhals/)),
returns results in whichever backend you handed it, and every estimator ships with:

- **influence-curve based inference**, plus cluster-robust variance, targeted bootstrap, and
  simultaneous (max-t) confidence intervals across estimands;
- **sensitivity analysis** — positivity/overlap diagnostics, truncation curves,
  omitted-variable-bias bounds with robustness values, E-values, and MNAR tilt analysis;
- **validation** — nuisance-model calibration and cross-validated risk, an explicit check that the
  efficient-influence-function score equation was solved, refutation tests, and a reusable
  simulation harness that measures bias and confidence-interval coverage.

## Install

```bash
pip install cleverly              # core: numpy, scipy, scikit-learn, narwhals, joblib
pip install "cleverly[all]"       # + pandas, polars, lightgbm, matplotlib
```

## Quickstart

```python
from cleverly import TMLE
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=2000, seed=0, backend="polars")  # or "pandas"

est = TMLE(estimands=("ate", "att", "atc", "ey1", "ey0"))
res = est.fit(
    frame,
    outcome="Y",
    treatment="A",
    covariates=["W1", "W2", "W3", "W4"],
)

print(res.summary())
print(f"true ATE = {truth['ate']:.4f}")

res.to_frame()  # tidy results, in the backend you passed in
res.estimates["ate"].psi  # point estimate
res.estimates["ate"].ci  # (lower, upper)
res.estimates["ate"].influence_curve
```

```
              psi     std_err      ci_lower    ci_upper     p_value
ate      1.982031    0.061142      1.862195    2.101867    0.000000
att      1.994884    0.078210      1.841595    2.148173    0.000000
...
```

### Collaborative TMLE

A propensity model fitted to predict treatment as well as possible is fitted to the wrong
objective. A covariate that predicts treatment but *not* the outcome — an instrument —
removes no confounding, and putting it in `g` pushes propensity scores towards 0 and 1,
inflating the variance of `1/g` and so of the estimate. `CTMLE` chooses the covariates
entering `g` by cross-validating the loss of the *targeted outcome model* instead, so the
two nuisance fits only have to be right between them (van der Laan & Gruber 2010).

```python
from cleverly import CTMLE
from cleverly.datasets import make_instrument

# W1 confounds; W2 predicts treatment but not the outcome; W3 predicts only the outcome.
frame, truth = make_instrument(n=2000, seed=0)
res = CTMLE(
    search="ordered", estimands=("ate",), outcome_learner="glm", treatment_learner="glm"
).fit(frame, outcome="Y", treatment="A")
print(res.extra["ctmle"].summary())
```

```
search = ordered; target = ate; criterion = cross-validated penalized squared-error loss

k  covariates in g  steps  risk     cv risk
0  (intercept)      1      7.19957  7.22364  <--
1  W1               2      7.20063  7.22640
2  W1, W3           2      7.20060  7.22598
3  W1, W3, W2       3      7.20876  7.24196

selected g: (intercept only)
left out: W1, W2, W3 -- adjusting for these would have cost more variance than the bias
they remove
```

Two things to read off. The instrument is last in the ordering and adding it costs the
largest jump in cross-validated risk of any covariate — that is instrument inflation,
priced. And the selector stops at the intercept: a GLM is correctly specified for the
outcome in this process, so `Qbar` has already done the adjusting and there is no
residual confounding left for `g` to remove. That is collaborative double robustness —
the two nuisance fits only have to be right between them — and it is why C-TMLE's
standard error here is about 25% below a plain TMLE's on the same samples.

`search="greedy"` (the default) builds the ordering by forward selection instead;
`search="ordered"` is the scalable variant (Ju et al. 2019) at `O(p)` propensity fits
rather than `O(p²)`; `search="discrete"` cross-validates an explicit list of candidate
models. One caveat worth knowing: the influence-curve standard error treats the selected
model as given, so it does not include the variability the selection itself contributes,
and is mildly anti-conservative as a result. Pass `n_bootstrap=` for inference that does
— each replicate re-runs the search.

### CV-TMLE

```python
res = TMLE(targeting_scheme="fold").fit(frame, outcome="Y", treatment="A")
res.cv_targeting.summary()  # per-fold psi and epsilon, cross-validated std errors
res.cv_targeting.variance["ate"]
```

### Sensitivity

```python
res.sensitivity.positivity()  # overlap, effective sample size, weight mass
res.sensitivity.truncation_curve()  # estimate vs propensity-truncation bound
res.sensitivity.omitted_variable(cf_y=0.03, cf_d=0.03)
res.sensitivity.robustness_value()  # confounding strength that would null the effect
res.sensitivity.benchmark(["W1", "W2"])  # calibrate cf_y/cf_d against observed covariates
res.sensitivity.evalue()  # VanderWeele-Ding E-value (binary outcomes)
res.sensitivity.missingness_tilt()  # MNAR exponential tilt (needs `delta=`)
```

### Validation

```python
res.validation.nuisance()  # CV AUC/Brier/calibration for g, CV R^2/MSE for Q, SL weights
res.validation.score_check()  # did targeting solve mean(EIF) = 0?
res.validation.refute()  # placebo treatment, random common cause, subset stability
```

```python
from cleverly.datasets import nonlinear_dgp
from cleverly.validation import CoverageStudy

study = CoverageStudy(
    dgp=nonlinear_dgp(),
    estimator=lambda: TMLE(estimands=("ate",)),
    n=1000,
    n_replicates=200,
    seed=0,
)
print(study.run().summary())  # bias, sqrt(n) bias, mc se, mean se, coverage, rejection rate
```

## What is implemented

Classic point-treatment TMLE for a binary treatment, at feature parity with R's `tmle` package
plus the pieces that matter from `tmle3` and the literature:

| Capability | Notes |
| --- | --- |
| Estimands | `EY1`, `EY0`, `ATE`, `ATT`, `ATC`, `RR`, `OR` |
| Outcome types | binary, and bounded continuous via Gruber & van der Laan (2010) scaling |
| Nuisance estimation | any scikit-learn estimator, or the built-in `SuperLearner` (ensemble + discrete) |
| Cross-fitting | out-of-fold nuisance fits; stratified and cluster-respecting folds |
| CV-TMLE | `targeting_scheme="fold"` — an `epsilon` per validation fold, plus the cross-validated variance and per-fold diagnostics |
| C-TMLE | `CTMLE` — greedy, scalable-ordered and discrete collaborative selection of the covariates entering `g` |
| Targeting | iterative fluctuation (Newton) or one-step universal least-favorable submodel |
| Fluctuation | logistic or linear; clever covariate or weighted (`target_weights`, R's `target.gwt`) |
| Missing outcomes | `delta=` with its own nuisance model, entering the clever covariate |
| Controlled direct effect | `intermediate=` (R's `Z`), with `P(Z=1 | A, W)` estimated |
| Weights | observation weights for biased sampling / survey designs |
| Clustering | `id=` for cluster-level influence-curve variance and cluster bootstrap |
| Bounds | propensity truncation (`g_bounds`), outcome bounds (`q_bounds`), `alpha` shrinkage |
| Screening | pre-screening of covariates for the treatment model (`prescreenW.g`, `min_retain`) |
| Inference | IC-based, cluster-robust, targeted bootstrap, multiplier bootstrap, delta method |

## Roadmap

The base classes (`estimators/base.py`, `inference/`, `learners/`, `fluctuation/`) are shared
infrastructure; the following variants plug into them:

- marginal structural model TMLE (`tmleMSM`) and multi-valued / categorical treatments
- longitudinal TMLE (`ltmle`) for time-varying treatments and censoring
- survival TMLE (`survtmle`) and competing risks
- stochastic-intervention / shift TMLE (`txshift`, `tmle3shift`)
- doubly-robust TMLE with nonparametric inference (`drtmle`)

### On native acceleration

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
is Python, but each body is a nuisance fit, so they stay scikit-learn-bound.

The measurement is reproducible — rerun the benchmark before revisiting this.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy src/cleverly
pytest -m "not slow" -q     # fast tier, ~3 minutes
pytest -m slow -q           # statistical validation tier (nightly in CI, ~1 hour)
python benchmarks/bench_tmle.py
```

`noxfile.py` wraps the same steps (`nox -s lint typecheck tests`), and the fast tier runs on
Python 3.10–3.13 in CI.

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
- Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), *Long story short: omitted variable bias
  in causal machine learning*.
- VanderWeele & Ding (2017), *Sensitivity analysis in observational research: introducing the
  E-value*.
- Scharfstein, Rotnitzky & Robins (1999), *Adjusting for nonignorable drop-out using semiparametric
  nonresponse models*.

## License

MIT
