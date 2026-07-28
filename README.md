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
| CV-TMLE | fold-wise targeting with pooled cross-validated influence curve |
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

- collaborative TMLE (C-TMLE) with data-adaptive covariate selection for `g`
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
  one covariance and a draw from an *m*-dimensional normal. That is the same distribution,
  not an approximation, and it is **80–360× faster** while never allocating a
  `(n_replicates, n)` array. The default stays `"rademacher"` so reported critical values
  do not move; prefer `"normal"` for large `n`.
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
pytest -m "not slow" -q     # fast tier, ~90 seconds
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
- van der Laan & Gruber (2016), *One-step targeted minimum loss-based estimation*.
- Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), *Long story short: omitted variable bias
  in causal machine learning*.
- VanderWeele & Ding (2017), *Sensitivity analysis in observational research: introducing the
  E-value*.
- Scharfstein, Rotnitzky & Robins (1999), *Adjusting for nonignorable drop-out using semiparametric
  nonresponse models*.

## License

MIT
