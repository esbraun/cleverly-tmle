# cleverly

Targeted maximum likelihood estimation (TMLE) for Python — with sensitivity analysis and
validation diagnostics treated as first-class parts of the estimator, not afterthoughts.

`cleverly` is a Python counterpart to the R targeted-learning ecosystem (`tmle`, `tlverse`/`tmle3`).
It takes **pandas or polars** dataframes interchangeably (via [narwhals](https://narwhals-dev.github.io/narwhals/)),
returns results in whichever backend you handed it — arrow-backed pandas
(`dtype_backend="pyarrow"`) and a bare `pyarrow.Table` included — and every estimator ships with:

- **influence-curve based inference**, plus cluster-robust variance, targeted bootstrap, and
  simultaneous (max-t) confidence intervals across estimands;
- **sensitivity analysis** — positivity/overlap diagnostics, truncation curves,
  omitted-variable-bias bounds with robustness values, E-values, and MNAR tilt analysis;
- **validation** — nuisance-model calibration and cross-validated risk, an explicit check that the
  efficient-influence-function score equation was solved, refutation tests, and a reusable
  simulation harness that measures bias and confidence-interval coverage.

It is written for two audiences at once: applied analysts who want a defensible effect estimate
out of observational data without leaving Python, and methodologists who need the estimand, the
influence function and the assumptions written down where they can be checked. Every parameter
the library reports is one it can name; everything else is [refused by name](#refusals) with a
reason, rather than approximated.

## Documentation

| | |
| --- | --- |
| [User guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md) | one runnable recipe per capability — multi-arm treatments, dynamic and stochastic regimes, continuous doses, incremental interventions, marginal structural models, longitudinal and survival fits, C-TMLE, cross-fitting, weights |
| [Technical appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md) | per algorithm: the estimand, its efficient influence curve, the second-order remainder, and the test that fails when it is built wrong |
| [Roadmap](https://github.com/esbraun/cleverly-tmle/blob/main/docs/roadmap.md) | what has landed, what is worth building next, and where native acceleration does and does not pay |

## Install

`cleverly` is not on PyPI yet, so install from source:

```bash
pip install "git+https://github.com/esbraun/cleverly-tmle.git"
pip install "cleverly[all] @ git+https://github.com/esbraun/cleverly-tmle.git"
```

The core depends only on numpy, scipy, scikit-learn, narwhals and joblib. The `all` extra adds
pandas, polars, lightgbm and matplotlib. Once the package is published the first line becomes
`pip install cleverly`.

One thing the backend promise does *not* cover: results are built from numpy, so a frame handed
in with `dtype_backend="pyarrow"` comes back as pandas but as **numpy-backed** pandas. Every
column a fit emits is a dense float with no nulls in it, so nothing an arrow dtype carries is
lost — but the promise is about the dataframe library, not about the dtype backend.

## Quickstart

```python
from cleverly import TMLE
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=2000, seed=0, backend="polars")  # or "pandas"

est = TMLE(estimands=("ate", "att", "atc", "ey1", "ey0"), random_state=0)
res = est.fit(
    frame,
    outcome="Y",
    treatment="A",
    covariates=["W1", "W2", "W3", "W4"],
).single()  # fit() returns one result per parameter; an ordinary fit has one

print(res.summary())
print(f"true ATE = {truth['ate']:.4f}")

res.to_frame()  # tidy results, in the backend you passed in
res.estimates["ate"].psi  # point estimate
res.estimates["ate"].ci  # (lower, upper)
res.estimates["ate"].influence_curve
```

```
              psi     std_err      ci_lower    ci_upper     p_value
ate      1.800893    0.052512      1.697971    1.903816    0.000000
att      1.974391    0.054585      1.867406    2.081376    0.000000
atc      1.652017    0.058129      1.538087    1.765947    0.000000
ey1      3.698363    0.050370      3.599639    3.797086    0.000000
ey0      1.897469    0.040687      1.817724    1.977215    0.000000

true ATE = 1.7500
```

`random_state=` is what makes those numbers reproducible: a fit draws a fold assignment, and
without a seed two runs of the same code give slightly different answers.

## An end-to-end fit

The quickstart returns an estimate. What follows is the rest of the surface — the diagnostics
that say whether to believe it — on one fit, with no refitting anywhere. Every analysis below
reads the cached nuisance fits and influence curves the fit already carries.

```python
import cleverly
from cleverly import TMLE
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=2000, seed=0)

res = (
    TMLE(estimands=("ate", "ey1", "ey0"), random_state=0)
    .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3", "W4"])
    .single()
)
print(res.summary())
```

```
estimand  psi     std_err  95% CI            p_value
--------  ------  -------  ----------------  -------
ate       1.8009  0.05251  [1.698, 1.9038]   <1e-4
ey1       3.6984  0.05037  [3.5996, 3.7971]  <1e-4
ey0       1.8975  0.04069  [1.8177, 1.9772]  <1e-4

simultaneous 95% bands (multiplier bootstrap, critical value 2.329 vs 1.960 pointwise):
  ate   [1.6786, 1.9232]
  ey1   [3.581, 3.8157]
  ey0   [1.8027, 1.9922]
```

**Is the identification plausible?** Positivity is the assumption an estimate degrades under
first, so it gets a report of its own, and confounding you did not measure gets a bound.

```python
print(res.sensitivity.positivity().summary())  # overlap, effective n, weight mass, verdict
res.sensitivity.robustness_value()  # confounding strength that would null the effect
res.sensitivity.benchmark(["W1", "W2"])  # calibrate that against covariates you did measure
```

```
truncated: 0 unit(s) (0.00%); most extreme untruncated g(W) = 0.04574
VERDICT: overlap looks adequate; no truncation-driven fragility detected.

{'rv': 0.5328, 'rva': 0.5127, 'max_bias': 2.3104}
```

**Did the estimator do what it claims?** `score_check()` verifies that targeting actually
solved the efficient score equation, and `nuisance()` reports how the learners did out of fold.

```python
print(res.validation.score_check().summary())
print(res.validation.nuisance().summary())
```

```
target  kind             |score|    before     threshold  ratio     ok
------  ---------------  ---------  ---------  ---------  --------  ---
mean    fluctuation      3.997e-18  1.401e-03  1.174e-06  3.40e-12  yes
ate     influence curve  2.132e-16  -          1.174e-06  1.82e-10  yes

PASS: the targeting step solved the estimated efficient score equation.

model       auc     brier   log_loss  r2      mse     cal_slope
----------  ------  ------  --------  ------  ------  ---------
propensity  0.6599  0.2284  0.6471    -       -       1.0093
outcome     -       -       -         0.6845  0.0049  1.0170

propensity: super learner weights glm=0.003, gam=0.832, boost=0.166
outcome: super learner weights gam=0.230, boost=0.770
```

**Anything the report did not name** comes from the joint influence curve by the delta method,
with no refit — here the risk ratio, which was not among the requested estimands:

```python
res.contrast(lambda psi: psi[0] / psi[1], ["ey1", "ey0"])
# contrast(ey1, ey0): 1.9491 (se 0.04092, 95% CI [1.8689, 2.0293])

res.save("fit.npz")  # arrays plus JSON, no pickle
again = cleverly.load("fit.npz")  # every retarget-based analysis is bit-for-bit identical
```

That is the whole shape of a fit: one call to estimate, and everything else read off what the
estimate already carries. The [user guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md)
takes each capability in turn.

## Architecture

The package is a pipeline with two registries bolted to the middle of it. Knowing where the
seams are is most of what it takes to extend it.

| directory | contents |
| --- | --- |
| `src/cleverly/data` | `CausalData` container and input validation |
| `src/cleverly/learners` | cross-fitting, screening, `SuperLearner`, thread limits |
| `src/cleverly/interventions` | regimes: static arms, dynamic rules, stochastic assignments; shifts of a continuous dose |
| `src/cleverly/msm.py` | the working model a fit projects the counterfactual means onto |
| `src/cleverly/fluctuation` | clever covariates and the targeting step |
| `src/cleverly/estimators` | nuisance orchestration, `TMLE`, result objects |
| `src/cleverly/longitudinal` | the time-ordered container, regimens, sequential regression, `LTMLE` |
| `src/cleverly/inference` | influence curves, clustering, bootstrap, simultaneous bands |
| `src/cleverly/sensitivity` | positivity, omitted-variable bias, E-values, MNAR tilt |
| `src/cleverly/validation` | score check, nuisance diagnostics, refutation, simulation |
| `src/cleverly/datasets` | synthetic processes with exactly known truth |
| `src/cleverly/targets` | the registry of estimands |

### The pipeline

A fit is four stages, and each one's output is the next one's only input:

```
CausalData  →  nuisances  →  fluctuation  →  influence curve  →  TMLEResult
              (cross-fitted   (clever covariate,  (the EIF, written
               g, Q̄, π)        Newton solve)       out explicitly)
```

Everything downstream of the third arrow — the variance, the cluster-robust variance, the delta
method, the simultaneous bands, and the score diagnostic — is computed *from* the influence
curve rather than from the fit. That is why a contrast nobody asked for needs no refit, and why
adding an estimand does not mean touching inference.

### Two registries

Estimands live in a registry, not in a `Literal`. A `Target` declares which fluctuation solves
its score equation, what scale its inference lives on, what it needs of the outcome, and — as a
required field — an `Identification` record naming its assumptions and what double robustness
buys for that estimand specifically. `register()` adds one.

Score equations live in a second registry. `group=` on a `Target` names a *score equation*, not
an estimand: six of the eight built-in targets share the `mean` fluctuation because they are
different functionals of one targeted distribution. `register_submodel()` adds a new one, and
`Target.group` is validated against that registry at registration time rather than at fit time.

The two are deliberately separate, because most new estimands need no new score equation. See
[Adding an estimand](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#adding-an-estimand)
and [Adding a fluctuation](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#adding-a-fluctuation).

### Five parameter axes, and why `LTMLE` is not a sixth

A `Target` declares via `parameter_axis=` what its parameters are indexed *by*:

| axis | declared with | the counterfactual is |
| --- | --- | --- |
| `arm` | *(default)* | a treatment level |
| `regime` | `interventions=` | a conditional distribution over the arms, `g*(a \| W)` |
| `shift` | `shifts=` | a change to the dose a unit received |
| `ipsi` | `incremental=` | the observed mechanism with its odds multiplied by `δ` |
| `msm` | `msm=` | still the arms — what changed is how they are summarised |

The five **partition** the registry rather than accumulating. Declaring one makes the other four
unavailable to that fit, because one fluctuation solves one set of score equations and a fit
reporting parameters from two axes would be putting two of them under one heading.

`LTMLE` is a separate estimator with its own container and result object rather than a sixth
axis. A regimen is a plan over nodes — not an arm, a regime, a shift, a tilt or an MSM
coefficient — and the point-treatment pipeline is built around one `Q̄(a, W)` and one `g(a | W)`.
What it *does* reuse is everything below the estimand: cross-fitting, the Super Learner, the
logistic fluctuation, the influence-curve variance, the delta method, the bands. What it cannot
reuse it [refuses by name](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#treatment-given-over-time-the-sequential-regression).

### `retarget`, and why the diagnostics are free

`TMLE.retarget` re-runs the targeting step against **cached** nuisance fits. Every sensitivity
analysis is exactly that operation with one input perturbed — a different truncation bound, a
tilted missingness mechanism — so it costs a fraction of a refit rather than a refit, and a
saved fit reproduces all of them bit for bit.

This is also the rule for new estimator variants: one that only changes *which* nuisance
estimate is targeted should override `TMLE._nuisances` and let the inherited `retarget` do the
rest. `CTMLE` is the worked example — because it swaps one array, every influence curve,
sensitivity analysis and validation diagnostic keeps working untouched, and the bootstrap
repeats the selection for free.

## What is implemented

Classic point-treatment TMLE for a binary treatment. The table below covers what R's `tmle`
package covers, plus the pieces that matter from `tmle3` and the literature — but read that
as a statement about *features*, not about numbers: **`cleverly` is not compared against R's
output, by decision.** No cross-language test exists here or in CI and none is planned —
agreement with another implementation is evidence about a transcription, and what this package
checks instead is that each method produces what its derivation predicts. There are also
known deliberate divergences (`g_bounds="auto"` resolves at Kish's effective sample size on
weighted fits). What the estimates *are* checked against is set out under
[How this is validated](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#how-this-is-validated).

| Capability | How you declare it | Where |
| --- | --- | --- |
| Point-treatment estimands | `estimands=("ate", "att", "atc", "ey1", "ey0", "rr", "or")` | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md) |
| Multi-valued treatment | nothing — a treatment with up to 20 levels is detected; `reference=` picks the contrast arm | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#multi-valued-treatment) |
| Dynamic and stochastic regimes | `interventions=(Static(...), Rule(...), Stochastic(...))` | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#dynamic-and-stochastic-regimes) · [appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#regimes-the-density-ratio-covariate) |
| Continuous dose | `shifts=[Shift(δ, cap=u)]` | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#shifting-a-continuous-dose) · [appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#shifting-a-continuous-dose-why-an-mtp-is-not-the-regime-it-induces) |
| Incremental interventions | `incremental=[Incremental(δ)]` | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#tilting-the-odds-of-treatment) · [appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#tilting-the-odds-of-treatment-two-score-equations) |
| Marginal structural model | `msm=MSM(design=..., terms=..., link=...)` | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#summarising-the-arms-a-marginal-structural-model) · [appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#the-msm-projection-its-matrix-and-its-remainder) |
| Treatment over time | `LTMLE({"always": 1, "never": 0}, ...)` — static or dynamic regimens | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#treatment-given-over-time) · [appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#treatment-given-over-time-the-sequential-regression) |
| Survival outcome | `outcome=["Y1", "Y2"]` — a list declares one absorbing event per node | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#a-survival-outcome) · [appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#survival-which-population-each-node-is-fitted-on) |
| Competing risks | `outcome={"relapse": [...], "death": [...]}` — a mapping declares causes | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#competing-risks) · [appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#competing-risks-the-cause-specific-recursion) |
| Collaborative TMLE | `CTMLE(search="greedy" \| "ordered" \| "discrete")` | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#collaborative-tmle) · [appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#c-tmle-how-the-selection-is-evidenced) |
| Doubly-robust inference — **in progress**, see [what is still open](https://github.com/esbraun/cleverly-tmle/blob/main/docs/roadmap.md#what-is-still-open) | `DRTMLE(guard=("Q", "g"))` — an interval that survives one inconsistent nuisance; valid under weaker conditions, not narrower and not efficient | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#doubly-robust-inference) · [appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#doubly-robust-inference-what-the-extra-equations-remove) |
| Cross-fitting and CV-TMLE | `n_folds=`, `targeting_scheme="fold"`, `cv_evaluation=True`, `repeats=` | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#cross-fitting-and-cv-tmle) · [appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#cross-fitting-what-the-folds-do-and-do-not-buy) |
| Observation weights | `weights=` on `.fit()`, with `id=` for a multi-stage design | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#observation-weights-and-which-population-they-define) |
| Missing outcomes, direct effects | `delta=`, `intermediate=` | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#options-with-no-section-of-their-own) |
| Nuisance estimation | `outcome_learner=`, `treatment_learner=` — any scikit-learn estimator, or `SuperLearner` | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#options-with-no-section-of-their-own) |
| Sensitivity | `res.sensitivity.*` — positivity, truncation curves, omitted-variable bounds, E-values, MNAR tilt | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#sensitivity) |
| Validation | `res.validation.*` — nuisance diagnostics, score check, refutation, `CoverageStudy` | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#validation) |
| Everything else | bounds, screening, clustering, contrasts, persistence, provenance, targeting diagnostics | [guide](https://github.com/esbraun/cleverly-tmle/blob/main/docs/user-guide.md#options-with-no-section-of-their-own) |

## Refusals

A good many things this library could be asked for are **refused rather than approximated**, and
a refusal is always *by name*: the keyword is accepted and rejected with a stated reason, rather
than arriving as an `unexpected keyword argument` that names none.

They are not all the same kind of thing, and what to do about one depends on where the problem
is. It can be **in this package** (the parameter is well defined and nobody has written it here —
ask, or compute it elsewhere), **in the question** (what was asked for is a different estimand
with its own identification, and no flag turns one into the other), or **in the method** (the
naive version runs and returns a plausible number that is wrong, usually with a known direction
of error). That last group is worth reading even by someone who never reaches for the keyword:
most are mistakes that are easy to make by hand in any framework, and none announces itself.

[How to read a refusal](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#how-to-read-a-refusal)
sets out the taxonomy and lists every one.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy src/cleverly
pytest -m "not slow" -q     # fast tier, ~3 minutes
pytest -m slow -q           # statistical validation tier (nightly in CI, ~1 hour)
python benchmarks/bench_tmle.py                         # where a whole fit's time goes
python -m benchmarks.numba.cli --config benchmarks/configs/sandbox.json
```

The second benchmark answers a narrower question than the first and answers it properly:
with the nuisance fits excluded from every timed region, does compiling or parallelising
the package's own arithmetic help? Its write-ups are
[the profile](https://github.com/esbraun/cleverly-tmle/blob/main/benchmarks/results/candidate_inventory.md)
and [the recommendation](https://github.com/esbraun/cleverly-tmle/blob/main/benchmarks/results/findings.md).
Neither is "numba everywhere" nor "numba nowhere".

`ruff` and `mypy` are both pinned exactly, and in three places that have to move together:
`pyproject.toml`'s `dev` extra, `.github/workflows/ci.yml`, and `noxfile.py`. `ruff` formats
the Python blocks inside Markdown as well as the source — so run it over the whole tree, not
just `src` and `tests`. `noxfile.py` wraps the same steps (`nox -s lint typecheck tests`) at
the same pins, and the fast tier runs on Python 3.11–3.13 in CI.

## Citing

There is no DOI yet. Cite the repository and the papers the estimator implements — the full list
is at the foot of the
[technical appendix](https://github.com/esbraun/cleverly-tmle/blob/main/docs/methodology.md#references).

## License

MIT
