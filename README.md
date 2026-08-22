# cleverly

**[Read the cleverly documentation →](https://esbraun.github.io/cleverly-tmle/)**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-1565c0.svg)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/license-GPL%20v3-087f8c.svg)](LICENSE)
[![Status: alpha](https://img.shields.io/badge/status-alpha-c97a00.svg)](https://pypi.org/classifiers/)

`cleverly` is the Python toolbox for causal studies built on targeted maximum likelihood
estimation (TMLE), organized around the causal question before the estimation method.

*It is named for TMLE's clever covariate—proof that even the jokes in this toolbox are targeted.*

```text
study design -> typed estimand -> identified effect -> estimation method -> causal result
```

`cleverly` supports point and longitudinal treatment settings, influence-curve inference,
cross-fitting, diagnostics, sensitivity analysis, and structured persistence. It accepts pandas,
polars, Arrow-backed pandas, and `pyarrow.Table` inputs through
[narwhals](https://narwhals-dev.github.io/narwhals/).

> [!WARNING]
> `cleverly` is alpha software and is not on PyPI. Pin a commit for reproducible work. Unsupported
> design, estimand, and method combinations fail before nuisance fitting instead of returning an
> approximation to a different causal question.

## Install

Install the core package from GitHub:

```bash
python -m pip install "git+https://github.com/esbraun/cleverly-tmle.git"
```

Add pandas, polars, and plotting support with the `all` extra. Third-party nuisance estimators
such as XGBoost or LightGBM can be installed separately and passed as sklearn-compatible objects:

```bash
python -m pip install "cleverly[all] @ git+https://github.com/esbraun/cleverly-tmle.git"
```

Python 3.11 or newer is required. See
[Installation](https://esbraun.github.io/cleverly-tmle/getting-started/installation.html) for a
development environment and reproducible commit-pinned installs.

## Quickstart

Declare the observed-data design and the causal estimand separately, inspect identification, then
estimate:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_nonlinear_ate

frame, truth = make_nonlinear_ate(n=2_000, seed=7)
study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3", "W4"),
    ),
)

effect = study.identify(ATE())
print(effect.summary())

result = effect.estimate(random_state=7)
print(result.summary())
print(result["ate"].ci)
```

The identified effect states the observed-data functional, assumptions, nuisance requirements,
and available methods before any learner is fit. The result retains estimates, influence curves,
joint covariance, structured parameter keys, normalized method configuration, provenance, and
post-fit assessment.

Continue with the
[full quickstart](https://esbraun.github.io/cleverly-tmle/getting-started/quickstart.html) or the
[analysis workflow](https://esbraun.github.io/cleverly-tmle/workflow.html).

## Documentation

The Sphinx/MyST documentation is published on
[GitHub Pages](https://esbraun.github.io/cleverly-tmle/) and builds from `docs/`.

| section | use it for |
| --- | --- |
| [Getting started](https://esbraun.github.io/cleverly-tmle/getting-started/) | installation, first fit, and result basics |
| [Workflow](https://esbraun.github.io/cleverly-tmle/workflow.html) | moving from a causal question through identification, estimation, assessment, and reporting |
| [User guide](https://esbraun.github.io/cleverly-tmle/user-guide/) | data roles, estimands, learners, methods, longitudinal designs, results, and refusals |
| [Technical reference](https://esbraun.github.io/cleverly-tmle/technical-reference/) | every implementation family, with theory, citations, local source, external provenance, and evidence |
| [Examples](https://esbraun.github.io/cleverly-tmle/examples/) | complete point, intervention, longitudinal, and post-fit workflows |
| [Python API](https://esbraun.github.io/cleverly-tmle/api/) | generated signatures, attributes, methods, and return types |

The [development reference](https://esbraun.github.io/cleverly-tmle/development/) contains the
roadmap, architecture invariants, and method-benchmarking strategy. The test-enforced evidence
manifest lives in the Technical reference.

## Implemented analysis families

- Counterfactual means, ATE, ATT, ATC, risk and odds ratios, natural-course means, population-
  attributable effects, multi-valued treatments, missing outcomes, and controlled direct effects.
- Static, dynamic, and stochastic regimes; continuous modified treatment policies; incremental
  propensity-score interventions; and point/longitudinal MSM projections.
- Longitudinal regimen means and contrasts for end-of-study, survival, and competing-risk outcomes.
- Observation weights, strata, cluster-robust inference, cross-fitting, repeated cross-fitting,
  CV-TMLE, simultaneous intervals, and bootstrap inference.
- Ordinary TMLE, collaborative TMLE, and DR-TMLE for their documented compatible estimands.
- Positivity, nuisance, and score diagnostics; omitted-variable, E-value, and missingness
  sensitivity analyses; refutation; variable importance; and trusted whole-result persistence.

## Roadmap

- Build the general nested Riesz engine and its initial evidence-gated catalog, including analytic
  and direct representers, nested composition, diagnostics, and persistence.
- Add optional DoWhy integration for graph-based identification and backdoor translation while
  keeping the core package standalone.
- Add EP learning for heterogeneous effects, beginning with conditional average treatment effects
  and conditional relative risks.
- Expand the estimand catalog target by target, with a separate derivation, evidence record,
  refusal contract, and statistical study for each family.

These are accepted directions, not implemented release claims. Their ordering, governing sources,
interfaces, refusals, and evidence requirements are in the single
[roadmap](docs/roadmap.md).

## Method configuration

Named shortcuts normalize into immutable configuration groups:

```python
from cleverly import CrossFitting, Inference, ModelSpec, Runtime, TMLEMethod

method = TMLEMethod(
    models=ModelSpec(
        outcome_learner=LinearRegression(), treatment_learner=LogisticRegression(max_iter=1000)
    ),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
    inference=Inference(alpha=0.05, simultaneous=False),
    runtime=Runtime(random_state=7, n_jobs=1),
)
result = effect.estimate(method=method)
```

`CollaborativeTMLEMethod` and `DRTMLEMethod` select estimator variants without changing the
identified causal question. `effect.available_methods()` reports support and refusal reasons.

## Development

```bash
uv venv
uv pip install -e ".[dev,docs]"
ruff check .
ruff format --check .
mypy src/cleverly
pytest -m "not slow" -q
sphinx-build -W --keep-going -b html docs docs/_build/html
```

`nox -s docs` runs that same warning-as-error build in an isolated environment, which is what CI
does; the direct call is the faster loop and is why the install above includes the `docs` extra.

The fast tier compiles every Python fence, executes the registered reader-facing guides, resolves
every relative link, and checks that the complete root API is represented in generated API source.
Scientific behavior belongs in ordinary fast tests or named slow statistical studies. Run the
relevant checks locally before handoff; a green GitHub Actions CI run is the final merge signal.

## Citing

There is no DOI yet. Cite the repository commit used in the analysis and the primary method papers
listed in the
[technical reference](https://esbraun.github.io/cleverly-tmle/technical-reference/) and
[references](https://esbraun.github.io/cleverly-tmle/references.html).

## License

[GNU General Public License v3.0](LICENSE)
