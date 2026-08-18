# cleverly

Targeted maximum likelihood estimation (TMLE) for Python, organized around the causal question
before the estimation method.

`cleverly` accepts pandas, polars, Arrow-backed pandas, and `pyarrow.Table` inputs through
[narwhals](https://narwhals-dev.github.io/narwhals/). It provides influence-curve inference,
cluster-robust variance, simultaneous intervals, sensitivity analyses, validation diagnostics,
point-treatment TMLE, and longitudinal TMLE for end-of-study, survival, and competing-risk
outcomes.

The package is alpha software. The public workflow is intentionally explicit:

```text
study design -> typed estimand -> identified effect -> estimation method -> causal result
```

That order keeps column roles, identification assumptions, and estimator settings from being
mixed into one constructor. Unsupported combinations fail before nuisance fitting.

## Documentation

| Document | Purpose |
| --- | --- |
| [User guide](docs/user-guide.md) | Point, intervention, longitudinal, and method recipes |
| [Migration guide](docs/migration.md) | Complete old-to-new argument map and examples |
| [Technical appendix](docs/methodology.md) | Estimands, influence curves, remainders, and evidence |
| [Architecture invariants](docs/architecture-invariants.md) | Cross-module scientific and engineering constraints |
| [Evidence](docs/evidence.md) | Test-enforced evidence for every registered estimand |
| [Public API redesign](docs/public-api-redesign.md) | Accepted object model and ordered implementation plan |

The complete index is [docs/README.md](docs/README.md).

## Install

`cleverly` is not on PyPI yet, so install from source:

```bash
pip install "git+https://github.com/esbraun/cleverly-tmle.git"
pip install "cleverly[all] @ git+https://github.com/esbraun/cleverly-tmle.git"
```

The core depends on NumPy, SciPy, scikit-learn, narwhals, and joblib. The `all` extra adds pandas,
polars, LightGBM, and matplotlib.

## Quickstart

Declare the design and the causal estimand separately, inspect identification if needed, then fit:

```python
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

An ordinary fit returns the result directly. There is no result-set wrapper to unwrap. The result
retains its typed `identified_effect`, normalized `method`, and structured `parameter_keys`, and
those records survive `save()`/`load()`.

For a randomized study with no adjustment variables, make that identifying claim explicitly:

```python
study = CausalStudy(
    frame[["Y", "A"]],
    design=PointTreatment(outcome="Y", treatment="A", randomized=True),
)
result = study.estimate(ATE(), outcome_learner="glm", treatment_learner="glm")
```

## Typed causal questions

The root package exposes typed estimands instead of string-driven estimator branches:

- arm contrasts: `ATE`, `ATT`, `ATC`, `RiskRatio`, and `OddsRatio`;
- means and population interventions: `CounterfactualMean`, `NaturalCourseMean`,
  `PopulationAttributableRisk`, and `PopulationAttributableFraction`;
- point-treatment interventions: `RegimeMean`, `RegimeContrast`, `ModifiedTreatmentPolicy`,
  `ModifiedTreatmentPolicyEffect`, `IncrementalMean`, and `IncrementalEffect`;
- projections and specialized effects: `MSMProjection` and `ControlledDirectEffect`;
- longitudinal regimens: `RegimeMean`, `RegimeContrast`, and `MSMProjection` with a
  `LongitudinalTreatment` design.

The [user guide](docs/user-guide.md) shows each family. The string target registry remains an
implementation detail used by the evidenced analytic engines; it is not a second beginner-facing
computational path.

## Methods and configuration

`"tmle"` is the ordinary method preset. Advanced choices use immutable groups:

```python
from cleverly import CrossFitting, Inference, ModelSpec, Runtime, TMLEMethod

method = TMLEMethod(
    models=ModelSpec(outcome_learner="glm", treatment_learner="glm"),
    cross_fitting=CrossFitting(n_folds=5, learner_folds=3),
    inference=Inference(alpha=0.10, simultaneous=False),
    runtime=Runtime(random_state=7, n_jobs=1),
)
result = effect.estimate(method=method)
```

Common shortcuts such as `n_folds=`, `alpha=`, and `random_state=` normalize into those same
objects. Collaborative TMLE and DR-TMLE are selected with `CollaborativeTMLEMethod` and
`DRTMLEMethod`; they are methods for compatible identified effects, not alternative study types.
If a normalized option cannot apply to the selected design, estimation raises
`MethodConfigurationError` before constructing an engine. In particular, longitudinal studies
refuse point-only controls such as `n_bootstrap=` rather than discarding them.

## Longitudinal treatment

```python
from cleverly import CausalStudy, LongitudinalTreatment, RegimeContrast
from cleverly.datasets import make_longitudinal

frame, truth = make_longitudinal(n=2_000, seed=11)
study = CausalStudy(
    frame,
    design=LongitudinalTreatment(
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    ),
)
result = study.estimate(
    RegimeContrast({"always": 1, "never": 0}, reference="always"),
    outcome_learner="glm",
    pseudo_learner="glm",
    treatment_learner="glm",
    n_folds=3,
    learner_folds=3,
    random_state=0,
)
print(result.summary())
```

An outcome sequence declares survival; a mapping of cause to outcome sequence declares competing
risks. Parameter keys retain regimen, horizon, and cause as fields rather than recovering them by
parsing display labels.

## Persistence

```python
from cleverly import load

result.save("analysis.npz")
restored = load("analysis.npz")
assert restored.parameter_keys == result.parameter_keys
assert restored.method == result.method
assert restored.validate() == result.validate()
```

The format stores arrays plus allow-listed structured metadata; it does not pickle arbitrary
objects. A learner given as a library name round-trips exactly. A learner given as an object — a
scikit-learn estimator, a `SuperLearner` — is recorded by identity instead, as custom callables
are: the file is still written and every cached analysis still replays, but the restored slot
refuses use rather than silently substituting a default and refitting.

## Post-fit assessment

```python
validation = result.validate()  # cached artifacts only; no refits
support = result.diagnostics.support()
scores = result.diagnostics.score_equations()
print(result.sensitivity.run_all().summary())
```

The same facade covers point and longitudinal results. Each operation declares the artifacts and
cost it requires; combined reports distinguish a question that does not apply from one whose
required derivation or fitted artifact is unavailable. Completed cache-only assessments and their
replayability metadata survive `save()` / `load()`.

## What is implemented and refused

The analytic engines cover the point and longitudinal estimands listed above, multi-valued
treatments, dynamic/stochastic regimes, continuous modified treatment policies, incremental
interventions, marginal structural models, missing outcomes, controlled direct effects,
observation weights, cross-fitting, C-TMLE, and DR-TMLE.

Graph identification, direct Riesz learning, EP learning, front-door, IV, mediation, and transport
are not placeholders. Requests for them are refused with a capability reason before fitting. See
[the public API plan](docs/public-api-redesign.md) and
[how to read a refusal](docs/methodology.md#how-to-read-a-refusal).

## Development

```bash
uv venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy src/cleverly
pytest -m "not slow" -q
pytest -m slow -q
pytest tests/unit/test_documentation_*.py -q
python benchmarks/bench_tmle.py
python -m benchmarks.numba.cli --config benchmarks/configs/sandbox.json
```

Ruff formats Python fences in Markdown, so run it across the whole tree. The documentation tests
resolve links and compile Python fences; behavioral claims belong in ordinary unit, integration,
or end-to-end tests.

Hosted GitHub Actions is currently out of budget and is not a correctness signal. Local runs of
the commands above are the release gate. One editable install is shared between Git worktrees;
`tests/conftest.py` detects a checkout/import mismatch and explains how to correct it.

## Citing

There is no DOI yet. Cite the repository and the relevant papers listed in the
[technical appendix](docs/methodology.md#references).

## License

MIT
