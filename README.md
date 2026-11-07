*It is named for TMLE's clever covariate. Even the jokes in this toolbox are targeted.*

# cleverly

[![PyPI](https://img.shields.io/pypi/v/cleverly.svg)](https://pypi.org/project/cleverly/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-1565c0.svg)](https://www.python.org/)
[![CI](https://github.com/esbraun/cleverly-tmle/actions/workflows/ci.yml/badge.svg)](https://github.com/esbraun/cleverly-tmle/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-087f8c.svg)](https://github.com/esbraun/cleverly-tmle/blob/main/LICENSE)

`cleverly` estimates what a treatment, an offer, or a policy did, from data where nobody assigned
it at random, with a confidence interval for each answer. It adjusts for the confounders you
recorded, mostly with targeted maximum likelihood estimation (TMLE). Afterward it checks how
strong a missing confounder would have to be to change the result. Read the **[documentation](https://esbraun.github.io/cleverly-tmle/)**, or start with a
[complete analysis](https://esbraun.github.io/cleverly-tmle/examples/point-treatment-tmle.html).

> Releases use `0.2.N` versions, and the public API can still change between them. Pin a version
> for reproducible work.

## Key features

- One-time treatments: average effects for everyone, the treated, or the untreated, as a
  difference, risk ratio, or odds ratio, with missing outcomes and treatments handled.
- Effects that vary with baseline columns such as age: declared curves with intervals, or a
  flexible curve (the EP learner) without them.
- Policies: treating by a rule, following a learned rule, raising the chance of treatment, or
  shifting a dose.
- Treatments over time, including time until an event with competing events such as death.
- Out-of-fold model fitting for flexible machine-learning models, with sampling weights and
  clustered data.
- Checks after the fit: overlap, attempts to break the result, and sensitivity to a missing
  confounder.
- Refusals instead of approximations: a request the library cannot answer correctly fails before
  any model is fit, and the error says why.
- Other estimators for comparison or special cases, such as collaborative TMLE and the parametric
  g-formula.
- pandas, polars, and `pyarrow.Table` inputs, through
  [narwhals](https://narwhals-dev.github.io/narwhals/).

The [technical reference](https://esbraun.github.io/cleverly-tmle/technical-reference/) lists every
method with its source papers and evidence.

## Install

```bash
python -m pip install cleverly          # Python 3.11 or newer
python -m pip install "cleverly[all]"   # adds pandas, polars, and plotting
```

The [installation guide](https://esbraun.github.io/cleverly-tmle/getting-started/installation.html)
covers other learners, development installs, and version pinning.

## Quick start

Simulated data with a true average effect of 1.5:

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_linear_ate

frame, truth = make_linear_ate(n=2_000, seed=7)

# 1. Describe the data: the outcome, the treatment, and the confounders to adjust for.
study = CausalStudy(
    frame,
    design=PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W1", "W2", "W3", "W4"),
    ),
)

# 2. Name the question: the average effect of A = 1 compared with A = 0 (ATE).
#    The summary lists the conditions the answer rests on and the methods that can fit it.
effect = study.identify(ATE(reference=0))
print(effect.summary())

# 3. Fit the outcome model and the treatment model, then apply the correction step.
result = effect.estimate(
    outcome_learner=LinearRegression(),
    # An unpenalized logistic regression, which matches how the data were simulated.
    treatment_learner=LogisticRegression(C=np.inf, max_iter=1000),
    cross_fit=False,
    random_state=7,
)
print(result.summary())
# The truth is known only because the data are simulated.
print(result["ate"].ci, "population ATE:", truth["ate"])

# 4. Check the fit and how sensitive the answer is to a missing confounder.
print(result.assess().summary())
```

The estimate is 1.513 with a 95% interval from 1.422 to 1.603, which contains the truth:

```text
                            Effect estimate (TMLE)
==============================================================================
Question:      average effect of A on   Rows:                            2,000
               Y (1 compared with 0)    Treated share:                   0.506
Outcome:       Y (continuous)           Out-of-fold predictions:            no
Adjusted for:  W1, W2, W3, W4           Outcome model:       linear regression
Method:        targeted maximum         Treatment model:   logistic regression
               likelihood (TMLE)        Probability limits:  [0.01471, 0.9853]
==============================================================================
      estimate   std. error           95% CI    p-value
------------------------------------------------------------------------------
ate      1.513        0.046   [1.422, 1.603]    <0.0001
==============================================================================
```

The question summary lists the conditions the answer rests on: a well-defined treatment, no
interference between people, no missing confounder, and overlap (every kind of person had some
chance of each treatment). The data cannot confirm the first three.

The two models here match how the data were simulated, so `cross_fit=False` is safe. On your own
data, use flexible models and keep the default out-of-fold fitting; a continuous outcome then needs
`Targeting(q_bounds=(lower, upper))`. The
[full quickstart](https://esbraun.github.io/cleverly-tmle/getting-started/quickstart.html)
explains the settings. The assessment's E-value of 3.52 means a missing confounder would need a
risk ratio of 3.52 with both treatment and outcome to explain the effect away.

`effect.available_methods()` lists the estimators that can answer the question and why the others
cannot; the
[methods and learners guide](https://esbraun.github.io/cleverly-tmle/user-guide/methods-learners.html)
shows how to configure them.

## Examples

Each example uses simulated data with a known answer, except the last (real birth records).

- [One treatment, one outcome](https://esbraun.github.io/cleverly-tmle/examples/point-treatment-tmle.html):
  why a simple treated-versus-untreated comparison misleads, and what happens when a model is wrong.
- [Flexible models and shared teams](https://esbraun.github.io/cleverly-tmle/examples/cross-fitting.html):
  gradient boosting, out-of-fold predictions, and patients whose results move together.
- [Three offer policies](https://esbraun.github.io/cleverly-tmle/examples/interventions.html):
  offering a program only to high-risk patients, raising each patient's chance of an offer, or
  raising how much each patient gets.
- [Who should get the offer](https://esbraun.github.io/cleverly-tmle/examples/treatment-rules.html):
  learning a treatment rule and scoring it on patients it was not learned from.
- [Readmission when death can come first](https://esbraun.github.io/cleverly-tmle/examples/longitudinal-survival.html):
  time until an event, with a competing event and dropout.
- [Weekly offer policies](https://esbraun.github.io/cleverly-tmle/examples/sequential-offers.html):
  a treatment repeated every week, where earlier offers change later behavior.
- [Smoking and twins' birth weight](https://esbraun.github.io/cleverly-tmle/examples/twins-causal-inference.html):
  a full analysis of real records, where no true answer is known.

The [examples index](https://esbraun.github.io/cleverly-tmle/examples/) lists them all.

## How cleverly is tested

More than 100 registered simulation studies check the shipped methods on data with a known answer:
whether estimates center on the truth and 95% intervals cover it as often as they should. Where
an established package computes the same quantity (R `tmle`, `tmle3`, `ltmle`, `lmtp`, `drtmle`,
`ctmle`, `survtmle`; Python zEpid and EconML), the study compares answers too. Some comparisons
have not run yet, and those pages say so. A result is marked experimental when no study covers its
setup. The
[validation grid](https://esbraun.github.io/cleverly-tmle/technical-reference/method-evidence/validation-grid.html)
lists every study.

## Documentation

- [Getting started](https://esbraun.github.io/cleverly-tmle/getting-started/): install, a first
  fit, and how to read a result.
- [Workflow](https://esbraun.github.io/cleverly-tmle/workflow.html): one analysis from the
  question to the report.
- [User guide](https://esbraun.github.io/cleverly-tmle/user-guide/): data, questions, models,
  methods, treatments over time, results, and refusals.
- [Examples](https://esbraun.github.io/cleverly-tmle/examples/): complete analyses to adapt.
- [Technical reference](https://esbraun.github.io/cleverly-tmle/technical-reference/): for
  statisticians, the exact quantities, conditions, papers, and evidence behind each method.
- [Python API](https://esbraun.github.io/cleverly-tmle/api/): every public object and its
  signature.

## Contributing

Report bugs and questions on the
[issue tracker](https://github.com/esbraun/cleverly-tmle/issues). See
[CONTRIBUTING.md](https://github.com/esbraun/cleverly-tmle/blob/main/CONTRIBUTING.md) for setup,
checks, and commit style.

## Citing

There is no DOI yet. Cite the version or commit you used and the method papers listed in the
[technical reference](https://esbraun.github.io/cleverly-tmle/technical-reference/).

## License

MIT. The R comparison runners under `tests/canonical/` are
[GPL-3.0](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/LICENSE) because they
call AGPL-3 R packages in-process; they are not distributed, so an installed `cleverly` is MIT
alone.
