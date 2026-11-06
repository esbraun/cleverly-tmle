# cleverly

[![PyPI](https://img.shields.io/pypi/v/cleverly.svg)](https://pypi.org/project/cleverly/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-1565c0.svg)](https://www.python.org/)
[![CI](https://github.com/esbraun/cleverly-tmle/actions/workflows/ci.yml/badge.svg)](https://github.com/esbraun/cleverly-tmle/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-087f8c.svg)](https://github.com/esbraun/cleverly-tmle/blob/main/LICENSE)

**[Read the documentation](https://esbraun.github.io/cleverly-tmle/)** for the getting-started
guide, the user guide, worked examples, and the API reference. To see a complete analysis first,
start with [one treatment, one outcome](https://esbraun.github.io/cleverly-tmle/examples/point-treatment-tmle.html).

`cleverly` is a Python library for estimating what a treatment, an offer, or a policy did, from
data where nobody assigned it at random. It answers questions such as "did the program lower
readmissions?", "what would sales be if every customer got the discount?", or "which patients
should get the offer?", and it reports each answer with a confidence interval.

In such data, a plain comparison of treated and untreated people mixes the treatment's effect with
differences in who got it. Those differences come from confounders: variables, such as age or
prior risk, that affect both who gets the treatment and the outcome. `cleverly` adjusts for the
confounders you recorded. Before it fits anything, it writes down what must be true for the
answer to count as a causal effect, such as "no important confounder is missing". After the fit,
it checks the models and asks how strong a missing confounder would have to be to change the
answer.

Most estimates come from targeted maximum likelihood estimation (TMLE). TMLE fits two models: an
outcome model, which predicts the outcome from the treatment and the confounders, and a treatment
model, which predicts who got the treatment. When the outcome model is imperfect, its estimate of
the effect is biased. A correction step uses the treatment model to remove that bias. Either model
can be any scikit-learn-compatible estimator, including gradient boosting or a stack of several
models.

*The name comes from TMLE's "clever covariate", the variable that drives that correction step.
Even the jokes in this library are targeted.*

> **Versions:** releases use `0.2.N` versions, and the public API can still change between them,
> so pin a version for reproducible work.

## Key features

- Effects of a one-time treatment: the average effect for everyone, for the treated, or for the
  untreated, as a difference, a risk ratio, or an odds ratio, with two or more treatment levels.
  Missing outcomes and missing treatments can be declared and handled.
- How an effect changes with a few baseline columns, such as age: a declared curve of the
  effect with intervals, a band, and a test of whether the effect varies at all, or, with no
  shape declared, a flexible curve (the EP learner) without intervals. A validation study covers
  the declared curve's intervals; the band and the flexible curve are labelled experimental.
- Effects of policies: treating only some people by a rule, following a rule learned from the
  data, raising each person's chance of treatment, or shifting a continuous dose or price.
- Treatments over time: treatments that repeat, outcomes measured at the end or every period, and
  time until an event, including events that another event, such as death, can prevent.
- Out-of-fold model fitting, so flexible machine-learning models can be used: each row's
  predictions come from models trained without that row, as in cross-validation. Intervals can
  also account for sampling weights, and for groups whose members are not independent, such as
  patients of one team.
- Checks after the fit: an overlap check of whether similar people received each option, attempts
  to break the result (for example, shuffling the treatment, so the estimate should be about
  zero), and sensitivity analyses for a missing confounder.
- Refusals instead of approximations: a request the library cannot answer correctly fails before
  any model is fit, and the error says why.
- Other estimators beside TMLE, for comparison or for special cases, such as collaborative TMLE,
  doubly robust TMLE, and, over time, sequential doubly robust estimation, sequential
  g-computation and weighting, and the parametric g-formula.
- pandas, polars, Arrow-backed pandas, and `pyarrow.Table` inputs, through
  [narwhals](https://narwhals-dev.github.io/narwhals/).

The [technical reference](https://esbraun.github.io/cleverly-tmle/technical-reference/) lists every
method the package ships, with its source papers and its evidence.

## Install

```bash
python -m pip install cleverly
```

Python 3.11 or newer is required. Add pandas, polars, and plotting support with the `all` extra:

```bash
python -m pip install "cleverly[all]"
```

Other scikit-learn-compatible models, such as XGBoost or LightGBM, work once you install them
yourself. The [installation guide](https://esbraun.github.io/cleverly-tmle/getting-started/installation.html)
covers development installs and version pinning.

## Quick start

This example uses simulated data in which the true average effect is 1.5.

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

The estimate summary gives the answer, an average effect of 1.513 with a 95% interval from 1.422
to 1.603, which contains the true 1.5:

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

The summary of the question, printed first, lists four conditions the answer rests on: the
treatment is well defined, one person's treatment does not change another's outcome, no
confounder is missing, and every kind of person had some chance of each treatment (overlap). The
data cannot confirm the first three, and they can only partly check overlap.

The simulated data come from a linear outcome model and a logistic treatment model, so the two
models above are correct. Because they are also simple, fitting and predicting on the same rows
still gives a valid interval, so this fit turns off out-of-fold predictions (`cross_fit=False`).
On your own data the right models are unknown. Use flexible models and keep the default
out-of-fold fitting. For a continuous outcome, that fit needs the smallest and largest values the
outcome can take, declared with `Targeting(q_bounds=(lower, upper))`; the
[full quickstart](https://esbraun.github.io/cleverly-tmle/getting-started/quickstart.html) explains
the settings. The intervals are valid in large samples when the models are reasonably accurate and
the overlap condition holds.

The final assessment reports that no check needs attention. Its E-value says that a missing
confounder would need a risk ratio of 3.52 with both the treatment and the outcome, beyond the
recorded confounders, to explain the effect away. The larger that number, the stronger a hidden
confounder must be to overturn the answer.

To choose a different estimator, call `effect.available_methods()`, which lists the methods that
can answer the question and says why the others cannot. The
[methods and learners guide](https://esbraun.github.io/cleverly-tmle/user-guide/methods-learners.html)
shows how to configure each one.

## Examples

Each example is a complete analysis on simulated data with a known true answer, except the last,
which uses real birth records.

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

The [examples index](https://esbraun.github.io/cleverly-tmle/examples/) has the full list, sorted by
the problem each one solves.

## How cleverly is tested

More than 100 registered simulation studies check the shipped methods on data where the true answer
is known. Each one tests whether the estimates center on the truth and whether the 95% intervals
contain it as often as they should. Where an established package computes the same quantity, such
as the R packages `tmle`, `tmle3`, `ltmle`, `lmtp`, `drtmle`, `ctmle`, and `survtmle` or the
Python packages zEpid and EconML, the study also compares `cleverly`'s answers with that package's.
Many methods have no such package, and their studies rely on the known truth and on the behavior
the method's theory predicts. For a few methods the comparison with another package has not run
yet, and their pages say so. `cleverly` marks a result experimental when no study covers its
setup. The
[validation grid](https://esbraun.github.io/cleverly-tmle/technical-reference/method-evidence/validation-grid.html)
lists every study and its results.

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

Bug reports and questions go to the
[issue tracker](https://github.com/esbraun/cleverly-tmle/issues).
[CONTRIBUTING.md](https://github.com/esbraun/cleverly-tmle/blob/main/CONTRIBUTING.md) gives the
development setup, the checks each change needs, and the commit style.

## Citing

There is no DOI yet. Cite the version or repository commit you used, and the method papers that
the [technical reference](https://esbraun.github.io/cleverly-tmle/technical-reference/) and the
[references page](https://esbraun.github.io/cleverly-tmle/references.html) list.

## License

`cleverly` is under the
[MIT License](https://github.com/esbraun/cleverly-tmle/blob/main/LICENSE). The R comparison
runners under `tests/canonical/` are under the
[GNU General Public License v3.0](https://github.com/esbraun/cleverly-tmle/blob/main/tests/canonical/LICENSE),
because they call reference R packages in the same process and one of those packages is AGPL-3.
No published distribution includes them, so an installed copy of `cleverly` is MIT alone.
