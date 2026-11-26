# User guide

The user guide explains the choices between the quickstart and a defensible applied analysis, one
stage at a time: identification (the design and the causal question), estimation, inference, and
checking the result. Use the [Python API](../api/index.md) for signatures and the [technical
reference](../technical-reference/index.md) for derivations and evidence.

The public workflow has four visible stages:

```text
CausalStudy -> identify(typed estimand) -> IdentifiedEffect -> estimate(method) -> CausalResult
```

The study design owns the column roles. The estimand owns the causal question. The method owns
learning, targeting, inference, and runtime settings. That separation is what lets `cleverly`
refuse an unsupported combination before it fits any nuisance model.

The examples here are explanatory. The behavioural guarantees live in the fast tests and
registered validation studies, not in an example.

## Find the page for your task

Pick the heading by the stage of the analysis. A treatment decided at several times, a time to an
event, and a trial pooled with external data each have their own heading for describing the data,
choosing what to compare and fitting. Intervals, checks and saving stay under their stage headings
for every design.

### Start here

- [What cleverly can estimate](capabilities.md) lists every question the package answers and the page for each.
- [Read cleverly's output](reading-output.md) reads the printed study description, question summary and estimate summary, its row names, and the `supported` or `experimental` label on each estimate.

### Identification

- [Data and study design](data-design.md) declares the outcome, the treatment, the adjustment set, weights, clusters, missing values and doses for a treatment decided once.

#### [Estimands](topics/estimands.md)

An estimand is the causal quantity you ask for, written as an object such as `ATE()`, and choosing it is the second half of identification after the design. These pages cover a treatment decided once. Pick the page by the kind of question; a treatment decided at several times has its own plans under [treatments over time](topics/treatments-over-time.md).

- [Average effects and fixed plans](estimands.md)
- [Learned treatment rules](learned-rules.md)
- [Dose shifts and odds changes](policies.md)
- [Conditional effects](conditional-effects.md)
- [Rank candidate exposures](variable-importance.md)

### Estimation

- [Methods, learners, and cross-fitting](methods-learners.md) chooses the estimator, the prediction models, and how the rows are split into folds.
- [Settings reference](settings.md) looks up a setting by name: its default and the fits that read it.
- [Run time and memory](run-time-and-memory.md) counts the fits before a long run and cuts its cost.

### Inference

- [Confidence intervals, the bootstrap and bands](inference.md) covers every interval a fit reports, for any design: the level, the bootstrap, bands, intervals of ratios and differences computed after the fit, and a coverage simulation of your own setup.

### Checking a result

- [Diagnostics](assessment.md) checks the fitted models, overlap and covariate balance, the weights and the correction step, and reads the combined report of every check, in numbers; [the plot guide](plots.md) lists the figures.
- [Refutation](refutation.md) refits the analysis on changed data whose answer is known.
- [Sensitivity analysis](sensitivity.md) asks how strong a missing confounder would have to be to change the answer (robustness value, E-value, worst-case bounds), and how far missing outcomes that differ from the observed ones could move it.

### Reporting

- [Which plot answers which question](plots.md) matches each question to the `cleverly.plotting` function that draws it, for an estimate or for a check; every figure is listed there.
- [Use and save a result](results-assessment.md) takes the estimates into code, such as a DataFrame, and saves and loads a result.

### Longitudinal, time-to-event and trial data

- [A small trial with external data](trial-with-external-data.md) pools a small randomized trial with external records on the same outcome.

#### [Treatments over time](topics/treatments-over-time.md)

These pages cover a treatment decided at several times, such as a weekly offer, where earlier treatments change later covariates and later decisions. They describe the data, choose the plans to compare, and fit them. Intervals, diagnostics, refutation, sensitivity analysis and saving work as for any fit, and each of those pages has a section for fits over time. An outcome that is a time to an event has its own pages under [time-to-event outcomes](topics/time-to-event.md).

- [Describe data over time](longitudinal.md)
- [Plans and policies over time](longitudinal-plans.md)
- [Estimators over time](longitudinal-estimators.md)

#### [Time-to-event outcomes](topics/time-to-event.md)

These pages cover an outcome that is whether and when an event happens, such as readmission or churn, when some people leave follow-up. They declare the data, choose what to report, and fit it. Intervals, diagnostics, refutation, sensitivity analysis and saving use the general pages, which note what differs for these fits.

- [Risks at set check times](survival.md)
- [Exact event times](survival-exact-times.md)
- [Competing events](survival-competing-events.md)
- [Fitting choices for event times](survival-fitting.md)

```{toctree}
:hidden:
:caption: Start here
:maxdepth: 1

What cleverly can estimate <capabilities>
Read cleverly's output <reading-output>
```

```{toctree}
:hidden:
:caption: Identification
:maxdepth: 1

Data and study design <data-design>
Estimands <topics/estimands>
```

```{toctree}
:hidden:
:caption: Estimation
:maxdepth: 1

Methods, learners, and cross-fitting <methods-learners>
Settings reference <settings>
Run time and memory <run-time-and-memory>
```

```{toctree}
:hidden:
:caption: Inference
:maxdepth: 1

Confidence intervals, the bootstrap and bands <inference>
```

```{toctree}
:hidden:
:caption: Checking a result
:maxdepth: 1

Diagnostics <assessment>
Refutation <refutation>
Sensitivity analysis <sensitivity>
```

```{toctree}
:hidden:
:caption: Reporting
:maxdepth: 1

Which plot answers which question <plots>
Use and save a result <results-assessment>
```

```{toctree}
:hidden:
:caption: Longitudinal, time-to-event and trial data
:maxdepth: 1

A small trial with external data <trial-with-external-data>
Treatments over time <topics/treatments-over-time>
Time-to-event outcomes <topics/time-to-event>
```
