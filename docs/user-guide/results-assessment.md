# Use and save a result

A fitted result (`CausalResult`) holds every estimate, its interval, the models and settings that
produced it, and what the checks need. This page reads it from code, and saves and loads it.
[Read cleverly's output](reading-output.md) reads the printed summary, and
[confidence intervals](inference.md) explains the intervals.

## Use a summary in code, and plot a result

A summary is also data. It is a string, so `print` and `str` give the text. The same object holds
the parts, so a program does not need to parse the text.

```python
summary = result.summary()
name = list(result.estimates)[0]
row = summary.table().row(name)
print(summary.heading)
print(name, row["estimate"].value, row["95% CI"].text)
print(summary.note_codes)
```

`summary.table()` returns the estimates table, and `.row(name)` returns its cells, each with the raw
`value` and the printed `text`. `summary.facts` maps each header label to its value, and
`summary.notes` lists the notes. A check's `summary().reading` gives its one-sentence verdict.
`summary.as_html()` returns an escaped HTML fragment, and a notebook shows it without a call.
`repr(result)` is one short line that names the method and the estimates.

To see a result instead of reading its table, draw it with `cleverly.plotting`: the [plot
results](../examples/plots.ipynb) tutorial draws the estimate, the overlap and balance checks, and
the sensitivity analyses. The overlap plot (how the predicted chances of treatment compare between
the groups), the learner plot and the calibration plot also draw a fit of a treatment over time, one
panel per time point, and a fit that adds rows from outside the study, such as an external control
group. [Plotting](../api/plotting.md) lists which results each plot draws.

## Result structure

A result (`CausalResult`) is a read-only mapping from each row name of the summary, such as `ate` or
`ey[1]`, to that row's estimate (`ParameterEstimate`).

```python
names = list(result.estimates)
table = result.to_frame()
covariance = result.covariance(names)
point = result[names[0]]
treated = next(alias for alias, key in result.parameter_keys.items() if key.value == 1)

print(point.psi, point.std_error, point.ci, point.pvalue)
```

`point.psi` is the point estimate, and `std_error`, `ci` and `pvalue` are the other three columns of
the summary table. `result.covariance(names)` gives how the estimates vary together.

A fit that withholds its interval, such as a collaborative fit with the default `"greedy"` strategy,
keeps its point estimate and refuses `std_error`, `ci` and `pvalue` with `CapabilityError`. The
[inference status](../technical-reference/inference.md#inference-status) table lists each case, and
[parametric per-arm
intervals](../technical-reference/collaborative-tmle.md#parametric-per-arm-intervals) gives the
conditions under which a collaborative `"oat"` fit reports one. The property
`result.inference_status` gives the status of the fit, and `point.supplies_inference` gives it for
one estimate. On a fit that withholds its interval, `point.plugin_std_error` and
`point.plugin_interval` give the standard error and interval that the usual formula would report,
which no theory or study shows to be valid for this fit. Each is a diagnostic, and neither is a
confidence statement.

`result.parameter_keys` maps each row name to a `ParameterKey`, which holds the parts of the name as
fields, such as the treatment level (`key.value`) and the subgroup (`key.stratum`). Select rows in
code by those fields, as the `treated` line above does, and not by parsing the printed name. On a
fit with baseline strata, `key.stratum == ("high",)` picks the estimate of the `high` stratum.

`result.estimate` is the sole `ParameterEstimate` only when the result holds one estimate, and
`result.psi()` is its numeric point value. With several, index the row name or pass a name to `psi`.

## Save and load a result

```python
from cleverly import load

result.save("analysis.joblib")
restored = load("analysis.joblib")
assert restored.parameter_keys == result.parameter_keys
assert type(restored.method) is type(result.method)
assert type(restored.method.models.outcome_learner) is type(result.method.models.outcome_learner)
```

Change the file name to your own. The three `assert` lines only show that the loaded result names
the same estimates, method and models; you do not need them. The file holds the whole result,
including unfitted copies of the outcome and treatment models, so a loaded result can still run
the checks that refit its models. `save()` writes the result with `cloudpickle`. It saves an
importable function or class by reference, and a lambda, a closure, or an object defined in a
notebook or in `__main__` by value, so a result keeps the treatment rules, models and other
functions you wrote inline. A by-value object loads only under the Python minor version that saved
it.

The loader is `joblib.load`, which uses pickle. Loading can execute arbitrary code, so load only
trusted files. Use an environment with the same Python minor version and compatible sklearn and
third-party estimator versions.

`result.save()` records the `cleverly` version that wrote the file. When another version saved it,
`cleverly.load()` warns with `VersionMismatchWarning` and loads the file as saved, with no
migration. Its support labels, check reports and saved check answers can differ from what this
version computes, so fit the analysis again with this version. Development snapshots between two
releases share one version string, so the check does not see a change between them.

## Go deeper

### What a loaded result can redo

Print `restored.replayability.summary()` to see what a loaded result can redo: show its stored
checks, re-estimate from its stored models, refit the models, or rerun on changed data. A note names
any part the result lacks. Read `result.diagnostics.capabilities` before you run a check on a loaded
file. Each row declares what it needs to redo, and `available` is `False` when the file cannot
supply it. The row names the missing part (its slot), so you learn the answer before the check runs.

A result can also hold an estimator that was copied or changed after the fit into a setup the
package refuses, such as a cross-fitted continuous outcome without its known range (`q_bounds`).
Such a result still gives its point estimates and the checks that rerun the correction step on its
stored models, such as the curve over caps on extreme weights. Checks that fit new models, such as
`refute()` and `simulated_confounding()`, read `unavailable`, and `estimator.refit()` raises
`CapabilityError` before it fits anything. Fit the analysis again under a setup the package accepts.
[Replay-only
unavailability](../technical-reference/scope-and-refusals.md#replay-only-unavailability) lists each
refused setup and the code the result records, and [truncation
stability](../technical-reference/validation-methods.md#truncation-stability) covers the weight-cap
curve of a guarded `DRTMLE` fit.

The saved file carries the answers of earlier checks (the assessment cache). A result you derive
with `dataclasses.replace` does not. The cache key records the check and its arguments, and it
records nothing about the result that answered them, so a derived result starts with an empty cache
of its own.

A curve from `fit.effect_curve()` on a [conditional effect](conditional-effects.md) saves and loads
the same way as a result. The loaded curve keeps its coefficients, their joint covariance, the
declared modifiers and basis, and any band, so `curve.predict(rows, interval=True)` evaluates it at
new rows after loading.

### Limit the memory of saved answers

A result keeps every check answer it has computed in its assessment cache, so a repeated call
returns at once. Most answers are a few kilobytes. The inputs to the missing-confounder bound that
`result.sensitivity.elements()` returns, and that `run_all()` and `assess()` compute along the way,
hold four numbers per row. At a million rows that is about 32 MB for each estimate and each choice
of `nu2_estimator`, the estimator of the bound's weight term ([sensitivity
analysis](sensitivity.md)). A derived risk ratio holds one number per row. Many distinct requests on
a large fit can therefore add up.

Set `result.assessment_cache.max_bytes` to a number of bytes to cap the memory of all saved answers.
When the total passes the cap, the cache drops the answers it can compute again exactly from the
fitted result, least recently used first, and a later call recomputes a dropped answer with
bit-for-bit the same numbers. It never drops an answer that refits a model, draws a random seed of
its own, fits a learner or runs work in parallel, because computing that again could give different
numbers. Those answers count toward the cap and can hold the cache above it.
`result.assessment_cache.info()` lists each answer with its size and whether the cap may drop it,
and `result.assessment_cache.evict()` drops every answer it may drop now.

There is no cap by default. A cap trades memory for the time to recompute, and `save()` writes only
the answers that survive it. `result.assessment_cache.clear()` removes every answer, including the
refits; a repeated unseeded refit then draws a new seed, as it would after any removal.

### Save a fit over time

`result.save()` and `cleverly.load()` carry a longitudinal result through a round trip. The
artifact keeps the folds, the fitted mechanisms, the sequential steps, the targeting state, the
causal metadata, and the truncation replay recipe. It keeps the plan that the fit used. A result
fitted on an adapted panel keeps its `PanelLayout` and `unit_ids`, so `to_long_panel()` works on
the loaded result.

`save()` writes the result with `cloudpickle`. A rule, a policy density, a map, or a learner that
you define inline is saved by value. That covers a lambda, a closure, and a function defined in a
notebook or in `__main__`. A loaded result evaluates each of them again. An importable function
or class is saved by reference, so its module must import on the machine that loads the artifact.

A by-value object loads only under the Python minor version that saved it. Load such an artifact
with the same minor version, for example 3.13 for an artifact that 3.13 wrote.
`tests/unit/test_long_panel.py::test_policies_defined_in_main_survive_a_save_in_a_fresh_interpreter`
saves a lambda rule, a closure density, and an `each_node` rule from `__main__`. It loads them in
a new interpreter and compares the assignments and the density.

[What a loaded result can redo](results-assessment.md#what-a-loaded-result-can-redo) states the
shared contract for every result, including the assessment cache and the capability rows that a
restored artifact refuses. `tests/unit/test_serialization.py`'s
`test_longitudinal_result_retains_the_complete_fitted_graph_and_assessment` checks the round trip
against one cross-fitted, weighted, censored fit.
