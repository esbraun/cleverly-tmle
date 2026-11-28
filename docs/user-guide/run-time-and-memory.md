# Run time and memory

A fit trains many models. It splits the rows into *folds*, parts of the data held out in turn so
each model is scored on rows it did not see, and it trains a model for the treatment and one for
the outcome on each fold. By default each of those models is an *ensemble*: several candidate
learners, combined by how well each predicts held-out rows. On a large table, or with many
treatment plans or time points, that takes a while and can need a lot of memory. Work through five
steps before a long run: count what it will fit, fit less, use more cores, set a memory limit, and
save progress on the longest checks so a stopped run can resume.

## Count the model fits before a long run

Call `planned_fits` with the same arguments you would pass to `estimate`. It reads your data,
folds, and learners and counts the models the fit would train, without training any of them.

```python
plan = effect.planned_fits(random_state=3)
print(plan.total_fits, plan.total_candidate_fits)
```

```text
20 360
```

With the default learners, a binary-treatment ATE fits two models, one for the treatment and one
for the outcome, on each of the 10 outer folds: 20 fits. Each of those fits is an ensemble of three
candidates, and the ensemble trains every candidate once per inner fold and once more on all the
rows it was given, so 3 × (5 + 1) = 18 candidate fits per model. That makes 360 candidate fits, of
which 120 are random forests. `print(plan.summary())` lists the fits by model and by candidate.
Lasso runs its own 5-fold search for a penalty inside each of its fits, and the summary shows those
folds beside the count rather than adding them to it.

`study.planned_fits(ATE(), random_state=3)` gives the same count straight from the study. Pass
`random_state` when you can. The count then uses the exact folds the fit will use. Without it the
count describes one random draw of the folds, and a count that depends on the folds can differ
from the draw the fit makes.

Some methods stop when a rule is met, so the count is the most they can fit. `plan.exact` is then
`False`, `plan.min_fits` gives the fewest fits, and the printed plan says "at most". On the same
data, `effect.planned_fits("drtmle", random_state=3)` counts at most 3170 fits and at least 230,
because its alternating updates can stop before their limit of 50 rounds. Adaptive IPTW updates,
bootstrap replicates, and folds where a regression target turns out constant give upper bounds for
the same kind of reason. The `why` column of `plan.to_frame()` names the reason for each row.

If the fit would refuse your data or settings before training its first model, `planned_fits`
raises the same error. To cut the count, use fewer outer folds, fewer inner folds, or a smaller
library, as [Two fold layers](methods-learners.md#two-fold-layers) describes.

`variable_importance` fits one model set per candidate exposure. `variable_importance_plan`,
called with the same arguments apart from `keep_fits`, counts those fits in the same way, with each
row labelled by its candidate ([rank candidate exposures](variable-importance.md)).
`result.diagnostics.planned_refits(...)`, called with the arguments you would pass to `refute()`,
counts the refits a refutation would make ([count the
refits](refutation.md#count-the-refits-before-a-long-test)).

## Fit less

The count multiplies the number of folds, the number of models per fold and the candidates in each
ensemble, so each of these is a lever:

- **Folds.** `n_folds=` sets the outer folds, and an ensemble's own inner folds come on top of
  them. [Two fold layers](methods-learners.md#two-fold-layers) says how to reduce either.
- **Learners.** The default ensemble trains several candidates, including random forests, on
  each fold. A smaller library, or one learner, cuts the count most;
  [methods and learners](methods-learners.md) shows how to pass your own.
- **Refutation on a sample.** `refute(result, sample_limit=20_000)` runs a refutation on a random
  20,000 of the fit's units. It answers a narrower question, as
  [refute a sample of a large fit](refutation.md#refute-a-sample-of-a-large-fit) explains.
- **Kept fits.** `variable_importance(..., keep_fits=False)` keeps only the estimates and the
  summary, so memory no longer grows with the number of candidates
  ([large runs](variable-importance.md#large-runs)).

## Use more cores

`n_jobs=` sets the most jobs a fit runs at the same time, such as folds fitted side by side. Pass
it to `estimate`; the [settings reference](settings.md) lists the fits that read it. With
`n_jobs=1`, the default, the fit runs one job at a time.

Some learners, such as gradient-boosted models and linear models built on numerical libraries,
start several threads inside one fit. `cleverly` limits those threads to one per model fit by
default, so a parallel run spreads across jobs rather than inside each model. A random forest's
threads are set by its own `n_jobs` argument, which this limit does not change; the default
library builds its forests with one.

```python
from cleverly.learners import get_thread_limit, set_thread_limit, thread_limit

get_thread_limit()  # 1, one thread per model fit
set_thread_limit(4)  # each model fit may use up to four threads
with thread_limit(None):  # no limit inside this block: each library's own default
    pass
set_thread_limit(1)
```

The limit applies in your Python process. Jobs that `n_jobs` starts as separate processes keep one
thread per model, so raising the limit helps mainly when `n_jobs` is 1 and one heavy learner does
most of the work.

## Set a memory limit

Two kinds of fit check their memory before they train anything: treatments over time (`LTMLE`,
`SDR`, `IPSI`) and fits with exact event times. They estimate what the whole fit needs and stop
with an error when that is more than the memory the machine has free. Pass `max_memory_bytes=` to
`estimate` to set the limit yourself, higher or lower; where the system does not report free
memory, as on macOS, only a limit you set is checked. Every other fit refuses the keyword, as the
[settings reference](settings.md#seeds-run-labels-and-parallel-jobs-runtime) shows. The limit changes how a fit runs, never the number it
returns.

| fit | what the limit controls | read more |
| --- | --- | --- |
| treatments over time | the check before the first model | [over time](longitudinal.md); [memory](../technical-reference/longitudinal-tmle.md#memory) |
| exact event times | the check, and how much of the data it keeps in memory between passes | [exact event times](survival-exact-times.md#exact-event-times-without-a-grid); [memory](../technical-reference/continuous-time-survival.md#memory) |
| a targeting curve, which checks how well a score sorts people by their effect | how many of its random draws it holds in memory at once, set by its own `max_memory_bytes=` keyword | [targeting curves](../technical-reference/conditional-effects.md#targeting-curves) |

Each check counts the arrays the fit itself holds. A learner's own working memory is not counted,
so a fit that passes can still run out of memory.

A result also keeps the answers of its checks, so a second call returns at once. Set
`result.assessment_cache.max_bytes` to cap that memory, as
[limit the memory of saved answers](results-assessment.md#limit-the-memory-of-saved-answers)
explains.

## Resume a stopped run

Two kinds of run repeat a full fit many times and can save each one as it finishes:

- A refutation with many draws, such as a test that simulates 400 new outcomes and refits each.
  Pass `checkpoint=RefutationCheckpoint("refute-draws")`, from `cleverly.validation`, and run the
  same call again after a stop: it loads the saved draws and refits only the rest
  ([save the draws as they finish](refutation.md#save-the-draws-as-they-finish)).
- A simulation study of your own (`CoverageStudy`), which takes a checkpoint object
  (`ReplicationCheckpoint`) as `checkpoint=` and runs draws in parallel with `n_jobs=`
  ([run a coverage study of your own setup](inference.md#run-a-coverage-study-of-your-own-setup)).

A refutation run over a ranking of candidate exposures saves each candidate's draws in its own
subfolder of the checkpoint folder.
