# Run time and memory

A fit trains many models: one or more for the treatment and the outcome, on each fold, often as an
ensemble of several candidates. On a large table, or with many treatment plans or time points, that
takes a while and can need a lot of memory. Work through five steps before a long run: count what
it will fit, shrink what you can, spread the work over your cores, set a memory limit, and save
progress on the longest checks so a stopped run can resume.

## Count the fits before you run

`planned_fits` takes the arguments you would pass to `estimate` and counts the models the fit
would train, without training any:

```python
plan = effect.planned_fits(random_state=3)
print(plan.total_fits, plan.total_candidate_fits)  # models, and candidates inside ensembles
print(plan.summary())  # the same count by model and by candidate
```

It raises the same error the fit would raise before its first model, so it also checks your
settings. [Count the model fits before a long run](methods-learners.md#count-the-model-fits-before-a-long-run)
explains the count and when it is an upper bound.

Two other calls count in the same way. `result.diagnostics.planned_refits(...)`, called with the
arguments you would pass to `refute()`, counts the refits a refutation would make
([count the refits](refutation.md#count-the-refits-before-a-long-test)).
`variable_importance_plan` counts the fits of a [ranking of candidate exposures](variable-importance.md).

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

`n_jobs=` sets the most jobs a fit runs at the same time, such as folds fitted side by side. It is
a setting of the `Runtime` group, and the [settings reference](settings.md) lists the fits that
read it. A fit with `n_jobs=1`, the default, runs one job at a time.

Inside each job, a learner such as a random forest or a gradient-boosted model can start its own
threads. `cleverly` keeps each model to one thread by default, so the parallel work happens across
jobs rather than inside a model and the jobs do not compete for the same cores. Change that limit
with `cleverly.learners.set_thread_limit`, or for one block of code with `thread_limit`:

```python
from cleverly.learners import get_thread_limit, set_thread_limit, thread_limit

get_thread_limit()  # 1, one thread per model fit
set_thread_limit(4)  # each model fit may use up to four threads
with thread_limit(None):  # no limit inside this block: each library's own default
    pass
set_thread_limit(1)
```

Raise the thread limit when `n_jobs` is 1 and a single heavy learner does most of the work. Keep
it at 1 when `n_jobs` already uses your cores, since threads times jobs should not exceed them.

## Set a memory limit

Fits whose memory grows with the data check it before they train anything. They estimate what the
whole fit needs and stop with an error when that is more than the memory the machine has free.
`max_memory_bytes=` sets that limit yourself, higher or lower; where the system does not report
free memory, as on macOS, only a limit you set is checked. The limit changes how a fit runs, never
the number it returns.

Each kind of fit reads the limit in its own way, and its page gives the details:

| fit | what the limit controls | read more |
| --- | --- | --- |
| treatments over time (`LTMLE`, `SDR`, `IPSI`) | the check before the first model | [over time](longitudinal.md); [memory](../technical-reference/longitudinal-tmle.md#memory) |
| exact event times | the check, and how many groups of people it keeps between passes | [exact event times](survival.md#exact-event-times-without-a-grid); [memory](../technical-reference/continuous-time-survival.md#memory) |
| a dose-response curve | the memory its numerical integration may use | [dose-response](../technical-reference/dose-response.md) |
| a targeting curve, which checks how well a score sorts people by their effect | the size of each block of its half-sample draws | [targeting curves](../technical-reference/conditional-effects.md#targeting-curves) |

The check counts the arrays the fit itself holds, measured with simple regression learners. A
heavier learner's own memory is not counted, so a fit that passes can still run out of memory.

A result also keeps the answers of its checks, so a second call returns at once. Set
`result.assessment_cache.max_bytes` to cap that memory, as
[limit the memory of saved answers](results-assessment.md#limit-the-memory-of-saved-answers)
explains.

## Resume a stopped run

Two kinds of run repeat a full fit many times and can save each one as it finishes:

- A refutation with many draws, such as a 400-draw generated-outcome test. Pass
  `checkpoint=RefutationCheckpoint("refute-draws")`, from `cleverly.validation`, and run the same
  call again after a stop: it loads the saved draws and refits only the rest
  ([save the draws as they finish](refutation.md#save-the-draws-as-they-finish)).
- A simulation study of your own (`CoverageStudy`), which takes `checkpoint=` and runs draws in
  parallel with `n_jobs=` ([run a coverage study of your own setup](results-assessment.md#run-a-coverage-study-of-your-own-setup)).

A refutation run over a ranking of candidate exposures saves each candidate's draws in its own
subfolder of the checkpoint folder.
