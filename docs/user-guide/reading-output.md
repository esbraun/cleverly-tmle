# Read cleverly's output

Every tutorial prints the same three kinds of output: the study description, the summary of the
causal question, and the summary of an estimate. This page reads each one, and the support label
that says whether a simulation study has checked an estimate's interval.
[Confidence intervals](inference.md) covers how the intervals are built, and [use and save a
result](results-assessment.md) covers the result object in code.

## The study description

A study description writes down the study you would run if you could assign the treatment
yourself: who is in it, when follow-up starts, what is compared, and what is measured. `cleverly`
stores it as a `StudyProtocol` and prints one line for each field. You write every field in your
own words; `cleverly` records them with the result and does not check them against the data.

```python
from cleverly import StudyProtocol

protocol = StudyProtocol(
    target_population="Adults discharged home from a participating hospital",
    eligibility=["Age 18 or older", "Discharged alive"],
    time_zero="Hospital discharge, before the navigation offer",
    treatment_strategies=["Offer navigation", "Usual discharge support"],
    treatment_versions=["Two navigator calls within 30 days", "No navigator calls"],
    outcome="Readmission within 30 days",
    horizon="30 days after discharge",
    intercurrent_event_handling=["Treat death before readmission as a competing event"],
    interference_unit="Individual patient",
    assumption_rationale=["The recorded baseline variables cover the common causes"],
)
print("\n".join(protocol.summary_lines()))
print(protocol.fingerprint)
```

| printed label | what it records |
| --- | --- |
| Population | the people the answer applies to |
| Eligibility | who enters the study |
| Follow-up starts | time zero: the moment that lines up eligibility, the treatment decision and the start of follow-up. Only information recorded before it should be adjusted for, that is, used to compare like with like |
| Strategies | the treatment plans compared, such as offering a program or not |
| Versions | what each plan delivers in practice, in the same order as the plans |
| Outcome and Horizon | what is measured, and by when |
| Intercurrent events | what to do about events after time zero that change what the outcome means, such as death before readmission or leaving the health plan (see below) |
| Interference unit | the unit within which one person's treatment may change another's outcome. "Individual patient" says that no patient's treatment affects another patient |
| Rationale | why you believe the conditions for a causal answer, listed under "The causal question" below, hold in this study |

The `fingerprint` is a short code computed from every field. Each result keeps the fingerprint of
the description it was fitted under. Different fingerprints mean the descriptions differ in at least
one field, even a reworded rationale, so compare the printed fields before you compare the numbers.

### Intercurrent events

An intercurrent event happens after time zero and changes what the outcome means. Each way of
handling one asks a different question, and a tutorial names the composite or hypothetical
strategy in parentheses where it uses one.

| strategy | what it does | example |
| --- | --- | --- |
| treatment policy | use the outcome whatever happened in between | analyze the offer regardless of how many contacts the patient completed |
| composite | fold the event into the outcome | score death before day 30 as the worst transition score |
| competing event | report each event as it happens in the real world, with the other events left as they are | readmission and death, where a patient who dies can no longer be readmitted |
| hypothetical | ask what would have happened had the event not occurred | the readmission risk had every patient stayed enrolled, or had death been prevented |

A hypothetical strategy needs extra conditions, because the data never show the world it
describes: whether the event happens must depend only on what was recorded before it, and every
kind of person must have had a real chance of avoiding it. `cleverly` cannot check either one.
Censoring columns on the design already ask the "had everyone stayed in follow-up" question, and
`eliminate=` on the design, such as `TimeToEvent(..., eliminate=["death"])`, asks "had death been
prevented". [Time-to-event outcomes](survival.md) shows how to ask each of
the event questions.

## The causal question

`study.identify(question)` checks that the design can pose the question and prints a summary before
any model is fitted; whether the answer is causal still rests on the conditions it lists. The heading names the question with a short code in brackets, such as
`Causal question (ate)` for an average effect or `(ey_regime)` for the mean outcome under each
treatment plan over time.

| part | what to read |
| --- | --- |
| Question, Outcome, Treatment, Adjusted for (Baseline over time) | the question in words and the role of each column. Check that every adjusted-for column was recorded before the treatment |
| Answered by | how the data answer the question. "adjusting for covariates" means comparing people with the same recorded characteristics; "adjusting for history" does the same at each treatment time, using everything recorded up to then; "randomized assignment" uses the recorded chances of a randomized treatment |
| What each treatment time adjusts for | for a treatment given more than once, the columns each decision is compared within |
| Conditions the answer rests on | each condition in plain words, and in the `checkable` column whether the data can check it. `no` means only your knowledge of the study can support it; `partly` means a check after the fit can find a problem but cannot prove the condition |
| Methods | the estimators that can fit this question, by the name you pass as `method=` |

The methods table lists the methods that fit the question, and a note says how many others cannot;
`summary(detail="full")` gives each reason. The tutorials use `tmle`, targeted maximum likelihood
estimation. It fits a model of the outcome and a model of who got the treatment, then corrects the
outcome model's estimate with the treatment model. Its interval is valid when both models
predict well, and its estimate stays unbiased in large samples if either model is right. Start
with `tmle`. The others answer special situations, such as a treatment model that a strong
predictor makes extreme (`collaborative_tmle`); the
[examples index](../examples/index.md#find-the-example-for-your-problem) groups the examples by
situation, and its topic pages match each situation to its example.

## The estimate

An estimate summary has a header that repeats the question and the fitted models, a table of
estimates, a table that says what each row estimates, and numbered warnings and notes.

### Reading a summary

The [quickstart](../getting-started/quickstart.md) fit asks how much a care-navigation offer
(the treatment) changes patients' transition score (the outcome), and prints this:

```text
                            Effect estimate (TMLE)
==============================================================================
Question:      average effect of        Rows:                            2,000
               transition_navigation    Treated share:                   0.506
               on transition_score (1   Out-of-fold predictions:            no
               compared with 0)         Outcome model:       linear regression
Outcome:       transition_score         Treatment model:   logistic regression
               (continuous)             Probability limits:  [0.01471, 0.9853]
Adjusted for:  discharge_risk,
               prior_utilization,
               medication_burden, age
Method:        targeted maximum
               likelihood (TMLE)
==============================================================================
      estimate   std. error           95% CI    p-value
------------------------------------------------------------------------------
ate      1.513        0.046   [1.422, 1.603]    <0.0001
==============================================================================
Notes:
[1] The interval of ate is widened on the side where a skewed outcome makes
    misses more likely, so it need not be centred on the estimate. Its t
    distribution has 250 or more degrees of freedom, so it is close to the
    normal one. The p-value is the level at which the interval would just
    reach the null value.
```

The row `ate` is the average treatment effect: the change in the mean score if every patient got the
offer, compared with no patient. The `estimate` is on the scale of the outcome, here score points.
The `std. error` measures how much the estimate would move from sample to sample, and the `p-value`
tests whether the effect is zero. The `95% CI` is the 95% confidence interval: a range built so
that, if the analysis were repeated on many new samples, about 95 in 100 such ranges would contain
the true effect. That holds in large samples when the models are reasonably accurate.

In words a manager can follow: offering navigation raised the mean score by about 1.5 points, and
the data are consistent with an effect between 1.42 and 1.60 points. That sentence is causal only if
the adjusted covariates include everything that drove both the offer and the score, and no data can
check that. [Sensitivity analysis](sensitivity.md) asks how strong a missing one would have to be.

The header records the question and how it was fitted. The fit uses two models: the
outcome model predicts the score from the offer and the covariates, and the treatment model predicts
each patient's chance of the offer from the covariates. The method, targeted maximum likelihood
estimation (TMLE), starts from the outcome model's predictions and adds a correction step that uses
the treatment model to remove most of the bias the outcome model leaves, when the treatment model is
close to right. The correction weights each patient by one over their predicted chance of the option
they got.

`Out-of-fold predictions: no` means both models were fitted and scored on every row
(`cross_fit=False`). That is sound for simple models like these linear and logistic regressions.
With flexible models, such as random forests or boosting, it can make the interval too narrow, and
the summary then warns. For those, keep the default, cross-fitting: each row's predictions come from
models trained without that row, like out-of-fold predictions in cross-validation. A cross-fitted
continuous outcome needs the range its values can take by how it is measured, such as
`q_bounds=(0, 100)` for a 0-to-100 score, passed to `estimate()`. The quickstart's score has no such
range, so fitting on every row with simple models is its only route ([quickstart, on your own
data](../getting-started/quickstart.md)).

The `Probability limits` cap each predicted chance, so that no patient gets an extreme weight. When
the cap changes some patients' weights, a note says so; no such note prints here, so the cap changed
no patient.

Numbered notes follow the table, and they appear only when something changes how to read a number,
with any warning first; a summary with no notes has nothing to flag. The quickstart's note says its
interval is widened on the side a skewed outcome misses on. A note can also say that an estimate is
experimental (see [the support label](#read-the-support-label) below), or that a row has no interval
and why. `summary(detail="full")` adds the assumptions under which the effect can be read as caused
by the treatment, one per line, and the settings that change how to read the numbers, such as the
limits on predicted chances and how many patients reached them.

### Row names

Each row's name reads as follows.

```text
ate_regimen[always vs never, readmission @ t=2]
 |    |       |                 |           |
 |    |       plans compared    cause       time point
 |    kind of plan
 what is estimated
```

| part | values you will meet |
| --- | --- |
| what is estimated | `ey`: the mean outcome under one plan (`ey1` and `ey0` for a yes-or-no treatment). `ate`: the difference between two plans, the first named minus the reference named after `vs`. `att` and `atc`: the effect among the treated and among the untreated. `risk`: the chance of the event by a time. `cif`: the cumulative incidence, the chance that one cause happens first by a time when other events compete. `rr` and `or`: the risk ratio and the odds ratio |
| kind of plan | none for one treatment given once at fixed levels, such as `ate`, or `ey[high]` with several levels. `regime`: a rule for one treatment given once, such as offering only to high-risk patients. `regimen`: a plan over time, or a baseline treatment held over follow-up. The question summary of a plan over time is headed `ey_regime` even though its rows say `regimen`. `policy`: a change to each person's dose. `ipsi`: a change to each person's odds of treatment. Other names, such as `ey_learned_rule`, follow the same pattern |
| inside the brackets | the plan or plans, named as in your code; a cause when several events compete; and `t=`, the position of the time point: on a grid of check days 30, 60 and 90, `t=2` is day 60 |

The "What each row estimates" table under the estimates spells out every row name in words, such
as "probability of readmission by time 2 if every unit followed plan always". Read that table
whenever a name in your output is unfamiliar. A `TimeToEvent` result also reports the actual grid
time in the `time` column of `curve()` and `to_frame()`.

[One treatment, one outcome](../examples/point-treatment-tmle.ipynb) prints a full summary and explains
it line by line.

### Labels, warnings and notes

A `support` column appears when an estimate is `experimental`, and a note below the table says
why. [Read the support label](#read-the-support-label) explains the two labels.

The word "licenses" in a warning means "covers". For example, the E-value says how strongly a
missing confounder would need to be tied to both the treatment and the outcome to explain the
effect away, and its "limit" version does the same for the end of the confidence interval nearer
no effect. "No validation study licenses the E-value of the limit" means no study covers that
number. You can still report it, with a plain caveat such as "this E-value describes this sample;
no simulation study has checked it for this kind of analysis".

On a fit without clusters a note often says that an interval need not be centred on the estimate:
it was built on the log-odds scale of a proportion, from two proportions' intervals, or widened on
the side where a skewed outcome makes misses more likely
([confidence intervals](inference.md#confidence-intervals-the-bootstrap-and-bands)).
Other notes appear only when something changes how to read a number: for example, that a model was
fitted and scored on the same rows, that a cap on extreme weights (the inverse chances of treatment that the correction step uses) changed
some people's weights, or that an
estimate's interval leans on few rows (`SparseInfluenceWarning`).

### Read the support label

Every estimate carries a support label, `supported` or `experimental`. It answers one question: has
a simulation study checked the coverage of this kind of interval for this kind of fit? Coverage is
the share of intervals that hold the true value across many simulated data sets; a 95% interval
should cover about 95% of the time. The package keeps a fixed list of the setups its simulation
studies checked, each one a combination of method, kind of treatment and data, and question. You do
not request the label, and no argument turns it off.

A `supported` label means that a simulation study checked a setup like your fit and passed. The
label names the study, and `result["ate"].support.limits` lists any conditions the study set, such
as the models it used for the outcome and the treatment or the smallest number of rows. An
`experimental` label means that the package computes the estimate but makes no coverage claim: no
study checked this setup, or a stated condition applies, such as too few clusters (groups of related
rows). The label carries the reason. The quickstart's `ate` reads `supported`, with no extra
conditions.

An `experimental` label does not mean the estimate is wrong. It means no simulation has checked the
number you read: the interval, a band over several estimates ([bands](inference.md#bands-that-cover-several-estimates-at-once)), or a value computed
from other estimates, such as a ratio. Report such an estimate with a plain caveat, such as "no
simulation study has checked this interval for this kind of analysis". [Run a coverage study of your
own setup](inference.md#run-a-coverage-study-of-your-own-setup) shows how to check it yourself. A `supported`
label says how the interval behaved in simulations; it does not show that the effect is causal.

Read the label with `result["ate"].support.label`, or in the `support` column of
`result.to_frame()`. The summary prints a `support` column and a note only when an estimate is
experimental. [Where the label appears](#where-the-label-appears) lists every place, and the
[validation grid](../technical-reference/method-evidence/validation-grid.md) lists the studies
behind the labels.

## Go deeper

### Where the label appears

| place | what it shows |
| --- | --- |
| `repr(estimate)` | the marker `[experimental]` after the estimate |
| `result.summary()` | a `support` column, only when an estimate is experimental, and an `experimental` note below the table that says in plain words why. `summary(detail="full")` prints each reason under Settings |
| `estimate.support` | a `Support` record with `label`, `reason`, `evidence`, `row` (the support entry) and `limits` |
| `estimate.to_dict()` and `result.to_frame()` | `support` on every row. An experimental row adds `support_reason` and `support_evidence`. A row with limits adds `support_limits` |
| `estimate.bootstrap.support` and `bootstrap_support` | the label of the percentile interval, which a support entry can cover apart from the usual interval built from the standard error (the Wald interval) |
| `result.simultaneous.support` | the label of the band |
| `result.config.support_key` | the key that matched the fit to a support entry. `result.save()` and `cleverly.load()` keep it with every label |

The `evidence` of a `supported` estimate holds the short names of its studies. The `evidence` of an
experimental estimate holds the ids of the [roadmap](../roadmap.md) items, such as `X54`, that would
supply the missing evidence. The reason of an experimental label names the support entry it read,
such as `T01.continuous`.

### Labels of ratios and other derived estimates

A derived estimate, computed from other estimates, such as a ratio or a contrast, is `supported`
only when its support entry lists derived estimates (the `derived` kind) and every input is
`supported`. A restricted mean (the average time free of the event up to a set horizon) read from a
grid of check times needs its own kind instead. There is one kind for a single treatment plan's mean
and one for a difference of two plans, since a study that checked the difference does not check
each plan's mean.
