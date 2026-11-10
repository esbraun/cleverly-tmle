# Read cleverly's output

Every tutorial prints the same three kinds of output: the study description, the summary of the
causal question, and the summary of an estimate. [Results, inference, and assessment](results-assessment.md)
covers the estimate summary and its checks in more depth.

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
[examples index](../examples/index.md#find-the-example-for-your-problem) matches each situation to
its method.

## The estimate

An estimate summary has a header that repeats the question and the fitted models, a table of
estimates, a table that says what each row estimates, and numbered warnings and notes. Each row's name reads
as follows.

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

### Labels, warnings and notes

A `support` column appears when an estimate is `experimental`, and a note below the table says
why. `supported` means a registered simulation study covers this configuration, such as the same
method, design and kind of plan, and its checks passed. `experimental` means no study covers it,
or a stated condition applies, such as too few clusters; the interval then has no checked
coverage. The label does not say the estimate is wrong.

The word "licenses" in a warning means "covers". For example, the E-value says how strongly a
missing confounder would need to be tied to both the treatment and the outcome to explain the
effect away, and its "limit" version does the same for the end of the confidence interval nearer
no effect. "No validation study licenses the E-value of the limit" means no study covers that
number. You can still report it, with a plain caveat such as "this E-value describes this sample;
no simulation study has checked it for this kind of analysis".
[Read the support label](results-assessment.md#read-the-support-label) lists where the label
appears and what each reason means.

Other notes appear only when something changes how to read a number: for example, that a model was
fitted and scored on the same rows, that a cap on extreme weights (the inverse chances of treatment that the correction step uses) changed
some people's weights, or that an
estimate's interval leans on few rows (`SparseInfluenceWarning`).
