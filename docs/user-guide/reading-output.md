# Read cleverly's output

Every tutorial prints the same three kinds of output: the study description, the summary of the
causal question, and the summary of an estimate. This page says what each line means, so that a
tutorial can stay on its own question. [Results, inference, and assessment](results-assessment.md)
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
| Follow-up starts | time zero: the moment that lines up eligibility, the treatment decision and the start of follow-up. Only information recorded before it goes into the adjustment |
| Strategies | the treatment plans compared, such as offering a program or not |
| Versions | what each plan delivers in practice, in the same order as the plans |
| Outcome and Horizon | what is measured, and by when |
| Intercurrent events | what to do about events after time zero that change what the outcome means, such as death before readmission or leaving the health plan (see below) |
| Interference unit | the unit within which one person's treatment may change another's outcome. "Individual patient" says that no patient's treatment affects another patient |
| Rationale | why you believe the conditions for a causal answer hold in this study |

The `fingerprint` is a short code computed from every field. Each result keeps the fingerprint of
the description it was fitted under, so two results with different fingerprints answer different
questions even when their numbers look comparable.

### Intercurrent events

An intercurrent event happens after time zero and changes what the outcome means. Each way of
handling one asks a different question, and the tutorials name the way they use in brackets.

| strategy | what it does | example |
| --- | --- | --- |
| treatment policy | use the outcome whatever happened in between | analyze the offer regardless of how many contacts the patient completed |
| composite | fold the event into the outcome | score death before day 30 as the worst transition score |
| competing event | report each event as it happens in the real world, with the other events left as they are | readmission and death, where a patient who dies can no longer be readmitted |
| hypothetical | ask what would have happened had the event not occurred | the readmission risk had every patient stayed enrolled, or had death been prevented |

A hypothetical strategy needs extra conditions, because the data never show the world it
describes: the event must depend only on recorded information, and every kind of person must have
had a real chance of avoiding it. [Time-to-event outcomes](survival.md) shows how to ask each of
the event questions.

## The causal question

`study.identify(question)` checks that the data can answer the question and prints a summary before
any model is fitted. The heading names the question with a short code in brackets, such as
`Causal question (ate)` for an average effect or `(ey_regime)` for the mean outcome under each
treatment plan over time.

| part | what to read |
| --- | --- |
| Question, Outcome, Treatment, Adjusted for | the question in words and the role of each column. Check that every column under "Adjusted for" was recorded before the treatment |
| Answered by | how the data answer the question. "adjusting for covariates" means comparing people with the same recorded characteristics; "adjusting for history" does the same at each treatment time, using everything recorded up to then |
| What each treatment time adjusts for | for a treatment given more than once, the columns each decision is compared within |
| Conditions the answer rests on | each condition in plain words, and whether the data can check it. `no` means only your knowledge of the study can support it; `partly` means a check after the fit can find a problem but cannot prove the condition |
| Methods | the estimators that can fit this question, by the name you pass as `method=` |

The methods table lists only the methods that fit the question. A note under it says how many
others cannot, and `summary(detail="full")` gives each reason. The tutorials use `tmle`, targeted
maximum likelihood estimation: it fits a model of the outcome and a model of who got the
treatment, then corrects the outcome model's estimate with the treatment model so its confidence
interval is valid when the models are good enough. The other entries are alternatives or
comparisons; the [technical reference](../technical-reference/index.md) describes each.

## The estimate

An estimate summary has a header that repeats the question and the fitted models, a table of
estimates, a table that says what each row estimates, and numbered notes. Each row has a name
built from three parts.

```text
ate_regimen[always vs never, readmission @ t=2]
 |    |       |                 |           |
 |    |       plans compared    cause       time point
 |    kind of plan
 what is estimated
```

| part | values you will meet |
| --- | --- |
| what is estimated | `ey`: the mean outcome under one plan. `ate`: the difference between two plans, the first minus the reference. `risk`: the chance of the event by a time. `cif`: the cumulative incidence, the chance that one cause happens first by a time when other events compete. `rr` and `or`: the risk ratio and the odds ratio |
| kind of plan | none for one treatment given once with fixed levels, such as `ate` or `ey[1]`. `regimen`: a plan over time, or a baseline treatment held over follow-up. `policy`: a change to each person's dose. `ipsi`: a change to each person's odds of treatment. Other names, such as `ey_learned_rule`, follow the same pattern |
| inside the brackets | the plan or plans, a cause when several events compete, and `t=` the time point on the grid, counted from 1 |

The "What each row estimates" table under the estimates spells out every row name in words, such
as "probability of readmission by time 2 if every unit followed plan always". Read that table
whenever a name is unfamiliar; it is the authority for the page you are on.

### Labels and notes

A `support` column appears when an estimate is `experimental`, and a note below the table says
why. `supported` means a registered simulation study has checked the method's intervals on that
kind of fit; `experimental` means no such study covers it yet, and the interval carries no
checked coverage. The label does not say the estimate is wrong. Some warnings say a study
"licenses" an interval, which means the same as `supported`: a study covers it.
[Read the support label](results-assessment.md#read-the-support-label) lists where the label
appears and what each reason means.

Other notes appear only when something changes how to read a number: for example, that a model was
fitted and scored on the same rows, that a cap on extreme weights changed some people, or that an
estimate rests on few people (`SparseInfluenceWarning`).
