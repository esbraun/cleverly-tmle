# Examples

## How the examples work

Each example is a complete analysis that you can run and adapt. It loads a dataset, estimates what
a treatment, offer, or policy did, tests whether the result holds up, and says what you can tell a
manager. Every example but one uses simulated data, so each page can compare its estimate with the
true answer. The [TWINS example](twins-causal-inference.ipynb) uses real birth records. If you
are new to causal inference, read [one treatment, one outcome](point-treatment-tmle.ipynb) first. It explains each step that the
other pages reuse, and [Read cleverly's output](../user-guide/reading-output.md) explains the labels,
tables and row names that every page prints.

Most examples use targeted maximum likelihood estimation (TMLE). It fits two models: an outcome
model, which predicts the outcome from the treatment and the patient's characteristics, and a
treatment model, which predicts who got the treatment. A correction step then uses the treatment
model to remove the bias that the outcome model leaves, giving more weight to patients whose own
treatment was unlikely.

## Find the example for your problem

The Start here table, and each topic page, pair a tutorial with the situation it fits and the mistake
it shows.

### Start here

The first page runs a whole analysis on simulated data and explains each step that the other
tutorials reuse. The second runs one on real records, where no true answer is known.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| a treatment given once and an outcome measured later | [one treatment, one outcome](point-treatment-tmle.ipynb) | a simple comparison of treated and untreated patients mixes the treatment's effect with who received it. The page also shows what happens to the estimate when one or both models are wrong |
| a full analysis of real records, where no true answer is known | [smoking and twins' birth weight](twins-causal-inference.ipynb) | adjusting for a variable that the treatment itself changes, here gestational age, removes part of the effect |

### One treatment, outcome measured once

Your data record one treatment decision per person, such as an offer at discharge, and one outcome
measured once, later. Pick the heading that matches your question.

#### [Overall effect of one or several treatments](topics/average-effect.md)

These tutorials estimate what giving a treatment to everyone would change on average, compared with
giving it to no one or with current practice. One of them ranks several practices at once.

- [A randomized pilot: TMLE with known treatment chances](randomized-pilot.ipynb)
- [Gain over current practice: TMLE for population impact](population-impact.ipynb)
- [Which discharge practices matter: ranking candidate exposures with TMLE](ranking-exposures.ipynb)

#### [Effects by characteristic or group](topics/effects-by-group.md)

These tutorials break the effect down by a recorded characteristic, such as age, or by groups of
people, such as those who live alone.

- [Who gains most from a coupon: an effect that changes with age](conditional-effects.ipynb)
- [Does navigation help patients who live alone more? Effects by group, with intervals](subgroup-effects.ipynb)

#### [Learn whom to treat from the data](topics/targeting-rules.md)

This tutorial learns a rule that decides who gets the treatment, from recorded characteristics, and
estimates how well the rule would do on new people. It also shows a random offer when there are
fewer places than people who qualify.

- [Who should get the offer: learned rules and random offers with TMLE](treatment-rules.ipynb)

#### [Compare set policies and doses](topics/amounts-and-policies.md)

These tutorials compare policies you specify in advance, such as offering the treatment only to
high-risk patients, giving more or less of it, or how renewals change with the size of a discount.

- [Three offer policies: TMLE for policy effects](interventions.ipynb)
- [More contacts and more service time: TMLE for policy effects](schedules-and-intensity.ipynb)
- [A trend across three contact schedules: TMLE for a trend line](msm-projections.ipynb)
- [Renewal discounts: how the chance of renewing changes with the discount](dose-response.ipynb)

### Repeated treatments, or time until an event

Your data record a treatment decided more than once per person, or the time until an event such as
readmission or churn. Look here first, whatever your question.

#### [Treatments repeated over time](topics/repeated-over-time.md)

These tutorials follow a treatment that is decided again and again, such as a weekly offer or a
monthly price, where earlier treatments change later behavior and later decisions.

- [Weekly offer policies: longitudinal TMLE and g-computation](sequential-offers.ipynb)
- [Weekly orders under repeated offers: longitudinal TMLE](weekly-outcomes.ipynb)
- [Two decisions after discharge: a learned treatment rule with longitudinal TMLE](two-decision-rules.ipynb)
- [Three offers after discharge: g-formula and blip models](model-based-over-time.ipynb)
- [Monthly prices: longitudinal TMLE for a continuous treatment](sequential-pricing.ipynb)

#### [Risk of an event by a given day (churn, readmission)](topics/time-to-event.md)

These tutorials estimate the risk of an event by a given day, when some people leave follow-up and
another event, such as death, can come first. They follow hospital readmission; the same analysis
applies to the time until a customer churns.

- [Readmission when death can come first: longitudinal TMLE](longitudinal-survival.ipynb)
- [Readmission by day 30, 60 and 90: longitudinal TMLE](discharge-to-readmission.ipynb)
- [A medicine with two effects on readmission and death: separable effects with TMLE](separable-effects.ipynb)

### Data issues and model choices (any question)

These problems can arise with any question: imperfect data, and choosing and checking the models
inside a fit. The tutorials show them with one treatment and one outcome.

#### [Missing answers, samples, or a trial plus outside records](topics/missing-and-combined-data.md)

These tutorials handle data that a textbook analysis assumes away. Outcomes go missing, a sample is
drawn with unequal chances, or a small randomized trial is combined with outside records that differ
from it.

- [Missing survey answers: TMLE with a model of who responds](survey-nonresponse.ipynb)
- [An interview of a sample of patients: TMLE with sampling weights](sampling-weights.ipynb)
- [A small pilot with outside records: adaptive TMLE](trial-with-external-data.ipynb)

#### [Choosing models and checking results](topics/models-and-checks.md)

These tutorials choose the prediction models inside a fit, such as gradient boosting. They keep the
interval honest when the models are flexible, when patients share a team, or when one variable
nearly decides who gets the treatment. The last one checks and explains a result with figures.

- [Choosing the prediction models: Super Learner ensembles in TMLE](learner-choice.ipynb)
- [Flexible models and shared teams: cross-validated TMLE](cross-fitting.ipynb)
- [A strong predictor of treatment: collaborative TMLE](collaborative-tmle.ipynb)
- [Plot results: see what an estimate's numbers hide](plots.ipynb)

## The health program

Most examples follow one imaginary health program. A regional health plan offers
**care-transition navigation** to adults who leave a participating hospital for home: a bedside
plan for the transition and two scheduled navigator contacts within 30 days. Hospital teams decided
who got the offer, so the data are observational, and each analysis adjusts for patient
characteristics recorded before each decision. The exception is the
[randomized pilot](randomized-pilot.ipynb), which drew the offer at random with a recorded chance.
If your program can still choose at random who gets an offer, do that: a randomized offer removes
the need to assume that no unrecorded factor drove it.

The main outcome is a 30-day transition score that patients report. Some pages use a variant: a
yes-or-no successful transition, a top-box rating, staying home for 30 days, a 0-to-100 recovery
score, or a simulated score with no fixed range. Others record an event: readmission by day 30, 60
and 90, or, on the [time-to-event page](longitudinal-survival.ipynb), readmission and death.

## What must hold for a causal answer

Every example, including the retailer and real-data pages below, needs at least these four
conditions for its answer to be causal, written here for the health program's offer:

- consistency: the offer means the same thing for every patient;
- exchangeability: no unrecorded factor (a confounder) drove both the offer and the outcome;
- positivity: every kind of patient had a real chance of each option the question compares. A
  policy that only scales each patient's current chances, such as doubling the odds of an offer,
  does not need it for the treatment;
- no interference: one patient's offer does not change another patient's outcome.

Pages with missing outcomes or dropout add two more: who goes missing depends only on recorded
information, and every kind of patient had a real chance of staying in the data (answering the
survey, or staying enrolled or tracked). The twins page adds that the way its records were
selected, two live births with unequal weights and a stated smoking status, does not bias the
effect.

The simulated data make these conditions true by construction. A real analysis has to argue for
them from how the program ran, and each page says which of them the data cannot check. The
[Oxford causal assumptions chapter](https://www.stats.ox.ac.uk/~evans/APTS/causassmp.html) defines
each one.

## The shared study design

The health-program pages share one study description. It says who is eligible and when follow-up
starts, what the offer and the comparison are, and what the outcome is. When you adapt a page to
your own program, this description is what you change. `navigation_protocol()` in
`cleverly.datasets` returns it as a `StudyProtocol`, and each page changes only the parts its
question changes, with `dataclasses.replace`. Three pages with decisions over time use
`longitudinal_navigation_protocol()`, which includes only patients discharged alive and starts
follow-up at discharge. The [time-to-event page](longitudinal-survival.ipynb) changes it with
`dataclasses.replace`. The team program of the [readmission page](discharge-to-readmission.ipynb)
and the [three-offers page](model-based-over-time.ipynb) call `team_readmission_protocol()` and
`recovery_score_protocol()`, which are built from it.

Two choices hold on every page of this program. The treatment is what the plan assigns (the offer,
or on some pages a navigation intensity, a contact schedule, or service time), not the navigation a
patient completes. Patients take up the offer to different degrees, and the plan controls only the
assignment, so analyzing it keeps the meaning of the treatment the same for everyone. And only
information recorded before each assignment goes into its adjustment. Length of stay and completed
contacts happen after it, and the assignment can change them, so adjusting for them would remove
part of the effect.

:::{dropdown} Details: the full study design
| design element | program definition |
| --- | --- |
| target population | adults discharged home from a participating hospital during the declared enrollment period |
| time zero (when follow-up starts) | after eligibility and baseline measurement, but before navigation assignment |
| treatment strategy | assignment to a standardized transition-navigation offer, not completed participation |
| variations | three schedules of contacts, an assigned navigation intensity, two service times, a randomized pilot, a learned rule or a random offer, or two or three decisions over time, each on its own page |
| standard offer | a bedside transition plan and two scheduled navigator contacts within 30 days |
| comparison | usual discharge support without access to the navigation protocol |
| primary outcome | a 30-day patient-reported transition score. The protocol scores death before day 30 as the worst transition score (a composite strategy) |
| death strategy on each page | the composite strategy needs an outcome with a worst value, such as a bounded score or a top-box indicator (whether the patient gave the best rating). The [trend page](msm-projections.ipynb) and the navigation-intensity raise on the [policy page](interventions.ipynb) use synthetic scores with no worst value, so they use the hypothetical strategy: the score the patient would have had without dying |
| death in the data | under the composite strategy, a death before day 30 enters the data as an observed outcome: the worst score, or not top box. Under the hypothetical strategy, a real analysis treats death as censoring (the outcome goes unseen), and then needs exchangeability and positivity for death given the recorded history. The [time-to-event page](longitudinal-survival.ipynb) fits that strategy with `eliminate=["death"]` |
| deaths in the synthetic data | the simulated data of the transition-score pages contain no deaths. Only the [time-to-event](longitudinal-survival.ipynb) data do |
| unreturned surveys | under the composite strategy, the composite score of a patient who died counts as observed, so only living patients who do not respond are missing. Under the hypothetical strategy, a death also leaves the score missing |
| missing at random | under the composite strategy, a death is always observed, so response depends on the composite outcome. Missing at random given treatment and baseline covariates then fails when some patients die and some survivors do not respond. The analysis needs missing at random among survivors, given treatment, covariates, and survival. Under the hypothetical strategy, it needs missing at random for death and for non-response, given treatment and covariates |
| baseline adjustment | discharge risk, prior utilization, medication burden, age, and any site or calendar factors measured before assignment |
| interference control | reserved navigator capacity and access controls keep one patient's assignment from changing another patient's navigation or outcome |
| dependence | patients can share a navigator team. The [shared-teams page](cross-fitting.ipynb) keeps each team together when it fits models and computes the interval |

The program records contamination, protocol changes, and capacity breaches, because any of them can
break consistency or no interference.

The simulated data draw each baseline characteristic independently from a standard normal
distribution, so a negative value is below the population average. Real characteristics would be
correlated. Scores are in simulated units.
:::

## The retailer program

Several examples follow a subscription retailer. On the [weekly offer
policies](sequential-offers.ipynb) page, the retailer sends each customer one offer in each of three
weeks: nothing, a 15% discount code, or a free-shipping code. Its system draws each offer at random,
with logged probabilities that depend on the customer's engagement that week, so the chance of each
offer is known. Engagement responds to earlier offers and drives later ones, and customers who
withdraw tracking consent leave the data. The data have one row per customer and week.
`offers_protocol()` in `cleverly.datasets` returns the study description.

On the [monthly prices](sequential-pricing.ipynb) page, a pricing engine sets a renewal price for
each subscriber each month. The page asks what raising each price by 12% would do, wherever the
raised price stays inside the band of prices the engine allows; elsewhere the engine's price stands.
To correct for how the engine chose prices, the method needs to know how much more likely each
recorded price is under the 12% rise than under the engine's own choices. The engine logged its
inputs but not the chance of each price, so the page estimates that ratio from the data. No one
leaves the data before the outcome. `pricing_protocol()` returns its study description.

On the [weekly orders under repeated offers](weekly-outcomes.ipynb) page, the retailer decides each
week, for three weeks, whether to send a free-shipping code. It records each week whether the
customer ordered. Its scoring system draws who gets a code but does not log the chance, so the page
estimates it, and customers who withdraw tracking consent leave the data. `weekly_orders_protocol()`
returns its study description.

On the [renewal discounts](dose-response.ipynb) page, the renewal engine gives each subscriber a
discount of 0% to 40% off the renewal price, with bigger discounts for less active and newer
subscribers. The page asks what share would renew if every subscriber got the same discount, for
discounts from 5% to 30%. Its data are simulated in the notebook, and its study description is
built in the notebook.

All four retailer datasets are simulated. The [datasets reference](../api/datasets.md) lists the
generators of the first three and their true answers.

## Pages outside the two programs

Two simulated pages tell their own story. On [who gains most from a
coupon](conditional-effects.ipynb), a retailer's marketing team chose who got a coupon, and the page
asks how the coupon's effect on buying changes with the customer's age. On [a medicine with two
effects](separable-effects.ipynb), a heart-failure clinic chooses the dose of a water pill at
discharge, and the page separates the pill's effect on readmission from its effect on death. The
[plot results](plots.ipynb) page draws its figures from health-program data and from other simulated
data.

The [twins analysis](twins-causal-inference.ipynb) uses US birth certificates of twin pairs. It
estimates how much smoking in pregnancy lowered the birth weight of twins whose mother smoked, in
grams: the effect among the treated. Because the data are real, the page cannot compare its
estimate with a true answer. It shows instead why the analysis stops short of adjusting for
gestational age, and how strong a missing confounder would have to be to change the conclusion.

## The methods behind each example

Each technical reference page gives exactly what each method estimates, its conditions, and the
simulation studies that check it. The [plot results](plots.ipynb) page has no row: it draws the
figures of fits that the pages below explain.

| example | method | reference |
| --- | --- | --- |
| [one treatment, one outcome](point-treatment-tmle.ipynb) | TMLE, with each row predicted by models trained without it | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [smoking and twins' birth weight](twins-causal-inference.ipynb) | TMLE for the effect among the treated | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [a randomized pilot](randomized-pilot.ipynb) | TMLE with the recorded chances of treatment, also for the effects among treated and untreated patients (ATT, ATC), beside doubly robust TMLE (DR-TMLE) and adaptive-propensity weighting | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md), [DR-TMLE](../technical-reference/dr-tmle/index.md) and [adaptive-propensity IPTW](../technical-reference/adaptive-iptw.md) |
| [gain over current practice](population-impact.ipynb) | TMLE for the population gain over current practice (attributable risk and fraction), risk and odds ratios, and the controlled direct effect | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [which discharge practices matter](ranking-exposures.ipynb) | one cross-fitted TMLE fit for each candidate practice, with p-values adjusted for testing several at once (step-down max-T) | [variable importance](../technical-reference/validation-methods.md#variable-importance) |
| [who gains most from a coupon](conditional-effects.ipynb) | the EP learner of a conditional effect curve, with a rule learned on half the customers and valued on the other half; a TMLE projection on cosine waves with intervals and a band; a targeting curve and a calibration check of a priority score; a learned rule's gain | [conditional effects](../technical-reference/conditional-effects.md) |
| [does navigation help patients who live alone more?](subgroup-effects.ipynb) | TMLE of the effect in each group, with intervals, a band over the groups and a test that they gain the same; the same with a straight line in age; the effect with a later doctor's visit held fixed, an age trend inside each group, and risk ratios by group | [conditional effects](../technical-reference/conditional-effects.md) |
| [who should get the offer](treatment-rules.ipynb) | the value of a treatment rule learned inside each fold (CV-TMLE), and a random offer with known chances | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [three offer policies](interventions.ipynb) | TMLE for policies | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [more contacts and more service time](schedules-and-intensity.ipynb) | TMLE for an odds change across three treatment levels, and for a joint raise of two treatment amounts | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [a trend across three contact schedules](msm-projections.ipynb) | a trend line across treatment levels (MSM projection) | [MSM projections](../technical-reference/msm-projections.md) |
| [renewal discounts](dose-response.ipynb) | cross-fitted TMLE for a dose-response curve averaged over nearby doses, with one band over every reported dose | [smoothed dose-response curve](../technical-reference/dose-response.md) |
| [weekly offer policies](sequential-offers.ipynb) | TMLE for treatments over time, compared with week-by-week g-computation and a simple comparison | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) and [sequential g-computation and IPW](../technical-reference/sequential-baselines.md) |
| [weekly orders under repeated offers](weekly-outcomes.ipynb) | TMLE for a weekly outcome curve under repeated treatments, with the sequentially doubly robust estimator and an odds-doubling policy over weeks | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) |
| [two decisions after discharge](two-decision-rules.ipynb) | longitudinal TMLE for the value of a pair of treatment rules, one for each decision, learned inside each fold | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md#learned-regimens) |
| [three offers after discharge](model-based-over-time.ipynb) | the parametric g-formula, which simulates each plan forward, and a structural nested mean (blip) model of what each offer adds, checked against longitudinal TMLE | [parametric g-formula](../technical-reference/parametric-gformula.md), [structural dynamic effects](../technical-reference/structural-dynamic-effects.md) and [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) |
| [monthly prices](sequential-pricing.ipynb) | TMLE for shifts of a continuous treatment over time | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) |
| [readmission when death can come first](longitudinal-survival.ipynb) | TMLE for treatments over time, when death can come first | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) |
| [readmission by day 30, 60 and 90](discharge-to-readmission.ipynb) | TMLE for the risk by each follow-up day after one treatment, and cross-fitted TMLE over two decisions with intervals that treat each team as one unit | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) |
| [a medicine with two effects](separable-effects.ipynb) | TMLE for separable effects: the risk of each of two competing events when the part of a treatment that acts on one event and the part that acts on the other are set to different doses | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) |
| [missing survey answers](survey-nonresponse.ipynb) | TMLE with a model of who responds | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [an interview of a sample of patients](sampling-weights.ipynb) | TMLE with sampling weights, with effects for two risk groups and missing treatment records | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [a small pilot with outside records](trial-with-external-data.ipynb) | adaptive TMLE, which learns how the outside records differ from the pilot and removes that difference | [adaptive TMLE](../technical-reference/adaptive-tmle.md) |
| [choosing the prediction models](learner-choice.ipynb) | TMLE with out-of-fold predictions, with each model a cross-validated blend of candidate models (a Super Learner), the default blend, and screening of the treatment model's columns | [nuisance learners](../technical-reference/nuisance-learners.md) and [CV-TMLE](../technical-reference/cv-tmle.md) |
| [flexible models and shared teams](cross-fitting.ipynb) | the same method (CV-TMLE), showing why those out-of-fold predictions matter with flexible models, and keeping each team together when the models are trained and the interval computed | [CV-TMLE](../technical-reference/cv-tmle.md) |
| [a strong predictor of treatment](collaborative-tmle.ipynb) | collaborative TMLE, here predicting the treatment from the outcome model's predictions instead of from every characteristic | [collaborative TMLE](../technical-reference/collaborative-tmle.md) |

```{toctree}
:caption: Start here
:hidden:
:maxdepth: 1

One treatment, one outcome <point-treatment-tmle>
Smoking and twins' birth weight <twins-causal-inference>
```

```{toctree}
:caption: One treatment, outcome measured once
:hidden:
:maxdepth: 2

topics/average-effect
topics/effects-by-group
topics/targeting-rules
topics/amounts-and-policies
```

```{toctree}
:caption: Repeated treatments, or time until an event
:hidden:
:maxdepth: 2

topics/repeated-over-time
topics/time-to-event
```

```{toctree}
:caption: Data issues and model choices (any question)
:hidden:
:maxdepth: 2

topics/missing-and-combined-data
topics/models-and-checks
```
