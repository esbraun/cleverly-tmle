# Examples

## Start here

Each example is a complete analysis that you can run and adapt. It loads a dataset, estimates what
a treatment, offer, or policy did, tests whether the result holds up, and says what you can tell a
manager. Sixteen examples use simulated data, so each page can compare its estimate with the true
answer. The seventeenth uses real birth records. If you are new to causal inference, read
[one treatment, one outcome](point-treatment-tmle.ipynb) first. It explains each step that the
other pages reuse.

## Find the example for your problem

Most examples use targeted maximum likelihood estimation (TMLE). It fits two models: an outcome
model, which predicts the outcome from the treatment and the patient's characteristics, and a
treatment model, which predicts who got the treatment. A correction step then uses the treatment
model to remove the bias that the outcome model leaves, giving more weight to patients whose own
treatment was unlikely.

| your situation | example | the mistake it shows |
| --- | --- | --- |
| a treatment given once and an outcome measured later | [one treatment, one outcome](point-treatment-tmle.ipynb) | a simple comparison of treated and untreated patients mixes the treatment's effect with who received it. The page also shows what happens to the estimate when one or both models are wrong |
| flexible models such as gradient boosting, or patients who share a care team | [flexible models and shared teams](cross-fitting.ipynb) | a model that predicts the same rows it was trained on looks more accurate than it is, and an analysis that treats teammates as independent ignores that their results move together. Both make the interval too narrow |
| one recorded variable strongly predicts who gets the treatment | [a strong predictor of treatment](collaborative-tmle.ipynb) | some patients become nearly certain or nearly unable to get the treatment, so the few exceptions get very large weights. In 2,000 simulated repeats, the 95% interval of standard TMLE contained the truth only 93% of the time, so it was slightly too narrow |
| many outcomes are missing, for example unanswered surveys | [missing survey answers](survey-nonresponse.ipynb) | analyzing only those who answered estimates the effect for people like the respondents, who differ from the whole group |
| the treatment was assigned at random with a recorded chance, as in a pilot | [a randomized pilot](randomized-pilot.ipynb) | because the chance of the offer differed by risk group, a simple comparison of offered and other patients is confounded even though the offer was random (here -0.006 against a true 0.142). Replacing the recorded chances with a fitted model of who got the treatment gives up the one model known to be right; when that fitted model and the outcome model are both wrong, the estimate is biased |
| the gain from offering everyone over current practice, or whether the offer still helps when a later step, such as a doctor's visit within 7 days, is guaranteed for everyone | [gain over current practice](population-impact.ipynb) | the usual average effect compares offering everyone with offering no one, so it also counts the gain current practice has already collected and overstates what offering everyone would add |
| policies to compare, such as offering a program only to high-risk patients, raising each patient's chance of an offer, or raising how much of the program each patient gets | [three offer policies](interventions.ipynb) | each policy has its own starting point: offering only to high-risk patients is compared with offering no one, and the other two with current practice, so the answers do not rank the policies or compare one with another directly. A staffing limit on how much each patient can get is part of the question: the same raise with and without the limit gives two different answers |
| a treatment that helps some patients and harms others, so you must decide who gets it, or share a limited number of places at random | [who should get the offer](treatment-rules.ipynb) | a rule scored on the same patients it was learned from looks better than it will do on new patients |
| a team chooses among several contact schedules, or sets two service times, and you want the effect of nudging those choices or raising both amounts | [more contacts and more service time](schedules-and-intensity.ipynb) | when two services help each other, the effects of raising each one alone do not add up to the effect of raising both |
| several treatment levels, summarized by one trend | [a trend across three contact schedules](msm-projections.ipynb) | a straight line through the levels still reports a slope when it does not fit them, so the slope alone can mislead |
| time until an event, when another event such as death can come first and some people leave follow-up | [readmission when death can come first](longitudinal-survival.ipynb) | treating death like dropout estimates readmission in a world where nobody dies, which the program cannot bring about |
| one treatment at discharge and an event such as readmission by day 30, 60 and 90, when some patients leave follow-up, or patients who share a team over two decisions | [readmission by day 30, 60 and 90](discharge-to-readmission.ipynb) | treating patients of one team as independent ignores that their risks move together, so the interval is too narrow |
| a treatment repeated weekly, where earlier offers change later behavior | [weekly offer policies](sequential-offers.ipynb) | treating two policies as if they had been tried on separate customers, when both are estimated from the same customers, overstates the uncertainty of their difference |
| an outcome recorded every week, such as whether a customer ordered, while the treatment repeats | [weekly orders under repeated offers](weekly-outcomes.ipynb) | fitting each week without the earlier orders leaves out what drove later offers, and adding the weekly variances as if the weeks were independent makes the interval for the total slightly too narrow |
| repeated treatments where you can model how the whole process unfolds | [three offers after discharge](model-based-over-time.ipynb) | a model of the process that is too simple gives a confident wrong answer; comparing two methods that rest on different models shows the problem |
| a price or dose that varies continuously and is set again each month | [monthly prices](sequential-pricing.ipynb) | one regression that holds later behavior and later prices fixed misses part of the effect of a price rise |
| a full analysis of real records, where no true answer is known | [smoking and twins' birth weight](twins-causal-inference.ipynb) | adjusting for a variable that the treatment itself changes, here gestational age, removes part of the effect |

## The health program

Thirteen examples follow one imaginary health program. A regional health plan offers
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
| death in the data | under the composite strategy, a death before day 30 enters the data as an observed outcome: the worst score, or not top box. Under the hypothetical strategy, a real analysis treats death as censoring (the outcome goes unseen), and then needs exchangeability and positivity for death given the recorded history |
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

Three examples follow a subscription retailer. On the [weekly offer
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

All three retailer datasets are simulated. The [datasets reference](../api/datasets.md) lists each
generator and its true answers.

## A real-data example

The [twins analysis](twins-causal-inference.ipynb) uses US birth certificates of twin pairs. It
estimates how much smoking in pregnancy lowered the birth weight of twins whose mother smoked, in
grams: the effect among the treated. Because the data are real, the page cannot compare its
estimate with a true answer. It shows instead why the analysis stops short of adjusting for
gestational age, and how strong a missing confounder would have to be to change the conclusion.

## The methods behind each example

Each technical reference page gives exactly what each method estimates, its conditions, and the
simulation studies that check it.

| example | method | reference |
| --- | --- | --- |
| [one treatment, one outcome](point-treatment-tmle.ipynb) | TMLE, with each row predicted by models trained without it | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [flexible models and shared teams](cross-fitting.ipynb) | the same method (CV-TMLE), showing why those out-of-fold predictions matter with flexible models, and keeping each team together when the models are trained and the interval computed | [CV-TMLE](../technical-reference/cv-tmle.md) |
| [a strong predictor of treatment](collaborative-tmle.ipynb) | collaborative TMLE, here predicting the treatment from the outcome model's predictions instead of from every characteristic | [collaborative TMLE](../technical-reference/collaborative-tmle.md) |
| [missing survey answers](survey-nonresponse.ipynb) | TMLE with a model of who responds | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [a randomized pilot](randomized-pilot.ipynb) | TMLE with the recorded chances of treatment, beside doubly robust TMLE (DR-TMLE) and adaptive-propensity weighting | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md), [DR-TMLE](../technical-reference/dr-tmle/index.md) and [adaptive-propensity IPTW](../technical-reference/adaptive-iptw.md) |
| [gain over current practice](population-impact.ipynb) | TMLE for the population gain over current practice (attributable risk and fraction), risk and odds ratios, and the controlled direct effect | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [three offer policies](interventions.ipynb) | TMLE for policies | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [who should get the offer](treatment-rules.ipynb) | the value of a treatment rule learned inside each fold (CV-TMLE), and a random offer with known chances | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [more contacts and more service time](schedules-and-intensity.ipynb) | TMLE for an odds change across three treatment levels, and for a joint raise of two treatment amounts | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |
| [a trend across three contact schedules](msm-projections.ipynb) | a trend line across treatment levels (MSM projection) | [MSM projections](../technical-reference/msm-projections.md) |
| [readmission when death can come first](longitudinal-survival.ipynb) | TMLE for treatments over time, when death can come first | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) |
| [readmission by day 30, 60 and 90](discharge-to-readmission.ipynb) | TMLE for the risk by each follow-up day after one treatment, and cross-fitted TMLE over two decisions with intervals that treat each team as one unit | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) |
| [weekly offer policies](sequential-offers.ipynb) | TMLE for treatments over time, compared with week-by-week g-computation and a simple comparison | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) and [sequential g-computation and IPW](../technical-reference/sequential-baselines.md) |
| [weekly orders under repeated offers](weekly-outcomes.ipynb) | TMLE for a weekly outcome curve under repeated treatments, with the sequentially doubly robust estimator and an odds-doubling policy over weeks | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) |
| [three offers after discharge](model-based-over-time.ipynb) | the parametric g-formula, which simulates each plan forward, and a structural nested mean (blip) model of what each offer adds | [parametric g-formula](../technical-reference/parametric-gformula.md) and [structural dynamic effects](../technical-reference/structural-dynamic-effects.md) |
| [monthly prices](sequential-pricing.ipynb) | TMLE for shifts of a continuous treatment over time | [longitudinal TMLE](../technical-reference/longitudinal-tmle.md) |
| [smoking and twins' birth weight](twins-causal-inference.ipynb) | TMLE for the effect among the treated | [point-treatment TMLE](../technical-reference/point-treatment-tmle.md) |

```{toctree}
:caption: First example
:maxdepth: 1

point-treatment-tmle
```

```{toctree}
:caption: One treatment, other designs
:maxdepth: 1

cross-fitting
collaborative-tmle
survey-nonresponse
randomized-pilot
population-impact
```

```{toctree}
:caption: Policies and treatment levels
:maxdepth: 1

interventions
treatment-rules
schedules-and-intensity
msm-projections
```

```{toctree}
:caption: Treatments and events over time
:maxdepth: 1

longitudinal-survival
discharge-to-readmission
sequential-offers
weekly-outcomes
model-based-over-time
sequential-pricing
```

```{toctree}
:caption: Real data
:maxdepth: 1

twins-causal-inference
```
