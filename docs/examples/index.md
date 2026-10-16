# Examples

These tutorials follow two programs. A regional health plan offers **care-transition navigation**
to adults who leave participating hospitals for home. Each of its tutorials answers one evaluation
question with one method, and the table below lists them. A subscription retailer sends weekly
offers. [The retailer program](#the-retailer-program) has its own section.

## The program

Start with [Point-treatment TMLE](point-treatment-tmle.ipynb).

| tutorial | the evaluation question | method entry | the failure mode it demonstrates |
| --- | --- | --- | --- |
| [Point-treatment TMLE](point-treatment-tmle.ipynb) | assignment to the standard navigation offer | [entry](../technical-reference/point-treatment-tmle.md) | one consistent nuisance gives a consistent point estimate. Two inconsistent nuisances have no general guarantee |
| [CV-TMLE](cross-fitting.ipynb) | the same offer, with flexible learners and patients nested in navigator teams | [entry](../technical-reference/cv-tmle.md) | in-sample nuisance evaluation and ignored teams can understate uncertainty |
| [Collaborative TMLE](collaborative-tmle.ipynb) | the offer's effect on a successful transition home, when one strong predictor of the offer has an uncertain role | [entry](../technical-reference/collaborative-tmle.md) | the predictor makes the fitted propensities extreme. The ordinary TMLE interval under-covers at this law in a review probe. OAT C-TMLE targets the same ATE, and a registered cell measures its coverage |
| [Choosing a navigation policy](interventions.ipynb) | target by risk, raise assigned navigation intensity, or change assignment odds | [entry](../technical-reference/point-treatment-tmle.md#variations) | the capacity cap is part of the question: two raises of the same size are two estimands |
| [Survey non-response](survey-nonresponse.ipynb) | many patients do not return the 30-day transition survey | [entry](../technical-reference/point-treatment-tmle.md#missing-outcomes-and-controlled-direct-effects) | a complete-case fit estimates the effect standardized to the respondents' covariate distribution |
| [Longitudinal TMLE](longitudinal-tmle.ipynb) | navigation at discharge and day seven | [entry](../technical-reference/longitudinal-tmle.md) | one regression cannot adjust for a time-varying confounder and preserve the causal path |
| [Time-to-event outcomes](longitudinal-survival.ipynb) | readmission, with death as a competing event and disenrollment as censoring | [entry](../technical-reference/longitudinal-tmle.md#survival-and-competing-risks) | coding death as censoring targets the readmission risk in a world where death is prevented. That is not the total effect the plan asks about, which leaves death alone |
| [MSM projections](msm-projections.ipynb) | three navigation cadences summarized as one trend | [entry](../technical-reference/msm-projections.md) | a linear working model that misses the arm means still defines the parameter it reports |

The health-program tutorials use observational assignment. A randomized offer would identify the
assignment effect with fewer assumptions. Use one when the program can still control assignment.

The synthetic laws make the causal assumptions true by construction. A real analysis must defend
them from its protocol and operational records. The relevant assumptions are consistency,
exchangeability, positivity, and no interference. The
[Oxford causal assumptions chapter](https://www.stats.ox.ac.uk/~evans/APTS/causassmp.html) defines
each assumption and explains why consistency requires a well-defined intervention.

## The shared study design

| design element | program definition |
| --- | --- |
| target population | adults discharged home from a participating hospital during the declared enrollment period |
| time zero | after eligibility and baseline measurement, but before navigation assignment |
| treatment strategy | assignment to a standardized transition-navigation offer, not completed participation |
| variations | three cadences, an assigned navigation intensity, or two decisions, each on its own tutorial |
| standard offer | a bedside transition plan and two scheduled navigator contacts within 30 days |
| comparison | usual discharge support without access to the navigation protocol |
| primary outcome | a 30-day patient-reported transition score. The protocol scores death before day 30 as the worst transition score (composite strategy) |
| death strategy on each page | the composite strategy needs an outcome with a worst value, such as a bounded instrument or a top-box indicator. The [MSM projections](msm-projections.ipynb) page uses a synthetic score with no worst value. It uses the hypothetical strategy: the score had the patient not died |
| death in the data | under the composite strategy, a death before day 30 enters the data as an observed outcome: the worst score, or not top box. Under the hypothetical strategy, a real analysis treats death as censoring. It then needs exchangeability and positivity for death given the recorded history |
| deaths in the synthetic laws | the laws of the transition-experience pages draw no deaths. Only the [time-to-event](longitudinal-survival.ipynb) law draws deaths |
| unreturned surveys | under the composite strategy, the composite score counts as observed. Only living patients who do not respond are missing. Under the hypothetical strategy, a death also leaves the score missing |
| missing at random | under the composite strategy, a death is always observed, so response depends on the composite outcome. Missing at random given treatment and baseline covariates fails when some patients die and some survivors do not respond. The analysis needs missing at random among survivors, given treatment, covariates, and survival. Under the hypothetical strategy, the analysis needs missing at random for death and for non-response, given treatment and covariates |
| baseline adjustment | discharge risk, prior utilization, medication burden, age, and applicable site or calendar factors measured before assignment |
| interference control | reserved navigator capacity and access controls prevent one assignment from changing another patient's protocol |
| dependence | patients can share a navigator team. The [CV-TMLE tutorial](cross-fitting.ipynb) keeps each team intact in fitting and inference |

`navigation_protocol()` in `cleverly.datasets` returns this design as the `StudyProtocol` of the
standard offer. Each point-treatment tutorial starts from it and uses `dataclasses.replace` to change
only the fields its question changes. The two longitudinal tutorials start from
`longitudinal_navigation_protocol()` in the same way. Their time zero is the discharge, and their
eligibility requires a live discharge.

Only baseline information enters a point-treatment adjustment set. In particular, actual length of
stay and completed contacts occur after assignment and cannot serve as baseline confounders.

The synthetic laws draw each baseline covariate independently from a standard normal distribution.
The laws do not standardize a sample. A negative value is below the population average. Real
covariates would be correlated. Outcome scores are in synthetic units. The longitudinal tutorial uses a binary top-box outcome, and its protocol scores
death before day 30 as not top box.

The intervention is the **offer**. This choice fixes treatment versions despite different patient
uptake. The program records contamination, protocol changes, and capacity breaches because any of
them can weaken consistency or no interference.

## Two outcome families

| outcome | what it is | which pages use it |
| --- | --- | --- |
| transition experience | a 30-day patient-reported score, or its top-box indicator | every tutorial except time-to-event outcomes |
| time-to-event outcomes | readmission or death | [time-to-event outcomes](longitudinal-survival.ipynb) |

The day-seven engagement score in [longitudinal TMLE](longitudinal-tmle.ipynb) is a time-varying
confounder, not an outcome.

## The retailer program

A subscription retailer sends one offer to each customer in each of three weeks: no code, a 15%
discount code, or a free-shipping code. The CRM draws each offer at random, with logged
probabilities that depend on the customer's engagement that week. Engagement responds to earlier
offers and drives later ones. Customers who withdraw tracking consent are lost to follow-up.

This program differs from the health program in four ways.

| element | the retailer program |
| --- | --- |
| assignment | randomized with logged probabilities, so the offer mechanism is known |
| treatment | a three-level offer at each of three weekly nodes |
| data layout | a long panel with one row per customer and week |
| data | synthetic only. No applied data access exists, so the pilot is a synthetic deployment check |

`offers_protocol()` in `cleverly.datasets` returns the `StudyProtocol` of the program. The
[datasets reference](../api/datasets.md) lists the generator and the truth.

| tutorial | the evaluation question | method entry | the failure mode it demonstrates |
| --- | --- | --- | --- |
| [Sequential offers](sequential-offers.ipynb) | which of five weekly offer policies gives the most full-price orders in week 4, and which policy beats which | [entry](../technical-reference/longitudinal-tmle.md#known-node-mechanisms) and [sequential baselines](../technical-reference/sequential-baselines.md) | an unpaired comparison of two policies, which ignores the shared sample, overstates the standard error of their difference |

## Real data

The [TWINS analysis](twins-causal-inference.ipynb) sits outside the program and is the only
real-data example. It estimates the effect of maternal smoking on first-year mortality among twins
of smoking mothers (ATT). It also shows why the adjustment set stops before birth weight.

```{toctree}
:maxdepth: 1

point-treatment-tmle
cross-fitting
collaborative-tmle
interventions
survey-nonresponse
longitudinal-tmle
longitudinal-survival
msm-projections
sequential-offers
twins-causal-inference
```
