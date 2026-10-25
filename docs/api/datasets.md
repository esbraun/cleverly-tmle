# Synthetic datasets

Generators return data together with known population quantities and are intended for examples,
tests, and method studies.

Each generator accepts a sample size, random seed, and dataframe backend. Its return value contains
the generated frame and a mapping of known population quantities.

Three functions serve the navigation program that the [examples](../examples/index.md) share.

| function | what it returns |
| --- | --- |
| `navigation_data` | the `make_nonlinear_bounded` draw, with the columns under the program's names |
| `navigation_protocol` | the `StudyProtocol` of the one-decision program |
| `longitudinal_navigation_protocol` | the `StudyProtocol` of the two-decision program |

`make_randomized_pilot` draws the randomized pilot of the
[randomized-pilot example](../examples/randomized-pilot.ipynb): a binary success outcome, an offer
drawn with a recorded chance by discharge-risk group (`PILOT_OFFER_CHANCES`), and the exact truth
of `randomized_pilot_truth`.

Two generators serve the [schedules and service doses example](../examples/schedules-and-intensity.ipynb).
`make_contact_schedules` draws a 0-to-1 transition score and a three-level contact schedule
(`SCHEDULE_LEVELS`), and `contact_schedules_truth` gives the exact mean under each declared odds
change. `make_navigation_doses` draws a binary success outcome and two doses, navigator hours and
pharmacist minutes, and `navigation_doses_truth` gives the exact success share under each capped
raise in `DOSE_PLANS`.

Eight names serve the subscription-retailer program of the [offers tutorial](../examples/sequential-offers.ipynb).

| name | what it holds |
| --- | --- |
| `make_sequential_offers` | a long panel of three weekly offers, with the logged probabilities and the exact truth. `policy=` draws a deployment-check cohort |
| `offers_truth` | the exact week-4 share under each plan, and each contrast with no offers |
| `offers_protocol` | the `StudyProtocol` of the campaign |
| `win_back`, `win_back_then_ship` | the two dynamic rules, as functions of the week's engagement |
| `OFFER_PLANS`, `OFFER_LEVELS`, `OFFER_WEEKS` | the five plans by name, the three offer levels, and the number of weekly sends |

`make_sequential_offers` returns a long panel and takes the keyword `policy`. It is not in `GENERATORS`,
which holds generators of one wide frame with the columns `Y` and `A`.

Seven names serve the pricing tutorial of the same program.

| name | what it holds |
| --- | --- |
| `make_sequential_pricing` | a wide frame of three monthly renewal prices with engagement, the renewal, and the revenue, with the exact truth |
| `pricing_truth` | the exact mean renewal and revenue under each plan, each effect against the engine's own prices, and the decomposition of the effect by path |
| `pricing_protocol` | the `StudyProtocol` of the pricing question |
| `price_rise` | a plan that raises each price by a factor and keeps the engine's price where the raised price would leave the band |
| `PRICE_PLANS`, `PRICE_BANDS`, `PRICE_NODES` | the three plans by name, the price band of each month, and the number of monthly prices |

`make_sequential_pricing` is not in `GENERATORS` either. Its columns are `usage`, `engaged_1`,
`price_1`, `engaged_2`, `price_2`, `engaged_3`, `price_3`, `renewed`, and `revenue`.

Seven names serve the [weekly-outcomes tutorial](../examples/weekly-outcomes.ipynb) of the same
program, where the order of every week is the outcome.

| name | what it holds |
| --- | --- |
| `make_weekly_orders` | a wide frame of three weekly free-shipping offers with an order recorded each week, with the exact truths |
| `weekly_orders_truth` | the exact weekly order share under each plan and odds tilt, and each contrast |
| `weekly_orders_protocol` | the `StudyProtocol` of the weekly-orders question |
| `offer_after_no_order` | the win-back rule, as a function of last week's order |
| `ORDER_PLANS`, `ORDER_TILTS`, `ORDER_WEEKS` | the three plans by name, the two odds tilts, and the number of weekly sends |

`make_weekly_orders` is not in `GENERATORS` either. Its columns are `tenure`, `order_0`, and
`engaged_t`, `offer_t`, `tracked_t`, and `order_t` for each week `t` from 1 to 3.

Three functions serve the team readmission part of the
[discharge-to-readmission tutorial](../examples/discharge-to-readmission.ipynb).

| function | what it returns |
| --- | --- |
| `make_team_readmission` | a wide frame of patients in navigator teams: a discharge offer, a day-30 call, dropout, and readmission by day 90, with the exact truth |
| `team_readmission_truth` | the exact share readmitted by day 90 under one plan of offer and call |
| `team_readmission_protocol` | the `StudyProtocol` of the two-decision readmission question |

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.datasets.available
   cleverly.datasets.make_linear_ate
   cleverly.datasets.make_nonlinear_ate
   cleverly.datasets.make_nonlinear_bounded
   cleverly.datasets.make_binary_outcome
   cleverly.datasets.make_multi_arm
   cleverly.datasets.make_heterogeneous
   cleverly.datasets.make_instrument
   cleverly.datasets.make_shift_dose
   cleverly.datasets.make_missing_outcome
   cleverly.datasets.make_missing_outcome_binary
   cleverly.datasets.make_missing_outcome_bounded
   cleverly.datasets.make_cde
   cleverly.datasets.make_clustered
   cleverly.datasets.make_biased_sample
   cleverly.datasets.make_weak_overlap
   cleverly.datasets.make_longitudinal
   cleverly.datasets.make_longitudinal_survival
   cleverly.datasets.make_longitudinal_repeated
   cleverly.datasets.make_longitudinal_competing
   cleverly.datasets.make_longitudinal_weighted
   cleverly.datasets.make_point_survival
   cleverly.datasets.point_survival_truth
   cleverly.datasets.navigation_data
   cleverly.datasets.navigation_protocol
   cleverly.datasets.longitudinal_navigation_protocol
   cleverly.datasets.make_randomized_pilot
   cleverly.datasets.randomized_pilot_truth
   cleverly.datasets.make_contact_schedules
   cleverly.datasets.contact_schedules_truth
   cleverly.datasets.make_navigation_doses
   cleverly.datasets.navigation_doses_truth
   cleverly.datasets.make_sequential_offers
   cleverly.datasets.offers_truth
   cleverly.datasets.offers_protocol
   cleverly.datasets.win_back
   cleverly.datasets.win_back_then_ship
   cleverly.datasets.make_sequential_pricing
   cleverly.datasets.pricing_truth
   cleverly.datasets.pricing_protocol
   cleverly.datasets.price_rise
   cleverly.datasets.make_weekly_orders
   cleverly.datasets.weekly_orders_truth
   cleverly.datasets.weekly_orders_protocol
   cleverly.datasets.offer_after_no_order
   cleverly.datasets.make_team_readmission
   cleverly.datasets.team_readmission_truth
   cleverly.datasets.team_readmission_protocol
```

Five names serve the [model-based-over-time tutorial](../examples/model-based-over-time.ipynb),
where navigation is offered at discharge, on day 10, and on day 20, and a recovery score is read on
day 30.

| name | what it holds |
| --- | --- |
| `make_recovery_score` | a wide frame of three navigation offers with a symptom flag before each later offer and the day-30 score, with the exact plan means and the effect of each offer |
| `recovery_score_truth` | the exact mean day-30 score under one plan of three offers |
| `recovery_score_protocol` | the `StudyProtocol` of the three-decision recovery question |
| `RECOVERY_PLANS`, `RECOVERY_BOUNDS` | the plans `always` and `never`, and the known range of the score, 0 to 100 |

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.datasets.make_recovery_score
   cleverly.datasets.recovery_score_truth
   cleverly.datasets.recovery_score_protocol
```
