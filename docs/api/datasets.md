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

`make_population_impact` draws the program data of the
[population-impact example](../examples/population-impact.ipynb): a binary top-box outcome with
unreturned surveys and a 7-day primary-care visit after the offer. `population_impact_truth` holds
its exact truths, including the population attributable risk and the controlled direct effect at
each visit level. `make_population_impact` is not in `GENERATORS`.

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
   cleverly.datasets.make_population_impact
   cleverly.datasets.population_impact_truth
   cleverly.datasets.navigation_data
   cleverly.datasets.navigation_protocol
   cleverly.datasets.longitudinal_navigation_protocol
   cleverly.datasets.make_sequential_offers
   cleverly.datasets.offers_truth
   cleverly.datasets.offers_protocol
   cleverly.datasets.win_back
   cleverly.datasets.win_back_then_ship
   cleverly.datasets.make_sequential_pricing
   cleverly.datasets.pricing_truth
   cleverly.datasets.pricing_protocol
   cleverly.datasets.price_rise
   cleverly.datasets.make_team_readmission
   cleverly.datasets.team_readmission_truth
   cleverly.datasets.team_readmission_protocol
```
