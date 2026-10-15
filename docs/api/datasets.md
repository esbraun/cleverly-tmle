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

Five names serve the subscription-retailer program of the offers tutorial.

| name | what it holds |
| --- | --- |
| `make_sequential_offers` | a long panel of three weekly offers, with the logged probabilities and the exact truth. `policy=` draws a deployment-check cohort |
| `offers_truth` | the exact week-4 share under each plan, and each contrast with no offers |
| `offers_protocol` | the `StudyProtocol` of the campaign |
| `win_back`, `win_back_then_ship` | the two dynamic rules, as functions of the week's engagement |
| `OFFER_PLANS`, `OFFER_LEVELS`, `OFFER_WEEKS` | the five plans by name, the three offer levels, and the number of weekly sends |

`make_sequential_offers` returns a long panel and takes the keyword `policy`. It is not in `GENERATORS`,
which holds generators of one wide frame with the columns `Y` and `A`.

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
   cleverly.datasets.make_sequential_offers
   cleverly.datasets.offers_truth
   cleverly.datasets.offers_protocol
   cleverly.datasets.win_back
   cleverly.datasets.win_back_then_ship
```
