# Synthetic datasets

Generators return data together with known population quantities and are intended for examples,
tests, and method studies.

Each generator accepts a sample size, random seed, and dataframe backend. Its return value contains
the generated frame and a mapping of known population quantities.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.datasets.available
   cleverly.datasets.make_linear_ate
   cleverly.datasets.make_nonlinear_ate
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
   cleverly.datasets.make_longitudinal_competing
   cleverly.datasets.make_longitudinal_weighted
```
