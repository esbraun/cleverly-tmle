# Point-treatment MSM projection fixture

Run `python -m tests.canonical.tmle3_msm.regenerate --smoke` before a full regeneration.
The runner fits pinned `tmle3::Param_MSM` with the declared fixed projection weights.

The R arm-indicator coefficients are transformed to the `cleverly` parameterization. The
transformation also applies to the joint influence curve, so the treatment-coefficient standard
error retains the covariance between both R arm indicators.
