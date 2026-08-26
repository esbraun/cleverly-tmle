# Ordinary longitudinal MSM projection fixture

Run `python -m tests.canonical.ltmle_msm.regenerate --smoke` before a full regeneration.
The runner fits each plan with pinned R `ltmle` and then applies the declared projection.

The projection acts on the joint regimen influence curves. This step retains within-sample
correlation and avoids raw `ltmleMSM` coefficients, which target a quasibinomial projection.
