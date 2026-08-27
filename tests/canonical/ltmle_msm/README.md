# Canonical R evidence for the ordinary longitudinal MSM projection

This directory compares the ordinary identity-link longitudinal MSM projection in `cleverly`
with a fixed projection of four pinned R `ltmle` regimen fits. Both implementations receive
the same censored two-time-point samples, the same four plans, known mechanisms, the same
sequential regression families, and the same declared projection measure.

R fits each plan separately. The runner then applies the declared projection to the four
estimates and to their joint influence curves. Projecting the joint curves is the point:
every regimen fit uses the same sample, so projecting four marginal standard errors would
discard the covariance that creates.

Raw `ltmleMSM` coefficients are not compared. Its quasibinomial projection targets a different
parameter from this outcome-scale weighted least-squares one, so coefficient parity would
compare two estimands rather than two implementations.

The comparison is secondary evidence. The parameter and its efficient influence curve come
from the finite-support law in `tests/discrete_law_longitudinal.py`. The property study in
`tests/studies/longitudinal_msm_properties.py` independently checks double robustness, root-n
behavior, calibration, type-I error, power, targeting, and the declared projection measure.

Run a disposable smoke study from the repository root:

```powershell
uv run python -m tests.canonical.ltmle_msm.regenerate `
  --replicates 12 --n 400 --skip-properties --allow-failures `
  --output .tmp/longitudinal-msm-smoke
```

Run the declared study and refresh its documentation:

```powershell
uv run python -m tests.canonical.ltmle_msm.regenerate
uv run python -m tests.studies.evidence.document --slug longitudinal-msm
```

The replication count and sample size come from the study record and are recorded in
`manifest.json`. Do not publish a smoke run over them. The Dockerfile pins R 4.5.2 by digest
and checks the `ltmle` 1.3-0 tarball hash. The manifest records every result-determining
module and reference file.
