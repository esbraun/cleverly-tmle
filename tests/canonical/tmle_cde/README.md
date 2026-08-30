# Controlled direct-effect TMLE evidence artifacts

This directory contains the registered comparison with digest-pinned R `tmle` 2.1.1. The runner
supplies exact nuisance predictions to both implementations on the same realized MAR samples.
Each primary replicate draws one observed sample. Both intervention levels use that sample and
produce separate Python and R rows.

Run a disposable primary smoke study before the declared run:

```console
python -m tests.canonical.tmle_cde.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/tmle-cde-smoke
```

Run the disposable property probe in the repository test suite. It checks both levels and every
single-mechanism control against the exact-law separation floor.

The declared run uses 3,200 paired primary replications, 2,400 replications at each rate-study
size, and 12,000 calibration replications. Run
`python -m tests.canonical.tmle_cde.regenerate --jobs 8 --r-jobs 8` to rebuild the eight committed
outputs, including the manifest. The manifest records the independent primary and property seeds.

The standard command also runs `probe_native_result2.R` on generated replication zero. The probe
rebuilds the committed `native-result2-defect.csv` artifact and deliberately selects the native
second result. The manifest hashes the probe and its output. The registered runner instead recodes
each requested level to result one because `tmle` 2.1.1 constructs the second result's observed
outcome offset from `Q`, not `Q.Z1`.
