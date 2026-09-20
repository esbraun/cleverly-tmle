# Cross-fitted weighted longitudinal TMLE evidence artifacts

This directory stores the registered comparison with R `lmtp` 1.5.4. Both fits use the same
five-fold assignment. The R nuisance adapter consumes the fixed weight from an auxiliary column.

Run this smoke command before the declared regeneration:

```bash
docker build -t cleverly-lmtp-crossfit:1.5.4 tests/canonical/lmtp_crossfit
docker run --rm -v "$PWD/tests/canonical:/fixture:ro" cleverly-lmtp-crossfit:1.5.4 /fixture/lmtp_crossfit/smoke_weighted.R
python -m tests.canonical.weighted_lmtp_ltmle.regenerate --replicates 4 --n 2000 --skip-properties --allow-failures --output build/weighted-lmtp-ltmle-smoke
```

The probe uses the declared sample size. At `--n 200` the weighted quasibinomial regression of one
training complement stops on `quasibinomial IRLS did not converge`, because the selected rows of a
fifth of 200 panels separate. A smaller probe measures the solver rather than the study.

Run `python -m tests.canonical.weighted_lmtp_ltmle.regenerate` to build the declared artifacts.
