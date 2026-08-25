# Cross-fitted survival-curve LTMLE evidence

This directory holds the registered paired study for Cleverly's five-fold survival recursion
against pinned R `lmtp` 1.5.4. Both implementations receive the identical panels, the identical
stored fold assignment, and the identical treatment and censoring mechanism. R fits one prefix for
each reported horizon.

R `ltmle` has no cross-fitting and cannot witness this construction. `lmtp` has no `gform`
argument, so the adapter substitutes exact per-node density ratios and checks them against
`lmtp`'s own estimate on every run. See the
[end-of-study README](../lmtp_ltmle/README.md), which carries the argument for why the mechanism
is supplied rather than estimated on each side.

The first horizon uses `lmtp`'s one-node binary mean, because `lmtp` requires two event nodes for
its survival path. That is the same first-horizon cumulative-risk parameter. At horizon two the
runner converts event-free survival to cumulative risk and reverses the influence-curve sign.

Run a disposable probe before the declared study.

```console
python -m tests.canonical.lmtp_ltmle_survival.regenerate --replicates 8 --n 500 --primary-only --output <temporary-directory> --cache <temporary-directory>
```

Then the complete registered study.

```console
python -m tests.canonical.lmtp_ltmle_survival.regenerate
python -m tests.studies.evidence.document --slug canonical-ltmle-survival-crossfit
```
