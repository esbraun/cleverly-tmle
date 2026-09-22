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

## Supplied-density validation correction

The shared adapter formerly screened the cumulative product of supplied density ratios against an
expectation of one. Later columns can be structurally zero after an event, so that anchor was not
valid in general. The shared helper now applies the expectation-one screen to the first node at
five standard errors, while retaining the zero-pattern and correlation checks. The competing
adapter uses the same helper instead of a duplicate.

This changes only whether an input is refused before fitting; a successful fit receives the same
supplied ratio matrix. The provenance ledger therefore records the eight shared-adapter manifest
transitions as result-neutral. The ordinary, weighted, clustered and competing fixture smokes all
pass, including the competing mutations for an opposite arm and a dropped censoring factor.
