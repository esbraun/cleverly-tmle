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

## Known defect in the shared adapter, due at the next regeneration

`lmtp_crossfit_adapter.R` screens the supplied density ratios with
`abs(mean(supplied) - 1) > 0.5` on the *cumulative* product. That anchor is wrong in general. A
unit whose follow-up ends at the first node has no second-node arm, so its later columns are
structurally zero, and the cumulative ratio's expectation is the probability of reaching the
second node under the plan rather than one.

This study is unaffected. Its law draws few first-node events, so the cumulative mean stays inside
the band. The competing-risk rows are not: their law ends follow-up at the first node for most
units, and the same screen rejected a correct matrix at 0.3188 against an expectation of 0.3125.
`lmtp_competing_adapter.R` carries the corrected form, which screens the first column, whose
expectation is exactly one whatever the event process does downstream.

The shared adapter keeps the wrong anchor because its bytes are hashed into four manifests, and
correcting it invalidates this study's provenance. Take the corrected form across when this row is
next regenerated for a reason of its own.
