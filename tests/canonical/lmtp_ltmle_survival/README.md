# Cross-fitted survival-curve LTMLE evidence

This directory audits Cleverly's five-fold survival recursion against pinned R `lmtp` 1.5.4.
Both implementations use the fold assignment stored with each realized panel. R fits one prefix
for each reported horizon.

The end-of-study audit rejected `lmtp`'s cross-fitted influence-curve intervals before this row
was registered. The survival row therefore uses the same zero-row comparator fallback. Its R
runner remains a source and parameter audit, but the registered claims rest on independent truth,
Gateaux, mutation, and repeated-sampling property gates.

Run a disposable probe before the declared study.

```console
python -m tests.canonical.lmtp_ltmle_survival.regenerate --replicates 8 --n 500 --primary-only --output <temporary-directory> --cache <temporary-directory>
python -m tests.canonical.lmtp_crossfit.audit survival <temporary-directory>/samples.csv.gz <temporary-directory>/truth.csv <temporary-directory>/lmtp-rows.csv --jobs 2
```

The registered regeneration does not invoke the R runner. Run the complete study only after the
probe succeeds. The complete run writes the artifacts and refuses failed replications or failed
evidence gates.
