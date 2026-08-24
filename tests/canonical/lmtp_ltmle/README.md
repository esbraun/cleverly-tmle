# Cross-fitted end-of-study LTMLE evidence

This directory audits Cleverly's five-fold end-of-study recursion against pinned R `lmtp`
1.5.4. Both implementations use the fold assignment stored with each realized panel.

The 1,600-replication audit rejected `lmtp` as the registered numeric comparator. Its point
estimates passed the truth-bias gates, but its influence-curve intervals did not. Coverage ranged
from 0.75 to 0.91, and every exact 99% coverage lower bound missed the declared 0.90 floor.
`lmtp-audit.csv` records the five truth and RMSE endpoints. The registered study therefore writes
the required zero-row equivalence artifact and relies on independent truth and property gates.

Run a disposable probe before the declared study.

```console
python -m tests.canonical.lmtp_ltmle.regenerate --replicates 8 --n 400 --primary-only --output <temporary-directory> --cache <temporary-directory>
python -m tests.canonical.lmtp_crossfit.audit end <temporary-directory>/samples.csv.gz <temporary-directory>/truth.csv <temporary-directory>/lmtp-rows.csv --jobs 2
```

The R runner remains a reproducible source audit. The registered regeneration does not invoke it.
Run the complete registered study only after the probe succeeds. The complete run writes the
artifacts and refuses failed replications or failed evidence gates.
