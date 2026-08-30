# Cross-fitted weighted longitudinal TMLE evidence artifacts

This directory stores the registered comparison with R `lmtp` 1.5.4. Both fits use the same
five-fold assignment. The R nuisance adapter consumes the fixed weight from an auxiliary column.

Run this smoke command before the declared regeneration:

```bash
python -m tests.canonical.weighted_lmtp_ltmle.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/weighted-lmtp-ltmle-smoke
```

Run `python -m tests.canonical.weighted_lmtp_ltmle.regenerate` to build the declared artifacts.
