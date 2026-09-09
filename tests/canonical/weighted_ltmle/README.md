# Ordinary weighted longitudinal TMLE evidence artifacts

This directory stores the registered comparison with R `ltmle` 1.3-0. The study draws exactly
2,000 selected rows per replication and passes fixed inverse-selection weights to both fits.

Run this smoke command before the declared regeneration:

```bash
python -m tests.canonical.weighted_ltmle.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/weighted-ltmle-smoke
```

Run `python -m tests.canonical.weighted_ltmle.regenerate` to build the declared artifacts.
