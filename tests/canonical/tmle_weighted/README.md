# Weighted point-treatment TMLE evidence artifacts

This directory contains the registered comparison with digest-pinned R `tmle` 2.1.1. The runner
passes exact nuisance predictions and inverse selection weights for each realized sample.

The comparison reuses `tests/canonical/tmle_mar/Dockerfile`. It also mounts the shared study
harness and point-treatment row adapter. The weighted study does not copy package pins.

Regenerate a disposable smoke study before the declared run:

```console
python -m tests.canonical.tmle_weighted.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/tmle-weighted-smoke
```

Run `python -m tests.canonical.tmle_weighted.regenerate` to rebuild the committed artifacts.
