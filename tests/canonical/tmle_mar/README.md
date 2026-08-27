# Ordinary missing-outcome TMLE evidence artifacts

This directory contains the registered comparison with digest-pinned R `tmle` 2.1.1. The runner
uses exact nuisance predictions on the same realized MAR samples as `cleverly`.

Regenerate a disposable smoke study before the declared run:

```console
python -m tests.canonical.tmle_mar.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/tmle-mar-smoke
```

Run `python -m tests.canonical.tmle_mar.regenerate` to rebuild the committed artifacts.
