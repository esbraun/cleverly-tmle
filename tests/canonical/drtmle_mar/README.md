# Randomized missing-outcome DR-TMLE evidence artifacts

This directory contains the registered comparison with pinned R `drtmle` 1.1.2. The comparator
covers the both-correct joint-mechanism limit. The property study covers `cleverly`'s separate
five-reduction correction cycle.

Regenerate a disposable smoke study before the declared run:

```console
python -m tests.canonical.drtmle_mar.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/drtmle-mar-smoke
```

Run `python -m tests.canonical.drtmle_mar.regenerate` to rebuild the committed artifacts.
