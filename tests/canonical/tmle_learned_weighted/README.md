# Learned weighted point-treatment TMLE evidence artifacts

This registered study compares Cleverly with digest-pinned R `tmle` 2.1.1. Both implementations
fit weighted main-effects regressions for the outcome and treatment mechanism.

The runner reuses `tests/canonical/tmle_mar/Dockerfile`. It also mounts the shared study harness
and continuous point-treatment row adapter.

Run a disposable smoke study before the declared run:

```console
python -m tests.canonical.tmle_learned_weighted.regenerate --replicates 4 --n 200 --skip-properties --allow-failures --output build/tmle-learned-weighted-smoke
```

Run `python -m tests.canonical.tmle_learned_weighted.regenerate` to build the declared artifacts.
