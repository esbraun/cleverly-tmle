# Deterministic-regime evidence artifacts

This directory contains the registered comparison with pinned R `lmtp` 1.5.4. The runner fits a
static plan and a nonconstant deterministic rule on each realized Python sample.

Regenerate a disposable smoke study before the declared run:

```console
python -m tests.canonical.lmtp_regimes.regenerate --replicates 4 --n 200 --skip-properties --output build/regime-smoke
```

Run `python -m tests.canonical.lmtp_regimes.regenerate` to rebuild the committed artifacts.
