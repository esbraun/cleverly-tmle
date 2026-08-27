# Incremental-intervention evidence artifacts

This directory contains the point-estimate comparison with R `imtp` at commit `d4b5204`.
`imtp` estimates the same incremental odds curve, but its reported influence curve omits the
treatment-mechanism derivative. The comparison is therefore a point-curve witness only; the
registered Python property study independently validates inference.

Regenerate a disposable smoke study before the declared run:

```console
python -m tests.canonical.imtp_incremental.regenerate --replicates 4 --n 200 --skip-properties --output build/incremental-smoke --allow-failures
```

Run `python -m tests.canonical.imtp_incremental.regenerate` to rebuild the committed artifacts.
