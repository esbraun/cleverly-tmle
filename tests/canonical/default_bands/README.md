# Default simultaneous bands across shipped fit shapes

This registered study measures the joint coverage of the default multiplier band for every
shipped fit shape that publishes one and that no other registered study measures. Each shape
refits a registered source study's subject fit with `simultaneous=True` at that study's primary
law and size, on samples this study draws. The primary scenario is the shipped `TMLE()` default
fit on the ordinary point-treatment binary law.

The `shift_grid` shape builds a 320-bin density design, about 1.9 GB per fit, so 16 workers run
out of memory on a 32 GB machine. The published run used 8.

No canonical implementation is compared. `equivalence.csv` is empty and schema-valid.

Regenerate from the repository root:

```powershell
.venv/Scripts/python.exe -m tests.canonical.default_bands.regenerate --jobs 8
```

The reader-facing results are in
[`default-simultaneous-bands.md`](../../../docs/technical-reference/method-evidence/default-simultaneous-bands.md).
