# Default simultaneous bands across shipped fit shapes

This registered study measures the joint coverage of the default multiplier band for every
shipped fit shape that publishes one and that no other registered study measures. Each shape
refits a registered source study's subject fit with `simultaneous=True` at that study's primary
law and size, on samples this study draws. The primary scenario is the shipped `TMLE()` default
fit on the ordinary point-treatment binary law.

No canonical implementation is compared. `equivalence.csv` is empty and schema-valid.

Regenerate from the repository root:

```powershell
.venv/Scripts/python.exe -m tests.canonical.default_bands.regenerate --jobs 16
```

The reader-facing results are in
[`default-simultaneous-bands.md`](../../../docs/technical-reference/method-evidence/default-simultaneous-bands.md).
