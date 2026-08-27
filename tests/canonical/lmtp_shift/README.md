# Modified-treatment-policy evidence artifacts

This directory contains the registered comparison with pinned R `lmtp` 1.5.4. The runner fits the
natural course, an uncapped shift, and an actively capped shift on each realized Python sample. A
shared point-treatment adapter supplies R with the analytic conditional density ratios. `cleverly`
uses its pooled-hazard representation of the same known conditional-normal law.

Regenerate a disposable smoke study before the declared run:

```console
python -m tests.canonical.lmtp_shift.regenerate --replicates 4 --n 200 --skip-properties --output build/shift-smoke --allow-failures
```

Run `python -m tests.canonical.lmtp_shift.regenerate` to rebuild the committed artifacts.
