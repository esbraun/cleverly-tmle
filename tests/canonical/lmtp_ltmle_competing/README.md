# R evidence for ordinary competing-risk longitudinal TMLE

This study compares ordinary `cleverly` LTMLE against pinned R `lmtp` 1.5.4. Both sides receive
the same panels and exact treatment and censoring mechanism. R fits each target cause with the
other cause declared as the competing event.

Run a disposable smoke study first:

```powershell
uv run --extra dev python -m tests.canonical.lmtp_ltmle_competing.regenerate `
  --replicates 8 --n 500 --primary-only --output $env:TEMP\lmtp-competing-probe
```

Regenerate the committed artifacts and document after the smoke study passes:

```powershell
uv run --extra dev python -m tests.canonical.lmtp_ltmle_competing.regenerate
uv run --extra dev python -m tests.studies.evidence.document --slug canonical-ltmle-competing
```

The pinned Docker image and both adapters are source inputs in the manifest. The exact-law and
Gateaux modules remain the scientific oracles.
