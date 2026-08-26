# R evidence for cross-fitted competing-risk longitudinal TMLE

This study compares five-fold `cleverly` LTMLE against pinned R `lmtp` 1.5.4. Both sides receive
the exact realized fold assignment and exact mechanism. R fits each cause separately and keeps the
other cause in its risk-set recursion.

Run a disposable smoke study first:

```powershell
uv run --extra dev python -m tests.canonical.lmtp_ltmle_competing_crossfit.regenerate `
  --replicates 8 --n 500 --primary-only --output $env:TEMP\lmtp-competing-crossfit-probe
```

Regenerate the committed artifacts and document after the smoke study passes:

```powershell
uv run --extra dev python -m tests.canonical.lmtp_ltmle_competing_crossfit.regenerate
uv run --extra dev python -m tests.studies.evidence.document `
  --slug canonical-ltmle-competing-crossfit
```

The pinned Docker image and both adapters are source inputs in the manifest. The exact-law and
Gateaux modules remain the scientific oracles.
