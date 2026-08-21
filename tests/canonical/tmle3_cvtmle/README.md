# Stacked point-treatment CV-TMLE versus R `tmle3`

This directory freezes a paired repeated-sampling comparison of cleverly's stacked
point-treatment CV-TMLE construction with R [`tmle3`](https://github.com/tlverse/tmle3) at
commit `ed72f8a` and [`sl3`](https://github.com/tlverse/sl3) at commit `0e8f236`.  The R
image is pinned by digest in the manifest and Dockerfile.

Python generates every sample and its treatment-stratified ten-fold assignment.  The R
runner reconstructs those exact validation indices with `origami`, wraps the corresponding
GLM learners in `Lrnr_cv`, and uses `tmle3_Update(cvtmle = TRUE)`.  Both implementations
therefore use out-of-fold nuisance predictions, one update over the stacked validation rows,
and a whole-sample plug-in evaluation.  The runner aborts on any failed fit, missing estimand,
or changed fold assignment.

The study covers arm means, ATE, ATT, ATC, observed mean, and PAR under binary and bounded
continuous outcome laws, plus PAF, RR, and OR under the binary law.  PAF interval scales are
declared incomparable: `tmle3` transforms a log-risk-ratio interval while cleverly reports
the fraction-scale influence-curve interval.

Its samples come from the seed on its own study record rather than from the ordinary-TMLE study's,
so this row is a separate draw and not the same experiment reported twice.

Regenerate from the repository root with Docker running:

```powershell
uv run --extra dev python -m tests.canonical.tmle3_cvtmle.regenerate
```

For a deliberately underpowered smoke run, write to a disposable directory and allow the
evidence gates to remain inconclusive:

```powershell
uv run --extra dev python -m tests.canonical.tmle3_cvtmle.regenerate --replicates 10 --allow-failures --output $env:TEMP\cvtmle-probe
```

The Python declaration is `tests/studies/canonical_cvtmle.py`; the reader-facing scope,
measurements, and limitations are in
[stacked point-treatment CV-TMLE study page](../../../docs/technical-reference/method-evidence/stacked-point-treatment-cvtmle.md).
Raw samples are temporary.  The committed replicate results and manifest retain everything
needed to reproduce the statistical summaries, paired decisions, provenance, and hashes.
