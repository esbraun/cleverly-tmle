# Canonical DR-TMLE evidence

This registered protocol compares Cleverly with `benkeser/drtmle` 1.1.2. The source is
pinned at commit `538a3a264c1ca984b6d88978ca7f96165f43152c`.

Both implementations use the same paper-law sample, initial out-of-fold nuisance arrays,
and ten-fold assignment. The R package fits its own reduced regressions and runs its own
targeting loop. The study reports arm means and the ATE for complete binary data.

Run a disposable smoke comparison first:

```powershell
uv run --extra dev python -m tests.canonical.drtmle.regenerate `
  --replicates 2 --n 100 --skip-properties --allow-failures `
  --output $env:TEMP\drtmle-smoke
```

Run the frozen primary and property protocols without overrides to publish evidence. The
study uses the `reporting` publication policy. Scientific failures remain in the artifacts
and generated documentation, but incomplete rows or invalid provenance stop regeneration.

The R reference is substantially more memory-intensive than the Python subject. Tune the
two pools independently; on a Docker host with about 16 GiB available, seven persistent R
workers and sixteen Python workers fit the full protocol:

```powershell
New-Item -ItemType Directory -Force .evidence-cache\drtmle | Out-Null
uv run --extra dev python -m tests.canonical.drtmle.regenerate `
  --jobs 16 --r-jobs 7 --cache .evidence-cache\drtmle
```

The cache holds the generated samples, truths, and Python rows so a reference-process failure
can be retried without changing or recomputing the paired subject phase.
