# Canonical DR-TMLE evidence

This registered protocol compares Cleverly with `benkeser/drtmle` 1.1.2. The source is
pinned at commit `538a3a264c1ca984b6d88978ca7f96165f43152c`.

Both implementations use the same paper-law sample, initial out-of-fold nuisance arrays,
and ten-fold assignment. That assignment comes from `cleverly.random_partition`, which reads
the row count and the sample's seed and no column. The R package fits its own reduced
regressions and runs its own targeting loop. The study reports arm means and the ATE for
complete binary data.

Two settings are declared on both sides and one diagnostic is not comparable.

- The alternation runs 100 rounds each way: `max_outer=100` here, `maxIter = 100` in R.
  `max_iter` is the Newton cap inside one fluctuation and has no counterpart in R.
- The score audit is `score_check`'s own bar, `1e-3 x se / sqrt(n)`, computed in Python from
  each side's own reported standard errors. `run_drtmle.R` emits the raw `score_max` and
  decides nothing.
- `solver_passed` exists on the Cleverly side only, because R `drtmle` reports no convergence
  flag. `solver_reported` says which side has one. The reference's cell is empty rather than
  filled with a pass.

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

The property phase now runs a size ladder for `double_robust_contraction` at n = 1,500, 3,000
and 6,000. The two outer rungs run 2,400 replications each and the middle rung runs 800,
because the middle rung's centred weight in the fitted slope is exactly zero. Under cost
proportional to n, that budget is 2.429 times the flat 800-per-cell ladder, so the measured
6.4 core-hours becomes roughly 15.5. Budget for it before regenerating.

The extra draws serve the fitted slope alone. Each rung's own coverage verdict is read at 800
replications, which `CONTRACTION_VERDICT_REPLICATES` in `tests/studies/drtmle_properties.py`
declares. A larger budget narrows the exact interval around the rung's true coverage and can
resolve an inconclusive verdict. The extra budget was declared for the slope and not for the
rung verdicts. The published `replicates` column reports the 800 each rung's verdict used, and
the three rate rows report every replication the slope consumed.

The run that produced these rows also moved the environment. SciPy moved from 1.17.1 to 1.18.0,
and Python moved from 3.11.13 to 3.13.7, on that run. `manifest.json` records the versions the
committed rows came from, and its git history records the versions they replaced. Every
difference from the rows these replaced therefore carries the environment move as well as the
ladder change. Do not attribute such a difference to either one alone.
