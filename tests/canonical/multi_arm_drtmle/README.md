# Multi-arm DR-TMLE versus R `drtmle`

This registered study gives Cleverly and pinned R `drtmle` 1.1.2 the same samples,
five-fold assignment, and initial out-of-fold nuisance predictions. `random_partition`
draws that assignment from the row count and the sample's seed, so it reads no column of
the data. Each implementation
runs its own armwise reduced regressions and correction cycle. The row uses the reporting
policy because the source theorem is binary; any multi-arm divergence remains visible in
the committed results instead of being converted into a theorem claim.

Regenerate from the repository root with Docker running. Set every thread variable
(`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and
`NUMEXPR_NUM_THREADS`) to 1, and run nothing else on the machine.

A declared run takes `--output` and `--jobs` only. `--output` names an empty scratch directory
outside the repository. The run refuses to start unless the tree is clean, `HEAD` equals its
upstream, `cleverly` imports from this tree's `src`, and every thread variable is 1. It writes
the artefacts, the manifest and `run.log` to the scratch directory, then copies them here. The
manifest then records the state of the tree at the start of the run. Run this command:

```bash
python -u -m tests.canonical.multi_arm_drtmle.regenerate --jobs 16 --output <empty scratch directory>
```

A smoke run is any other `--replicates`. It writes to a scratch `--output` outside the
repository and skips the property study:

```bash
python -u -m tests.canonical.multi_arm_drtmle.regenerate --replicates 48 --jobs 16 --allow-failures --skip-properties --output <scratch>
```

The run form is in `tests/canonical/declared_run.py`.

The reader-facing results are in
[`multi-arm-dr-tmle.md`](../../../docs/technical-reference/method-evidence/multi-arm-dr-tmle.md).
