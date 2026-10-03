# Randomized multi-arm missing-outcome DR-TMLE evidence artifacts

This directory holds the registered study of `DRTMLE` with `delta=` at three arms. The law is L3
of `tests/studies/mar_arm_indexed_laws.py`. Its treatment table is a known W-stratified
randomization, which the fit receives by level.

The comparator is pinned R `drtmle` 1.1.2 at the both-correct limit. R `drtmle` fluctuates one
joint treatment-response mechanism with a composite response. Díaz and van der Laan (2017,
page 25) reject that construction for this problem. The pairing therefore shows the shared limit
only. The property study tests the armwise five-reduction construction.

Run a disposable smoke study into a scratch directory outside the repository:

```console
python -m tests.canonical.drtmle_mar_multi_arm.regenerate --replicates 8 --output <scratch> --skip-properties --allow-failures
```

A declared run passes `--output <scratch>` and optionally `--jobs`, and nothing else.
`tests/canonical/declared_run.py` describes the guard, the run log and the copy into this
directory.
