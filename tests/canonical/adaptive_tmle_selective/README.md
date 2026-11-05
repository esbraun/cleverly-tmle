# adaptive-tmle-selective (X90 T3, declared, not yet run)

The gated study of cell S6 (selective enrollment, a large wiggly bias, Super Learner nuisances,
both forms) and control K6 (main-terms GLM nuisances on the same draws, reported only: a pilot found it does not under-cover at n = 3,200).
The module docstring of `tests/studies/adaptive_tmle_selective.py` is the declaration. The
flexible fits cost two orders more than study A's, so this study goes to the local queue.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m tests.canonical.adaptive_tmle_selective.regenerate --output /tmp/atmle-selective-run --jobs 4
```

## Cost (1 core, one timed replicate, four workers busy elsewhere)

| cell | n | reps | s/rep | core-hours |
| --- | --- | --- | --- | --- |
| s6_cv (SuperLearner, 5 outer folds) | 3200 | 800 | 264 | 58.7 |
| s6_pooled | 3200 | 800 | 87 | 19.4 |

78 core-hours: about 20 wall-hours on 4 workers, so this study goes to the local queue.
