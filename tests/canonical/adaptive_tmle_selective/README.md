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

## Sizing and cost

`PRIMARY_REPLICATES` is `repeated_sizing.primary_replicates` at the pilot that
`tests/unit/test_adaptive_tmle_trial_design.py` pins as `SELECTIVE_PILOT_COVERED`. With no pilot
pinned it is the rule's floor, 2,400 (the bias clause and the per-cell coverage clause). Before
the declared run:

```bash
python -m tests.canonical.adaptive_tmle_trial.pilot --study selective --replicates 400 --jobs 16 > pilot-selective.json
```

Copy its covered counts into `SELECTIVE_PILOT_COVERED` and set `PRIMARY_REPLICATES` to its
`primary_replicates`. One timed replicate (four workers busy elsewhere) took 264 s for `s6_cv`
and 87 s for `s6_pooled`, so 2,400 replicates cost about 235 core-hours, about 15 wall-hours on
the local queue's 16 cores.
