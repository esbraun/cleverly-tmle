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
`tests/unit/test_adaptive_tmle_trial_design.py` pins as `SELECTIVE_PILOT_COVERED`. The pinned
pilot has 1,200 replicates and covered counts of 1,159 (`s6_cv`) and 1,126 (`s6_pooled`), which
give `PRIMARY_REPLICATES = 2,400`, the rule's floor. A replicate set costs 545.46 core-seconds, so
the declared run costs about 364 core-hours, about 23 wall-hours on the local queue's 16 cores.

The first pilot had 400 replicates (388 and 371 covered), which priced the run at 7,600. Replicate
seeds depend only on the record seed, the scenario and the replicate index, so the 1,200-replicate
pilot reproduced those 400 and added 800:

```bash
python -m tests.canonical.adaptive_tmle_trial.pilot --study selective --replicates 1200 --jobs 16 > pilot-selective.json
```
