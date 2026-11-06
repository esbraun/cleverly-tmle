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
pilot has 400 replicates and covered counts of 388 (`s6_cv`) and 371 (`s6_pooled`), which give
`PRIMARY_REPLICATES = 7,600`. A replicate set costs 546.07 core-seconds, so the declared run
costs about 1,150 core-hours, about 72 wall-hours on the local queue's 16 cores.

The bound comes from the pilot's size. A larger pilot is the valid way to tighten it: 1,200
replicates at the same coverage would give about 3,600. A 1,200-replicate pilot is planned
before the declared run. Replicate seeds depend only on the record seed, the scenario and the
replicate index, so this command reproduces the first 400 replicates and adds 800 more:

```bash
python -m tests.canonical.adaptive_tmle_trial.pilot --study selective --replicates 1200 --jobs 16 > pilot-selective.json
```

Copy its covered counts into `SELECTIVE_PILOT_COVERED`, set `SELECTIVE_PILOT_REPLICATES = 1200`
and set `PRIMARY_REPLICATES` to its `primary_replicates`.
