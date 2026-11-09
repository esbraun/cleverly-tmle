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

## Declaration history

- 2026-11-09, fresh declaration after the selective study's red `s6_cv` cell (SE ratio 1.160,
  interval 1.114 to 1.209, run at bd99bbd4; artifacts kept on branch
  `claude/x90-selective-red-run`, commit 7be1f9b2). Diagnosis: a defect, not a finite-sample
  limit. The CV form's variance applied target directions estimated on the training rows to the
  validation rows, which estimates the fold variance at the wrong direction and overstates it
  (proxy on 100 `s6_cv` draws: SE ratio 1.233 before, 0.993 after; point estimates
  bit-identical). Fixed in 3b29c757: the directions are estimated on every row with the fold's
  selected columns. Also result-determining since the last run: outer and inner folds stratified
  by trial membership (and by outcome for a binary outcome), as atmle does (a779f233), and a
  `force_arm` option whose default leaves the fit unchanged. Laws, margins, cells, seeds, learners
  and replication counts are unchanged. The sizing pilots predate these changes; the rule's count
  is unchanged at the proxy coverages (the selective count stays at the rule's floor of 2,400 for
  `s6_cv` coverage down to 0.94 at the 1,200-replicate pilot).
