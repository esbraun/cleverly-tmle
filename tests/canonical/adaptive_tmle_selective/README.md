# adaptive-tmle-selective (X90 T3, run 2 accepted by a named exception)

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

## Replaying the committed run

The committed run predates two result-determining changes, so a replay at the current code does
not reproduce its rows bit for bit. It took the lasso candidate from `default_library(task)` with
no seed, so for a binary target the SAGA solver drew its sample order from numpy's global
generator, and no two runs agreed: an A/B replay of 20 replicates moved every row, by a median of
2e-5 and at most 0.007 in the estimate. `SuperLearner(random_state=...)` now gives that candidate
the ensemble's seed. The run also predates `meta_learner="auto"` picking `cc_ls` for the
regression outcome ensemble, where the run used NNLS. A replay therefore agrees with these rows
only to within those two changes, until the study is rerun.

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
- 2026-11-10, run 2 accepted by the owner's named exception, with no repeat run (declared at
  c6f9e51c; artifacts kept on branch `claude/x90-selective-red-run-2`, commit 8f2a91a5, and
  committed unedited here). The run read `s6_cv` with a bias of 0.0290 (0.20 SD, 99% interval
  0.0215 to 0.0364, margin 0.0355), a coverage of 0.9367 and an SE ratio of 0.953, and
  `s6_pooled` inside every margin (bias 0.0129, 0.10 SD). Diagnosis: no defect. The projection
  truth is within 0.0035 of Psi and the projection coverage is 0.939, so it is no projection gap.
  The cause is a finite-sample O(K/n_v) term of the CV form's targeting: about 37 selected bias
  columns on 640 validation rows, and 0.0161 more bias than the pooled form on different draws.
  The owner decided in writing on 2026-11-10: "if the bias is not a defect, i don't want to run
  a repeat study - that's wasteful. let's make any exception and accept the study as valid." The
  study keeps `publication_policy="gated"`, the policy its run recorded. The exception is
  `ACCEPTED_BY_EXCEPTION` in `tests/studies/evidence/red_cells.py`, keyed to this study and the
  `s6_cv/psi` truth row and to the `replicates.parquet` hash of this run. It fails if the cell
  turns green or the study is rerun. No margin, count, law, seed, learner or label changed, and
  no roadmap owner row exists for the cell.
