# Diagnosis of the pilot's failed in-sample refits

The committed pilot lost 2 of its 120 in-sample fits (draws 38 and 45). Each withheld its bootstrap
(refusal C6) because 7 and 3 of its 200 refits raised `DataError: An initial hazard fit puts a
combined jump of one or more at some time`. Over the pilot, 27 of 24,000 in-sample refits failed
and none of the 2,400 cross-fitted ones did. The probe
`tests/studies/continuous_survival_bootstrap_ties.py` rebuilds those bootstraps from their seeds,
reads the hazard jump that trips the guard, and screens original samples for the same failure.
Each fit stops after the engine's checks of the initial hazards, so no walk runs.

```
python -m tests.studies.continuous_survival_bootstrap_ties --jobs 4 --draws 400 --bootstrap 38 45 0 1
```

This note diagnoses only. The engine is unchanged.

## What trips the guard

The guard in `ContinuousSurvivalTMLE._initial` sums the cause hazard jumps of each unit at both
arms, on the grid up to the last target time. It refuses when that sum reaches 1 for any unit,
arm and time. Censoring is not part of the sum. All 10 failing refits read the same way.

| reading | draws 38 and 45, 10 failing refits |
| --- | --- |
| hazard | cause 2 every time; the cause-1 jump at that time is 0 |
| selected member for cause 2 | `CoxHazard('interaction')` every time (cause 1: interaction or main) |
| time | 1.62 to 2.32, before the last target time 2.32; 107 to 153 distinct event times lie later, and follow-up runs to 12.8 to 23.2 |
| at risk | 255 to 353 rows, 163 to 231 distinct units; the most copies of one unit at risk is 5 or 6 |
| events at that time | 3 to 5 tied events, all copies of one unit |
| offending unit and arm | a control unit read at the treated arm (its counterfactual arm); unit 833 in 6 of draw 38's 7 failures, unit 961 in all 3 of draw 45's |
| that unit's risk score $e^{x^\top\hat\beta}$ at the treated arm | 159 to 428 |
| largest risk score of any at-risk row at its own arm | 8.3 to 28; the median is 1.1 to 1.4, and the sum is 515 to 861 |
| Breslow baseline jump | 0.004 to 0.008, equal to (tied events) / (sum of the risk set's scores) to 1e-15 |
| combined jump | 1.06 to 2.30 |

The mechanism is the product of two factors, and the tied-copies hypothesis holds only in part:

1. **Tied copies of the event unit.** In a resample, a unit drawn k times that has a cause-2 event
   at time t contributes k tied events. The Breslow baseline jump at t is k times what one event
   gives. The tied unit is not the offending unit; the offending unit itself had 1 to 4 copies.
2. **Counterfactual extrapolation of the interaction model.** The guard reads both arms, and the
   cause-2 interaction Cox model gives a few control units a treated-arm score 9 to 35 times the
   largest score in the risk set. A unit's jump is its score times the baseline jump.

With one event at the time (k = 1), the same units' jumps would be 0.24 to 0.50. In draws 0 and 1,
no refit failed, and the largest combined jump per refit reached 0.82 and 0.39. In the two
failing draws, the 99th percentile of the largest jump per refit was 1.22 and 1.33. The failures
cluster in the original samples that contain such a unit, which is why one draw lost 7 of 200
refits while most lost none.

## On original samples

The same failure needs no tie: a counterfactual score near the risk set's total score suffices.
Screen of 400 base-law samples per cell, stopped after the checks:

| n | fit | combined jump ≥ 1 | ≥ 0.5 | median of the largest jump | 99th percentile |
| --- | --- | --- | --- | --- | --- |
| 1,000 | in sample | 0 of 400 | 0 | 0.047 | 0.15 (largest 0.29) |
| 1,000 | cross-fitted | 0 of 400 | 0 | 0.056 | 0.15 (largest 0.29) |
| 200 | in sample | 7 of 400 (1.8%) | 24 | 0.15 | 1.4 |
| 200 | cross-fitted | 27 of 400 (6.8%) | 66 | 0.23 | 4.0 |

At n = 200, another 43 samples raised earlier because an arm had no cause-2 event by the first
target time; their largest jump was not read. The registered records agree:

- `continuous-survival-variance` lost 8 of 6,000 `v3_half_n` fits (n = 500, cross-fitted) to this
  refusal, 0.13% (`failures.csv`).
- `continuous-survival-functionals` lost 1 of 14,800.
- `continuous-time-survival` (7,600 of 7,600) and `continuous-survival-causes` (7,200 of 7,200 per
  cell) lost none.

So at n = 1,000 the refusal is rare on original data, and the bootstrap's ties raise the largest
jump by a factor of about 2.5 to 3.5 (a 99th percentile of 0.38 to 0.51 in the draws without
failures, against 0.15).

The existing last-time handling does not cover it:

- `last_time_floor_share` counts (unit, arm) pairs whose $\hat\pi \hat S_c$ is below the
  positivity floor at the last time. It bounds the treatment and censoring weights and never
  touches a cause hazard.
- The walk's `frozen_share` holds a pair whose survival a step would take below $10^{-8}$. It acts
  on the walk's steps, after this check.
- No step of the original fit treats the last at-risk time of a cause hazard specially. The
  failing times sit mid-follow-up with hundreds at risk, so a rule for the end of follow-up would
  not reach them.

## Options

| option | theory and effect on the estimand | registered results it changes | fast witness |
| --- | --- | --- | --- |
| A. Efron ties instead of Breslow | For k tied copies of one unit, Efron's baseline jump is $\sum_{l<k} 1/(\Sigma_R - l s)$, which is at least $k/\Sigma_R$. It raises the jump, so it cannot help. | none (original times are continuous, so there are no ties) | none; rejected |
| B. Bound the combined jump (recommended) | Where a unit's cause jumps at one arm and time sum to at least $1 - \delta$, with $\delta$ the walk's floor $10^{-8}$, scale them to sum to $1 - \delta$, keeping each cause's share. This projects the step onto a discrete hazard (survival factor in $[\delta, 1]$), so the initial survival stays a probability. The continuous-time hazard has no atoms, and the bound acts only where the Breslow atom of an extrapolated score passes 1, an event whose probability tends to 0 at a fixed law. So the initial fit's consistency, the walk's score equation and the estimand are unchanged. In finite samples the affected pair's survival falls to $\delta$, the model's own claim that the event is certain, and the walk then holds the pair as it does now. Applied in every fit, since a replicate must be the estimator. | `continuous-survival-variance` gains 8 `v3_half_n` fits and `continuous-survival-functionals` 1; `continuous-time-survival` and `continuous-survival-causes` lost no fit, so they stay bit for bit (unless some kept fit had a combined jump within $10^{-8}$ of 1) | a user hazard learner whose counterfactual-arm jump exceeds 1 at one time: raises now, fits after, with that pair's survival $\delta$ after the time and the cause split kept. An anchored dense-walk fit stays bit for bit. Mutations: a threshold of 0.5 breaks the anchor, and dropping the share breaks the split. Pilot draws 38 and 45 refit with no failure. |
| C. Break the bootstrap's ties (jitter copies by a vanishing amount) | A smoothed bootstrap. It removes the factor k but not the extrapolation. Its validity needs its own reading of the smoothed bootstrap delta method, and the in-sample label reason would change. | none (bootstrap only) | copies of a unit get distinct times; a replicate equals the refit on the jittered resample |
| D. Count these refits differently in C6 and the failure bound | A bootstrap over the resamples that fit conditions on fitting. C6 exists to refuse that, and the decided design keeps it. A per-refit bound pooled over the study hides draws such as 38, where 3.5% of refits fail. | none | boundary tests of the new count |
| E. Exponential product $S = \prod e^{-d\Lambda}$ everywhere | Asymptotically equivalent (the Breslow survival rather than the product limit), and never fails, but every fit moves. | all four | every exact-law test of the walk |
| F. Clip counterfactual scores to the observed range | Changes the initial estimator wherever it extrapolates, not only where it fails. | all four | clipped scores |

**Recommendation: B.** It treats the estimator's refusal itself, which also costs the registered
studies fits at small n (1.8% in sample and 6.8% cross-fitted at n = 200), rather than a symptom
the bootstrap amplifies. It moves no fit that runs today. Its asymptotic theory is unchanged
because the bound is inactive with probability tending to one. Its cost is regenerating
`continuous-survival-variance` and `continuous-survival-functionals`, whose verdicts 9 added fits
cannot plausibly move. That claim needs the regenerations, not this note. Until B lands, the declared
run stays refused on the failure line, and the extension to 400 draws should wait for it, since
its fits would be redrawn under the new guard.
