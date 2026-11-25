# Diagnosis of the bootstrap pilot

This note holds two diagnoses: the failed in-sample refits of the first pilot, which led to the
jump bound, and [the calibration rows below 0.90](#calibration-rows-below-090-in-the-pilot-under-the-bound)
in the pilot rerun under the bound.

The first pilot lost 2 of its 120 in-sample fits (draws 38 and 45). Each withheld its bootstrap
(the 1% failure rule) because 7 and 3 of its 200 refits raised `DataError: An initial hazard fit puts a
combined jump of one or more at some time`. Over the pilot, 27 of 24,000 in-sample refits failed
and none of the 2,400 cross-fitted ones did. The probe
`tests/studies/continuous_survival_bootstrap_ties.py` rebuilds those bootstraps from their seeds,
reads the hazard jump that trips the guard, and screens original samples for the same failure.
Each fit stops after the engine's checks of the initial hazards, so no walk runs.

```
python -m tests.studies.continuous_survival_bootstrap_ties --jobs 4 --draws 400 --bootstrap 38 45 0 1
```

This note was written as a diagnosis, with the engine unchanged. The owner adopted option B on
2026-11-20; [the decision](#decision-option-b-adopted) at the end says what shipped.

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
- `continuous-time-survival` lost none of its 7,600 gated fits, but its reported positivity cell
  `positivity_4k` lost replicate 292 to this refusal (see the correction below).
  `continuous-survival-causes` (7,200 of 7,200 per cell) lost none.

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
| B. Bound the combined jump (recommended) | Where a unit's cause jumps at one arm and time sum to at least $1 - \delta$, with $\delta$ the walk's floor $10^{-8}$, scale them to sum to $1 - \delta$, keeping each cause's share. This projects the step onto a discrete hazard (survival factor in $[\delta, 1]$), so the initial survival stays a probability. The continuous-time hazard has no atoms, and the bound acts only where the Breslow atom of an extrapolated score passes 1, an event whose probability tends to 0 at a fixed law. So the initial fit's consistency, the walk's score equation and the estimand are unchanged. In finite samples the affected pair's survival falls to $\delta$, the model's own claim that the event is certain, and the walk's survival floor is affected (see the correction below). Applied in every fit, since a replicate must be the estimator. | `continuous-survival-variance` gains 8 `v3_half_n` fits, `continuous-survival-functionals` 1 and `continuous-time-survival` 1 (`positivity_4k` replicate 292); `continuous-survival-causes` lost no fit. That nothing else moves needs an A/B replay at `rtol 0`, not the failure files alone | a user hazard learner whose counterfactual-arm jump exceeds 1 at one time: raises now, fits after, with that pair's survival $\delta$ after the time and the cause split kept. An anchored dense-walk fit stays bit for bit. Mutations: a threshold of 0.5 breaks the anchor, and dropping the share breaks the split. Pilot draws 38 and 45 refit with no failure. |
| C. Break the bootstrap's ties (jitter copies by a vanishing amount) | A smoothed bootstrap. It removes the factor k but not the extrapolation. Its validity needs its own reading of the smoothed bootstrap delta method, and the in-sample label reason would change. | none (bootstrap only) | copies of a unit get distinct times; a replicate equals the refit on the jittered resample |
| D. Count these refits differently in the 1% rule and the failure bound | A bootstrap over the resamples that fit conditions on fitting. The 1% rule exists to refuse that, and the decided design keeps it. A per-refit bound pooled over the study hides draws such as 38, where 3.5% of refits fail. | none | boundary tests of the new count |
| E. Exponential product $S = \prod e^{-d\Lambda}$ everywhere | Asymptotically equivalent (the Breslow survival rather than the product limit), and never fails, but every fit moves. | all four | every exact-law test of the walk |
| F. Clip counterfactual scores to the observed range | Changes the initial estimator wherever it extrapolates, not only where it fails. | all four | clipped scores |

**Recommendation at first writing: B.** It reverses the recorded decision that this refusal is a
recordable `DataError`, so it waits for the owner's call, and the review below lists what it must
carry. It treats the estimator's refusal itself, which also costs the registered
studies fits at small n (1.8% in sample and 6.8% cross-fitted at n = 200), rather than a symptom
the bootstrap amplifies. It moves no fit that runs today. Its asymptotic theory is unchanged
because the bound is inactive with probability tending to one. Its cost is regenerating
`continuous-survival-variance`, `continuous-survival-functionals` and `continuous-time-survival`,
whose gated verdicts 10 added fits cannot plausibly move (`positivity_4k` is reported, not gated). That claim needs the regenerations, not this note. Until B lands, the declared
run stays refused on the failure line, and the extension to 400 draws should wait for it, since
its fits would be redrawn under the new guard.

## Correction after an independent review

The review reproduced draw 45's three failures exactly and read the offending units in their
original samples. Its readings correct this note in three places.

**Positivity is the main cause, not the ties.** The two offending units are extreme in X3, and the
treated arm does not cover them:

| draw, unit | X3 | rank in n = 1,000 | true $\pi(1 \mid x)$ | lowest treated X3 | original-fit treated-arm score, interaction / main | largest own-arm score, interaction / main |
| --- | --- | --- | --- | --- | --- | --- |
| 45, 961 | -3.28 | lowest | 0.014 | -2.12 | 21.7 / 8.8 | 18.2 / 19.1 |
| 38, 833 | -3.46 | lowest | 0.012 | -2.48 | 38.9 / 16.0 | 25.2 / 26.9 |

Their true treatment probability is about 0.013. On the original data the interaction member
already scores them past every factual score. The bootstrap adds noise to that member's
interaction coefficients, raising the score to 159 to 428, and the ties multiply the baseline atom.
Their $\hat\pi \hat S_c$ at the last time most likely lies under the `auto` positivity floor
$5/(\sqrt n \log n) = 0.023$, so `last_time_floor_share` most likely already counts them; the review
read the true $\pi$, not $\hat\pi$. The three refitted `v3_half_n` failures of the
switched-variance study read the same way on original samples, with no ties: a unit whose X3 is
among the sample's three most extreme, read at the arm whose range does not cover it, score 22 to 39
times the largest in a risk set of 141 to 148 units.

**`continuous-time-survival` has a fit of this kind.** Its reported cell `positivity_4k` (the
positivity-violation law, n = 4,000) lost replicates 283 and 292 to `DataError`
(`reported-replicates.csv`). Replicate 292 trips this guard, with a combined jump of 6.83 (cause 1,
unit 843 at the control arm). Replicate 283 trips the survival-underflow guard instead, with a
largest jump of 0.977. So option B moves `continuous-time-survival` too. The guard is mainly a
positivity signal.

**Option B's walk claim is wrong.** The walk's floor is `min(WALK_SURVIVAL_FLOOR, lowest)` over
every unit's initial survival at the grid's end. A bounded pair's survival there is below
$10^{-8}$, so in every bounded fit it would lower the floor for all pairs rather than be held. B
also needs a record field, a warning and an experimental label for a fit in which it acts, and a
witness with two nonzero causes at the bounded cell. All 10 failures here have a cause-1 jump of
0, so a witness of that shape cannot see the cause split. The review in the plans repository
lists these accompaniments.

The refusal's message no longer blamed a late time with few units at risk. It named the
extrapolation to a counterfactual arm outside that arm's covariate range, which ties can amplify.

## Decision: option B adopted

The owner adopted option B, amended as the review asked, on 2026-11-20. The refusal is gone; an
infinite combined jump is still refused, and a NaN jump is walked as before.

- **Placement.** `cleverly.survival._chunked.bound_jumps` runs inside `InitialHazards.jumps`, the
  one source of the engine's checks, its replay check and every walk block, in every fit and
  bootstrap refit alike. It acts on cells whose sum is at least 1, exactly the cells the old
  code refused, and scales them to $1 - \delta$, $\delta = 10^{-8}$ (`JUMP_BOUND`, a constant of
  its own beside the walk's floor). It writes nothing when no cell reaches 1, so every fit that
  ran before computes the same arrays. (The first version acted from $1 - \delta$; the
  implementation review narrowed it to 1 so the bound moves provably nothing that ran.)
- **The walk's floor.** Each bounded pair takes its own floor, the lower of the other pairs'
  floor and its own initial survival at the grid's end; the other pairs' floor reads the other
  pairs only (`run_walk(own_floor=...)`). A fit with no bounded pair runs the old code path.
- **Record, warning and label.** `ContinuousSurvivalRecord.jump_bound_share` and
  `jump_bound_at_risk_share`; one `PositivityWarning` per fit naming the arm, the causes and the
  pairs; every estimate experimental with a reason that names the extrapolation;
  `result.bootstrap_jump_bounded` counts the refits where the bound acted, and every bootstrap
  interval's reason names it. The continuous studies' `targeting-readings.csv` gain
  `mean_jump_bound_share` and `mean_jump_bound_at_risk_share`, and the bootstrap study's `mean_jump_bounded_refits`, at their next
  regeneration.
- **Witnesses.** `tests/unit/test_continuous_survival_jump_bound.py` (two nonzero causes at a
  counterfactual cell, a cell at the received arm while at risk, the threshold) and
  `tests/unit/test_continuous_survival_blocks.py` (the bounded walk against the dense walk, the
  own floor, the underflow that a bounded survival can still reach).
- **A/B replay.** `tests/studies/continuous_survival_jump_bound_replay.py` refits replicates of
  the four registered continuous studies at the base commit and with the bound, and compares
  them at `rtol 0`. Its readings are below.

The replay ran 395 fits on each side: 40 replicates spread over each of the nine primary cells,
six of each reported cell of `continuous-time-survival`, and every fit the committed records lost
to the refusal, with `positivity_4k` replicate 283 (an underflow) as a control. It ran again after the
bound was narrowed to sums of at least 1, with the same counts.

| outcome | fits |
| --- | --- |
| identical estimates and standard errors at `rtol 0` | 384 (causes 80, functionals 40, variance 200, continuous-time 64) |
| refused before, fit now | 7 (`v3_half_n` replicates 657, 1209, 2184, 3179, 4433, 4586, 5932) |
| refused before, refused now on survival underflow | 3 (`v3_half_n` 4272, `functionals_cf_1k` 10023, `positivity_4k` 292) |
| refused on underflow on both sides | 1 (`positivity_4k` 283) |
| changed, or newly failing | 0 |

In the three that still fail, one pair's jumps reach the bound at 32, 27 and 1,141 grid times,
with raw sums up to 1.9, 1.3 and 6.8, so its survival is $\delta$ to that power and falls below
$10^{-250}$. The refuter calibration's trial 465, outside the replay, fails the same way
at subset refit 8. The bound keeps the refusal for a pair the model extrapolates at many times; it
rescues a pair that crosses one at a few times, as every bootstrap refit of draws 38 and 45 did.
Refitted whole, draw 38 lost 0 of 200 refits with the bound acting in 7, and draw 45 lost 0
with the bound acting in 3
(`python -m tests.studies.continuous_survival_bootstrap_ties --whole --bootstrap 38 45`).

## Calibration rows below 0.90 in the pilot under the bound

The pilot was rerun whole under the bound (main 7b4d0f08): 400 in-sample and 50 cross-fitted draws,
each a fit with 200 refits. Fifteen of the first pilot's 120 in-sample draws had refits the old
engine refused, so extending that file would have mixed two engines. No fit withheld its bootstrap.
3 of 80,000 in-sample refits failed (one in each of 3 draws), and the bound acted in 101 refits of
46 draws. No cross-fitted refit failed, and the bound acted in 17 refits of 7 draws.

At the declared 3,200 fits, 11 of the 12 gated rows pass the calibration rule with a plug-in
probability below 0.90 (`regenerate.pilot_refusals`), four of them at 0.000. The rule reads each
row's pilot bootstrap SE ratio and percentile coverage. Both rest on the pilot's 400 estimates,
whose empirical SD has a relative Monte Carlo SD of $1/\sqrt{2 \cdot 399} = 0.035$.

The shortfall is shared with the Wald interval on the same fits, so it is not the bootstrap's. On
each row the mean bootstrap SE is 0.983 to 1.030 times the mean Wald SE, a paired reading with a
Monte Carlo SE of 0.0025 to 0.0056 (delta method) that does not involve the empirical SD. The percentile coverage
minus the Wald coverage on the same draws is -0.015 to +0.010 by row, -0.003 on average. The low
SE ratios come from the denominator: the pilot's Wald SE ratios on these rows read 0.912 to 1.044.

`continuous-survival-variance`'s cell `v4_in_sample` measures the same Wald interval on the same
fit: the same law (its sampler returns the registered sampler's frame for the same seed, and its
truths are equal), $n = 1{,}000$, `cross_fit=False`, the default libraries and folds, on 6,000 fits.
There the Wald SE ratios of these 12 rows are 0.954 to 1.003 and the coverage 0.937 to 0.951. The
pilot's Wald SE ratios sit 0.3 to 2.3 of their Monte Carlo SDs below those of `v4_in_sample` on
10 of the 12 rows, deepest on the treated cause-2 risks, which share draws with the cause-2
differences. A largest deviation near 2.3 SDs over 12 correlated rows is within chance, so this is
the Monte Carlo spread of the pilot's 400 samples, not a property of the law. The 6,000 disjoint
fits of `v4_in_sample` stand in for the extension of the pilot that `guide/studies.md` asks for
when a row's plug-in pass is below 0.90: they measure the same estimator's Wald SE ratio and
coverage on 15 times the pilot's draws, and an extension would add only each row's own percentile
minus Wald coverage difference.

The anchored reading below multiplies each row's paired bootstrap-to-Wald SE by its
`v4_in_sample` Wald SE ratio, and takes `v4_in_sample`'s coverage, since the percentile and Wald
coverages agree on average.

| row | pilot percentile coverage | pilot bootstrap SE ratio | pilot Wald SE ratio | bootstrap / Wald SE | `v4_in_sample` Wald SE ratio | `v4_in_sample` coverage | anchored bootstrap SE ratio | plug-in pass, pilot | plug-in pass, anchored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cif_regimen[treated, cause1 @ t=1.12]` | 0.940 | 0.991 | 0.995 | 0.996 | 0.983 | 0.943 | 0.979 | 0.955 | 0.917 |
| `cif_regimen[treated, cause1 @ t=2.32]` | 0.938 | 0.941 | 0.939 | 1.002 | 0.984 | 0.946 | 0.987 | 0.046 | 0.978 |
| `cif_regimen[treated, cause2 @ t=1.12]` | 0.917 | 0.914 | 0.923 | 0.991 | 0.981 | 0.940 | 0.972 | 0.000 | 0.780 |
| `cif_regimen[treated, cause2 @ t=2.32]` | 0.930 | 0.897 | 0.912 | 0.983 | 0.990 | 0.938 | 0.974 | 0.000 | 0.760 |
| `cif_regimen[control, cause1 @ t=1.12]` | 0.932 | 0.975 | 0.952 | 1.024 | 0.964 | 0.937 | 0.987 | 0.447 | 0.844 |
| `cif_regimen[control, cause1 @ t=2.32]` | 0.935 | 0.967 | 0.950 | 1.017 | 0.980 | 0.944 | 0.996 | 0.497 | 0.994 |
| `cif_regimen[control, cause2 @ t=1.12]` | 0.938 | 0.964 | 0.973 | 0.990 | 1.003 | 0.948 | 0.993 | 0.526 | 0.993 |
| `cif_regimen[control, cause2 @ t=2.32]` | 0.940 | 1.044 | 1.044 | 1.000 | 1.003 | 0.951 | 1.003 | 0.276 | 0.997 |
| `ate_regimen[treated vs control, cause1 @ t=1.12]` | 0.935 | 0.959 | 0.930 | 1.030 | 0.954 | 0.937 | 0.983 | 0.311 | 0.831 |
| `ate_regimen[treated vs control, cause1 @ t=2.32]` | 0.930 | 0.959 | 0.944 | 1.016 | 0.970 | 0.941 | 0.985 | 0.133 | 0.948 |
| `ate_regimen[treated vs control, cause2 @ t=1.12]` | 0.925 | 0.932 | 0.927 | 1.006 | 0.980 | 0.937 | 0.986 | 0.000 | 0.830 |
| `ate_regimen[treated vs control, cause2 @ t=2.32]` | 0.920 | 0.939 | 0.930 | 1.010 | 0.977 | 0.939 | 0.987 | 0.000 | 0.915 |

Anchored, every row's bootstrap SE ratio is 0.972 to 1.003, inside the band (0.93, 1.07). What
remains is the coverage half. At 3,200 fits the 99% Clopper-Pearson interval of a coverage near
0.94 has half-width about 0.011, so a row whose true coverage is about 0.931 fails the band's
lower edge of 0.92 with probability about one half, and a row below it more likely fails than
passes. The in-sample fit's own Wald coverage sits at 0.937 to 0.951, close to that edge. Taking
each row's percentile coverage at the pooled difference of -0.003 from `v4_in_sample`'s, the
anchored family pass at 3,200 is 0.26, with 1.2 red rows expected; that family reaches 0.93 at
5,600 fits and 0.97 at 6,400. The row differences (-0.015 to +0.010, Monte Carlo SE 0.004 to
0.012) do not reject a common value (chi-squared 14.6 on 11 degrees of freedom, p = 0.20, ignoring
the correlation between rows), and they do not confirm one. Taking each row's own difference, the
family pass at 3,200 is 0.003 with 3.1 red rows expected, and only 0.07 at 5,600, 0.09 at 6,400
and 0.15 at 7,600, because `ate_regimen[treated vs control, cause1 @ t=2.32]` (0.941 - 0.015) and
`cif_regimen[control, cause1 @ t=1.12]` (0.937 - 0.0075) sit near 0.93 at any count. So 1.2 to
3.1 red rows are expected at 3,200.

Diagnosis: no defect. The low plug-in passes come from the chance spread of the pilot's samples,
which the Wald interval on the same fits shares, and from the in-sample fit's own coverage near
0.94, which the bootstrap inherits and does not cause. So the declaration gates the 12 rows on
the primary rule alone (coverage, SE ratio and bias), and reports the calibration rule and the
shrunk-SE control in `calibration.csv` with no verdict that stops the run. It keeps 3,200 fits,
the calibration rule's at-target count, so that the reported calibration reading keeps the
resolution it was sized for. A calibration row that misses the rule is a finite-sample shortfall
only when its SE-ratio interval contains 1 and its coverage is consistent with 0.95. Otherwise its
declared diagnosis, like that of a red primary row, starts from `paired-wald.csv`, which compares
the row with the Wald interval on the same fits, as `regenerate.py`'s red-cell readings state.
Doubling the count would cost about 580 CPU-h more and buy a calibration family pass of 0.97 under
the pooled difference but 0.09 under the rows' own differences. It could move no label either: the
in-sample bootstrap stays experimental until the conditions of the bootstrap delta method are
read.

The rows this diagnosis covers: `cif_regimen[treated, cause1 @ t=2.32]`,
`cif_regimen[treated, cause2 @ t=1.12]`, `cif_regimen[treated, cause2 @ t=2.32]`,
`cif_regimen[control, cause1 @ t=1.12]`, `cif_regimen[control, cause1 @ t=2.32]`,
`cif_regimen[control, cause2 @ t=1.12]`, `cif_regimen[control, cause2 @ t=2.32]`,
`ate_regimen[treated vs control, cause1 @ t=1.12]`, `ate_regimen[treated vs control, cause1 @ t=2.32]`,
`ate_regimen[treated vs control, cause2 @ t=1.12]` and
`ate_regimen[treated vs control, cause2 @ t=2.32]`.
