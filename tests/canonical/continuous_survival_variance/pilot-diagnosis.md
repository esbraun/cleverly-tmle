# Pilot diagnosis: continuous-survival-variance

Written by the X6 row thread before the declaration, from the committed pilot
(`tests/studies/continuous_survival_variance_pilot.csv.gz` at f30799f2: V1, V3 and V4 at 2,000
draws, V2 and V7 at 1,000, all fitted at the stable conditional risk of bc630ef4). The pilot's
plug-in check (`continuous_survival_variance_pilot.plug_in_checks`) reads each first-quartile row's
calibration pass probability at the declared 6,000 replicates, taking the pilot's SE ratio of the
shipped (switched) interval as true. Six rows of gated cells stay below 0.90 after the extension.
First-quartile rows are no gates (regenerate.py, red-cell readings), so these rows decide no
merge; this file records why each one falls short and that none points at a defect.

## The confirmed departures

SE ratio is the mean standard error over the empirical SD of the estimate across draws; "S > IC"
is the share of draws where the switched variance raised the influence-curve variance.

| row | draws | bias / SD | S SE ratio | IC SE ratio | median `variance_rel_sd` | S > IC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1_rare_cause2/cif_regimen[treated, cause2 @ t=0.52] | 2,000 | +0.034 | 0.960 | 0.867 | 0.458 | 0.68 |
| v3_half_n/cif_regimen[treated, cause2 @ t=0.46] | 1,995 | +0.025 | 0.959 | 0.882 | 0.465 | 0.64 |
| v4_in_sample/ate_regimen[treated vs control, cause1 @ t=0.46] | 2,000 | +0.025 | 0.933 | 0.897 | 0.168 | 0.61 |
| v4_in_sample/cif_regimen[control, cause1 @ t=0.46] | 2,000 | -0.013 | 0.935 | 0.885 | 0.216 | 0.68 |
| v4_in_sample/cif_regimen[control, cause2 @ t=0.46] | 2,000 | +0.024 | 1.034 | 0.995 | 0.175 | 0.73 |
| v4_in_sample/cif_regimen[treated, cause2 @ t=0.46] | 2,000 | +0.052 | 0.953 | 0.864 | 0.360 | 0.74 |

## Reading

- **Sparse first-quartile risks (V1 and V3, treated cause 2).** These are the sparsest cells of
  the study: the rare second cause in the treated arm by the first quartile, at 1,000 units with a
  halved cause-2 hazard (V1) or at 500 units (V3). The median relative SD of the variance estimate
  is 0.46, far above the 0.125 switch point, so the influence-curve SE runs 12 to 13% short (IC
  ratio 0.87 and 0.88). The switched variance closes most of that gap (0.96) but not all of it.
  This is the early-time shortfall the plan diagnosed before PR A (`rows/X6/plan.md`, "the
  early-time SE finding"; the earlier study read treated cause 2 at 0.46 with IC coverage 0.898
  and S 0.927), and it is the open X6 (a2) item: a corrected first-quartile interval. The bias is
  under 0.04 SD, so the point estimate is not at fault, and every such estimate reads
  experimental under the share floor of 0.40 and the `variance_rel_sd` floor.
- **The in-sample cell (V4).** V4 fits the nuisance models on the same rows it targets
  (`cross_fit=False`). Its influence-curve SE runs short at every sparse first-quartile row
  (0.86 to 0.90), which is the known optimism of in-sample nuisance fits where few events inform
  the hazard: the empirical variance of the estimated influence curve is taken on the rows the
  learner already fitted. S again recovers most of it (0.93 to 0.95). V4's median and
  upper-quartile rows, which are its gates, read S SE ratios of 0.97 to 1.02 on the pilot. Bias is
  at most 0.05 SD.
- **V4 control cause 2 (over-width).** The one departure above 1: S reads 1.034 where the IC
  reads 0.995. This is S's designed over-width in a row where the curve is not sparse enough to
  need it (73% of draws switched at a median relative SD of 0.175). Its plug-in pass is 0.900,
  at the edge of the check. If the declared run reads this row red with its SE-ratio interval
  above 1 while the IC's paired row is green, it goes to Eric as a red-cell decision
  (regenerate.py), not an owner row.

No row shows a defect: every bias is small, the IC's shortfall in each row matches the sparsity
the plan documented, and S moves every under-width row toward 1. The declared run publishes these
rows under the calibration rule, and their estimates stay experimental through the share floor
until X6 (a2) supplies a corrected first-quartile interval.

## Failures in the gated cells

V3 lost 5 of 2,000 pilot fits (V1, V4 and V7 lost none). Refitting draws 1182 and 1521 gives the
two kinds of refusal the code keeps on purpose at 500 units:

- `DataError` (draws 1182, 1587, 1870): an initial hazard fit puts a combined jump of one or more
  at a late time with few units at risk, so the product-limit survival is not a probability.
- `CapabilityError` (draws 1521, 1776): the treated arm has no cause-2 event by the first target
  time, so that risk's influence curve is zero and its interval undefined.

Neither is the removed survival-floor refusal. The pilot's predicted chance that V3 exceeds its
failure bound of 60 at 6,000 replicates is 0.0001 (`failure_exceedance`), so the bound is not at
risk.
