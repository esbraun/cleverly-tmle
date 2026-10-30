# What the nuisances must satisfy

The union condition, `Q̄ = Q̄_0` **or** `g = g_0`, is assumed and not checked. It is not
enough. The release claim rests on the rate conditions of
[the remainder section](theorem.md#the-remainder-terms-and-the-rate-conditions) and on
[the conditioning terms](theorem.md#the-conditioning-terms-are-linear-in-one-error). With one nuisance wrong, those terms vanish only in the cases
that section lists. The rate conditions fall on **five** estimated functions rather than two
for the univariate construction:

1. the primary outcome regression `Q̄_n`;
2. the primary propensity `g_n`;
3. the reduced outcome regression `Q_r`;
4. the reduced propensity `g_{r,1}`;
5. the reduced propensity `g_{r,2}`.

The bivariate construction instead has four: the same two primary nuisances, `Q_r`, and its
single two-column reduced probability `g_r`.

The reduced regressions are the part it is easy to forget. They estimate conditional means
given the *estimated* nuisance. A univariate rate for them does not control the conditioning
term, which conditions on the true nuisance as well.

Their consistency is **estimated,
unmeasured**: a saturated learner recovers them exactly on the exact law, which is consistency
at one learner on one law and not a rate. An archived study over 6,000 fits measured
`0.844`/`0.848` coverage against a plain TMLE's `0.532`/`0.472` in the cell built for it. It was
**nominal nowhere**, the best reading being `0.880`, with the three reductions fitted by `glm`.
A committed probe tested the conditioning term as its cause, under a rule fixed before it ran.
The shortfall is not attributed. The conditioning term at the mean fit does not account for it
([the validation programme](validation-programme.md)).

Practical consequences:

- **A flexible learner for the consistent nuisance does not buy the one-wrong interval.** Its
  bias must keep the level sets of the truth, or be negligible in the direction
  [the theorem page](theorem.md#the-conditioning-terms-are-linear-in-one-error) names. No fit can check either. Declare a known mechanism where the
  design gives one.
- **Choose the reduction learners deliberately.** Each reduced regression has its own keyword.
  The table below gives the routing. A `None` slot uses the primary specification of its side.

  | regression | target, on design | task | keyword | `None` resolves to |
  | --- | --- | --- | --- | --- |
  | `Q_r` | `Y − Q̄_n` on `g_n` | regression | `reduced_outcome_learner=` | `outcome_learner=` |
  | `g_{r,1}`, or bivariate `g_r` | `1_a` on `Q̄_n` | classification | `reduced_treatment_learner=` | `treatment_learner=` |
  | univariate `g_{r,2}` | `(1_a − g_n)/g_n` on `Q̄_n` | regression | `reduced_mechanism_residual_learner=` | `outcome_learner=` |

  The two regression slots resolve independently. So `reduced_outcome_learner=` alone leaves
  `g_{r,2}` on the default. R `drtmle` routes the same way: its `gr` learner (`glm_gr`,
  `SL_gr`) fits both reduced propensities, and its `Qr` learner fits only `Q_r`
  (`tests/canonical/drtmle/run_drtmle.R`). `tests/unit/test_drtmle_reduced_routing.py` pins
  the routing.

  Theory asks for a flexible `Q_r` learner when only the mechanism is consistent. The same
  learner on `g_{r,2}` fits noise when `Q̄_n` is near constant. A
  [fold-wise study cell](../method-evidence/dr-tmle-foldwise.md#the-spline-cell-and-the-red-cell-it-replaced)
  that fitted one spline on both was red, and its paired diagnosis put the bias in `g_{r,2}`.
  Its replacement fits the spline on `Q_r` only and passes.

  `Q_r` and `g_{r,2}` have signed targets, so a classifier cannot fit them. `DRTMLE` refuses a
  classifier in either regression slot before it fits a learner. A `None` slot takes
  `outcome_learner=`, which is often a classifier on a binary outcome. Pass a regressor to each
  regression slot in that case. On the randomized missing-outcome route,
  `reduced_mechanism_residual_learner=` fits `r_a` and `r_m`, and `reduced_outcome_learner=`
  fits `e`.
- **A flexible primary nuisance does not buy a flexible reduction.** The reductions are on one
  scalar by default and two for bivariate `g_r`, so their bias is a low-dimensional smoothing
  question independent of how well the primary fit adjusted for `W`.
- **A random-forest reduction is outside the entropy condition** of
  [reduced-regression cross-fitting](targeting.md#reduced-regression-cross-fitting), and is not warned about at runtime. The automatic
  Super Learner includes a forest, so choose explicit reduced learners when claiming that condition.
