# What the nuisances must satisfy

The union condition, `Q̄ = Q̄_0` **or** `g = g_0`, is assumed and not checked. Beyond it, the
release claim rests on the rate conditions of
[the remainder section](theorem.md#the-remainder-terms-and-the-rate-conditions), which fall on **five**
estimated functions rather than two for the univariate construction:

1. the primary outcome regression `Q̄_n`;
2. the primary propensity `g_n`;
3. the reduced outcome regression `Q_r`;
4. the reduced propensity `g_{r,1}`;
5. the reduced propensity `g_{r,2}`.

The bivariate construction instead has four: the same two primary nuisances, `Q_r`, and its
single two-column reduced probability `g_r`.

The reduced regressions are the part it is easy to forget, and they are where the guarantee is
bought. Their consistency is **estimated, unmeasured**: a saturated learner recovers them exactly
on the exact law, which is consistency at one learner on one law and not a rate. A study over
6,000 fits found the interval demonstrably better than a plain TMLE's where one nuisance is badly
estimated: `0.844`/`0.848` against `0.532`/`0.472` in the cell built for it. It was **nominal
nowhere**, the best reading being `0.880`, with the three reductions fitted by `glm`. That is a
measurement of a configuration, not of the theorem's condition.

Practical consequences:

- **Choose the reduction learners deliberately.** They default to the primary specification.
  `reduced_outcome_learner=` and `reduced_treatment_learner=` are two keywords rather than one
  because the tasks differ: `g_{r,1}` (or bivariate `g_r`) is a conditional probability and
  `Q_r` plus univariate `g_{r,2}` are conditional means of signed quantities. A learner
  *instance* built for classification cannot serve `Q_r`, whose target is an outcome residual.
- **A flexible primary nuisance does not buy a flexible reduction.** The reductions are on one
  scalar by default and two for bivariate `g_r`, so their bias is a low-dimensional smoothing
  question independent of how well the primary fit adjusted for `W`.
- **A random-forest reduction is outside the entropy condition** of
  [section 3](targeting.md#reduced-regression-cross-fitting), and is not warned about at runtime. The automatic
  Super Learner includes a forest, so choose explicit reduced learners when claiming that condition.
