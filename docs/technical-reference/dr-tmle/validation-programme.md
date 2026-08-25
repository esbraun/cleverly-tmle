# What the validation programme established

The active registered [canonical DR-TMLE study](../method-evidence/canonical-dr-tmle.md)
now adds a theory-first comparison with pinned R `drtmle` on the paper's binary complete-data law.
Most paired cells establish bounded equivalence, but none establishes the prespecified
coverage-superiority route; one paired cell is inconclusive. The both-correct calibration and
root-n cells pass, while both one-correct robustness cells fail their finite-sample bias rule.
Those red results are committed evidence, not exceptions to the theorem: the theorem remains
conditional on rate and remainder premises that a fitted dataset cannot certify.

The older drift-law programme below asks a different, harder question. It is retained as
historical evidence rather than silently generalized to the registered paper law.

This page's claims rest on a closed programme of six pieces: a theoretical audit against the
sources, a targeting-and-exit study, a controlled coverage demonstration, a reference study for
the reduced regressions, a construction ablation, and a terminal experiment. The
[evidence index](../evidence.md) records the acceptance instruments. The
programme itself, including its study harnesses, replicate records, differential diagnostics,
dispatch workflows, and working notes, is archived at the `drtmle-validation-archive-2026-08` tag
rather than on `main`. In summary:

**Established.** The implementation is faithful to Theorem 1 in five respects.

- The corrected curve is the Gateaux derivative of the parameter.
- The sign of the mechanism correction is the appendices' orientation.
- The reported variance is Theorem 1's.
- The three score equations are solved at the state returned.
- The interval beats a plain TMLE's by a material margin where one nuisance is badly estimated.

**Not established in that archived drift-law programme, and recorded as such.** Three things.
Nominal coverage anywhere in that study,
the best reading being `0.880`. A localized cause for that shortfall: a six-contrast construction
ablation over 2,496 fits returned a **null** on its primary column, and a terminal experiment over
both a selection and an independent audit cohort nominated **nothing**. And any `src/` change
justified against the theorem.

Two measured quantities account for the shortfall, and they are one premise measured twice. The
second-order remainder that Theorem 1 assumes negligible does not vanish at these sizes. The
reported `se` runs about 10% short of the spread it covers in one drift cell, and about 16%
*long* in the other. The second is therefore not a separate defect in the variance estimator. `σ̂²_n` is
Theorem 1's own quantity, valid to first order exactly when the condition the first quantity fails
holds.

That is why the release claim is conditional validity, and why the conditions are stated on this
page rather than assumed away.
