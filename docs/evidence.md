# What each estimand's correctness rests on

`tests/unit/test_registry.py`'s `test_every_target_has_an_oracle` answers one question —
*does this target have a branch in an oracle law?* — and answers it yes or no. That gate is
what stops an estimand shipping "on the strength of its author's arithmetic alone", and it
is not what this page is for.

**This page is the other half: which instruments a target actually has, and which it does
not.** The distinction matters because the instruments go blind in different places, and
`docs/methodology.md`'s [*How to read a
refusal*](methodology.md#how-to-read-a-refusal) is not the only thing worth writing down —
so is how to read a *pass*. The trap that survives a green suite is:

> an exact-law instrument goes blind wherever a quantity vanishes at the truth, which is
> where a parity check is blind too. At correct nuisances `Q_r` and `g_{r,2}` are zero row
> by row, so every `test_influence_gateaux*` module passes against a flipped sign.

A row whose only evidence is a Gateaux comparison is therefore *not* the same claim as a
row that also carries a remainder rate and an exact identity, and a reader deciding how far
to trust a number should be able to see which they are looking at without opening five test
modules. That is the whole of what this table is.

**It is a gate, not a note.** `tests/unit/test_registry.py::TestEvidenceManifest` checks it
in both directions against `TARGETS`, checks that every module named here exists, and —
the part that makes it more than a document — checks that the *oracle law* column agrees
with the law whose `functional` really has the branch, read through the same `oracle_for`
the coverage gate uses. A row cannot claim an oracle the laws do not provide.

## The instruments

| kind | what it is | what it cannot see |
| --- | --- | --- |
| **oracle law** | the parameter written down longhand on an exactly representable discrete law, sharing no code with `src/` | nothing about a term that is zero at the truth |
| **Gateaux** | the reported influence curve against a complex-step derivative of that functional, to ~1e-14 absolute with `rtol=0` | a sign on any block that vanishes at correct nuisances; a counterfactual block, at `epsilon = 0` |
| **remainder** | the von Mises expansion's second-order term, measured as a rate under one wrong nuisance | a first-order error that cancels in the product |
| **exact identity** | an algebraic relation that holds by definition and so must hold bit-for-bit | anything symmetric in whatever the identity is symmetric in |
| **theorem** | a check against the source's own theorem, *at values where the quantity does not vanish* | — this is the anchor the others need |
| **bounded implementation witness** | a frozen comparison with an independently maintained implementation, scoped to a named finite-sample choice that the scientific oracles cannot exercise | the estimand's derivation; any behavior outside the deliberately matched nuisance, bound, and targeting settings |

## The table

One row per registered target. `—` in a cell means there is no such instrument for this
target, which is a statement rather than an omission; the **not covered** column says what
follows from it.

| target | oracle law | Gateaux | remainder | exact identity | not covered |
| --- | --- | --- | --- | --- | --- |
| `ate` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py`, `tests/unit/test_influence_gateaux_multi.py` | `tests/unit/test_remainder.py` | `IC_ate == IC_ey1 - IC_ey0`, and a null outcome model gives zero in every population (`tests/unit/test_invariants.py`) | no theorem anchor; the derivation is checked only where the oracle law can represent it |
| `att` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py`, `tests/unit/test_influence_gateaux_multi.py` | `tests/unit/test_remainder.py` | relabelling the arms gives `ATT' == -ATC` (`tests/unit/test_invariants.py`) | swapping the two conditioning populations outright — symmetric in the arms, so the symmetry test cannot see it; the closed-form comparison is what catches it |
| `atc` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py`, `tests/unit/test_influence_gateaux_multi.py` | `tests/unit/test_remainder.py` | relabelling the arms gives `ATC' == -ATT` (`tests/unit/test_invariants.py`) | as `att`, and for the same reason |
| `ey` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux_multi.py` | `tests/unit/test_remainder.py` | — | no identity of its own; it is the per-arm level the contrasts are built from, so its errors surface in them |
| `ey1` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | `tests/unit/test_remainder.py` | `IC_ate == IC_ey1 - IC_ey0` (`tests/unit/test_influence_gateaux.py`) | binary-only by declaration; the multi-arm path reports `ey` instead |
| `ey0` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | `tests/unit/test_remainder.py` | `IC_ate == IC_ey1 - IC_ey0` (`tests/unit/test_influence_gateaux.py`) | binary-only by declaration, as `ey1`; and the identity it shares with `ey1` is symmetric in the two arms, so a defect that swaps them survives it |
| `ey_obs` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | — | its empirical influence curve is `w(Y - E_w[Y])`; it is zero-mean without a targeting equation | missing outcomes are refused until the MAR natural-course score is derived; no separate remainder exists because the complete-data estimator is empirical |
| `par` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | — | `IC_par == IC_ey_obs - IC_ey0` on the binary oracle | missing outcomes and controlled intermediates are refused; the exact-law identity is blind to a defect shared by both component curves |
| `paf` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | — | its curve is the delta-method transform of `ey_obs` and the reference-arm mean | defined only for a binary outcome with positive observed risk; small-sample coverage near zero risk is not established |
| `rr` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | `tests/unit/test_remainder.py` | the ratio's curve is the delta-method transform of the levels' (`tests/unit/test_influence_gateaux.py`) | the log-scale interval's small-sample coverage is a simulation claim, not an identity |
| `or` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux.py` | `tests/unit/test_remainder.py` | the odds ratio's curve is the delta-method transform of the levels' (`tests/unit/test_influence_gateaux.py`) | as with `rr`, the log-scale interval's small-sample coverage is a simulation claim rather than an identity; and nothing here pins the odds ratio apart from the risk ratio at values where the two are close |
| `ey_regime` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux_regime.py` | `tests/unit/test_remainder_regime.py` | a degenerate regime equals the static arm it puts all its mass on (`tests/unit/test_regimes.py`) | a rule that is not deterministic; the density is evaluated once at fit time and the fit answers for that evaluation |
| `ate_regime` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux_regime.py` | `tests/unit/test_remainder_regime.py` | the contrast is the difference of the means (`tests/unit/test_regimes.py`) | a rule that is not deterministic, as for `ey_regime`; and the contrast's identity is a relation between two reported numbers, so a defect common to both means survives it |
| `ey_ipsi` | `tests/discrete_law.py`, `tests/discrete_law_mar.py` | `tests/unit/test_influence_gateaux_ipsi.py`, `tests/unit/test_influence_gateaux_ipsi_mar.py` | `tests/unit/test_remainder_ipsi.py`, `tests/unit/test_remainder_ipsi_mar.py` | `psi(delta=1)` equals `mean(Y)` row by row whatever the nuisances are (`tests/unit/test_ipsi_fit.py`) — the canary for an alternation exiting with one equation open | this is the one estimand that is **not** doubly robust: every remainder term carries `(ghat - g0)`, so a consistent `Qbar` cannot substitute for a consistent mechanism. The alternation's linear rate is measured, not bounded |
| `ate_ipsi` | `tests/discrete_law.py`, `tests/discrete_law_mar.py` | `tests/unit/test_influence_gateaux_ipsi.py`, `tests/unit/test_influence_gateaux_ipsi_mar.py` | `tests/unit/test_remainder_ipsi.py`, `tests/unit/test_remainder_ipsi_mar.py` | `psi(delta=1)` equals `mean(Y)`, and the contrast of two tilts at `delta=1` is zero (`tests/unit/test_ipsi_fit.py`) | not doubly robust, as for `ey_ipsi`; and with `delta=` present the `psi(1)` canary changes meaning -- it is then the MAR-identified `E[Y]` and the complete-case mean is the wrong answer, so the canary must not be read across that case |
| `ey_shift` | `tests/discrete_law_shift.py` | `tests/unit/test_influence_gateaux_shift.py`, `tests/unit/test_influence_gateaux_shift_cde.py` | `tests/unit/test_remainder_shift_cde.py` | the negative control: a shift's mean equals the induced stochastic regime's, and its **curve does not** (`tests/unit/test_influence_gateaux_shift.py`) | there is no plain-shift remainder module — the rate is measured only in the presence of a third nuisance (`_shift_cde`). And a Gateaux check on an exact law cannot see a counterfactual block, which is why `tests/unit/test_shift_submodel.py` and `tests/unit/test_shift_fit.py` pin those structurally and at `epsilon != 0` instead |
| `ate_shift` | `tests/discrete_law_shift.py` | `tests/unit/test_influence_gateaux_shift.py`, `tests/unit/test_influence_gateaux_shift_cde.py` | `tests/unit/test_remainder_shift_cde.py` | the same negative control as `ey_shift`, taken on the contrast (`tests/unit/test_influence_gateaux_shift.py`) | no plain-shift remainder module, as for `ey_shift`; and the MNAR tilt is refused on this axis by name, so nothing here measures sensitivity to it |
| `msm` | `tests/discrete_law.py` | `tests/unit/test_influence_gateaux_msm.py` | `tests/unit/test_remainder_msm.py` | a **saturated** working model reproduces the per-arm report exactly, at the covariate and at the estimate (`tests/unit/test_msm_submodel.py`, `tests/e2e/test_msm.py`); continuous-dose quadrature and its nonzero density-ratio score are pinned in `tests/unit/test_msm.py` and `tests/unit/test_continuous_msm.py` | the saturated identity is blind to every link-specific mistake — the curvature term, the alternation's restart, the loss of exact double robustness — because a saturated model *fits*, which is what a projection does not promise. The continuous test uses a linear truth; a nonlinear continuous-dose Gateaux oracle remains absent |

## Longitudinal estimands outside the target registry

`LTMLE` parameters are indexed by regimen, horizon, and sometimes cause rather than by a
`Target`, so they do not belong in the registry-gated table above. They have their own
bidirectional oracle gates in the named Gateaux modules; this table makes the parallel
evidence structure explicit.

| longitudinal variant | parameter and EIF oracle | nonzero or mutation witness | canonical implementation witness | not covered |
| --- | --- | --- | --- | --- |
| end-of-study, static and dynamic regimens | `tests/discrete_law_longitudinal.py`, `tests/unit/test_influence_gateaux_longitudinal.py` | dynamic-rule arm evaluation and dropped-censoring mutations in the Gateaux module; loss/design decomposition in `tests/unit/test_longitudinal_msm_submodel.py` | R `ltmle` 1.3-0, fixed g and intercept-only Q, active cumulative bound, nonzero epsilon, and a censoring-active mask witness (`tests/unit/test_ltmle_canonical_r.py`) | the R witness covers one static regimen, no cross-fitting or observation weights, and `variance.method="ic"`; cleverly's IC-only standard errors do not implement R's default robust truncation-aware variance |
| absorbing survival curve | `tests/discrete_law_survival.py`, `tests/unit/test_influence_gateaux_survival.py` | the `t-1` risk-set mutation and end-of-study reduction in `tests/e2e/test_ltmle.py` | the same R fixture at horizon two, with cumulative event nodes and explicit post-event missingness | the R witness does not cover every horizon jointly or competing events |
| competing-risks cumulative incidence | `tests/discrete_law_competing.py`, `tests/unit/test_influence_gateaux_competing.py` | mutation from all-cause to cause-specific survival in the Gateaux module; one-cause reduction in `tests/e2e/test_ltmle.py` | — | no canonical R comparison: the fixture would not add evidence beyond the exact law unless a distinct finite-sample blind spot is first named |
| working model over regimen/horizon cells | `tests/discrete_law_longitudinal.py`, `tests/unit/test_influence_gateaux_longitudinal_msm.py` | non-saturated, nonuniform projection law plus exact pooled-design/loss-weight checks in `tests/unit/test_longitudinal_msm_submodel.py` | — | R `ltmleMSM` uses a quasibinomial working-model projection; cleverly declares an outcome-scale weighted least-squares projection, so raw coefficient parity would compare different estimands |

## What this table says is missing

Read down the **not covered** column and one thing recurs: the **theorem** column is empty
for every row. There is one such instrument in the tree — `tests/unit/test_theorem_drtmle.py`,
which checks `DRTMLE` against Benkeser et al.'s Theorem 1 at values where the correction
does not vanish — and it exists because that variant's corrections are zero row by row at
correct nuisances, so the exact-law instruments were blind exactly where it mattered.

The arm, regime, shift, tilt and MSM axes are not in that position: their influence curves
do **not** vanish at the truth, so the Gateaux comparison is anchored where the quantity
lives rather than where it disappears. That is the argument for the empty column, and it is
an argument rather than a measurement — which is why it is written here, where the next
person to add an estimand will read it, instead of being left to be re-derived.

**The condition that would fill the column** is a target whose curve contains a block that
is zero at the truth. Registering one means the row above is no longer available, and the
new row needs a check against its source's own theorem, at a value where its block does not
vanish, before the estimand is reported.
