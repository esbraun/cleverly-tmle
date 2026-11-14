"""Regenerate the ``few-cluster-adaptive-learners`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
Run it from the repository root as a module,
``python -m tests.canonical.few_cluster_adaptive_learners.regenerate``, so that ``tests``
imports, with every BLAS and OpenMP thread variable at 1 (``README.md``). The record has no
comparator, so the run starts no container. Laws, learners, fits, arms, cells, margins, seeds
and replicate counts are those of :mod:`tests.studies.few_cluster_adaptive_learners` and
:mod:`tests.studies.few_cluster_adaptive_properties`, fixed before any run and recomputed by
``tests/unit/test_few_cluster_adaptive_design.py``. The record is not in ``registered()`` until
its run is committed; until then its property module is an uncommitted subject of the property
differential, on synthetic rows.

Order of the work (plan X54 rev 3.2, section 5.4): the laws (L2 in
:mod:`tests.studies.clustered_nonlinear_law`, its truth by quadrature checked at two resolutions
and against a large draw); the pilot of :mod:`tests.studies.few_cluster_adaptive_pilot` on its
own seed stream, committed as ``pilot.csv.gz``; this declaration; the declared run.

Budget. Measured CPU seconds per fit on L1, at 10, 20 and 40 clusters: in-sample default
library 3.8, 4.0 and 5.9; stacked 17.8, 20.6 and 23.3; cross-fitted ``LTMLE`` on L3 about 26;
every GLM and control fit under 0.1. At 4,000 draws per cell, with L2 at the L1 times:
stacked 137 CPU-h, in sample 30, L3 29, the GLM and control fits 2, the 1,000-draw primary 6,
and the summary's bootstraps under 1, so about 205 CPU-h. The pilot's wall times, taken on a
shared machine, read stacked L2 fits up to 1.5 times slower than L1; at that ratio the total is
about 240 CPU-h, still under the plan's declared 270 CPU-h. On the local 16-core queue that is
13 to 15 hours of wall time, and at most 17 at the declared 270. The heavy keys run first (``few_cluster_adaptive_properties.keys``), so the
tail of the run is short fits. The pilot spent 1.33 CPU-h (4,782 CPU seconds on 3 workers),
inside the 1.5 CPU-h cap.

What is gated, and why (plan 5.4, review findings B3 and N3). Each positive row
(``t_reference`` of ``tmle_in_sample``, ``tmle_stacked`` and ``ltmle_crossfit`` at each law and
count, 13 rows) passes when all of these hold:

* ``coverage floor`` (G1): the exact 99% lower end of all-draw coverage at least 0.90, a
  withheld fit a miss and a lost fit a miss in the worst case of the failure accounting;
* ``bias equivalent`` (G2): the 99% bias interval inside 0.25 empirical SD;
* ``variance ratio lower end``: the 99% percentile-bootstrap lower end of mean(SE^2) over the
  variance of the estimates, on shown fits, at least 0.8649. The variance ratio and not the SE
  ratio, because the SE ratio of a calibrated variance sits at the Jensen factor;
* ``shown coverage lower end``: the exact 99% lower end of shown-only coverage at least 0.92. It
  reads coverage directly, so a few outlier SEs cannot carry it as they can the variance ratio;
* ``paired non-inferiority``, on L1 only: the exact 99% lower end of the paired all-draw
  coverage difference to the ``glm_reference`` arm of the same draws above -0.02.

Each ``iid_t_control`` passes the two-clause rule of the registered few-cluster studies, and each
must-fail control (``interpolating_control``, ``stacked_control``) passes when the exact 99% upper
end of its all-draw coverage is below 0.90. A configuration's caveat goes at a law and count only
where its positive row, its IID control and its must-fail control all pass
(``caveat_removed``); the cross-fitted ``LTMLE`` cell reads the stacked control of L1 at 20
clusters, because no control plants an overfit ``LTMLE`` outcome fit.

The ``glm_reference`` arm fits a main-terms logistic outcome and an estimated main-terms logistic
propensity. The committed parametric baseline uses the exact propensity; this arm estimates it,
so that the paired arms differ only in the learner library.

Pilot readings (``pilot.csv.gz``, 60 in-sample and 20 stacked draws per point cell, 20 on L3,
no fit lost, no interval withheld). All-draw coverage of the positive rows, in sample on L1 at
10, 20 and 40 clusters: 0.933, 0.900 and 0.917; on L2: 0.900, 0.950 and 0.850. Stacked on L1:
0.850, 0.850 and 0.950; on L2: 0.800, 0.950 and 0.950. Cross-fitted ``LTMLE``: 1.000. The
variance ratio reads 0.52 to 1.36 across the positive rows, standardized bias at most 0.23, and
the Super Learner put all its weight on one candidate on 0.00 to 0.43 of draws. The GLM arms
of the point laws cover 0.85 to 0.95. Both must-fail controls cover 0.42 to 0.60 at every law and count, and give
the same estimate on every draw: the interpolating tree leaves no outcome residual, so the
targeting step and the curve do not read the propensity. The IID control's paired SE ratio reads
0.66 to 0.75 against its ceiling of 0.80.

Pass probabilities (``few_cluster_adaptive_learners.PREDICTED_PASS``, recomputed by the design
test; predictive, each pilot rate drawn from its Jeffreys posterior and each pilot ratio from its
normal approximation). Every must-fail control fails with probability 1.000, so every control
has power at every count. The IID controls pass with probability 0.67 to 1.00. The positive rows
read 0.000 to 0.494; nine are below 0.10 (``PREDICTED_REDS``). At this pilot size the figures are
set by the pilot's own uncertainty: raising a cell from 4,000 to 4,800 draws moves no positive
row's probability by more than 0.013. The plan's raise rule (a cell to 4,800 when a clause reads
below 0.90) is therefore not applied: raising every cell would cost about 40 CPU-h and change no
prediction. At a
calibrated variance the plan's sizing gives the variance-ratio clause 0.99 (SE relative variance
0.5) and 0.92 (2.0), the non-inferiority clause 0.995 and 0.923 at 6% and 10% discordant draws,
and the shown-coverage clause 1.000 at a true coverage of 0.95.

Red-cell readings (declared before any run). The record is ``reporting``, so a red gated verdict
is published, and it takes the route of plan section 5.8 named here. No cell is re-sized,
re-drawn, dropped or relabelled after a verdict is seen, and no margin differs from the harness
defaults except the declared thresholds above.

* A positive row passes G1 and G2 but fails the variance-ratio, shown-coverage or
  non-inferiority clause at a count, or its control fails: the configuration takes the
  ``adaptive_nuisance`` shortfall of its t-tier support row at that count and below
  (``cleverly._support_rows.ADAPTIVE_NUISANCE_SHORTFALLS``, experimental label, interval shown).
  The floor reading is unchanged, and F28 records the limit.
* A positive row fails G1 or G2 at 10 or 20 clusters: the same shortfall, recorded with its
  coverage, withheld share and bias.
* Stacked passes and in sample fails at a count, or the reverse: the failing configuration
  alone takes the shortfall below its smallest passing count.
* Every clause of a configuration passes at every count: no shortfall for it, and the docs
  sentences of plan 5.8 row 1 change for it.
* The in-sample rows at 40 clusters are predicted red (0.023 on L1, 0.000 on L2). At 40 clusters
  the reference is normal and the qualifier does not apply, so a red row there is not a
  few-cluster limit. It is read as the in-sample default library's overfit, the failure that
  cross-fitting exists to remove, when the stacked row at 40 passes on the same law. Otherwise
  it is a defect suspect for the library or the law, diagnosed before any reading. The stacked
  row of L2 at 40 is predicted red at 0.065, and takes the same diagnosis if red.
* A red row of the cross-fitted ``LTMLE`` cell takes the shortfall for cross-fitted ``LTMLE``
  at 20 to 39 clusters.
* A must-fail control passes (covers): the study cannot see an overfit outcome fit at that law
  and count, so the caveat of the configuration that reads it stays there. That is not a
  shortfall of the positive row.
* The primary (L1 at 20 clusters, stacked, 1,000 draws) reads the configuration of the
  ``tmle_stacked__equal10__j20`` cell, predicted red at the pilot (coverage 0.85 and variance
  ratio 0.52 on 20 draws). A red primary leg takes that cell's reading; it is not a separate
  defect.
* A replication that raises is never redrawn. A lost fit costs its key in its fit's cells only,
  under the declared bound of 0.05, and every declared rule reads it at the worst fill. A cell
  past its bound is red with the count stated.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import few_cluster_adaptive_learners, few_cluster_adaptive_properties

if __name__ == "__main__":
    run(
        few_cluster_adaptive_learners,
        few_cluster_adaptive_properties,
        here=Path(__file__).resolve().parent,
    )
