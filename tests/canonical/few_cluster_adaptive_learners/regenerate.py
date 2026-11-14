"""Regenerate the ``few-cluster-adaptive-learners`` evidence under the declared run form.

**State: not declared.** The design is fixed below, but the pilot that sizes it is the 400-draw
pilot of :mod:`tests.studies.few_cluster_adaptive_pilot` (design review B1), which has not run.
The predicted pass probabilities, the predicted red cells and every reading that names one are
derived from it in the redeclaration that follows it. Do not queue this run until that
redeclaration is committed.

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

Order of the work (plan X54 rev 3.2, section 5.4, and the S-h design review): the laws (L2 in
:mod:`tests.studies.clustered_nonlinear_law`, its truth by quadrature checked at two
resolutions and against a large draw); the 400-draw pilot on its own seed stream, committed as
``pilot.csv.gz``; a diagnosis of any departure it confirms (a stacked variance-ratio interval
that excludes 1, read against the paired GLM arms) before anything is declared; the
redeclaration; the declared run.

Budget. Measured CPU seconds per fit on L1, at 10, 20 and 40 clusters: in-sample default
library 3.8, 4.0 and 5.9; stacked 17.8, 20.6 and 23.3; cross-fitted ``LTMLE`` on L3 about 26;
the stacked control about 0.5; every GLM fit under 0.1. At 4,000 draws per cell, with L2 at the
L1 times: stacked 137 CPU-h, in sample 30, L3 29, the GLM and control fits 4, the 1,000-draw
primary 6, and the summary's bootstraps under 1, so about 207 CPU-h, under the plan's declared
270 CPU-h; about 13 hours on the local 16-core queue, and at most 17 at 270. The heavy keys run
first (``few_cluster_adaptive_properties.keys``). Runner rule (plan 5.4, design review m2): the
smoke and the pilot re-measure each per-fit time, and a time more than 1.5 times the figure
above (a stacked fit over 27 to 35 s by count, an ``LTMLE`` fit over 39 s) goes back to the row
thread before the run. A shared machine read 56 to 81 s per stacked fit at 10 clusters and 62 s
per ``LTMLE`` fit, which the 16-core queue's pilot settles.

What is gated, and why (plan 5.4, review findings B3 and N3, design review M1 and M2). The
gated fits are stacked ``TMLE`` and cross-fitted ``LTMLE``. Each positive row (``t_reference``
of ``tmle_stacked`` at each point law and count, and of ``ltmle_crossfit``, 7 rows) passes when
all of these hold:

* ``coverage floor`` (G1): the exact 99% lower end of all-draw coverage at least 0.90, a
  withheld fit a miss and a lost fit a miss in the worst case of the failure accounting;
* ``bias equivalent`` (G2): the 99% bias interval inside 0.25 empirical SD;
* ``variance ratio lower end``: the 99% percentile-bootstrap lower end of mean(SE^2) over the
  variance of the estimates, on shown fits, at least 0.8649. The variance ratio and not the SE
  ratio, because the SE ratio of a calibrated variance sits at the Jensen factor;
* ``shown coverage lower end``: the exact 99% lower end of shown-only coverage at least 0.92. It
  reads coverage directly, so a few outlier SEs cannot carry it as they can the variance ratio;
* ``paired non-inferiority``, on stacked L1 rows: the exact 99% lower end of the paired
  all-draw coverage difference to the ``glm_reference`` arm of the same draws above -0.02.

Each gated ``iid_t_control`` passes the two-clause rule of the registered few-cluster studies,
and ``stacked_control`` passes when the exact 99% upper end of its all-draw coverage is below
0.90. A configuration's caveat goes at a law and count only where its positive row, its IID
control and its must-fail control all pass (``caveat_removed``). The stacked control's outcome
is an unpruned random forest fitted on the full sample, standing in for every out-of-fold
prediction, so its residuals shrink but do not vanish, and the out-of-fold assembly, the
stacked targeting and the stacked variance all act on it (design review M1). If the pilot reads
its power below 0.90 at a count, the stacked caveat cannot be removed at that count (plan 5.4).
No control plants an overfit ``LTMLE`` outcome fit, so the cross-fitted ``LTMLE`` cell reads the
stacked control of L1 at 20 clusters, a point-``TMLE`` control; the evidence page states that
limit.

The in-sample default-library fit is reported, not gated (design review M2, option (b)). The
Super Learner cross-validates only its weights and predicts from each candidate's full-sample
fit (``cleverly/learners/super_learner.py``), so an in-sample fit's residuals shrink at any
cluster count and no Donsker argument covers it. It therefore takes the ``adaptive_nuisance``
shortfall on the ``t`` tier by theory (``cleverly._support.ADAPTIVE_IN_SAMPLE_REASON``), and at
40 or more clusters no clustered in-sample row covers it. Its cells, its IID control and the
interpolating control are published with every reading, among them the paired coverage
difference to the GLM arm on L1 and L2.

The ``glm_reference`` arm fits a main-terms logistic outcome and an estimated main-terms logistic
propensity. The committed parametric baseline uses the exact propensity. On L1 the main-terms
propensity is correctly specified, and an estimated correct propensity makes the influence-curve
variance conservative under a misspecified outcome, so the paired comparator is at least as
strict as the exact-propensity one and the arms differ only in the learner library.

Red-cell readings (declared before any run). The record is ``reporting``, so a red gated verdict
is published, and it takes the route of plan section 5.8 named here. No cell is re-sized,
re-drawn, dropped or relabelled after a verdict is seen, and no margin differs from the harness
defaults except the declared thresholds above.

* A stacked positive row passes G1 and G2 but fails the variance-ratio, shown-coverage or
  non-inferiority clause at a count below 40, or its control fails or lacks power there: the
  stacked configuration takes the ``adaptive_nuisance`` shortfall of its ``t``-tier row
  (``T11.stacked``) at that count and below
  (``cleverly._support_rows.ADAPTIVE_NUISANCE_SHORTFALLS``, experimental label, interval
  shown). The floor reading is unchanged, and F28 records the limit.
* A stacked positive row fails G1 or G2 at 10 or 20 clusters: the same shortfall, recorded with
  its coverage, withheld share and bias.
* A stacked positive row is red at 40 clusters (design review M3). It is first diagnosed as a
  defect suspect: the stacked targeting and variance code, read against the paired GLM arm and
  the stacked control on the same draws. A code defect is fixed and the study re-run under a
  fresh declaration. If the diagnosis finds no code defect, the red is the default library's
  limit at 400 rows: the clustered row ``T10.unweighted`` takes an entry with ``below=None``, so
  every clustered stacked fit with an adaptive nuisance learner is experimental at every cluster
  count, with the sentence of ``AdaptiveNuisanceShortfall.sentence`` ("measured with parametric
  nuisance learners only: ... fails at 40 clusters for stacked TMLE, and no study measures a
  larger count"). The qualifier is set on the ``clustered`` tier for this, and
  ``tests/unit/test_adaptive_nuisance_support.py`` holds its witness. The ``t``-tier row takes
  the shortfall at every count below 40 too, because a limit measured at 40 holds below it.
* Every clause of the stacked configuration passes at every count: no shortfall for it, and the
  docs sentences of plan 5.8 row 1 change for stacked fits; the in-sample sentence stays.
* A red row of the cross-fitted ``LTMLE`` cell takes the shortfall for cross-fitted ``LTMLE``
  at 20 to 39 clusters (``G07``).
* A must-fail control passes (covers): the study cannot see an overfit stacked outcome fit at
  that law and count, so the stacked caveat stays there. That is not a shortfall of the positive
  row.
* The primary (L1 at 20 clusters, stacked, 1,000 draws) reads the configuration of the
  ``tmle_stacked__equal10__j20`` cell. A red primary leg takes that cell's reading; it is not a
  separate defect.
* A replication that raises is never redrawn. A lost fit costs its key in its fit's cells only,
  under the declared bound of 0.05, and every declared rule reads it at the worst fill. A cell
  past its bound is red with the count stated.

Left to the redeclaration (design review B1 and m3): the predicted pass probabilities and the
at-target probabilities recomputed from the 400-draw pilot; the raise to 4,800 draws where the
recomputation calls for it; whether a declared interim at 1,000 draws may stop a stacked cell
that cannot reach its gates at 4,000 (stopping a cell at target with probability below 0.01);
and whether the reported in-sample cells keep 4,000 draws.
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
