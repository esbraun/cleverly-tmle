"""Regenerate the ``few-cluster-floor-cross-fitted-ltmle`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container. Law, fit, ladder, arms, margins,
seeds and replicate counts are those of :mod:`tests.studies.few_cluster_floor_ltmle`,
:mod:`tests.studies.few_cluster_floor_ltmle_properties` and :mod:`tests.studies.few_cluster_floor`,
fixed before any run and recomputed by ``tests/unit/test_few_cluster_floor_design.py``. The
record is not in ``registered()`` until its run is committed. Until then its property module is
a synthetic subject of the property differential, on the subset it declares
(``DIFFERENTIAL_BUDGET`` and ``differential_cells()``).

**State: redeclared after the pilot (committed here), awaiting the design-review confirmation
and the run.** The pilot (:mod:`tests.studies.few_cluster_floor_pilot`, 400 draws per cell on
the ``"pilot"`` stream) wrote the tables listed in ``few-cluster-floor-tmle`` here; the
declaration copies its readings into the study module, and the design test recomputes them from
``pilot-summary.csv``. It changed no rule. R is 4,000 (the family-wise rule M5 is met there:
plug-in 1.000, predictive 0.814, expected floor 19). The two predicted reds,
``unequal40`` J7 and J8 (pilot coverage 0.900 and 0.9125), read ``below floor
(finite-sample)``. (C) has no predicted shortfall (discordance 1.5% over 26 cells, clause (i)
probability 1.000). No key was lost. ``pilot-timing.csv`` projects 13.3 CPU-h at R = 4,000
(14.9 at 6,000), under the declared 22.

**What is gated**: the registered two-clause rule of ``clustered-few-cluster-tmle``, as in
``few-cluster-floor-tmle``: G1 and G2 on each ``t_reference`` row, both control clauses on each
``iid_t_control`` row, with ``shown_only`` (the SE ratios read the shown draws; a cell with fewer
than two shown draws fails its control). Every other arm is reported, and so are the package-GLM columns of each
positive row (the raised share by count with its exact interval, and the study learner's shown
coverage split by whether the package GLM raised, M2 and Q15).

**Floor** (plan 5.8). A count passes when both size laws pass G1, G2 and the control. The floor
is the smallest ladder count from which every ladder count passes; a failure at or above 20
keeps the current floor of 20 and is diagnosed as a defect suspect. A passing floor certifies an
all-draw coverage whose exact 99% lower end is at least 0.90, with the study quasibinomial GLM
at ``numerical_separation=True`` (it keeps the separated last iterate, also for a target separated
once rounded, and a fit that still fails is a lost key); the package GLM's raised share is
reported beside it.

**Part (c)** (plan 5.7): (B) on every draw, (C) on the first 2,000 draws at 10 clusters or more
(below 10 every one of the :math:`\\min(5, J)` folds holds one cluster, R5 refuses, and (C) reads
as withheld on every draw). The rule and its consequences are those of
``few-cluster-floor-tmle``; (B) becomes the default reference only if it wins here and in every
family compared there.

**Budget**, measured on this machine from 3 serial draws of every cell with one thread: the
property phase at R = 4,000 is 14.2 CPU-h, of which the jackknife arms (10 to 39 clusters, 2,000
draws) are 9.9 and the package-GLM refits about 1; about 16.5 at R = 6,000. The pilot is about
2.4 CPU-h and the 1,000-draw primary under 0.1. A two-draw smoke of every key, timed per key,
projects 13.1 CPU-h at the declared R (a three-draw smoke 15.2 at R = 6,000). The pilot's
``pilot-timing.csv`` projects it again from 400 draws, at the sized R and at 6,000. Declared 22 CPU-h, the plan's figure.

**Failures**: as ``few-cluster-floor-tmle``. A 24-draw smoke of every cell lost no key.

**Red-cell readings**: as ``few-cluster-floor-tmle``, owner ``X54-cluster-floor``. ``unequal40`` at 6 to 10 clusters may read
all-draw coverage near 0.92, where G1 passes or fails by chance (plan 2.2); the floor reports
the count honestly either way.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import few_cluster_floor_ltmle, few_cluster_floor_ltmle_properties

if __name__ == "__main__":
    run(
        few_cluster_floor_ltmle,
        few_cluster_floor_ltmle_properties,
        here=Path(__file__).resolve().parent,
    )
