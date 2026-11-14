"""Regenerate the ``few-cluster-floor-tmle`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container. Laws, fits, ladders, arms,
margins, seeds and replicate counts are those of :mod:`tests.studies.few_cluster_floor_tmle`,
:mod:`tests.studies.few_cluster_floor_tmle_properties` and :mod:`tests.studies.few_cluster_floor`,
fixed before any run and recomputed by ``tests/unit/test_few_cluster_floor_design.py``. The
record is not in ``registered()`` until its run is committed; until then its property module is
an uncommitted subject of the property differential.

**State: declared for design review, pilot not run.** The pilot
(:mod:`tests.studies.few_cluster_floor_pilot`, 400 draws per cell on the ``"pilot"`` stream) runs
after the design review and before the run. It sets each family's R by the family-wise rule
(steps of 400 from 4,000 to 6,000, M5), records the predicted reds and the predicted
admissibility of (C) in ``README.md``, and is committed as ``pilot.csv.gz``. It changes no rule.

**What is gated** (the registered two-clause rule of ``clustered-few-cluster-tmle``,
``property_verdicts._few_cluster_rule``). On each ``t_reference`` row: G1, the exact 99% lower
end of coverage over every requested draw at least 0.90 (a withheld row is a miss; a lost draw
is judged by the failure accounting's worst case); G2, the bias within 0.25 empirical SD. On
each ``iid_t_control`` row: the paired IID/cluster SE ratio's 99% upper end below 0.80, and the
paired positive arm's SE-ratio upper end at most 1.20. Every other arm is reported.

**Floors** (plan 5.8). A count passes for a family when both mechanism configurations
(``_known`` and ``_estimated``; ``drtmle_known`` and ``ltmle_in_sample`` have one) pass G1, G2
and the control on every size law they run on. The family floor is the smallest ladder count
from which every ladder count passes (:func:`tests.studies.few_cluster_floor.family_floor`); a
failure at or above the anchor keeps the current floor (10, or 20 for ``LTMLE``) and is
diagnosed as a defect suspect. A passing floor certifies
an all-draw coverage whose exact 99% lower end is at least 0.90.

**Part (c)** (plan 5.7), for ``tmle_in_sample``, ``tmle_stacked``, ``tmle_stacked_d5`` and
``ltmle_in_sample``, on the first 2,000 draws of each cell for all three candidates. (B) and (C)
are admissible from (A)'s floor (for cross-fitted (C), from 10, where R5 admits it) when the
paired all-draw coverage difference to (A), withheld and refused draws as misses, has a 99%
lower end above -0.03 in every cell and above -0.01 pooled over the family's cells there. The
winner leads by at least 2 calibrated counts (shown-only exact interval inside [0.92, 0.98]),
else is the shortest candidate whose half-width ratio to (A) has a 99% upper end below 1; ties
go to A, then B, then C (:func:`tests.studies.few_cluster_floor.reference_decision`). The
consequences are the table of plan 5.8: (B) becomes the default only if it wins every compared
family (with ``few-cluster-floor-cross-fitted-ltmle``); (C) ships as the option it already is
when admissible, becomes a family default when it wins, and is refused for a family where it is
not admissible.

**Budget**, measured on this machine from 3 serial draws of every cell with one thread
(``fit_rows``, the package-GLM refit and the jackknife included): the property phase at
R = 4,000 is 33.6 CPU-h, of which the jackknife arms are 10.9 and the DR-TMLE fits about 20; the
1,000-draw primary under 0.1. With every family raised to 6,000 the property phase is about 45
CPU-h. The pilot is about 4.5 CPU-h. Declared 60 CPU-h (plan 5.1 declared 250, which assumed
0.15 s per jackknife refit; a refit here costs 0.005 to 0.02 s).

**Red-cell readings** (declared before any run). The policy is ``reporting``. A positive cell
whose shown-only SE-ratio interval contains 1, whose shown-only coverage is consistent with
0.95 and whose diagnosis finds no defect is ``finite-sample``, owned by F28; its family's floor
stays above that count. Any other red cell is a defect suspect and is diagnosed first. A
replication that raises is never redrawn: it is a lost draw, a miss in G1 by the accounting's
worst case. No cell is re-sized, re-drawn, dropped or relabelled after a verdict is seen.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import few_cluster_floor_tmle, few_cluster_floor_tmle_properties

if __name__ == "__main__":
    run(
        few_cluster_floor_tmle,
        few_cluster_floor_tmle_properties,
        here=Path(__file__).resolve().parent,
    )
