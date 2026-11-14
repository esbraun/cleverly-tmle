"""Regenerate the ``few-cluster-floor-tmle`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container. Laws, fits, ladders, arms,
margins, seeds and replicate counts are those of :mod:`tests.studies.few_cluster_floor_tmle`,
:mod:`tests.studies.few_cluster_floor_tmle_properties` and :mod:`tests.studies.few_cluster_floor`,
fixed before any run and recomputed by ``tests/unit/test_few_cluster_floor_design.py``. The
record is not in ``registered()`` until its run is committed. Until then its property module is
a synthetic subject of the property differential, on the subset it declares
(``DIFFERENTIAL_BUDGET`` and ``differential_cells()``).

**State: declared for design review, pilot not run.** The pilot
(:mod:`tests.studies.few_cluster_floor_pilot`, 400 draws per cell on the ``"pilot"`` stream) runs
after the design review and before the run. It sets each family's R by the family-wise rule
(steps of 400 from 4,000 to 6,000, M5) and writes ``pilot.csv.gz``, ``pilot-failures.csv``,
``pilot-summary.csv``, ``pilot-cells.csv``, ``pilot-sizing.csv``, ``pilot-candidates.csv`` and
``pilot-timing.csv`` here. The declaration then copies R per family, the predicted reds with
their readings and (C)'s predicted shortfalls into the study module (``PROPERTY_REPLICATES``,
``PILOT_PREDICTED_REDS``, ``PILOT_SHORTFALLS``, ``JACKKNIFE_SHORTFALLS``), and the design test
checks them against the committed tables. It changes no rule.

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

**Part (c)** (plan 5.7), for ``tmle_in_sample``, ``tmle_stacked`` (with its five-covariate
configurations, which give (B) a cell where :math:`p > 2` and report their own floor reading
outside the floor rule) and ``ltmle_in_sample``, on the first 2,000 draws of each cell for all
three candidates. (C) runs as its own key, so its refusals, failed refits and errors are its own
withheld misses and never cost (A) a row. (B) and (C) are admissible from (A)'s floor (for
cross-fitted (C), from 10, where R5 admits it) when the 99% lower end of the paired all-draw
coverage gain over (A), withheld and refused draws as misses and the S7 bootstrap interval
(``few_cluster_candidate``), is at least -0.03 in every cell, and the pooled gain
(``few_cluster_candidate_pooled``) is above -0.01 in every configuration there. The winner leads
by at least 2 calibrated counts (shown-only exact interval inside [0.92, 0.98]), else is the
shortest candidate whose half-width ratio to (A) has a 99% upper end below 1; ties go to A, then
B, then C (:func:`tests.studies.few_cluster_floor.reference_decision`). The consequences are
:func:`tests.studies.few_cluster_floor.outcomes`: (B) becomes the default only if it wins every
compared family (with ``few-cluster-floor-cross-fitted-ltmle``); (C) ships as the option it
already is when admissible, becomes a family default when it wins (the family's floor is then
(C)'s admissibility count), and is refused for a family where it is not admissible.

**Budget**, measured on this machine from 3 serial draws of every cell with one thread
(``fit_rows``, the package-GLM refit and the jackknife included): the property phase at
R = 4,000 is 33.6 CPU-h, of which the jackknife arms are 10.9 and the DR-TMLE fits about 20; the
1,000-draw primary under 0.1. With every family raised to 6,000 the property phase is about 45
CPU-h. Two-draw smokes of every key, timed per key, project 33 to 35 CPU-h at the declared R. The
pilot is about 4.5 CPU-h. Its ``pilot-timing.csv`` projects the run from 400 draws at the R that
``pilot-sizing.csv`` picks, read against 1.25 times the declared budget, and at R = 6,000. Declared 60 CPU-h (plan 5.1 declared 250, which assumed 0.15 s
per jackknife refit; a refit here costs 0.005 to 0.02 s).

**Failures.** A key that raises is a lost draw, never redrawn, and a miss in G1 by the
accounting's worst case. The declared run records ``DataError`` (the package's data refusals:
training support, GLM separation or a singular information matrix, and the DR-TMLE
leave-two-out refusal of ``reduced_crossfit="foldwise"``, a ``DataError`` since the second
pilot stopped on it as a bare ``ValueError``), ``numpy.linalg.LinAlgError`` with
``QuasiBinomialConvergenceError``, and a plain ``ValueError`` or ``RuntimeError`` raised inside
numpy, scipy, scikit-learn or statsmodels (:func:`cleverly.validation._failures.recordable`).
``CapabilityError`` (a request the study must not make), ``AssertionError`` (the study's own
checks) and every other class stop the run. ``drtmle_known`` keeps ``"foldwise"``: the shipped
floor family ``drtmle_cross_fitted_known_mechanism`` is that configuration, as in the
committed ``clustered-few-cluster-tmle`` study, so a draw it refuses is a lost draw of that
family, not a reason to measure ``"pooled"``. The pilot records every exception of a key, so it
lists any class this paragraph omits; a 24-draw smoke of every cell lost 11 keys, all
``DataError``, at 4 and 5 clusters.

**Red-cell readings** (declared before any run, owner ``X54-cluster-floor`` in
``docs/roadmap.md``). The policy is ``reporting``. A red below the family floor is held under
one of three readings (:func:`tests.studies.few_cluster_floor.declared_route`). A positive cell
whose withheld-and-lost share exceeds 3% while its shown-only coverage is consistent with 0.95
and its SE ratio with 1 reads ``below floor (availability)``; any other red positive reads
``below floor (finite-sample)``; a red control reads ``control power``. A red at or above the
anchor makes the family a defect suspect. It is held under ``finite-sample (plug-in sizing)``
only after a diagnosis that finds no defect is recorded, and it is otherwise held by nobody. A
part (c) candidate cell that fails below its admissibility count reads ``candidate not
non-inferior``. A replication that raises is never redrawn: it is a lost draw, a miss in G1 by
the accounting's worst case. No cell is re-sized, re-drawn, dropped or relabelled after a
verdict is seen.
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
