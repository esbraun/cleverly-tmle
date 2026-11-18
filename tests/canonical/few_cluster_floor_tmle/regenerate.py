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

**State: redeclared after the pilot (committed here), awaiting the design-review confirmation
and the run.** The pilot (:mod:`tests.studies.few_cluster_floor_pilot`, 400 draws per cell on
the ``"pilot"`` stream) wrote ``pilot.csv.gz``, ``pilot-failures.csv``, ``pilot-summary.csv``,
``pilot-cells.csv``, ``pilot-sizing.csv``, ``pilot-candidates.csv`` and ``pilot-timing.csv``
here. The declaration copies its readings into the study module, and the design test recomputes
each from ``pilot-summary.csv`` (``pilot-sizing.csv`` is rewritten from it by ``--resize``). The
pilot changed no rule, margin, law or ladder.

- **R per family** (``PROPERTY_REPLICATES``, the family-wise rule M5: from the expected floor
  up, a plug-in probability of at least 0.95 that every cell passes, R in steps of 400 from
  4,000 to 6,000). ``tmle_stacked`` 6,000 (plug-in 0.966, met); ``tmle_in_sample`` 6,000
  (0.0047), ``tmle_fold_evaluated`` 6,000 (1.8e-12) and ``drtmle_known`` 6,000 (0.920), each
  short of the target at the cap; ``ltmle_in_sample`` 4,000 (0.999). The d5 configurations read
  the stacked R. Every expected floor is the anchor (10, and 20 for ``LTMLE``).
- **Shortfalls** (``PILOT_SHORTFALLS``). R stays 6,000 and the shortfall is declared, as M5
  requires. Each is driven by a cell at or above the anchor: ``tmle_in_sample_known``
  ``equal10`` J10 covers 0.900 in the pilot; the fold-evaluated controls at J10 read a paired
  IID/cluster ratio of 0.80 to 0.81, so clause 1 has no power at the anchor, and
  ``tmle_fold_evaluated_known`` ``equal10`` J10 covers 0.9125; ``drtmle_known`` ``equal10`` J30
  covers 0.915. A red there takes the declared route of a red at or above the anchor: the floor
  stays at the anchor, the family is a defect suspect and is diagnosed first, and the red is
  held by ``X54-cluster-floor`` under ``finite-sample (plug-in sizing)`` only once a no-defect
  diagnosis is recorded in ``NO_DEFECT_DIAGNOSES`` and the cell agrees with its committed
  counterpart (below). No floor or threshold moves. The four J10 reds share one set of 400
  ``equal10`` draws (one draw serves every fit of a ``(law, J)``), where every family reads low
  together, so they are one event, not four; the run draws new samples on the ``property``
  stream.
- **Fold-evaluated: declared F = 10.** Its paired IID/cluster ratio is 0.80 to 0.81 wherever
  every validation fold holds exactly 2 clusters (J6, J8, J10), against 0.75 to 0.76 at J9 and
  0.73 at J15, so its J10 control is a structural red and ``family_floor`` keeps the anchor:
  the declared outcome for ``tmle_fold_evaluated`` is F = 10 (``defect suspect: 10`` until
  diagnosed). The first item of that diagnosis is the comparison of the pilot's J10 ratio,
  0.804 (99% interval 0.775 to 0.836), with the committed ``tmle_cv_evaluation`` cell at the
  same J and V = 5, 0.758 (upper end 0.767).
- **Predicted reds** (``PILOT_PREDICTED_REDS``, 65 rows, 10 of them the d5 reading's): a positive
  whose plug-in G1 and G2 figure at its R is below 0.90, and a control whose power under both
  clauses is below 0.90, which lists a control that turns red jointly with its positive
  (``tmle_stacked_estimated`` ``unequal_informative`` J7, clause 2: the positive's shown-only SE
  ratio is 1.153). Each carries the reading the declared route gives it and is keyed as the
  ledger keys a property row. The summary is computed with the shown set below (not withheld,
  finite positive SE), rebuilt from the raw rows; four reds at J4 and J5 read availability.
  The control figures multiply the two clauses as if independent. Both read the positive's
  cluster SE, and an inflated cluster SE helps clause 1 while it hurts clause 2, so the pass
  events are negatively dependent and the product overstates joint power. It feeds only the
  predicted-red list and the M5 plug-in, never a gate.
- **Merge of main.** The J4 to J7 loss and withheld shares were measured before ``origin/main``
  was merged (its plain-TMLE constant-outcome rule fits some draws the pilot lost to
  training-support refusals, and its fold-evaluated training floor labels J4 and J5). The
  predicted reds there stay records, and ``red_cells.CLAIMS`` follows the run. Every cell at or
  above an expected floor lost and withheld nothing, and a two-draw refit of every key at
  J >= 10 after the merge reproduced the pilot rows exactly.
- **(C)** (``JACKKNIFE_SHORTFALLS``): none. A (C) as good as (A) passes clause (i) in every cell
  with probability 0.993 for ``tmle_in_sample`` (discordance 4.7%), 1.000 for ``tmle_stacked``
  and for ``ltmle_in_sample``.
- **Losses**: 105 lost keys, all ``DataError``, at 4 to 7 clusters: 58 singular GLM
  information matrices (43 of them the d5 treatment GLM at 4 clusters, which loses 47 of 400
  keys there), 31
  training-support refusals, 10 GLM separations and 6 DR-TMLE leave-two-out refusals.
- **Budget**: ``pilot-timing.csv`` projects 42.9 CPU-h at the declared R (43.2 at 6,000 for
  every family), under the declared 60 and the 75 that would go back to the row thread.

**What is gated** (the registered two-clause rule of ``clustered-few-cluster-tmle``,
``property_verdicts.few_cluster_rule``). On each ``t_reference`` row: G1, the exact 99% lower
end of coverage over every requested draw at least 0.90 (a withheld row is a miss; a lost draw
is judged by the failure accounting's worst case); G2, the bias within 0.25 empirical SD. On
each ``iid_t_control`` row: the paired IID/cluster SE ratio's 99% upper end below 0.80, and the
paired positive arm's SE-ratio upper end at most 1.20. The rule runs with ``shown_only``: both SE
ratios read only the shown draws (not withheld, with a finite positive SE; a withheld row may
keep its computed SE), the paired ratio the draws both arms
show, while coverage still counts every withheld draw as a miss; a cell with fewer than two
shown draws publishes no SE ratio and its control fails. The failure accounting's worst cases
read the same shown draws (``few_cluster_rule(shown_only=True)`` in the module's
``PROPERTY_RULES``): the lost count comes from the unfiltered cell, then the withheld rows are
dropped. Every other arm is reported.

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
``below floor (finite-sample)``; a red control reads ``control power``, whichever clause it
fails (including a control that turns red with its positive through clause 2, or a cell with
fewer than two shown draws). A red at or above the
anchor makes the family a defect suspect. It is held under ``finite-sample (plug-in sizing)``
only after a diagnosis that finds no defect is recorded **and** the cell agrees with its
committed counterpart: the cell of ``clustered-few-cluster-tmle`` at the same family, law and J
(an estimated configuration reads the committed oracle cell), compared by the two-sample 99%
interval of the coverage difference (positive) or the paired-ratio difference (control), which
must hold 0 (:func:`tests.studies.few_cluster_floor.agrees_with_committed`). Otherwise nobody
holds it. A red that disagrees with the committed anchor, with no code defect found, means the
shipped floor is not supported for that family by the code that ships (plan 5.8):
``CLUSTER_INTERVAL_FLOORS[f]`` keeps its value only under a new roadmap item that owns the
re-measurement, and no page says the family is certified from 10. A red at or above the anchor
that no committed cell can be compared with (J15 and J39 of the point families, J39 of
in-sample ``LTMLE``, and every d5 cell from 10) is never read as chance: the family's route is
``redeclaration: {counts}``, its floor stays at the anchor, and it goes to a redeclaration. A d5
red there sends ``tmle_stacked`` to that route. A
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
