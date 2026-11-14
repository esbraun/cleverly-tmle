"""Regenerate the ``boundary-node-ltmle`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container. Laws, learners, cells, sizes,
margins, seeds and replicate counts are those of :mod:`tests.studies.boundary_node_ltmle` and
:mod:`tests.studies.boundary_node_properties`, fixed before any run and recomputed by
``tests/unit/test_boundary_node_study_design.py``. The record is not in ``registered()`` until
its run is committed; until then its property module is an uncommitted subject of the property
differential.

Order of the work (plan X54 rev 3.1, section 5.5): the laws and their declared grids; the pilot
of :mod:`tests.studies.boundary_node_pilot` on its own seed stream, 400 draws of each of the
eight cells, committed as ``pilot.csv.gz``; this declaration; the declared run. Budget: about
0.06 s per fit at the pilot (one fit per key, its three arms read off that fit), so 8 cells by
4,000 draws plus the 1,000-draw primary is about 0.6 CPU-h of fitting; declared 6 CPU-h.

What is gated, and why (declared before any run, from the law and not from a pilot value).
The supplement acts only on a draw whose node 4 is flagged (a firing draw). A non-firing
draw's interval is the one the package shipped before the supplement, and in these laws it sits
in the rare-event regime: a node flagged with probability ``f`` holds about ``-ln f`` expected
events, so a non-firing draw carries one or two events at the node that holds most of the
variance. The same identity makes flagging common, so no law can have both. The gates are
therefore the statistics the supplement changes:

* ``bias`` (G2) on every ``supplemented`` row;
* ``firing floor`` on every ``supplemented`` row whose exact expected firing count at 4,000
  draws is 600 or more (``boundary_node_properties.FIRING_CELLS``, 19 rows): the exact 99% lower
  end of coverage on the firing draws at least 0.90;
* ``zero-cell coverage`` on ``risk[always]`` of ``zero_n200`` and ``held_zero_n200``, where every
  draw fires: the exact 99% lower end of all-draw coverage at least 0.92;
* ``paired gain`` and ``plug-in loses coverage`` on the ``plugin`` row of ``interior_n200``
  ``always``, the ``paired_attributable`` control of plan B1.

All-draw coverage of every arm is published with its exact interval and its G1 reading
(``all_draw_floor_met``), with the firing, non-firing and multi-flag coverages and the median
half-width ratio to the ``oracle_node`` interval. Law B has no attributable control: under a
held treatment a node flagged half the time carries about 0.04 of the variance, so no clause
has power. Law B shows that the hazard path, with its coefficient bound, does not lose coverage
on firing draws and what it costs in width; it cannot show that the supplement is needed there.

Pilot readings at 400 draws (``pilot.csv.gz``): supplemented coverage on firing draws 0.95 to
1.00 in every gated row; plug-in coverage of the control 0.824 on firing draws against 0.892 on
non-firing draws, supplemented 1.000, so both paired clauses have power 1.0 and 0.9995 at
4,000 draws (``boundary_node_pilot.control_power``); all-draw supplemented coverage 0.918 at
``interior_n400`` ``always`` and non-firing coverage 0.78 to 0.93 in the law A interior cells
under ``always``; half-width ratio of ``risk[always]`` to the oracle-node interval 1.6 to 1.9
in the law A interior cells, 3.3 in the zero cell, 1.07 to 1.08 on law B; standardized bias at
most 0.14.

Red-cell readings (declared before any run). The record is ``reporting``, so a red gated verdict
is published, and it takes the route of plan section 5.8 named here. No cell is re-sized,
re-drawn, dropped or relabelled after a verdict is seen.

* Every gated statistic passes and the paired control passes: the supplement ships on both
  paths. A published non-firing coverage below 0.925 in an interior cell, and an all-draw
  reading below its G1 floor (predicted at ``interior_n400`` ``always``), are recorded under
  roadmap F31 as the measured rare-event limit of the shipped interval (R4), declared here before
  the run. They are not part (i) red cells.
* A law B gated statistic is red while law A passes: the iterated-mean path ships alone, and the
  hazard path refuses a flagged node with its R2 text, reopened by F31.
* A law A ``firing floor`` is red and the diagnosis finds no defect (the finite-sample reading of
  the plans guide: the 99% interval of the cell's SE ratio contains 1 and its coverage is
  consistent with 0.95): the supplement ships under a new shown status ``"boundary_node_plugin"``
  (experimental label, interval shown), reopened by F31. Any other red firing floor is a defect
  suspect and nothing ships until it is fixed and re-run under a fresh declaration.
* The law A zero cell is red: a defect suspect, because the conservative argument of plan 2.5(5)
  failed. Diagnose it; nothing ships until it is fixed.
* The primary (``interior_n200``, 1,000 draws, the shipped interval of each target) is read by
  the harness's G1, G2 and SE-ratio band of 0.8 to 1.2. The supplement is a bound, so the SE
  ratio of ``risk[always]`` is predicted above the band (pilot 1.30, contrast 1.18). That reading
  is the declared width cost of the bound, published under ``reporting``, not a red cell. A red
  primary G2 is a defect suspect.
* A red ``bias`` row is a defect suspect: the supplement does not move a point estimate, every
  learner is correctly specified, and the pilot read at most 0.14 SD.
* A paired clause fails, or has no power at the declared law: the study cannot license the
  construction and nothing ships. That is a design-test failure, redesigned under a fresh
  declaration, not a red cell.
* A replication that raises is never redrawn. A recordable failure, including the study GLM's
  IRLS failure on a nearly separated pseudo-outcome, costs its key in every cell of its law
  under the declared bound of 0.05 (the pilot lost 6 of 400 cross-fitted draws and 2 of 400 at
  ``n = 100``), and every declared rule reads a lost draw as a miss wherever it can fall. A cell
  past its bound is red with the count stated.
"""

from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import boundary_node_ltmle, boundary_node_properties

if __name__ == "__main__":
    run(
        boundary_node_ltmle,
        boundary_node_properties,
        here=Path(__file__).resolve().parent,
    )
