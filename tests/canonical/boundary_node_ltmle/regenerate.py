"""Regenerate the ``boundary-node-ltmle`` evidence under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
Run it from the repository root as a module,
``python -m tests.canonical.boundary_node_ltmle.regenerate``, so that ``tests`` imports.
The record has no comparator, so the run starts no container. Laws, learners, cells, sizes,
margins, seeds and replicate counts are those of :mod:`tests.studies.boundary_node_ltmle` and
:mod:`tests.studies.boundary_node_properties`, fixed before any run and recomputed by
``tests/unit/test_boundary_node_study_design.py``. The record is not in ``registered()`` until
its run is committed; until then its property module is an uncommitted subject of the property
differential.

Order of the work (plan X54 rev 3.1, section 5.5): the laws and their declared grids; the pilot
of :mod:`tests.studies.boundary_node_pilot` on its own seed stream, 2,000 draws of each of the
eight cells, committed as ``pilot.csv.gz``; this declaration; the declared run. Budget: about
0.04 to 0.08 s per fit at the pilot (one fit per key, its three arms read off that fit), so 8
cells by 4,000 draws plus the 1,000-draw primary is under 1 CPU-h of fitting; declared 6 CPU-h.

What is gated, and why (declared before any run, from the law and not from a pilot value).
The supplement acts only on a draw whose node 4 is flagged (a firing draw). A draw that is
neither firing nor supplemented at another node (a quiet draw) shows the interval the package
shipped before the supplement, and in these laws it sits in the rare-event regime: a node
flagged with probability ``f`` holds about ``-ln f`` expected events, so a quiet draw carries
one or two events at the node that holds most of the variance. The same identity makes
flagging common, so no law can have both. The gates are therefore the statistics the
supplement changes:

* ``bias`` (G2) on every ``supplemented`` row;
* ``firing floor`` on every ``supplemented`` row whose exact expected firing count at 4,000
  draws is 600 or more (``boundary_node_properties.FIRING_CELLS``, 19 rows): the exact 99% lower
  end of coverage on the firing draws at least 0.90;
* ``zero-cell coverage`` on ``risk[always]`` of ``zero_n200`` and ``held_zero_n200``, where every
  draw fires: the exact 99% lower end of all-draw coverage at least 0.92;
* ``paired gain`` and ``plug-in loses coverage`` on the ``plugin`` row of ``interior_n200``
  ``always``, the ``paired_attributable`` control of plan B1.

All-draw coverage of every arm is published with its exact interval and its G1 reading
(``all_draw_floor_met``), with the firing, quiet and multi-flag coverages, the SE ratio on the
firing draws with its 99% bootstrap interval, and the median half-width ratio to the
``oracle_node`` interval. Law B has no attributable control: under a held treatment a node
flagged half the time carries about 0.04 of the variance, so no clause has power. Law B
measures the coverage of the hazard-path interval on draws with a flagged node; the coefficient
bound rests on its unit witness (``tests/unit/test_boundary_node_variance.py``).

Pilot readings at 2,000 draws per cell (``pilot.csv.gz``): supplemented coverage on firing draws
0.957 to 1.000 in every gated row; plug-in coverage of the control 0.828 on firing draws against
0.900 on non-firing draws, supplemented 0.999; quiet coverage of ``risk[always]`` 0.79 to 0.94
in the law A interior cells; half-width ratio of ``risk[always]`` to the oracle-node interval
1.6 to 1.9 in the law A interior cells, 3.4 in the zero cell, 1.07 to 1.08 on law B;
standardized bias at most 0.09. Lost keys: 6 at ``n = 100`` and 3 cross-fitted, each because no
unit followed ``always`` through node 4 in the sample or in an outer training fold; none
elsewhere.

Pass probabilities (``tests/unit/test_boundary_node_study_design.py``, predictive: each
simulation draws the pilot's rates from their posteriors). Every G2 row passes with probability
at least 0.99 (all read 1.00) with the 95th percentile of its lost keys filled at the worst
vertex. Every firing floor and zero cell reads at least 0.9999. The paired clauses have power
1.00 and 0.993. The product over the 47 gated clauses (24 G2 rows, 19 firing floors, 2 zero
cells, 2 paired clauses) is 0.993. In the primary, the bias legs of all three targets read 1.00;
the coverage legs read 0.995, 0.945 and 1.00 for ``risk[always]``, ``risk[never]`` and the
contrast; the SE-ratio leg of ``risk[never]`` reads 1.00, and those of ``risk[always]`` and the
contrast read 0.00 and 0.05 (below). Each primary row's declared reading names the one leg it
covers (``boundary_node_ltmle.PRIMARY_RED_READINGS``).

Red-cell readings (declared before any run). The record is ``reporting``, so a red gated verdict
is published, and it takes the route of plan section 5.8 named here. No cell is re-sized,
re-drawn, dropped or relabelled after a verdict is seen, and no margin differs from the harness
defaults.

* The primary SE-ratio legs of ``risk[always]`` and the contrast are predicted red: the pilot
  reads SE ratios of 1.29 and 1.19 against the band's upper end of 1.20, a pass probability
  below 0.10 for each. The supplement adds a flagged node's variance at the one-sided bound of
  its hazard, not at the hazard, so this is the bound's width cost and no defect. Both rows are
  declared here under the roadmap's red-cell owner ``X54-supplement-width``
  (``boundary_node_ltmle.PRIMARY_RED_READINGS``), and the claims are added to
  ``tests/studies/evidence/red_cells.py`` with the run. That owner covers the SE-ratio leg
  only. A red coverage or bias leg of these rows is a defect suspect, and at registration
  ``boundary_node_ltmle.unowned_primary_failures`` must find no such leg in
  ``performance-tests.csv`` (``test_a_registered_primary_fails_only_its_owned_legs``).
* The primary coverage leg of ``risk[never]`` is predicted to pass (pilot coverage 0.940, pass
  probability 0.945). If it is red, it takes the one reading this study gives a shortfall of
  the ordinary interval in the rare-event regime, the property family's ``rare_event_limit``:
  node 4 is flagged under ``never`` on one draw in ten, so most draws show the ordinary interval
  at a node with one or two events. It is recorded under roadmap F31, owned in the red-cell
  ledger by ``X54-rare-event-coverage``. That owner covers the coverage leg only; a red bias or
  SE-ratio leg of the row is a defect suspect.
* Every gated statistic passes and the paired control passes: the supplement ships on both
  paths. A quiet coverage whose exact lower end is below 0.90 (``rare_event_limit``) in an
  interior cell, predicted at ``interior_n100``, ``interior_n200`` and the cross-fitted cell, is
  recorded under roadmap F31 as the measured rare-event limit of the shipped interval (R4),
  declared here before the run. It is not a part (i) red cell.
* A law B gated statistic is red while law A passes: the iterated-mean path ships alone, and the
  hazard path refuses a flagged node with its R2 text, reopened by F31.
* A law A ``firing floor`` is red and the diagnosis finds no defect (the finite-sample reading of
  the plans guide: the 99% interval of the cell's firing SE ratio contains 1 and its coverage is
  consistent with 0.95): the supplement ships under a new shown status ``"boundary_node_plugin"``
  (experimental label, interval shown), reopened by F31. Any other red firing floor is a defect
  suspect and nothing ships until it is fixed and re-run under a fresh declaration.
* The law A zero cell is red: a defect suspect, because the conservative argument of plan 2.5(5)
  failed. Diagnose it; nothing ships until it is fixed.
* A red ``bias`` row is a defect suspect: the supplement does not move a point estimate, every
  learner is correctly specified, and the pilot read at most 0.09 SD.
* A paired clause fails, or has no power at the declared law: the study cannot license the
  construction and nothing ships. That is a design-test failure, redesigned under a fresh
  declaration, not a red cell.
* A replication that raises is never redrawn. A recordable failure costs its key in every cell
  of its law under the declared bound of 0.05, and every declared rule reads a lost draw as a
  miss wherever it can fall. The study GLM keeps its last iterate on a pseudo-outcome that is
  binary to within ``1e-4`` and completely separated once rounded, where it raised in the first
  pilot (:class:`tests.studies.fractional_glm.QuasiBinomialGLM`, ``numerical_separation``); no
  other study opts in, and no fit that converged changes. Where a target separated once rounded
  has a value farther from 0 or 1 than that, as on property draw 130 of the cross-fitted cell,
  which stopped the first declared run at ``03e75140``, the learner continues IRLS for up to ten
  times its steps before it raises. A fit that still fails raises
  ``QuasiBinomialConvergenceError``, which the harness records as a lost key, so no learner
  failure stops the run. A cell past its bound is red with the count stated.
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
