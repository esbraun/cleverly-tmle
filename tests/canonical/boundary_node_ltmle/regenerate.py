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

Order of the work (plan X54 rev 3.1, section 5.5; rev 3.3 for this redeclaration): the laws and
their declared grids; the pilot of :mod:`tests.studies.boundary_node_pilot` on its own seed
stream, 2,000 draws of each of the eight cells, committed as ``pilot.csv.gz``; the first
declaration and its declared run; this redeclaration; its declared run.

The first declared run (declaration ``91cb39ae``, artifacts ``cbbd634c``, kept in history and in
this directory until the rerun overwrites them; ``boundary_node_ltmle.FIRST_DECLARED_RUN``)
followed the declared route for a failed paired clause. Its clause (ii) compared the plug-in's
coverage on firing draws (0.845) with its coverage on the non-firing draws of the same cell
(0.861), and its Newcombe interval, -0.0456 to +0.0122, did not lie below 0. Those non-firing
draws are not a well-covered reference: they hold one or two events at node 4 and under-cover
for the rare-event reason, so the clause compared the wrong groups and measured the difference
between two shortfalls, while the supplement gained 304 firing draws and lost none. This
redeclaration is informed by that run. It changes three things and nothing else:

* clause (ii') replaces clause (ii): on the firing draws of the control, ``oracle_node`` minus
  plug-in coverage, with its exact 99% lower end at least 0.05. The ``oracle_node`` interval has
  the plug-in's centre and the flagged node's true variance added, so it contains the plug-in
  interval and the paired difference is one-sided. The margin is the gap between nominal
  coverage and the study's floor (0.95 against 0.90), set from the floor's meaning; the first
  run's lower end, 0.134, would pass any margin up to 0.13. The clause reads no non-firing draw;
* a withheld target's three rows carry ``std_error`` NaN and read as misses, and the draw's other
  targets keep their rows. The first run wrote the plug-in row of an estimate at 0 or 1 with a
  curve variance of 0, the row check rejected it, and 8 ``interior_n100`` keys were lost in all
  nine cells. The row check now accepts NaN on a withheld row and nothing else;
* fresh ``SEED`` and ``RESAMPLING_SEED`` and fresh property stream labels
  (``boundary_node_properties.SAMPLE_STREAM`` and ``SE_RATIO_STREAM``), so the rerun reads the new
  clause on draws that did not inform it.

Every cell, R, gate, margin and owner other than these is unchanged, and the first run's rows
serve only as a second pilot for the power of clause (ii'). The control row also publishes, as
a reading that gates nothing, whether the exact 99% upper end of the plug-in's firing coverage
is below 0.95 (``firing_below_nominal``; the first run read 0.845), so the attribution states
that the plug-in falls short of nominal as well as that the node's true variance moves coverage.

All eight cells and the primary rerun, not only the control cell: the harness records one
declaration per manifest, so a partial rerun would leave cells under two declarations, and the
whole rerun costs about 0.15 CPU-h. Fresh draws can only add reds to the cells that passed. The
first run's passing cells are history, not evidence: the registration cites the rerun's rows
and names the first run only as the record of why the study was redeclared. Budget: about
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
* ``paired gain`` (clause (i)) and ``node variance matters`` (clause (ii')) on the ``plugin``
  row of ``interior_n200`` ``always``, the ``paired_attributable`` control of plan B1.

All-draw coverage of every arm is published with its exact interval and its G1 reading
(``all_draw_floor_met``), with the firing, quiet and multi-flag coverages, the SE ratio on the
firing draws with its 99% bootstrap interval, and the median half-width ratio to the
``oracle_node`` interval. Law B has no attributable control: under a held treatment a node
flagged half the time carries about 0.04 of the variance, so no clause has power. Law B
measures the coverage of the hazard-path interval on draws with a flagged node; the coefficient
bound rests on its unit witness (``tests/unit/test_boundary_node_variance.py``).

Pilot readings at 2,000 draws per cell (``pilot.csv.gz``): supplemented coverage on firing draws
0.957 to 1.000 in every gated row; on the control's 1,002 firing draws, plug-in coverage 0.828,
supplemented 0.999 and ``oracle_node`` 0.998, the oracle covering 170 draws the plug-in misses
and missing none it covers (clause (ii') lower end 0.140; the first declared run read 303 of
1,958, lower end 0.134); quiet coverage of ``risk[always]`` 0.79 to 0.94
in the law A interior cells; half-width ratio of ``risk[always]`` to the oracle-node interval
1.6 to 1.9 in the law A interior cells, 3.4 in the zero cell, 1.07 to 1.08 on law B;
standardized bias at most 0.09. Lost keys: 6 at ``n = 100`` and 3 cross-fitted, each because no
unit followed ``always`` through node 4 in the sample or in an outer training fold; none
elsewhere.

Pass probabilities (``tests/unit/test_boundary_node_study_design.py``, predictive: each
simulation draws the pilot's rates from their posteriors). Every G2 row passes with probability
at least 0.99 (all read 1.00) with the 95th percentile of its lost keys filled at the worst
vertex. Every firing floor and zero cell reads at least 0.9999. The paired clauses (i) and (ii')
have power 1.00 and 1.00 from the pilot, and clause (ii') also 1.00 from the first run's rates;
on law B's first cell, where the flagged node carries about 0.03 of the variance, clause (ii')
has power 0.00, so the clause can fail. The product over the 47 gated clauses (24 G2 rows, 19
firing floors, 2 zero cells, 2 paired clauses) reads 1.00. In the primary, the bias legs of all three targets read 1.00;
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
* A paired clause fails (the supplement does not change firing-draw coverage, clause (i), or
  the true node variance changes it by less than 0.05, clause (ii')), or has no power at the
  declared law: the study cannot license the construction and nothing ships. That is a design-test failure, redesigned under a fresh
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
