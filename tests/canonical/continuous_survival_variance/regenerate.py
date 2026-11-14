"""Regenerate the continuous-survival-variance evidence (X6 part (a)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  The laws, cells, learners, sizes,
rules and controls are those of :mod:`tests.studies.continuous_survival_variance` and its readings
module, declared from X6 plan rev 5, section 5.1 ("if S"), and checked by
``tests/unit/test_continuous_survival_variance_design.py``.

Merge condition.  PR A does not merge before this study's committed readings pass the ship rule
(``ship-rule.csv``, candidate ``s``) and the regenerated ``continuous-time-survival`` rows, written
under the switched default, are committed (X6 plan 5.2).  Until then the branch's default
``variance="switched"`` and ``MEASURED_DEFAULTS["variance"]`` describe a configuration no committed
study measures.  Once ``ship-rule.csv`` is committed the design test ties both to it: they equal
``"switched"`` if and only if the rule ships S.

Pilot.  ``python -m tests.studies.continuous_survival_variance_pilot --jobs N`` draws 1,000 fits
per cell on its own stream and writes ``continuous_survival_variance_pilot.csv.gz``
(:mod:`tests.studies.continuous_survival_variance_pilot` states each reading).  This script refuses
a declared run until that file is committed and agrees with the declaration
(:func:`pilot_refusals`):

* ``PRIMARY_REPLICATES`` equals the pilot's count (the shared primary rule on the gated cells, or
  the 3,600 at-target count of the first-quartile rows if larger), and the count is at most the
  cap of 7,200.  A count above the cap stops the declaration and goes back to the plan; the run
  is never truncated to the cap;
* V7's coefficient follows the halving rule read on the draws at 0.3, and V7's pilot holds draws
  at the declared coefficient;
* ``V7_GATED`` agrees with V7's non-null witness at the declared coefficient;
* no gated cell's pilot loses more than the failure bound of 0.01 of its fits;
* no gated cell's pilot predicts, with posterior predictive probability above 0.10 (Jeffreys
  prior on its failure rate over all its pilot draws), that the declared run loses more fits than
  the failure accounting allows (``pilot.failure_refusals``); such a cell's ``DataError`` is a
  defect suspect, diagnosed in ``pilot-diagnosis.md`` before the declaration;
* each gated cell flagged by the first-quartile plug-in check holds 2,000 draws, and each row still
  below 0.90 on them is named in a committed ``pilot-diagnosis.md`` beside this script.

Cells (a pilot-informed change, declared before any primary run).  ``v1_rare_cause2``,
``v3_half_n``, ``v4_in_sample`` and ``v7_quadratic`` are gated cells; ``v2_heavy_weights`` alone
is a reported cell (``REPORTED_CELLS``, ``StudyRecord.reported_scenarios``).  The first committed
pilot (1,000 draws per cell, 789258e6) read V2's switched interval covering control cause 1 at
1.12 at 0.894 with SE ratio 0.88 (the influence curve's at 0.874), the cause-1 difference at 1.12
at 0.907 and treated cause 2 at 1.12 at 0.915; with V2 the shared rule finds no count up to
50,000, on that pilot and on the rerun.  That pilot also lost 12 of 1,000 V3 fits to
``DataError``, so V3 was first reported too.  The defect fix (bc630ef4) removed those failures:
the rerun pilot (aa2c4167) loses no fit in any cell, and with V3 in the sizing the count stays
6,000, so V3 is gated (decided 2026-11-14).  On the rerun V1, V3, V4 and V7 read worst coverages
of 0.928 to 0.939.  V2 runs at the same count on the same seeds, is left out of the count, the
pilot's failure-share refusal and the merge gate, and still feeds ``r_prim`` (a red V2 row caps
it) and both clauses of the ship rule.  V2's failure share is published in the failure
accounting, red above 0.01, and gates nothing (``failures.BOUND_REPORTED``; the record opts out
with ``StudyRecord.reported_bounds_gate=False``, which it keeps only while it holds a reported
cell).  V3's failure share gates under the bound of 0.01 like every gated cell's.  Roadmap item
X6 (a2) owns V2, the weight-driven shortfall.

Sizes.  6,000 per cell, the shared primary rule on the 48 rows of the gated cells (clause (a)
binds: a row whose true bias is 0.15 SD passes the 0.2 SD bias gate with 0.903 at 6,000), above
the 3,600 at-target count of the first-quartile rows.  Four cells at n = 1,000 and ``v3_half_n``
at 500; V7's witness held (0.45 SD) and halving did not fire.

Budget.  Median seconds per fit in the rerun pilot, on one core: ``v1_rare_cause2`` 5.80,
``v2_heavy_weights`` 5.97, ``v3_half_n`` 2.45, ``v4_in_sample`` 2.34, ``v7_quadratic`` 5.75, so
22.3 s per replicate across the five cells; the declared fit adds the no-square reading, about
5% (one fit per cell, 2026-11-14).  At 6,000 per cell the primary takes about 39 CPU-h, of which
V3, gated at 6,000, takes about 4 CPU-h at 2.45 s per fit.  V1 and V4 already hold 2,000 pilot
draws; the plug-in check now also flags V3, whose extension to 2,000 draws takes about 0.7 CPU-h.
About 40 CPU-h, 60 with a 1.5 margin, and about 71 with margin at the cap.  Cloud, one job per
cell; the slowest cell (V2) takes about 10.4 CPU-h, under 3 h on a 4-core runner.

Red-cell readings (declared before any run).  The record is ``gated``: every median and
upper-quartile row of every gated cell is a merge gate under the primary rule, and any red
verdict fails the run.  A reported cell's rows publish bias, coverage and the SE ratio with no
verdict; they decide no gate and feed ``r_prim`` and the ship rule as stated above.  The size
floor ``MEASURED_N`` falls from 1,000 to 500 only if every primary row of ``v3_half_n`` passes; a
lost V3 replicate counts against it, because a V3 failure share above 0.01 fails the gated record
and no floor is read from a failed record.  The support floors only label estimates; a floor that falls below a red
row's rung does not turn the row green or the record into ``reporting``.  No cell is re-sized,
re-drawn or downgraded after a verdict is seen.

* A red primary row reads as a finite-sample red only when all three of ``guide/studies.md``'s
  conditions hold: the 99% interval of its SE ratio contains 1, its coverage is consistent with
  0.95, and the diagnosis finds no defect.  It then caps ``r_prim`` below its rung, the record
  moves to ``reporting`` with owner row X6 and the measured ratio stated, and the claim the cell
  carries stays experimental.  Any other red row is a defect suspect: it is diagnosed, fixed, and
  rerun under a fresh declaration that records the change; it never ends as an owner row.  A red
  bias row is a defect suspect of the walk (every cell but V7 holds the truth in its library).
* If the ship rule fails, S does not ship: the engine default and ``MEASURED_DEFAULTS["variance"]``
  revert to ``"influence_curve"``, the floors read the IC's rows (``floors.csv`` does so already),
  and the study's S rows are reported, not gated: the record moves to ``reporting`` and its
  performance verdicts on S describe a candidate the library does not ship.  The
  ``continuous-time-survival`` study then does not regenerate (plan 5.2).
* First-quartile rows are no gates.  On the S path a first-quartile red whose SE-ratio interval
  lies above 1 while the IC's paired row is green is S's designed over-width: it goes to Eric as a
  red-cell decision, not an automatic owner row.
* An instrument check that fails (``s_pv130`` not firing clause 1, ``s_never`` passing clause 2,
  ``s_shrunk`` passing the calibration rule at a gated row) voids the run, and the design test
  fails on the committed ``ship-rule.csv``.
* The row ``ic_sparse_calibration`` of ``ship-rule.csv`` reads whether the IC fails the
  calibration rule at each cell's sparsest first-quartile row; where it does not, the cell does
  not exercise the sparse-cell failure, and the reading says so.
* A replication that raises is never redrawn; a ``DataError`` is a recordable fit failure bounded
  by 0.01 per gated cell, V3 included.  V2's failures are recorded and published with its bound
  leg, and gate nothing; a share above 0.01 there is a reading for X6 (a2), its owner.
* V2's rows carry no verdict.  Their readings are X6 (a2)'s evidence of the heavy-weight
  shortfall.  A heavy-weight fit carries no configuration label: nothing detects heavy weights,
  and its estimates read experimental only where their ``variance_rel_sd`` lies above ``r_prim``;
  (a2) owns measuring that configuration.
* A red V3 row is read like any gated red row above, and it keeps ``MEASURED_N`` at 1,000, so a
  500-unit fit keeps reading experimental under the size floor.
"""

import sys
from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import (
    continuous_survival_variance,
    continuous_survival_variance_law,
    continuous_survival_variance_pilot,
    continuous_survival_variance_properties,
)


def pilot_refusals() -> list[str]:
    """What the committed pilot says the declaration must change, or that it is missing."""
    study = continuous_survival_variance
    pilot = continuous_survival_variance_pilot
    rows = pilot.read()
    if rows.empty:
        return [f"{pilot.OUTPUT.name} is not committed; run the pilot first"]
    out = []
    halving = pilot.v7_halving(rows)
    if halving is None:
        out.append("the pilot holds no V7 draws at 0.3; run V7's pilot")
    else:
        expected = 0.15 if halving else pilot.V7_INITIAL
        if expected != continuous_survival_variance_law.QUADRATIC:
            out.append(f"V7's halving rule sets the coefficient to {expected}")
    witness = pilot.v7_witness(rows)
    if witness is None:
        out.append(
            f"the pilot holds no V7 draws at {continuous_survival_variance_law.QUADRATIC}; "
            "run V7's pilot at the declared coefficient"
        )
    elif (witness > pilot.WITNESS_BIAS) != study.V7_GATED:
        out.append(f"V7's witness reads {witness:.3f}; set V7_GATED = {not study.V7_GATED}")
    for cell, share in pilot.failure_shares(rows).items():
        if cell in study.GATED_CELLS and share > study.FAILURE_BOUND:
            out.append(f"{cell} lost {share:.3f} of its pilot fits; diagnose it")
    out.extend(pilot.failure_refusals(rows))
    count = pilot.primary_count(rows)
    if count > study.REPLICATE_CAP:
        out.append(f"the pilot asks for {count} replicates, above the cap; stop and report")
    elif count != study.PRIMARY_REPLICATES:
        out.append(
            f"the pilot sets {count} primary replicates and the record declares "
            f"{study.PRIMARY_REPLICATES}"
        )
    for cell in pilot.extension_cells(rows, count):
        if pilot.draws(rows, cell) < study.EXTENSION_REPLICATES:
            out.append(
                f"{cell} has a first-quartile plug-in pass below 0.90; extend its pilot "
                f"to {study.EXTENSION_REPLICATES} draws"
            )
    diagnosed = pilot.DIAGNOSIS.read_text(encoding="utf-8") if pilot.DIAGNOSIS.exists() else ""
    for row in pilot.confirmed_departures(rows, count):
        cell = row.split("/", 1)[0]
        if pilot.draws(rows, cell) >= study.EXTENSION_REPLICATES and row not in diagnosed:
            out.append(
                f"{row} stays below 0.90 on the extended pilot; commit its diagnosis in "
                f"{pilot.DIAGNOSIS.name}"
            )
    return out


def refuse_undeclared() -> None:
    """Refuse a declared run until the committed pilot agrees with the declaration."""
    if any(argument.startswith("--replicates") for argument in sys.argv[1:]):
        return  # a smoke run
    reasons = pilot_refusals()
    if reasons:
        raise SystemExit("refused: " + "; ".join(reasons))


if __name__ == "__main__":
    refuse_undeclared()
    run(
        continuous_survival_variance,
        continuous_survival_variance_properties,
        here=Path(__file__).resolve().parent,
    )
