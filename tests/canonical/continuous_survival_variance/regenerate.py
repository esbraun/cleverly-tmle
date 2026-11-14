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

* ``PRIMARY_REPLICATES`` equals the pilot's count (the shared primary rule on the gated rows, or
  the 3,600 at-target count of the first-quartile rows if larger), and the count is at most the
  cap of 7,200.  A count above the cap stops the declaration and goes back to the plan; the run
  is never truncated to the cap;
* V7's coefficient follows the halving rule read on the draws at 0.3, and V7's pilot holds draws
  at the declared coefficient;
* ``V7_GATED`` agrees with V7's non-null witness at the declared coefficient;
* no record cell's pilot loses more than the failure bound of 0.01 of its fits;
* each cell flagged by the first-quartile plug-in check holds 2,000 draws, and each row still
  below 0.90 on them is named in a committed ``pilot-diagnosis.md`` beside this script.

Sizes.  Pending the pilot.  The design value is about 6,000 per cell (clause (a) of the shared rule
binds: a row whose true bias is 0.15 SD passes the 0.2 SD bias gate with 0.664 at 3,600, 0.903 at
6,000).  Four cells at n = 1,000 and ``v3_half_n`` at 500; V7 leaves the record if its witness
fails.

Budget.  Per fit, measured on one core with the no-square reading the record carries (one fit
each on replicate 0, 2026-11-14): ``v1_rare_cause2`` 8.5 s, ``v2_heavy_weights`` 6.2 s,
``v3_half_n`` 3.0 s, ``v4_in_sample`` 3.7 s, ``v7_quadratic`` 6.3 s, 27.7 s per replicate across the
five cells.  At 6,000 per cell the primary takes about 46 CPU-h and the pilot (1,000 per cell)
about 7.2.  Extensions are likely: at 6,000 a calibrated first-quartile row reads a plug-in pass
below 0.90 when its pilot SE ratio is outside (0.964, 1.034), so pilot noise alone flags a cell
with probability about 0.54 at 1,000 draws, and leaves a row flagged after 2,000 draws with about
0.16 per cell (each such row then needs its diagnosis).  At three extended cells (1,000 more draws
each, about 2 CPU-h a cell): about 59 CPU-h, 89 with a 1.5 margin, and about 102 with margin at
the cap.  The runner brief restates it from the pilot's own ``seconds``.  Cloud, one job per cell;
the slowest cell (V1) takes about 14 CPU-h at 6,000, under 4 h on a 4-core runner.

Red-cell readings (declared before any run).  The record is ``gated``: every median and
upper-quartile row of every record cell is a merge gate under the primary rule, and any red
verdict fails the run.  The support floors only label estimates; a floor that falls below a red
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
  by 0.01 per cell.
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
        if share > study.FAILURE_BOUND:
            out.append(f"{cell} lost {share:.3f} of its pilot fits; diagnose it")
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
