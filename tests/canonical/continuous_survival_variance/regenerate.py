"""Regenerate the continuous-survival-variance evidence (X6 part (a)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  The laws, cells, learners, sizes,
rules and controls are those of :mod:`tests.studies.continuous_survival_variance` and its readings
module, declared from X6 plan rev 5, section 5.1 ("if S"), and checked by
``tests/unit/test_continuous_survival_variance_design.py``.

Pilot.  ``python -m tests.studies.continuous_survival_variance_pilot --jobs N`` draws 500 fits per
cell on its own stream and writes ``continuous_survival_variance_pilot.csv.gz``.  This script
refuses a declared run until that file is committed and agrees with the declaration: V7's
coefficient follows the halving rule (0.3, or 0.15 if the influence-curve interval covers a
cause-1 target at the median or upper quartile below 0.93); V7's non-null witness holds (the
plug-in's standardized bias of a cause-1 risk at the median above 0.2 SD, else V7 is re-declared
as reported); and every cell with a first-quartile row whose plug-in calibration pass at 3,600 is
below 0.90 holds an extended pilot, diagnosed before the declaration (``guide/studies.md``).

Sizes.  3,600 replicates per cell (the ship rule's power is stated there; at SE ratio 1 and
coverage 0.95 a calibration row passes with 0.99923), cap 7,200.  Five cells: ``v1_rare_cause2``,
``v2_heavy_weights``, ``v4_in_sample`` and ``v7_quadratic`` at n = 1,000, ``v3_half_n`` at 500.

Budget.  Per fit, measured on one core with the no-square reading the record carries (one fit
each on replicate 0, 2026-11-14): ``v1_rare_cause2`` 8.5 s, ``v2_heavy_weights`` 6.2 s,
``v3_half_n`` 3.0 s, ``v4_in_sample`` 3.7 s, ``v7_quadratic`` 6.3 s.  At 3,600 per cell the primary
takes about 28 CPU-h and the pilot (500 per cell, without the no-square reading) about 3.6, so
about 32 CPU-h, 47 with a 1.5 margin, and about 89 with margin at the cap of 7,200.  The plan's
68 (90 with margin) assumed 9 s per fit at n = 1,000.  Cloud, one job per cell; the slowest cell
(V1) takes about 8.5 CPU-h at 3,600, under 3 h on a 4-core runner.

Red-cell readings (declared before any run).  The record is ``gated`` on the twelve median and
upper-quartile targets of each cell under the primary rule.  A red primary row caps
``r_prim`` below its rung and is diagnosed under ``guide/studies.md``; it moves the record to
``reporting`` with owner row X6, and the run is repeated under the new commit with the same
seeds.  No cell is re-sized, re-drawn or downgraded after a verdict is seen.

* First-quartile rows are no gates.  On the S path a first-quartile red whose SE-ratio interval
  lies above 1 while the IC's paired row is green is S's designed over-width: it goes to Eric as a
  red-cell decision, not an automatic owner row.
* A failed ship rule keeps the IC: PR A then takes the IC path of plan 5.1.
* An instrument check that fails (``s_pv130`` not firing clause 1, ``s_never`` passing clause 2,
  ``s_shrunk`` passing a gated row) voids the run.
* If the IC passes the calibration rule at every sparsest first-quartile row, the cell does not
  exercise the sparse-cell failure, and the reading says so.
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
    pilot = continuous_survival_variance_pilot
    rows = pilot.read()
    if rows.empty:
        return [f"{pilot.OUTPUT.name} is not committed; run the pilot first"]
    out = []
    expected = 0.15 if pilot.v7_halving(rows) else 0.3
    if expected != continuous_survival_variance_law.QUADRATIC:
        out.append(f"V7's halving rule sets the coefficient to {expected}")
    if pilot.v7_witness(rows) <= pilot.WITNESS_BIAS:
        out.append("V7's plug-in bias is inside 0.2 SD; re-declare V7 as reported")
    draws = rows.groupby("label")["replicate"].nunique()
    for cell in pilot.extension_cells(rows):
        if int(draws.get(cell, 0)) <= continuous_survival_variance.PILOT_REPLICATES:
            out.append(f"{cell} has a first-quartile plug-in pass below 0.90; extend its pilot")
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
