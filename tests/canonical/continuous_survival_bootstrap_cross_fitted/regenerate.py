"""Regenerate the cross-fitted continuous-survival-bootstrap evidence (X6 part (d)), reported only.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  Its law, learners, margins and
readings are those of the gated in-sample record
(``tests/canonical/continuous_survival_bootstrap/regenerate.py``); its cell, count and seeds are
:mod:`tests.studies.continuous_survival_bootstrap_cross_fitted`'s, checked by
``tests/unit/test_continuous_survival_bootstrap_design.py``.

Cell.  ``b2_cross_fitted``: the default cross-fitted fit with 500 refits of the base law at
n = 1,000, 1,000 replicates, all 18 risks and risk differences at the three quartiles with the
percentile, BCa and Wald readings.  The record is ``reporting``, its one cell is a reported
scenario, its failure bound gates nothing and its calibration table gives no row a verdict, so no
run of it gates anything: no theorem covers a bootstrap of a cross-fitted fit with a selected
library, and no reading moves its label.

Pilot.  The ``cross_fitted`` kind of ``continuous_survival_bootstrap_pilot.csv.gz``, on its own
streams, reads this record's failures and seconds.  This script refuses a declared run until the
pilot holds the declared 50 draws (:func:`pilot_refusals`).

PILOT READINGS.  The committed pilot, rerun under the jump bound, holds 50 draws, each a fit with
200 refits.  No refit failed in 10,000, and the bound acted in 17.  Percentile coverage reads 0.90
to 1.00 and the bootstrap SE ratio 0.999 to 1.211 (mean 1.072), beside Wald SE ratios of 0.995 to
1.205 on the same fits, too few fits to read a row.  The failure rule, published rather than
gated, puts the probability that 1,000 fits lose more than 10 at 0.30, an artifact of 50 draws
under a Jeffreys prior.

Budget.  At the pilot's median 3.46 s per fit on one thread, 1,000 replicates of 501 fits cost
about 480 CPU-h, local.

Readings (declared before any run).  Every reading is a diagnostic.  A replication that raises, a
withheld bootstrap included, is never redrawn; it is published against the bound of 0.01.  No
cell is re-sized or re-drawn after a reading is seen.
"""

import sys
from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import (
    continuous_survival_bootstrap_cross_fitted,
    continuous_survival_bootstrap_pilot,
    continuous_survival_bootstrap_properties,
)


def pilot_refusals() -> list[str]:
    """What the committed pilot says the declaration must change, or that it is missing."""
    pilot = continuous_survival_bootstrap_pilot
    rows = pilot.read()
    if rows.empty:
        return [f"{pilot.OUTPUT.name} is not committed; run the pilot first"]
    out = []
    declared = pilot.KINDS["cross_fitted"][1]
    held = pilot.draws(rows, "cross_fitted")
    if held < declared:
        out.append(
            f"the pilot holds {held} draws of cross_fitted, short of the declared {declared}; "
            "extend it with --resume"
        )
    out.extend(pilot.failure_refusal(rows, "cross_fitted"))
    return out


def refuse_undeclared() -> None:
    """Refuse a declared run until the committed pilot holds the declared draws."""
    if any(argument.startswith("--replicates") for argument in sys.argv[1:]):
        return  # a smoke run
    reasons = pilot_refusals()
    if reasons:
        raise SystemExit("refused: " + "; ".join(reasons))


if __name__ == "__main__":
    refuse_undeclared()
    run(
        continuous_survival_bootstrap_cross_fitted,
        continuous_survival_bootstrap_properties,
        here=Path(__file__).resolve().parent,
    )
