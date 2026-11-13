"""Regenerate the continuous-survival-functionals evidence (X6 part (c)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  The law, learners, sizes, margins
and replicate counts are those of :mod:`tests.studies.continuous_survival_functionals` and its
properties module, declared from X6 plan rev 2, section 5.4, and checked by
``tests/unit/test_continuous_survival_functionals_design.py``.

Sizes.  The pilot (``python -m tests.studies.continuous_survival_functionals_pilot``, 800 draws
of the primary fit on its own stream) writes ``continuous_survival_functionals_pilot.csv.gz``.
The primary count is the shared primary rule's count from it (about 6,000 to 7,200 at the
registered study's coverages); until it is committed the count is the placeholder 0 and this
script refuses a declared run.  The calibration cell of the RMST difference takes 3,200
replicates by the at-target rule, checked against the pilot's SE ratio; the targeting-necessity
reading takes 1,000 risk-only fits of the primary draws.  All fits are at n = 1,000.

Scope.  The record gates 25 targets of the default cross-fitted fit with
``restricted_horizons=(2.32,)``: each arm's risk of each cause and each risk difference at the
median and upper quartile (1.12 and 2.32), each cause's risk ratio there on the log scale, and at
tau = 2.32 the RMTL of each arm and cause, the RMST of each arm and their differences.  Every row
reads the shipped interval: range-logit for a risk on (0, 1) and for an RMTL or RMST on
(0, tau), log for a ratio, identity for a difference until Fix-3 PR 3.  The joint band row is a
second phase that waits for the band thread's PR 2.

Budget.  One fit at n = 1,000 with the restricted horizon took 6.4 to 6.7 s on one core (two
draws, 4 and 5 walk steps), against 5.8 to 5.9 s without it.  At 6.5 s: pilot 800 fits
(1.4 CPU-h), primary 6,000 to 7,200 (10.8 to 13.0), calibration 3,200 (5.8) and necessity 1,000
(1.8): about 20 to 22 CPU-h, 30 to 33 with a 1.5 margin, so 8 h on 4 cores.  Cloud.

Red-cell readings (declared before any run).  The record is ``gated``, so any red verdict fails
the run.  A red that the diagnosis finds is not a defect follows the route of
``docs/development/method-benchmarking.md``: the record moves to ``reporting`` with the owner row
X6, the run is repeated under the new commit with the same deterministic seeds, and the claim the
cell carries stays experimental.  No cell is re-sized, re-drawn or downgraded after a verdict is
seen.

* A red bias row is a defect suspect of the walk or the restricted rows: every nuisance library
  holds the truth.  Diagnose it against ``tests/unit/test_continuous_survival_functionals.py``
  (the Gateaux-derivative witnesses past the last target time and their mutation controls).
* A red coverage or SE-ratio row, primary or calibration, reads as a finite-sample red only when
  all three of ``guide/studies.md``'s conditions hold: the 99% interval of its SE ratio contains
  1, its coverage is consistent with 0.95, and the diagnosis finds no defect.  It then takes the
  route above with the measured ratio stated.  Any other red coverage or SE-ratio row is a
  variance defect suspect, including one whose SE ratio's interval lies wholly below 1 at the
  median: the registered study's early-time decision covers its first target time only.
* A red risk-ratio row with a green row for each of its two risks is a defect suspect of the
  log-scale curve.
* A derived control (``shrunken_se_control``, ``noise_control``) that does not leave the band is
  an instrument defect, and the run is void.
* The targeting-necessity reading carries no verdict.  If the plug-in RMTL's bias is inside the
  0.2 SD margin, the reading says this law does not separate the plug-in from the TMLE, and the
  claim rests on the score test of the fast tier.
* A replication that raises is never redrawn.  A ``DataError`` (an initial hazard fit whose
  survival reaches zero before tau, or whose combined jump reaches one) is a recordable fit
  failure, bounded by the failure bound of 0.01.
"""

import sys
from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import (
    continuous_survival_functionals,
    continuous_survival_functionals_pilot,
    continuous_survival_functionals_properties,
)


def refuse_unsized() -> None:
    """Refuse a declared run until the committed pilot sets the primary count."""
    if any(argument.startswith("--replicates") for argument in sys.argv[1:]):
        return  # a smoke run
    pilot = continuous_survival_functionals_pilot
    rows = pilot.read()
    if rows.empty:
        raise SystemExit(f"refused: {pilot.OUTPUT.name} is not committed; run the pilot first")
    count = pilot.primary_count(rows)
    if count != continuous_survival_functionals.PRIMARY_REPLICATES:
        raise SystemExit(
            f"refused: the pilot sets {count} primary replicates and the record declares "
            f"{continuous_survival_functionals.PRIMARY_REPLICATES}"
        )


if __name__ == "__main__":
    refuse_unsized()
    run(
        continuous_survival_functionals,
        continuous_survival_functionals_properties,
        here=Path(__file__).resolve().parent,
    )
