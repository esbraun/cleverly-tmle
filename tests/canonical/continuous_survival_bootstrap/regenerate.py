"""Regenerate the continuous-survival-bootstrap evidence (X6 part (d)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  The law, cells, learners, size,
margins and control are those of :mod:`tests.studies.continuous_survival_bootstrap`, declared from
X6 plan rev 5, section 5.5, and checked by ``tests/unit/test_continuous_survival_bootstrap_design.py``.

The cross-fitted fit is a separate, reported record with its own script
(``tests/canonical/continuous_survival_bootstrap_cross_fitted/regenerate.py``); this one runs the
in-sample record alone.

Pilot.  ``python -m tests.studies.continuous_survival_bootstrap_pilot --jobs N`` draws 400 bootstrap
fits of ``b1`` and 50 of the cross-fitted fit, each with 200 refits, on its own streams, and writes
``continuous_survival_bootstrap_pilot.csv.gz`` (:mod:`tests.studies.continuous_survival_bootstrap_pilot`
states each reading).  This script refuses a declared run until that file is committed, holds the
declared 400 in-sample draws and agrees with the declaration (:func:`pilot_refusals`):

* ``PRIMARY_REPLICATES`` equals the family-wise rule's count at the design point, at most the cap
  of 7,600; a count above the cap stops the declaration, and the run is never truncated;
* each of the 12 gated calibration rows passes with plug-in probability at least 0.90 at the
  declared count, or is named in a committed ``pilot-diagnosis.md`` beside this script;
* the in-sample pilot does not predict, with posterior predictive probability above 0.10, that the
  run loses more fits than the failure bound of 0.01 allows.

Cell.  ``b1_in_sample``, a bootstrap fit with 500 refits of the base law at n = 1,000, gated: the
primary rule on its 12 percentile rows at the median and upper quartile, and the calibration rule
with the shrunk-SE control in ``calibration.csv``.  The first quartile's six rows come from the same
fits and publish in ``early-time.csv`` with no verdict.

PILOT READINGS.  The committed pilot (local, four workers on one thread each, about 3.4 h of wall
time) holds 120 draws of ``b1`` and 12 of the cross-fitted fit, each a fit with 200 refits; the
declared 400 and 50 extend it on the same streams with ``--resume``, and this script refuses a declared run until
they are committed.  At this size the readings are a screen:

* ``b1``, 118 kept fits, on the 12 gated rows: percentile coverage 0.890 to 0.966 (mean 0.933),
  BCa 0.907 to 0.975 (0.944), Wald on the same fits 0.907 to 0.966 (0.937); bootstrap SE ratio
  0.856 to 1.040 (mean 0.950) beside a Wald SE ratio of 0.884 to 1.019 (0.950).  The two ratios
  move together row by row, so the shortfall is the fits' own, at a Monte Carlo SD of about 0.065
  per row.  The lowest rows are treated cause 2 (0.856 at t = 2.32) and the cause-2 differences
  (0.884 and 0.907).  9 of the 12 rows pass the calibration rule at 3,200 with plug-in probability
  below 0.90, and the shrunk-SE control leaves the band with probability at least 0.94.  The
  first quartile's six reported rows read SE ratios of 0.905 to 1.014.
* ``b1`` lost 2 of 120 fits: each withheld its bootstrap because 7 and 3 of its 200 refits
  raised ``DataError`` (an initial hazard fit puts a combined jump of one or more at some time).
  ``pilot-diagnosis.md`` traces them to a positivity failure: a unit with extreme X3, which the
  treated arm does not cover, read at that counterfactual arm by the interaction Cox model, with
  tied copies of an event unit amplifying the jump.  Over all draws 27 of 24,000 refits failed
  (0.11%).  The failure rule predicts that 3,200 fits lose more than 32 with probability 0.78.
* The owner adopted the jump bound on 2026-11-20: the engine now scales such a jump to sum to
  1 - 1e-8, keeping each cause's share, in every fit and refit, and records it
  (``jump_bound_share``, ``bootstrap_jump_bounded``) and labels the fit experimental.  Refitted
  under it, draws 38 and 45 lose no refit; the pilot's failure line was read before the bound
  and is stale.

The declared run is refused today on three lines: the pilot is short (120 of 400 in-sample draws),
the failure rule, and the 9 gated rows below plug-in 0.90.  The 400-draw extension (about 280 more
fits of 201, roughly 100 CPU-h) must read clean on all three before the run.  It runs under the
jump bound, so its draws are fitted by the engine that will run the study, and its failure line
reads the refits that still fail (the bound leaves the survival-underflow refusal in place).

Budget.  At the pilot's median seconds per fit on one thread (1.34 s in sample, timed while the
machine was shared), 3,200 replicates of 501 fits cost about 600 CPU-h, local.

Red-cell readings (declared before any run).  The record is ``gated``.  No cell is re-sized,
re-drawn or downgraded after a verdict is seen.

* A red ``b1`` row keeps the in-sample bootstrap experimental, which it is regardless (plan 8.9),
  and the reading is stated in the docs, as roadmap ``X20-bootstrap`` does for LTMLE.
* A red ``b1`` row reads as a finite-sample red only when the 99% interval of its SE ratio contains
  1, its coverage is consistent with 0.95, and the diagnosis finds no defect.  Any other red row is
  a defect suspect: diagnose it against ``tests/unit/test_continuous_survival_bootstrap.py``
  (T-d1 to T-d4 and their mutations).
* A shrunk-SE control that does not leave the band is an instrument defect, and the run is void.
* A replication that raises, a withheld bootstrap included, is never redrawn; it is a recordable
  fit failure bounded by 0.01 per cell.
"""

import sys
from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import (
    continuous_survival_bootstrap,
    continuous_survival_bootstrap_pilot,
    continuous_survival_bootstrap_properties,
)


def pilot_refusals() -> list[str]:
    """What the committed pilot says the declaration must change, or that it is missing."""
    study = continuous_survival_bootstrap
    pilot = continuous_survival_bootstrap_pilot
    rows = pilot.read()
    if rows.empty:
        return [f"{pilot.OUTPUT.name} is not committed; run the pilot first"]
    out = []
    declared = pilot.KINDS["in_sample"][1]
    if pilot.draws(rows, "in_sample") < declared:
        out.append(
            f"the pilot holds {pilot.draws(rows, 'in_sample')} draws of in_sample, short of the "
            f"declared {declared}; extend it with --resume"
        )
    count = pilot.at_target_count()
    if count > study.REPLICATE_CAP:
        out.append(f"the rule asks for {count} replicates, above the cap; stop")
    elif count != study.PRIMARY_REPLICATES:
        out.append(
            f"the rule sets {count} replicates and the record declares {study.PRIMARY_REPLICATES}"
        )
    out.extend(pilot.failure_refusal(rows, "in_sample"))
    diagnosed = pilot.DIAGNOSIS.read_text(encoding="utf-8") if pilot.DIAGNOSIS.exists() else ""
    checks = pilot.calibration_checks(rows, study.PRIMARY_REPLICATES)
    for name, passed in zip(checks["estimand"], checks["plug_in_pass"], strict=True):
        if passed < pilot.PLUG_IN_PASS and name not in diagnosed:
            out.append(
                f"{name} passes with plug-in probability {passed:.3f}, below "
                f"{pilot.PLUG_IN_PASS:.2f}; commit its diagnosis in {pilot.DIAGNOSIS.name}"
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
        continuous_survival_bootstrap,
        continuous_survival_bootstrap_properties,
        here=Path(__file__).resolve().parent,
    )
