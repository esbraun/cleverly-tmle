"""Regenerate the continuous-survival-bootstrap evidence (X6 part (d)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  The law, cells, learners, size,
margins and control are those of :mod:`tests.studies.continuous_survival_bootstrap`, declared from
X6 plan rev 5, section 5.5, and checked by ``tests/unit/test_continuous_survival_bootstrap_design.py``.

Pilot.  ``python -m tests.studies.continuous_survival_bootstrap_pilot --jobs N`` draws 400 bootstrap
fits of ``b1`` and 50 of ``b2``, each with 200 refits, on its own streams, and writes
``continuous_survival_bootstrap_pilot.csv.gz`` (:mod:`tests.studies.continuous_survival_bootstrap_pilot`
states each reading).  This script refuses a declared run until that file is committed, holds the
declared draws and agrees with the declaration (:func:`pilot_refusals`):

* ``PRIMARY_REPLICATES`` equals the family-wise rule's count at the design point, at most the cap
  of 7,600; a count above the cap stops the declaration, and the run is never truncated;
* each calibration row of ``b1`` passes with plug-in probability at least 0.90 at the declared
  count, or is named in a committed ``pilot-diagnosis.md`` beside this script;
* no declared cell's pilot predicts, with posterior predictive probability above 0.10, that the run
  loses more fits than the failure bound of 0.01 allows.

Cells.  ``b1_in_sample`` (gated: the primary rule on its 18 percentile rows, and the calibration rule
with the shrunk-SE control in ``calibration.csv``) and ``b2_cross_fitted`` (reported), each a
bootstrap fit with 500 refits of the base law at n = 1,000.

PILOT READINGS.  The committed container pilot holds 100 draws of ``b1`` and 8 of ``b2``; the
declared 400 and 50 extend it on the same streams with ``--resume``, and the readings are written
here once the extension is committed.  The design review reads the container pilot's readings in
the study's evidence page draft and in the X6 progress notes.

Budget.  At the container pilot's seconds per fit (about 1.5 s in sample and 3.7 s cross-fitted on
one thread), 3,200 replicates of 501 fits cost about 670 CPU-h for ``b1`` and about 1,650 CPU-h for
``b2``: about 2,300 CPU-h, local.  ``b2`` runs at the record's count because the evidence framework
declares one count per record; plan 5.5's 1,000 for ``b2`` needs a record of its own.

Red-cell readings (declared before any run).  The record is ``gated``.  No cell is re-sized,
re-drawn or downgraded after a verdict is seen.

* A red ``b1`` row keeps the in-sample bootstrap experimental, which it is regardless (plan 8.9),
  and the reading is stated in the docs, as roadmap ``X20-bootstrap`` does for LTMLE.
* A red ``b1`` row reads as a finite-sample red only when the 99% interval of its SE ratio contains
  1, its coverage is consistent with 0.95, and the diagnosis finds no defect.  Any other red row is
  a defect suspect: diagnose it against ``tests/unit/test_continuous_survival_bootstrap.py``
  (T-d1 to T-d4 and their mutations).
* A shrunk-SE control that does not leave the band is an instrument defect, and the run is void.
* ``b2``'s readings are a diagnostic: no theorem covers a cross-fitted bootstrap with a selected
  library, so no reading moves its label.
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
    for kind in ("in_sample", "cross_fitted"):
        declared = pilot.KINDS[kind][1]
        if pilot.draws(rows, kind) < declared:
            out.append(
                f"the pilot holds {pilot.draws(rows, kind)} draws of {kind}, short of the "
                f"declared {declared}; extend it with --resume"
            )
    count = pilot.at_target_count()
    if count > study.REPLICATE_CAP:
        out.append(f"the rule asks for {count} replicates, above the cap; stop")
    elif count != study.PRIMARY_REPLICATES:
        out.append(
            f"the rule sets {count} replicates and the record declares {study.PRIMARY_REPLICATES}"
        )
    out.extend(pilot.failure_refusals(rows, study.PRIMARY_REPLICATES))
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
