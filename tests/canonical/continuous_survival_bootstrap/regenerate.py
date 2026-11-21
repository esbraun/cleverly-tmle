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
fits and publish in ``early-time.csv`` with no verdict, and ``paired-wald.csv`` compares each gated
row with the Wald interval on the same fits, also with no verdict.

PILOT READINGS.  The committed pilot (cloud, under the jump bound at main 7b4d0f08) holds 400
draws of ``b1`` and 50 of the cross-fitted fit, each a fit with 200 refits.  Its ``seconds`` column
sums to 39.6 CPU-h (29.9 in sample, 9.7 cross-fitted); the runner's wall clock reported about 46.
It replaced a first pilot of 120 and 12 draws fitted before the bound.

* ``b1`` lost no fit.  3 of 80,000 refits failed and the bound acted in 101, so the failure rule
  predicts that 3,200 fits lose more than 32 with probability 0.005.
* On the 12 gated rows: percentile coverage 0.918 to 0.940 (mean 0.932), BCa 0.925 to 0.950
  (0.940), Wald on the same fits 0.913 to 0.953 (0.934); bootstrap SE ratio 0.897 to 1.044 (mean
  0.957) beside a Wald SE ratio of 0.912 to 1.044 (0.952).  The first quartile's six reported rows
  read bootstrap SE ratios of 0.910 to 1.003.  The shrunk-SE control leaves the band with
  probability at least 0.90 on every row.
* 11 of the 12 rows pass the calibration rule at 3,200 with plug-in probability below 0.90.
  ``pilot-diagnosis.md`` diagnoses them: the bootstrap SE is 0.983 to 1.030 times the Wald SE on
  the same fits, and the pilot's Wald SE ratios run below those of ``continuous-survival-variance``'s
  ``v4_in_sample``, the same fit on the same law at 6,000 fits (0.954 to 1.003), by the chance
  spread of 400 samples.  Those 6,000 disjoint fits stand in for the pilot extension that
  ``guide/studies.md`` asks for, since an extension of the same estimator would add only each
  row's own percentile-minus-Wald coverage difference.  Anchored to ``v4_in_sample``, every
  bootstrap SE ratio is 0.972 to 1.003.  The in-sample fit's own coverage near 0.94 leaves 1.2 to
  3.1 red rows expected at 3,200: 1.2 (family pass 0.26) if every row's percentile coverage sits
  at the pooled difference of -0.003 from the Wald, 3.1 (family pass 0.003) with each row's own
  difference.  The count stays at the at-target 3,200.

Budget.  At the pilot's median seconds per fit on one thread (1.30 s in sample), 3,200 replicates
of 501 fits cost about 580 CPU-h, local.

Red-cell readings (declared before any run).  The record is ``gated``.  No cell is re-sized,
re-drawn or downgraded after a verdict is seen.

* A red ``b1`` row keeps the in-sample bootstrap experimental, which it is regardless (plan 8.9),
  and the reading is stated in the docs, as roadmap ``X20-bootstrap`` does for LTMLE.
* A red ``b1`` calibration row reads as a finite-sample red only when the 99% interval of its SE
  ratio contains 1, its coverage interval contains 0.95, and the diagnosis finds no defect.  Any
  other red row is a defect suspect.
* A defect suspect's diagnosis is declared now.  It first compares the row with the Wald
  interval on the same fits, from ``paired-wald.csv``
  (:func:`~tests.studies.continuous_survival_bootstrap.paired_wald`): the 99% delta-method
  interval of the paired SE ratio (mean bootstrap SE over mean Wald SE) inside (0.95, 1.05), and
  the lower 99% limit of the paired coverage difference (percentile minus Wald, Newcombe's
  interval) above -0.025.  A row that meets both shares its shortfall with the in-sample Wald on
  the same fits, and the diagnosis turns to the in-sample fit.  A row that misses either points at
  the bootstrap, and is diagnosed against ``tests/unit/test_continuous_survival_bootstrap.py``
  (T-d1 to T-d4 and their mutations).  These readings are diagnostics, not a further route to a
  finite-sample red: a row that meets both stays a defect suspect unless the owner approves in
  writing its routing to an owner row as a departure from ``guide/studies.md``.
* A red gated row makes the declared run copy its artifacts and manifest into this directory and
  then stop with the driver's error (``tests/canonical/declared_run.py``).  A finite-sample red
  then moves the record to ``reporting`` with an X6 row in the roadmap's Red-cell owners table.
  At registration the record enters ``EXCLUDED_HOOKS`` (``tests/studies/evidence/red_cells.py``),
  as every study with a ``scientific_failures`` hook does.
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
