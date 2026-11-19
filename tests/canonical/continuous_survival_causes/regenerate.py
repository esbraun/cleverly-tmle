"""Regenerate the continuous-survival-causes evidence (X6 part (b)) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`).
The record has no comparator, so the run starts no container.  The laws, cells, learners, sizes,
margins and controls are those of :mod:`tests.studies.continuous_survival_causes` and its
properties module, declared from X6 plan rev 5, section 5.3, and checked by
``tests/unit/test_continuous_survival_causes_design.py``.

Pilot.  ``python -m tests.studies.continuous_survival_causes_pilot --jobs N`` draws 800 primary
fits per cell and 400 control draws per law (each fitted with the wrong hazards twice, shipped
and factual tilt) on its own streams and writes ``continuous_survival_causes_pilot.csv.gz``
(:mod:`tests.studies.continuous_survival_causes_pilot` states each reading).  The primary cells
were then extended to 2,000 draws, because the arm-pooled rule could not size the calibration
family on 800 (``PILOT READINGS``).  This script refuses a
declared run until that file is committed and agrees with the declaration (:func:`pilot_refusals`):

* ``PRIMARY_REPLICATES`` equals the shared primary rule's count on the 24 gated rows, at least
  1,000, and ``CALIBRATION_REPLICATES`` equals the arm-pooled predictive rule's count on the two
  calibration cells; each is at most the cap of 7,600.  A count above the cap stops the
  declaration and goes back to the plan; the run is never truncated to the cap;
* each calibration cell's plug-in pass probability at the declared count is at least 0.90, or the
  cell is named in a committed ``pilot-diagnosis.md`` beside this script;
* each double-robustness control the pilot reads as uninformative (pass probability at 1,000 draws
  below 0.80) is named in ``UNINFORMATIVE_CONTROLS`` and so published as such, and no other is;
* no cell's pilot predicts, with posterior predictive probability above 0.10 (Jeffreys prior on
  its failure rate), that the declared run loses more fits than the failure bound of 0.01 allows.

Cells.  ``l1_cf_1k`` (one cause) and ``l3_cf_1k`` (three causes), the default cross-fitted fit at
n = 1,000, each gated on every risk and risk difference at its median and upper quartile (6 and
18 rows).  The first quartile is not gated: part (a) kept the share floor at 0.40.

PILOT READINGS (the committed pilot: 2,000 primary draws per cell, 400 control draws per law,
no fit raised).

* Primary count 7,200, read on each cell's first 800 draws and set by L3's control cause-1 risk
  at its median (coverage 0.926).  On all 2,000 draws the gated rows read coverage 0.9465 to
  0.959, SE ratio 0.990 to 1.037 and |bias| / SD at most 0.047 for L1, and coverage 0.9345 to
  0.959, SE ratio 0.979 to 1.028 and |bias| / SD at most 0.032 for L3.  The first-quartile rows,
  not gated, read coverage 0.9355 to 0.9555 (L1) and 0.927 to 0.965 (L3).
* Calibration count 3,200 by the arm-pooled rule on 2,000 draws (SE ratios 0.995 for
  ``l1_rd`` and 1.001 for ``l3_rd``); at-target count 2,400.  Plug-in pass at 3,200: 0.996 and
  0.997, so no cell needs a diagnosis.  On the declared 800 draws the ratios read 0.970 and 0.969
  and the rule found no count up to 50,000; the extension was ``guide/studies.md``'s branch for
  that case, and its readings did not depart.  Shrunk-SE power at 3,200: 1.000 in both cells.
* Double robustness (400 draws per law; standardized bias, then pass probability at 1,000):
  ``mechanism_correct`` +0.053 (0.981) for L1 cause 1, -0.005 (1.000) for L3 cause 1, +0.065
  (0.954) for L3 cause 3; ``plug_in_control`` 3.42, 2.42 and 1.54, and ``factual_tilt_control``
  2.24, 1.65 and 1.08, each with power 1.000.  No control is uninformative.
* Failures: none in 4,800 pilot fits; the predicted chance of exceeding the 0.01 bound at 7,200
  is below 1e-8 per cell.
* Seconds per fit (median, one thread each, four jobs): L1 2.48, L3 5.37; the wrong-hazard and
  factual-tilt control fits 1.98 and 1.97 (L1), 4.29 and 4.30 (L3).
* Budget: primary 7,200 x 7.86 s = 15.7 CPU-h; calibration 3,200 x 7.86 s = 7.0 CPU-h;
  double robustness 1,000 draws per law, two fits each, 3.5 CPU-h.  About 26 CPU-h, 39 with a
  1.5 margin: 7 to 10 hours on 4 cores.  Cloud.

Red-cell readings (declared before any run).  The record is ``gated``: every primary row, every
calibration cell, every double-robustness cell and every derived control is a merge gate of PR B's
support change, and any red verdict fails the run.  No cell is re-sized, re-drawn or downgraded
after a verdict is seen, and no law is changed to make a row pass.

* A red L1 row keeps one cause experimental (``MEASURED_CAUSES`` stays without 1); a red L3 row
  keeps three causes experimental.  Either reading is a result: no rerun with a new law.  The
  design test ties ``MEASURED_CAUSES`` to the committed ``performance-tests.csv`` and
  ``properties.csv``: a number of causes joins only when every gated row and property cell of its
  law passes.
* A red primary or calibration row reads as a finite-sample red only when all three of
  ``guide/studies.md``'s conditions hold: the 99% interval of its SE ratio contains 1, its
  coverage is consistent with 0.95, and the diagnosis finds no defect.  The record then moves to
  ``reporting`` with owner row X6, the measured ratio stated, and the number of causes stays
  experimental.  Any other red row is a defect suspect: diagnose it against
  ``tests/unit/test_continuous_survival_causes.py`` (T-b1, T-b2 and their mutations), fix it, and
  rerun under a fresh declaration that records the change; it never ends as an owner row.  A red
  bias row is a defect suspect of the walk (both laws' hazards are in the default library).
* A red ``mechanism_correct`` cell is a defect suspect of the J-cause tilt: the TMLE is doubly
  robust with the right propensity and censoring.
* A red informative control (its bias inside the margin) is an instrument failure: the run is
  void and goes back to the plan.  A control declared uninformative publishes its reading with no
  verdict claim beyond its role, and the three-cause tilt then rests on T-b2.
* A derived calibration control (``shrunken_se_control``, ``noise_control``) that does not leave
  the band is an instrument defect, and the run is void.
* A replication that raises is never redrawn.  A ``DataError`` is a recordable fit failure bounded
  by 0.01 per cell.
"""

import sys
from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import (
    continuous_survival_causes,
    continuous_survival_causes_pilot,
    continuous_survival_causes_properties,
)


def pilot_refusals() -> list[str]:
    """What the committed pilot says the declaration must change, or that it is missing."""
    study = continuous_survival_causes
    properties = continuous_survival_causes_properties
    pilot = continuous_survival_causes_pilot
    rows = pilot.read()
    if rows.empty:
        return [f"{pilot.OUTPUT.name} is not committed; run the pilot first"]
    out = []
    for label in (*study.CELLS, *(f"{law}_controls" for law in properties.ROBUSTNESS_LABELS)):
        if not pilot.draws(rows, label):
            out.append(f"the pilot holds no draws of {label}; run its pilot")
    if out:
        return out
    primary = pilot.primary_count(rows)
    calibration = pilot.calibration_count(rows)
    for name, count, declared in (
        ("primary", primary, study.PRIMARY_REPLICATES),
        ("calibration", calibration, properties.CALIBRATION_REPLICATES),
    ):
        if count > study.REPLICATE_CAP:
            out.append(f"the pilot asks for {count} {name} replicates, above the cap; stop")
        elif count != declared:
            out.append(
                f"the pilot sets {count} {name} replicates and the record declares {declared}"
            )
    for cell in study.CELLS:
        if pilot.draws(rows, cell) != pilot.EXTENSION_DRAWS:
            out.append(f"the pilot of {cell} does not hold its {pilot.EXTENSION_DRAWS} draws")
    out.extend(pilot.failure_refusals(rows, study.PRIMARY_REPLICATES))
    diagnosed = pilot.DIAGNOSIS.read_text(encoding="utf-8") if pilot.DIAGNOSIS.exists() else ""
    checks = pilot.calibration_checks(rows, properties.CALIBRATION_REPLICATES)
    for cell, passed in zip(checks["cell"], checks["plug_in_pass"], strict=True):
        if passed < pilot.PLUG_IN_PASS and cell not in diagnosed:
            out.append(
                f"{cell} passes with plug-in probability {passed:.3f}, below "
                f"{pilot.PLUG_IN_PASS:.2f}; commit its diagnosis in {pilot.DIAGNOSIS.name}"
            )
    readings = pilot.control_readings(rows)
    uninformative = set(readings.loc[~readings["informative"].astype(bool), "cell"])
    if uninformative != set(properties.UNINFORMATIVE_CONTROLS):
        out.append(
            f"the pilot reads {sorted(uninformative)} as uninformative and the declaration "
            f"names {sorted(properties.UNINFORMATIVE_CONTROLS)}"
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
        continuous_survival_causes,
        continuous_survival_causes_properties,
        here=Path(__file__).resolve().parent,
    )
