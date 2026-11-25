"""Regenerate the continuous-time-longitudinal evidence (X6 part (e), E2) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else (:mod:`tests.canonical.declared_run`,
which also refuses unless ``HEAD`` equals its upstream on a clean tree and every thread variable
is 1).  The record has no comparator, so the run starts no container.  The laws, cells, learners,
sizes, rules and controls are those of :mod:`tests.studies.continuous_time_longitudinal` and its
readings module, declared from X6 plan rev 7, section 5.7, and checked by
``tests/unit/test_continuous_longitudinal_design.py``.

Pilot.  ``python -m tests.studies.continuous_time_longitudinal_pilot --jobs N --resume`` writes
``continuous_time_longitudinal_pilot.csv.gz`` (:mod:`tests.studies.continuous_time_longitudinal_pilot`
states each reading): the truths, the feasibility readings, the plug-in limit with the IPW-SD
witness, and 400 draws of each of ``day_cf``, ``cont_cf``, ``mechanism_correct`` and
``outcome_correct``.  This script refuses a declared run until that file is committed and agrees
with the declaration (:func:`pilot_refusals`):

* the law module's truths equal the pilot's truth rows, and the feasibility readings lie inside
  the plan's bounds (at least 30 expected events of each cause by day 10 among each regime's
  baseline followers, a 99th weight percentile at most 20);
* the plug-in limit separates: some day-30 risk of always or never sits at least 0.5 true-weight
  IPW SDs from its truth (review S8).  If not, :math:`L`'s coefficients in the cause hazards and
  the :math:`L` transition are raised and every truth and pilot reading is read again;
* every pilot cell holds its 400 draws, and the family rule's counts equal ``PRIMARY_REPLICATES``
  and ``MECHANISM_REPLICATES``.  When no pair of rungs carries the family, the rows whose pilot
  coverage is below 0.925 move to reported before the declaration, which is a change to the
  record, not to this script;
* each calibration row's plug-in pass at the primary count is at least 0.90, or its cell holds
  the 800-draw extension and the row is named in a committed ``pilot-diagnosis.md``;
* the at-target pass of the calibration pair at the primary count is at least 0.95, the
  plug-in control fires with probability at least 0.80 at ``mechanism_correct``'s count, and each
  shrunk-SE control leaves the band with probability at least 0.999;
* no gated cell's pilot predicts, with posterior predictive probability above 0.10 (Jeffreys
  prior), that the declared run loses more fits than the failure bound of 0.01 allows;
* the pilot's seconds per fit project the declared run at no more than 200 CPU-h (plan 5.7's
  hold).

Cells (plan 5.7).  ``day_cf`` (CT-day, every regime, 30 gated rows, 12 of them rule rows) and
``cont_cf`` (CT, always and never, 18 gated rows), the default cross-fitted fit at n = 1,000,
under the primary rule.  ``mechanism_correct`` (12 gated rows and the plug-in control),
``outcome_correct`` and ``both_wrong`` (reported) and the X70 pairing ``ltmle_daily`` (reported)
run inside ``day_cf``'s replicates, each on its own stream and at its own count.

Deviation (declared before any run).  The pairing reads the first 400 ``day_cf`` draws on a
9-node panel whose nodes end at days 3, 6, 10, 13, 16, 20, 23, 26 and 30, in place of the plan's
1,000 draws on a 30-node daily panel, which E1's smoke found infeasible
(:mod:`tests.studies.continuous_time_longitudinal` states why).

Budget.  ``pilot-diagnosis.md`` holds the smoke's measured seconds per fit and peak memory, the
budget table and the ``--jobs`` rule.

Red-cell readings (declared before any run).  The record is ``gated``: every primary row of
``day_cf`` and ``cont_cf``, every row of ``mechanism_correct``, the plug-in control, both
calibration rows and both shrunk-SE controls are gates, and any red verdict fails the run.  No
cell is re-sized, re-drawn or downgraded after a verdict is seen, and no law is changed to make a
row pass.

* A red calibration row is a finite-sample red only when the 99% interval of its SE ratio
  contains 1, its coverage is consistent with 0.95, and the diagnosis finds no defect.  It then
  goes to ``reporting`` with an owner in the roadmap's Red-cell owners table.  Any other red row
  is a defect suspect, and an unfixed defect never ends as an owner row (``guide/studies.md``,
  "Red cells").
* One exception is declared (plan rev 7, review S6).  With :math:`G` right and the continuation
  approximated, the influence-curve variance is expected to be conservative (plan 3.5.7), so an
  SE ratio above 1 is the expected direction.  A calibration row red only because its SE-ratio
  interval lies above 1.07, with its coverage interval at or above 0.95 and no defect found,
  reads as conservative: it goes to ``reporting`` with an owner, and the labels follow the
  primary rows.  A red on the low side stays a defect suspect.
* Diagnosis first reads where the red falls.  Red in ``cont_cf`` only points at the untied path
  (the reduced grid, the continuation held at many visit times).  Red in ``day_cf`` only points at
  the tie rule.  Red at day 30 in both, with day 10 green, points at the visit or covariate term,
  which grows with the visits.
* Red rule rows alone point at the rule's slot, its evaluation or the decision visits.  Red in
  ``mechanism_correct`` points at targeting and the slot weight (plan 3.5.6).  Each diagnosis
  runs paired arms on fresh seeds and reruns at most the affected cell, only for a code defect.
* A red rule row keeps only rules experimental (plan 8.10).  Static regimes are labelled from
  their own rows, so a red rule row holds back no static label.
* A plug-in control that does not fire, or a shrunk-SE control that stays in the band, is an
  instrument failure: the run is void and goes back to the plan.
* ``outcome_correct``, ``sparse_calibration`` and ``both_wrong`` are reported and start no
  diagnosis.  ``ltmle_daily`` gates nothing either; a mean paired difference above 0.5 of the
  continuous engine's SE (``diagnose`` in ``ltmle-pairing.csv``) starts a diagnosis of both
  engines on the same draws.
* A replication that raises is never redrawn.  A ``DataError`` is a recordable fit failure
  bounded by 0.01 per cell; a ``day_cf`` replicate that raised also costs each secondary cell
  below its count one fit (``account_failures``).
"""

import sys
from pathlib import Path

from tests.canonical.declared_run import run
from tests.studies import (
    continuous_time_longitudinal,
    continuous_time_longitudinal_law,
    continuous_time_longitudinal_pilot,
    continuous_time_longitudinal_properties,
)

#: The plan's hold: a projection above this stops the run before it starts (plan 5.7).
HOLD_CPU_HOURS = 200.0
#: The smallest power of the plug-in control and of the shrunk-SE controls.
CONTROL_POWER = 0.80
SHRUNK_POWER = 0.999


def pilot_refusals() -> list[str]:
    """What the committed pilot says the declaration must change, or that it is missing."""
    study = continuous_time_longitudinal
    laws = continuous_time_longitudinal_law
    pilot = continuous_time_longitudinal_pilot
    rows = pilot.read()
    if rows.empty:
        return [f"{pilot.OUTPUT.name} is not committed; run the pilot first"]
    out = []
    truths = pilot.truth_table(rows)
    if not truths:
        out.append("the pilot holds no truth rows; run --kinds truth")
    elif truths != laws.TRUTH_RISKS:
        out.append("the law module's TRUTH_RISKS differ from the pilot's truth rows")
    out.extend(pilot.feasibility_refusals(rows))
    separated = pilot.separated(rows)
    if separated is None:
        out.append("the pilot holds no plug-in limit; run --kinds plugin_limit")
    elif not separated:
        out.append(
            "no day-30 risk of always or never separates its plug-in limit by 0.5 IPW SDs; "
            "raise L's coefficients in the cause hazards and the L transition"
        )
    for cell in pilot.CELLS:
        if pilot.draws(rows, cell) < pilot.DRAWS:
            out.append(
                f"the pilot holds {pilot.draws(rows, cell)} draws of {cell}, not {pilot.DRAWS}"
            )
    if out:
        return out
    counts = pilot.counts(rows)
    if counts is None:
        out.append(
            "no pair of rungs carries the family; move these rows to reported before the "
            f"declaration: {pilot.binding_rows(rows)}"
        )
        return out
    primary, mechanism = counts
    if (primary, mechanism) != (study.PRIMARY_REPLICATES, study.MECHANISM_REPLICATES):
        out.append(
            f"the pilot sets {primary} primary and {mechanism} mechanism_correct replicates and "
            f"the record declares {study.PRIMARY_REPLICATES} and {study.MECHANISM_REPLICATES}"
        )
    diagnosed = pilot.DIAGNOSIS.read_text(encoding="utf-8") if pilot.DIAGNOSIS.exists() else ""
    checks = pilot.calibration_checks(rows, primary)
    for cell, passed in zip(checks["cell"], checks["plug_in_pass"], strict=True):
        if passed >= pilot.PLUG_IN_PASS:
            continue
        if pilot.draws(rows, cell) < pilot.EXTENSION_DRAWS:
            out.append(
                f"{cell}'s calibration row passes with plug-in probability {passed:.3f}; extend "
                f"its pilot to {pilot.EXTENSION_DRAWS} draws"
            )
        elif f"{cell}/{study.CALIBRATION_ROW}" not in diagnosed:
            out.append(
                f"{cell}/{study.CALIBRATION_ROW} stays below 0.90 on the extended pilot; commit "
                f"its diagnosis in {pilot.DIAGNOSIS.name}"
            )
    if pilot.at_target_pass(primary) < 0.95:
        out.append(
            f"the calibration pair passes at target with {pilot.at_target_pass(primary):.3f}"
        )
    power = pilot.control_power(rows, mechanism)
    if power < CONTROL_POWER:
        out.append(f"the plug-in control fires with probability {power:.3f}; back to the plan")
    for cell, probability in pilot.shrunken_power(rows, primary).items():
        if probability < SHRUNK_POWER:
            out.append(f"{cell}'s shrunk-SE control leaves the band with {probability:.4f}")
    out.extend(pilot.failure_refusals(rows, primary, mechanism))
    projected = sum(pilot.budget(rows, primary, mechanism).values())
    if projected > HOLD_CPU_HOURS:
        out.append(f"the declared run projects {projected:.0f} CPU-h, above the plan's hold of 200")
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
        continuous_time_longitudinal,
        continuous_time_longitudinal_properties,
        here=Path(__file__).resolve().parent,
    )
