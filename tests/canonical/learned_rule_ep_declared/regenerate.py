"""Regenerate the learned-rule-ep-declared evidence (P1 PR 1) under the RM30 run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``: :func:`tests.canonical.learned_rule_run.run`. A
``--replicates N`` smoke runs the primary phase on the throwaway record of rule L9. A smoke of
the property phase passes ``--property-smoke N --output <scratch>``: ``N`` draws of every
property cell on the ``smoke`` stream (sample label ``smoke_sample``, fold label
``smoke_fold_partition``), through the shared driver's property step
(:mod:`tests.canonical.property_smoke`), with one injected recordable failure for the failure
accounting. With ``--write-timings`` it writes each cell's seconds per fit to
:data:`tests.studies.learned_rule_ep_declared.SMOKE_TIMINGS`, which the design test reads against
:data:`~tests.studies.learned_rule_ep_declared.SECONDS` with the cost gate. The files the run
writes are :data:`tests.studies.learned_rule_ep_declared.RUN_FILES`.

Red-cell readings, declared before any run and before any pilot was read (P1 plan section 5.2,
rev 2.2 after the design review, carrying over the shipped ``learned-rule-ep`` readings). A red
cell is diagnosed before a route is chosen; no cell is re-sized, re-drawn or downgraded after the
run.

* An ``S`` cell (``nonexc__S``, ``misspec__S``, ``optimal__S``) whose SE-ratio interval lies
  above the band while its limit-rule fit's SE ratio (``limit_se_ratio``: ``limit_std_error``
  over the SD of ``limit_estimate - limit_truth`` on the same rows) sits above one by about the
  same amount reads as the fixed-rule CV-TMLE's own conservative variance, which a wrong outcome
  regression gives. It is not a rule-learner defect: the cell is published red under
  ``reporting`` with the owner P1 (F42 after PR 4), and the declared-sieve condition stays for
  that law.
* A learned-value cell (``nonexc__S``, ``misspec__S``, ``nonexc__L``, ``misspec__L``) whose
  SE-ratio interval lies below the band while its oracle-SE ratio (``oracle_se_ratio``: the
  fixed-rule SE of the realized rules over the error SD) sits below one by about the same amount
  reads as the rule learner's fold covariance: each fold's curve is fitted on rows that are
  other folds' validation rows, the second-order excess the shipped study measured at smaller
  n. It is a finding about the size, not a defect: the cell is published red under
  ``reporting`` with the owner above, and its condition is not removed.
* An SE-ratio cell red while neither ratio of the two readings above explains it, below or above
  the band, is a variance defect suspect. An ``L`` cell red above the band has no pair and is a
  defect suspect.
* A learned-value cell whose coverage is red while its SE ratio is green reads as a bias of the
  realized-rule target: a fold-locality or harness defect suspect. Diagnose it against the unit
  witnesses W-c1 to W-c3, the published ``rule_rows_checked``, ``harness_verified`` and the
  standardized bias. A ``HarnessMismatch`` on a verified replicate stops the run.
* ``optimal__S`` red on its coverage while ``nonexc__S`` and ``misspec__S`` are green is not
  explained by the declared sieve's limit, whose regret is under 0.01 SE at n = 16,000: it is a
  defect suspect. With no defect found it is published red under ``reporting`` with the owner
  P1.
* ``optimal__L`` states no verdict (design review S4). Its truth is the value of ``d0``, so it
  measures the greedy learner's regret, which no claim covers; it publishes its coverage, its
  SE ratio with its interval and ``regret_mean``.
* ``nonexc_S__shrunken_se_control`` that does not fall below the band is an instrument defect,
  and the run is void.
* ``nonexc_L__validation_rule_control`` whose coverage interval does not lie below 0.92 means the
  instrument does not resolve an ``L`` curve fitted on its own validation rows at n = 2,000. The
  run's fold-locality evidence then rests on the unit witnesses W-c2 and W-c3 alone; the
  conditions are not removed until a stronger control is declared under a fresh run.
* The primary phase (n = 2,000, 2,000 replicates per cell) red on its coverage floor 0.90 or its
  SE sanity screen while the n = 16,000 cells of its curve are green reads as the finite-sample
  rule-learner deficit: reported. Red with a red n = 16,000 cell reads with that cell.
* No replicate count, law, learner, margin or seed moves after a verdict is seen. No re-run
  buys a pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tests.canonical import property_smoke
from tests.canonical.declared_run import outside
from tests.canonical.learned_rule_run import run
from tests.studies import learned_rule_ep_declared, learned_rule_ep_declared_properties


def smoke(replicates: int, output: Path, *, write_timings: bool) -> pd.DataFrame:
    """The property smoke; returns the property summary."""
    study, properties = learned_rule_ep_declared, learned_rule_ep_declared_properties
    first = study.POSITIVE_CELLS[0]
    order = list(study.PROPERTY_CELLS["interval_calibration"])
    with property_smoke.instrumented(
        properties, "_replicate", group_of=lambda key: str(key[0]), fail=(first, 0, "smoke")
    ) as timings:

        def generate(*, n_jobs: int) -> pd.DataFrame:
            return properties.generate_property_rows(n_jobs=1, budget=replicates, stream="smoke")

        summary, failure_files = property_smoke.run(
            properties, study.STUDY, generate=generate, output=output
        )
    frame = property_smoke.timing_frame(timings, order)
    frame.to_csv(output / "timings.csv", index=False)
    if write_timings:
        property_smoke.write_timings(frame, study.SMOKE_TIMINGS)
    print(f"failure files: {failure_files}", flush=True)
    print(frame.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--property-smoke", type=int, default=None)
    parser.add_argument("--write-timings", action="store_true")
    known, rest = parser.parse_known_args(sys.argv[1:])
    if known.property_smoke is None:
        run(
            learned_rule_ep_declared,
            learned_rule_ep_declared_properties,
            here=Path(__file__).resolve().parent,
        )
    else:
        options = argparse.ArgumentParser(add_help=False)
        options.add_argument("--output", type=Path, required=True)
        arguments = options.parse_args(rest)
        smoke(
            known.property_smoke,
            outside(arguments.output, "smoke"),
            write_timings=known.write_timings,
        )
