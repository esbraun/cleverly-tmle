"""Regenerate the projection-compositions evidence (P1 PR 1) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory. The study has no
comparator, so the run starts no container. The groups, sizes, learners, controls and design
numbers are those of :mod:`tests.studies.projection_compositions`, fixed before the run, and the
files the run writes are :data:`tests.studies.projection_compositions.RUN_FILES`.

A smoke passes ``--property-smoke N --output <scratch>``. It draws ``N`` replicates of every
group on the ``smoke`` stream (sample label ``smoke_sample``, fold label
``smoke_fold_partition``) and two primary replicates on ``smoke_primary``, none of which a
declared run draws, and runs the shared driver's property step on them
(:mod:`tests.canonical.property_smoke`): the rows, ``summarize_properties``, one injected
recordable failure and its accounting under ``PROPERTY_RULES``. With ``--write-timings`` it
writes each group's seconds per fit to
:data:`tests.studies.projection_compositions.SMOKE_TIMINGS`, which the design test reads against
the declared seconds with the cost gate
(:data:`~tests.studies.projection_compositions.COST_GATE`). The shared driver's own smoke form
(``--replicates``) is not used: it draws the declared primary seeds.

Red-cell readings, declared before any run and before any pilot was read (P1 plan section 5.0,
rev 2.2 after the design review). A red cell is diagnosed before a route is chosen; no cell is
re-sized, re-drawn or downgraded after the run.

* A red calibration row is finite-sample only when its 99% SE-ratio interval contains 1, its
  coverage interval contains 0.95 and the diagnosis finds no defect. It then goes to
  ``reporting`` with the owner P1 (F42 after PR 4). Every other red is a defect suspect:
  diagnose it with paired arms on fresh seeds, fix the defect, and rerun the affected cell
  under a fresh declaration.
* A ``rep3_*`` row red above the band reads as the median rule's split adjustment, a
  conservative variance and not a defect, only when the 99% interval of its
  ``se_ratio_without_displacement`` (published with
  ``se_ratio_without_displacement_ci_lower`` and ``_ci_upper``) lies inside 0.93 to 1.07. It is
  then published red under ``reporting`` with the owner above, and its support row is not
  added. A row whose ratio without the displacement is also outside the band is a defect
  suspect. At GLM nuisances the displacement adds under 0.1% to the SE (design review S1), so
  this reading is not expected to fire; the median rule's displacement term rests on the unit
  witnesses RP2 and RP3.
* A bootstrap row (``__boot_*``) whose ``bootstrap_failed_fraction`` exceeds 0.01 reports
  ``bootstrap_conditional`` and claims no coverage, the shipped bootstrap study's rule.
* ``boot_stacked_2k`` publishes its coverage and SE ratio under the diagnostic role and states
  no verdict: it is the evidence the cross-fitted bootstrap's experimental reason (F2) cites.
* A ``bias`` row of ``mtreat_dr_4k`` red reads as a defect of the composite route under a wrong
  outcome regression (the design's GLM-limit bias is -0.035 and -0.001 SD): diagnose against
  MT2, MT4 and MT6.
* A control that does not detect (``__shrunken_se_control`` or
  ``__independent_splits_control`` inside or above the band, or ``complete_case``,
  ``no_treatment_observation`` or ``marginal_as_stratum`` inside the bias margin) is a red of
  the study, not of the estimator: the cell it guards cannot see the component it is named for,
  and that cell's claim is withheld until the control is repaired under a fresh declaration.
* The primary phase (``mtreat_4k``, 1,000 replicates) red on its coverage floor or SE screen
  while ``mtreat_4k_*__mtreat`` is green reads as the primary phase's finite-sample noise and is
  reported; red with a red calibration row reads with that row.
* No replicate count, law, learner, margin or seed moves after a verdict is seen. No re-run
  buys a pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tests.canonical import property_smoke
from tests.canonical.declared_run import outside, run
from tests.studies import projection_compositions, projection_compositions_properties
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import stream_seed

#: Primary smoke draws.
PRIMARY_SMOKE_REPLICATES = 2


def smoke_primary(replicates: int) -> pd.DataFrame:
    """The primary rows of ``replicates`` draws of ``mtreat_4k`` on the ``smoke_primary`` stream."""
    study = projection_compositions
    rows = []
    for scenario in study.SCENARIOS:
        for replicate in range(replicates):
            seed = stream_seed(study.STUDY, "smoke_primary", scenario, replicate)
            frame, truths = study.draw_from_seed(scenario, study.PRIMARY_N, seed)
            rows += study.primary_rows(frame, truths, scenario, replicate)
    return pd.DataFrame(rows).loc[:, list(REPLICATE_COLUMNS)]


def smoke(replicates: int, output: Path, *, write_timings: bool) -> pd.DataFrame:
    """The property smoke and the primary smoke; returns the property summary."""
    study, properties = projection_compositions, projection_compositions_properties
    first = study.GROUPS[0].label
    order = [group.label for group in study.GROUPS]
    with property_smoke.instrumented(
        properties, "_replicate", group_of=lambda key: str(key[0]), fail=(first, 0, "smoke")
    ) as timings:

        def generate(*, n_jobs: int) -> pd.DataFrame:
            return properties.generate_property_rows(n_jobs=1, budget=replicates, stream="smoke")

        summary, failure_files = property_smoke.run(
            properties, study.STUDY, generate=generate, output=output
        )
    primary = smoke_primary(PRIMARY_SMOKE_REPLICATES)
    primary.to_csv(output / "primary-smoke.csv", index=False)
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
            projection_compositions,
            projection_compositions_properties,
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
