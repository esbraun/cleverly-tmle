"""Regenerate the score-assessment-compositions evidence (P1 PR 2) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory. The study has no
comparator, so the run starts no container. The groups, sizes, learners, controls and design
numbers are those of :mod:`tests.studies.score_assessment_compositions`, fixed before the run,
and the files the run writes are :data:`tests.studies.score_assessment_compositions.RUN_FILES`.

A smoke passes ``--property-smoke N --output <scratch>``. It draws ``N`` replicates of every
group on the ``smoke`` stream and two primary replicates on ``smoke_primary``, none of which a
declared run or the pilot draws, and runs the shared driver's property step on them
(:mod:`tests.canonical.property_smoke`). With ``--write-timings`` it writes each group's
seconds per fit to :data:`tests.studies.score_assessment_compositions.SMOKE_TIMINGS`, which the
design test reads against the declared seconds with the cost gate.

Red-cell readings, declared before any pilot or run (P1 PR 2 plan rev 3, sections 5.0, 5.5 and
5B). A red cell is diagnosed before a route is chosen; no cell is re-sized, re-drawn or
downgraded after the run.

* A red calibration row is finite-sample only when its 99% SE-ratio interval contains 1, its
  coverage interval contains 0.95 and the diagnosis finds no defect. It then goes to
  ``reporting`` with the owner P1 (F42 after PR 4). Every other red row is a defect suspect.
* Pilot ratios off 1, on either side. ``Margins`` is two-sided because a departure is a defect,
  not a regime. An arm whose pilot 99% interval excludes 1, wholly above or wholly below, gets a
  targeted diagnosis before the declaration. Candidate causes above 1: a cluster sum plus row
  terms counted twice, weights entering squared, the variance of an in-sample spline fit.
  Below 1: a missing curve term, a fold covariance, the in-sample shortfall of the P1 pilot
  diagnosis. The shipped calibration rows read 1.02 to 1.03, so the above-band reading is the
  likely one here. The point-reading fallback (:mod:`tests.studies.point_reading`, family
  target 0.95) may absorb a pilot point within [0.97, 1.03] only after that diagnosis finds no
  defect. A confirmed departure beyond goes to ``reporting`` with the owner P1 before the run;
  buying replications is not the remedy.
* No clustered red-cell reading: at 800 clusters the sandwich's small-sample factor
  ``G / (G - 1)`` is about 1.001, so a red ``assess_clustered_4k`` row is a defect suspect.
* If either clustered control row's pilot row-level SE ratio exceeds 0.85, the clustered law is
  too weak to show that the cluster sum matters: the cluster effect is raised before the
  declaration, and the control is not dropped.
* A control that does not detect (``shrunken_se_control`` or ``row_variance_control`` inside or
  above the band) is a red of the study, not of the estimator: the cell it guards cannot see
  the component it is named for, and that cell's claim is withheld until the control is
  repaired under a fresh declaration.
* The primary phase (the weighted paired row) red on its coverage floor while the
  ``assess_weighted_2k`` risk rows are green reads as the primary phase's finite-sample noise
  and is reported; red with a red risk row reads with that row.
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
from tests.studies import score_assessment_compositions, score_assessment_compositions_properties
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import stream_seed

#: Primary smoke draws.
PRIMARY_SMOKE_REPLICATES = 2


def smoke_primary(replicates: int) -> pd.DataFrame:
    """The primary rows of ``replicates`` draws on the ``smoke_primary`` stream."""
    study = score_assessment_compositions
    rows = []
    for scenario in study.SCENARIOS:
        for replicate in range(replicates):
            seed = stream_seed(study.STUDY, "smoke_primary", scenario, replicate)
            frame, truths = study.draw_from_seed(scenario, study.PRIMARY_N, seed)
            rows += study.primary_rows(frame, truths, scenario, replicate)
    return pd.DataFrame(rows).loc[:, list(REPLICATE_COLUMNS)]


def smoke(replicates: int, output: Path, *, write_timings: bool) -> pd.DataFrame:
    """The property smoke and the primary smoke; returns the property summary."""
    study, properties = score_assessment_compositions, score_assessment_compositions_properties
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
            score_assessment_compositions,
            score_assessment_compositions_properties,
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
