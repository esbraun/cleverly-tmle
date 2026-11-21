"""Regenerate the projection-methods evidence (P1 PR 3) under the declared run form.

A declared run passes ``--output`` (an empty scratch directory outside the repository) and
optionally ``--jobs`` and ``--fresh``, and nothing else. :mod:`tests.canonical.declared_run`
describes the guard, the run log and the copy into this directory. The study has no
comparator (R ``drtmle`` fits no projection), so the run starts no container. The groups,
sizes, learners, controls and design numbers are those of
:mod:`tests.studies.projection_methods`, fixed before the run, and the files the run writes are
:data:`tests.studies.projection_methods.RUN_FILES`.

A smoke passes ``--property-smoke N --output <scratch>``. It draws ``N`` replicates of every
group on the ``smoke`` stream and two primary replicates on ``smoke_primary``, none of which a
declared run or the pilot draws, and runs the shared driver's property step on them
(:mod:`tests.canonical.property_smoke`). With ``--write-timings`` it writes each group's
seconds per fit to :data:`tests.studies.projection_methods.SMOKE_TIMINGS`, which the design
test reads against the declared seconds with the cost gate.

Red-cell readings, declared before any pilot or run (P1 PR 3 plan rev 2, sections 5.0 and
5.5). A red cell is diagnosed before a route is chosen; no cell is re-sized, re-drawn or
downgraded after the run.

* A red calibration row is finite-sample only when its 99% SE-ratio interval contains 1, its
  coverage interval contains 0.95 and the diagnosis finds no defect. It then goes to
  ``reporting`` with the owner P1 (F42 after PR 4). Every other red row is a defect suspect.
* Pilot ratios off 1, on either side. An arm whose pilot 99% interval excludes 1, wholly above
  or wholly below, gets a targeted diagnosis before the declaration. Candidate causes below 1:
  a correction the curve fails to subtract or subtracts unweighted (``phi_j`` outside
  ``Q_r``), equation (9) left unsolved, the declared mechanism read where the tilted one
  belongs. Above 1: a correction subtracted twice, ``D_Y`` kept beside a declared mechanism.
  The point-reading fallback (:mod:`tests.studies.point_reading`, family target 0.95) may
  absorb a pilot point within [0.97, 1.03] only after that diagnosis finds no defect.
* ``dr_glm_4k`` intercept above the band is the predicted conservative case and passes its
  gated ``conservative`` clauses; its slope is gated ``valid``. A miss of either is a defect
  suspect, diagnosed against the quadrature's predicted SE ratios (1.19 and 1.00).
* A ``bias`` row of ``dr_known_4k``, ``dr_both_4k`` or ``ctmle_points_4k`` red reads as a
  defect of the projection's targeting: diagnose against the exact laws of
  ``tests/unit/test_drtmle_projection.py`` and ``tests/unit/test_ctmle_projection.py``.
* A control that does not detect (``shrunken_se_control`` inside or above its band, a
  ``wrong_q_witness`` row inside the bias margin, a ladder rung whose bias interval does not
  clear the margin on the negative side, or a growth interval that reaches 0) is a red of the
  study, not of the estimator: the cell it guards cannot see the component it is named for,
  and that cell's claim is withheld until the control is repaired under a fresh declaration.
* The ``spread`` readings (``d_star_reading``, ``plugin_reading``) publish under the
  diagnostic role and state no verdict.
* The primary phase (``dr_known_4k`` intercept) red on its coverage floor while the
  calibration rows are green reads as the primary phase's finite-sample noise and is
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
from tests.studies import projection_methods, projection_methods_properties
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import stream_seed

#: Primary smoke draws.
PRIMARY_SMOKE_REPLICATES = 2


def smoke_primary(replicates: int) -> pd.DataFrame:
    """The primary rows of ``replicates`` draws of ``dr_known_4k`` on ``smoke_primary``."""
    study = projection_methods
    rows = []
    for scenario in study.SCENARIOS:
        for replicate in range(replicates):
            seed = stream_seed(study.STUDY, "smoke_primary", scenario, replicate)
            frame, truths = study.draw_from_seed(scenario, study.PRIMARY_N, seed)
            rows += study.primary_rows(frame, truths, scenario, replicate)
    return pd.DataFrame(rows).loc[:, list(REPLICATE_COLUMNS)]


def smoke(replicates: int, output: Path, *, write_timings: bool) -> pd.DataFrame:
    """The property smoke and the primary smoke; returns the property summary."""
    study, properties = projection_methods, projection_methods_properties
    first = properties.payloads(1, "smoke")[0][0]
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
            projection_methods,
            projection_methods_properties,
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
