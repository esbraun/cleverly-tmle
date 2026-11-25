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

The first declared run and the second (P1 PR 3 plan rev 3). The first run was declared at
``32f9d8ac`` and ran at ``c566a4c2``. Two gated cells were red:
``dr_glm__intercept__conservative`` (SE ratio 1.108, 99% interval 1.066 to 1.151, which does not
clear the band's 1.07; coverage 0.9675) and ``dr_slow_n1000__intercept__first_order`` (-0.26 SD,
inside the margin). The diagnosis (``rows/P1/pilot-diagnosis-pr3.md``) found no estimator
defect: the ``dr_glm`` cell was sized from a harness of the estimator that solves equations (8)
and (9), not the declared one that also solves (10), and the 1,000-row rung measured an opposing
finite-sample bias rather than the first-order term. The study page publishes both red rows
("The re-declared cells"), and their claims are withheld for that run. The second run fits the
whole study at the rev 3 declaration: ``dr_glm_4k`` at R = 17,600 and the ladder at 4,000,
16,000 and 64,000 rows on the fresh ``property_r3`` stream, every other group and the primary on
the streams the first run drew. Those groups must reproduce the first run's rows and verdicts:
:mod:`tests.canonical.projection_methods.first_run` checks them through the shared driver's
``check_rows`` hook before the manifest is written. Exact equality is expected, a relative
difference up to 1e-9 is noted in ``run.log`` with the stack difference, and anything larger, or
a flipped verdict, stops the run as a defect suspect and publishes nothing. The run replaces
every artifact in this directory, and the ``n1000`` cells cease to exist.

Red-cell readings, declared before any pilot or run (P1 PR 3 plan rev 2, sections 5.0 and
5.5). A red cell is diagnosed before a route is chosen. No cell was to be re-sized, re-drawn or
downgraded after the run. Rev 3 departed from that once, under the design-bug route of
``docs/development/method-benchmarking.md`` ("Red cells"), for ``dr_glm_4k`` and the ladder
only: the first verdicts stand and are published, the draws are new, R follows the declared
sizing rule at a lower planning value, and no threshold, band or rule moved.

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
  suspect, diagnosed against the planning ratio 1.105 (pooled finite-n pilot reading 1.108)
  and 1.00, beside the declared-guard fits at n = 200,000 (1.108 to 1.148).
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
  buys a pass. Rev 3's redesign of ``dr_glm_4k`` and the ladder, above, is the one departure.

Readings of the second run, declared before it (rev 3 R3.4). The record publishes under
``publication_policy="reporting"``, declared before the run, so a red row publishes red.

* A red rev 3 cell (``dr_glm__intercept__conservative``, ``dr_glm__W1__valid``, any
  ``dr_slow_n{4000,16000,64000}`` rung or either growth cell) publishes red, and its claim is
  withheld in the docs and the evidence grid. Its owner goes in ``docs/roadmap.md`` under
  ``reporting`` (P1, then F42 after PR 4) and in ``OWNERS`` of
  ``tests/studies/evidence/red_cells.py``, which ``tests/unit/test_red_cell_ledger.py`` requires
  for every red row of a registered study. PR 3 does not resize, redraw or re-ladder that cell
  again: there is no rev 4 for these cells.
* A red on an unchanged group cannot occur without a determinism failure, which stops the run
  before it publishes.
* Over the two attempts at each re-declared claim the false-pass rate at the boundary is at
  most about 1% (0.5% each). The predictive pass probability of the redesigned cells together
  is about 0.93 x 0.97, roughly 0.90.
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
