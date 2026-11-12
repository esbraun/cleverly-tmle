"""Refit every fit that read ``cell_separated`` and check that separation explains it.

A fit reads ``cell_separated`` when rows the clip holds carry most of its remaining score, read
as the net residual of their cells.  Separation explains that only when no clipped logit sits on
a row that pulls it inward: an event of that cause at the lower clip, or a row without one at the
upper clip.  Such a row is the defect fixed in the iterative TMLE's inner fit (plan
``X91-D-stall-plan.md`` rev 2.1), and the declared readings of
``tests/canonical/separable_effects/regenerate.py`` name it a defect suspect.

Run it as ``python -m tests.studies.separable_effects_separation pilot --jobs N`` on the rerun
pilot, or ``python -m tests.studies.separable_effects_separation run DIR --jobs N`` on a declared
run's output directory (its ``convergence.csv`` and ``property-replicates.parquet``).  It prints
one line per fit and exits with status 1 when any fit has an inward-pulling clipped row.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.longitudinal import separable as separable_module
from cleverly.utils.parallel import map_parallel
from tests.studies import separable_effects as study
from tests.studies import separable_effects_law as law
from tests.studies import separable_effects_pilot as pilot
from tests.studies import separable_effects_properties as properties
from tests.studies.evidence.seeds import stream_seed

#: ``(source, cell, replicate)``: where the fit's draw comes from.
Job = tuple[str, str, int]

#: A logit this close to the clip counts as at the clip.
AT_CLIP = 1e-9


def inward_rows(state: Any, inputs: Any, targets: Any) -> int:
    """The rows whose logit sits at the clip although their outcome pulls it inward.

    Only the levels some target moves count: a factor at a level no target reads keeps its
    start, which may sit at the clip, and enters no score.
    """
    low, high = inputs.bounds
    rows, observed = inputs.rows, inputs.observed
    count = 0
    for factor, beta, events, factor_rows in (
        (0, state.beta_d, inputs.event_d, inputs.rows_d()),
        (1, state.beta_y, inputs.event_y, inputs.rows_y()),
    ):
        moved = np.isin(observed, [target.levels[factor] for target in targets])[:, None]
        logit = beta[rows, :, observed]
        finite = np.isfinite(logit) & factor_rows & moved
        count += int(np.sum(finite & (logit <= low + AT_CLIP) & (events == 1.0)))
        count += int(np.sum(finite & (logit >= high - AT_CLIP) & (events == 0.0)))
    return count


def _refit(job: Job) -> dict[str, Any]:
    source, cell, replicate = job
    captured: list[tuple[Any, Any, Any]] = []
    walk = separable_module.walk_separable

    def recording(initial: Any, inputs: Any, targets: Any, **settings: Any) -> Any:
        state, record = walk(initial, inputs, targets, **settings)
        captured.append((state, inputs, targets))
        return state, record

    separable_module.walk_separable = recording
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if source == "pilot":
                scenario, n, _, robustness = pilot.PILOTS[cell]
                frame = law.sample(n, stream_seed(study.STUDY, "pilot", cell, replicate))
                if robustness:
                    properties.robustness_fit(frame, robustness)
                else:
                    study.fit(frame, scenario)
            elif source == "primary":
                frame, _ = study.draw_scenario(cell, study.PRIMARY_N, replicate)
                study.fit(frame, cell)
            elif cell in properties.ROBUSTNESS_CELLS:
                frame = properties._draw((properties.ROBUSTNESS, cell, replicate, 0))
                properties.robustness_fit(frame, cell)
            else:
                label = cell.removesuffix("__correctly_specified")
                frame = properties._draw((properties.CALIBRATION, label, replicate, 0))
                study.fit(frame, properties.CALIBRATION_CELLS[label][0])
    finally:
        separable_module.walk_separable = walk
    inward = sum(inward_rows(state, inputs, targets) for state, inputs, targets in captured)
    return {"source": source, "cell": cell, "replicate": replicate, "inward_rows": inward}


def pilot_jobs() -> list[Job]:
    """The rerun pilot's ``cell_separated`` fits."""
    rows = pd.read_csv(pilot.OUTPUT, low_memory=False)
    fits = rows.drop_duplicates(["label", "replicate"])
    separated = fits.loc[fits["failure"].fillna("").eq("cell_separated")]
    return [("pilot", str(r.label), int(r.replicate)) for r in separated.itertuples()]


def run_jobs(directory: Path) -> list[Job]:
    """A declared run's ``cell_separated`` primary and property fits."""
    convergence = pd.read_csv(directory / "convergence.csv")
    primary = convergence.loc[convergence["failure"].fillna("").eq("cell_separated")]
    jobs: list[Job] = [("primary", str(r.scenario), int(r.replicate)) for r in primary.itertuples()]
    rows = pd.read_parquet(directory / "property-replicates.parquet")
    fitted = rows.loc[rows["cell"].isin(list(properties.CROSS_FITTED_CELLS))]
    separated = fitted.loc[fitted["failure"].fillna("").eq("cell_separated")]
    jobs += [("property", str(r.cell), int(r.replicate)) for r in separated.itertuples()]
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["pilot", "run"])
    parser.add_argument("directory", nargs="?", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args(argv)
    if arguments.source == "pilot":
        jobs = pilot_jobs()
    else:
        if arguments.directory is None:
            parser.error("run needs the declared run's output directory")
        jobs = run_jobs(arguments.directory)
    outcomes = map_parallel(_refit, [(job,) for job in jobs], n_jobs=arguments.jobs)
    table = pd.DataFrame(outcomes, columns=["source", "cell", "replicate", "inward_rows"])
    print(table.to_string(index=False))
    bad = int((table["inward_rows"] > 0).sum())
    print(f"{len(table)} cell_separated fits refitted; {bad} with inward-pulling clipped rows")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
