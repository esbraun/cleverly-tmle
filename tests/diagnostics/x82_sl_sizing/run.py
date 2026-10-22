"""The sizing probe of ``refuter-calibration-longitudinal``: each defect on large samples.

:mod:`tests.unit.test_refuter_calibration_longitudinal_design` reads every power-cell size of the
declaration off this probe (X82 plan 8.2, AGENT-BRIEF "Power-cell sizing"). The probe fits each
cell's own declared estimator (:func:`tests.studies.refuter_calibration_longitudinal.fit_cell`)
on samples of :data:`SIZE` rows, on seeds outside every declared stream, so it cannot change a
verdict.

==========================  =====================================================================
quantity                    what one draw records
==========================  =====================================================================
``stale``                   the refit on one stale-row placebo (the study's own
                            :func:`~tests.studies.refuter_calibration_longitudinal.stale_rows`
                            at the declared share)
``control``                 the refit on the valid control ``NC`` (law E only)
``confounded``              the refit on the confounded control ``NCU`` (law E only)
==========================  =====================================================================

Each row records the declared stale share and control strength it ran at, the refit's
estimate, and its standard error rescaled to the study's
``n = 1,000``, ``se_at_n = SE sqrt(SIZE / n)``, and their ratio ``shift``. For ``stale`` the
ratio is the population shift of the placebo in units of its SD at ``n``; for ``confounded`` it
is the adjusted contrast of the control in control SE at ``n``. One draw's ratio carries a Monte
Carlo SD of about ``sqrt(n / SIZE) = 0.071``.

    python -m tests.diagnostics.x82_sl_sizing.run

writes ``reading.csv`` beside this file.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cleverly.validation._markov_placebo import MarkovKernel
from tests.studies import refuter_calibration_longitudinal as study

HERE = Path(__file__).resolve().parent
#: The rows of each probe sample.
SIZE = 200_000
#: Independent probe samples per cell.
DRAWS = 4
SEED_BASE = 820_000_000


def _record(cell: str, quantity: str, draw: int, refit: Any) -> dict[str, Any]:
    spec = study.CELLS[cell]
    estimate = refit[spec.estimand]
    se_at_n = float(estimate.std_error) * np.sqrt(SIZE / study.N)
    return {
        "cell": cell,
        "quantity": quantity,
        "draw": draw,
        "size": SIZE,
        "n": study.N,
        "stale_share": study.STALE_SHARE[spec.law],
        "control_delta": study.CONTROL_DELTA if spec.law == "end_of_study" else float("nan"),
        "estimate": float(estimate.psi),
        "se_at_n": se_at_n,
        "shift": float(estimate.psi) / se_at_n,
    }


def probe(cell: str, draw: int) -> list[dict[str, Any]]:
    """Every quantity of one cell on one probe sample."""
    spec = study.CELLS[cell]
    seed = SEED_BASE + 1_000 * list(study.CELLS).index(cell) + draw
    frame = study.draw_law(spec.law, SIZE, seed)
    result = study.fit_cell(cell, frame, seed)
    data = result.data
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rng = np.random.default_rng(seed + 1)
        kernel = MarkovKernel.fit(data)
        with study.stale_rows(study.STALE_SHARE[spec.law]):
            placebo = kernel.placebo(data, rng)
        stale = result.estimator.refit(data.with_treatment(placebo), random_state=seed + 2)
        rows.append(_record(cell, "stale", draw, stale))
        if "negative_control" in spec.readings:
            for quantity, column in (("control", "NC"), ("confounded", "NCU")):
                values = frame[column].to_numpy(dtype=float)
                refit = result.estimator.refit(
                    data.with_outcome(values, name=column), random_state=seed + 3
                )
                rows.append(_record(cell, quantity, draw, refit))
    return rows


def main() -> None:
    """Run every cell and draw, and write ``reading.csv``."""
    rows = []
    for cell in study.CELLS:
        for draw in range(DRAWS):
            start = time.perf_counter()
            rows += probe(cell, draw)
            print(f"{cell} draw {draw}: {time.perf_counter() - start:.1f} s", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "reading.csv", index=False, float_format="%.6f", lineterminator="\n")
    print(table.groupby(["cell", "quantity"])["shift"].mean().to_string())


if __name__ == "__main__":
    main()
