"""Refit the declared run's investigated stalls and read where their remaining score lies.

The declared ``separable-effects`` run flagged three cross-fitted fits that stopped with
``failure="newton_stalled"`` at 1.2 to 2.4 times ``tol``
(:data:`tests.studies.separable_effects.INVESTIGATED_STALLS`).  For each, this refits the draw,
computes the Jacobian of the factor scores at the reported state by central differences, as the
Newton phase does, and splits the remaining score along the Jacobian's left singular vectors.
The finding (:data:`~tests.studies.separable_effects.STALL_FINDING`) holds when nearly all of
the remaining score lies in directions whose singular value is below ``1e-10`` of the largest:
directions the Newton phase cannot resolve, so the stall is set by rounding and near
collinearity, not by a defect of the targeting.

Run it as ``python -m tests.studies.separable_effects_stalls --jobs N``.  It prints one line per
fit and exits with status 1 when a fit does not stall again or when less than 99% of its
remaining score lies in those directions.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd

from cleverly.fluctuation._score import relative_score
from cleverly.longitudinal import separable as separable_module
from cleverly.utils.parallel import map_parallel
from tests.studies import separable_effects as study
from tests.studies import separable_effects_properties as properties

#: A singular value below this share of the largest counts as singular to rounding.
SINGULAR = 1e-10
#: The share of the remaining squared score that must lie in those directions.
SHARE = 0.99


def score_jacobian(state: Any, inputs: Any, targets: Any) -> np.ndarray:
    """The central-difference Jacobian of the ``2M`` factor scores, as the Newton phase reads it."""
    covariates = [separable_module._covariates(state, inputs, target) for target in targets]
    step = separable_module._JACOBIAN_STEP
    columns = []
    for m, target in enumerate(targets):
        for factor in (0, 1):
            level = target.levels[factor]
            delta = np.zeros_like(state.lam_d if factor == 0 else state.h_y)
            delta[:, :, level] = step * covariates[m][factor]
            zero = np.zeros_like(delta)
            pair = (delta, zero) if factor == 0 else (zero, delta)
            up = state.moved(pair[0], pair[1], inputs.bounds)
            down = state.moved(-pair[0], -pair[1], inputs.bounds)
            columns.append(
                (
                    separable_module.scores(up, inputs, targets)[0]
                    - separable_module.scores(down, inputs, targets)[0]
                )
                / (2.0 * step)
            )
    return np.stack(columns, axis=1)


def _refit(key: tuple[str, int]) -> dict[str, Any]:
    cell, replicate = key
    captured: list[tuple[Any, Any, Any, Any]] = []
    walk = separable_module.walk_separable

    def recording(initial: Any, inputs: Any, targets: Any, **settings: Any) -> Any:
        state, record = walk(initial, inputs, targets, **settings)
        captured.append((state, inputs, targets, record))
        return state, record

    separable_module.walk_separable = recording
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cell in study.SCENARIOS:
                frame, _ = study.draw_scenario(cell, study.PRIMARY_N, replicate)
                study.fit(frame, cell)
            else:
                label = cell.removesuffix("__correctly_specified")
                frame = properties._draw((properties.CALIBRATION, label, replicate, 0))
                study.fit(frame, properties.CALIBRATION_CELLS[label][0])
    finally:
        separable_module.walk_separable = walk
    (state, inputs, targets, record), *_ = captured
    score, scale = separable_module.scores(state, inputs, targets)
    jacobian = score_jacobian(state, inputs, targets)
    left, singular, _ = np.linalg.svd(jacobian)
    along = (left.T @ score) ** 2
    unresolved = singular < SINGULAR * singular[0]
    return {
        "cell": cell,
        "replicate": replicate,
        "failure": record.failure,
        "relative_score": relative_score(score, scale),
        "recorded": study.INVESTIGATED_STALLS[key],
        "directions_below": int(np.sum(unresolved & (singular > 0.0))),
        "unresolved_share": float(along[unresolved].sum() / along.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args(argv)
    keys = list(study.INVESTIGATED_STALLS)
    table = pd.DataFrame(map_parallel(_refit, [(key,) for key in keys], n_jobs=arguments.jobs))
    print(table.to_string(index=False))
    again = table["failure"].eq("newton_stalled") & np.isclose(
        table["relative_score"], table["recorded"], rtol=1e-3, atol=0.0
    )
    bad = int((~again | (table["unresolved_share"] < SHARE)).sum())
    print(f"{len(table)} investigated stalls refitted; {bad} not explained by {SHARE:.0%}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
