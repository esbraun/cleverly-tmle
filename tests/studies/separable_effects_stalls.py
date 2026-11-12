"""Refit the declared run's investigated stalls and read where their remaining score lies.

The declared ``separable-effects`` run flagged three cross-fitted fits that stopped with
``failure="newton_stalled"`` at 1.2 to 2.4 times ``tol``
(:data:`tests.studies.separable_effects.INVESTIGATED_STALLS`).  For each, this refits the draw,
computes the Jacobian of the factor scores at the reported state by central differences, as the
Newton phase does, and splits the remaining score along the Jacobian's left singular vectors.
It then forces Newton steps along every direction (a longer difference and no cut-off) until the
relative score falls to ``tol``, and compares every estimate at that root with the reported one,
in units of its standard error.

The finding (:data:`~tests.studies.separable_effects.STALL_FINDING`) holds when the remaining
score lies in two directions whose singular values are below ``1e-10`` of the largest, which
the Newton phase's cut-off of ``1e-10`` discards, and when the root within ``tol`` that the
forced steps reach moves no estimate by more than :data:`SHIFT` standard errors.

Run it as ``python -m tests.studies.separable_effects_stalls --jobs N``.  It prints one line per
fit and exits with status 1 when any part of the finding fails for a fit.
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
#: The share of the remaining squared score that must lie in those directions, the share a
#: direction must carry to count as one the score lies in, and how many there must be.
SHARE = 0.99
CARRIES = 0.01
DIRECTIONS = 2
#: The forced Newton steps: their central-difference length, their cut-off, and their number.
FORCED_STEP = 1e-4
FORCED_RCOND = 1e-15
FORCED_STEPS = 12
#: The largest move of an estimate to the forced root, in standard errors, that the finding
#: allows.
SHIFT = 0.01
#: How closely a refit must reproduce the recorded relative score.
RECORDED_RTOL = study.STALL_RTOL


def _increments(state: Any, covariates: Any, targets: Any, weights: np.ndarray) -> Any:
    delta_d = np.zeros_like(state.lam_d)
    delta_y = np.zeros_like(state.h_y)
    for k, weight in enumerate(weights):
        m, factor = divmod(k, 2)
        level = targets[m].levels[factor]
        (delta_d if factor == 0 else delta_y)[:, :, level] += weight * covariates[m][factor]
    return delta_d, delta_y


def score_jacobian(
    state: Any, inputs: Any, targets: Any, step: float = separable_module._JACOBIAN_STEP
) -> np.ndarray:
    """The central-difference Jacobian of the ``2M`` factor scores, as the Newton phase reads it."""
    covariates = [separable_module._covariates(state, inputs, target) for target in targets]
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
    left, singular, _ = np.linalg.svd(score_jacobian(state, inputs, targets))
    along = (left.T @ score) ** 2
    unresolved = singular < SINGULAR * singular[0]
    carrying = along >= CARRIES * along.sum()
    levels, curves = separable_module.influence(state, inputs, targets)
    error = curves.std(axis=1) / np.sqrt(curves.shape[1])
    forced = state
    for _ in range(FORCED_STEPS):
        forced_score, forced_scale = separable_module.scores(forced, inputs, targets)
        if relative_score(forced_score, forced_scale) <= record.tol:
            break
        jacobian = score_jacobian(forced, inputs, targets, FORCED_STEP)
        weights = np.linalg.lstsq(jacobian, -forced_score, rcond=FORCED_RCOND)[0]
        covariates = [separable_module._covariates(forced, inputs, t) for t in targets]
        forced = forced.moved(*_increments(forced, covariates, targets, weights), inputs.bounds)
    forced_score, forced_scale = separable_module.scores(forced, inputs, targets)
    moved, _ = separable_module.influence(forced, inputs, targets)
    return {
        "cell": cell,
        "replicate": replicate,
        "failure": record.failure,
        "relative_score": relative_score(score, scale),
        "recorded": study.INVESTIGATED_STALLS[key],
        "directions": int(np.sum(carrying)),
        "carried_unresolved": bool(np.all(unresolved[carrying])),
        "unresolved_share": float(along[unresolved].sum() / along.sum()),
        "gap": float(singular[~unresolved].min() / singular[unresolved & carrying].max()),
        "forced_score": relative_score(forced_score, forced_scale),
        "tol": record.tol,
        "shift_se": float(np.max(np.abs(moved - levels) / error)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args(argv)
    keys = list(study.INVESTIGATED_STALLS)
    table = pd.DataFrame(map_parallel(_refit, [(key,) for key in keys], n_jobs=arguments.jobs))
    print(table.to_string(index=False))
    again = table["failure"].eq("newton_stalled") & np.isclose(
        table["relative_score"], table["recorded"], rtol=RECORDED_RTOL, atol=0.0
    )
    explained = (
        again
        & table["directions"].eq(DIRECTIONS)
        & table["carried_unresolved"]
        & (table["unresolved_share"] >= SHARE)
        & (table["forced_score"] <= table["tol"])
        & (table["shift_se"] <= SHIFT)
    )
    bad = int((~explained).sum())
    print(f"{len(table)} investigated stalls refitted; {bad} not explained")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
