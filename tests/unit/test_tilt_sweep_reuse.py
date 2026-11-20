"""X106 G4: the tilt sweep builds its gamma-invariant inputs once and evaluates each iterate once.

Both changes are equivalent implementations (mode E, bitwise).  The solve is compared byte for
byte with the pre-G4 loop, kept verbatim below as the reference; the shared inputs are compared
with a fresh construction at every gamma of a sweep, on the constructions that read them
differently (conditional contrasts under fold evaluation, three arms with a declared direction,
several cross-fitting draws).  Each guard has a mutation control.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import cleverly.sensitivity._tilt_targeting as tilt_module
from cleverly.datasets import make_missing_outcome_binary
from cleverly.estimators import TMLE
from cleverly.sensitivity._tilt_targeting import (
    TiltInputs,
    TiltSolve,
    solve_tilt,
    targeted_tilt_estimates,
)
from cleverly.sensitivity.missingness import _tilt_direction, _tilt_targets, tipping_gamma
from cleverly.utils.bounds import expit, logit

COVARIATES = ("W1", "W2", "W3")


def _reference_solve_tilt(
    scaled: Any,
    observed: Any,
    received: Any,
    weights: Any,
    outcome: Any,
    mechanism: Any,
    gamma: Any,
    arm_weights: Any,
    *,
    tol: float = 1e-10,
) -> TiltSolve:
    """``solve_tilt`` as it stood before X106 G4, which evaluated each iterate three times."""
    edge = tilt_module._EDGE
    q = np.clip(np.asarray(outcome, dtype=float), edge, 1.0 - edge)
    p = np.clip(np.asarray(mechanism, dtype=float), edge, 1.0 - edge)
    b = np.asarray(arm_weights, dtype=float)
    g = np.asarray(gamma, dtype=float)[None, :]
    a = np.asarray(received, dtype=int)
    w = np.asarray(weights, dtype=float)
    d = np.asarray(observed, dtype=bool)
    y = np.where(d, np.asarray(scaled, dtype=float), 0.0)
    m = q.shape[0]
    at = np.arange(m)
    delta = d.astype(float)

    def covariates(q: Any, p: Any) -> tuple[Any, Any]:
        h = expit(logit(q) + g)
        kappa = p + (1.0 - p) * h * (1.0 - h) / (q * (1.0 - q))
        return b * (kappa / p)[:, :, None], b * (q - h)[:, :, None]

    def score(q: Any, p: Any) -> tuple[Any, Any, float]:
        hq, hp = covariates(q, p)
        own_q, own_p = hq[at, a], hp[at, a]
        total = (w * delta * (y - q[at, a])) @ own_q / m + (w * (delta - p[at, a])) @ own_p / m
        scale = np.maximum(
            (np.abs(w * delta) @ np.abs(own_q) + np.abs(w) @ np.abs(own_p)) / m, 1e-300
        )
        return total, scale, float(np.max(np.abs(total) / scale, initial=0.0))

    converged = False
    iteration = 0
    _, _, relative = score(q, p)
    while iteration < tilt_module._MAX_ITER:
        if relative <= tol:
            converged = True
            break
        iteration += 1
        hq, hp = covariates(q, p)
        own_q, own_p = hq[at, a], hp[at, a]
        vq = w * delta * q[at, a] * (1.0 - q[at, a])
        vp = w * p[at, a] * (1.0 - p[at, a])
        gram = ((own_q * vq[:, None]).T @ own_q + (own_p * vp[:, None]).T @ own_p) / m
        total, _, _ = score(q, p)
        step = np.asarray(np.linalg.lstsq(gram, total, rcond=None)[0], dtype=float)
        q = np.clip(expit(logit(q) + hq @ step), edge, 1.0 - edge)
        p = np.clip(expit(logit(p) + hp @ step), edge, 1.0 - edge)
        _, _, relative = score(q, p)
    else:
        converged = relative <= tol
    return TiltSolve(q, p, converged, iteration, relative)


def _solve_inputs(seed: int, n: int, k: int, j: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    received = rng.integers(0, k, size=n)
    observed = rng.uniform(size=n) < 0.7
    return {
        "scaled": rng.uniform(size=n),
        "observed": observed,
        "received": received,
        "weights": rng.uniform(0.3, 2.0, size=n),
        "outcome": rng.uniform(0.05, 0.95, size=(n, k)),
        "mechanism": rng.uniform(0.3, 0.95, size=(n, k)),
        "gamma": rng.normal(scale=1.5, size=k),
        "arm_weights": rng.normal(size=(n, k, j)),
    }


@pytest.mark.parametrize(
    ("seed", "n", "k", "j", "tol"),
    [(0, 400, 2, 2, 1e-10), (1, 257, 3, 3, 1e-10), (2, 300, 2, 1, 1e-12), (3, 200, 3, 2, 0.0)],
)
def test_the_solve_is_bitwise_the_pre_g4_loop(
    seed: int, n: int, k: int, j: int, tol: float
) -> None:
    # tol = 0 never converges, so the last case runs every one of the 200 steps.
    inputs = _solve_inputs(seed, n, k, j)
    with np.errstate(all="ignore"):
        new = solve_tilt(**inputs, tol=tol)
        old = _reference_solve_tilt(**inputs, tol=tol)
    assert new.outcome.tobytes() == old.outcome.tobytes()
    assert new.mechanism.tobytes() == old.mechanism.tobytes()
    assert (new.converged, new.n_iter) == (old.converged, old.n_iter)
    assert np.float64(new.score).tobytes() == np.float64(old.score).tobytes()
    assert new.n_iter > 0


def test_the_solve_evaluates_each_iterate_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # One expit per evaluation of the covariates and two per update: 1 + 3 * n_iter.  The
    # pre-G4 loop evaluated each iterate three times, 1 + 5 * n_iter.
    calls = [0]

    def counted(x: Any) -> Any:
        calls[0] += 1
        return expit(x)

    inputs = _solve_inputs(5, 300, 2, 2)
    monkeypatch.setattr(tilt_module, "expit", counted)
    solved = solve_tilt(**inputs)
    assert solved.n_iter >= 2
    assert calls[0] == 1 + 3 * solved.n_iter


def _fit(**options: Any) -> Any:
    frame, _ = make_missing_outcome_binary(n=500, seed=17)
    if options.pop("three_arm", False):
        rng = np.random.default_rng(17)
        frame = frame.assign(A=rng.integers(0, 3, size=len(frame)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                outcome_learner=LogisticRegression(max_iter=1000),
                treatment_learner=LogisticRegression(max_iter=1000),
                missingness_learner=LogisticRegression(max_iter=1000),
                cross_fit=True,
                simultaneous=False,
                random_state=0,
                **options,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES, delta="Delta")
            .single()
        )


def _digest(estimates: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    return {
        name: (
            np.float64(estimate.psi).tobytes(),
            np.asarray(estimate.influence_curve, dtype=float).tobytes(),
            np.float64(estimate.std_error).tobytes(),
            np.asarray(estimate.ci, dtype=float).tobytes(),
        )
        for name, estimate in estimates.items()
    }


_SWEEP = (-3.0, -0.5, 0.25, 1.0, 4.0)


@pytest.mark.parametrize(
    "options",
    [
        {"estimands": ("ate", "att", "atc", "rr", "ey_obs", "par")},
        {"estimands": ("ate", "att"), "cv_evaluation": True},
        {"three_arm": True},
        {"repeats": 2},
    ],
    ids=["contrasts", "fold-evaluated", "three-arm", "repeats"],
)
def test_shared_inputs_give_the_fresh_estimate_at_every_gamma(options: dict[str, Any]) -> None:
    result = _fit(**options)
    targets = [
        target
        for target in _tilt_targets(result).values()
        if target.kind in tilt_module.INFERENTIAL_KINDS
    ]
    levels = list(result.data.treatment_levels)
    arm_gamma = {level: (1.0 if i == 0 else -0.5 - i) for i, level in enumerate(levels)}
    direction = _tilt_direction(result.data, arm_gamma if options.get("three_arm") else None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for repeat in result.repeats:
            shared = TiltInputs(result, repeat)
            for gamma in _SWEEP:
                fresh, fresh_ok = targeted_tilt_estimates(result, repeat, targets, gamma, direction)
                reused, reused_ok = targeted_tilt_estimates(
                    result, repeat, targets, gamma, direction, shared
                )
                assert fresh_ok == reused_ok
                assert _digest(fresh) == _digest(reused)


def test_inputs_of_another_draw_change_the_estimate() -> None:
    # Mutation control for the per-draw key: reading draw 0's inputs on draw 1 moves the
    # estimate, so a cache keyed on anything coarser than the draw would be seen.
    result = _fit(repeats=2)
    targets = [_tilt_targets(result)["ate"]]
    direction = dict.fromkeys(result.data.arm_codes, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrong = TiltInputs(result, result.repeats[0])
        right, _ = targeted_tilt_estimates(result, result.repeats[1], targets, 1.0, direction)
        mixed, _ = targeted_tilt_estimates(
            result, result.repeats[1], targets, 1.0, direction, wrong
        )
    assert _digest(right) != _digest(mixed)


def test_the_received_column_matches_the_per_row_lookup() -> None:
    result = _fit(three_arm=True)
    repeat = result.repeats[0]
    inputs = TiltInputs(result, repeat)
    column = {arm: tuple(repeat.nuisance.arms).index(arm) for arm in repeat.nuisance.arms}
    expected = np.array(
        [column[float(code)] for code in np.asarray(result.data.treatment, dtype=float)],
        dtype=int,
    )
    assert inputs.received.dtype == expected.dtype
    assert np.array_equal(inputs.received, expected)


def test_a_search_builds_each_draws_inputs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    # The probe grid has about a thousand points and the roots more; the gamma-invariant
    # inputs are built once per draw for all of them.
    result = _fit(repeats=2)
    built = [0]
    original = TiltInputs.__init__

    def counted(self: Any, *args: Any, **kwargs: Any) -> None:
        built[0] += 1
        original(self, *args, **kwargs)

    monkeypatch.setattr(TiltInputs, "__init__", counted)
    levels = list(result.data.treatment_levels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tipping_gamma(result, arm_gamma={levels[0]: 1.0, levels[1]: -0.5})
    assert built[0] == len(result.repeats)
