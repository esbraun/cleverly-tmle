"""The Cox fit builds its information in blocks and halves steps on the likelihood alone.

Both are exact rewrites: every float is the same operation in the same order as the one-array
construction, so the fitted model is byte-identical. The reference below is that construction,
kept here so the comparison does not depend on the code under test.
"""

from __future__ import annotations

import tracemalloc
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from cleverly.exceptions import ConvergenceWarning
from cleverly.survival import CoxHazard, cox
from cleverly.survival.cox import _risk_sets, _RiskSets, _tail

Parts = tuple[float, np.ndarray, np.ndarray]


def _reference_parts(beta: np.ndarray, Z: np.ndarray, sets: _RiskSets) -> Parts:
    """The one-array construction: the ``(n, q, q)`` products in a single running sum."""
    eta = Z @ beta
    shift = float(np.max(eta)) if eta.size else 0.0
    risk = np.exp(eta - shift)
    zs = Z[sets.order]
    rs = risk[sets.order]
    s0 = _tail(rs, sets.start)
    s1 = _tail(rs[:, None] * zs, sets.start)
    s2 = _tail(rs[:, None, None] * zs[:, :, None] * zs[:, None, :], sets.start)
    zbar = s1 / s0[:, None]
    loglik = float(np.sum(eta[sets.event_sum_rows]) - np.sum(sets.deaths * (np.log(s0) + shift)))
    grad = Z[sets.event_sum_rows].sum(axis=0) - (sets.deaths[:, None] * zbar).sum(axis=0)
    info = (
        sets.deaths[:, None, None] * (s2 / s0[:, None, None] - zbar[:, :, None] * zbar[:, None, :])
    ).sum(axis=0)
    return loglik, grad, info


def _sample(
    n: int = 600, p: int = 4, seed: int = 20261115, ties: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.binomial(1, 0.5, n), rng.normal(size=(n, p - 1))])
    T = rng.exponential(1 / (0.1 * np.exp(0.3 * X[:, 1] - 0.4 * X[:, 0])))
    C = rng.exponential(1 / 0.05, n)
    time = np.minimum(T, C)
    if ties:
        time = np.round(time, 1)
    return X, time, (T <= C).astype(float)


def _state(model: CoxHazard, X: np.ndarray) -> tuple[bytes, ...]:
    grid = np.linspace(0.0, float(np.nanmax(model.baseline_times_)), 9)
    with np.errstate(all="ignore"):  # a boundary fit's hazards overflow, identically
        hazards = model.cumulative_hazard(X[:25], grid)
    return (
        model.coef_.tobytes(),
        np.float64(model.intercept_shift_).tobytes(),
        model.baseline_times_.tobytes(),
        model.baseline_jumps_.tobytes(),
        bytes([model.converged_]),
        hazards.tobytes(),
    )


def _fit(
    model: CoxHazard,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    parts: Callable[..., Parts] | None = None,
) -> tuple[CoxHazard, list[tuple[str, str]]]:
    """Fit, with the reference construction in place of both likelihood functions if asked."""
    with monkeypatch.context() as patch:
        if parts is not None:
            patch.setattr(cox, "_loglik_parts", parts)
            patch.setattr(cox, "_loglik", lambda beta, Z, sets: parts(beta, Z, sets)[0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, time, event)
    return model, [(item.category.__name__, str(item.message)) for item in caught]


def _separated() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Failure order is a monotone function of one covariate: its coefficient runs to a
    # monotone-likelihood boundary, step halving cannot keep the likelihood finite, and the
    # fit warns.
    rng = np.random.default_rng(0)
    n = 200
    W = rng.normal(size=n)
    X = np.column_stack([rng.binomial(1, 0.5, n), W])
    return X, np.exp(-5.0 * W), np.ones(n)


def _heavy_tailed() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A Cauchy covariate makes a full Newton step overshoot, so at least one halving trial
    # is rejected and the halving loop runs for real.
    rng = np.random.default_rng(5)
    n = 300
    X = np.column_stack([rng.binomial(1, 0.5, n), rng.standard_cauchy(n)])
    T = rng.exponential(1 / (0.1 * np.exp(0.3 * np.clip(X[:, 1], -3, 3) - 0.4 * X[:, 0])))
    C = rng.exponential(20.0, n)
    return X, np.minimum(T, C), (T <= C).astype(float)


def _singular() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A duplicated covariate makes the information singular, so the Newton step takes the
    # least-squares route, which reads both triangles.
    X, time, event = _sample(n=300, p=3, seed=7)
    return np.column_stack([X, X[:, 1]]), time, event


def _with_nan_times() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, time, event = _sample(n=300, p=3, seed=9)
    time = time.copy()
    time[[4, 17]] = np.nan
    return X, time, event


CASES: dict[str, Callable[[], tuple[CoxHazard, tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {
    "main": lambda: (CoxHazard("main"), _sample()),
    "interaction": lambda: (CoxHazard("interaction"), _sample(p=6)),
    "ties": lambda: (CoxHazard("main"), _sample(ties=True)),
    "penalty": lambda: (CoxHazard("interaction", penalty=1.0), _sample(p=5)),
    "treatment": lambda: (CoxHazard("treatment"), _sample()),
    "nan times": lambda: (CoxHazard("main"), _with_nan_times()),
    "boundary": lambda: (CoxHazard("main"), _separated()),
    "halving": lambda: (CoxHazard("main"), _heavy_tailed()),
    "singular": lambda: (CoxHazard("main"), _singular()),
}


@pytest.mark.parametrize("case", list(CASES))
def test_the_fit_is_byte_identical_to_the_one_array_construction(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, (X, time, event) = CASES[case]()
    reference, reference_warnings = _fit(
        CoxHazard(model.terms, model.penalty), X, time, event, monkeypatch, _reference_parts
    )
    fitted, fitted_warnings = _fit(model, X, time, event, monkeypatch)
    assert _state(fitted, X) == _state(reference, X)

    # The package's own warnings are identical. NumPy's floating-point warnings can only be
    # fewer: a halving trial no longer forms the means and second moments it discards, which
    # at a boundary divide zero by zero.
    def own(caught: list[tuple[str, str]]) -> list[tuple[str, str]]:
        return [item for item in caught if item[0] != "RuntimeWarning"]

    assert own(fitted_warnings) == own(reference_warnings)
    assert set(fitted_warnings) <= set(reference_warnings)


def test_the_hard_cases_reach_the_paths_they_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # The boundary fixture must exhaust the halving loop, and the singular one must reach the
    # least-squares step; otherwise their witnesses above would not cover those paths.
    model, (X, time, event) = CASES["boundary"]()
    _, caught = _fit(model, X, time, event, monkeypatch)
    assert not model.converged_
    assert any("monotone-likelihood boundary" in text for _, text in caught)
    calls = []
    original = np.linalg.lstsq

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    model, (X, time, event) = CASES["singular"]()
    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, "lstsq", counting)
        _fit(model, X, time, event, monkeypatch)
    assert calls
    # More trials than accepted steps: some trial was rejected and its step halved.
    counts = {"parts": 0, "trials": 0}
    parts, loglik = cox._loglik_parts, cox._loglik

    def counted_parts(*args: Any) -> Parts:
        counts["parts"] += 1
        return parts(*args)

    def counted_trial(*args: Any) -> float:
        counts["trials"] += 1
        return loglik(*args)

    model, (X, time, event) = CASES["halving"]()
    with monkeypatch.context() as patch:
        patch.setattr(cox, "_loglik_parts", counted_parts)
        patch.setattr(cox, "_loglik", counted_trial)
        _fit(model, X, time, event, monkeypatch)
    assert counts["trials"] > counts["parts"] - 1


@pytest.mark.parametrize("block_bytes", [1, 2, 3, 10**9])
def test_no_block_size_changes_a_byte(block_bytes: int, monkeypatch: pytest.MonkeyPatch) -> None:
    # One row, two rows, a block that does not divide q = 11, and the whole matrix at once.
    X, time, event = _sample(p=6)
    Z = cox._design(X, "interaction")
    sets = _risk_sets(time, event)
    beta = np.random.default_rng(1).normal(scale=0.1, size=Z.shape[1])
    n, q = Z.shape
    monkeypatch.setattr(cox, "_BLOCK_BYTES", block_bytes * 8 * n * q)
    got = cox._loglik_parts(beta, Z, sets)
    want = _reference_parts(beta, Z, sets)
    assert np.float64(got[0]).tobytes() == np.float64(want[0]).tobytes()
    assert got[1].tobytes() == want[1].tobytes()
    assert got[2].tobytes() == want[2].tobytes()
    assert cox._loglik(beta, Z, sets) == got[0]


def _blocked_parts(
    beta: np.ndarray, Z: np.ndarray, sets: _RiskSets, block: int, skip: bool
) -> Parts:
    """A test-local blocked construction; ``skip`` leaves the last row of each block unset."""
    loglik, grad, _ = _reference_parts(beta, Z, sets)
    eta = Z @ beta
    risk = np.exp(eta - float(np.max(eta)))
    zs, rs = Z[sets.order], risk[sets.order]
    s0 = _tail(rs, sets.start)
    zbar = _tail(rs[:, None] * zs, sets.start) / s0[:, None]
    q = Z.shape[1]
    info = np.zeros((q, q))
    for first in range(0, q, block):
        rows = slice(first, min(q, first + block) - (1 if skip else 0))
        s2 = _tail(rs[:, None, None] * zs[:, rows, None] * zs[:, None, :], sets.start)
        info[rows] = (
            sets.deaths[:, None, None]
            * (s2 / s0[:, None, None] - zbar[:, rows, None] * zbar[:, None, :])
        ).sum(axis=0)
    return loglik, grad, info


def test_a_block_that_skips_a_row_is_caught() -> None:
    # Deliberate mutation: the byte comparison above sees a blocked construction that leaves
    # one Hessian row of each block unset, on the fixture it runs.
    X, time, event = _sample(p=6)
    Z = cox._design(X, "interaction")
    sets = _risk_sets(time, event)
    beta = np.random.default_rng(1).normal(scale=0.1, size=Z.shape[1])
    want = _reference_parts(beta, Z, sets)[2].tobytes()
    assert _blocked_parts(beta, Z, sets, 3, skip=False)[2].tobytes() == want
    assert _blocked_parts(beta, Z, sets, 3, skip=True)[2].tobytes() != want


def test_a_wrong_trial_likelihood_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deliberate mutation of the halving trial's likelihood: a value one unit low makes the
    # trial reject steps the true likelihood accepts, and the fit moves.
    model, (X, time, event) = CASES["main"]()
    reference, _ = _fit(CoxHazard("main"), X, time, event, monkeypatch, _reference_parts)
    original = cox._loglik

    def low(beta: np.ndarray, Z: np.ndarray, sets: _RiskSets) -> float:
        return original(beta, Z, sets) - 1.0

    with monkeypatch.context() as patch:
        patch.setattr(cox, "_loglik", low)
        mutated, _ = _fit(model, X, time, event, monkeypatch)
    assert _state(mutated, X) != _state(reference, X)


def test_the_information_needs_no_n_by_q_by_q_array(monkeypatch: pytest.MonkeyPatch) -> None:
    X, time, event = _sample(n=5_000, p=20)
    Z = cox._design(X, "interaction")  # q = 39
    sets = _risk_sets(time, event)
    beta = np.zeros(Z.shape[1])
    full = 8 * Z.shape[0] * Z.shape[1] ** 2  # one (n, q, q) array: 61 MB
    monkeypatch.setattr(cox, "_BLOCK_BYTES", 2**20)

    def peak(parts: Callable[..., Parts]) -> int:
        tracemalloc.start()
        try:
            parts(beta, Z, sets)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    blocked = peak(cox._loglik_parts)
    assert blocked < full / 5
    assert peak(_reference_parts) > 5 * blocked


def test_halving_trials_build_no_second_moments(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each Newton step builds the information once; a trial reads the likelihood alone.
    X, time, event = _sample(p=6)
    calls = {"parts": 0, "loglik": 0}
    parts, loglik = cox._loglik_parts, cox._loglik

    def counted_parts(beta: np.ndarray, Z: np.ndarray, sets: _RiskSets) -> Parts:
        calls["parts"] += 1
        return parts(beta, Z, sets)

    def counted_loglik(beta: np.ndarray, Z: np.ndarray, sets: _RiskSets) -> float:
        calls["loglik"] += 1
        return loglik(beta, Z, sets)

    monkeypatch.setattr(cox, "_loglik_parts", counted_parts)
    monkeypatch.setattr(cox, "_loglik", counted_loglik)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        CoxHazard("interaction").fit(X, time, event)
    # Without the split, every trial would be a second-moment pass as well.
    assert calls["loglik"] >= 1
    assert calls["parts"] <= calls["loglik"] + 1


def test_a_continuous_survival_fit_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default hazard library selects among the treatment-only, main-terms and interaction
    # Cox models by cross-validated partial likelihood; the fit is compared with the same fit
    # under the one-array construction, in this process.
    from cleverly.survival import ContinuousSurvivalTMLE
    from tests.unit.test_continuous_survival import _sample as survival_sample

    def run() -> tuple[tuple[str, bytes, bytes], ...]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ContinuousSurvivalTMLE(
                [1.0, 2.0], n_folds=2, learner_folds=3, random_state=0
            ).fit(
                survival_sample(400, seed=11),
                time="time",
                event="event",
                treatment="Z",
                baseline=["X3"],
            )
        return tuple(
            (name, np.float64(e.psi).tobytes(), np.float64(e.std_error).tobytes())
            for name, e in fit.items()
        )

    got = run()
    with monkeypatch.context() as patch:
        patch.setattr(cox, "_loglik_parts", _reference_parts)
        patch.setattr(cox, "_loglik", lambda beta, Z, sets: _reference_parts(beta, Z, sets)[0])
        want = run()
    assert got == want
