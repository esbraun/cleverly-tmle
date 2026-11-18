"""The fit-scoped cache of the longitudinal fits (X106 J2a).

A fit shares the arrays it reads more than once while some record or reader holds them, marks
them read-only, and returns exactly what it returns without the cache.  These tests pin the
equality on every estimator that runs a backward recursion, the sharing the records rely on,
the entries' lifetime, and the keys that keep two different arrays apart.
"""

from __future__ import annotations

import gc
import warnings
import weakref
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.interventions import Incremental
from cleverly.learners import GLM
from cleverly.longitudinal import IPSI, LTMLE, SDR, LongitudinalData, _fit_cache
from cleverly.longitudinal import data as data_module
from cleverly.longitudinal import sequential as sequential_module
from cleverly.longitudinal.regimen import resolve_plans, resolve_regimens


def _panel(n: int, n_times: int, *, survival: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Two baseline covariates, one covariate per node after the first, a binary treatment."""
    rng = np.random.default_rng(20261118)
    data: dict[str, np.ndarray] = {"W1": rng.normal(size=n), "W2": rng.normal(size=n)}
    previous, covariate, linear = np.zeros(n), np.zeros(n), 0.3 * data["W1"]
    time_varying: list[list[str]] = []
    for t in range(1, n_times + 1):
        names: list[str] = []
        if t >= 2:
            covariate = 0.5 * covariate + 0.3 * previous + 0.3 * data["W1"] + rng.normal(size=n)
            data[f"L{t}"] = covariate.copy()
            names.append(f"L{t}")
        treat = rng.binomial(1, expit(0.2 * data["W1"] + 0.3 * covariate + 0.4 * previous))
        data[f"A{t}"] = treat.astype(float)
        time_varying.append(names)
        previous = data[f"A{t}"]
        linear = linear + 0.2 * previous + 0.1 * covariate
        if survival:
            data[f"Y{t}"] = rng.binomial(1, expit(-2.5 + 0.5 * linear)).astype(float)
    if not survival:
        data["Y"] = rng.binomial(1, expit(-0.4 + 0.5 * linear)).astype(float)
    frame = pd.DataFrame(data)
    if survival:
        order = list(frame.columns)
        event = np.zeros(n, dtype=bool)
        for t in range(1, n_times + 1):
            later = order[order.index(f"Y{t}") + 1 :]
            event = event | (frame[f"Y{t}"].to_numpy() == 1)
            frame.loc[event, later] = np.nan
    outcome: Any = [f"Y{t}" for t in range(1, n_times + 1)] if survival else "Y"
    columns = {
        "outcome": outcome,
        "treatment": [f"A{t}" for t in range(1, n_times + 1)],
        "baseline": ["W1", "W2"],
        "time_varying": time_varying,
    }
    return frame, columns


def _learners() -> dict[str, Any]:
    return {
        "outcome_learner": GLM(family="binomial"),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "random_state": 7,
        "simultaneous": False,
    }


FITS: dict[str, tuple[type, dict[str, Any], bool]] = {
    "ltmle_survival": (LTMLE, {"always": 1, "never": 0}, True),
    "ltmle_end_of_study": (LTMLE, {"always": 1, "never": 0}, False),
    "sdr": (SDR, {"always": 1, "never": 0}, False),
    "ipsi": (IPSI, {"double": Incremental(2.0), "half": Incremental(0.5)}, False),
}


def _fit(case: str, *, n_folds: int = 3, **options: Any) -> Any:
    cls, regimens, survival = FITS[case]
    frame, columns = _panel(600, 3, survival=survival)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(regimens, n_folds=n_folds, **_learners(), **options).fit(frame, **columns)


def _uncached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every fit without the cache: the context never becomes active."""
    monkeypatch.setattr(_fit_cache, "fit_cache", _NoCache)


class _NoCache:
    """Stands in for the cache context: nothing becomes active."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def _numbers(result: Any) -> list[tuple[str, float, float, bytes]]:
    return [
        (
            name,
            estimate.psi,
            estimate.variance,
            np.asarray(estimate.influence_curve).tobytes(),
        )
        for name, estimate in result.items()
    ]


@pytest.mark.parametrize("case", sorted(FITS))
def test_the_cache_changes_no_estimate_bit(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every estimate, variance and influence curve is bitwise the fit without the cache."""
    cached = _fit(case)
    _uncached(monkeypatch)
    plain = _fit(case)
    assert _numbers(cached) == _numbers(plain)
    assert str(cached.summary()) == str(plain.summary())


def test_the_cache_is_inactive_outside_a_fit() -> None:
    """The context variable is unset once the fit returns, and nothing else is held."""
    _fit("ltmle_survival")
    assert not _fit_cache.active()


def _regimen_fits(result: Any) -> dict[str, list[Any]]:
    """The records of each regimen, over its horizons."""
    by_regimen: dict[str, list[Any]] = {}
    for label, fit in result.fits.items():
        by_regimen.setdefault(label.split(" @ ")[0], []).append(fit)
    return by_regimen


def test_horizons_share_the_cumulative_products_and_clever_covariates() -> None:
    """At three horizons a regimen's records hold one cumulative product and one clever
    covariate per node, not one per horizon; the targeted arm is the observed fluctuation."""
    by_regimen = _regimen_fits(_fit("ltmle_survival", n_folds=1))
    assert sorted(by_regimen) == ["always", "never"]
    for records in by_regimen.values():
        assert len(records) == 3
        assert len({id(fit.cumulative) for fit in records}) == 1
        assert len({id(fit.cumulative_unbounded) for fit in records}) == 1
        clever: dict[int, set[int]] = {}
        for fit in records:
            for step in fit.steps:
                clever.setdefault(step.time, set()).add(id(step.clever))
                if step.fluctuation is not None:
                    assert step.targeted is step.fluctuation.targeted.observed
        assert all(len(ids) == 1 for ids in clever.values())


def test_shared_result_arrays_are_read_only() -> None:
    """A write through one record would change another, so it raises instead."""
    fit = _regimen_fits(_fit("ltmle_survival", n_folds=1))["always"][0]
    with pytest.raises(ValueError, match="read-only"):
        fit.cumulative[0, 0] = 0.5
    with pytest.raises(ValueError, match="read-only"):
        fit.steps[0].clever[0] = 0.5


def test_an_entry_lives_only_while_its_value_does() -> None:
    """A value nobody holds is dropped with its sources, so the cache never keeps alive an
    array the fit without it would have freed."""

    class Source:
        pass

    source = Source()
    source_ref = weakref.ref(source)
    with _fit_cache.fit_cache():
        first = _fit_cache.cached("probe", (source,), 1, lambda: np.arange(5.0))
        again = _fit_cache.cached("probe", (source,), 1, lambda: np.arange(5.0))
        assert again is first
        assert not first.flags.writeable
        cache = _fit_cache._ACTIVE.get()
        assert cache is not None and len(cache.entries) == 1
        del first, again
        assert not cache.entries  # dropped by reference counting, with no collection
        del source
        assert source_ref() is None  # the entry released its source
        other = Source()
        rebuilt = _fit_cache.cached("probe", (other,), 1, lambda: np.arange(5.0))
        assert np.array_equal(rebuilt, np.arange(5.0))
    assert not cache.entries and not cache.strong


def test_a_tuple_value_is_shared_while_every_part_lives() -> None:
    class Source:
        pass

    source = Source()
    with _fit_cache.fit_cache():
        pair = _fit_cache.cached("pair", (source,), None, lambda: (np.zeros(2), np.ones(2)))
        again = _fit_cache.cached("pair", (source,), None, lambda: (np.zeros(2), np.ones(2)))
        assert again[0] is pair[0] and again[1] is pair[1]
        kept = pair[0]
        del pair, again
        gc.collect()
        fresh = _fit_cache.cached("pair", (source,), None, lambda: (np.zeros(2), np.ones(2)))
        assert fresh[0] is not kept


def test_an_arm_that_is_a_distinct_array_moves_along_its_own_covariate() -> None:
    """a3: only an arm that is the observed array itself reuses the observed result; an arm
    with equal-shaped but distinct predictions and covariate moves on its own (the mutation
    "always reuse" fails here), and the reused arm is the observed result object."""
    from cleverly.fluctuation.iterative import InitialFit, apply_logistic
    from cleverly.fluctuation.submodel import Submodel

    rng = np.random.default_rng(3)
    observed = rng.uniform(0.2, 0.8, 20)
    other = rng.uniform(0.2, 0.8, 20)
    h_observed = rng.normal(size=(20, 1))
    h_other = rng.normal(size=(20, 1))
    epsilon = np.array([0.3])
    fit = InitialFit(observed, {0.0: other, 1.0: observed})
    submodel = Submodel(h_observed, {0.0: h_other, 1.0: h_observed}, ("h",), "mean")
    moved = apply_logistic(fit, submodel, epsilon, 0.9995)
    plain = apply_logistic(
        InitialFit(observed, {0.0: other.copy(), 1.0: observed.copy()}),
        Submodel(h_observed, {0.0: h_other.copy(), 1.0: h_observed.copy()}, ("h",), "mean"),
        epsilon,
        0.9995,
    )
    assert moved.arms[1.0] is moved.observed
    assert moved.arms[0.0].tobytes() == plain.arms[0.0].tobytes()
    assert moved.arms[1.0].tobytes() == plain.arms[1.0].tobytes()
    assert not np.array_equal(moved.arms[0.0], moved.observed)


def test_two_bounds_give_two_cumulative_products() -> None:
    """The bounds are part of the key: a mutation that drops them fails here."""
    result = _fit("ltmle_end_of_study", n_folds=1)
    data, mechanism = result.data, result.mechanism
    plan = resolve_plans(resolve_regimens({"always": 1}, data.n_times), data)[0]
    with _fit_cache.fit_cache():
        loose = mechanism.cumulative_with_unbounded(data, plan, (1e-6, 1.0))
        tight = mechanism.cumulative_with_unbounded(data, plan, (0.4, 1.0))
        assert np.array_equal(loose[0], tight[0])
        assert not np.array_equal(loose[1], tight[1])


def test_two_numerator_columns_give_two_clever_covariates() -> None:
    """The numerator view and the node are part of the key: the same node with two columns of
    one numerator gets two weights, and one product read at two nodes gets two, so a mutation
    that drops either from the key fails here."""
    n = 50
    at_risk = np.ones(n, dtype=bool)
    cumulative = np.full((n, 3), 0.5)
    numerator = np.column_stack([np.full(n, 0.2), np.full(n, 0.4), np.full(n, 0.8)])
    with _fit_cache.fit_cache():
        first = sequential_module._clever_covariate(
            at_risk, at_risk, cumulative, 2, numerator[:, 0]
        )
        second = sequential_module._clever_covariate(
            at_risk, at_risk, cumulative, 2, numerator[:, 1]
        )
        assert np.allclose(first[1], 0.4) and np.allclose(second[1], 0.8)
        # The node is part of the key: one cumulative product read at two nodes.
        halving = np.column_stack([np.full(n, 0.5), np.full(n, 0.25), np.full(n, 0.125)])
        second_node = sequential_module._clever_covariate(at_risk, at_risk, halving, 2, None)
        third_node = sequential_module._clever_covariate(at_risk, at_risk, halving, 3, None)
        assert np.allclose(second_node[1], 4.0) and np.allclose(third_node[1], 8.0)


def test_the_design_with_and_without_the_current_arm_differ() -> None:
    """``include_current`` is part of the design key, so the two designs never share."""
    frame, columns = _panel(300, 3, survival=False)
    data = LongitudinalData.from_frame(frame, **columns)
    with _fit_cache.fit_cache():
        without = data.history_design(2, include_current=False)
        with_arm = data.history_design(2, include_current=True)
        assert with_arm.shape[1] > without.shape[1]
        assert np.array_equal(data.covariate_history(2), data._covariate_history(2))


def test_the_covariate_blocks_are_cleaned_once_per_panel() -> None:
    frame, columns = _panel(300, 4, survival=False)
    data = LongitudinalData.from_frame(frame, **columns)
    with _fit_cache.fit_cache():
        held = [data.covariate_history(t) for t in range(1, 5)]
        cache = _fit_cache._ACTIVE.get()
        assert cache is not None and len(cache.strong) == 1
        outside = [data._covariate_history(t) for t in range(1, 5)]
    for t, (inside, plain) in enumerate(zip(held, outside, strict=True), start=1):
        # The shipped construction: join the raw blocks, then fill.
        blocks = [data.baseline, *data.time_varying[:t]]
        shipped = np.nan_to_num(np.hstack([b for b in blocks if b.shape[1]]), nan=0.0)
        for matrix in (inside, plain):
            assert matrix.tobytes(order="A") == shipped.tobytes(order="A")
            assert matrix.flags.c_contiguous == shipped.flags.c_contiguous


class _WritesIntoX(LogisticRegression):
    def predict_proba(self, X: Any) -> Any:
        X[0, 0] = 0.0
        return super().predict_proba(X)


def test_a_learner_that_writes_into_its_design_is_refused_by_numpy() -> None:
    """The shared designs are read-only, so a learner that writes into ``X`` raises numpy's
    error rather than changing the design another model reads."""
    frame, columns = _panel(300, 2, survival=False)
    options = {**_learners(), "treatment_learner": _WritesIntoX(max_iter=1000)}
    with pytest.raises(ValueError, match="read-only"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LTMLE({"always": 1, "never": 0}, n_folds=1, **options).fit(frame, **columns)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([1.0, np.nan, np.inf, 2.0], dtype=object), [True, False, False, True]),
        (np.array([np.float32(1.0), np.float16(np.nan)], dtype=object), [True, False]),
        (np.array([np.longdouble("1e400"), 1.0], dtype=object), [True, True]),
        (np.array([2**1100, 1], dtype=object), [True, True]),
        (np.array(["a", None, "b"], dtype=object), [True, False, True]),
        (np.array(["a", pd.NA, 1.0], dtype=object), [True, False, True]),
        (np.array(["a", "b"], dtype=object), [True, True]),
    ],
)
def test_the_presence_check_is_the_per_cell_reading(
    values: np.ndarray, expected: list[bool]
) -> None:
    """The fast paths answer what the per-cell reading answers, including a long double above
    the ``float64`` range and a Python integer too large for a float."""
    per_cell = [data_module._categorical_value_is_present(value) for value in values]
    assert per_cell == expected
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert data_module._present_objects(values).tolist() == expected


def test_a_float_subclass_holding_nan_is_read_one_cell_at_a_time() -> None:
    class Odd(float):
        pass

    values = np.array([Odd("nan"), 1.0], dtype=object)
    assert data_module._present_objects(values).tolist() == [False, True]


def test_an_object_column_of_floats_is_not_read_one_cell_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(value: Any) -> bool:
        raise AssertionError("per-cell reading")

    monkeypatch.setattr(data_module, "_categorical_value_is_present", refuse)
    values = np.array([1.0, np.nan, 0.0] * 100, dtype=object)
    assert data_module._present_objects(values).sum() == 200
