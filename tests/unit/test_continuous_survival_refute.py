"""Refits of a continuous-time survival fit (X101 PR 1): storage, refutation, truncation curve.

A :class:`~cleverly.survival.ContinuousSurvivalResult` keeps a copy of its estimator, with
the seed resolved and the outer fold labels recorded, and its data. Every check here refits
that estimator, so it runs the shipped engine on a perturbed copy of the fit's own data.

The witnesses guard parts an exact check at the truth would miss: a refit at another seed
moves the estimate, an unseeded user learner breaks the replay unless the fit seeds it, a
placebo that returned the data unchanged, a stale placebo row, a subset refit that redrew its
split, a noise column that moved nothing, a floor that binds only counterfactual cells, a
floor applied to the propensity alone, and a stored datum that no longer reproduces the fit.

Plan rows T1 to T12 of ``rows/X101/plan.md`` (rev 2); T13 is the sweep kind in
``tests/unit/_capability_sweep_support.py``.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard, engine
from cleverly.survival.data import ContinuousSurvivalData
from cleverly.validation import _continuous_survival_refute as adapter
from cleverly.validation import _interpretation as interpretation
from cleverly.validation import refute
from cleverly.validation._interpretation import (
    DESCRIPTIVE_SENTENCE,
    CalibrationScope,
    VerdictEvidence,
    continuous_survival_configuration,
)
from cleverly.validation._readings import subset_reading
from tests.unit.test_continuous_survival_assessment import _arm_censored

#: The arm risk difference of cause 1 at the last target time, the contrast most tests read.
RD = "ate_regimen[1 vs 0, 1 @ t=2]"
#: The risk ratio of the same cause and time.
RR = "rr_regimen[1 vs 0, 1 @ t=2]"
#: An arm risk, which has no fixed placebo null.
LEVEL = "cif_regimen[1, 1 @ t=2]"


def _frame(n: int = 300, seed: int = 0, *, intercept: float = -1.5, arm: float = -1.0) -> Any:
    """The sweep's law: two causes, a binary treatment, independent censoring."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    h1 = np.exp(intercept + 0.5 * w + arm * a)
    h2 = np.exp(-2.0 - 0.5 * w - 0.5 * a)
    t = rng.exponential(1 / (h1 + h2))
    cause = np.where(rng.uniform(size=n) < h1 / (h1 + h2), 1, 2)
    c = rng.exponential(1 / 0.15, n)
    return pd.DataFrame(
        {"time": np.minimum(t, c), "event": np.where(t <= c, cause, 0), "A": a, "W": w}
    )


def _estimator(horizons: Any = (1.0, 2.0), **settings: Any) -> ContinuousSurvivalTMLE:
    options: dict[str, Any] = {
        "cross_fit": False,
        "hazard_library": [CoxHazard("main")],
        "censoring_library": [CoxHazard("main")],
        "random_state": 0,
    }
    options.update(settings)
    return ContinuousSurvivalTMLE(list(horizons), **options)


def _fit(frame: Any, horizons: Any = (1.0, 2.0), **settings: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _estimator(horizons, **settings).fit(
            frame, time="time", event="event", treatment="A", baseline=["W"]
        )


@functools.cache
def _in_sample() -> Any:
    return _fit(_frame())


@functools.cache
def _cross_fitted() -> Any:
    return _fit(_frame(), cross_fit=True, n_folds=3)


def _fresh(result: Any) -> Any:
    """A copy with an empty report cache, sharing the fit."""
    return dataclasses.replace(result)


def _same(first: Any, second: Any) -> bool:
    """Whether two fits agree bit for bit in every estimate, standard error and record field."""
    if list(first) != list(second):
        return False
    for name in first:
        a, b = first[name], second[name]
        if a.psi != b.psi or a.plugin_std_error != b.plugin_std_error:
            return False
    r, s = first.record, second.record
    return (
        r.initial_psi == s.initial_psi
        and r.standardized_score == s.standardized_score
        and (r.bounded_share, r.last_time_floor_share, r.frozen_share, r.off_grid_share)
        == (s.bounded_share, s.last_time_floor_share, s.frozen_share, s.off_grid_share)
        and (r.steps, r.halvings, r.weight_floor) == (s.steps, s.halvings, s.weight_floor)
        and np.array_equal(r.p1, s.p1)
        and np.array_equal(r.retention_before, s.retention_before)
    )


@pytest.fixture
def refits(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[Any, dict[str, Any], Any]]]:
    """Record every ``refit`` call: the data, the keyword arguments and the new result."""
    calls: list[tuple[Any, dict[str, Any], Any]] = []
    original = ContinuousSurvivalTMLE.refit

    def recording(self: Any, data: Any, **kwargs: Any) -> Any:
        result = original(self, data, **kwargs)
        calls.append((data, kwargs, result))
        return result

    monkeypatch.setattr(ContinuousSurvivalTMLE, "refit", recording)
    yield calls


# ----------------------------------------------------------------- T1, T2 storage and seeds


@pytest.mark.parametrize("fitted", [_in_sample, _cross_fitted], ids=["in_sample", "cross_fitted"])
def test_t1_the_frame_and_the_container_fit_identically(fitted: Any) -> None:
    result = fitted()
    container = ContinuousSurvivalData.from_frame(
        _frame(), time="time", event="event", treatment="A", baseline=["W"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = copy.copy(result.estimator).fit_data(container)
    assert _same(result, direct)
    np.testing.assert_array_equal(direct.estimator._outer_folds, result.estimator._outer_folds)


@pytest.mark.parametrize("fitted", [_in_sample, _cross_fitted], ids=["in_sample", "cross_fitted"])
def test_t2_the_stored_estimator_reproduces_the_fit(fitted: Any) -> None:
    result = fitted()
    stored = result.estimator
    assert isinstance(stored.random_state, int) and stored.random_state == 0
    assert isinstance(result.data, ContinuousSurvivalData) and result.data.n == result.n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = stored.refit(result.data, random_state=stored.random_state)
    assert _same(result, again)
    # The stored copy shares no mapping with the estimator it came from.
    assert stored.arms is None or stored.arms is not again.estimator.arms


def test_t2_a_refit_at_another_seed_moves_the_cross_fitted_estimate() -> None:
    """Mutation control of T2: the identity above is not vacuous, the seed reaches the folds."""
    result = _cross_fitted()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        moved = result.estimator.refit(result.data, random_state=result.estimator.random_state + 1)
    assert any(moved[name].psi != result[name].psi for name in result)


def test_t2_an_unseeded_fit_stores_its_seed_and_reproduces() -> None:
    result = _fit(_frame(), cross_fit=True, n_folds=3, random_state=None)
    seed = result.estimator.random_state
    assert isinstance(seed, int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = result.estimator.refit(result.data, random_state=seed)
        second = result.estimator.refit(result.data, random_state=seed)
    assert _same(first, second) and _same(result, first)


def _saga_fit() -> Any:
    from sklearn.linear_model import LogisticRegression

    learner = LogisticRegression(solver="saga", max_iter=3, tol=1e-12)
    return _fit(_frame(), cross_fit=True, n_folds=3, treatment_learner=learner, simultaneous=True)


def _replays(result: Any) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = result.estimator.refit(result.data, random_state=result.estimator.random_state)
    band, other = result.simultaneous, again.simultaneous
    return (
        _same(result, again)
        and band.critical_value == other.critical_value
        and band.bands == other.bands
    )


def test_t2_an_unseeded_user_learner_and_the_band_take_the_fit_seed() -> None:
    result = _saga_fit()
    assert result.estimator.treatment_learner.random_state is None
    assert _replays(result)


def test_t2_without_the_seed_an_unseeded_saga_learner_breaks_the_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control of M4: an unseeded SAGA fit draws from the global generator."""
    monkeypatch.setattr(engine, "seed_unset", lambda learner, seed: learner)
    assert not _replays(_saga_fit())


# ----------------------------------------------------------------- T3 to T5 the placebo


def _placebo_witness(original: Any, copies: list[Any]) -> bool:
    """Each copy permutes the treatment, moves at least one row and keeps everything else."""
    for data in copies:
        if sorted(data.treatment.tolist()) != sorted(original.treatment.tolist()):
            return False
        if np.array_equal(data.treatment, original.treatment):
            return False
        for name in ("time", "event", "baseline"):
            if not np.array_equal(getattr(data, name), getattr(original, name)):
                return False
        if data.names != original.names:
            return False
    return True


def test_t3_the_placebo_permutes_the_treatment_and_keeps_the_rest(refits: list[Any]) -> None:
    result = _fresh(_in_sample())
    refute(result, estimand=RD, tests=("placebo",), n_replicates=3)
    assert len(refits) == 3
    assert _placebo_witness(result.data, [data for data, _, _ in refits])


def test_t3_a_placebo_that_keeps_the_treatment_fails_the_witness(
    refits: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fresh(_in_sample())
    monkeypatch.setattr(adapter, "_placebo_treatment", lambda treatment, rng, mode: treatment)
    refute(result, estimand=RD, tests=("placebo",), n_replicates=2)
    assert not _placebo_witness(result.data, [data for data, _, _ in refits])


def test_t3_the_drawn_placebo_keeps_both_arms() -> None:
    rng = np.random.default_rng(0)
    treatment = np.array([0] * 5 + [1] * 95)
    for _ in range(50):
        assert set(adapter._placebo_treatment(treatment, rng, "draw").tolist()) == {0, 1}


@functools.cache
def _strong_effect() -> Any:
    """The T4 law: a cause-1 effect far from the null, so a leak of it would show."""
    return _fit(_frame(intercept=-1.0, arm=-1.5), horizons=(0.5, 1.0))


#: The T4 contrast: the cause-1 risk difference at the last target time.
STRONG = "ate_regimen[1 vs 0, 1 @ t=1]"


def test_t4_the_placebo_reads_the_null_beside_a_strong_effect() -> None:
    result = _fresh(_strong_effect())
    estimate = result[STRONG]
    assert abs(estimate.psi / estimate.plugin_std_error) > 5
    report = refute(result, estimand=STRONG, tests=("placebo",))
    test = report["placebo"]
    assert len(test.values) == 20
    assert -3.0 <= test.statistic <= 3.0


def test_t4_a_stale_placebo_row_share_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation control: keeping 80% of the recorded treatments leaks the effect."""

    def stale(treatment: Any, rng: Any, mode: str) -> Any:
        out = np.asarray(treatment).copy()
        moved = rng.choice(out.size, size=out.size // 5, replace=False)
        out[moved] = rng.permutation(out[moved])
        return out

    monkeypatch.setattr(adapter, "_placebo_treatment", stale)
    report = refute(_fresh(_strong_effect()), estimand=STRONG, tests=("placebo",))
    assert abs(report["placebo"].statistic) > 5


def test_t5_the_ratio_placebo_reads_the_log_against_zero() -> None:
    report = refute(_fresh(_in_sample()), estimand=RR, tests=("placebo",), n_replicates=5)
    test = report["placebo"]
    logs = np.log(np.asarray(test.values))
    expected = logs.mean() / (logs.std(ddof=1) / np.sqrt(logs.size))
    assert test.statistic == pytest.approx(expected, rel=1e-12)
    assert test.expectation == "~ 1"
    # Mutation control: a reading of psi - 1 is another statistic.
    shifted = np.asarray(test.values) - 1.0
    wrong = shifted.mean() / (shifted.std(ddof=1) / np.sqrt(shifted.size))
    assert abs(wrong - test.statistic) > 1e-3 * abs(test.statistic)


# ----------------------------------------------------------------- T6 the subset


def _rows_of(original: Any, subset: Any) -> np.ndarray:
    """The original row of each subset row: the follow-up times are distinct."""
    order = np.argsort(original.time)
    found = order[np.searchsorted(original.time, subset.time, sorter=order)]
    assert np.array_equal(original.time[found], subset.time)
    return found


def test_t6_the_subset_keeps_each_units_fold(refits: list[Any]) -> None:
    result = _fresh(_cross_fitted())
    n = result.n
    report = refute(result, estimand=RD, tests=("subset",), n_replicates=4)
    size = min(n, max(20, int(0.7 * n)))
    assert len(refits) == 4
    labels = result.estimator._outer_folds
    for data, kwargs, refitted in refits:
        rows = _rows_of(result.data, data)
        assert rows.size == size == np.unique(rows).size
        assert kwargs["random_state"] == report.random_state
        np.testing.assert_array_equal(refitted.estimator._outer_folds, labels[rows])
    # Mutation control: a refit that redraws its split fails the equality above.
    data, _, refitted = refits[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        redrawn = result.estimator.refit(data, random_state=report.random_state)
    assert not np.array_equal(redrawn.estimator._outer_folds, labels[_rows_of(result.data, data)])
    # rho is the shared reading of the values, the fit's SE, n and m.
    test = report["subset"]
    reading = subset_reading(list(test.values), std_error=result[RD].plugin_std_error, n=n, m=size)
    assert test.statistic == reading.rho
    assert test.passed is None
    # Mutation control: m = n gives another rho.
    wrong = subset_reading(list(test.values), std_error=result[RD].plugin_std_error, n=n, m=n)
    assert wrong.rho != reading.rho


def test_t6_an_in_sample_subset_keeps_one_fold(refits: list[Any]) -> None:
    refute(_fresh(_in_sample()), estimand=RD, tests=("subset",), n_replicates=2)
    for _, kwargs, refitted in refits:
        assert np.unique(kwargs["_frozen_folds"]).tolist() == [0]
        assert refitted.record.n_folds == 1


def test_t6_a_draw_that_loses_events_refuses_before_any_refit(refits: list[Any]) -> None:
    result = _fresh(_in_sample())
    with pytest.raises(CapabilityError, match=r"subset draw 1 of 20 leaves arm") as caught:
        refute(result, estimand=RD, tests=("subset",), subset_fraction=0.05)
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    assert "Raise subset_fraction, or drop subset from tests=." in str(caught.value)
    assert refits == []
    # With the default tests the placebo and noise refits come first in the order; the guard
    # still refuses before any of them runs.
    with pytest.raises(CapabilityError, match=r"subset draw 1 of 20 leaves arm"):
        refute(result, estimand=RD, subset_fraction=0.05)
    assert refits == []


def test_t6_a_subset_refit_reproduces_under_its_own_refit(refits: list[Any]) -> None:
    """A fit made with frozen folds keeps them when refitted without naming them."""
    refute(_fresh(_cross_fitted()), estimand=RD, tests=("subset",), n_replicates=2)
    _, _, refitted = refits[0]
    stored = refitted.estimator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = stored.refit(refitted.data, random_state=stored.random_state)
    np.testing.assert_array_equal(again.estimator._outer_folds, stored._outer_folds)
    assert again[RD].psi == refitted[RD].psi


def test_the_stored_data_does_not_alias_the_callers_arrays() -> None:
    result = _fresh(_in_sample())
    # A private copy: the fitted results are shared across tests.
    data = dataclasses.replace(result.data, time=result.data.time.copy())
    before = data.time.copy()
    refitted = result.estimator.refit(data, random_state=result.estimator.random_state)
    data.time[:] = 0.0
    np.testing.assert_array_equal(refitted.data.time, before)


# ----------------------------------------------------------------- T7, T8 noise and seeds


def test_t7_the_noise_column_is_standard_normal_and_keeps_the_fits_seed_and_folds(
    refits: list[Any],
) -> None:
    result = _fresh(_cross_fitted())
    report = refute(result, estimand=RD, tests=("random_common_cause",), n_replicates=3)
    original = result.data
    for index, (data, kwargs, refitted) in enumerate(refits):
        assert data.baseline.shape == (original.n, original.baseline.shape[1] + 1)
        np.testing.assert_array_equal(data.baseline[:, :-1], original.baseline)
        for name in ("time", "event", "treatment"):
            np.testing.assert_array_equal(getattr(data, name), getattr(original, name))
        assert data.names == (*original.names, f"_noise_{index}")
        noise = data.baseline[:, -1]
        assert abs(noise.mean()) < 0.25 and 0.8 < noise.std() < 1.2
        assert kwargs["random_state"] == result.estimator.random_state
        np.testing.assert_array_equal(
            refitted.estimator._outer_folds, result.estimator._outer_folds
        )
    test = report["random_common_cause"]
    assert test.passed is None
    # Nonzero witness: the noise column moved the estimate.
    assert any(value != result[RD].psi for value in test.values)


def test_t8_an_unseeded_request_repeats_and_records_its_seed() -> None:
    result = _in_sample()
    first = refute(_fresh(result), estimand=RD, tests=("placebo", "subset"), n_replicates=3)
    second = refute(_fresh(result), estimand=RD, tests=("placebo", "subset"), n_replicates=3)
    assert first.random_state == second.random_state == result.estimator.random_state
    assert first["placebo"].values == second["placebo"].values
    assert first["subset"].values == second["subset"].values


# ----------------------------------------------------------------- T9 refusals


#: Each refused request, its reason, a fragment of its text and its roadmap item.
REFUSED: dict[str, tuple[dict[str, Any], RefusalReason, str, str | None]] = {
    "level_placebo": (
        {"estimand": LEVEL, "tests": ("placebo",)},
        RefusalReason.INVALID_DECLARATION,
        "the placebo has no fixed null for 'cif_regimen[1, 1 @ t=2]'",
        None,
    ),
    "negative_control": (
        {"estimand": RD, "tests": ("negative_control_outcome",), "negative_control_outcome": 0},
        RefusalReason.IMPLEMENTATION_GAP,
        "no negative-control adapter is implemented for a continuous-time survival fit",
        "X105",
    ),
    "dummy_outcome": (
        {"estimand": RD, "tests": ("dummy_outcome",)},
        RefusalReason.IMPLEMENTATION_GAP,
        "no generated event-time process is declared",
        "X105",
    ),
    "simulated_outcome": (
        {"estimand": RD, "tests": ("simulated_outcome",)},
        RefusalReason.IMPLEMENTATION_GAP,
        "no generated event-time process is declared",
        "X105",
    ),
    "plasmode": (
        {"estimand": RD, "tests": ("plasmode",)},
        RefusalReason.IMPLEMENTATION_GAP,
        "no generated event-time process is declared",
        "X105",
    ),
    "bootstrap_stability": (
        {"estimand": RD, "tests": ("bootstrap_stability",)},
        RefusalReason.IMPLEMENTATION_GAP,
        "The full-refit bootstrap of X6 (d) needs the same fold rule.",
        "X105",
    ),
    "bootstrap_measurement_error": (
        {"estimand": RD, "tests": ("bootstrap_measurement_error",)},
        RefusalReason.IMPLEMENTATION_GAP,
        "a unit drawn twice must share a cross-fitting fold",
        "X105",
    ),
    "future_treatment": (
        {"estimand": RD, "tests": ("future_treatment",), "future_treatment_control": 0},
        RefusalReason.INVALID_DECLARATION,
        "a continuous-time survival fit has one baseline treatment.",
        None,
    ),
    "markov": (
        {"estimand": RD, "tests": ("placebo",), "placebo_mode": "markov"},
        RefusalReason.INVALID_DECLARATION,
        "which the placebo permutes ('permute') or draws ('draw')",
        None,
    ),
    "noise_nodes": (
        {"estimand": RD, "tests": ("random_common_cause",), "noise_nodes": True},
        RefusalReason.INVALID_DECLARATION,
        "random_common_cause appends one baseline column",
        None,
    ),
    "unreported": (
        {"estimand": "ate", "tests": ("subset",)},
        RefusalReason.INVALID_DECLARATION,
        "estimand 'ate' was not requested in this fit",
        None,
    ),
}


@pytest.mark.parametrize("case", list(REFUSED))
def test_t9_each_refused_request_raises_its_record_everywhere(case: str) -> None:
    arguments, reason, text, roadmap = REFUSED[case]
    result = _fresh(_in_sample())
    with pytest.raises(CapabilityError) as direct:
        refute(result, **arguments)
    refusal = direct.value.refusal
    assert refusal.reason is reason and text in refusal.condition and refusal.roadmap == roadmap
    if reason is RefusalReason.IMPLEMENTATION_GAP:
        assert refusal.alternative == adapter._RUNS
    with pytest.raises(CapabilityError) as facade:
        result.diagnostics.refute(**arguments)
    assert facade.value.refusal.reason is reason
    assert text in facade.value.refusal.condition
    row = result.diagnostics._capability_for_arguments("refute", arguments)
    assert not row.available and row.refusal is not None
    assert text in row.refusal.condition


def test_t9_the_level_placebo_defers_the_row_on_tests_and_the_rest_runs() -> None:
    result = _fresh(_in_sample())
    row = result.diagnostics._capability_for_arguments("refute", {"estimand": LEVEL})
    assert row.status.value == "deferred" and "tests" in row.requires_arguments
    report = result.diagnostics.refute(
        estimand=LEVEL, tests=("random_common_cause", "subset"), n_replicates=2
    )
    assert [test.name for test in report.tests] == ["random_common_cause", "subset"]


def test_t9_a_contrast_runs_the_default_tests() -> None:
    result = _fresh(_in_sample())
    row = result.diagnostics._capability_for_arguments("refute", {"estimand": RD})
    assert row.available
    report = result.diagnostics.refute(estimand=RD, n_replicates=2)
    assert [test.name for test in report.tests] == ["placebo", "random_common_cause", "subset"]


def test_t9_a_result_without_its_estimator_or_data_refuses() -> None:
    for missing in ("estimator", "data"):
        bare = dataclasses.replace(_in_sample(), **{missing: None})
        with pytest.raises(CapabilityError, match="refute needs the fitted estimator and the data"):
            refute(bare, estimand=RD, tests=("subset",))


def test_t9_simulated_confounding_runs_on_a_continuous_fit() -> None:
    """PR 3 lifts the X101 refusal: the row is live and asks for a grid and an estimand."""
    result = _fresh(_in_sample())
    row = result.sensitivity.capability("simulated_confounding")
    assert row.available and row.refusal is None
    assert set(row.requires_arguments) == {"grid", "estimand"}


# ----------------------------------------------------------------- T10 readings and scope


def test_t10_every_reading_is_descriptive() -> None:
    report = refute(_fresh(_in_sample()), estimand=RD, n_replicates=3)
    for test in report.tests:
        assert test.passed is None, test.name
        assert test.rung == "L0"
    assert DESCRIPTIVE_SENTENCE in report["placebo"].scope_sentence
    assert DESCRIPTIVE_SENTENCE in report["subset"].scope_sentence


def test_t10_the_configuration_reads_the_fit() -> None:
    result = _in_sample()
    fit = continuous_survival_configuration(result, RD)
    data = result.data
    observed = data.time[data.event > 0]
    assert fit.method == "ContinuousSurvivalTMLE"
    assert fit.estimand_kind == "continuous:ate_regimen"
    assert continuous_survival_configuration(result, RR).estimand_kind == "continuous:rr_regimen"
    assert continuous_survival_configuration(result, LEVEL).estimand_kind == "continuous:risk"
    assert fit.outcome_family == "competing_risks" and fit.causes == 2
    assert fit.treatment_kind == "discrete" and fit.arms == 2 and fit.nodes is None
    assert fit.learners == ("CoxHazard('main')", "CoxHazard('main')", "default")
    assert fit.cross_fitting == "off" and not fit.known_mechanism and fit.n == result.n
    assert fit.event_share == float(np.mean(observed <= 2.0))
    in_arm = [
        int(np.sum((result.record.arm == a) & (data.event == code) & (data.time <= 2.0)))
        for a in (0, 1)
        for code in (1, 2)
    ]
    assert fit.min_event_count == min(in_arm)
    assert continuous_survival_configuration(_cross_fitted(), RD).cross_fitting == "folds=3"


def _placebo_row(fit: Any, share: tuple[float, float]) -> VerdictEvidence:
    return VerdictEvidence(
        "placebo",
        "t",
        "L2",
        "x",
        CalibrationScope(
            methods=frozenset({fit.method}),
            estimand_kinds=frozenset({fit.estimand_kind}),
            outcome_families=frozenset({fit.outcome_family}),
            treatment_kinds=frozenset({fit.treatment_kind}),
            learners=frozenset(fit.learners),
            cross_fitting=frozenset({fit.cross_fitting}),
            inference=frozenset({fit.inference}),
            known_mechanism=frozenset({False}),
            arms=frozenset({2}),
            nodes=frozenset({None}),
            replicates=frozenset({3}),
            placebo_draws=frozenset({"permute"}),
            n_min=fit.n,
            causes=frozenset({2}),
            event_share_range=share,
        ),
        "a licensed sentence",
        study=("study", "s1", "p1"),
    )


def test_t10_a_row_with_the_fits_scope_licenses_and_an_event_share_outside_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _in_sample()
    fit = continuous_survival_configuration(result, RD)
    share = float(fit.event_share)
    monkeypatch.setattr(
        interpretation, "VERDICT_EVIDENCE", (_placebo_row(fit, (share - 0.01, share + 0.01)),)
    )
    licensed = refute(_fresh(result), estimand=RD, tests=("placebo",), n_replicates=3)
    assert licensed["placebo"].passed is not None
    assert licensed["placebo"].rung == "L2"
    monkeypatch.setattr(
        interpretation, "VERDICT_EVIDENCE", (_placebo_row(fit, (share + 0.01, share + 0.2)),)
    )
    outside = refute(_fresh(result), estimand=RD, tests=("placebo",), n_replicates=3)
    assert outside["placebo"].passed is None
    assert "event share" in outside["placebo"].scope_sentence


# ----------------------------------------------------------------- T11, T12 truncation curve


def _rows(curve: Any, name: str) -> pd.DataFrame:
    frame = curve.to_frame()
    return frame[frame["estimand"] == name].sort_values("bound").reset_index(drop=True)


def _initial(result: Any) -> Any:
    estimator = result.estimator
    return estimator._initial(result.data, estimator.random_state)


def _block_product(plan: Any, k: int) -> np.ndarray:
    """``(2, c, M)`` pi S_c(u-) of the walk's block ``k``, as the engine builds its weight."""
    lo, hi = plan.bounds[k]
    jumps = plan.fit.hazards.censoring_jumps(lo, hi)
    retention = np.cumprod(1.0 - np.clip(jumps, 0.0, 1.0), axis=-1)
    before = np.concatenate([np.ones((2, hi - lo, 1)), retention[..., :-1]], axis=-1)
    return plan.pi[:, lo:hi, None] * before


def _product(initial: Any) -> np.ndarray:
    """``(2, n, M)`` pi S_c(u-) of every unit."""
    plan = engine._FitPlan(initial, 0.5)
    return np.concatenate([_block_product(plan, k) for k in range(len(plan.bounds))], axis=1)


def _cells(result: Any) -> tuple[float, float]:
    """The global minimum of pi S_c(u-) and the minimum over received-arm followed cells."""
    initial = _initial(result)
    product = _product(initial)
    factual = initial.A[None, :, None] == np.arange(2)[:, None, None]
    reach = initial.T[None, :, None] >= initial.grid[None, None, :]
    return float(product.min()), float(product[factual & reach].min())


def test_t11_floors_below_every_cell_change_nothing() -> None:
    result = _fresh(_in_sample())
    lowest, _ = _cells(result)
    curve = result.diagnostics.truncation_curve([lowest / 4, lowest / 2])
    for name in result:
        rows = _rows(curve, name)
        assert rows["psi"].iloc[0] == rows["psi"].iloc[1]
    assert set(curve.to_frame()["truncated_fraction"]) == {0.0}


def test_t11_the_shares_rise_with_the_floor() -> None:
    result = _fresh(_in_sample())
    curve = result.diagnostics.truncation_curve()
    rows = _rows(curve, RD)
    assert len(rows) == 5
    # truncated_fraction is the unit share: a unit counts once if any of its cells, in
    # either arm, falls below the floor.
    product = _product(_initial(result))
    for bound, share in zip(rows["bound"], rows["truncated_fraction"], strict=True):
        assert share == float(np.mean(product.min(axis=(0, 2)) < bound))
    # Mutation control: the cell share is another number at some floor of the grid.
    assert any(
        share != float(np.mean(product < bound))
        for bound, share in zip(rows["bound"], rows["truncated_fraction"], strict=True)
    )
    assert np.all(np.diff(rows["truncated_fraction"]) >= 0)
    assert np.all(np.diff(rows["bounded_share"]) >= 0)
    assert rows["truncated_fraction"].iloc[-1] > rows["truncated_fraction"].iloc[0]


def _counterfactual_floor(result: Any) -> tuple[float, float]:
    lowest, received = _cells(result)
    assert lowest < received
    return lowest / 2, (lowest + received) / 2


def test_t11_a_floor_on_counterfactual_cells_alone_moves_the_estimate() -> None:
    """Nonzero witness of M3: the walk weights both arms of every unit."""
    result = _fresh(_in_sample())
    below, between = _counterfactual_floor(result)
    curve = result.diagnostics.truncation_curve([below, between])
    at = curve.to_frame()
    at = at[at["bound"] == between]
    assert set(at["bounded_share"]) == {0.0}
    assert set(at["truncated_fraction"]).pop() > 0
    assert any(_rows(curve, name)["psi"].diff().iloc[1] != 0 for name in result)


def test_t11_a_curve_that_floors_received_cells_only_stays_flat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control of the witness above: received-arm flooring leaves psi flat."""
    result = _fresh(_in_sample())
    below, between = _counterfactual_floor(result)
    fitted = result.record.weight_floor
    original = engine._FitPlan._weight

    def received_only(self: Any, k: int) -> Any:
        floored = original(self, k)
        if self.floor == fitted:
            return floored
        lo, hi = self.bounds[k]
        factual = self.fit.A[None, lo:hi, None] == np.arange(2)[:, None, None]
        return np.where(factual, floored, 1.0 / _block_product(self, k))

    monkeypatch.setattr(engine._FitPlan, "_weight", received_only)
    curve = result.diagnostics.truncation_curve([below, between])
    assert all(_rows(curve, name)["psi"].diff().iloc[1] == 0 for name in result)


@functools.cache
def _binding() -> Any:
    """The binding-floor law of the assessment tests: censoring three times as fast if treated."""
    return _fit(_arm_censored(), horizons=(0.5, 1.0))


def _matches_refits(result: Any, curve: Any) -> bool:
    """Whether each curve row equals a refit of the stored estimator at that floor."""
    frame = curve.to_frame()
    estimator = result.estimator
    for bound in sorted(set(frame["bound"])):
        changed = copy.copy(estimator)
        changed.g_bounds = float(bound)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refit = changed.refit(result.data, random_state=estimator.random_state)
        rows = frame[frame["bound"] == bound]
        for _, row in rows.iterrows():
            estimate = refit[row["estimand"]]
            if row["psi"] != estimate.psi or row["std_err"] != estimate.plugin_std_error:
                return False
        record = refit.record
        if (
            set(rows["bounded_share"]) != {record.bounded_share}
            or set(rows["last_time_floor_share"]) != {record.last_time_floor_share}
            or set(rows["frozen_share"]) != {record.frozen_share}
        ):
            return False
    return True


@pytest.mark.parametrize("fitted", [_in_sample, _binding], ids=["sweep_law", "arm_censored"])
def test_t11_each_row_equals_a_refit_at_its_floor(fitted: Any) -> None:
    result = _fresh(fitted())
    curve = result.diagnostics.truncation_curve()
    assert _matches_refits(result, curve)


def test_t11_flooring_the_propensity_alone_fails_the_exact_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation control: on arm-dependent censoring a floor on pi alone is another weight."""
    result = _fresh(_binding())
    assert result.record.bounded_share > 0
    fitted = result.record.weight_floor
    original = engine._FitPlan._weight

    def propensity_only(self: Any, k: int) -> Any:
        floored = original(self, k)
        if self.floor == fitted:
            return floored
        lo, hi = self.bounds[k]
        pi = self.pi[:, lo:hi, None]
        retention = _block_product(self, k) / pi
        return 1.0 / (np.maximum(pi, self.floor) * retention)

    monkeypatch.setattr(engine._FitPlan, "_weight", propensity_only)
    curve = result.diagnostics.truncation_curve()
    monkeypatch.setattr(engine._FitPlan, "_weight", original)
    assert not _matches_refits(result, curve)


def test_t12_the_fitted_row_is_the_fit() -> None:
    result = _fresh(_in_sample())
    frame = result.diagnostics.truncation_curve().to_frame()
    fitted = frame[frame["is_fitted_bound"]]
    assert set(fitted["bound"]) == {result.record.weight_floor}
    assert len(fitted) == len(result)
    for _, row in fitted.iterrows():
        estimate = result[row["estimand"]]
        assert row["psi"] == row["fitted_psi"] == estimate.psi
        assert row["delta_from_fitted"] == 0.0
        # The fitted row keeps the fit's settings, so its support is the fit's, reason and all.
        assert estimate.support.label == row["support"] == "experimental"
        assert row["support_reason"] == estimate.support.reason
        assert "g_bounds" not in row["support_reason"]
    # Every other floor sets g_bounds away from "auto", which the support names.
    others = frame[~frame["is_fitted_bound"]]
    assert set(others["support"]) == {"experimental"}
    assert all("g_bounds" in reason for reason in others["support_reason"])


def test_t12_a_corrupted_datum_is_refused_with_the_replay_text() -> None:
    result = _in_sample()
    time = result.data.time.copy()
    time[0] = time[0] * 1.5
    corrupted = dataclasses.replace(result, data=dataclasses.replace(result.data, time=time))
    with pytest.raises(CapabilityError, match="do not reproduce this fit at its own floor") as e:
        corrupted.diagnostics.truncation_curve()
    assert "VersionMismatchWarning" in str(e.value)
    assert e.value.refusal.reason is RefusalReason.INVALID_DECLARATION


def test_t12_mechanism_and_a_floor_outside_the_unit_interval_are_refused() -> None:
    result = _fresh(_in_sample())
    with pytest.raises(CapabilityError, match="mechanism=True is a point-treatment option"):
        result.diagnostics.truncation_curve(mechanism=True)
    with pytest.raises(CapabilityError, match=r"lies in \(0, 1\)"):
        result.diagnostics.truncation_curve([1.5])
    from cleverly import sensitivity

    with pytest.raises(CapabilityError, match="mechanism=True is a point-treatment option"):
        sensitivity.truncation_curve(result, mechanism=True)


def test_t12_the_free_function_and_the_plot_read_the_curve() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from cleverly import plotting, sensitivity

    result = _fresh(_in_sample())
    frame = sensitivity.truncation_curve(result, estimands=[RD])
    curve = result.diagnostics.truncation_curve(estimands=[RD])
    pd.testing.assert_frame_equal(frame, curve.to_frame())
    figure = plotting.sensitivity(curve)
    assert len(figure.axes) == 2
