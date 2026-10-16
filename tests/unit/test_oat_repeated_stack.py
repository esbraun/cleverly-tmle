"""Witnesses of the repeated per-arm OAT stack (X58 plan, 13.3, RW1 to RW3).

A fit with ``repeats=3`` reports the median of three cross-fitted draws with the
split-adjusted variance. Each draw keeps its own status, standard error, curve and shipped
standard error in ``RepeatFit.inference``. A parameter status of one draw passes to the
median; different fit statuses across draws still raise, also beside a parameter status
(GUARD_STATUSES). Each witness has a
deliberate-mutation control that fails when its term is wrong.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import cleverly
from cleverly._inference_status import PARAMETER_STATUSES
from cleverly.estimators import CTMLE, _oat_stack
from cleverly.estimators import ctmle as ctmle_module
from cleverly.estimators._oat_stack import corrected_curves, fitted_stack
from cleverly.exceptions import DataWarning
from cleverly.inference import influence as influence_module
from cleverly.inference.influence import median_estimates
from cleverly.learners import ParametricMechanism, ParametricOutcome
from tests.studies import oat_per_arm_laws as laws

pytestmark = pytest.mark.xdist_group("oat_repeated_stack")

ESTIMANDS = ("ey0", "ey1", "ate")


def _fit(frame: Any, **options: Any) -> Any:
    settings: dict[str, Any] = {
        "strategy": "oat",
        "outcome_learner": ParametricOutcome(),
        "treatment_learner": ParametricMechanism(),
        "cross_fit": True,
        "n_folds": 5,
        "simultaneous": False,
        "estimands": ESTIMANDS,
        "g_bounds": laws.G_BOUNDS,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
    }
    settings.update(options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(**settings)
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
            .single()
        )


@pytest.fixture(scope="module")
def repeated() -> Any:
    return _fit(laws.BINARY_ACTIVE.sample(1500, 71)[0], repeats=3)


@pytest.fixture(scope="module")
def singles(repeated: Any) -> list[Any]:
    """A ``repeats=1`` fit at each draw's seed."""
    frame = laws.BINARY_ACTIVE.sample(1500, 71)[0]
    return [_fit(frame, random_state=repeat.seed) for repeat in repeated.repeats]


class TestRW1PerDrawIdentity:
    """RW1: each stored draw is the ``repeats=1`` fit at its seed, and the median is theirs."""

    @pytest.mark.parametrize("draw", [0, 1, 2])
    def test_each_draw_is_its_single_fit(
        self, repeated: Any, singles: list[Any], draw: int
    ) -> None:
        record = repeated.repeats[draw].inference
        single = singles[draw]
        assert set(record) == set(ESTIMANDS)
        for name in ESTIMANDS:
            reference = single.repeats[0].inference[name]
            assert repeated.repeats[draw].psi[name] == pytest.approx(single[name].psi, rel=1e-12)
            assert record[name].curve is not None
            np.testing.assert_allclose(
                record[name].curve, single[name].influence_curve, rtol=1e-12, atol=1e-15
            )
            assert record[name].se == pytest.approx(single[name].std_error, rel=1e-12)
            assert record[name].shipped_se == pytest.approx(reference.shipped_se, rel=1e-12)
            assert record[name].status == "influence_curve"

    def test_the_shipped_se_is_the_uncorrected_curve(self, singles: list[Any]) -> None:
        single = singles[0]
        report = _oat_stack.stack_report(fitted_stack(single), (0.0, 1.0))
        for j, name in enumerate(("ey0", "ey1")):
            expected = report.shipped_std_error[float(j)] * fitted_stack(single).scale
            assert single.repeats[0].inference[name].shipped_se == pytest.approx(expected, rel=1e-9)
            assert single.repeats[0].inference[name].shipped_se < single[name].std_error

    def test_the_median_is_the_singles(self, repeated: Any, singles: list[Any]) -> None:
        combined = median_estimates([single.estimates for single in singles])
        for name in ESTIMANDS:
            assert repeated[name].psi == pytest.approx(combined[name].psi, rel=1e-12)
            assert repeated[name].std_error == pytest.approx(combined[name].std_error, rel=1e-12)

    @pytest.mark.parametrize("draw", [0, 1, 2])
    def test_fitted_stack_reads_the_requested_draw(
        self, repeated: Any, singles: list[Any], draw: int
    ) -> None:
        """m3: ``fitted_stack(result, draw=r)`` is the stack of the single fit at draw r's seed."""
        np.testing.assert_allclose(
            fitted_stack(repeated, draw=draw).theta,
            fitted_stack(singles[draw]).theta,
            rtol=1e-12,
            atol=1e-15,
        )

    def test_mutation_every_draw_reads_draw_zero(self, repeated: Any, singles: list[Any]) -> None:
        stack = fitted_stack(repeated, draw=0)
        curves = corrected_curves(stack.problem, stack.theta).curves * stack.scale
        ate = curves[:, 1] - curves[:, 0]
        assert np.max(np.abs(ate - singles[1]["ate"].influence_curve)) > 1e-4


def _fail_second_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force arm 1's stack to fail on the second draw's report."""
    original = ctmle_module.stack_report
    calls: list[int] = []

    def failing(stack: Any, arms: Any) -> Any:
        report = original(stack, arms)
        calls.append(1)
        if len(calls) == 2:
            failures = {**report.failures, 1.0: "forced"}
            corrections = {**report.corrections, 1.0: np.zeros(stack.problem.n)}
            return replace(report, failures=failures, corrections=corrections)
        return report

    monkeypatch.setattr(ctmle_module, "stack_report", failing)


def _draw_statuses(result: Any, name: str) -> list[str]:
    return [repeat.inference[name].status for repeat in result.repeats]


class TestRW2StatusPrecedence:
    """RW2: one draw's failed arm withholds that arm's estimates in the median."""

    @pytest.fixture(scope="class")
    def failed(self) -> Any:
        with pytest.MonkeyPatch.context() as patch:
            _fail_second_draw(patch)
            return _fit(laws.BINARY_ACTIVE.sample(1500, 71)[0], repeats=3)

    def test_the_failed_arm_is_withheld(self, failed: Any) -> None:
        assert failed["ey0"].inference == "influence_curve"
        assert failed["ey1"].inference == "stack_failure_plugin"
        assert failed["ate"].inference == "stack_failure_plugin"

    def test_each_draw_keeps_its_own_status(self, failed: Any) -> None:
        assert _draw_statuses(failed, "ey1") == [
            "influence_curve",
            "stack_failure_plugin",
            "influence_curve",
        ]
        assert _draw_statuses(failed, "ey0") == ["influence_curve"] * 3

    def test_the_record_round_trips(self, failed: Any, tmp_path: Any) -> None:
        loaded = cleverly.load(failed.save(tmp_path / "failed.pkl"))
        assert _draw_statuses(loaded, "ey1") == _draw_statuses(failed, "ey1")
        assert loaded["ate"].inference == "stack_failure_plugin"
        for repeat, original in zip(loaded.repeats, failed.repeats, strict=True):
            for name in ESTIMANDS:
                assert repeat.inference[name].shipped_se == original.inference[name].shipped_se
                np.testing.assert_array_equal(
                    repeat.inference[name].curve, original.inference[name].curve
                )

    def test_mutation_no_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fail_second_draw(monkeypatch)
        monkeypatch.setattr(
            influence_module,
            "restamp_parameter_statuses",
            lambda per_repeat: [dict(report) for report in per_repeat],
        )
        with pytest.raises(ValueError, match="different inference statuses"):
            _fit(laws.BINARY_ACTIVE.sample(1500, 71)[0], repeats=3)

    def test_mutation_restamped_record(self, failed: Any) -> None:
        """A record written after the restamp loses the draws' own statuses."""
        mutant = [
            {name: replace(record, status=failed[name].inference) for name, record in draw.items()}
            for draw in (repeat.inference for repeat in failed.repeats)
        ]
        assert [draw["ey1"].status for draw in mutant] != _draw_statuses(failed, "ey1")


def _draw_status(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    draw: int = 1,
    names: tuple[str, ...] | None = None,
) -> None:
    """Stamp ``status`` on draw ``draw``'s estimates (0-based), on ``names`` or on every one."""
    original = CTMLE._retarget_detailed
    calls: list[int] = []

    def stamping(self: CTMLE, *args: Any, **kwargs: Any) -> Any:
        estimates, fluctuations, detail = original(self, *args, **kwargs)
        calls.append(1)
        if len(calls) == draw + 1:
            estimates = {
                name: replace(estimate, inference=status)
                if names is None or name in names
                else estimate
                for name, estimate in estimates.items()
            }
        return estimates, fluctuations, detail

    monkeypatch.setattr(CTMLE, "_retarget_detailed", stamping)


#: The draw statuses that reach the fit-status guard of the restamp (M1): one draw holds a
#: parameter status, and another draw holds a fit status other than ``"influence_curve"``.
GUARD_STATUSES = ("influence_curve", "stack_failure_plugin", "generated_design_plugin")
#: The estimates that read arm 1, whose stack ``_fail_second_draw`` fails.
FAILED = ("ey1", "ate")


def _guard_draws(repeated: Any) -> list[dict[str, Any]]:
    """Three draws of ``repeated``'s estimates whose ``ate`` carries :data:`GUARD_STATUSES`."""
    return [
        {
            **repeated.estimates,
            "ate": replace(repeated.estimates["ate"], inference=status),
        }
        for status in GUARD_STATUSES
    ]


def _restamp_without_guard(per_repeat: Any) -> list[dict[str, Any]]:
    """Mutation [15]: the restamp without its fit-status guard."""
    out = [dict(report) for report in per_repeat]
    for name in out[0]:
        statuses = [report[name].inference for report in out]
        if not any(status in PARAMETER_STATUSES for status in statuses):
            continue
        status = influence_module.precedent_status(statuses)
        for report in out:
            report[name] = replace(report[name], inference=status)
    return out


class TestRW3FitStatusGuard:
    """RW3: draws with different fit statuses still raise in the median."""

    def test_different_fit_statuses_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _draw_status(monkeypatch, "generated_design_plugin")
        with pytest.raises(ValueError, match="different inference statuses"):
            _fit(laws.BINARY_ACTIVE.sample(1500, 71)[0], repeats=3)

    def test_the_guard_keeps_every_status(self, repeated: Any) -> None:
        """M1, function level: a parameter status beside another fit status is not restamped."""
        out = influence_module.restamp_parameter_statuses(_guard_draws(repeated))
        assert [report["ate"].inference for report in out] == list(GUARD_STATUSES)
        with pytest.raises(ValueError, match="different inference statuses"):
            median_estimates(out)

    def test_mutation_the_guard_dropped(self, repeated: Any) -> None:
        """Mutation [15] restamps the three draws to one status, so the median returns."""
        out = _restamp_without_guard(_guard_draws(repeated))
        assert len({report["ate"].inference for report in out}) == 1
        median_estimates(out)

    def test_a_failed_draw_beside_another_fit_status_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M1, fit level: draw 2 fails arm 1's stack, and draw 3 stamps another fit status.

        Draw 3 stamps only the estimates that read arm 1, so ``ey0`` agrees over the draws
        and the median reaches the guard on ``ey1`` and ``ate``.
        """
        _fail_second_draw(monkeypatch)
        _draw_status(monkeypatch, "generated_design_plugin", draw=2, names=FAILED)
        with pytest.raises(ValueError, match="different inference statuses"):
            _fit(laws.BINARY_ACTIVE.sample(1500, 71)[0], repeats=3)

    def test_mutation_the_guard_dropped_in_the_fit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _fail_second_draw(monkeypatch)
        _draw_status(monkeypatch, "generated_design_plugin", draw=2, names=FAILED)
        monkeypatch.setattr(influence_module, "restamp_parameter_statuses", _restamp_without_guard)
        result = _fit(laws.BINARY_ACTIVE.sample(1500, 71)[0], repeats=3)
        expected = influence_module.precedent_status(list(GUARD_STATUSES))
        assert result["ate"].inference == expected
        assert result["ey1"].inference == expected

    def test_mutation_restamp_over_every_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _draw_status(monkeypatch, "generated_design_plugin")

        def over_every(per_repeat: Any) -> list[dict[str, Any]]:
            out = [dict(report) for report in per_repeat]
            for name in out[0]:
                status = influence_module.precedent_status(
                    [report[name].inference for report in out]
                )
                for report in out:
                    report[name] = replace(report[name], inference=status)
            return out

        monkeypatch.setattr(influence_module, "restamp_parameter_statuses", over_every)
        result = _fit(laws.BINARY_ACTIVE.sample(1500, 71)[0], repeats=3)
        assert result["ate"].inference == "generated_design_plugin"


def test_restamp_returns_agreeing_draws_unchanged(repeated: Any) -> None:
    draws = [dict(repeated.estimates) for _ in range(3)]
    out = influence_module.restamp_parameter_statuses(draws)
    for report in out:
        for name, estimate in report.items():
            assert estimate is repeated.estimates[name]


def test_a_failed_draw_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failed draw warns; the rebuild of its uncorrected estimates adds no warning."""
    _fail_second_draw(monkeypatch)
    frame = laws.BINARY_ACTIVE.sample(1500, 71)[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CTMLE(
            strategy="oat",
            outcome_learner=ParametricOutcome(),
            treatment_learner=ParametricMechanism(),
            cross_fit=True,
            n_folds=5,
            repeats=3,
            simultaneous=False,
            estimands=ESTIMANDS,
            g_bounds=laws.G_BOUNDS,
            random_state=0,
        ).fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3"])
    forced = [
        w for w in caught if issubclass(w.category, DataWarning) and "forced" in str(w.message)
    ]
    assert len(forced) == 1
