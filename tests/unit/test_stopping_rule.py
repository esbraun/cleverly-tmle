"""The stopping rules of ``Targeting(stopping=)``: thresholds, ruled solvers, plumbing, refusals.

Every threshold below is recomputed by hand from the ``tmle3`` ``check_convergence`` formula
and the fit's own targeted arrays, with ``pandas`` for the per-unit sums, so the solver's
helper is never compared with itself.  The mutation controls break one line of a helper each
and ask the same hand check to fail.

``scaled_var`` is ``sd_j / (sqrt(n_u) * min(log n_u, 10))`` and ``sample_size`` is ``1 / n_u``,
both against ``|P_n S_j| * span``.  ``n_u`` counts the units among the rows the solver
receives and ``sd_j`` is the ``ddof=1`` SD of the per-unit influence ``(n_u / n) * sum_{i in
u} c_ij``.  A unit-weight column that targets one arm mean takes the full arm curve; every
other column takes the score column (declared difference D4).
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import Targeting
from cleverly._refusal import RefusalReason
from cleverly.estimators import TMLE
from cleverly.estimators.targeting import TargetingSpec, solve_submodel
from cleverly.exceptions import ConvergenceWarning, MethodConfigurationError
from cleverly.fluctuation import (
    InitialFit,
    Submodel,
    att_submodel,
    mean_submodel,
    solve_fluctuation,
    solve_one_step,
)
from cleverly.fluctuation import _score as score_module
from cleverly.fluctuation import iterative as iterative_module
from cleverly.fluctuation._score import (
    LINEAR_RULE_REFUSAL,
    STOP_CONTEXT_MISSING,
    StopContext,
    check_stopping_rule,
    rule_ratio_at,
    ruled_warning,
    score_columns,
)
from cleverly.fluctuation.iterative import apply_logistic
from cleverly.methods import SHORTCUTS, TMLEMethod
from cleverly.utils.bounds import expit
from tests.conftest import linear_in_sample
from tests.unit._source_mutation import mutate

LO, HI = -3.0, 9.0
SPAN = HI - LO
N_UNITS, PER_UNIT = 60, 4


def _setting(seed: int = 3) -> dict[str, Any]:
    """Clustered continuous outcome on ``[LO, HI]``, rescaled to ``[0, 1]`` for the solver."""
    rng = np.random.default_rng(seed)
    n = N_UNITS * PER_UNIT
    units = np.repeat(np.arange(N_UNITS), PER_UNIT)
    w = rng.normal(size=n) + np.repeat(rng.normal(size=N_UNITS), PER_UNIT)
    g1 = expit(0.5 * w)
    a = rng.binomial(1, g1).astype(float)
    raw = 2 + 1.5 * a + w + rng.normal(size=n) + np.repeat(rng.normal(size=N_UNITS), PER_UNIT)
    return {
        "n": n,
        "units": units,
        "g1": g1,
        "a": a,
        "y": np.clip((raw - LO) / SPAN, 0.01, 0.99),
        "initial": InitialFit(np.full(n, 0.4), {0.0: np.full(n, 0.4), 1.0: np.full(n, 0.4)}),
        "ones": np.ones(n),
    }


def _hand_thresholds(
    setting: dict[str, Any],
    fit: Any,
    submodel: Submodel,
    *,
    rule: str,
    units: np.ndarray | None,
    weights: np.ndarray,
    full_curve: bool,
) -> np.ndarray:
    """The tmle3 formula, written out from the fit's targeted arrays with pandas."""
    y = setting["y"]
    q = fit.targeted.observed
    h = submodel.observed
    n = y.shape[0]
    cols = []
    for j in range(h.shape[1]):
        score_part = weights * (y - q) * h[:, j]
        column = SPAN * score_part
        if full_curve:
            level = {col: lvl for lvl, col in submodel.arm_columns.items()}[j]
            qa = fit.targeted.arms[level]
            column = SPAN * (score_part + qa - qa.mean())
        cols.append(column)
    contribution = pd.DataFrame(np.column_stack(cols))
    key = np.arange(n) if units is None else units
    sums = contribution.groupby(key).sum()
    n_u = sums.shape[0]
    influence = sums * (n_u / n)
    if rule == "sample_size":
        return np.full(h.shape[1], 1.0 / n_u)
    sd = influence.std(axis=0, ddof=1).to_numpy()
    return sd / (np.sqrt(n_u) * min(np.log(n_u), 10.0))


def _ruled_iterative(setting: dict[str, Any], rule: str, **overrides: Any) -> Any:
    submodel = mean_submodel(setting["a"], setting["g1"])
    stop = StopContext(setting["units"], SPAN)
    kwargs = {"stopping": rule, "stop": stop, **overrides}
    fit = solve_fluctuation(
        setting["y"], setting["initial"], submodel, setting["ones"], None, **kwargs
    )
    return fit, submodel


class TestThresholdsByHand:
    @pytest.mark.parametrize("rule", ["scaled_var", "sample_size"])
    def test_a_clustered_arm_mean_fit_matches_the_tmle3_formula(self, rule: str) -> None:
        setting = _setting()
        fit, submodel = _ruled_iterative(setting, rule)
        hand = _hand_thresholds(
            setting,
            fit,
            submodel,
            rule=rule,
            units=setting["units"],
            weights=setting["ones"],
            full_curve=True,
        )
        assert fit.stopping == rule
        assert fit.thresholds is not None
        np.testing.assert_allclose(fit.thresholds, hand, rtol=1e-12, atol=0.0)
        assert fit.converged
        assert fit.failure is None
        assert np.max(np.abs(fit.score) * SPAN / hand) <= 1.0

    def test_sample_size_counts_units_rather_than_rows(self) -> None:
        setting = _setting()
        fit, _ = _ruled_iterative(setting, "sample_size")
        assert fit.thresholds is not None
        np.testing.assert_allclose(fit.thresholds, 1.0 / N_UNITS)
        assert not np.allclose(fit.thresholds, 1.0 / setting["n"])

    def test_without_units_every_row_is_a_unit(self) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        stop = StopContext(None, SPAN)
        fit = solve_fluctuation(
            setting["y"],
            setting["initial"],
            submodel,
            setting["ones"],
            None,
            stopping="scaled_var",
            stop=stop,
        )
        hand = _hand_thresholds(
            setting,
            fit,
            submodel,
            rule="scaled_var",
            units=None,
            weights=setting["ones"],
            full_curve=True,
        )
        assert fit.thresholds is not None
        np.testing.assert_allclose(fit.thresholds, hand, rtol=1e-12, atol=0.0)

    def test_a_contrast_column_takes_the_score_column(self) -> None:
        setting = _setting()
        submodel = att_submodel(setting["a"], setting["g1"], arm_fractions=setting["a"].mean())
        assert submodel.arm_columns == {}
        fit = solve_fluctuation(
            setting["y"],
            setting["initial"],
            submodel,
            setting["ones"],
            None,
            stopping="scaled_var",
            stop=StopContext(setting["units"], SPAN),
        )
        hand = _hand_thresholds(
            setting,
            fit,
            submodel,
            rule="scaled_var",
            units=setting["units"],
            weights=setting["ones"],
            full_curve=False,
        )
        assert fit.thresholds is not None
        np.testing.assert_allclose(fit.thresholds, hand, rtol=1e-12, atol=0.0)

    def test_a_weighted_arm_column_takes_the_score_column(self) -> None:
        setting = _setting()
        weights = 1.0 + 0.6 * np.sin(np.arange(setting["n"]))
        weights = weights / weights.mean()
        submodel = mean_submodel(setting["a"], setting["g1"])
        fit = solve_fluctuation(
            setting["y"],
            setting["initial"],
            submodel,
            weights,
            None,
            stopping="scaled_var",
            stop=StopContext(setting["units"], SPAN),
        )
        hand = _hand_thresholds(
            setting,
            fit,
            submodel,
            rule="scaled_var",
            units=setting["units"],
            weights=weights,
            full_curve=False,
        )
        curve = _hand_thresholds(
            setting,
            fit,
            submodel,
            rule="scaled_var",
            units=setting["units"],
            weights=weights,
            full_curve=True,
        )
        assert fit.thresholds is not None
        np.testing.assert_allclose(fit.thresholds, hand, rtol=1e-12, atol=0.0)
        assert np.max(np.abs(hand - curve) / curve) > 1e-3

    def test_the_log_cap_binds_above_22026_units(self) -> None:
        rng = np.random.default_rng(5)
        n = 30_000
        contributions = rng.normal(size=(n, 2))
        stop = StopContext(None, 1.0)
        thresholds, _ = rule_ratio_at(np.zeros(2), contributions, stop, "scaled_var")
        sd = contributions.std(axis=0, ddof=1)
        np.testing.assert_allclose(thresholds, sd / (np.sqrt(n) * 10.0), rtol=1e-12)
        assert np.log(n) > 10.0

    def test_a_rule_that_needs_a_spread_refuses_a_single_unit(self) -> None:
        stop = StopContext(np.zeros(5, dtype=np.int64), 1.0)
        with pytest.raises(ValueError, match="at least two units"):
            rule_ratio_at(np.zeros(1), np.ones((5, 1)), stop, "scaled_var")
        thresholds, ratio = rule_ratio_at(np.zeros(1), np.ones((5, 1)), stop, "sample_size")
        assert thresholds.tolist() == [1.0]
        assert ratio == 0.0

    def test_the_context_must_describe_the_rows_it_receives(self) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        with pytest.raises(ValueError, match="exactly the rows it receives"):
            solve_fluctuation(
                setting["y"],
                setting["initial"],
                submodel,
                setting["ones"],
                None,
                stopping="sample_size",
                stop=StopContext(setting["units"][:-1], SPAN),
            )

    def test_restrict_keeps_the_span_and_the_units_of_the_rows(self) -> None:
        stop = StopContext(np.array([5, 5, 6, 7, 7, 7]), 4.0)
        rows = np.array([0, 2, 3, 4])
        restricted = stop.restrict(rows)
        assert restricted.span == 4.0
        assert restricted.units is not None
        assert restricted.units.tolist() == [5, 6, 7, 7]
        masked = stop.restrict(np.array([True, True, False, False, False, False]))
        assert masked.units is not None
        assert masked.units.tolist() == [5, 5]
        assert StopContext(None, 2.0).restrict(rows).units is None
        _, ratio_all = rule_ratio_at(np.ones(1), np.ones((6, 1)), stop, "sample_size")
        thresholds, _ = rule_ratio_at(np.ones(1), np.ones((4, 1)), restricted, "sample_size")
        assert thresholds.tolist() == [1.0 / 3.0]
        assert ratio_all == pytest.approx(4.0 * 3.0)

    @pytest.mark.parametrize("span", [0.0, -1.0, float("nan"), float("inf")])
    def test_the_span_must_be_positive_and_finite(self, span: float) -> None:
        with pytest.raises(ValueError, match="span must be a positive finite number"):
            StopContext(None, span)


class TestRuledIterativeSolver:
    def test_the_default_fluctuation_records_the_shipped_rule(self) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        fit = solve_fluctuation(setting["y"], setting["initial"], submodel, setting["ones"])
        assert fit.stopping == "relative"
        assert fit.thresholds is None

    def test_a_pinned_score_that_meets_the_rule_converges_where_relative_cannot(self) -> None:
        """Separation: the shrink bound holds the score near ``1 - alpha`` for ever."""
        setting = _setting()
        rng = np.random.default_rng(11)
        y = np.where(setting["a"] == 1.0, 1.0, (rng.random(setting["n"]) < 0.5).astype(float))
        initial = InitialFit(
            np.full(setting["n"], 0.5),
            {0.0: np.full(setting["n"], 0.5), 1.0: np.full(setting["n"], 0.5)},
        )
        submodel = mean_submodel(setting["a"], setting["g1"])
        stop = StopContext(setting["units"], 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            shipped = solve_fluctuation(y, initial, submodel, setting["ones"], None, alpha=0.99)
            ruled = solve_fluctuation(
                y,
                initial,
                submodel,
                setting["ones"],
                None,
                alpha=0.99,
                stopping="sample_size",
                stop=stop,
            )
        assert not shipped.converged
        assert shipped.n_iter == 20
        assert ruled.converged
        assert ruled.failure is None
        assert ruled.n_iter == 1
        # The estimate moves: the shipped walk keeps pushing the pinned arm out.
        assert abs(ruled.epsilon[1] - shipped.epsilon[1]) > 100.0
        hand = _hand_thresholds(
            {**setting, "y": y},
            ruled,
            submodel,
            rule="sample_size",
            units=setting["units"],
            weights=setting["ones"],
            full_curve=True,
        )
        assert np.max(np.abs(ruled.score) / hand) <= 1.0
        assert np.max(np.abs(ruled.score)) > 1e-3

    def test_an_unmet_rule_warns_with_the_pinned_text(self) -> None:
        setting = _setting()
        rng = np.random.default_rng(11)
        y = np.where(setting["a"] == 1.0, 1.0, (rng.random(setting["n"]) < 0.5).astype(float))
        initial = InitialFit(
            np.full(setting["n"], 0.5),
            {0.0: np.full(setting["n"], 0.5), 1.0: np.full(setting["n"], 0.5)},
        )
        submodel = mean_submodel(setting["a"], setting["g1"])
        with pytest.warns(ConvergenceWarning) as record:
            fit = solve_fluctuation(
                y,
                initial,
                submodel,
                setting["ones"],
                None,
                alpha=0.99,
                stopping="scaled_var",
                stop=StopContext(setting["units"], 1.0),
            )
        assert not fit.converged
        assert fit.failure == "separation_suspected"
        message = str(record[0].message)
        assert message.startswith(
            "targeting step did not meet the scaled_var stopping rule after 20 iteration(s) "
            "(largest score over its threshold = "
        )
        ratio = float(re.search(r"threshold = ([0-9.e+-]+)\)", message).group(1))  # type: ignore[union-attr]
        assert ratio > 1.0
        assert "separation" in message
        assert message.endswith(
            "Inspect res.fluctuations[group].failure and res.diagnostics.support()."
        )

    def test_the_warning_template(self) -> None:
        assert ruled_warning("sample_size", 3, 12.3456, "why") == (
            "targeting step did not meet the sample_size stopping rule after 3 iteration(s) "
            "(largest score over its threshold = 12.3): why"
        )

    def test_a_carried_fit_moves_with_the_ruled_steps(self) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        carried = InitialFit(
            np.full(setting["n"], 0.3),
            {0.0: np.full(setting["n"], 0.3), 1.0: np.full(setting["n"], 0.3)},
        )
        fit = solve_fluctuation(
            setting["y"],
            setting["initial"],
            submodel,
            setting["ones"],
            None,
            carry=(carried,),
            stopping="sample_size",
            stop=StopContext(setting["units"], SPAN),
        )
        assert len(fit.carried) == 1
        moved = apply_logistic(carried.shrunk(0.9995), submodel, fit.epsilon, 0.9995)
        np.testing.assert_allclose(fit.carried[0].observed, moved.observed, rtol=1e-12)

    def test_a_missing_outcome_row_contributes_a_zero_score_part(self) -> None:
        setting = _setting()
        observed = np.ones(setting["n"], dtype=bool)
        observed[::7] = False
        submodel = mean_submodel(setting["a"], setting["g1"])
        fit = solve_fluctuation(
            setting["y"],
            setting["initial"],
            submodel,
            setting["ones"],
            observed,
            stopping="scaled_var",
            stop=StopContext(setting["units"], SPAN),
        )
        y = setting["y"].copy()
        residual = np.where(observed, y - fit.targeted.observed, 0.0)
        columns = []
        for level, column in submodel.arm_columns.items():
            qa = fit.targeted.arms[level]
            columns.append(SPAN * (residual * submodel.observed[:, column] + qa - qa.mean()))
        order = [submodel.arm_columns[0.0], submodel.arm_columns[1.0]]
        frame = pd.DataFrame(np.column_stack(columns), columns=order).sort_index(axis=1)
        sums = frame.groupby(setting["units"]).sum() * (N_UNITS / setting["n"])
        hand = sums.std(axis=0, ddof=1).to_numpy() / (np.sqrt(N_UNITS) * min(np.log(N_UNITS), 10))
        assert fit.thresholds is not None
        np.testing.assert_allclose(fit.thresholds, hand, rtol=1e-12, atol=0.0)


def _hand_one_step(setting: dict[str, Any], step_size: float, rule: str) -> tuple[int, np.ndarray]:
    """The universal least-favorable walk, transcribed, stopped by the hand ratio."""
    submodel = mean_submodel(setting["a"], setting["g1"])
    y, w = setting["y"], setting["ones"]
    mask = np.ones(setting["n"], dtype=bool)
    current = setting["initial"].shrunk(0.9995)
    epsilon = np.zeros(2)
    dx = step_size
    steps = 0
    fake = type("Fit", (), {})()
    while True:
        fake.targeted = current
        threshold = _hand_thresholds(
            setting,
            fake,
            submodel,
            rule=rule,
            units=setting["units"],
            weights=w,
            full_curve=True,
        )
        score = score_columns(y, current.observed, submodel.observed, w, mask)
        if np.max(np.abs(score) * SPAN / threshold) <= 1.0:
            return steps, epsilon
        norm = np.linalg.norm(score)
        direction = score / norm
        candidate = apply_logistic(current, submodel, dx * direction, 0.9995)
        candidate_score = score_columns(y, candidate.observed, submodel.observed, w, mask)
        if np.linalg.norm(candidate_score) > norm:
            dx *= 0.5
            continue
        epsilon = epsilon + dx * direction
        current = candidate
        steps += 1


class TestRuledOneStepWalk:
    @pytest.mark.parametrize("rule", ["sample_size", "scaled_var"])
    def test_the_walk_stops_at_the_hand_computed_index(self, rule: str) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        steps, epsilon = _hand_one_step(setting, 0.01, rule)
        ruled = solve_one_step(
            setting["y"],
            setting["initial"],
            submodel,
            setting["ones"],
            None,
            step_size=0.01,
            stopping=rule,
            stop=StopContext(setting["units"], SPAN),
        )
        assert steps > 5
        assert ruled.n_iter == steps
        np.testing.assert_allclose(ruled.epsilon, epsilon, rtol=1e-12, atol=1e-15)
        assert ruled.converged
        assert ruled.method == "one_step"
        assert ruled.stopping == rule
        assert ruled.thresholds is not None

    def test_the_ruled_walk_is_the_shipped_walk_cut_short(self) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        ruled = solve_one_step(
            setting["y"],
            setting["initial"],
            submodel,
            setting["ones"],
            None,
            step_size=0.01,
            stopping="sample_size",
            stop=StopContext(setting["units"], SPAN),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            truncated = solve_one_step(
                setting["y"],
                setting["initial"],
                submodel,
                setting["ones"],
                None,
                step_size=0.01,
                max_steps=ruled.n_iter,
            )
            full = solve_one_step(
                setting["y"], setting["initial"], submodel, setting["ones"], None, step_size=0.01
            )
        np.testing.assert_array_equal(ruled.epsilon, truncated.epsilon)
        np.testing.assert_array_equal(ruled.score, truncated.score)
        assert full.n_iter > ruled.n_iter
        # Witness: the looser rule reports a different estimate.
        ruled_mean = ruled.targeted.arms[1.0].mean() * SPAN
        full_mean = full.targeted.arms[1.0].mean() * SPAN
        assert abs(ruled_mean - full_mean) > 1e-6

    def test_a_walk_that_cannot_meet_the_rule_warns_with_the_pinned_text(self) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        with pytest.warns(ConvergenceWarning, match="did not meet the sample_size stopping rule"):
            fit = solve_one_step(
                setting["y"],
                setting["initial"],
                submodel,
                setting["ones"],
                None,
                step_size=0.01,
                max_steps=3,
                stopping="sample_size",
                stop=StopContext(setting["units"], SPAN),
            )
        assert not fit.converged
        assert fit.failure == "max_iter_reached"
        assert fit.n_iter == 3

    def test_a_rule_that_is_met_at_the_start_takes_no_step(self) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        fit = solve_one_step(
            setting["y"],
            setting["initial"],
            submodel,
            setting["ones"],
            None,
            step_size=0.01,
            stopping="sample_size",
            stop=StopContext(setting["units"], 1e-9),
        )
        assert fit.n_iter == 0
        assert fit.converged
        np.testing.assert_array_equal(fit.epsilon, np.zeros(2))


class TestPlumbing:
    def test_a_declared_rule_without_a_context_fails_loudly(self) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        observed = np.ones(setting["n"], dtype=bool)
        for algorithm in ("iterative", "one_step"):
            spec = TargetingSpec(targeting=algorithm, stopping="scaled_var")  # type: ignore[arg-type]
            with pytest.raises(ValueError, match=re.escape(STOP_CONTEXT_MISSING)):
                solve_submodel(
                    setting["y"], setting["initial"], submodel, setting["ones"], observed, spec
                )
        with pytest.raises(ValueError, match=re.escape(STOP_CONTEXT_MISSING)):
            solve_fluctuation(
                setting["y"],
                setting["initial"],
                submodel,
                setting["ones"],
                None,
                stopping="sample_size",
            )
        with pytest.raises(ValueError, match=re.escape(STOP_CONTEXT_MISSING)):
            solve_one_step(
                setting["y"],
                setting["initial"],
                submodel,
                setting["ones"],
                None,
                stopping="sample_size",
            )

    def test_the_shipped_rule_ignores_a_context_and_matches_a_call_without_one(self) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        observed = np.ones(setting["n"], dtype=bool)
        spec = TargetingSpec()
        plain = solve_submodel(
            setting["y"], setting["initial"], submodel, setting["ones"], observed, spec
        )
        given = solve_submodel(
            setting["y"],
            setting["initial"],
            submodel,
            setting["ones"],
            observed,
            spec,
            stop=StopContext(setting["units"], SPAN),
        )
        np.testing.assert_array_equal(plain.epsilon, given.epsilon)
        np.testing.assert_array_equal(plain.score, given.score)
        assert plain.stopping == given.stopping == "relative"
        assert plain.thresholds is None and given.thresholds is None

    @pytest.mark.parametrize("algorithm", ["iterative", "one_step"])
    def test_solve_submodel_passes_the_rule_and_the_context_to_the_solver(
        self, algorithm: str
    ) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        observed = np.ones(setting["n"], dtype=bool)
        spec = TargetingSpec(targeting=algorithm, stopping="sample_size", step_size=0.01)  # type: ignore[arg-type]
        stop = StopContext(setting["units"], SPAN)
        fit = solve_submodel(
            setting["y"], setting["initial"], submodel, setting["ones"], observed, spec, stop=stop
        )
        assert fit.stopping == "sample_size"
        assert fit.thresholds is not None
        np.testing.assert_allclose(fit.thresholds, 1.0 / N_UNITS)

    def test_a_fold_solve_counts_the_units_of_its_rows(self) -> None:
        setting = _setting()
        rows = np.arange(setting["n"])[setting["units"] < 40]
        stop = StopContext(setting["units"], SPAN).restrict(rows)
        submodel = mean_submodel(setting["a"][rows], setting["g1"][rows])
        initial = InitialFit(
            setting["initial"].observed[rows],
            {k: v[rows] for k, v in setting["initial"].arms.items()},
        )
        fit = solve_fluctuation(
            setting["y"][rows],
            initial,
            submodel,
            np.ones(rows.shape[0]),
            None,
            stopping="sample_size",
            stop=stop,
        )
        assert fit.thresholds is not None
        np.testing.assert_allclose(fit.thresholds, 1.0 / 40)

    def test_shortcuts_route_stopping_to_the_targeting_group(self) -> None:
        assert SHORTCUTS["targeting"]["stopping"] == "stopping"
        method = TMLEMethod().with_overrides(stopping="scaled_var")
        assert method.targeting.stopping == "scaled_var"
        assert method.estimator_kwargs()["stopping"] == "scaled_var"
        assert TMLEMethod().targeting.stopping == "relative"
        assert TMLEMethod().estimator_kwargs()["stopping"] == "relative"

    def test_the_engine_carries_the_rule_into_its_targeting_spec(self) -> None:
        estimator = TMLE(**linear_in_sample(), stopping="sample_size")
        assert estimator.targeting_spec().stopping == "sample_size"
        assert TMLE(**linear_in_sample()).targeting_spec().stopping == "relative"

    def test_a_longitudinal_design_carries_a_declared_rule_to_the_engine(self) -> None:
        method = TMLEMethod().with_overrides(stopping="sample_size")
        assert method.estimator_kwargs(longitudinal=True)["stopping"] == "sample_size"
        assert TMLEMethod().estimator_kwargs(longitudinal=True)["stopping"] == "relative"

    def test_the_default_rule_fits_the_same_as_leaving_it_out(self) -> None:
        rng = np.random.default_rng(2)
        n = 300
        w = rng.normal(size=n)
        a = rng.binomial(1, expit(0.4 * w)).astype(float)
        frame = pd.DataFrame({"W": w, "A": a, "Y": 1.0 + a + w + rng.normal(size=n)})
        plain = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        named = (
            TMLE(**linear_in_sample(), stopping="relative")
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert plain.estimates.keys() == named.estimates.keys()
        for key, estimate in plain.estimates.items():
            assert estimate.psi == named.estimates[key].psi
            assert estimate.variance == named.estimates[key].variance
            np.testing.assert_array_equal(
                estimate.influence_curve, named.estimates[key].influence_curve
            )
        for key, fluctuation in plain.fluctuations.items():
            np.testing.assert_array_equal(fluctuation.epsilon, named.fluctuations[key].epsilon)
            assert fluctuation.stopping == "relative"


class TestRefusals:
    def test_r_b1_names_the_three_rules(self) -> None:
        message = "stopping must be 'relative', 'scaled_var' or 'sample_size'; got 'bogus'"
        with pytest.raises(ValueError, match=re.escape(message)):
            Targeting(stopping="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=re.escape(message)):
            TMLE(**linear_in_sample(), stopping="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=re.escape(message)):
            TargetingSpec(stopping="bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=re.escape(message)):
            check_stopping_rule("bogus", "logistic")
        for rule in ("relative", "scaled_var", "sample_size"):
            assert Targeting(stopping=rule).stopping == rule  # type: ignore[arg-type]

    def test_r_abc1_refuses_a_declared_rule_beside_the_linear_fluctuation(self) -> None:
        expected = (
            "fluctuation='linear' solves the score equation in one closed-form step, so it "
            "has no iterate, stopping rule or step bound to choose. Remove iterate=, "
            "stopping= and step_bound=, or use fluctuation='logistic'"
        )
        assert expected == LINEAR_RULE_REFUSAL.message()
        assert LINEAR_RULE_REFUSAL.reason is RefusalReason.INVALID_DECLARATION
        for rule in ("scaled_var", "sample_size"):
            with pytest.raises(MethodConfigurationError, match=re.escape(expected)):
                Targeting(fluctuation="linear", stopping=rule)  # type: ignore[arg-type]
            with pytest.raises(MethodConfigurationError, match=re.escape(expected)):
                TMLE(**linear_in_sample(), fluctuation="linear", stopping=rule)  # type: ignore[arg-type]
            with pytest.raises(MethodConfigurationError, match=re.escape(expected)):
                TargetingSpec(fluctuation="linear", stopping=rule)  # type: ignore[arg-type]
            with pytest.raises(MethodConfigurationError, match=re.escape(expected)):
                TMLEMethod().with_overrides(fluctuation="linear", stopping=rule)
        setting = _setting()
        with pytest.raises(MethodConfigurationError, match=re.escape(expected)):
            solve_fluctuation(
                setting["y"],
                setting["initial"],
                mean_submodel(setting["a"], setting["g1"]),
                setting["ones"],
                None,
                kind="linear",
                stopping="sample_size",
                stop=StopContext(None, SPAN),
            )

    def test_the_linear_fluctuation_still_takes_the_shipped_rule(self) -> None:
        assert Targeting(fluctuation="linear").stopping == "relative"
        assert TargetingSpec(fluctuation="linear").stopping == "relative"


class TestMutationControls:
    """One broken line each; the hand check of the same property must fail."""

    @staticmethod
    def _deviation(rule: str = "scaled_var") -> float:
        setting = _setting()
        fit, submodel = _ruled_iterative(setting, rule)
        hand = _hand_thresholds(
            setting,
            fit,
            submodel,
            rule=rule,
            units=setting["units"],
            weights=setting["ones"],
            full_curve=True,
        )
        assert fit.thresholds is not None
        return float(np.max(np.abs(fit.thresholds - hand) / hand))

    @staticmethod
    def _cap_deviation() -> float:
        rng = np.random.default_rng(5)
        n = 30_000
        contributions = rng.normal(size=(n, 2))
        thresholds, _ = rule_ratio_at(
            np.zeros(2), contributions, StopContext(None, 1.0), "scaled_var"
        )
        sd = contributions.std(axis=0, ddof=1)
        return float(np.max(np.abs(thresholds - sd / (np.sqrt(n) * 10.0)) / thresholds))

    def test_the_controls_pass_on_the_shipped_code(self) -> None:
        assert self._deviation() < 1e-12
        assert self._deviation("sample_size") < 1e-12
        assert self._cap_deviation() < 1e-12

    def test_rows_in_place_of_units_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mutate(
            monkeypatch,
            score_module,
            score_module,
            "unit_influence",
            [("n_units = int(inverse.max()) + 1 if n else 0", "n_units = n")],
        )
        assert self._deviation() > 1e-2
        assert self._deviation("sample_size") > 1e-2

    def test_dropping_the_span_from_the_contributions_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            score_module,
            score_module,
            "rule_contributions",
            [
                ("out = span * score_part", "out = score_part"),
                ("out[:, column] = span * (score_part", "out[:, column] = 1.0 * (score_part"),
            ],
        )
        monkeypatch.setattr(iterative_module, "rule_contributions", score_module.rule_contributions)
        assert self._deviation() > 1e-2

    def test_dropping_the_span_from_the_statistic_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setting = _setting()
        submodel = mean_submodel(setting["a"], setting["g1"])
        hand_steps, _ = _hand_one_step(setting, 0.01, "sample_size")

        def walk() -> int:
            return solve_one_step(
                setting["y"],
                setting["initial"],
                submodel,
                setting["ones"],
                None,
                step_size=0.01,
                stopping="sample_size",
                stop=StopContext(setting["units"], SPAN),
            ).n_iter

        assert walk() == hand_steps
        mutate(
            monkeypatch,
            score_module,
            score_module,
            "rule_ratio_at",
            [("statistic = np.abs(score) * stop.span", "statistic = np.abs(score)")],
        )
        monkeypatch.setattr(iterative_module, "rule_ratio_at", score_module.rule_ratio_at)
        assert walk() < hand_steps

    def test_the_uncapped_log_fails_at_30000_units(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mutate(
            monkeypatch,
            score_module,
            score_module,
            "rule_thresholds",
            [("min(np.log(n_units), 10.0)", "np.log(n_units)")],
        )
        assert self._cap_deviation() > 1e-2
        # Below the cap the two agree, so the clustered fixture is blind to this mutation.
        assert self._deviation() < 1e-12

    def test_the_score_column_on_an_arm_column_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mutate(
            monkeypatch,
            iterative_module,
            iterative_module,
            "_rule_state",
            [('submodel.group == "mean"', 'submodel.group == "never"')],
        )
        assert self._deviation() > 1e-3

    def test_ddof_zero_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mutate(
            monkeypatch,
            score_module,
            score_module,
            "rule_thresholds",
            [("axis=0, ddof=1)", "axis=0, ddof=0)")],
        )
        assert self._deviation() > 1e-3
